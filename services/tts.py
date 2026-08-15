"""Text to speech: Kokoro-82M on ONNX Runtime, with a disk cache.

TTS stays local deliberately. It is the one model here that is *cheap* to run
and *expensive* to call remotely: synthesis is pure CPU with no per-request fee,
the cache means most requests never synthesize anything at all, and the voice
must be byte-identical across browsers — which it is, because the client just
plays back whatever bytes this returns.

Clips are cached on disk keyed by (voice, lang, speed, text). Because answer
text is the cache key, the semantic answer cache upstream is what makes this
work: two visitors who ask the same thing in different words get the *same
answer string*, so the second one is a disk read.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from config import (
    APP_DIR,
    CACHE_DIR,
    GEMINI_TTS_MODEL,
    KOKORO_MODEL,
    KOKORO_VOICES,
    ONNX_THREADS,
    SYNTH_WORKERS,
    TRIM_SILENCE,
    TTS_MEM_ARENA,
    TTS_VOICE,
    TTS_VOICE_OVERRIDES,
)
from kb import debug

# One Kokoro session per worker thread.
#
# Sentence streaming only works while synthesis outruns playback, and measured
# on this corpus (a 4-sentence, 20-second answer, 12-core CPU) it does not:
#
#   int8  1 session  x 4 threads     RTF 0.95     <- what production shipped
#   int8  2 sessions x 2 threads     RTF 0.55
#   int8  4 sessions x 2 threads     RTF 0.43
#   fp32  1 session  x 4 threads     RTF 0.32
#
# At RTF 0.95 the margin is gone: any jitter — a second request, the prewarm
# thread, a slower host — pushes it past 1.0, and then playback drains faster
# than clips arrive and a gap opens at every sentence boundary. That is the
# "part by part" the UI was producing.
#
# Kokoro is a small model, so a single session cannot use many cores well
# (4 threads is its peak; 8 is slower). Running several *sessions* side by side
# does use them, which is why two workers beat one by 1.7x. Each costs another
# copy of the weights — ~92 MB for int8 — so the default of 2 is what fits a
# 512 MB instance while restoring the margin the design depends on.
_sessions = threading.local()
_session_count = 0
_count_lock = threading.Lock()

# Fastest first. The fp32 weights are ~2x faster than the int8 ones on CPU
# (RTF 0.42 vs 0.86 measured) — ONNX Runtime's dynamically-quantized MatMul
# kernels are slower than plain fp32 GEMM here — but cost ~325 MB instead of
# ~92 MB resident. The Docker image only bakes int8, so containers stay lean; a
# local checkout that has both automatically gets the fast one.
_MODEL_PREFERENCE = ("kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx", "kokoro-v1.0.int8.onnx")


def resolve_model_path() -> str:
    """Path to the Kokoro weights to load.

    An explicit KOKORO_MODEL wins if that file exists; otherwise fall back to
    the fastest of the weights this checkout/image actually has.
    """
    if KOKORO_MODEL != "auto":
        for base in (Path(KOKORO_MODEL), APP_DIR / KOKORO_MODEL):
            if base.exists():
                return str(base)
    for candidate in _MODEL_PREFERENCE:
        path = APP_DIR / candidate
        if path.exists():
            return str(path)
    # Nothing on disk — let Kokoro raise a clear "file not found" for the
    # configured name rather than failing somewhere more confusing.
    return KOKORO_MODEL if KOKORO_MODEL != "auto" else _MODEL_PREFERENCE[-1]


def _voices_path() -> str:
    path = Path(KOKORO_VOICES)
    return str(path if path.exists() else APP_DIR / KOKORO_VOICES)


def get_kokoro():
    """This thread's Kokoro session, created on first use.

    Thread-local rather than a shared singleton: ONNX Runtime can run one
    session concurrently, but the Kokoro wrapper around it carries per-call
    state through its phonemizer, so sharing one across the synthesis pool
    would be a correctness risk for no gain — the point of the pool is to have
    several sessions running at once anyway.
    """
    global _session_count
    session = getattr(_sessions, "kokoro", None)
    if session is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ONNX_THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The arena is why a warmed session costs far more than its 92 MB of
        # weights. See config.TTS_MEM_ARENA for the measured trade.
        opts.enable_cpu_mem_arena = TTS_MEM_ARENA
        model_path = resolve_model_path()
        ort_session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        session = Kokoro.from_session(ort_session, _voices_path())
        _sessions.kokoro = session
        with _count_lock:
            _session_count += 1
            print(
                f"[tts] {Path(model_path).name} session {_session_count} "
                f"on {ONNX_THREADS} thread(s)"
            )
    return session


def session_count() -> int:
    return _session_count


def voices() -> list[str]:
    return sorted(get_kokoro().get_voices())


# Kokoro pads every utterance with silence — measured on this voice, ~25 ms
# before and ~110 ms after. Harmless in a single clip, but sentence streaming
# butt-joins clips, so every seam inherits both: ~135 ms of dead air between
# each sentence, on top of any gap from synthesis running late.
#
# The padding is removed and replaced with one deliberate, uniform pause. A
# short gap between sentences is *wanted* — speech without one sounds rushed and
# breathless — but it should be a chosen duration, identical at every seam,
# rather than whatever the vocoder happened to emit.
SILENCE_THRESHOLD = 0.02   # fraction of peak amplitude
SENTENCE_PAUSE_S = 0.06    # the deliberate seam

# Bump whenever a change alters the audio a given request produces, so stale
# clips cannot be served from an existing cache. 2 = edge-trimmed.
AUDIO_FORMAT_VERSION = 2


def _trim_edges(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Strip the vocoder's leading/trailing silence, then add a uniform pause."""
    if samples.size == 0:
        return samples
    amplitude = np.abs(samples)
    peak = float(amplitude.max())
    if peak <= 0:
        return samples
    voiced = np.nonzero(amplitude > peak * SILENCE_THRESHOLD)[0]
    if voiced.size == 0:
        return samples
    # A few milliseconds either side of the detected speech, so the trim never
    # clips a soft consonant onset or a trailing fricative.
    guard = int(0.012 * sample_rate)
    start = max(0, int(voiced[0]) - guard)
    end = min(len(samples), int(voiced[-1]) + guard)
    trimmed = samples[start:end]
    pause = np.zeros(int(SENTENCE_PAUSE_S * sample_rate), dtype=trimmed.dtype)
    return np.concatenate([trimmed, pause])


# espeak-ng is a C library with process-global state, and Kokoro's phonemizer
# calls straight into it. Two threads phonemizing at once corrupt each other:
# running the same six sentences serially failed 0 times, and concurrently
# raised "number of lines in input and output must be equal, we have: input=1,
# output=3" intermittently — a mangled phoneme string, not a crash, which is the
# dangerous kind of failure because a clip that *does* come back is wrong.
#
# So phonemization is serialized and inference is not. That split is what makes
# parallel synthesis safe: espeak is pure text processing and quick, while the
# vocoder pass is the slow part and genuinely does run concurrently.
_phonemize_lock = threading.Lock()


def _synth_kokoro(text: str, voice: str, lang: str, speed: float) -> bytes:
    kokoro = get_kokoro()
    with _phonemize_lock:
        phonemes = kokoro.tokenizer.phonemize(text, lang)
    samples, sample_rate = kokoro.create(
        phonemes, voice=voice, speed=speed, lang=lang, is_phonemes=True
    )
    if TRIM_SILENCE:
        samples = _trim_edges(samples, sample_rate)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _to_wav(encoded: bytes, speed: float) -> bytes:
    """Decode a remote backend's compressed audio into this service's WAV format.

    Every clip in the system is 24 kHz PCM-16 WAV, and keeping that true for the
    remote voices too is worth the decode. Two reasons, and the second is the
    one that matters:

    * The client hands each clip straight to ``decodeAudioData``, which is
      format-agnostic — so this is not about playback.
    * ``_trim_edges`` is. Sentence streaming butt-joins clips on the audio
      timeline, so every clip's own leading and trailing silence becomes a gap
      at a seam. The trimmer works on samples, so a clip that stays compressed
      cannot be trimmed, and Tamil would inherit a pause at every sentence
      boundary that no other language has.

    The size cost is real (a 26 KB MP3 becomes ~210 KB of WAV) but it is exactly
    what English already ships today, and it is paid once per sentence ever
    because the result goes in the disk cache.
    """
    from faster_whisper.audio import decode_audio

    samples = decode_audio(io.BytesIO(encoded), sampling_rate=24000)
    if TRIM_SILENCE:
        samples = _trim_edges(samples, 24000)
    buf = io.BytesIO()
    sf.write(buf, samples, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _synth_edge(text: str, voice: str, speed: float) -> bytes:
    """Microsoft Edge's neural voices — used for Tamil (see config).

    Synchronous on purpose despite the library being async: this is called from
    the synthesis thread pool, where every other backend is blocking too, and
    handing an event loop up through `cached_wav` would mean rewriting the pool,
    the prewarm thread and the streaming generator around it for no gain.
    """
    import asyncio

    import edge_tts

    # edge-tts expresses rate as a percentage delta from the voice's natural
    # pace. Kokoro's 0.9 is "a touch slower, calm and gentle"; the same intent
    # here is -10%.
    rate = f"{round((speed - 1.0) * 100):+d}%"

    async def run() -> bytes:
        chunks: list[bytes] = []
        async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    # A fresh loop per call rather than asyncio.run() at import scope: these run
    # on pool threads, which have no running loop of their own, and asyncio.run
    # refuses to nest inside one if this is ever called from an async context.
    loop = asyncio.new_event_loop()
    try:
        mp3 = loop.run_until_complete(run())
    finally:
        loop.close()

    if not mp3:
        raise RuntimeError("edge-tts returned no audio")
    return _to_wav(mp3, speed)


def _synth_gemini(text: str, voice: str, speed: float) -> bytes:
    """Gemini's TTS models. Highest measured fidelity, 3 requests/minute free.

    Kept behind the same interface as the others so a paid key can switch to it
    with an env var, but not the default — see config.TTS_VOICE_OVERRIDES for
    why the quota makes it unusable for streamed answers on the free tier.
    """
    import wave

    from google.genai import types

    from kb import gemini

    response = gemini.with_retry(
        lambda: gemini.client().models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
            ),
        ),
        what="tts",
        retries=2,
    )
    pcm = response.candidates[0].content.parts[0].inline_data.data
    if not pcm:
        raise RuntimeError("gemini tts returned no audio")
    # Raw 24 kHz PCM-16 mono, so it only needs a container.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(pcm)
    return _to_wav(buf.getvalue(), speed)


def _synth_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    """Synthesize one clip with whichever backend `voice` names.

    The backend is encoded in the voice string ("edge:ta-IN-PallaviNeural")
    rather than passed separately, because the voice is already part of the
    disk-cache key. That means switching Tamil to a neural voice gives it fresh
    keys while leaving every English, Hindi and Malay clip already on disk
    reachable — where bumping AUDIO_FORMAT_VERSION would have thrown away the
    whole cache to change one language.
    """
    backend, _, name = voice.partition(":")
    if not name:
        return _synth_kokoro(text, voice, lang, speed)
    if backend == "edge":
        return _synth_edge(text, name, speed)
    if backend == "gemini":
        return _synth_gemini(text, name, speed)
    raise ValueError(f"unknown TTS backend {backend!r} in voice {voice!r}")


def voice_for(lang_code: str) -> str:
    """The voice to synthesize `lang_code` with.

    Returns a plain Kokoro voice name, or a "backend:voice" string for a
    language that config delegates elsewhere.
    """
    return TTS_VOICE_OVERRIDES.get(lang_code, TTS_VOICE)


def backend_of(voice: str) -> str:
    return voice.partition(":")[0] if ":" in voice else "kokoro"


def cached_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    """WAV bytes for this exact request, synthesizing only on a cache miss.

    Shared by the /tts endpoints and the startup prewarm so both write the same
    cache keys — that is what makes a prewarmed answer a pure disk read at
    request time.
    """
    # The format version is part of the key. Edge-trimming changed the bytes a
    # given request produces, and without this an old cache would keep serving
    # untrimmed clips forever — the seams would stay audible on exactly the
    # deployments that had been running longest.
    key = hashlib.sha256(
        f"v{AUDIO_FORMAT_VERSION}|{voice}|{lang}|{speed}|{text}".encode("utf-8")
    ).hexdigest()
    path = CACHE_DIR / f"{key}.wav"
    if path.exists():
        if debug.ENABLED:
            debug.log(f"[tts] cache HIT  {len(text):4d} ch  {key[:8]}  {text[:60]!r}")
        return path.read_bytes()

    started = time.perf_counter()
    try:
        data = _synth_wav(text, voice, lang, speed)
    except Exception as exc:  # noqa: BLE001 - the avatar must never go silent
        if backend_of(voice) == "kokoro":
            raise
        # A remote voice failed: no network, a changed upstream, an exhausted
        # quota. Kokoro can always speak this text — accented for Tamil, which
        # is the whole reason the override exists, but intelligible and instant.
        print(f"[tts] {voice} failed ({type(exc).__name__}: {exc}); using Kokoro")
        debug.log(f"[tts] {voice} failed -> kokoro fallback")
        # Deliberately NOT cached. Writing it under the neural voice's key would
        # pin the degraded clip forever, so a transient outage would permanently
        # cost Tamil the voice it was configured to use. Returning it uncached
        # means the next request tries the real backend again.
        return _synth_kokoro(text, TTS_VOICE, lang, speed)

    if debug.ENABLED:
        seconds = (len(data) - 44) / (24000 * 2)
        elapsed = time.perf_counter() - started
        debug.log(
            f"[tts] SYNTH {len(text):4d} ch -> {seconds:5.2f}s audio in {elapsed:5.2f}s "
            f"(RTF {elapsed / max(seconds, 0.01):.2f}) voice={voice} {text[:50]!r}"
        )
    # Write via a temp file + rename so a concurrent reader never sees a
    # half-written clip (two requests for the same text can race here).
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return data


_executor = None
_executor_lock = threading.Lock()


def _pool():
    """The synthesis pool — created once and reused for the process lifetime.

    It has to be persistent, and the reason is easy to miss: sessions are
    thread-local, so a fresh ``ThreadPoolExecutor`` per request means fresh
    threads, which means a fresh Kokoro session built on every single call.
    That leaked ~92 MB per request and made each one pay first-inference warmup
    again — the opposite of what the pool is for. Reusing the threads is what
    keeps exactly ``SYNTH_WORKERS`` sessions alive.
    """
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                from concurrent.futures import ThreadPoolExecutor

                _executor = ThreadPoolExecutor(
                    max_workers=SYNTH_WORKERS, thread_name_prefix="tts"
                )
    return _executor


def warm_sessions() -> None:
    """Build and prime every pooled session before the first real request.

    Priming matters as much as loading: ONNX Runtime allocates its arenas on the
    first inference, so an unwarmed session charges that cost to whichever
    sentence arrives first — measured at roughly double the steady-state time.
    Warming one session is not enough once there are several, because each
    allocates its own.
    """
    if SYNTH_WORKERS <= 1:
        get_kokoro()
        _synth_wav("Warming up.", "hf_alpha", "en-us", 1.0)
        return

    # A barrier, not `map`: submitting N tasks to N threads does not guarantee
    # each thread runs one — a thread that finishes early takes the next task
    # and leaves a sibling session cold, which then charges its warmup to a real
    # sentence later. Holding every task until all N have started forces each
    # thread to build its own session.
    barrier = threading.Barrier(SYNTH_WORKERS, timeout=120)

    def warm(_):
        get_kokoro()  # build this thread's session
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        _synth_wav("Warming up.", "hf_alpha", "en-us", 1.0)

    pool = _pool()
    futures = [pool.submit(warm, i) for i in range(SYNTH_WORKERS)]
    for future in futures:
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001 - warming must never block boot
            print(f"[tts] session warm failed: {exc}")


def synthesize_ordered(sentences: list[str], voice: str, lang: str, speed: float):
    """Yield each sentence's WAV in order, synthesizing several at once.

    The generator this replaced was strictly serial — synthesize, yield,
    synthesize — so the stream advanced at exactly the speed of one session. At
    the measured int8 rate of RTF 0.95 that is slower than the listener consumes
    it, and the gap it opens is audible at every sentence.

    Work is submitted for all sentences immediately and collected in order, so
    later sentences are already being rendered while the first is still playing.
    Order is preserved because the client schedules clips back to back on its
    audio timeline; delivering clip 3 before clip 2 would reorder the answer.

    A failed clip yields ``None`` rather than raising: one bad sentence should
    cost a sentence, not the rest of the reply.
    """
    if not sentences:
        return
    if SYNTH_WORKERS <= 1 or len(sentences) == 1:
        for sentence in sentences:
            try:
                yield cached_wav(sentence, voice, lang, speed)
            except Exception as exc:  # noqa: BLE001
                print(f"[tts] sentence failed ({exc}); skipping")
                yield None
        return

    # Everything is submitted at once, and that is a deliberate choice against
    # the tempting alternative of rendering the opening clip exclusively first.
    #
    # Giving clip 1 the whole machine does reach first audio sooner — measured
    # 3.4s against 4.7s — but it delays every other clip until clip 1 is
    # finished, and the answer then runs dry mid-sentence: 1.3 SECONDS of
    # silence at the first seam, against none when all clips overlap. A listener
    # forgives a slightly later start; a gap in the middle of a sentence is the
    # exact defect this is meant to remove.
    futures = [_pool().submit(cached_wav, s, voice, lang, speed) for s in sentences]
    for future in futures:
        try:
            yield future.result()
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] sentence failed ({exc}); skipping")
            yield None
