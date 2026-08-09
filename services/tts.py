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

import soundfile as sf

from config import APP_DIR, CACHE_DIR, KOKORO_MODEL, KOKORO_VOICES, ONNX_THREADS
from kb import debug

_kokoro = None
_kokoro_lock = threading.Lock()

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
    """Kokoro on a hand-tuned ONNX Runtime session (see ONNX_THREADS)."""
    global _kokoro
    if _kokoro is None:
        with _kokoro_lock:
            if _kokoro is None:
                import onnxruntime as ort
                from kokoro_onnx import Kokoro

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = ONNX_THREADS
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                model_path = resolve_model_path()
                session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                print(f"[tts] {Path(model_path).name} on {ONNX_THREADS} thread(s)")
                _kokoro = Kokoro.from_session(session, _voices_path())
    return _kokoro


def voices() -> list[str]:
    return sorted(get_kokoro().get_voices())


def _synth_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    samples, sample_rate = get_kokoro().create(text, voice=voice, speed=speed, lang=lang)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def cached_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    """WAV bytes for this exact request, synthesizing only on a cache miss.

    Shared by the /tts endpoints and the startup prewarm so both write the same
    cache keys — that is what makes a prewarmed answer a pure disk read at
    request time.
    """
    key = hashlib.sha256(f"{voice}|{lang}|{speed}|{text}".encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.wav"
    if path.exists():
        if debug.ENABLED:
            debug.log(f"[tts] cache HIT  {len(text):4d} ch  {key[:8]}  {text[:60]!r}")
        return path.read_bytes()

    started = time.perf_counter()
    data = _synth_wav(text, voice, lang, speed)
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
