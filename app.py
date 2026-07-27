"""SIA voice service — open-source TTS + STT behind one small FastAPI app.

TTS  : Kokoro-82M via kokoro-onnx (torch-free, ONNX Runtime). The voice
       ``hf_alpha`` is a Hindi female speaker, so English text comes out in a
       warm Indian-accented "girl" voice — identical on every browser/device
       because the client just plays the returned audio.
STT  : faster-whisper (CTranslate2). Robust on Indian-accented English and
       works from any browser (the client records audio and POSTs it here).

Both models are loaded lazily and cached; synthesized clips are cached on disk
keyed by (voice, lang, speed, text). Because the knowledge base is a *fixed*
set of answers, the service pre-synthesizes all of them on startup (PREWARM),
so by the time a visitor asks anything the reply is already a disk hit and the
avatar starts speaking immediately instead of waiting on the vocoder.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from knowledge import GREETING, public_suggestions, resolve_answer, spoken_texts

APP_DIR = Path(__file__).resolve().parent

# --- configuration (all overridable via environment variables) ---------------
# "auto" picks the fastest weights actually present (see _resolve_model_path).
# Pin it to kokoro-v1.0.int8.onnx on a memory-constrained host.
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "auto")
KOKORO_VOICES = os.getenv("KOKORO_VOICES", "voices-v1.0.bin")
# hf_alpha = Hindi female → Indian-accented English. lang="en-us" keeps English
# pronunciation correct while the speaker embedding supplies the Indian accent.
# (Set TTS_LANG=hi for a stronger Hindi phonology if you prefer.)
TTS_VOICE = os.getenv("TTS_VOICE", "hf_alpha")
TTS_LANG = os.getenv("TTS_LANG", "en-us")
TTS_SPEED = float(os.getenv("TTS_SPEED", "0.9"))  # a touch slower → calm, gentle

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny|base|small ...
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")
WHISPER_LANG = os.getenv("WHISPER_LANG", "en")

# ONNX Runtime intra-op threads. Kokoro is a small model: throughput peaks
# around 4 threads and *regresses* past that (measured RTF on a 12-core CPU:
# 1 thread 3.1, 4 threads 0.42, 12 threads 0.60), because the per-op sync cost
# outweighs the extra parallelism. 0/unset → pick a sane value for this host.
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "0")) or max(1, min(4, os.cpu_count() or 1))

# Default next to the app (not /tmp, which isn't a real path on Windows) so the
# cache survives restarts. Docker/Render override it with TTS_CACHE_DIR.
CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", str(APP_DIR / "tts-cache")))
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

# Pre-synthesize the knowledge base at boot (background thread). This is what
# makes the site feel instant; turn it off (PREWARM=0) only on a host too small
# to hold the models. WARMUP=1 is honoured as the older name for the same flag.
PREWARM = os.getenv("PREWARM", os.getenv("WARMUP", "1")) != "0"
# Also load Whisper at boot. Costs ~200 MB resident but takes the model load
# off the first mic press.
PREWARM_STT = os.getenv("PREWARM_STT", "1") != "0"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _start_prewarm()
    yield


app = FastAPI(title="SIA Voice Service", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cors_origin(request: Request) -> str:
    origin = request.headers.get("origin", "")
    if "*" in ALLOW_ORIGINS or not ALLOW_ORIGINS:
        return "*"
    return origin if origin in ALLOW_ORIGINS else ALLOW_ORIGINS[0]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Guarantee CORS headers even on unexpected 500s, so a real server error
    never masquerades as a CORS problem in the browser. (A bare 500 from the
    default handler skips the CORS middleware.)"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {exc}"},
        headers={"Access-Control-Allow-Origin": _cors_origin(request)},
    )


# --- lazy model singletons ---------------------------------------------------
_kokoro = None
_whisper = None
_kokoro_lock = threading.Lock()
_whisper_lock = threading.Lock()

# Fastest first. The fp32 weights are ~2x faster than the int8 ones on CPU
# (RTF 0.42 vs 0.86 measured) — ONNX Runtime's dynamically-quantized MatMul
# kernels are slower than plain fp32 GEMM here — but cost ~325 MB instead of
# ~92 MB resident. The Docker image only bakes int8, so containers stay lean;
# a local checkout that has both automatically gets the fast one.
_MODEL_PREFERENCE = ("kokoro-v1.0.onnx", "kokoro-v1.0.fp16.onnx", "kokoro-v1.0.int8.onnx")


def _resolve_model_path() -> str:
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
                model_path = _resolve_model_path()
                session = ort.InferenceSession(
                    model_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
                print(f"[tts] {Path(model_path).name} on {ONNX_THREADS} thread(s)")
                _kokoro = Kokoro.from_session(session, _voices_path())
    return _kokoro


def get_whisper():
    global _whisper
    if _whisper is None:
        with _whisper_lock:
            if _whisper is None:
                from faster_whisper import WhisperModel

                _whisper = WhisperModel(
                    WHISPER_MODEL,
                    device="cpu",
                    compute_type=WHISPER_COMPUTE,
                    cpu_threads=int(os.getenv("CPU_THREADS", str(ONNX_THREADS))),
                )
                print(f"[stt] faster-whisper '{WHISPER_MODEL}' ({WHISPER_COMPUTE})")
    return _whisper


# --- startup prewarm ---------------------------------------------------------
# Progress is exposed on /health so you can tell "still warming" apart from
# "broken" without reading the logs.
_prewarm_status = {"enabled": PREWARM, "done": 0, "total": 0, "ready": not PREWARM, "error": None}


def _prewarm_worker() -> None:
    texts = spoken_texts()
    _prewarm_status["total"] = len(texts)
    started = time.perf_counter()
    try:
        for text in texts:
            _cached_wav(text, TTS_VOICE, TTS_LANG, TTS_SPEED)
            _prewarm_status["done"] += 1
        if PREWARM_STT:
            get_whisper()
    except Exception as exc:  # noqa: BLE001 - never take the service down
        _prewarm_status["error"] = str(exc)
        print(f"[prewarm] failed: {exc}")
        return
    _prewarm_status["ready"] = True
    # Keep log output ASCII: a Windows console defaults to cp1252 and raises
    # UnicodeEncodeError on anything fancier.
    print(
        f"[prewarm] {_prewarm_status['done']}/{len(texts)} clips cached "
        f"in {time.perf_counter() - started:.1f}s -> {CACHE_DIR}"
    )


def _start_prewarm() -> None:
    """Synthesize every knowledge-base answer into the disk cache at boot.

    Runs in a daemon thread so it never blocks the health check or the first
    request — an uncached request just synthesizes on demand as before.
    """
    if not PREWARM:
        return
    threading.Thread(target=_prewarm_worker, name="prewarm", daemon=True).start()


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang: Optional[str] = None
    speed: Optional[float] = None


class AskRequest(BaseModel):
    question: str


def _synth_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    samples, sample_rate = get_kokoro().create(text, voice=voice, speed=speed, lang=lang)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _cached_wav(text: str, voice: str, lang: str, speed: float) -> bytes:
    """WAV bytes for this exact request, synthesizing only on a cache miss.

    Shared by /tts and the startup prewarm so both write the same cache keys —
    that's what makes a prewarmed answer a pure disk read at request time.
    """
    key = hashlib.sha256(f"{voice}|{lang}|{speed}|{text}".encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.wav"
    if path.exists():
        return path.read_bytes()

    data = _synth_wav(text, voice, lang, speed)
    # Write via a temp file + rename so a concurrent reader never sees a
    # half-written clip (two requests for the same text can race here).
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return data


@app.get("/health")
def health():
    return {
        "ok": True,
        "tts_voice": TTS_VOICE,
        "tts_lang": TTS_LANG,
        "tts_model": Path(_resolve_model_path()).name,
        "onnx_threads": ONNX_THREADS,
        "whisper_model": WHISPER_MODEL,
        "prewarm": dict(_prewarm_status),
    }


@app.get("/voices")
def voices():
    """Handy for confirming hf_alpha is available on this build."""
    return {"voices": sorted(get_kokoro().get_voices())}


@app.get("/content")
def content():
    """Greeting + suggestion chips for the frontend to render on load."""
    return {"greeting": GREETING, "suggestions": public_suggestions()}


@app.post("/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` is required")
    return {"answer": resolve_answer(question)}


@app.post("/tts")
def tts(req: TtsRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` is required")

    voice = req.voice or TTS_VOICE
    lang = req.lang or TTS_LANG
    speed = req.speed or TTS_SPEED

    try:
        data = _cached_wav(text, voice, lang, speed)
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        raise HTTPException(status_code=500, detail=f"tts failed: {exc}") from exc

    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def _transcribe(raw: bytes, suffix: str) -> str:
    """Blocking transcription — always call this off the event loop."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        segments, _info = get_whisper().transcribe(
            tmp_path,
            language=WHISPER_LANG,
            beam_size=1,
            vad_filter=True,  # drop silence → fewer hallucinated words
            condition_on_previous_text=False,  # short clips: no context to carry
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio upload")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    try:
        # Whisper is CPU-bound and synchronous; running it inline in this async
        # handler would block the event loop (and every other request) for the
        # length of the transcription.
        text = await run_in_threadpool(_transcribe, raw, suffix)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"stt failed: {exc}") from exc

    return {"text": text}
