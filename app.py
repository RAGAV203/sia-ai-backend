"""SIA voice service — open-source TTS + STT behind one small FastAPI app.

TTS  : Kokoro-82M via kokoro-onnx (torch-free, ONNX Runtime). The voice
       ``hf_alpha`` is a Hindi female speaker, so English text comes out in a
       warm Indian-accented "girl" voice — identical on every browser/device
       because the client just plays the returned audio.
STT  : faster-whisper (CTranslate2). Robust on Indian-accented English and
       works from any browser (the client records audio and POSTs it here).

Both models are loaded lazily and cached; synthesized clips are cached on disk
keyed by (voice, lang, speed, text), so the site's fixed set of answers is only
ever generated once.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from knowledge import GREETING, public_suggestions, resolve_answer

# --- configuration (all overridable via environment variables) ---------------
# int8 by default → low RAM so small instances don't OOM (which shows up as a
# 502 + a misleading "CORS missing" in the browser).
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "kokoro-v1.0.int8.onnx")
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

CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", "/tmp/tts-cache"))
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SIA Voice Service", version="1.0.0")
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


def _resolve_model_path() -> str:
    """Use the configured model if present, else whichever Kokoro model the
    image actually has (build-time download may have used a fallback)."""
    if Path(KOKORO_MODEL).exists():
        return KOKORO_MODEL
    for candidate in ("kokoro-v1.0.int8.onnx", "kokoro-v1.0.fp16.onnx", "kokoro-v1.0.onnx"):
        if Path(candidate).exists():
            return candidate
    return KOKORO_MODEL  # let Kokoro raise a clear error


def get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(_resolve_model_path(), KOKORO_VOICES)
    return _kokoro


def get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=WHISPER_COMPUTE)
    return _whisper


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


@app.get("/health")
def health():
    return {
        "ok": True,
        "tts_voice": TTS_VOICE,
        "tts_lang": TTS_LANG,
        "whisper_model": WHISPER_MODEL,
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

    key = hashlib.sha256(f"{voice}|{lang}|{speed}|{text}".encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{key}.wav"

    if path.exists():
        data = path.read_bytes()
    else:
        try:
            data = _synth_wav(text, voice, lang, speed)
        except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
            raise HTTPException(status_code=500, detail=f"tts failed: {exc}") from exc
        path.write_bytes(data)

    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio upload")

    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
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
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"stt failed: {exc}") from exc
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {"text": text}
