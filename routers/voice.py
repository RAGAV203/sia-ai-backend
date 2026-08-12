"""The audio endpoints: synthesis in, transcription out."""

from __future__ import annotations

import struct
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import languages
from config import TTS_SPEED, TTS_VOICE
from services import stt as stt_service
from services import tts as tts_service
from speech import split_sentences

router = APIRouter(tags=["voice"])


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang: Optional[str] = None
    speed: Optional[float] = None


def _espeak(lang: Optional[str]) -> str:
    """Phonemizer language for a request's `lang` field.

    The wire now carries our two-letter codes ("ta"), but this endpoint took raw
    espeak codes ("en-us") before there was more than one language, and the TTS
    disk cache is keyed on whatever is passed down from here. Resolving through
    the registry — which accepts both spellings and returns the espeak code —
    keeps every clip synthesized before this change reachable, instead of
    orphaning the whole cache on a rename.
    """
    return languages.get(lang).espeak


@router.get("/voices")
def voices():
    """Handy for confirming hf_alpha is available on this build."""
    return {"voices": tts_service.voices()}


@router.get("/languages")
def language_catalog():
    """The languages this deployment speaks, for the frontend's picker."""
    return {"languages": languages.public_catalog(), "default": languages.DEFAULT_LANG}


@router.post("/tts")
def tts(req: TtsRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` is required")

    try:
        data = tts_service.cached_wav(
            text, req.voice or TTS_VOICE, _espeak(req.lang), req.speed or TTS_SPEED
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        raise HTTPException(status_code=500, detail=f"tts failed: {exc}") from exc

    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.post("/tts/stream")
def tts_stream(req: TtsRequest):
    """Synthesize sentence by sentence and stream each clip as it is ready.

    Whole-answer synthesis is the wrong shape for a voice UI: Kokoro runs at
    ~0.4x real time, so a 40-second answer means ~16 seconds of silence first.
    Sentence streaming turns that into ~2 seconds — the listener hears sentence
    one while sentence two renders, and because the real-time factor is below 1,
    synthesis stays ahead of playback for the rest of the answer.

    Wire format is a sequence of ``[4-byte big-endian length][WAV bytes]``
    frames. Each frame is a complete WAV file so the browser can hand it
    straight to ``decodeAudioData`` without reassembling anything.

    Each sentence is cached individually, so stock phrasings ("Is there anything
    else I can help with?") are already on disk across different answers — a
    finer-grained hit than the whole-answer cache can give.

    Clips are synthesized **several at a time** and delivered in order. The
    serial version of this loop only advanced as fast as one Kokoro session,
    which measured slower than real time on int8 (RTF 0.95) — so the client ran
    out of audio mid-answer and a gap opened at every sentence boundary. See
    ``config.SYNTH_WORKERS``.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` is required")

    voice = req.voice or TTS_VOICE
    lang = _espeak(req.lang)
    speed = req.speed or TTS_SPEED
    sentences = split_sentences(text)

    def frames():
        for wav in tts_service.synthesize_ordered(sentences, voice, lang, speed):
            if wav is None:
                continue  # one bad clip must not kill the stream
            yield struct.pack(">I", len(wav)) + wav

    return StreamingResponse(
        frames(),
        media_type="application/octet-stream",
        headers={
            "X-Sentence-Count": str(len(sentences)),
            "Cache-Control": "no-store",
            # Without this an intermediate proxy may buffer the whole response
            # and re-create exactly the delay this endpoint exists to remove.
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stt")
async def stt(audio: UploadFile = File(...), lang: str = Form(default=languages.DEFAULT_LANG)):
    """Transcribe a clip. `lang` is what the speaker is speaking, not a hint.

    Both engines need telling. Whisper will auto-detect, but on a short noisy
    clip it detects badly and a wrong language guess produces confident nonsense
    rather than an error; Gemini needs the script named or it romanizes Tamil.
    """
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio upload")

    try:
        # Decoding and transcription are both blocking; running them inline in
        # this async handler would stall the event loop — and every other
        # request — for the length of the call.
        result = await run_in_threadpool(
            stt_service.transcribe, raw, audio.filename or "", lang
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"stt failed: {exc}") from exc

    # `text` stays the contract the client reads; the rest is diagnostic, and
    # tells you which engine produced a bad transcript without reading logs.
    return result
