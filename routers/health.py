"""Health and introspection.

Deliberately detailed. The failure this service is most prone to is a *silent
degradation* — the API still answers, but from the curated fallback instead of
the knowledge base, or with the browser's voice instead of Kokoro — and from the
frontend those look like "the avatar sounds wrong", not like an error. Every
subsystem reports what it can actually do right now, so one request distinguishes
"warming up" from "misconfigured" from "broken".
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from config import KB_ENABLED, ONNX_THREADS, TTS_LANG, TTS_VOICE
from services import prewarm, runtime
from services import stt as stt_service
from services import tts as tts_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {
        "ok": True,
        "tts": {
            "voice": TTS_VOICE,
            "lang": TTS_LANG,
            "model": Path(tts_service.resolve_model_path()).name,
            "onnx_threads": ONNX_THREADS,
        },
        "stt": stt_service.status(),
        "prewarm": dict(prewarm.status),
        "knowledge_base": _kb_status(),
    }


def _kb_status() -> dict:
    """What the answering path can actually do right now."""
    if not KB_ENABLED:
        return {"enabled": False}
    if runtime.kb_error:
        return {"enabled": True, "error": runtime.kb_error}
    try:
        from kb import answering, embed, gemini
        from kb.index import load as load_index

        index = load_index()
        return {
            "enabled": True,
            "chunks": len(index.chunks) if index else 0,
            "backend": answering.backend_name(),
            "model": answering.model_name(),
            "ready": answering.available(),
            "top_k": answering.TOP_K,
            "gate": answering.MIN_SIMILARITY,
            "embeddings": embed.status(),
            "gemini": gemini.status(),
            "answer_cache": runtime.answer_cache.stats() if runtime.answer_cache else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}
