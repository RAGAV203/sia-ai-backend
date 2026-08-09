"""Boot-time warming: synthesize what SIA will say before anyone asks.

Synthesis, not generation, is the latency that matters here. Kokoro runs at
roughly 0.4x real time, so a fresh 40-second answer costs ~16 s of CPU before
the avatar makes a sound, while a Gemini answer arrives in under a second.
Pre-rendering every fixed string at boot turns the first visit into a disk read.

The suggestion chips get the same treatment one level up: they are answered
through the real pipeline at boot, which puts them in the semantic answer cache
*and* puts their sentences in the TTS cache. A tap on a chip then costs no model
call and no synthesis.
"""

from __future__ import annotations

import threading
import time

from config import CACHE_DIR, PREWARM, PREWARM_STT, TTS_LANG, TTS_SPEED, TTS_VOICE
from knowledge import public_suggestions, spoken_texts
from services.tts import cached_wav, warm_sessions
from speech import split_sentences

# Exposed on /health so "still warming" is distinguishable from "broken".
status: dict = {
    "enabled": PREWARM,
    "done": 0,
    "total": 0,
    "ready": not PREWARM,
    "error": None,
}


def _prewarm_answers(answer_cache) -> None:
    """Pre-answer the suggestion chips and cache both the text and its audio."""
    if answer_cache is None:
        return
    from kb import answering

    if not answering.available():
        return
    for suggestion in public_suggestions():
        question = suggestion["question"]
        if answer_cache.get(question):
            continue
        try:
            result = answering.answer(question)
        except Exception as exc:  # noqa: BLE001
            print(f"[prewarm] answer failed for {question[:40]!r}: {exc}")
            continue
        if result.get("grounded"):
            answer_cache.put(question, result, top_chunk_id=result.get("top_chunk_id"))
            for sentence in split_sentences(result["answer"]):
                cached_wav(sentence, TTS_VOICE, TTS_LANG, TTS_SPEED)


def _worker(answer_cache) -> None:
    texts = spoken_texts()
    status["total"] = len(texts)
    started = time.perf_counter()
    try:
        # Build and prime every synthesis session first. Otherwise the first
        # streamed answer pays ONNX arena allocation on each session, which
        # measured as roughly double the steady-state synthesis time — and it
        # lands on the one clip the listener is actually waiting for.
        warm_sessions()
        for text in texts:
            cached_wav(text, TTS_VOICE, TTS_LANG, TTS_SPEED)
            status["done"] += 1
        _prewarm_answers(answer_cache)
        if PREWARM_STT:
            from services.stt import get_whisper

            get_whisper()
    except Exception as exc:  # noqa: BLE001 - never take the service down
        status["error"] = str(exc)
        print(f"[prewarm] failed: {exc}")
        return
    status["ready"] = True
    # Keep log output ASCII: a Windows console defaults to cp1252 and raises
    # UnicodeEncodeError on anything fancier.
    print(
        f"[prewarm] {status['done']}/{len(texts)} clips cached "
        f"in {time.perf_counter() - started:.1f}s -> {CACHE_DIR}"
    )


def start(answer_cache=None) -> None:
    """Warm in a daemon thread so it never blocks health checks or first request."""
    if not PREWARM:
        return
    threading.Thread(target=_worker, args=(answer_cache,), name="prewarm", daemon=True).start()
