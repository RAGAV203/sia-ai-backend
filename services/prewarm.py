"""Boot-time warming: synthesize what SIA will say before anyone asks.

Synthesis, not generation, is the latency that matters here. Kokoro runs at
roughly 0.4x real time, so a fresh 40-second answer costs ~16 s of CPU before
the avatar makes a sound, while a Gemini answer arrives in under a second.
Pre-rendering every fixed string at boot turns the first visit into a disk read.

The suggestion chips get the same treatment one level up: they are answered
through the real pipeline at boot, which puts them in the semantic answer cache
*and* puts their sentences in the TTS cache. A tap on a chip then costs no model
call and no synthesis.

**Languages warm one after another, English first.** The work is now roughly
proportional to the number of languages — each one needs its own translations
and its own clips, because both caches are keyed on the exact text. Running them
serially in a fixed order is what keeps that affordable: English is complete
within seconds of boot, so the common case is instant, and Tamil, Hindi and
Malay fill in over the following minute while the service is already answering.
A visitor who picks Tamil during that window is not broken, only slower — their
answer synthesizes on demand exactly as it would have without any prewarm.
"""

from __future__ import annotations

import threading
import time

import languages
from config import (
    CACHE_DIR,
    PREWARM,
    PREWARM_LANGS,
    PREWARM_STT,
    TTS_SPEED,
    TTS_VOICE,
)
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
    "languages": {},
}


def _prewarm_answers(answer_cache) -> None:
    """Pre-answer the suggestion chips and cache both the text and its audio.

    English only, and deliberately so. The answer cache is keyed on the English
    question (see ``routers/chat``), so warming it once serves every language —
    what each language still needs is the *translation* of the resulting answer,
    which `_prewarm_language` picks up because the answers are already here.
    """
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


def _texts_for(lang: str, answer_cache) -> list[str]:
    """Every string SIA might say in `lang`, translated and ready to synthesize.

    The English set comes from ``knowledge.spoken_texts`` plus whatever the
    chips actually answered with, so a deployment whose corpus has changed warms
    the answers it will really give rather than the ones it used to.
    """
    english = list(spoken_texts())

    # The chip answers are only known after `_prewarm_answers` has run, and they
    # are the longest things SIA says — the ones where cold synthesis is most
    # obvious.
    if answer_cache is not None:
        for suggestion in public_suggestions():
            cached = answer_cache.get(suggestion["question"])
            if cached and cached.get("answer"):
                english.append(cached["answer"])

    # Deduplicate while preserving order: the greeting and several intent
    # replies overlap with chip answers on some corpora, and synthesizing the
    # same string twice is pure waste.
    seen: set[str] = set()
    unique = []
    for text in english:
        if text and text not in seen:
            seen.add(text)
            unique.append(text)

    if lang == languages.DEFAULT_LANG:
        return unique

    from services import translate as translate_service

    return [t for t in translate_service.to_language_many(unique, lang) if t]


def _prewarm_language(lang: str, answer_cache) -> None:
    language = languages.get(lang)
    texts = _texts_for(language.code, answer_cache)
    entry = {"done": 0, "total": len(texts), "ready": False}
    status["languages"][language.code] = entry
    # Counted as the language starts, not when it finishes. Accumulating it
    # afterwards made /health report `done` higher than `total` for the whole
    # time a language was warming — the completed languages were in `done` but
    # the one in progress had not yet contributed to `total`, which reads as a
    # corrupted counter rather than as progress.
    status["total"] += len(texts)

    for text in texts:
        # Synthesized per sentence, matching how `/tts/stream` will request it.
        # Warming the whole paragraph as one clip would fill the cache with keys
        # the streaming path never looks up.
        for sentence in split_sentences(text):
            cached_wav(sentence, TTS_VOICE, language.espeak, TTS_SPEED)
        entry["done"] += 1
        status["done"] += 1

    entry["ready"] = True


def _worker(answer_cache) -> None:
    wanted = [c for c in PREWARM_LANGS if c in languages.LANGUAGES] or [languages.DEFAULT_LANG]
    # English first whatever the env var says. It is the default the first
    # visitor lands on, and the ordering is the only thing making a multi-
    # language prewarm tolerable.
    ordered = [languages.DEFAULT_LANG] + [c for c in wanted if c != languages.DEFAULT_LANG]
    ordered = [c for c in ordered if c in wanted or c == languages.DEFAULT_LANG]

    started = time.perf_counter()
    try:
        # Build and prime every synthesis session first. Otherwise the first
        # streamed answer pays ONNX arena allocation on each session, which
        # measured as roughly double the steady-state synthesis time — and it
        # lands on the one clip the listener is actually waiting for.
        warm_sessions()

        # Before any language: the chip answers have to exist before they can be
        # translated or synthesized.
        _prewarm_answers(answer_cache)

        for lang in ordered:
            _prewarm_language(lang, answer_cache)
            elapsed = time.perf_counter() - started
            print(
                f"[prewarm] {lang}: {status['languages'][lang]['done']}"
                f"/{status['languages'][lang]['total']} texts cached "
                f"({elapsed:.1f}s elapsed)"
            )

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
        f"[prewarm] {status['done']} texts across {len(ordered)} language(s) "
        f"in {time.perf_counter() - started:.1f}s -> {CACHE_DIR}"
    )


def start(answer_cache=None) -> None:
    """Warm in a daemon thread so it never blocks health checks or first request."""
    if not PREWARM:
        return
    threading.Thread(target=_worker, args=(answer_cache,), name="prewarm", daemon=True).start()
