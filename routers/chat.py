"""The conversational endpoints: what SIA offers, and what she answers.

**Where the language boundary sits.** Everything between the two translation
calls in `ask()` is English and knows nothing about languages. That is the whole
design (see ``services/translate.py`` for why answering natively in Tamil breaks
the guards, the scope triage and the answer cache), and the ordering below is
what enforces it:

    guard -> intent -> TRANSLATE IN -> [English pipeline] -> TRANSLATE OUT

The two steps before the boundary are there deliberately. The guard is
script-agnostic and rejecting a prompt-extraction attempt should not cost a
translation call; the intent gate matches social turns in their own language, so
"நன்றி" is answered from a cached string without any model call at all.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import languages
from config import KB_ENABLED
from knowledge import FALLBACK_ANSWER as KEYWORD_FALLBACK
from knowledge import GREETING, public_suggestions, resolve_answer
from services import runtime
from services import translate as translate_service

router = APIRouter(tags=["chat"])


class AskRequest(BaseModel):
    question: str
    lang: Optional[str] = None


def _localize(result: dict, language) -> dict:
    """Translate a finished English answer into the visitor's language.

    Runs **after** every guard, which is the point: `check_answer` has already
    decided whether this text is speakable, whether it leaked the prompt, and
    whether it is really a refusal, all against the English it was calibrated
    on. Translation cannot change those verdicts because it cannot run before
    them.

    The English is kept alongside as `answer_en`. It costs a few bytes and makes
    a bad translation diagnosable from the client without server logs.
    """
    answer = (result.get("answer") or "").strip()
    if language.code == languages.DEFAULT_LANG or not answer:
        return {**result, "lang": language.code}

    localized = translate_service.to_language(answer, language.code)
    return {
        **result,
        "answer": localized,
        "answer_en": answer,
        "lang": language.code,
        # A translation that silently fell back to English is worth surfacing:
        # from the frontend it looks like the picker was ignored.
        "translated": localized != answer,
    }


@router.get("/content")
def content(lang: str = languages.DEFAULT_LANG):
    """Greeting + suggestion chips for the frontend to render on load.

    Both are fixed strings, so their translations are prewarmed at boot and this
    is a disk read in every language — which matters because it sits on the
    critical path of the very first paint.
    """
    language = languages.get(lang)
    suggestions = public_suggestions()

    if language.code == languages.DEFAULT_LANG:
        return {
            "greeting": GREETING,
            "suggestions": suggestions,
            "lang": language.code,
            "languages": languages.public_catalog(),
        }

    # One batched call rather than nine, and after the first boot it is nine
    # cache hits and no call at all.
    source = [GREETING]
    for s in suggestions:
        source.extend([s["chip"], s["question"]])
    rendered = translate_service.to_language_many(source, language.code)

    greeting = rendered[0]
    localized = []
    for i, s in enumerate(suggestions):
        localized.append(
            {
                "id": s["id"],
                "chip": rendered[1 + i * 2],
                # `question` is what the chip actually asks when tapped, so it
                # must be in the visitor's language too: it is echoed straight
                # into the transcript as their message.
                "question": rendered[2 + i * 2],
            }
        )

    return {
        "greeting": greeting,
        "suggestions": localized,
        "lang": language.code,
        "languages": languages.public_catalog(),
    }


@router.post("/ask")
def ask(req: AskRequest, request: Request):
    """Answer a question, grounded in the scraped college knowledge base.

    Degrades in stages rather than failing: knowledge base -> curated keyword
    answers -> a safe fallback line. The avatar always gets something to say.
    """
    if not (req.question or "").strip():
        raise HTTPException(status_code=400, detail="`question` is required")

    language = languages.get(req.lang)

    client = request.client.host if request.client else "unknown"
    if runtime.rate_limiter and not runtime.rate_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    from kb.guard import check_question

    # Script-agnostic: it normalizes, strips control characters and matches
    # prompt-extraction patterns, none of which assume English. Running it
    # before the translation means an attack never buys a model call, and — more
    # to the point — never gets laundered into English by one.
    verdict = check_question(req.question)
    if not verdict.ok:
        if verdict.reason == "prompt_extraction":
            # Refuse rather than sanitize-and-answer: honouring a rewritten
            # instruction is exactly the outcome being defended against.
            from kb.prompts import OFF_CONTRACT_ANSWER

            return _localize(
                {"answer": OFF_CONTRACT_ANSWER, "grounded": False, "reason": "refused"},
                language,
            )
        raise HTTPException(status_code=400, detail=f"invalid question ({verdict.reason})")

    question = verdict.value

    # Social turns are answered before anything else runs. The corpus cannot
    # answer "thanks" — there is no right chunk to retrieve — so retrieving
    # anyway returns whichever page is nearest and speaks it as fact. See
    # kb/intent.py for the testimonial this used to read out.
    from kb import debug, intent

    social = intent.match(question)
    if social:
        name, reply = social
        debug.section(f"ASK [{language.code}]: {question}")
        debug.log(f"intent={name} -> canned reply (no retrieval, no generation)")
        return _localize(
            {"answer": reply, "grounded": True, "reason": f"intent:{name}"}, language
        )

    # --- the language boundary ------------------------------------------------
    # Everything from here to `_localize` runs on English and is unchanged from
    # the monolingual service.
    english = translate_service.to_english(question, language.code)
    if english != question:
        debug.log(f"translated in [{language.code}]: {question!r} -> {english!r}")

    if not KB_ENABLED:
        return _localize(
            {"answer": resolve_answer(english), "grounded": True, "reason": "keyword_kb"},
            language,
        )

    from kb import answering

    cache = runtime.answer_cache
    if cache:
        # Keyed on the English question, so all four languages share one cache:
        # a Tamil visitor asking about fees hits the entry an English visitor
        # created, and gets the already-translated, already-synthesized answer.
        #
        # The top retrieved chunk corroborates a merely-similar question, so a
        # paraphrase can reuse the cached answer without loosening the
        # similarity bar.
        #
        # Passed as a callable because computing it is no longer cheap. When
        # embeddings were a local ONNX model this was a few milliseconds; it is
        # now a metered API call against a 1000/day free-tier cap, and eager
        # evaluation spent one on every request — including the verbatim cache
        # hits that exist precisely to avoid touching the network.
        cached = cache.get(english, top_chunk_id=lambda: answering.top_chunk_id(english))
        if cached:
            debug.section(f"ASK [{language.code}]: {english}")
            debug.log(
                f"answer cache HIT ({cached.get('match')}, sim={cached.get('similarity')}) "
                f"-> {cached.get('matched_question', '')[:60]!r}"
            )
            debug.block("CACHED ANSWER", cached.get("answer", ""))
            # Byte-identical to a previous answer, so its clips are already on
            # disk — in every language, because the translation is a pure
            # function of this string and is cached on it.
            return _localize(cached, language)

    result = answering.answer(english)

    # Fall back to the curated answers only when the pipeline could not *reach*
    # a judgement — no key, or a model error. Never when it reached one.
    #
    # This used to also fire on `no_sources`, which is how "write a python
    # program" came to be answered with the academic programmes blurb: retrieval
    # scored it 0.597, just under the gate, so the reason was `no_sources`, and
    # the curated keyword table matched the substring "program". A deliberate
    # decision that the corpus cannot answer something is not an invitation to
    # keyword-match it — and an out-of-scope verdict least of all.
    if result["reason"] in ("no_api_key", "model_error"):
        curated = resolve_answer(english)
        if curated != KEYWORD_FALLBACK:
            return _localize(
                {
                    "answer": curated,
                    "grounded": True,
                    "reason": f"keyword_kb:{result['reason']}",
                },
                language,
            )

    if cache:
        # `put` stores only grounded answers, so out-of-scope replies and "I
        # don't have that detail" are never pinned to a question that a better
        # corpus would answer later. Stored under the English question and with
        # the English answer, so the entry stays language-neutral.
        cache.put(english, result, top_chunk_id=result.get("top_chunk_id"))
    return _localize(result, language)
