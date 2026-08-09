"""The conversational endpoints: what SIA offers, and what she answers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from config import KB_ENABLED
from knowledge import FALLBACK_ANSWER as KEYWORD_FALLBACK
from knowledge import GREETING, public_suggestions, resolve_answer
from services import runtime

router = APIRouter(tags=["chat"])


class AskRequest(BaseModel):
    question: str


@router.get("/content")
def content():
    """Greeting + suggestion chips for the frontend to render on load."""
    return {"greeting": GREETING, "suggestions": public_suggestions()}


@router.post("/ask")
def ask(req: AskRequest, request: Request):
    """Answer a question, grounded in the scraped college knowledge base.

    Degrades in stages rather than failing: knowledge base -> curated keyword
    answers -> a safe fallback line. The avatar always gets something to say.
    """
    if not (req.question or "").strip():
        raise HTTPException(status_code=400, detail="`question` is required")

    client = request.client.host if request.client else "unknown"
    if runtime.rate_limiter and not runtime.rate_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    from kb.guard import check_question

    verdict = check_question(req.question)
    if not verdict.ok:
        if verdict.reason == "prompt_extraction":
            # Refuse rather than sanitize-and-answer: honouring a rewritten
            # instruction is exactly the outcome being defended against.
            from kb.prompts import OFF_CONTRACT_ANSWER

            return {"answer": OFF_CONTRACT_ANSWER, "grounded": False, "reason": "refused"}
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
        debug.section(f"ASK: {question}")
        debug.log(f"intent={name} -> canned reply (no retrieval, no generation)")
        return {"answer": reply, "grounded": True, "reason": f"intent:{name}"}

    if not KB_ENABLED:
        return {"answer": resolve_answer(question), "grounded": True, "reason": "keyword_kb"}

    from kb import answering

    cache = runtime.answer_cache
    if cache:
        # The top retrieved chunk corroborates a merely-similar question, so a
        # paraphrase can reuse the cached answer without loosening the
        # similarity bar.
        #
        # Passed as a callable because computing it is no longer cheap. When
        # embeddings were a local ONNX model this was a few milliseconds; it is
        # now a metered API call against a 1000/day free-tier cap, and eager
        # evaluation spent one on every request — including the verbatim cache
        # hits that exist precisely to avoid touching the network.
        cached = cache.get(question, top_chunk_id=lambda: answering.top_chunk_id(question))
        if cached:
            debug.section(f"ASK: {question}")
            debug.log(
                f"answer cache HIT ({cached.get('match')}, sim={cached.get('similarity')}) "
                f"-> {cached.get('matched_question', '')[:60]!r}"
            )
            debug.block("CACHED ANSWER", cached.get("answer", ""))
            # Byte-identical to a previous answer, so its clips are already on disk.
            return cached

    result = answering.answer(question)

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
        curated = resolve_answer(question)
        if curated != KEYWORD_FALLBACK:
            return {
                "answer": curated,
                "grounded": True,
                "reason": f"keyword_kb:{result['reason']}",
            }

    if cache:
        # `put` stores only grounded answers, so out-of-scope replies and "I
        # don't have that detail" are never pinned to a question that a better
        # corpus would answer later.
        cache.put(question, result, top_chunk_id=result.get("top_chunk_id"))
    return result
