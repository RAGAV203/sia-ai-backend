"""Grounded answering: retrieve, then answer only from what was retrieved.

Two interchangeable backends behind one interface:

* ``local`` (default) — Qwen2.5-1.5B-Instruct via llama.cpp on CPU. No API key,
  no network, ~1 GB resident. See ``llm_local.py`` for how the model was chosen
  and what it costs in latency and injection resistance.
* ``anthropic`` — the Claude API, for when quality matters more than
  self-containment.

Retrieval is tuned per backend. Local CPU inference is dominated by prefill at
roughly 7 ms per prompt token, so the local path passes fewer and shorter chunks
(measured: 3 chunks ≈ 6 s, 6 chunks ≈ 12 s). The API path has no such pressure
and passes more context for better recall.
"""

from __future__ import annotations

import os
import threading

from . import debug
from .embed import model_name as embed_model_name
from .guard import check_answer
from .index import load as load_index
from .prompts import FALLBACK_ANSWER, build_user_turn, system_prompt

BACKEND = os.getenv("LLM_BACKEND", "local").strip().lower()

# The relevance gate is a *cost* filter, not the off-topic defense.
#
# It cannot be the defense, because retrieval scores do not separate on- from
# off-topic on this corpus. Measured across four candidate signals, every one
# overlapped:
#
#   top-1 cosine  on-topic 0.569-0.699   off-topic 0.422-0.583
#   z-score       on-topic 2.222-5.129   off-topic 2.290-3.639
#   margin@11     on-topic 0.034-0.156   off-topic 0.021-0.081
#   bm25 top      on-topic 0.858-9.382   off-topic 0.000-8.643
#
# That is not a tuning failure — the corpus genuinely contains sports results,
# poetry events and student achievements, so "who won the cricket world cup"
# really does match a cricket tournament page. No threshold can fix that.
#
# What does work is the model: given the retrieved chunks, it declined 4/4
# off-topic questions ("the reference material does not contain information
# about..."). So the gate is set low, purely to skip a pointless generation when
# nothing matches at all, and the model makes the actual judgement with the
# evidence in front of it. Erring low costs a few seconds; erring high silently
# refuses questions the corpus can answer.
#
# Scales differ per encoder, so the default follows the selected model.
_GATE_BY_MODEL = {"minilm": 0.30, "multiqa": 0.30, "bge": 0.30}
MIN_SIMILARITY = float(
    os.getenv("RETRIEVAL_MIN_SIMILARITY", str(_GATE_BY_MODEL.get(embed_model_name(), 0.30)))
)

# Local `top_k` was raised 3 -> 5 against tests/eval_accuracy.py, which scores
# retrieval at the settings the service actually serves:
#
#   k=3 per_doc=2   recall 11/15 (73%)   2.73 distinct pages
#   k=4 per_doc=2   recall 11/15 (73%)   3.33
#   k=5 per_doc=2   recall 12/15 (80%)   4.07     <- chosen
#
# The cost is prefill: ~7 ms per prompt token on CPU, so two more 700-char
# chunks add roughly 2.5s before the first word is spoken. Accepted deliberately
# — a wrong answer is worse than a slower one here, and the two paths that
# dominate real traffic are unaffected: the suggestion chips are prewarmed at
# boot, and repeat questions are served verbatim from the semantic cache.
_LOCAL_DEFAULTS = {"top_k": 5, "chunk_chars": 700}
_API_DEFAULTS = {"top_k": 6, "chunk_chars": 1000}
_defaults = _LOCAL_DEFAULTS if BACKEND == "local" else _API_DEFAULTS

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", str(_defaults["top_k"])))
MAX_CHUNK_CHARS = int(os.getenv("RETRIEVAL_CHUNK_CHARS", str(_defaults["chunk_chars"])))

# How many chunks one page may occupy in a single answer's context.
#
# Pages here average 15k characters, so a page that matches at all usually
# matches several times and takes every slot: "do you provide job placement
# assistance" and "tell me about the placement cell" both retrieved the
# placements page three times over, spending the whole context on one page's
# worth of information.
#
# Measured on tests/eval_accuracy.py (15 cases, k=3):
#
#   per_doc  recall   distinct pages   depth on best page
#   none     11/15    2.60             3
#   2        11/15    2.73             2      <- chosen
#   1        11/15    3.00             1
#
# Recall is flat because it only asks whether the right page appeared at all —
# the gain is in what the model actually receives. 1 was rejected despite
# scoring best at higher k: it strips a topic page down to a single chunk, so
# "tell me about the library" arrived with one library chunk and three
# unrelated pages. 2 keeps enough of the winning page to answer from while
# still freeing a slot.
PER_DOC = int(os.getenv("RETRIEVAL_PER_DOC", "2"))

# --- Anthropic backend config (unused when BACKEND == "local") ---------------
MODEL = os.getenv("ANSWER_MODEL", "claude-opus-5")
EFFORT = os.getenv("ANSWER_EFFORT", "low")
MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "400"))

_client = None
_client_lock = threading.Lock()
_fallbacks_supported = True


def backend_name() -> str:
    return BACKEND


def available() -> bool:
    """True when the configured backend can actually answer."""
    if BACKEND == "local":
        from . import llm_local

        return llm_local.available()
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import anthropic

                _client = anthropic.Anthropic()
    return _client


def retrieve(question: str) -> list[dict]:
    """Retrieve supporting chunks, or nothing at all if the question is off-topic.

    The gate is applied to the *best* hit rather than per chunk: once the top
    match is clearly relevant, the weaker neighbours are still useful context,
    but if even the best match is poor the question isn't answerable from this
    corpus and the model should not be handed material to stretch.
    """
    index = load_index()
    if index is None:
        return []
    hits = index.search(question, k=TOP_K, per_doc=PER_DOC)
    if not hits or max(h.similarity for h in hits) < MIN_SIMILARITY:
        return []
    return [
        {
            "id": h.chunk["id"],
            "title": h.chunk["title"],
            "text": h.chunk["text"][:MAX_CHUNK_CHARS],
            "url": h.chunk["url"],
            "score": h.score,
            "similarity": h.similarity,
        }
        for h in hits
    ]


def top_chunk_id(question: str) -> str | None:
    """Id of the best-matching chunk, used to corroborate a semantic cache hit."""
    index = load_index()
    if index is None:
        return None
    hits = index.search(question, k=1)
    if not hits or hits[0].similarity < MIN_SIMILARITY:
        return None
    return hits[0].chunk["id"]


def _call_local(question: str, sources: list[dict]) -> str:
    from . import llm_local

    return llm_local.generate(system_prompt("local"), build_user_turn(question, sources))


def _call_anthropic(question: str, sources: list[dict]) -> str:
    global _fallbacks_supported
    client = _get_client()
    system = [{"type": "text", "text": system_prompt("anthropic"), "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": build_user_turn(question, sources)}]
    common = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
        output_config={"effort": EFFORT},
    )

    response = None
    if _fallbacks_supported:
        try:
            # Opus 5's classifiers can decline a request; "default" re-serves it
            # on Anthropic's recommended fallback inside the same call.
            response = client.beta.messages.create(
                **common,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except Exception as exc:  # noqa: BLE001 - beta may be unavailable on this key
            _fallbacks_supported = False
            print(f"[answer] server-side fallbacks unavailable ({type(exc).__name__}); plain endpoint")

    if response is None:
        response = client.messages.create(**common)

    if getattr(response, "stop_reason", None) == "refusal":
        return ""

    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()


def _generate(question: str, sources: list[dict]) -> str:
    return _call_local(question, sources) if BACKEND == "local" else _call_anthropic(question, sources)


def answer(question: str) -> dict:
    """Answer a vetted question. Always returns something speakable."""
    debug.section(f"ASK: {question}")
    debug.log(f"backend={BACKEND} embed={embed_model_name()} top_k={TOP_K} gate={MIN_SIMILARITY}")

    with debug.timed("retrieve"):
        sources = retrieve(question)
    debug.sources(sources)

    if not available():
        reason = "no_model" if BACKEND == "local" else "no_api_key"
        debug.log(f"-> fallback ({reason})")
        return {"answer": FALLBACK_ANSWER, "grounded": False, "sources": [], "reason": reason}
    if not sources:
        debug.log(f"-> fallback (nothing above the {MIN_SIMILARITY} similarity gate)")
        return {"answer": FALLBACK_ANSWER, "grounded": False, "sources": [], "reason": "no_sources"}

    prompt = system_prompt(BACKEND)
    user_turn = build_user_turn(question, sources)
    debug.block("SYSTEM PROMPT", prompt)
    debug.block("USER TURN (exact model input)", user_turn)

    try:
        with debug.timed("generate"):
            raw = _generate(question, sources)
    except Exception as exc:  # noqa: BLE001 - never take the voice path down
        print(f"[answer] generation failed: {type(exc).__name__}: {exc}")
        debug.log(f"-> fallback (model error: {type(exc).__name__})")
        return {"answer": FALLBACK_ANSWER, "grounded": False, "sources": [], "reason": "model_error"}

    debug.block("RAW MODEL OUTPUT (before guards)", raw)

    checked = check_answer(raw)
    if not checked.ok:
        # The output guard matters more with a small local model than it would
        # with a frontier one — it is the last layer before the avatar speaks.
        print(f"[answer] rejected by output guard: {checked.flags}")
        debug.log(f"-> fallback (output guard rejected: {checked.flags})")
        return {"answer": FALLBACK_ANSWER, "grounded": False, "sources": [], "reason": "guard"}

    grounded = "no_answer" not in checked.flags
    debug.log(f"guard flags={checked.flags or '[]'} grounded={grounded} cacheable={grounded}")
    debug.block("FINAL ANSWER", checked.text)

    return {
        "answer": checked.text,
        "grounded": grounded,
        "sources": [{"title": s["title"], "url": s["url"]} for s in sources[:3]],
        "reason": "ok",
        "flags": checked.flags,
        "top_chunk_id": sources[0]["id"],
    }
