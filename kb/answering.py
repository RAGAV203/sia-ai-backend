"""Grounded answering: understand the question, retrieve, verify, then answer.

The earlier version of this module did retrieve-then-generate, and gated on a
similarity threshold. That shape has a specific failure, and it is worth stating
because the whole design here follows from it:

    "write a python program"
        -> retrieved "Digital Marketing Proficiency Program" at 0.597
        -> answered about the college's academic programmes

Nothing in that path ever asked the two questions a person would ask. *Is this
even something I do?* and *does what I found actually answer it?* A similarity
score cannot express either. It measures topical overlap, and "write a python
program" really does overlap with a page containing the word "Program" — so the
threshold was being asked to separate two things it cannot see the difference
between, and every choice of value was wrong somewhere.

So the pipeline now has four stages instead of two:

    1. TRIAGE    cheap local check for obviously out-of-scope requests
                 (kb/triage.py) — costs nothing, catches the blatant cases
    2. RETRIEVE  hybrid dense + BM25 + RRF, with one adaptive rewrite when the
                 question is too thin to match on
    3. VERIFY    the model returns `scope` and `sufficiency` as *fields*, judging
                 the question against the sources it was given
    4. ROUTE     scope and sufficiency choose which of three very different
                 replies is correct

Verification costs nothing extra: it is the same call that writes the answer,
constrained to a schema, measured at the same ~870 ms as the free-text version.
What it buys is that the judgement is explicit and checkable, rather than
something the caller has to infer from the tone of a sentence.
"""

from __future__ import annotations

import os

from . import debug, gemini, rerank, triage
from .embed import model_name as embed_model_name
from .guard import check_answer
from .index import load as load_index
from .prompts import (
    ANSWER_SCHEMA,
    FALLBACK_ANSWER,
    OFF_CONTRACT_ANSWER,
    REWRITE_PROMPT,
    build_user_turn,
    system_prompt,
)

BACKEND = os.getenv("LLM_BACKEND", "gemini").strip().lower()

# Two thresholds, because "confident match" and "worth looking at" are different
# questions and were previously answered by one number.
#
# Measured on the eval set with gemini-embedding-001 at 768 dims:
#
#   on-topic   0.638 - 0.741
#   off-topic  0.499 - 0.567
#
# MIN_SIMILARITY (0.60) sits in the clean gap and still means what it did: a
# confident topical match. It is what corroborates an answer-cache hit.
#
# FLOOR (0.45) is new, and is the point below which nothing in the corpus is
# even loosely related. Between the two, retrieval is uncertain — and that band
# is exactly where the interesting failures live, "write a python program" among
# them at 0.597. Those used to be refused by threshold with the *wrong* message.
# They are now sent to the model, which reads the request rather than the word
# overlap and routes them correctly.
#
# The cost of admitting that band is one generation for some off-topic
# questions. Triage absorbs the obvious ones for free, and being right is worth
# a twentieth of a cent.
MIN_SIMILARITY = float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.60"))
FLOOR = float(os.getenv("RETRIEVAL_FLOOR", "0.45"))

# How far below the best hit a chunk may be and still earn a slot in the prompt.
# See _trim_weak: `k` is a ceiling, not a quota.
DROP_MARGIN = float(os.getenv("RETRIEVAL_DROP_MARGIN", "0.07"))

# Retrieval breadth. The local backend was pinned to k=5 / 700 chars purely by
# prefill cost; on the API path a chunk costs about 0.02 cents and no latency
# worth measuring, so both go up.
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
MAX_CHUNK_CHARS = int(os.getenv("RETRIEVAL_CHUNK_CHARS", "1000"))

# How many chunks one page may occupy. Pages here average 15k characters, so a
# page that matches at all usually matches several times and takes every slot.
# 2 keeps enough of the winning page to answer from while freeing slots for
# corroborating pages.
PER_DOC = int(os.getenv("RETRIEVAL_PER_DOC", "2"))

# One rewrite attempt when the question is too thin to retrieve on. Off by
# setting REWRITE_WEAK_QUERIES=0.
REWRITE_ENABLED = os.getenv("REWRITE_WEAK_QUERIES", "1") != "0"

MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "500"))


def backend_name() -> str:
    return BACKEND


def available() -> bool:
    return gemini.available()


def model_name() -> str:
    return gemini.TEXT_MODEL


# --- retrieval ----------------------------------------------------------------

def _search(question: str) -> list[dict]:
    index = load_index()
    if index is None:
        return []
    # With reranking on, retrieve a wider pool so the reranker has something to
    # promote from; without it, k is the pool.
    fetch = rerank.POOL if rerank.available() else TOP_K
    hits = index.search(question, k=fetch, per_doc=PER_DOC)
    sources = [
        {
            "id": h.chunk["id"],
            "title": h.chunk["title"],
            "section": h.chunk.get("section", ""),
            "text": h.chunk["text"][:MAX_CHUNK_CHARS],
            "url": h.chunk["url"],
            "score": h.score,
            "similarity": h.similarity,
        }
        for h in hits
    ]
    if rerank.available():
        # Relevance judgement replaces the similarity margin: once a model has
        # read each chunk against the question, a score of 0 is a better reason
        # to drop it than being 0.07 below the leader.
        return rerank.rerank(question, sources, TOP_K)
    return _trim_weak(sources)


def _trim_weak(sources: list[dict]) -> list[dict]:
    """Drop trailing chunks that are much worse than the best one.

    ``k`` is a ceiling, not a quota, and treating it as a quota is how unrelated
    material ends up in the prompt. "Who can apply for admission?" scored 0.638
    at the top and still received an "Outcome Based Education" chunk at 0.564,
    purely because eight slots existed and something had to fill the eighth.

    Similarity is only meaningful *relative to the query* — 0.60 is a strong
    match for one question and a weak one for another — so the cut is relative to
    the best hit rather than a second absolute threshold. Everything within
    ``DROP_MARGIN`` of the leader stays; the tail that is merely the least-bad
    remaining option goes.

    This matters more than it looks. Every weak chunk is another passage the
    model is invited to stretch into an answer, and the sufficiency judgement
    downstream is only as good as the evidence it is asked to judge.
    """
    if not sources:
        return sources
    best = sources[0]["similarity"]
    for source in sources:
        best = max(best, source["similarity"])
    cutoff = max(FLOOR, best - DROP_MARGIN)
    kept = [s for s in sources if s["similarity"] >= cutoff]
    # Always keep the leader even if the margin maths says otherwise.
    return kept or sources[:1]


def _best(sources: list[dict]) -> float:
    return max((s["similarity"] for s in sources), default=0.0)


def _rewrite(question: str) -> str | None:
    """Turn a thin spoken question into something the index can match."""
    try:
        rewritten = gemini.generate(REWRITE_PROMPT, question, max_tokens=40)
    except Exception as exc:  # noqa: BLE001 - a failed rewrite is not fatal
        debug.log(f"rewrite failed: {type(exc).__name__}")
        return None
    rewritten = " ".join(rewritten.split()).strip('"').strip()
    # Guard against the model answering instead of rewriting.
    if not rewritten or len(rewritten) > 120:
        return None
    return rewritten


def retrieve(question: str) -> tuple[list[dict], str]:
    """Retrieve supporting chunks. Returns (sources, the query actually used).

    A single rewrite is attempted when the first pass comes back weak *and* the
    question has too few content words to match on. Both conditions matter: a
    well-specified question that retrieves badly is usually a genuine gap in the
    corpus, and rewriting it just spends a call to fail again.
    """
    sources = _search(question)
    if not REWRITE_ENABLED or _best(sources) >= MIN_SIMILARITY:
        return sources, question
    if not triage.is_underspecified(question):
        return sources, question

    rewritten = _rewrite(question)
    if not rewritten:
        return sources, question

    retried = _search(rewritten)
    debug.log(f"rewrite: {question!r} -> {rewritten!r} ({_best(sources):.3f} -> {_best(retried):.3f})")
    if _best(retried) > _best(sources):
        return retried, rewritten
    return sources, question


def top_chunk_id(question: str) -> str | None:
    """Id of the best-matching chunk, used to corroborate a semantic cache hit."""
    index = load_index()
    if index is None:
        return None
    hits = index.search(question, k=1)
    if not hits or hits[0].similarity < MIN_SIMILARITY:
        return None
    return hits[0].chunk["id"]


# --- replies ------------------------------------------------------------------

def _off_contract(reason: str) -> dict:
    return {
        "answer": OFF_CONTRACT_ANSWER,
        "grounded": False,
        "sources": [],
        "reason": reason,
        "scope": "out_of_scope",
    }


def _no_detail(reason: str, sources: list[dict] | None = None) -> dict:
    return {
        "answer": FALLBACK_ANSWER,
        "grounded": False,
        "sources": [],
        "reason": reason,
        "scope": "college",
        "sufficiency": "none",
    }


# --- the pipeline -------------------------------------------------------------

def answer(question: str) -> dict:
    """Answer a vetted question. Always returns something speakable.

    The returned ``grounded`` flag means "this answer states facts the corpus
    supports", and it is what decides whether the answer may be cached. A
    refusal, an out-of-scope reply, and an "I don't have that detail" are all
    correct responses and none of them are grounded — caching any of them would
    pin a non-answer to a question that might work once the corpus improves.
    """
    debug.section(f"ASK: {question}")
    debug.log(
        f"backend={BACKEND} model={model_name()} embed={embed_model_name()} "
        f"top_k={TOP_K} gate={MIN_SIMILARITY} floor={FLOOR}"
    )

    # 1. TRIAGE — free, and skips retrieval plus generation entirely.
    rule = triage.out_of_scope(question)
    if rule:
        debug.log(f"triage: out of scope ({rule}) -> no retrieval, no generation")
        return _off_contract(f"out_of_scope:{rule}")

    if not available():
        debug.log("-> fallback (no_api_key)")
        return _no_detail("no_api_key")

    # 2. RETRIEVE
    with debug.timed("retrieve"):
        sources, used_query = retrieve(question)
    debug.sources(sources)
    best = _best(sources)

    if best < FLOOR:
        # Nothing in the corpus is even loosely related. Generating here would
        # only ask the model to decline from material about something else.
        debug.log(f"-> nothing above the {FLOOR} floor (best {best:.3f})")
        return _off_contract("no_sources")

    # 3. VERIFY + ANSWER (one call)
    prompt = system_prompt(BACKEND)
    user_turn = build_user_turn(question, sources)
    debug.block("SYSTEM PROMPT", prompt)
    debug.block("USER TURN (exact model input)", user_turn)

    try:
        with debug.timed("generate"):
            verdict = gemini.generate_json(prompt, user_turn, ANSWER_SCHEMA, max_tokens=MAX_TOKENS)
    except ValueError as exc:
        # Unparseable or empty completion — a safety filter, or a truncated
        # object. There is no answer to speak either way.
        print(f"[answer] malformed verdict: {exc}")
        debug.log(f"-> fallback (malformed verdict: {exc})")
        return _no_detail("model_error")
    except Exception as exc:  # noqa: BLE001 - never take the voice path down
        print(f"[answer] generation failed: {type(exc).__name__}: {exc}")
        debug.log(f"-> fallback (model error: {type(exc).__name__})")
        return _no_detail("model_error")

    scope = str(verdict.get("scope", "")).strip().lower()
    sufficiency = str(verdict.get("sufficiency", "")).strip().lower()
    raw = str(verdict.get("answer", "")).strip()
    debug.log(f"verdict: scope={scope} sufficiency={sufficiency} best_sim={best:.3f}")
    debug.block("RAW MODEL OUTPUT (before guards)", raw)

    # 4. ROUTE
    if scope == "out_of_scope":
        # The model's own wording is discarded in favour of the canonical line.
        # It is prewarmed in the TTS cache, it keeps the refusal consistent
        # however the model phrased it, and a refusal is the one reply where
        # improvisation has no upside.
        debug.log("-> off-contract (model judged out of scope)")
        return _off_contract("out_of_scope:model")

    if not raw:
        debug.log("-> fallback (empty answer field)")
        return _no_detail("model_error")

    checked = check_answer(raw)
    if not checked.ok:
        # The output guard is the last layer before the avatar speaks.
        print(f"[answer] rejected by output guard: {checked.flags}")
        debug.log(f"-> fallback (output guard rejected: {checked.flags})")
        return _no_detail("guard")

    # `sufficiency` and the guard's refusal detection are independent readings of
    # the same thing, so an answer is grounded only when both agree. The guard
    # catches a model that says "full" and then hedges in prose; `sufficiency`
    # catches a confident-sounding sentence the sources never supported.
    refused = "no_answer" in checked.flags
    grounded = sufficiency in ("full", "partial") and not refused
    debug.log(
        f"guard flags={checked.flags or '[]'} sufficiency={sufficiency} "
        f"grounded={grounded} cacheable={grounded}"
    )
    debug.block("FINAL ANSWER", checked.text)

    return {
        "answer": checked.text,
        "grounded": grounded,
        "sources": [{"title": s["title"], "url": s["url"]} for s in sources[:3]],
        "reason": "ok" if grounded else f"insufficient:{sufficiency or 'unknown'}",
        "flags": checked.flags,
        "scope": "college",
        "sufficiency": sufficiency,
        "top_chunk_id": sources[0]["id"],
        "query": used_query,
    }
