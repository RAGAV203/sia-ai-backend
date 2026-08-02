"""Semantic answer-cache correctness.

    python -m tests.test_answer_cache

The cache exists for latency, but a wrong hit is a correctness bug: it speaks a
confident answer to a question nobody asked. These tests pin the behaviour that
keeps that from happening — in particular that a *different question about the
same topic* does not hit, even though it retrieves the same chunk.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

from kb.answer_cache import CORROBORATED_THRESHOLD, SIMILARITY_THRESHOLD, AnswerCache

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))


def fresh() -> AnswerCache:
    return AnswerCache(path=pathlib.Path(tempfile.mkdtemp()) / "cache.jsonl")


# --- hit / miss behaviour -----------------------------------------------------

cache = fresh()
cache.put(
    "What programmes does the college offer?",
    {"answer": "We offer B.Com and B.Sc programmes.", "grounded": True},
    top_chunk_id="pages-1#0",
)
cache.put(
    "What is the fee for B.Com?",
    {"answer": "Fee details are on the admissions portal.", "grounded": True},
    top_chunk_id="pages-2#0",
)

CASES = [
    ("What programmes does the college offer?", "pages-1#0", True, "identical question"),
    ("What programmes does the college offer?", None, True, "identical needs no corroboration"),
    ("What academic programs do you offer?", "pages-1#0", True, "paraphrase corroborated by same chunk"),
    ("What academic programs do you offer?", "pages-9#9", False, "paraphrase with different chunk"),
    ("What academic programs do you offer?", None, False, "paraphrase without corroboration"),
    # The important one: same topic, same retrieved chunk, different answer wanted.
    ("How long is the B.Com course?", "pages-2#0", False, "different question about the same topic"),
    ("What subjects are in B.Com?", "pages-2#0", False, "another question about the same topic"),
    ("What is the hostel fee?", "pages-1#0", False, "unrelated question"),
]

for question, chunk_id, want_hit, label in CASES:
    hit = cache.get(question, top_chunk_id=chunk_id)
    check(
        (hit is not None) == want_hit,
        f"{'hits' if want_hit else 'misses'}: {label}",
        f"sim={hit['similarity']} match={hit['match']}" if hit else "",
    )

# A hit must return the stored answer verbatim — that is what keeps the TTS
# clips on disk reusable.
hit = cache.get("What programmes does the college offer?")
check(
    bool(hit) and hit["answer"] == "We offer B.Com and B.Sc programmes.",
    "hit returns the answer byte-for-byte",
)
check(bool(hit) and hit.get("cached") is True, "hit is flagged as cached")

# --- what must never be cached ------------------------------------------------

ungrounded = fresh()
ungrounded.put("Anything?", {"answer": "I do not have that detail.", "grounded": False}, top_chunk_id="x")
check(ungrounded.stats()["entries"] == 0, "ungrounded fallbacks are not cached")

empty = fresh()
check(empty.get("Any question at all") is None, "empty cache returns None")

# --- persistence --------------------------------------------------------------

path = pathlib.Path(tempfile.mkdtemp()) / "persist.jsonl"
first = AnswerCache(path=path)
first.put("What is the NAAC grade?", {"answer": "A++ since 2023.", "grounded": True}, top_chunk_id="n#0")
reloaded = AnswerCache(path=path)
hit = reloaded.get("What is the NAAC grade?")
check(bool(hit) and hit["answer"] == "A++ since 2023.", "cache survives a restart")

# --- thresholds are ordered sensibly -----------------------------------------

check(
    CORROBORATED_THRESHOLD < SIMILARITY_THRESHOLD,
    "corroborated bar is looser than the verbatim bar",
)
check(
    CORROBORATED_THRESHOLD > 0.58,
    "corroborated bar clears the measured different-question band (max 0.580)",
    f"threshold={CORROBORATED_THRESHOLD}",
)

# --- report -------------------------------------------------------------------

failed = [r for r in results if not r[0]]
for ok, name, detail in results:
    if not ok:
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))
print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f", {len(failed)} FAILED" if failed else ""))
sys.exit(1 if failed else 0)
