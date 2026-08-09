"""Scope and sufficiency: does SIA answer the right *kind* of question, and does
she admit when the corpus does not actually answer one?

    python -m tests.test_scope            # local layers only, no API calls
    python -m tests.test_scope --live     # + the full pipeline through Gemini

The regression this suite exists for: asked to "write a python program", SIA
answered with the college's academic programmes. Retrieval scored the request
0.597 against pages containing the word "Program", and two separate paths then
turned that into a confident answer — the similarity gate, and the curated
keyword table matching the bare substring "program".

Both paths are covered here, because fixing one and not the other leaves the
system able to make the same mistake by a different route.
"""

from __future__ import annotations

import argparse
import sys

from kb.triage import is_underspecified, out_of_scope
from knowledge import FALLBACK_ANSWER, OFF_TOPIC_ANSWER, SUGGESTIONS, resolve_answer
from tests.eval_accuracy import CASES

# Requests that are not about the college, however much vocabulary they share
# with it. The "program"/"programme" cases are the original bug.
OUT_OF_SCOPE = [
    "write a python program",
    "write me a python program to sort a list",
    "can you write code in java",
    "give me some python code",
    "generate a website for me",
    "write an essay about climate change",
    "write me a poem about the sea",
    "give me a recipe for pasta",
    "what is the capital of France",
    "who won the cricket world cup",
    "translate this to Hindi",
    "solve this equation for x",
    "tell me a joke",
    "do my homework",
    "what is 5 * 7",
]

# Genuine college questions, several deliberately sharing vocabulary with the
# out-of-scope list. None of these may ever be refused.
IN_SCOPE = [q for q, _, _ in CASES] + [s["question"] for s in SUGGESTIONS] + [
    "Is there a certificate program in Python?",
    "Do you teach programming in the computer science course?",
    "What is the Digital Marketing Proficiency Program?",
    "What is the student induction program?",
    "Who wrote the college anthem?",
    "Do you have a coding club?",
    "Is Java part of the syllabus?",
    "Can you tell me about the B.Sc Computer Science program?",
]

# In scope, but the corpus has no answer. These must produce an "I don't have
# that detail" reply — never an invented one, and never an out-of-scope refusal,
# which would tell a prospective student their fair question was unwelcome.
IN_SCOPE_UNKNOWN = [
    "What is the exact tuition fee for B.Com in rupees?",
    "How many students were placed in 2019 exactly?",
    "What is the WiFi password?",
]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        failures.append(f"{name}: {detail}")


def test_triage_precision() -> None:
    """The cheap layer must never refuse a real question."""
    for q in IN_SCOPE:
        rule = out_of_scope(q)
        check("triage false positive", rule is None, f"{q!r} refused by rule {rule!r}")


def test_triage_recall() -> None:
    for q in OUT_OF_SCOPE:
        check("triage miss", out_of_scope(q) is not None, f"{q!r} not caught locally")


def test_curated_fallback_respects_scope() -> None:
    """The keyword table is the path that actually produced the reported bug."""
    for q in OUT_OF_SCOPE:
        answer = resolve_answer(q)
        check(
            "curated leak",
            answer == OFF_TOPIC_ANSWER,
            f"{q!r} -> {answer[:70]!r}",
        )


def test_curated_fallback_still_answers() -> None:
    for q in ("What academic programs do you offer?", "how do I apply", "tell me about placements"):
        answer = resolve_answer(q)
        check(
            "curated regression",
            answer not in (FALLBACK_ANSWER, OFF_TOPIC_ANSWER),
            f"{q!r} no longer matches a curated answer",
        )


def test_underspecified() -> None:
    for q in ("tell me more", "what about fees", "and the timings?"):
        check("underspecified", is_underspecified(q), f"{q!r} not flagged")
    for q in ("What are the admission requirements and how can I apply?",):
        check("underspecified", not is_underspecified(q), f"{q!r} wrongly flagged")


def test_live_pipeline() -> None:
    """The whole path through Gemini, including the model's own scope verdict."""
    from kb import answering

    if not answering.available():
        print("  (no API key — skipping live pipeline)")
        return

    for q in OUT_OF_SCOPE[:6]:
        result = answering.answer(q)
        check(
            "live scope",
            result.get("scope") == "out_of_scope",
            f"{q!r} -> scope={result.get('scope')} reason={result.get('reason')} "
            f"answer={result['answer'][:60]!r}",
        )
        check("live scope grounded", not result.get("grounded"), f"{q!r} marked grounded")

    for q in IN_SCOPE_UNKNOWN:
        result = answering.answer(q)
        check(
            "live unknown",
            not result.get("grounded"),
            f"{q!r} claimed grounded: {result['answer'][:70]!r}",
        )
        check(
            "live unknown scope",
            result.get("scope") != "out_of_scope",
            f"{q!r} wrongly refused as out of scope",
        )

    for q in ("What NAAC grade does the college have?", "Tell me about the library"):
        result = answering.answer(q)
        check(
            "live grounded",
            result.get("grounded") and result.get("scope") == "college",
            f"{q!r} -> grounded={result.get('grounded')} reason={result.get('reason')}",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also exercise the Gemini pipeline")
    args = ap.parse_args()

    tests = [
        test_triage_precision,
        test_triage_recall,
        test_curated_fallback_respects_scope,
        test_curated_fallback_still_answers,
        test_underspecified,
    ]
    if args.live:
        tests.append(test_live_pipeline)

    for test in tests:
        before = len(failures)
        test()
        status = "ok" if len(failures) == before else f"{len(failures) - before} FAILED"
        print(f"  {test.__name__:40s} {status}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nall scope checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
