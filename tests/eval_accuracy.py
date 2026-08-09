"""Accuracy evaluation for the knowledge base.

    python -m tests.eval_accuracy              # retrieval only (fast, no model)
    python -m tests.eval_accuracy --answers    # + end-to-end answer quality

Retrieval and answering fail in different ways and are scored separately: if
retrieval never surfaces the right page, no amount of prompt work will fix the
answer, and tuning the model against a broken retriever wastes effort. The
retrieval pass needs no LLM, so it is cheap enough to run after every change to
chunking, embeddings, or fusion.

Ground truth is drawn from facts verified directly against the live site.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from kb.index import load

# (question, substrings that identify the right source page, fact the answer should carry)
# `expect_page` matches against chunk title or url; `expect_fact` is a list of
# alternatives, any one of which counts as correct.
CASES: list[tuple[str, list[str], list[str]]] = [
    ("What NAAC grade does the college have?", ["naac"], ["a++", "a ++"]),
    ("Is the college accredited?", ["naac", "about"], ["a++", "naac", "accredit"]),
    ("Where is the college located?", ["contact", "about"], ["madley", "t.nagar", "chennai"]),
    ("What is the college email address?", ["contact"], ["shasuncollege.edu.in"]),
    ("How big is the campus?", ["infrastructure", "about"], ["2.1", "acre"]),
    ("What undergraduate programmes are offered?", ["academic", "programme", "school", "commerce"],
     ["b.com", "b.sc", "bachelor", "commerce"]),
    ("Tell me about the placement cell", ["placement"], ["placement", "recruit", "training"]),
    ("Do you have a computer science course?", ["computer science"], ["computer science"]),
    ("What is the visual communication course?", ["visual communication"], ["visual communication"]),
    ("Is there a journalism programme?", ["journalism"], ["journalism"]),
    ("Who can apply for admission?", ["admission", "eligib"], ["women", "girl", "female", "12"]),
    ("What clubs and activities are there?", ["club", "extension", "nss", "activit"],
     ["club", "nss", "activit"]),
    ("Tell me about the library", ["library"], ["library", "book"]),
    ("What is the college anthem?", ["anthem"], ["anthem", "shasun"]),
    ("Does the college have an IQAC?", ["iqac", "quality"], ["iqac", "quality"]),
]

# Questions the corpus genuinely cannot answer — the retriever must return
# nothing rather than the nearest college page, and the model must decline.
NEGATIVES = [
    "Who won the cricket world cup?",
    "How do I bake sourdough bread?",
    "What is the capital of France?",
    "Write me a poem about the sea",
    "What is the share price of Apple?",
]


# SIA is instructed to write what she says out loud — "10 plus 2" rather than
# "10+2" — because the text goes straight to a speech synthesizer that would
# otherwise read punctuation as punctuation.
#
# That collided with this eval, which scored answers by substring. Two correct
# answers were counted wrong for obeying the prompt:
#
#   "A plus plus grade from the National Assessment..."   vs expected "a++"
#   "info at shasuncollege dot edu dot in"                vs expected "shasuncollege.edu.in"
#
# Rather than weaken the expectations to spoken spellings — which would stop
# catching a genuinely missing grade — both sides are folded back to the written
# form before comparing. The check then tests the fact, not the transcription.
_SPOKEN = (
    (" plus plus", "++"),
    (" plus ", "+"),
    (" dot ", "."),
    (" at ", "@"),
    (" underscore ", "_"),
    (" dash ", "-"),
    # Numbers are spoken as words for the same reason, so "class twelve" has to
    # satisfy an expectation written as "12".
    (" twelve", " 12"),
    (" ten", " 10"),
    (" eleven", " 11"),
    (" two point one", " 2.1"),
)


def _fold_spoken(text: str) -> str:
    low = f" {text.lower()} "
    for spoken, written in _SPOKEN:
        low = low.replace(spoken, written)
    return low


def matches(text: str, needles: list[str]) -> bool:
    low = text.lower()
    folded = _fold_spoken(text)
    return any(n.lower() in low or n.lower() in folded for n in needles)


def corpus_ceiling() -> tuple[int, list[str]]:
    """How many eval questions the corpus could answer even with perfect retrieval.

    Retrieval scores are meaningless without this. A question whose source page
    was never scraped is unanswerable no matter how good the embeddings are, and
    tuning the retriever against it wastes effort — the fix belongs in ingest.
    """
    import json

    from kb.index import CORPUS

    if not CORPUS.exists():
        return len(CASES), []
    docs = [json.loads(ln) for ln in CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    blob = " ".join((d["title"] + " " + d["url"] + " " + d["text"]).lower() for d in docs)

    answerable, missing = 0, []
    for question, expect_page, expect_fact in CASES:
        if matches(blob, expect_page) and matches(blob, expect_fact):
            answerable += 1
        else:
            missing.append(question)
    return answerable, missing


def evaluate_retrieval(index, k: int | None = None, verbose: bool = True) -> dict:
    """Score retrieval using the settings the service actually answers with.

    This used to default to k=6 while the local backend ran k=3, so the number
    printed here described a configuration nobody was serving — and it read 80%
    while production was managing 73%. Defaults now come from `answering`, and
    `-k` overrides it for sweeps.
    """
    from kb.answering import MIN_SIMILARITY, PER_DOC, TOP_K

    k = TOP_K if k is None else k
    hits = misses = 0
    latencies = []
    for question, expect_page, _ in CASES:
        t = time.perf_counter()
        results = index.search(question, k=k, per_doc=PER_DOC)
        latencies.append((time.perf_counter() - t) * 1000)
        found = any(
            matches(f"{r.chunk['title']} {r.chunk['url']}", expect_page) for r in results
        )
        best = max((r.similarity for r in results), default=0.0)
        gated = best < MIN_SIMILARITY
        if found and not gated:
            hits += 1
        else:
            misses += 1
            if verbose:
                top = results[0].chunk["title"][:40] if results else "-"
                why = "below gate" if gated else f"got {top!r}"
                print(f"    MISS  {question[:52]:52s} sim={best:.3f}  {why}")

    false_positives = 0
    for question in NEGATIVES:
        results = index.search(question, k=k, per_doc=PER_DOC)
        best = max((r.similarity for r in results), default=0.0)
        if best >= MIN_SIMILARITY:
            false_positives += 1
            if verbose:
                print(f"    LEAK  {question[:52]:52s} sim={best:.3f} -> {results[0].chunk['title'][:34]!r}")

    latencies.sort()
    return {
        "recall": hits / len(CASES),
        "hits": hits,
        "total": len(CASES),
        "false_positives": false_positives,
        "negatives": len(NEGATIVES),
        "median_ms": latencies[len(latencies) // 2] if latencies else 0,
        "k": k,
        "per_doc": PER_DOC,
    }


def evaluate_answers(verbose: bool = True) -> dict:
    from kb import answering

    if not answering.available():
        print("  (answering backend unavailable — skipping)")
        return {}

    correct = wrong = 0
    latencies = []
    for question, _, expect_fact in CASES:
        t = time.perf_counter()
        result = answering.answer(question)
        latencies.append(time.perf_counter() - t)
        text = result["answer"]
        if result["reason"] == "ok" and matches(text, expect_fact):
            correct += 1
        else:
            wrong += 1
            if verbose:
                print(f"    WRONG {question[:46]:46s} [{result['reason']}] {' '.join(text.split())[:70]}")

    # This is the metric that actually matters for off-topic questions. The
    # retrieval gate cannot separate them on this corpus (see answering.py), so
    # correctness depends on the model declining once it sees the chunks —
    # measured here end to end rather than assumed.
    declined = 0
    for question in NEGATIVES:
        result = answering.answer(question)
        if result["reason"] != "ok" or not result.get("grounded"):
            declined += 1
        elif verbose:
            print(f"    ANSWERED OFF-TOPIC: {question[:44]!r} -> {' '.join(result['answer'].split())[:60]}")

    latencies.sort()
    return {
        "accuracy": correct / len(CASES),
        "correct": correct,
        "total": len(CASES),
        "declined": declined,
        "negatives": len(NEGATIVES),
        "median_s": latencies[len(latencies) // 2] if latencies else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also evaluate generated answers (slow)")
    ap.add_argument("-k", type=int, default=None, help="chunks to retrieve (default: the service's RETRIEVAL_TOP_K)")
    args = ap.parse_args()

    index = load()
    if index is None:
        print("no index — run `python -m kb.ingest && python -m kb.index`")
        return 1

    from kb.embed import model_name

    print(f"index: {len(index.chunks)} chunks  [embed: {model_name()}]\n")

    ceiling, unanswerable = corpus_ceiling()
    print("CORPUS COVERAGE")
    print(f"  answerable        {ceiling}/{len(CASES)}  ({100*ceiling/len(CASES):.0f}%) — the retrieval ceiling")
    for question in unanswerable:
        print(f"    NOT IN CORPUS  {question[:60]}")

    print("\nRETRIEVAL")
    r = evaluate_retrieval(index, k=args.k)
    k = r["k"]
    lost = ceiling - r["hits"]
    print(f"  recall@{k}        {r['hits']}/{r['total']}  ({100*r['recall']:.0f}%)"
          f"  — {100*r['hits']/max(ceiling,1):.0f}% of what the corpus allows")
    if lost > 0:
        print(f"  lost to retrieval {lost}  (present in corpus but not surfaced)")
    # Not a correctness metric: retrieval scores provably cannot separate
    # on-topic from off-topic here, so these only cost a wasted generation. The
    # model declining is what makes them harmless, and that is scored below.
    print(f"  off-topic past gate {r['false_positives']}/{r['negatives']}  (cost only — model declines these)")
    print(f"  median latency    {r['median_ms']:.1f} ms")

    # Judge retrieval against what the corpus can actually support, not an
    # absolute number it has no way to reach.
    ok = r["hits"] >= 0.8 * ceiling

    if args.answers:
        print("\nANSWERS")
        a = evaluate_answers()
        if a:
            print(f"  accuracy          {a['correct']}/{a['total']}  ({100*a['accuracy']:.0f}%)")
            print(f"  declined off-topic {a['declined']}/{a['negatives']}  (want all)")
            print(f"  median latency    {a['median_s']:.1f} s")
            ok = ok and a["accuracy"] >= 0.6 and a["declined"] == a["negatives"]

    print("\n" + ("PASS" if ok else "BELOW TARGET"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
