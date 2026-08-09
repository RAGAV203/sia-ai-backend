"""Semantic answer cache — the piece that keeps dynamic answers fast to *speak*.

Generation latency is not the bottleneck in this service; **synthesis** is.
Kokoro runs at roughly 0.4x real time, so a fresh 40-second answer costs ~16s of
CPU before the avatar makes a sound. The existing TTS disk cache is keyed on the
exact answer text, which is why the fixed knowledge base could be prewarmed and
served in ~25ms.

Free-form generated answers would defeat that: two visitors asking the same
thing in slightly different words get two different strings, so every reply is a
TTS cache miss. This cache restores the property by matching on *meaning* and
returning the previous answer **verbatim** — identical bytes, therefore the same
TTS cache key, therefore instant audio.

So a hit here is worth far more than the model call it saves.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .embed import embed_symmetric

CACHE_PATH = Path(__file__).resolve().parent.parent / "kb-data" / "answer-cache.jsonl"

# Two ways to call a cached question "the same question".
#
# The dangerous case is not an unrelated question — it is a *different question
# about the same topic*, which retrieves the same chunk and is worded similarly.
# "What is the fee for B.Com?" and "How long is the B.Com course?" want
# completely different answers. Measured on real pairs:
#
#   same intent, should hit          0.627 - 0.906
#     0.906  "is there a placement cell" / "do you have a placement cell"
#     0.718  "what programmes does the college offer" / "what academic programs do you offer"
#     0.627  "what programmes does the college offer" / "what courses can I study there"
#
#   different question, same topic   0.361 - 0.580   <- must NOT hit
#     0.580  "what is the NAAC grade" / "when was the NAAC visit"
#     0.567  "what is the fee for B.Com" / "how long is the B.Com course"
#     0.361  "who is the principal" / "how many staff does the college have"
#
# The bands separate, but only by ~0.05, so the threshold sits at 0.65 — above
# every measured different-question pair with real headroom, at the cost of
# missing the weakest paraphrase. A miss costs one generation; a false hit
# speaks a confident answer to a question nobody asked.
#
# Corroboration (same top-retrieved chunk) is required as well, but note it does
# not guard this case — same-topic questions retrieve the same chunk by
# definition. It guards the opposite failure: two unrelated questions that
# happen to land close in embedding space.
SIMILARITY_THRESHOLD = 0.93   # near-identical: serve on similarity alone
CORROBORATED_THRESHOLD = 0.65  # reworded: also requires the same top chunk
MAX_ENTRIES = 2000


class AnswerCache:
    def __init__(self, path: Path = CACHE_PATH, threshold: float = SIMILARITY_THRESHOLD):
        self.path = path
        self.threshold = threshold
        self._lock = threading.Lock()
        self._questions: list[str] = []
        self._answers: list[dict] = []
        self._vectors: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return
        self._questions = [r["question"] for r in rows]
        self._answers = [r["payload"] for r in rows]
        self._vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
        print(f"[cache] {len(rows)} cached answers loaded")

    def get(self, question: str, top_chunk_id=None) -> dict | None:
        """Return a cached answer for an equivalent question, or None.

        ``top_chunk_id`` is the id of the best chunk this question retrieves.
        When supplied it corroborates a merely-similar question, letting the
        cache serve paraphrases without loosening the similarity bar itself.

        **Pass a callable, not a value.** Computing it costs a query embedding —
        a metered API call, capped at 1000/day on the free tier — and it is only
        consulted for scores in the corroboration band. Passing it eagerly meant
        every request paid for it, including the verbatim cache hits whose whole
        purpose is to cost nothing. A callable is invoked only if it is actually
        needed. A plain value still works, for callers that already have one.
        """
        with self._lock:
            if not len(self._vectors):
                return None
            # embed_symmetric is the pinned *local* encoder — free and offline,
            # which is what lets a cache lookup happen before any API call.
            sims = self._vectors @ embed_symmetric(question)
            best = int(np.argmax(sims))
            score = float(sims[best])
            payload = dict(self._answers[best])
            matched = self._questions[best]

        if score >= self.threshold:
            match_kind = "verbatim"
        elif score >= CORROBORATED_THRESHOLD:
            resolved = top_chunk_id() if callable(top_chunk_id) else top_chunk_id
            if not resolved or payload.get("top_chunk_id") != resolved:
                return None
            match_kind = "corroborated"
        else:
            return None

        payload["cached"] = True
        payload["match"] = match_kind
        payload["matched_question"] = matched
        payload["similarity"] = round(score, 3)
        return payload

    def put(self, question: str, payload: dict, top_chunk_id: str | None = None) -> None:
        # Only cache genuinely grounded answers — caching a fallback would pin
        # "I don't know" to a question that might work on the next attempt.
        if not payload.get("grounded"):
            return
        payload = {**payload, "top_chunk_id": top_chunk_id} if top_chunk_id else dict(payload)
        vector = embed_symmetric(question)
        entry = {
            "question": question,
            "payload": {k: v for k, v in payload.items() if k != "cached"},
            "vector": vector.tolist(),
            "at": time.time(),
        }
        with self._lock:
            self._questions.append(question)
            self._answers.append(entry["payload"])
            self._vectors = (
                np.vstack([self._vectors, vector[None, :]]) if len(self._vectors) else vector[None, :]
            )
            if len(self._questions) > MAX_ENTRIES:
                drop = len(self._questions) - MAX_ENTRIES
                self._questions = self._questions[drop:]
                self._answers = self._answers[drop:]
                self._vectors = self._vectors[drop:]
                self._rewrite()
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError as exc:
                print(f"[cache] persist failed: {exc}")

    def _rewrite(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for q, a, v in zip(self._questions, self._answers, self._vectors):
                    fh.write(
                        json.dumps(
                            {"question": q, "payload": a, "vector": v.tolist(), "at": time.time()},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            tmp.replace(self.path)
        except OSError as exc:
            print(f"[cache] rewrite failed: {exc}")

    def stats(self) -> dict:
        return {"entries": len(self._questions), "threshold": self.threshold}

    def vet(self, apply: bool = False) -> list[tuple[str, str]]:
        """Re-check every cached answer against the *current* guards.

        The cache is append-only and long-lived, so it accumulates answers that
        were acceptable under an older guard and are not any more. That is not a
        cosmetic problem: a refusal admitted by mistake is served verbatim from
        then on, so a question the corpus can answer is permanently answered
        with "I can't assist with that" — and because the entry is marked
        grounded, nothing ever revisits it.

        Run after changing `guard.py` or `intent.py`:

            python -m kb.answer_cache --vet        # report
            python -m kb.answer_cache --vet --apply

        Returns the (question, reason) pairs that would be dropped.
        """
        from .guard import check_answer
        from .intent import match as intent_match

        drops: list[tuple[str, str]] = []
        keep_q, keep_a, keep_v = [], [], []
        with self._lock:
            for q, payload, vec in zip(self._questions, self._answers, self._vectors):
                reason = ""
                if intent_match(q):
                    reason = "social turn — now answered by the intent gate"
                else:
                    checked = check_answer(payload.get("answer", ""))
                    if not checked.ok:
                        reason = f"rejected by guard: {','.join(checked.flags)}"
                    elif "no_answer" in checked.flags:
                        reason = f"refusal cached as grounded: {','.join(checked.flags)}"
                if reason:
                    drops.append((q, reason))
                else:
                    keep_q.append(q)
                    keep_a.append(payload)
                    keep_v.append(vec)

            if apply and drops:
                self._questions = keep_q
                self._answers = keep_a
                self._vectors = (
                    np.array(keep_v, dtype=np.float32)
                    if keep_v
                    else np.zeros((0, self._vectors.shape[1]), dtype=np.float32)
                )
                self._rewrite()
        return drops


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="answer cache maintenance")
    ap.add_argument("--vet", action="store_true", help="re-check entries against current guards")
    ap.add_argument("--apply", action="store_true", help="actually remove the failing entries")
    args = ap.parse_args()

    cache = AnswerCache()
    if not args.vet:
        raise SystemExit(f"{cache.stats()}  (pass --vet to re-check entries)")

    failing = cache.vet(apply=args.apply)
    for question, reason in failing:
        print(f"  DROP  {question[:52]:52s} {reason}")
    verb = "removed" if args.apply else "would remove"
    print(f"{verb} {len(failing)} of {len(failing) + len(cache._questions)} entries")
    if failing and not args.apply:
        print("re-run with --apply to rewrite the cache")
