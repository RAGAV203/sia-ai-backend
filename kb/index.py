"""Chunk the corpus and build a hybrid retrieval index.

**No vector database.** The corpus is ~490 chunks; as float32 that is ~750 KB of
vectors, so exact search is a single ``(N, 384) @ (384,)`` matmul — a fraction of
a millisecond, and *exact*. Approximate indexes (HNSW/IVF) only start paying for
themselves in the 100K+ vector range; here one would be slower to build and
merely approximate. A separate vector service would add a process, a network
hop, and a backup story in exchange for nothing. Revisit above ~50K chunks, or
if the index ever needs to be shared across replicas.

Retrieval is **hybrid**: dense embeddings for paraphrase ("where do students
live" → hostel) plus BM25 for the exact proper nouns a college site is full of
("B.Com Corporate Secretaryship", "SAYE", "NAAC"), which dense vectors blur
together. The two rankings are merged with Reciprocal Rank Fusion, which needs
no score normalization and no tuning.

    python -m kb.index          # build from corpus.jsonl
    python -m kb.index --probe "how do I apply"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import embed_passages, embed_query, model_name

DATA_DIR = Path(__file__).resolve().parent.parent / "kb-data"
CORPUS = DATA_DIR / "corpus.jsonl"
VECTORS = DATA_DIR / "vectors.npy"
CHUNKS = DATA_DIR / "chunks.jsonl"

# Chunk size, swept against tests/eval_accuracy.py rather than guessed:
#
#   400 chars -> 1661 chunks, recall 80%
#   600       -> 1068,        recall 80%
#   800       ->  810,        recall 80%   <- chosen
#   1000      ->  655,        recall 73%
#   1400      ->  508,        recall 73%
#
# Recall falls off above ~800 because this site has very large pages (15k chars
# on average), so a big chunk mixes several topics and a generic page starts
# out-ranking the specific one. 800 matches the recall of the smaller sizes with
# half as many chunks, and leaves each retrieved source substantial enough to
# actually answer from — which matters on the local backend, where every chunk
# passed to the model is prefill time.
TARGET_CHARS = 800
OVERLAP_CHARS = 120
MIN_CHUNK_CHARS = 80

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the to was what "
    "when where which who will with you your do does can could would should about".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


# --- chunking -----------------------------------------------------------------

def _split_at_word(text: str, limit: int) -> tuple[str, str]:
    """Cut ``text`` at or before ``limit``, never through a word.

    The old code sliced on raw character offsets, which is why retrieved chunks
    read like ``'ills on aptitude, personality and social growth'`` and
    ``'h 55% marks in the qualifying examination'``. A chunk beginning mid-word
    is damaged twice over: the embedding is computed from a broken token
    sequence, and if it is retrieved the model is asked to answer from a
    fragment whose first sentence has no beginning.
    """
    if len(text) <= limit:
        return text, ""
    cut = text.rfind(" ", 0, limit + 1)
    if cut <= 0:
        cut = limit  # a single unbroken token longer than the window
    return text[:cut].strip(), text[cut:].strip()


def _tail_at_word(text: str, size: int) -> str:
    """The last ``size``-ish characters of ``text``, starting at a word boundary."""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()


def split_document(text: str) -> list[str]:
    """Paragraph-aware windows of ~800 chars with ~120 chars of overlap.

    Splitting on blank lines first keeps related sentences together; the overlap
    stops an answer that straddles a boundary from being cut in half. Every cut
    lands on a word boundary — see ``_split_at_word`` for why that matters more
    than it sounds.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        # A single oversized paragraph (a long table or list) is hard-split.
        while len(para) > TARGET_CHARS * 2:
            head, rest = _split_at_word(para, TARGET_CHARS)
            if not head:
                break
            chunks.append(head)
            para = f"{_tail_at_word(head, OVERLAP_CHARS)} {rest}".strip()
        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= TARGET_CHARS:
            buf = f"{buf}\n{para}"
        else:
            chunks.append(buf)
            overlap = _tail_at_word(buf, OVERLAP_CHARS)
            buf = f"{overlap}\n{para}" if overlap else para
    if buf:
        chunks.append(buf)

    return [c for c in (c.strip() for c in chunks) if len(c) >= MIN_CHUNK_CHARS]


# A line repeated across this fraction of documents is site chrome, not
# knowledge. Real prose does not recur across ~18 unrelated pages, but menu
# items ("Home", "Overview", "Vision, Mission"), comment-form furniture
# ("Leave a comment"), and event widgets ("Share this event") do.
CHROME_DOC_FRACTION = 0.05
CHROME_MAX_LINE_CHARS = 90


def _chrome_lines(docs: list[dict]) -> set[str]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update({ln.strip() for ln in doc["text"].splitlines() if ln.strip()})
    cutoff = max(3, int(CHROME_DOC_FRACTION * len(docs)))
    return {
        line
        for line, n in df.items()
        if n >= cutoff and len(line) <= CHROME_MAX_LINE_CHARS
    }


def _strip_chrome(text: str, chrome: set[str]) -> str:
    """Remove site chrome while keeping repeated *headings*.

    Frequency alone cannot tell a menu item from a section heading: on this site
    "Eligibility For Admission" appears on 27 programme pages and "Home" on 152,
    and a naive frequency filter deletes both. Deleting the heading is what made
    "Who can apply for admission?" unanswerable — the eligibility text survived,
    but the one line saying what it *was* did not, so nothing in the chunk
    matched the word "admission".

    The two are distinguishable by shape rather than by count. A navigation menu
    is a *run* of consecutive repeated lines; a section heading is a single
    repeated line surrounded by prose unique to its page. So a chrome line is
    only dropped when its neighbour is chrome too, which removes menus whole and
    leaves headings standing.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    is_chrome = [ln in chrome for ln in lines]
    out: list[str] = []
    for i, line in enumerate(lines):
        if is_chrome[i]:
            prev_chrome = i > 0 and is_chrome[i - 1]
            next_chrome = i + 1 < len(lines) and is_chrome[i + 1]
            if prev_chrome or next_chrome:
                continue  # part of a menu block
        out.append(line)
    return "\n".join(out)


# A heading is short, has no sentence-ending punctuation, and introduces the
# text under it. Carrying it into the embedding is what lets a chunk from deep
# in a programme page still match "admission" or "career opportunities".
HEADING_MAX_CHARS = 70


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > HEADING_MAX_CHARS:
        return False
    if line[-1] in ".!?,;:":
        return False
    return len(line.split()) <= 9


def _section_for(lines: list[str], upto: int) -> str:
    """The nearest heading at or above line ``upto``."""
    for i in range(min(upto, len(lines) - 1), -1, -1):
        if _looks_like_heading(lines[i]):
            return lines[i].strip()
    return ""


def build_chunks() -> list[dict]:
    """Chunk every document, after subtracting site chrome corpus-wide.

    Chrome removal happens here rather than at ingest because it needs the whole
    corpus to measure line frequency, and because it must apply to both ingest
    paths: Elementor pages come back from the REST API as nothing *but* their
    in-page nav widget, so without this the index fills with menus.
    """
    docs = [json.loads(ln) for ln in CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    chrome = _chrome_lines(docs)

    chunks: list[dict] = []
    dropped = 0
    for doc in docs:
        body_text = _strip_chrome(doc["text"], chrome)
        if len(body_text) < 100:
            dropped += 1  # nothing but chrome — indexing it would only add noise
            continue

        lines = body_text.splitlines()
        cursor = 0
        for i, body in enumerate(split_document(body_text)):
            # Which section this chunk fell in, so a chunk from deep inside a
            # page still carries the heading that says what it is about.
            first_line = body.splitlines()[0].strip() if body else ""
            try:
                cursor = lines.index(first_line, cursor)
            except ValueError:
                pass  # overlap text won't match a line exactly; keep the last section
            section = _section_for(lines, cursor)

            # Title and section are prepended for embedding/scoring only — the
            # `text` handed to the model stays exactly what the page said.
            context = "\n".join(p for p in (doc["title"], section) if p)
            chunks.append(
                {
                    "id": f"{doc['id']}#{i}",
                    "url": doc["url"],
                    "title": doc["title"],
                    "section": section,
                    "kind": doc.get("kind", ""),
                    "text": body,
                    "embed_text": f"{context}\n{body}" if context else body,
                }
            )
    print(f"chrome   : {len(chrome)} repeated menu lines stripped; {dropped} chrome-only docs dropped")
    return chunks


# --- BM25 ---------------------------------------------------------------------

@dataclass
class BM25:
    """Okapi BM25 over the chunk set (~30 lines; no extra dependency)."""

    postings: dict[str, list[tuple[int, int]]]  # term -> [(chunk idx, tf)]
    doc_len: np.ndarray
    avg_len: float
    n_docs: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: list[list[str]]) -> "BM25":
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, terms in enumerate(docs):
            lengths[i] = len(terms)
            for term, tf in Counter(terms).items():
                postings.setdefault(term, []).append((i, tf))
        return cls(postings, lengths, float(lengths.mean() or 1.0), len(docs))

    def scores(self, query_terms: list[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(query_terms):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for idx, tf in plist:
                norm = 1 - self.b + self.b * (self.doc_len[idx] / self.avg_len)
                out[idx] += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return out

    def to_json(self) -> dict:
        return {
            "postings": {t: p for t, p in self.postings.items()},
            "doc_len": self.doc_len.tolist(),
            "avg_len": self.avg_len,
            "n_docs": self.n_docs,
        }

    @classmethod
    def from_json(cls, d: dict) -> "BM25":
        return cls(
            {t: [tuple(x) for x in p] for t, p in d["postings"].items()},
            np.array(d["doc_len"], dtype=np.float32),
            d["avg_len"],
            d["n_docs"],
        )


# --- searcher -----------------------------------------------------------------

RRF_K = 60  # standard Reciprocal Rank Fusion constant

# Above this cosine, two chunks are the same content on two pages rather than
# two views of a topic. See Index._drop_near_duplicates for the measurements.
NEAR_DUPLICATE_COSINE = float(os.getenv("RETRIEVAL_NEAR_DUPLICATE", "0.98"))


@dataclass
class Hit:
    chunk: dict
    score: float        # RRF fusion score — ranks hits, does NOT measure relevance
    similarity: float   # dense cosine — absolute and comparable across queries


class Index:
    def __init__(self, chunks: list[dict], vectors: np.ndarray, bm25: BM25):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = bm25
        # A third BM25 ranker over titles alone was tried here and measured
        # *worse* twice (recall 47% -> 40% with default b, -> 33% with b=0):
        # titles are 2-4 words, so a single shared common term dominates and
        # "journalism programme" retrieved "Faculty Training Programme". Left
        # out deliberately — don't re-add it without an eval run.

    def search(
        self, query: str, k: int = 6, pool: int = 30, per_doc: int = 0, dedupe: bool = True
    ) -> list[Hit]:
        """Hybrid dense + BM25 search fused with RRF.

        Each hit carries both scores because they answer different questions.
        RRF is built from *rank position*, so its value is bounded and roughly
        constant whenever a chunk tops both rankers — an off-topic query still
        ranks something first and scores as highly as a good one. Only the dense
        cosine says how relevant the match actually is, so callers must gate on
        ``similarity`` and use ``score`` purely for ordering.

        ``per_doc`` caps how many chunks one source page may occupy. Pages on
        this site run to 15k characters, so a page that matches at all tends to
        match several times over and fill every slot — "do you provide job
        placement assistance" retrieved the placements page three times and gave
        the model one page's worth of context in three times the prefill. The
        cap is applied as a re-ordering, not a filter: if fewer than ``k``
        distinct pages match, the overflow chunks come back to fill the gap, so
        a question that genuinely lives on one page still gets it in full.
        """
        if not self.chunks:
            return []
        pool = min(pool, len(self.chunks))

        terms = tokenize(query)
        dense = self.vectors @ embed_query(query)        # exact cosine, one matmul
        lexical = self.bm25.scores(terms)

        # RRF: rank position, not raw score, so the scales never need
        # normalizing against each other.
        fused: dict[int, float] = {}

        def fuse(scores: np.ndarray, weight: float = 1.0) -> None:
            order = np.argsort(-scores)[:pool]
            for rank, idx in enumerate(order):
                if scores[idx] <= 0:
                    continue
                fused[int(idx)] = fused.get(int(idx), 0.0) + weight / (RRF_K + rank + 1)

        # Dense always contributes, including zero/negative scores, because it is
        # the only ranker that fires when the query shares no vocabulary.
        dense_order = np.argsort(-dense)[:pool]
        for rank, idx in enumerate(dense_order):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        fuse(lexical)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        if per_doc > 0:
            ranked = self._diversify(ranked, k, per_doc)
        # After the per-page cap, before truncating to k: a duplicate that is
        # dropped here should give its slot to the next distinct chunk, not
        # leave a hole.
        top = self._drop_near_duplicates(ranked, k) if dedupe else ranked[:k]
        return [Hit(self.chunks[i], s, float(dense[i])) for i, s in top]

    def _diversify(self, ranked: list[tuple[int, float]], k: int, per_doc: int) -> list[tuple[int, float]]:
        """Re-order so no single page takes more than ``per_doc`` of the top k."""
        taken: Counter[str] = Counter()
        picked: list[tuple[int, float]] = []
        spill: list[tuple[int, float]] = []
        for idx, score in ranked:
            # Chunk ids are "{document id}#{n}", so the page is the prefix.
            doc = self.chunks[idx]["id"].rsplit("#", 1)[0]
            if taken[doc] < per_doc:
                taken[doc] += 1
                picked.append((idx, score))
            else:
                spill.append((idx, score))
        # Overflow keeps its original rank order, so it only ever fills slots
        # that no other page was able to claim.
        return picked + spill

    def _drop_near_duplicates(self, ranked: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        """Suppress chunks that repeat content already selected.

        ``per_doc`` caps how much of one *page* is used, which does nothing about
        the same text living on two different pages — and this site is full of
        that. Every programme runs as "Shift I" and "Shift II" on separate URLs
        with near-identical prose, so "who can apply for admission" spent two of
        its eight slots on byte-identical Visual Communication text.

        Measured cosine between chunk vectors separates the cases cleanly:

            0.992  Visual Communication Shift I vs Shift II (identical text)
            0.915  two different chunks of the same SANKALP page
            0.808  Contact Us vs Home

        So 0.98 removes the duplicates while leaving genuinely distinct chunks of
        a shared topic alone. 31 pairs in the corpus are above it; 330 are above
        0.95, which is why the threshold is not lower.
        """
        kept: list[tuple[int, float]] = []
        for idx, score in ranked:
            if any(
                float(self.vectors[idx] @ self.vectors[other]) > NEAR_DUPLICATE_COSINE
                for other, _ in kept
            ):
                continue
            kept.append((idx, score))
            if len(kept) >= k:
                break
        return kept

    def dense_only(self, query: str, k: int = 1) -> list[Hit]:
        sims = self.vectors @ embed_query(query)
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[int(i)], float(sims[int(i)]), float(sims[int(i)])) for i in order]


_index: Index | None = None
_index_lock = threading.Lock()


META = DATA_DIR / "index-meta.json"


def load() -> Index | None:
    """Load the persisted index into memory (idempotent, thread-safe).

    Refuses an index built by a different encoder than the one that will embed
    queries against it. Dimensions usually differ, so the mismatch would surface
    as a matmul error — but two encoders of the same width would produce a
    working matmul over meaningless geometry, ranking chunks by nothing at all.
    That failure is invisible from the outside: retrieval still returns five
    confident-looking sources. Better to refuse and serve the curated answers.
    """
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is not None:
            return _index
        if not (VECTORS.exists() and CHUNKS.exists()):
            return None

        if META.exists():
            try:
                meta = json.loads(META.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            built_with = meta.get("embed_model")
            if built_with and built_with != model_name():
                raise RuntimeError(
                    f"index was built with embed model {built_with!r} but this process "
                    f"queries with {model_name()!r} — rebuild with `python -m kb.index`"
                )

        chunks = [json.loads(ln) for ln in CHUNKS.read_text(encoding="utf-8").splitlines() if ln.strip()]
        vectors = np.load(VECTORS)
        if vectors.shape[0] != len(chunks):
            raise RuntimeError(
                f"index is inconsistent: {vectors.shape[0]} vectors for {len(chunks)} chunks "
                "— rebuild with `python -m kb.index`"
            )
        bm25 = BM25.build([tokenize(c["embed_text"]) for c in chunks])
        _index = Index(chunks, vectors, bm25)
        return _index


def build() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    if not chunks:
        raise SystemExit(f"no chunks — run `python -m kb.ingest` first ({CORPUS} missing/empty)")

    print(f"chunking : {len(chunks)} chunks from {CORPUS.name}")
    vectors = embed_passages([c["embed_text"] for c in chunks])
    np.save(VECTORS, vectors)
    with CHUNKS.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    (DATA_DIR / "index-meta.json").write_text(
        json.dumps({"embed_model": model_name(), "chunks": len(chunks), "dim": int(vectors.shape[1])}),
        encoding="utf-8",
    )
    mb = vectors.nbytes / 1e6
    print(f"vectors  : {vectors.shape} float32 = {mb:.2f} MB resident  [{model_name()}]")
    print(f"written  : {VECTORS.name}, {CHUNKS.name}, index-meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", help="run a test query against the built index")
    args = ap.parse_args()
    if args.probe:
        idx = load()
        if idx is None:
            raise SystemExit("no index — run `python -m kb.index` first")
        for hit in idx.search(args.probe):
            print(f"  {hit.score:.4f}  {hit.chunk['title'][:44]:44s} {hit.chunk['text'][:90]!r}")
    else:
        build()
