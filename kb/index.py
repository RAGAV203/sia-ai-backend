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

def split_document(text: str) -> list[str]:
    """Paragraph-aware windows of ~1000 chars with ~150 chars of overlap.

    Splitting on blank lines first keeps related sentences together; the overlap
    stops an answer that straddles a boundary from being cut in half.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        # A single oversized paragraph (a long table or list) is hard-split.
        while len(para) > TARGET_CHARS * 2:
            head, para = para[:TARGET_CHARS], para[TARGET_CHARS - OVERLAP_CHARS :]
            chunks.append(head)
        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= TARGET_CHARS:
            buf = f"{buf}\n{para}"
        else:
            chunks.append(buf)
            buf = (buf[-OVERLAP_CHARS:] + "\n" + para) if len(buf) > OVERLAP_CHARS else para
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
        body_text = "\n".join(
            ln for ln in doc["text"].splitlines() if ln.strip() and ln.strip() not in chrome
        )
        if len(body_text) < 100:
            dropped += 1  # nothing but chrome — indexing it would only add noise
            continue
        doc = {**doc, "text": body_text}
        for i, body in enumerate(split_document(doc["text"])):
                chunks.append(
                    {
                        "id": f"{doc['id']}#{i}",
                        "url": doc["url"],
                        "title": doc["title"],
                        "kind": doc.get("kind", ""),
                        "text": body,
                        # The title is prepended for embedding/scoring only: a chunk
                        # from deep in a page otherwise loses all topic context.
                        "embed_text": f"{doc['title']}\n{body}" if doc["title"] else body,
                    }
                )
    print(f"chrome   : {len(chrome)} repeated lines stripped; {dropped} chrome-only docs dropped")
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

    def search(self, query: str, k: int = 6, pool: int = 30, per_doc: int = 0) -> list[Hit]:
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
        top = ranked[:k]
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

    def dense_only(self, query: str, k: int = 1) -> list[Hit]:
        sims = self.vectors @ embed_query(query)
        order = np.argsort(-sims)[:k]
        return [Hit(self.chunks[int(i)], float(sims[int(i)]), float(sims[int(i)])) for i in order]


_index: Index | None = None
_index_lock = threading.Lock()


def load() -> Index | None:
    """Load the persisted index into memory (idempotent, thread-safe)."""
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is not None:
            return _index
        if not (VECTORS.exists() and CHUNKS.exists()):
            return None
        chunks = [json.loads(ln) for ln in CHUNKS.read_text(encoding="utf-8").splitlines() if ln.strip()]
        vectors = np.load(VECTORS)
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
