"""Embeddings for retrieval — Gemini on the query/passage path, local ONNX for
question-to-question comparison.

**Retrieval is asymmetric**: a short spoken question has to match a long scraped
passage, and the two sides must be encoded differently. ``gemini-embedding-001``
takes that as an explicit ``task_type`` (``RETRIEVAL_QUERY`` vs
``RETRIEVAL_DOCUMENT``) rather than the instruction-prefix hack BGE needed.
Measured on a real pair from this corpus at 768 dimensions:

    query "what is the NAAC grade"
      vs  "NAAC Accreditation: A++ Grade (2023)..."   cos 0.739   <- relevant
      vs  "The canteen provides hygienic food..."     cos 0.573   <- not

Two properties of this model are easy to get wrong and both are silent failures:

1. **Truncated vectors are not normalized.** It is a Matryoshka model, so asking
   for 768 of 3072 dimensions is legitimate — but the returned vector has norm
   ~0.59, not 1.0. Retrieval is a plain dot product against a matrix of these, so
   skipping renormalization does not error, it just quietly ranks by magnitude as
   much as by direction.
2. **A batch is capped at 100 inputs**, and the free tier returns 429 well below
   that (measured: 100 fails immediately, 32 is comfortable).

Because a rebuild is thousands of API calls against a quota that throttles,
every vector is cached on disk keyed by content. Re-running the index after a
chunker tweak then re-embeds only what actually changed — the difference between
a 20-second rebuild and one that dies partway through and has to be restarted.

Set ``EMBED_BACKEND=local`` to fall back to the bundled BGE ONNX encoder (no
network, no key); the index records which backend built it and refuses to serve
a mismatch, because comparing vectors from two different models is meaningless.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import gemini

APP_DIR = Path(__file__).resolve().parent.parent

BACKEND = os.getenv("EMBED_BACKEND", "gemini").strip().lower()
EMBED_DIM = gemini.EMBED_DIM if BACKEND == "gemini" else 384

# 100 is the hard API cap; 32 is what the free tier actually sustains.
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "32"))

VECTOR_CACHE = Path(os.getenv("EMBED_CACHE_PATH", APP_DIR / "kb-data" / "embed-cache.jsonl"))

# Repeat questions are common (suggestion chips, a visitor rephrasing), and a
# query embedding is a network round-trip on the path the user is waiting on.
QUERY_CACHE_SIZE = int(os.getenv("EMBED_QUERY_CACHE", "512"))

_query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_query_lock = threading.Lock()

_disk_cache: dict[str, list[float]] | None = None
_disk_lock = threading.Lock()
_disk_dirty = 0


def model_name() -> str:
    """Identity of the encoder that produced a vector set.

    Written into ``index-meta.json`` at build time and checked at load time —
    an index built by one model and queried by another produces plausible
    nonsense rather than an error, so the mismatch has to be caught explicitly.
    """
    return f"gemini:{gemini.EMBED_MODEL}:{EMBED_DIM}" if BACKEND == "gemini" else "bge"


# --- disk cache ---------------------------------------------------------------

def _cache_key(text: str, task: str) -> str:
    payload = f"{model_name()}|{task}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_disk_cache() -> dict[str, list[float]]:
    global _disk_cache
    if _disk_cache is not None:
        return _disk_cache
    with _disk_lock:
        if _disk_cache is not None:
            return _disk_cache
        cache: dict[str, list[float]] = {}
        if VECTOR_CACHE.exists():
            for line in VECTOR_CACHE.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    cache[row["k"]] = row["v"]
                except (json.JSONDecodeError, KeyError):
                    continue  # a partial line from an interrupted run
        _disk_cache = cache
        return _disk_cache


def _cache_put(entries: list[tuple[str, np.ndarray]]) -> None:
    """Append new vectors to the on-disk cache.

    Appending after every batch rather than at the end is deliberate: a rebuild
    that is killed by a quota error keeps everything it had already paid for.
    """
    if not entries:
        return
    cache = _load_disk_cache()
    with _disk_lock:
        try:
            VECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with VECTOR_CACHE.open("a", encoding="utf-8") as fh:
                for key, vec in entries:
                    values = [round(float(x), 6) for x in vec]
                    cache[key] = values
                    fh.write(json.dumps({"k": key, "v": values}) + "\n")
        except OSError as exc:
            print(f"[embed] cache write failed: {exc}")


# --- Gemini path --------------------------------------------------------------

def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity is a plain dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-9, None)).astype(np.float32)


_pace_lock = threading.Lock()
_pace_window: list[float] = []


def _pace(count: int) -> None:
    """Block until ``count`` more texts fit inside the per-minute quota.

    Being throttled is recoverable — `with_retry` honours the server's stated
    delay — but it is far cheaper to not trip the limit at all. Waiting here
    turns an index build from a sequence of failed calls and 25-second penalties
    into a steady walk through the quota.
    """
    if gemini.EMBED_RPM <= 0:
        return
    with _pace_lock:
        while True:
            now = time.monotonic()
            _pace_window[:] = [t for t in _pace_window if now - t < 60.0]
            if len(_pace_window) + count <= gemini.EMBED_RPM:
                _pace_window.extend([now] * count)
                return
            sleep_for = 60.0 - (now - _pace_window[0]) + 0.5
            print(f"[embed] quota pacing: waiting {sleep_for:.0f}s")
            time.sleep(max(sleep_for, 1.0))


def _gemini_encode(texts: list[str], task: str, *, use_cache: bool = True) -> np.ndarray:
    from google.genai import types

    out = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    cache = _load_disk_cache() if use_cache else {}

    pending: list[int] = []
    for i, text in enumerate(texts):
        key = _cache_key(text, task) if use_cache else ""
        hit = cache.get(key) if use_cache else None
        if hit is not None:
            out[i] = np.array(hit, dtype=np.float32)
        else:
            pending.append(i)

    if pending:
        print(f"[embed] {len(pending)} to embed, {len(texts) - len(pending)} from cache")

    for start in range(0, len(pending), BATCH_SIZE):
        idxs = pending[start : start + BATCH_SIZE]
        batch = [texts[i] if texts[i].strip() else " " for i in idxs]

        _pace(len(batch))

        def call(batch=batch):
            return gemini.client().models.embed_content(
                model=gemini.EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=EMBED_DIM
                ),
            )

        response = gemini.with_retry(call, what=f"embed[{task}]")
        raw = np.array([e.values for e in response.embeddings], dtype=np.float32)
        # Renormalize: truncated Matryoshka vectors arrive with norm != 1.
        vectors = _normalize(raw)
        for slot, vec in zip(idxs, vectors):
            out[slot] = vec
        if use_cache:
            _cache_put([(_cache_key(texts[i], task), out[i]) for i in idxs])
        if len(pending) > BATCH_SIZE:
            print(f"[embed]   {min(start + BATCH_SIZE, len(pending))}/{len(pending)}")

    return out


# --- local ONNX path (fallback / offline) -------------------------------------

_LOCAL_DIR = Path(os.getenv("EMBED_MODEL_DIR", APP_DIR / "models/bge"))
_LOCAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_TOKENS = 256

_lock = threading.Lock()
_session = None
_tokenizer = None


def _load_local():
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(_LOCAL_DIR / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(length=None)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "0")) or max(
            1, min(4, os.cpu_count() or 1)
        )
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _tokenizer = tok
        _session = ort.InferenceSession(
            str(_LOCAL_DIR / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )


def _local_encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    _load_local()
    assert _tokenizer is not None and _session is not None

    names = {i.name for i in _session.get_inputs()}
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = [t if t.strip() else " " for t in texts[start : start + batch_size]]
        encoded = _tokenizer.encode_batch(chunk)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in names}
        hidden = _session.run(None, feed)[0]
        out.append(_normalize(hidden[:, 0]))  # BGE pools [CLS]
    return np.vstack(out)


# --- public API ---------------------------------------------------------------

def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed indexable content (the document side of the asymmetric pair)."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    if BACKEND == "gemini":
        return _gemini_encode(texts, "RETRIEVAL_DOCUMENT")
    return _local_encode(texts, batch_size)


def embed_queries(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed search queries (the query side of the asymmetric pair)."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    if BACKEND == "gemini":
        # Queries are not written to the disk cache: they are unbounded and
        # mostly one-off, so they would grow the file without ever being reused.
        # The in-memory LRU in `embed_query` covers the repeats that matter.
        return _gemini_encode(texts, "RETRIEVAL_QUERY", use_cache=False)
    return _local_encode([_LOCAL_QUERY_PREFIX + t for t in texts], batch_size)


def embed_query(text: str) -> np.ndarray:
    """Embed one query, memoized — this sits on the latency path of every ask."""
    key = text.strip().lower()
    with _query_lock:
        hit = _query_cache.get(key)
        if hit is not None:
            _query_cache.move_to_end(key)
            return hit

    vector = embed_queries([text])[0]

    with _query_lock:
        _query_cache[key] = vector
        _query_cache.move_to_end(key)
        while len(_query_cache) > QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return vector


embed = embed_passages


def embed_one(text: str) -> np.ndarray:
    return embed_passages([text])[0]


def status() -> dict:
    return {
        "backend": BACKEND,
        "model": model_name(),
        "dim": EMBED_DIM,
        "cached_vectors": len(_load_disk_cache()) if BACKEND == "gemini" else 0,
        "query_cache": len(_query_cache),
    }


# --- symmetric encoder, pinned local ------------------------------------------
#
# Comparing two *questions* ("did someone already ask this?") is a symmetric
# task, and it must not move when the retrieval model changes.
#
# It did once: the answer cache shared the retrieval encoder, and switching that
# from MiniLM to BGE shifted every score upward. Two genuinely different
# questions about one topic — "what is the fee for B.Com" and "how long is the
# B.Com course" — went from 0.567 to 0.764, sailing past a threshold calibrated
# against the old scale and turning into a false cache hit.
#
# That argument applies with more force now that retrieval is a *remote* model:
# pinning the cache to a local symmetric encoder keeps its 0.65/0.93 thresholds
# valid, keeps a cache lookup free, and keeps it working with no network at all.
_SYMMETRIC_DIR = Path(os.getenv("SYMMETRIC_MODEL_DIR", APP_DIR / "models/minilm"))
_sym_lock = threading.Lock()
_sym_session = None
_sym_tokenizer = None


def _load_symmetric():
    global _sym_session, _sym_tokenizer
    if _sym_session is not None:
        return
    with _sym_lock:
        if _sym_session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(_SYMMETRIC_DIR / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(length=None)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # a single short string at a time
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _sym_tokenizer = tok
        _sym_session = ort.InferenceSession(
            str(_SYMMETRIC_DIR / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )


def embed_symmetric(text: str) -> np.ndarray:
    """Encode text for question-to-question comparison (mean-pooled MiniLM)."""
    _load_symmetric()
    assert _sym_tokenizer is not None and _sym_session is not None

    encoded = _sym_tokenizer.encode_batch([text if text.strip() else " "])
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    names = {i.name for i in _sym_session.get_inputs()}
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in names:
        feed["token_type_ids"] = np.zeros_like(ids)
    feed = {k: v for k, v in feed.items() if k in names}

    hidden = _sym_session.run(None, feed)[0]
    m = mask[..., None].astype(np.float32)
    pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    return _normalize(pooled)[0]
