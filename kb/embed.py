"""Sentence embeddings via ONNX Runtime — 384 dims, int8, ~23-34 MB.

Deliberately the same runtime Kokoro already uses, so the knowledge base adds a
small model and no new heavy dependency: no PyTorch, no sentence-transformers.
Tokenization uses the `tokenizers` library that faster-whisper already pulls in.

**Queries and passages are embedded differently.** Retrieval here is asymmetric —
a short question has to match a long passage — and the two most obvious model
choices behave very differently at that:

* ``all-MiniLM-L6-v2`` is trained for *symmetric* similarity (is sentence A like
  sentence B). Used for search it compresses short questions into a narrow band,
  which is what made the relevance gate hard to place: genuinely relevant pages
  landed at 0.37-0.43 while an off-topic question reached 0.468.
* ``multi-qa-MiniLM-L6-cos-v1`` and ``bge-small-en-v1.5`` are trained for
  question-to-passage search. BGE additionally expects an instruction prefix on
  the query side only, which is why prefixes are part of the model config rather
  than the caller's problem.

Select with ``EMBED_MODEL=minilm|multiqa|bge``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent

MAX_TOKENS = 256          # trained window for these encoders; longer input loses its tail
EMBED_DIM = 384

# name -> (directory, query prefix, passage prefix)
_MODELS = {
    "minilm": ("models/minilm", "", ""),
    "multiqa": ("models/multiqa", "", ""),
    # BGE was trained with this exact instruction on queries only. Omitting it,
    # or adding it to passages too, measurably degrades retrieval.
    "bge": ("models/bge", "Represent this sentence for searching relevant passages: ", ""),
}

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge").strip().lower()
if EMBED_MODEL not in _MODELS:
    EMBED_MODEL = "bge"

_dir, QUERY_PREFIX, PASSAGE_PREFIX = _MODELS[EMBED_MODEL]
MODEL_DIR = Path(os.getenv("EMBED_MODEL_DIR", APP_DIR / _dir))

_lock = threading.Lock()
_session = None
_tokenizer = None


def model_name() -> str:
    return EMBED_MODEL


def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_TOKENS)
        tok.enable_padding(length=None)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.getenv("ONNX_THREADS", "0")) or max(1, min(4, os.cpu_count() or 1))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _tokenizer = tok
        _session = ort.InferenceSession(
            str(MODEL_DIR / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )


def _encode(texts: list[str], batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    _load()
    assert _tokenizer is not None and _session is not None

    inputs = {i.name for i in _session.get_inputs()}
    out: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = [t if t.strip() else " " for t in texts[start : start + batch_size]]
        encoded = _tokenizer.encode_batch(chunk)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in inputs}

        hidden = _session.run(None, feed)[0]  # (batch, seq, dim)

        if EMBED_MODEL == "bge":
            # BGE pools the [CLS] token rather than averaging.
            pooled = hidden[:, 0]
        else:
            # Mean-pool over real tokens only — padding must not drag vectors
            # toward zero.
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)

        # Normalizing here makes cosine similarity a plain dot product, so the
        # whole search stays one matmul.
        pooled = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
        out.append(pooled.astype(np.float32))

    return np.vstack(out)


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed indexable content."""
    return _encode([PASSAGE_PREFIX + t for t in texts] if PASSAGE_PREFIX else texts, batch_size)


def embed_queries(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed search queries (may carry a model-specific instruction prefix)."""
    return _encode([QUERY_PREFIX + t for t in texts] if QUERY_PREFIX else texts, batch_size)


def embed_query(text: str) -> np.ndarray:
    return embed_queries([text])[0]


embed = embed_passages


def embed_one(text: str) -> np.ndarray:
    return embed_passages([text])[0]


# --- symmetric encoder, pinned ------------------------------------------------
#
# Comparing two *questions* ("is this the same thing someone already asked?") is
# a symmetric task, and it must not move when the retrieval model changes.
#
# It did once: the answer cache shared the retrieval encoder, and switching that
# from MiniLM to BGE shifted every score upward. Two genuinely different
# questions about one topic — "what is the fee for B.Com" and "how long is the
# B.Com course" — went from 0.567 to 0.764, sailing past a threshold calibrated
# against the old scale and turning into a false cache hit. Pinning the cache to
# its own symmetric model keeps that calibration valid no matter what the
# retriever uses.
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
        opts.intra_op_num_threads = 1  # single short string at a time
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
    pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None)
    return pooled[0].astype(np.float32)
