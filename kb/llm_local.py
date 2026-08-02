"""Local generation with a small instruct model on CPU — no API, no network.

Model choice was made by benchmarking candidates on this exact task (grounded
answering over retrieved college-site chunks), not on general leaderboards:

| model              | size   | grounded fact | refuses politely | prompt injection |
|--------------------|--------|---------------|------------------|------------------|
| Qwen2.5-1.5B Q4_K_M| 986 MB | correct       | yes              | **resisted**     |
| Llama-3.2-1B Q4_K_M| 808 MB | correct       | hedged           | **"OWNED"** ✗    |
| Qwen2.5-0.5B Q4_K_M| 398 MB | **wrong** ✗   | yes              | resisted         |

Llama-3.2-1B obeyed an instruction planted inside a `<source>` block and replied
"OWNED OWNED OWNED" — unusable for a system whose knowledge base is scraped from
a CMS. Qwen2.5-0.5B stayed safe but answered "A grade" when the accreditation is
A++ (it picked a decade-old value out of the same chunk), which is exactly the
kind of confident-but-wrong answer a college assistant cannot give.

**Small models are markedly weaker at resisting injection than frontier models.**
That is the real cost of running locally, and it is why the guard layers in
`guard.py` and the structural containment in `prompts.py` matter more here, not
less. The model is still given no tools, so the blast radius stays bounded.

Latency on CPU is dominated by *prefill*, roughly 7 ms per prompt token, which
is what caps how much context the retriever may pass:

| chunks | prompt tokens | time to answer |
|--------|---------------|----------------|
| 2      | 649           | 5.8 s          |
| 3      | 920           | 6.4 s          |
| 6      | 1803          | 12.2 s         |

So the local path retrieves fewer, shorter chunks than the API path would, and
leans on the semantic answer cache to keep repeat questions instant.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(os.getenv("LLM_MODEL_PATH", APP_DIR / "models/llm/qwen2.5-1.5b.gguf"))
N_CTX = int(os.getenv("LLM_CTX", "4096"))
# Measured best on a 12-core CPU; large batches speed up prefill, which is the
# bottleneck, without hurting the short generations this service produces.
N_THREADS = int(os.getenv("LLM_THREADS", "0")) or max(1, min(12, os.cpu_count() or 4))
N_BATCH = int(os.getenv("LLM_BATCH", "2048"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "180"))

_llm = None
_load_lock = threading.Lock()
# llama.cpp keeps mutable state per context, so concurrent generations on one
# instance corrupt each other. FastAPI runs sync endpoints in a threadpool, so
# requests genuinely can overlap — serialize them.
_gen_lock = threading.Lock()


def available() -> bool:
    if not MODEL_PATH.exists():
        return False
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return True


def _get_llm():
    global _llm
    if _llm is None:
        with _load_lock:
            if _llm is None:
                from llama_cpp import Llama

                _llm = Llama(
                    model_path=str(MODEL_PATH),
                    n_ctx=N_CTX,
                    n_threads=N_THREADS,
                    n_threads_batch=N_THREADS,
                    n_batch=N_BATCH,
                    verbose=False,
                )
                print(f"[llm] {MODEL_PATH.name} on {N_THREADS} threads (ctx {N_CTX})")
    return _llm


def generate(system: str, user: str, max_tokens: int | None = None) -> str:
    """Deterministic grounded generation.

    Temperature 0: this is extraction from supplied text, where sampling variety
    buys nothing and costs both faithfulness and — because the TTS cache is keyed
    on exact answer text — cache hits.

    **Never call ``llm.reset()`` here.** llama.cpp keeps the KV cache from the
    previous call and reuses however much of the new prompt shares a prefix with
    it. The system prompt is constant and comes first, so that is ~457 tokens of
    prefill skipped on every request after the first — measured 3.16 s with a
    reset between calls versus 0.89 s without, a 3.6x difference on the dominant
    cost. The lock below is what makes this safe: the cache is single-sequence
    shared state, so overlapping generations would corrupt each other's prefix.
    """
    llm = _get_llm()
    with _gen_lock:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens or MAX_TOKENS,
            temperature=0.0,
            seed=7,
        )
    return (out["choices"][0]["message"]["content"] or "").strip()


def warm() -> None:
    """Load weights and prime the KV cache with the real system prompt.

    Priming matters as much as loading: the cache only helps a request whose
    prefix it already holds, so warming with a *different* system prompt would
    leave the first real question paying the full ~3 s prefill anyway.
    """
    try:
        from .prompts import system_prompt

        generate(system_prompt("local"), "<question>\nhello\n</question>", max_tokens=4)
    except Exception as exc:  # noqa: BLE001
        print(f"[llm] warm failed: {exc}")
