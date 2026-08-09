"""Fetch model weights at image-build time so Render cold starts don't
re-download them every time the instance spins back up.

Defaults to the **int8** Kokoro model (~88 MB) to keep resident memory low so
small instances don't get OOM-killed (which surfaces as a 502). Override with
KOKORO_MODEL=kokoro-v1.0.onnx (fp32, ~310 MB) for maximum quality on a big box.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
# app.py accepts KOKORO_MODEL=auto ("use the fastest weights on disk"), but a
# downloader needs a concrete filename — resolve it to the lean int8 build.
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "kokoro-v1.0.int8.onnx")
if KOKORO_MODEL == "auto":
    KOKORO_MODEL = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = os.getenv("KOKORO_VOICES", "voices-v1.0.bin")
FALLBACK_MODEL = "kokoro-v1.0.onnx"  # always present in the release


def _download(name: str) -> bool:
    dest = Path(name)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name} already present")
        return True
    url = f"{RELEASE}/{name}"
    try:
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, name)
        print(f"[done] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"[warn] {name} failed: {exc}")
        return False


def main() -> None:
    _download(KOKORO_VOICES)
    # If the configured model can't be fetched, grab the fp32 fallback so the
    # image always has *a* usable model (app.py picks whichever file exists).
    if not _download(KOKORO_MODEL) and KOKORO_MODEL != FALLBACK_MODEL:
        print(f"[warn] falling back to {FALLBACK_MODEL}")
        _download(FALLBACK_MODEL)

    # Pre-download the Whisper weights into the image's HF cache. Whisper is the
    # STT *fallback* now, but baking it in is still right: the moment it is
    # needed is the moment the network to Google is already failing, which is
    # the worst possible time to discover the weights were never fetched.
    model = os.getenv("WHISPER_MODEL", "base")
    compute = os.getenv("WHISPER_COMPUTE", "int8")
    print(f"[get ] faster-whisper '{model}' ({compute})")
    from faster_whisper import WhisperModel

    WhisperModel(model, device="cpu", compute_type=compute)

    _download_kb_models()
    print("[done] all models cached")


# Knowledge-base models.
#
# Only MiniLM is still fetched. It is the *symmetric* encoder behind the answer
# cache — comparing two questions to each other — which is pinned local on
# purpose: its 0.65/0.93 thresholds were calibrated against this model's score
# distribution, and it keeps a cache lookup free and offline. See kb/embed.py.
#
# Two downloads that used to live here are gone: the 986 MB Qwen2.5-1.5B GGUF
# (generation moved to Gemini) and the BGE retrieval encoder (retrieval
# embeddings are a Gemini call now). BGE is still supported via
# EMBED_BACKEND=local, but it is no longer baked into every image for a path
# most deployments never take.
_HF_FILES = [
    ("Xenova/all-MiniLM-L6-v2", "onnx/model_quantized.onnx", "models/minilm/model.onnx"),
    ("Xenova/all-MiniLM-L6-v2", "tokenizer.json", "models/minilm/tokenizer.json"),
]


def _download_kb_models() -> None:
    if os.getenv("KB_ENABLED", "1") == "0":
        print("[skip] knowledge-base models (KB_ENABLED=0)")
        return
    import shutil

    from huggingface_hub import hf_hub_download

    for repo, filename, dest in _HF_FILES:
        path = Path(dest)
        if path.exists() and path.stat().st_size > 0:
            print(f"[skip] {dest} already present")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[get ] {dest} <- {repo}/{filename}")
        try:
            cached = hf_hub_download(repo_id=repo, filename=filename)
            shutil.copyfile(cached, path)
            print(f"[done] {dest} ({path.stat().st_size / 1e6:.0f} MB)")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {dest} failed: {exc}")


if __name__ == "__main__":
    main()
