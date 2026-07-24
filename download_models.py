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
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "kokoro-v1.0.int8.onnx")
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

    # Pre-download the Whisper weights into the image's HF cache.
    model = os.getenv("WHISPER_MODEL", "base")
    compute = os.getenv("WHISPER_COMPUTE", "int8")
    print(f"[get ] faster-whisper '{model}' ({compute})")
    from faster_whisper import WhisperModel

    WhisperModel(model, device="cpu", compute_type=compute)
    print("[done] all models cached")


if __name__ == "__main__":
    main()
