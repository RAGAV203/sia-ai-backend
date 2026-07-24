"""Fetch model weights at image-build time so Render cold starts don't
re-download ~340 MB every time the free instance spins back up."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
FILES = {
    "kokoro-v1.0.onnx": f"{RELEASE}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{RELEASE}/voices-v1.0.bin",
}


def main() -> None:
    for name, url in FILES.items():
        dest = Path(name)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} <- {url}")
        urllib.request.urlretrieve(url, name)
        print(f"[done] {name} ({dest.stat().st_size / 1e6:.1f} MB)")

    # Pre-download the Whisper weights into the image's HF cache.
    model = os.getenv("WHISPER_MODEL", "base")
    compute = os.getenv("WHISPER_COMPUTE", "int8")
    print(f"[get ] faster-whisper '{model}' ({compute})")
    from faster_whisper import WhisperModel

    WhisperModel(model, device="cpu", compute_type=compute)
    print("[done] all models cached")


if __name__ == "__main__":
    main()
