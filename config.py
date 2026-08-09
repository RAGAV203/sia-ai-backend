"""Every environment-driven setting in one place.

The settings used to be declared beside the code that read them, spread across
``app.py``, ``kb/answering.py``, ``kb/embed.py`` and ``kb/llm_local.py``. That
was survivable while the service was one file, but it made two things hard that
matter now: seeing what a deployment can actually be configured to do, and
noticing when two modules had drifted onto different defaults for the same idea.

Gemini-specific settings stay in ``kb/gemini.py`` — they belong with the client
that validates and uses them.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default) not in ("0", "false", "False", "")


# --- knowledge base -----------------------------------------------------------
# Without it (or without an API key) the service still answers from the curated
# keyword set, whose clips are prewarmed.
KB_ENABLED = _flag("KB_ENABLED")

# --- TTS ----------------------------------------------------------------------
# "auto" picks the fastest weights actually present (see services/tts.py).
KOKORO_MODEL = os.getenv("KOKORO_MODEL", "auto")
KOKORO_VOICES = os.getenv("KOKORO_VOICES", "voices-v1.0.bin")
# hf_alpha = Hindi female -> Indian-accented English. lang="en-us" keeps English
# pronunciation correct while the speaker embedding supplies the accent.
TTS_VOICE = os.getenv("TTS_VOICE", "hf_alpha")
TTS_LANG = os.getenv("TTS_LANG", "en-us")
TTS_SPEED = float(os.getenv("TTS_SPEED", "0.9"))  # a touch slower -> calm, gentle

# Default next to the app (not /tmp, which isn't a real path on Windows) so the
# cache survives restarts. Docker/Render override it with TTS_CACHE_DIR.
CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", str(APP_DIR / "tts-cache")))

# ONNX Runtime intra-op threads. Kokoro is a small model: throughput peaks around
# 4 threads and *regresses* past that (measured RTF on a 12-core CPU: 1 thread
# 3.1, 4 threads 0.42, 12 threads 0.60), because per-op sync cost outweighs the
# extra parallelism. 0/unset -> pick a sane value for this host.
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "0")) or max(1, min(4, os.cpu_count() or 1))

# --- startup ------------------------------------------------------------------
# Pre-synthesize the knowledge base at boot (background thread). This is what
# makes the site feel instant; turn it off only on a host too small to hold the
# models. WARMUP=1 is honoured as the older name for the same flag.
PREWARM = _flag("PREWARM", os.getenv("WARMUP", "1"))

# Load the Whisper *fallback* at boot. Now that Gemini serves the STT path, this
# defaults off: it costs ~200 MB resident to shave a model load off a code path
# that only runs when the API is unreachable. Set PREWARM_STT=1 on a deployment
# that expects to run without a key.
PREWARM_STT = _flag("PREWARM_STT", "0")

# --- HTTP ---------------------------------------------------------------------
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
