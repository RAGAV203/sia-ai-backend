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

# Sentences synthesized concurrently, each on its own Kokoro session.
#
# This is what keeps streamed speech continuous. Measured on a 4-sentence,
# 20-second answer:
#
#   int8  1 worker  x 4 threads   RTF 0.95    gaps at every seam
#   int8  2 workers x 2 threads   RTF 0.55    smooth
#   int8  4 workers x 2 threads   RTF 0.43
#   fp32  1 worker  x 4 threads   RTF 0.32
#
# Synthesis has to outrun playback or the client runs dry mid-answer, and 0.95
# leaves no margin for a second listener or the prewarm thread. Each worker
# holds its own copy of the weights (~92 MB int8, ~325 MB fp32), so 2 is the
# default: it restores the margin and still fits a 512 MB instance.
SYNTH_WORKERS = max(1, int(os.getenv("TTS_WORKERS", "2")))

# ONNX Runtime intra-op threads *per session*, so the pool as a whole stays
# inside the core budget rather than oversubscribing it.
#
# Kokoro is small: one session peaks around 4 threads and is slower at 8,
# because per-op sync cost overtakes the extra parallelism. That budget is now
# shared — measured with 2 workers, 2 threads each (RTF 0.55) beat 4 threads
# each (RTF 0.63), because eight busy threads on twelve cores spend more time
# contending than computing. Extra cores are used by adding *sessions*, not by
# widening one.
_thread_budget = min(4, os.cpu_count() or 1)
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "0")) or max(1, _thread_budget // SYNTH_WORKERS)

# Strip the vocoder's leading/trailing silence from each clip and replace it
# with one uniform pause, so butt-joined sentences do not inherit ~135 ms of
# dead air at every seam.
TRIM_SILENCE = _flag("TTS_TRIM_SILENCE")

# ONNX Runtime's CPU arena allocator.
#
# It trades memory for speed, and the ratio is extreme enough to be worth a
# switch. Measured, two warmed Kokoro sessions:
#
#   arena on    598 MB RSS   RTF 0.96
#   arena off   336 MB RSS   RTF 1.21
#
# Off is ~25% slower per session, which can be bought back by running more of
# them — 3 sessions without the arena reached RTF 0.95 at 455 MB, still less
# than 2 sessions with it. Worth setting to 0 on a memory-capped host; leave on
# where RAM is available, since fewer sessions also means faster first audio.
TTS_MEM_ARENA = _flag("TTS_MEM_ARENA")

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
