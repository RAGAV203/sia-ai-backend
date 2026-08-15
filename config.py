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
#
# One voice serves all four languages. hf_alpha is a Hindi speaker, and every
# phoneme espeak-ng produces for Tamil and Malay falls inside Kokoro's 114-symbol
# vocabulary, so they synthesize in the same voice through the same cache. See
# languages.py for what that does and does not buy.
TTS_VOICE = os.getenv("TTS_VOICE", "hf_alpha")

# Languages whose synthesis is delegated to a neural voice instead of Kokoro.
#
# Kokoro carries Tamil *phonetically* — every espeak phoneme lands inside its
# vocabulary — but it was never trained on Tamil prosody, and the result is
# audibly a Hindi speaker reading Tamil aloud. Measured by round-tripping each
# clip back through speech-to-text, which is the closest thing to an objective
# pronunciation score available here:
#
#   Kokoro hf_alpha, Tamil        73.8%   hallucinated words, mangled compounds
#   edge-tts ta-IN-Pallavi        ~96%    1.1-1.6 s per sentence
#   gemini-3.1-flash-tts          98.6%   5-7 s per sentence
#
# Gemini scores highest and is unusable here anyway: the free tier allows **3
# requests per minute** for the TTS models, and a four-sentence answer
# synthesizes four clips concurrently, so a single Tamil reply exhausts the
# quota before it finishes speaking. Set `ta=gemini:Sulafat` on a paid key.
#
# Format: `code=backend:voice`, comma separated. Anything not listed uses
# Kokoro with TTS_VOICE, so English, Hindi and Malay are untouched — Hindi is
# native to hf_alpha and Malay measured fine.
_DEFAULT_VOICE_OVERRIDES = "ta=edge:ta-IN-PallaviNeural"


def _parse_overrides(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        code, voice = item.split("=", 1)
        code, voice = code.strip().lower(), voice.strip()
        if code and voice:
            out[code] = voice
    return out


# Set TTS_VOICE_OVERRIDES="" to force Kokoro everywhere (no network, no
# unofficial endpoint) at the cost of the Tamil accent.
TTS_VOICE_OVERRIDES = _parse_overrides(
    os.getenv("TTS_VOICE_OVERRIDES", _DEFAULT_VOICE_OVERRIDES)
)

# Only consulted when a voice override names the `gemini` backend.
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")

# The phonemizer language for *English* only; every other language takes its
# espeak code from the registry in languages.py. This stays configurable because
# it was, and a deployment that set it to en-gb should keep getting en-gb.
TTS_LANG = os.getenv("TTS_LANG", "en-us")
TTS_SPEED = float(os.getenv("TTS_SPEED", "0.9"))  # a touch slower -> calm, gentle

# Default next to the app (not /tmp, which isn't a real path on Windows) so the
# cache survives restarts. Docker/Render override it with TTS_CACHE_DIR.
CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", str(APP_DIR / "tts-cache")))

# --- languages ----------------------------------------------------------------
# Which of the four registered languages this deployment offers. Trimming the
# list is the supported way to run English-only on a small host: it shrinks the
# boot prewarm proportionally, since every language warms its own clips.
SUPPORTED_LANGS = [
    code.strip().lower()
    for code in os.getenv("SUPPORTED_LANGS", "en,ta,hi,ms").split(",")
    if code.strip()
]

# Translations of the fixed strings live here, keyed by content hash. Same
# reasoning as CACHE_DIR: beside the app so it survives a restart, because a
# cold translation cache also means a cold TTS cache — the clips are keyed on
# the translated text.
TRANSLATION_CACHE_DIR = Path(
    os.getenv("TRANSLATION_CACHE_DIR", str(APP_DIR / "translation-cache"))
)

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

# Languages to prewarm, and in what order. English first is not cosmetic: the
# prewarm thread is serial, so whichever language is listed first is the one
# that is instant for the first visitor. The rest fill in over the following
# minute or two while the service is already answering.
#
# Defaults to every supported language. Set to `en` on a host where the extra
# boot CPU matters more than a first Tamil visitor waiting for synthesis.
PREWARM_LANGS = [
    code.strip().lower()
    for code in os.getenv("PREWARM_LANGS", ",".join(SUPPORTED_LANGS)).split(",")
    if code.strip()
]

# Load the Whisper *fallback* at boot. Now that Gemini serves the STT path, this
# defaults off: it costs ~200 MB resident to shave a model load off a code path
# that only runs when the API is unreachable. Set PREWARM_STT=1 on a deployment
# that expects to run without a key.
PREWARM_STT = _flag("PREWARM_STT", "0")

# --- HTTP ---------------------------------------------------------------------
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
