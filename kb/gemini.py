"""One Gemini client, shared by answering, embeddings and speech-to-text.

Everything that talks to Google goes through here so that retries, timeouts and
model selection are decided once rather than three times slightly differently.

**Model choice was measured, not assumed.** The candidates were probed on this
exact task — grounded answering over retrieved college chunks, plus a refusal on
an off-topic question:

| model                   | grounded fact | off-topic        | latency | audio in |
|-------------------------|---------------|------------------|---------|----------|
| gemini-3.1-flash-lite   | correct       | declined exactly | 774 ms  | 1272 ms  |
| gemini-3.5-flash-lite   | correct       | declined exactly | 777 ms  | 1662 ms  |
| gemini-2.0-flash-lite   | 429 quota     | -                | -       | -        |
| gemini-2.5-flash-lite   | 404 retired   | -                | -       | -        |

All three working lite models emitted `NO_ANSWER_SENTENCE` verbatim on the
off-topic probe, which the local Qwen2.5-1.5B never managed reliably. That is the
single biggest correctness win in this change: the relevance gate provably cannot
separate on- from off-topic on this corpus (see `answering.py`), so the model
declining *is* the defense.

Two different defaults, because the price lists differ by modality:

* text  -> ``gemini-3.1-flash-lite``  ($0.25 /Mtok in, $1.50 /Mtok out)
* audio -> ``gemini-3.5-flash-lite``  ($0.30 /Mtok audio in, cheapest audio rate)

A grounded answer is ~2000 tokens in and ~30 out, so roughly $0.0005 per
generated answer — and most traffic never reaches the model at all, because the
semantic answer cache serves it first.

``*-latest`` aliases are deliberately **not** used. They move under you, and a
changed answer string is not just a quality risk here: answer text is the TTS
cache key, so a silent model swap invalidates every prewarmed clip at once.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Read ``.env`` into the environment without adding a dependency.

    Real environment variables always win, so a deployed service configured
    through its dashboard is never overridden by a file that happened to ship
    in the image.
    """
    path = APP_DIR / ".env"
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# The SDK reads GEMINI_API_KEY; GOOGLE_API_KEY is accepted as the older name.
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.1-flash-lite")
AUDIO_MODEL = os.getenv("GEMINI_AUDIO_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# 768 of the available 3072 dimensions. gemini-embedding-001 is a Matryoshka
# model, so a truncated prefix is still a valid embedding — at 810-2000 chunks
# the extra dimensions buy nothing measurable and cost 4x the memory and 4x the
# matmul. Truncated vectors come back UNNORMALIZED (measured norm 0.59 at 768),
# so `embed.py` must renormalize or every cosine score is silently wrong.
EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

REQUEST_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "6"))

# Free-tier embedding quota, in *texts* per minute — every item in a batch is
# counted separately, so a batch of 32 spends 32 of the 100. Measured by walking
# an index build into the limit: it throttles after roughly three batches.
# Overridden upward on a paid key, where this pacing is pure lost time.
EMBED_RPM = int(os.getenv("GEMINI_EMBED_RPM", "90"))

_client = None
_client_lock = threading.Lock()


def available() -> bool:
    """True when a key is configured and the SDK is importable."""
    if not API_KEY:
        return False
    try:
        import google.genai  # noqa: F401
    except ImportError:
        return False
    return True


def client():
    """The shared ``genai.Client`` (lazy, thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from google import genai
                from google.genai import types

                _client = genai.Client(
                    api_key=API_KEY,
                    http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),
                )
    return _client


# Errors worth trying again: quota (the free tier throttles hard and recovers),
# transient 5xx, and connection resets. A 400/404 never becomes valid by waiting.
_RETRYABLE = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE")


def is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in _RETRYABLE)


# A 429 body carries `"retryDelay": "23s"` — how long the quota window actually
# needs. Guessing with exponential backoff loses to simply reading it: the free
# embedding tier refills on a 60-second window, so a self-chosen 8-second retry
# is three wasted attempts before the one that was always going to work.
_RETRY_DELAY = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


def _server_retry_delay(exc: Exception) -> float | None:
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


def with_retry(fn, *, what: str, retries: int | None = None):
    """Call ``fn``, honouring the server's own retry advice on a 429.

    The free tier allows 100 embedded texts per minute, so an index rebuild is
    inherently rate-limited rather than merely flaky: it *will* be throttled
    repeatedly and must keep going anyway. Retries are therefore generous and
    the wait comes from the response when the server states one.
    """
    attempts = MAX_RETRIES if retries is None else retries
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified immediately below
            last = exc
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            advised = _server_retry_delay(exc)
            if advised is not None:
                # A small margin over the stated delay: the window is measured
                # server-side and coming back a hair early just burns a retry.
                delay = advised + 2.0
            else:
                # Full jitter: a rebuild fans out many identical calls, and
                # retrying in lockstep recreates the burst that caused the 429.
                delay = random.uniform(1.0, min(2.0 * (2**attempt), 30.0))
            print(
                f"[gemini] {what}: {type(exc).__name__} — "
                f"retry {attempt + 1}/{attempts - 1} in {delay:.1f}s"
            )
            time.sleep(delay)
    raise last if last else RuntimeError(f"{what} failed")


def generate(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> str:
    """Deterministic grounded generation.

    Temperature 0 for the same reason the local backend used it: this is
    extraction from supplied text, where sampling variety costs faithfulness and
    — because answer text is the TTS cache key — costs cache hits too.
    """
    from google.genai import types

    def call():
        return client().models.generate_content(
            model=model or TEXT_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
                # The corpus is a public college site and the questions are from
                # prospective students, so the default filters only ever fire as
                # false positives here. A blocked response would surface as an
                # empty answer; `answering.py` treats that as a model error and
                # falls back, so the avatar still speaks.
                safety_settings=[
                    types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
                    for c in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            ),
        )

    response = with_retry(call, what="generate")
    return (getattr(response, "text", None) or "").strip()


def generate_json(
    system: str,
    user: str,
    schema: dict,
    *,
    model: str | None = None,
    max_tokens: int = 500,
) -> dict:
    """Generation constrained to a JSON schema.

    Used for the answering verdict, where the model must commit to *judgements*
    (is this in scope, do the sources actually answer it) and not merely produce
    prose that implies them. A free-text answer can hedge its way into sounding
    grounded; a required ``sufficiency`` field cannot.

    Raises ``ValueError`` if the response is not parseable JSON, so the caller
    can fall back rather than speak a fragment.
    """
    import json

    from google.genai import types

    def call():
        return client().models.generate_content(
            model=model or TEXT_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=schema,
                safety_settings=[
                    types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
                    for c in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            ),
        )

    response = with_retry(call, what="generate_json")
    raw = (getattr(response, "text", None) or "").strip()
    if not raw:
        raise ValueError("empty completion")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"non-JSON completion: {raw[:120]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def transcribe(audio: bytes, mime_type: str, *, prompt: str, model: str | None = None) -> str:
    """Transcribe speech from raw audio bytes.

    Inline bytes rather than the Files API: clips here are a few seconds and a
    handful of KB, so an upload round-trip would double the latency of the one
    call the user is actually waiting on.
    """
    from google.genai import types

    def call():
        return client().models.generate_content(
            model=model or AUDIO_MODEL,
            contents=[types.Part.from_bytes(data=audio, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                safety_settings=[
                    types.SafetySetting(category=c, threshold="BLOCK_ONLY_HIGH")
                    for c in (
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                    )
                ],
            ),
        )

    response = with_retry(call, what="transcribe", retries=2)
    return (getattr(response, "text", None) or "").strip()


def status() -> dict:
    """What the Gemini path is configured to do, for ``/health``."""
    return {
        "configured": bool(API_KEY),
        "sdk": _sdk_version(),
        "text_model": TEXT_MODEL,
        "audio_model": AUDIO_MODEL,
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
    }


def _sdk_version() -> str | None:
    try:
        import google.genai

        return getattr(google.genai, "__version__", None)
    except ImportError:
        return None
