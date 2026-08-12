"""Phonetic transliteration, so a QWERTY keyboard can type Tamil and Hindi.

Most visitors who read Tamil or Hindi do not have a Tamil or Hindi keyboard
installed — especially on a desktop, where there is no equivalent of the phone
keyboard's language switcher. Offering the interface in Tamil and then requiring
a Tamil keyboard to use it is offering it to almost nobody, so the composer lets
them type "vanakkam" and turns it into வணக்கம் as they go.

**Why this is proxied rather than called from the browser.** Google Input Tools
answers cross-origin requests, so the frontend *could* call it directly, and
three things argue against that:

* The upstream URL and its parameters stay server-side, so swapping the provider
  — or dropping in an offline transliterator — is a change to this file rather
  than a frontend release.
* Responses are cached here, shared across every visitor. Transliteration
  requests are extraordinarily repetitive: every prefix of every word is its own
  lookup, so a single typed word issues five or six of them and the next visitor
  typing the same word issues none.
* A third-party request from the page would send the visitor's keystrokes to a
  host the site does not control, from their own IP, on a page that does not
  otherwise talk to Google.

**It must never be load-bearing.** This sits on the keystroke path: a slow or
failed lookup has to leave the Latin text alone and get out of the way. Every
failure here returns empty candidates rather than an error status, and the
timeout is deliberately shorter than a person's tolerance for a laggy keyboard.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Request

import languages
from kb.guard import RateLimiter

router = APIRouter(tags=["input"])

UPSTREAM = "https://inputtools.google.com/request"

# Short on purpose. Above roughly a quarter second the suggestion arrives after
# the next keystroke has already been typed, at which point it is not a
# suggestion but an interruption — better to show nothing and let the Latin
# stand.
TIMEOUT_S = float(os.getenv("TRANSLIT_TIMEOUT_S", "1.5"))

MAX_INPUT_CHARS = 64
MAX_CANDIDATES = 5

# Its own limiter, far above the /ask budget. This endpoint is called per word
# while typing, so the 20-per-minute budget that suits a metered model call
# would cut off a visitor mid-sentence. It still needs *a* limit: without one
# the service is an open transliteration proxy.
_limiter = RateLimiter(
    limit=int(os.getenv("TRANSLIT_RATE_LIMIT", "600")),
    window=60.0,
)

# Prefix lookups repeat heavily — "v", "va", "van", "vana"… — and the answers
# never change, so an LRU is the whole caching story. Bounded because the key
# space is attacker-controlled.
_CACHE_SIZE = int(os.getenv("TRANSLIT_CACHE", "4096"))
_cache: "OrderedDict[tuple[str, str], list[str]]" = OrderedDict()
_cache_lock = threading.Lock()


def _cached(key: tuple[str, str]) -> list[str] | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
        return hit


def _store(key: tuple[str, str], value: list[str]) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)


def _parse(payload: str) -> list[str]:
    """Pull the candidate list out of Google's positional JSON.

    The shape is ``["SUCCESS", [[input, [candidates], [], {...}]]]`` — positional
    rather than keyed, so every index here is a guess about a format nobody
    promised us. It is parsed defensively for that reason: an unexpected shape
    means no suggestions, never an exception on the typing path.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list) or len(data) < 2 or data[0] != "SUCCESS":
        return []
    try:
        candidates = data[1][0][1]
    except (IndexError, TypeError, KeyError):
        return []
    if not isinstance(candidates, list):
        return []
    return [c for c in candidates if isinstance(c, str) and c][:MAX_CANDIDATES]


@router.get("/translit")
def transliterate(text: str, lang: str, request: Request):
    """Native-script candidates for romanized `text`.

    Returns ``{"input", "lang", "candidates"}``. An empty `candidates` list is a
    normal, expected answer — it means "keep what you typed" — and is what every
    failure degrades to.
    """
    source = (text or "").strip()
    if not source:
        return {"input": "", "lang": lang, "candidates": []}
    if len(source) > MAX_INPUT_CHARS:
        # A transliteration request is one word. Anything longer is not a
        # keystroke, so it is refused rather than forwarded upstream.
        raise HTTPException(status_code=400, detail="`text` too long")

    if not languages.is_supported(lang):
        raise HTTPException(status_code=400, detail=f"unsupported language: {lang}")

    language = languages.get(lang)
    if not language.transliteration:
        # Malay and English are already written in the Latin alphabet, so there
        # is nothing to convert. Answering plainly beats a 400: the frontend
        # asks generically and this is the honest reply.
        return {"input": source, "lang": language.code, "candidates": []}

    key = (language.code, source.lower())
    hit = _cached(key)
    if hit is not None:
        return {"input": source, "lang": language.code, "candidates": hit, "cached": True}

    client = request.client.host if request.client else "unknown"
    if not _limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    import httpx

    try:
        response = httpx.get(
            UPSTREAM,
            params={
                "text": source,
                "itc": language.transliteration,
                "num": MAX_CANDIDATES,
                "cp": 0,
                "cs": 1,
                "ie": "utf-8",
                "oe": "utf-8",
            },
            timeout=TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - a keyboard must not fail loudly
        print(f"[translit] upstream failed: {type(exc).__name__}: {exc}")
        return {"input": source, "lang": language.code, "candidates": []}

    if response.status_code != 200:
        print(f"[translit] upstream {response.status_code}")
        return {"input": source, "lang": language.code, "candidates": []}

    candidates = _parse(response.text)
    # Negative results are cached too. A word with no transliteration is asked
    # about on every subsequent keystroke otherwise, which is precisely the
    # traffic pattern the cache exists to stop.
    _store(key, candidates)
    return {"input": source, "lang": language.code, "candidates": candidates}
