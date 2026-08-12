"""Translation, so that everything else in this service can stay English.

**The pivot design, and why it is not the obvious one.**

The obvious way to answer a Tamil question is to answer it in Tamil: hand the
model the Tamil question and the English sources and ask for a Tamil reply. It
is one API call instead of two, and it is wrong here, because of what lives
downstream of the model:

* ``kb/guard.py`` decides whether an answer is a refusal by matching English
  verb families ("does not contain", "I cannot find"). A Tamil refusal matches
  none of them, so it is marked ``grounded`` and **cached** — pinning "I don't
  know" to a question the corpus can answer, permanently and invisibly.
* ``kb/answer_cache.py`` compares questions with a local MiniLM encoder that was
  trained on English. Tamil embeddings from it are not merely worse, they are
  meaningless, so the cache would both miss real paraphrases and occasionally
  fire on unrelated ones.
* ``kb/triage.py`` reads the question as a *request* using English imperative
  patterns. In Tamil it sees nothing and every out-of-scope question reaches the
  model.

So the language boundary is drawn at the very edge instead. A non-English
question is translated to English on the way in, the entire existing pipeline
runs on English exactly as it always has, and the guarded English answer is
translated on the way out. Every calibrated threshold, pattern list and cache
keeps its meaning, and adding a fifth language later touches this file and
``languages.py`` rather than every guard in the tree.

**Why the cache matters more than the latency it saves.** Answer text is the TTS
cache key (see ``services/tts.py``). If the same English answer translated to
Tamil twice produced two different Tamil strings — and at temperature 0 it
usually would not, but "usually" is not a guarantee across model revisions —
every clip would be a synthesis miss and the avatar would pause for seconds.
Caching on the exact English input makes the Tamil string a pure function of it,
which is what keeps the prewarmed clips reachable.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

from config import APP_DIR, TRANSLATION_CACHE_DIR
from kb import debug, gemini
from languages import DEFAULT_LANG, get as get_language

# Bump when a prompt change alters the text a given input produces, so stale
# translations cannot be served from an existing cache — and, more importantly,
# so the TTS clips keyed off them are regenerated too.
#
# 2 = sentence punctuation required. Version 1 produced Tamil and Hindi with the
# full stops stripped, which is invisible on screen and severe in the ear: the
# splitter in speech.py had nothing to split on, so a four-sentence answer was
# synthesized as one 300-character clip and the listener waited through all of
# it before hearing a word.
TRANSLATION_VERSION = 2

# Translation is a formatting-sensitive task, not a creative one: the output is
# going to be *spoken aloud* by a phonemizer that expects one script, and shown
# in a chat bubble. Both of those constrain it more than fidelity alone does.
_SYSTEM = """\
You are a translator for the voice assistant of an Indian women's college. \
You translate short spoken replies and questions between English and {name}.

Output ONLY the translation. No quotes, no notes, no alternatives, no \
transliteration in brackets, no explanation of choices.

Rules:
- Translate into natural, spoken {name} as a helpful receptionist would say it \
aloud. Not formal written register, and not word-for-word.
- Write the ENTIRE output in the {script} of {name}, including proper nouns and \
place names. The result is fed to a speech synthesizer configured for {name}, \
and text left in another alphabet is mispronounced.
- PUNCTUATE FULLY. End every sentence with {terminator}, and keep the commas. \
Match the source's sentence count: if it is three sentences, write three \
sentences. The output is split on sentence punctuation before it is spoken, so \
text without it is synthesized as one long unbroken utterance.
- Keep degree and qualification names recognizable when you render them, so \
B.Com, B.Sc, B.B.A and M.A. stay identifiable as those qualifications.
- Write numbers as words the way they are said aloud, never as digits.
- No markdown, no bullet points, no headings, no URLs, no emoji.
- Preserve the meaning exactly. Never add a fact, a number, a date, a fee or a \
contact detail that is not in the source, and never drop one that is.
- If the source is already in {name}, return it unchanged.
"""

# The character that ends a sentence in each script. Hindi's danda is the one
# that actually differs; naming it explicitly is what stops the model ending
# Devanagari sentences with a Latin period some of the time and nothing at all
# the rest, which is what it did when the prompt said only "punctuate".
_TERMINATORS = {
    "ta": "a full stop (.)",
    "hi": "a danda (।)",
    "ms": "a full stop (.)",
    "en": "a full stop (.)",
}


def _system_for(language) -> str:
    return _SYSTEM.format(
        name=language.english_name,
        script=_SCRIPTS.get(language.script, "native script"),
        terminator=_TERMINATORS.get(language.code, "a full stop (.)"),
    )


_TO_ENGLISH = """\
You translate a visitor's spoken question into English for a search system.

Output ONLY the English translation. No quotes, no notes, no explanation.

Rules:
- Translate the question faithfully, keeping it a question.
- Render Indian proper nouns in their standard English spelling: the college is \
"Shasun Jain College", or "Shri Shankarlal Sundarbai Shasun Jain College for \
Women". Place names use their usual English forms, so Chennai and T. Nagar.
- Keep degree names as B.Com, B.Sc, B.B.A, M.A., M.Com, M.Sc.
- Do not answer the question. Do not add context it did not contain.
- If the input is already English, return it unchanged.
"""

# Scripts named explicitly, because "write it in Tamil" is ambiguous to a model
# handed romanized input — it will sometimes answer in romanized Tamil, which
# the Tamil phonemizer then reads as if it were English.
_SCRIPTS = {
    "tamil": "Tamil script",
    "devanagari": "Devanagari script",
    "latin": "Latin alphabet",
}

_memory: dict[str, str] = {}
_memory_lock = threading.Lock()

# Translations are small and read far more often than written, so they get the
# same treatment as TTS clips: one file per input, written atomically. A single
# growing JSONL would need a lock on every read and a load at boot; this needs
# neither, and a half-written file is impossible rather than merely unlikely.
_CACHE_DIR = Path(TRANSLATION_CACHE_DIR)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Translation is on the latency path for every non-English answer that is not
# already cached, so it gets a tighter retry budget than an index rebuild.
_RETRIES = int(os.getenv("TRANSLATE_RETRIES", "2"))


def available() -> bool:
    return gemini.available()


def _key(direction: str, lang: str, text: str) -> str:
    return hashlib.sha256(
        f"v{TRANSLATION_VERSION}|{direction}|{lang}|{text}".encode("utf-8")
    ).hexdigest()


def _cached(key: str) -> str | None:
    with _memory_lock:
        hit = _memory.get(key)
    if hit is not None:
        return hit
    path = _CACHE_DIR / f"{key}.txt"
    if not path.exists():
        return None
    try:
        value = path.read_text(encoding="utf-8")
    except OSError:
        return None
    with _memory_lock:
        _memory[key] = value
    return value


def _store(key: str, value: str) -> None:
    with _memory_lock:
        _memory[key] = value
    path = _CACHE_DIR / f"{key}.txt"
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.part")
    try:
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[translate] cache write failed: {exc}")


def _clean(text: str) -> str:
    """Strip the framing an instructed model still occasionally adds."""
    out = " ".join((text or "").split())
    # A model asked for "only the translation" sometimes still wraps it, and a
    # stray quote is audible: the phonemizer reads it as a pause.
    if len(out) > 1 and out[0] in "\"'“‘" and out[-1] in "\"'”’":
        out = out[1:-1].strip()
    return out


def _call(system: str, text: str, what: str) -> str | None:
    try:
        # Roughly 3x the input in tokens: Indic scripts tokenize far less
        # efficiently than English, and a translation truncated mid-sentence is
        # worse than no translation at all — it would be spoken, and cached.
        budget = max(256, min(1200, len(text) * 3))
        out = gemini.generate(system, text, max_tokens=budget)
    except Exception as exc:  # noqa: BLE001 - never take the voice path down
        print(f"[translate] {what} failed: {type(exc).__name__}: {exc}")
        return None
    return _clean(out) or None


def to_language(text: str, lang: str) -> str:
    """Translate an English string into `lang`, for display and for speech.

    Returns the English unchanged on any failure. That is the right degradation
    for this service: an English answer spoken to a Tamil visitor is a poor
    experience, while a raised exception is a silent avatar.
    """
    language = get_language(lang)
    source = (text or "").strip()
    if not source or language.code == DEFAULT_LANG:
        return source

    key = _key("out", language.code, source)
    hit = _cached(key)
    if hit is not None:
        if debug.ENABLED:
            debug.log(f"[translate] cache HIT  en->{language.code}  {source[:50]!r}")
        return hit

    if not available():
        return source

    system = _system_for(language)
    out = _call(system, source, f"en->{language.code}")
    if not out:
        return source

    if debug.ENABLED:
        debug.log(f"[translate] en->{language.code}  {source[:40]!r} -> {out[:40]!r}")
    _store(key, out)
    return out


def to_english(text: str, lang: str) -> str:
    """Translate a visitor's question into English for the answering pipeline.

    Returns the input unchanged on failure, which lets the pipeline try its luck
    — `gemini-embedding-001` is multilingual, so cross-lingual retrieval often
    works even without this — rather than refusing outright.
    """
    language = get_language(lang)
    source = (text or "").strip()
    if not source or language.code == DEFAULT_LANG:
        return source

    key = _key("in", language.code, source)
    hit = _cached(key)
    if hit is not None:
        if debug.ENABLED:
            debug.log(f"[translate] cache HIT  {language.code}->en  {source[:50]!r}")
        return hit

    if not available():
        return source

    out = _call(_TO_ENGLISH, source, f"{language.code}->en")
    if not out:
        return source

    if debug.ENABLED:
        debug.log(f"[translate] {language.code}->en  {source[:40]!r} -> {out[:40]!r}")
    _store(key, out)
    return out


def to_language_many(texts: list[str], lang: str) -> list[str]:
    """Translate several English strings into `lang`, in one call where possible.

    Only used by the boot prewarm, which has a few dozen fixed strings to render
    across three languages. One call per language instead of ~15 matters there
    for a reason beyond speed: the free tier is metered per request, and a cold
    start that spends 45 of them competes with the visitor who arrives during it.

    Falls back to translating one at a time if the batch comes back the wrong
    shape — a dropped or reordered item would silently pair an answer with the
    wrong question, which is far worse than a slow boot.
    """
    language = get_language(lang)
    if language.code == DEFAULT_LANG:
        return [(t or "").strip() for t in texts]

    pending: list[tuple[int, str]] = []
    out: list[str] = []
    for i, text in enumerate(texts):
        source = (text or "").strip()
        out.append(source)
        if not source:
            continue
        hit = _cached(_key("out", language.code, source))
        if hit is not None:
            out[i] = hit
        else:
            pending.append((i, source))

    if not pending or not available():
        return out

    system = _system_for(language)
    schema = {
        "type": "OBJECT",
        "properties": {
            "translations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "text": {"type": "STRING"},
                    },
                    "required": ["id", "text"],
                },
            }
        },
        "required": ["translations"],
    }
    numbered = "\n\n".join(f"[{n}] {source}" for n, (_, source) in enumerate(pending))
    user = (
        "Translate each numbered item below. Return one object per item, with "
        "`id` set to the item's number and `text` set to its translation. "
        "Return every item exactly once.\n\n" + numbered
    )

    try:
        parsed = gemini.generate_json(
            system, user, schema, max_tokens=max(1024, len(numbered) * 3)
        )
        rows = parsed.get("translations") or []
        by_id = {int(r["id"]): _clean(str(r.get("text", ""))) for r in rows if "id" in r}
    except Exception as exc:  # noqa: BLE001 - fall back to one at a time
        print(f"[translate] batch en->{language.code} failed: {type(exc).__name__}: {exc}")
        by_id = {}

    for n, (index, source) in enumerate(pending):
        value = by_id.get(n)
        if not value:
            # The batch dropped this one. One-at-a-time is the safe repair, and
            # it is also what populates the cache correctly for next boot.
            value = to_language(source, language.code)
        else:
            _store(_key("out", language.code, source), value)
        out[index] = value or source

    return out


def status() -> dict:
    """What the translation layer can do right now, for ``/health``."""
    try:
        cached = sum(1 for _ in _CACHE_DIR.glob("*.txt"))
    except OSError:
        cached = 0
    return {
        "ready": available(),
        "cached": cached,
        "dir": str(_CACHE_DIR.relative_to(APP_DIR)) if _CACHE_DIR.is_relative_to(APP_DIR) else str(_CACHE_DIR),
    }
