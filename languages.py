"""The four languages SIA speaks, and what each subsystem needs to know about them.

One registry rather than four scattered lookups, because a language is not a
single string: espeak wants ``ta``, Whisper wants ``ta``, Google Input Tools
wants ``ta-t-i0-und``, the browser's SpeechRecognition wants ``ta-IN``, and the
translator wants "Tamil". Those drifted apart the moment they were declared next
to the code that read them, and a mismatch is silent — espeak falls back to
English and synthesizes Tamil text as gibberish rather than raising.

**English is the pivot, not merely the default.** Every other language is
translated to English on the way in and back out again (``services/translate``),
so retrieval, the guards, the scope triage and the answer cache all keep
operating on the English they were calibrated against. See ``routers/chat`` for
why that ordering matters.

**Why Kokoro covers all four.** ``hf_alpha`` is a Hindi female speaker, and
Kokoro's phoneme vocabulary is a 114-symbol IPA set shared across every language
it was trained on. espeak-ng phonemizes Tamil and Malay into symbols that all
happen to fall inside that set, so they synthesize in the same voice with the
same cache and no second engine. Measured coverage on a sample sentence per
language: en 100%, hi 100%, ta 100%, ms 100%.

That is a real result but not a free lunch: Kokoro was *trained* on Hindi and
English, not on Tamil or Malay. Those two come out intelligible and in the right
voice, but carrying an Indian-English accent and losing some vowel-length
distinctions. If that ever becomes the limiting factor, the fix is a second
synthesis backend for ``ta``/``ms`` behind ``services.tts.cached_wav`` — nothing
above that function needs to know.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import TTS_LANG


@dataclass(frozen=True)
class Language:
    code: str
    """Our canonical two-letter code — the value the API speaks."""

    english_name: str
    """How the translator is told to name it. Must be unambiguous to a model."""

    native_name: str
    """How it is offered to the visitor, in its own script."""

    espeak: str
    """Phonemizer language for Kokoro. NOT the same as `code` for English."""

    whisper: str
    """faster-whisper language code (the STT fallback path)."""

    speech_recognition: str
    """BCP-47 tag for the browser's own SpeechRecognition/speechSynthesis, used
    only when the voice service is unreachable and the frontend degrades to
    Web Speech."""

    transliteration: str | None
    """Google Input Tools `itc` code, or None for a language already written in
    the Latin alphabet. This is what lets someone type "vanakkam" on a QWERTY
    keyboard and get வணக்கம்."""

    script: str
    """Writing system, so the UI can pick a font stack that actually has the
    glyphs. Latin-script languages need no extra webfont."""

    script_sample: str
    """A short phrase in this language's own script, shown to the transcription
    model as an example of what its output must look like.

    This is load-bearing, not decorative. Told only to "write the transcript in
    the native script of Tamil", Gemini returns romanized Tamil — measured 0% of
    letters in Tamil script across every test clip. Shown a concrete sample and
    told romanization is wrong, the same model returns 100%. See
    ``services/stt._prompt_for``."""


LANGUAGES: dict[str, Language] = {
    "en": Language(
        code="en",
        english_name="English",
        native_name="English",
        # en-us by default, not en-in: the accent comes from the *speaker
        # embedding* (hf_alpha is Hindi), while the phonemizer decides
        # pronunciation. An Indian English phoneme set on top of an Indian voice
        # overshoots. Overridable via TTS_LANG, which predates this registry.
        espeak=TTS_LANG,
        whisper="en",
        speech_recognition="en-IN",
        transliteration=None,
        script="latin",
        script_sample="How do I apply for admission?",
    ),
    "ta": Language(
        code="ta",
        english_name="Tamil",
        native_name="தமிழ்",
        espeak="ta",
        whisper="ta",
        speech_recognition="ta-IN",
        transliteration="ta-t-i0-und",
        script="tamil",
        script_sample="கல்லூரியில் சேர்க்கை எப்படி?",
    ),
    "hi": Language(
        code="hi",
        english_name="Hindi",
        native_name="हिन्दी",
        espeak="hi",
        whisper="hi",
        speech_recognition="hi-IN",
        transliteration="hi-t-i0-und",
        script="devanagari",
        script_sample="कॉलेज में एडमिशन कैसे लें?",
    ),
    "ms": Language(
        code="ms",
        english_name="Malay",
        native_name="Bahasa Melayu",
        espeak="ms",
        whisper="ms",
        speech_recognition="ms-MY",
        transliteration=None,  # already Latin script
        script="latin",
        script_sample="Bagaimana cara memohon kemasukan?",
    ),
}

DEFAULT_LANG = "en"

# Every espeak code the registry can produce, so a legacy caller passing
# `lang="en-us"` straight through (the pre-multilingual API shape) is still
# understood rather than silently normalized to English.
_ESPEAK_CODES = {lang.espeak: lang.code for lang in LANGUAGES.values()}


def get(code: str | None) -> Language:
    """The Language for a code, falling back to English rather than raising.

    Deliberately forgiving. This sits on the request path for every endpoint,
    and a bad `lang` should cost the visitor an English answer, not a 400 — the
    frontend already degrades in several places, and an exception here would
    turn a cosmetic mismatch into a dead avatar.
    """
    return LANGUAGES.get(normalize(code), LANGUAGES[DEFAULT_LANG])


def normalize(code: str | None) -> str:
    """Canonicalize whatever the client sent into one of our four codes.

    Accepts our own codes, BCP-47 tags the browser might volunteer (``ta-IN``),
    and the raw espeak codes the ``/tts`` endpoint took before this module
    existed (``en-us``) — that last one is why this is not simply a dict lookup
    on the first two characters.
    """
    if not code:
        return DEFAULT_LANG
    raw = code.strip().lower().replace("_", "-")
    if raw in LANGUAGES:
        return raw
    if raw in _ESPEAK_CODES:
        return _ESPEAK_CODES[raw]
    base = raw.split("-", 1)[0]
    return base if base in LANGUAGES else DEFAULT_LANG


def is_supported(code: str | None) -> bool:
    """Whether this code names a language we actually speak, before normalizing.

    Separate from `normalize` because "unsupported" and "English" must be
    distinguishable at the edges — `/translit` needs to refuse an unknown
    language rather than silently transliterate into English, which would return
    the input unchanged and look like a broken keyboard.
    """
    if not code:
        return False
    raw = code.strip().lower().replace("_", "-")
    return raw in LANGUAGES or raw in _ESPEAK_CODES or raw.split("-", 1)[0] in LANGUAGES


def codes() -> list[str]:
    """Supported language codes, English first."""
    return [DEFAULT_LANG] + [c for c in LANGUAGES if c != DEFAULT_LANG]


def public_catalog() -> list[dict]:
    """What the frontend needs to render a language picker and degrade well.

    `speech_recognition` is included because the browser fallback path lives
    entirely client-side: when the voice service is unreachable there is nobody
    to ask for the right BCP-47 tag.
    """
    return [
        {
            "code": lang.code,
            "label": lang.english_name,
            "native": lang.native_name,
            "script": lang.script,
            "speechRecognition": lang.speech_recognition,
            "transliteration": bool(lang.transliteration),
        }
        for lang in (LANGUAGES[c] for c in codes())
    ]
