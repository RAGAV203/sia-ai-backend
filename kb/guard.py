"""Request-side and response-side guards for the answering pipeline.

Layered defense, because no single layer is reliable:

* **Ingest** (``sanitize.py``) removes invisible text and quarantines
  injection-shaped pages before they ever reach the index.
* **Prompt** (``prompts.py``) frames retrieved content as untrusted data, so a
  chunk that slips through is read as *quotation*, not *instruction*.
* **Here** — bound what a caller can spend, refuse prompt-extraction attempts,
  and check the answer before it is spoken.

The structural defense underneath all of it: the model is given **no tools**.
It can emit text and nothing else, so the worst outcome of a successful
injection is a wrong sentence, never an action.
"""

from __future__ import annotations

import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field

MAX_QUESTION_CHARS = 400      # a spoken question is short; also caps cost + TTS work
MIN_QUESTION_CHARS = 2
MAX_ANSWER_CHARS = 700        # ~40s of speech; see prompts.py for why this is capped

# Per-client budget. The expensive resources are the model call and — far more
# so — TTS synthesis, which is CPU-bound and roughly real-time.
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_S = 60.0


@dataclass
class Verdict:
    ok: bool
    reason: str = ""
    value: str = ""


# --- input --------------------------------------------------------------------

# Attempts to extract or override the operator prompt. Unlike corpus injection,
# a user typing this can only mislead themselves — but honoring it would leak
# the prompt, so these are refused rather than sanitized.
_EXTRACTION = [
    re.compile(p, re.I)
    for p in (
        r"(ignore|disregard|forget)\b.{0,32}\b(previous|prior|above|earlier|initial|original|system|all)\b",
        r"(reveal|repeat|print|show|output|display|recite|translate)\b.{0,32}\b(system|initial|original)\b.{0,16}\b(prompt|instruction|message|rule)",
        r"what (are|were) your\b.{0,24}\b(instruction|rule|prompt|guideline)s?\b",
        r"you are (now|no longer|actually)\b",
        r"\b(dan|developer|jailbreak|god)\s*mode\b",
        r"</?(system|assistant|human)>",
        r"\[\s*(system|inst|/inst)\s*\]",
        r"repeat (everything|the text|all text) above",
        r"(begin|start) (your )?(reply|response|answer) with\b",
    )
]


def check_question(raw: str) -> Verdict:
    """Normalize and vet an inbound question."""
    if not raw or not raw.strip():
        return Verdict(False, "empty")

    # NFKC first: otherwise fullwidth/math-styled glyphs slip past the patterns.
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(c for c in text if c == " " or not unicodedata.category(c).startswith("C"))
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < MIN_QUESTION_CHARS:
        return Verdict(False, "too_short")
    if len(text) > MAX_QUESTION_CHARS:
        # Truncate rather than reject: a rambling spoken question is normal, and
        # the leading clause is almost always the real question.
        text = text[:MAX_QUESTION_CHARS].rsplit(" ", 1)[0]

    for pattern in _EXTRACTION:
        if pattern.search(text):
            return Verdict(False, "prompt_extraction", text)

    return Verdict(True, value=text)


class RateLimiter:
    """Fixed-window per-client counter. In-process by design — this service runs
    as a single instance; a shared store would only matter behind replicas."""

    def __init__(self, limit: int = RATE_LIMIT_REQUESTS, window: float = RATE_LIMIT_WINDOW_S):
        self.limit = limit
        self.window = window
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits.get(client, []) if now - t < self.window]
            if len(hits) >= self.limit:
                self._hits[client] = hits
                return False
            hits.append(now)
            self._hits[client] = hits
            # Opportunistic sweep so idle clients don't accumulate forever.
            if len(self._hits) > 4096:
                self._hits = {
                    c: [t for t in ts if now - t < self.window]
                    for c, ts in self._hits.items()
                    if any(now - t < self.window for t in ts)
                }
            return True


# --- output -------------------------------------------------------------------

# Distinctive strings from the operator prompt. If any surfaces in an answer,
# the model is echoing its instructions and the response must not be spoken.
_LEAK_MARKERS = [
    re.compile(p, re.I)
    for p in (
        # `\b` not `>` — the real tags carry attributes (`<source id="1" ...>`).
        r"</?(source|reference_material|question)\b",
        r"untrusted (reference )?(data|material|content)",
        r"you are sia\b",
        r"reference material (below|above)",
        r"never follow instructions",
        r"system prompt",
    )
]

# "I could not answer this", detected as a pattern family rather than a list of
# exact phrasings.
#
# This matters beyond reporting: an answer flagged here is not `grounded`, and
# only grounded answers are cached — so a missed refusal pins "I don't know" to
# a question that might well succeed later, once the corpus is re-scraped.
#
# Enumerating strings failed one wording at a time. "does not contain" was
# covered while "does not provide" and the plural "sources do not contain" were
# not, and each gap silently cached a refusal. Matching the verb family closes
# the class. The prompt also asks the model for a canonical refusal sentence
# (`prompts.NO_ANSWER_SENTENCE`), which `_is_refusal` checks first — these
# patterns only catch the cases where it improvises anyway.
#
# Three wordings reached the cache before the verb lists below were widened, and
# each is now pinned to a question the corpus can answer:
#
#   "BBA fees structure is not detailed in the reference material"
#        -> "detailed" was not in the verb list
#   "I'm not able to provide information about joining the college"
#        -> the auxiliary list had do/does/is/are but not am/'m
#   "I can't assist with that"
#        -> the modal list covered find/locate/answer/help but not assist
#
# The pattern is always the same: a verb family enumerated by example, closed
# one miss at a time. Prefer widening a family over appending a phrase.
_REFUSAL_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\b(do|does|did|is|are|was|were|am|'m|be|been)(n't|\s+not)\s+(\w+\s+){0,3}"
        r"(contain|provide|mention|include|specify|say|state|list|cover|available|"
        r"detail(ed)?|describ(e|ed)|discuss(ed)?|given|give|found|find|offer(ed)?|"
        r"able|clear|sure|certain)",
        r"\bi\s+(do\s+not|don't|cannot|can't|couldn't|could not)\s+"
        r"(have|know|see|find|access)\b",
        r"\bno\s+(information|mention|details?|reference|data|record)\b",
        r"\bnot\s+(mentioned|provided|specified|stated|listed|available|included|"
        r"detailed|described|discussed|given|found|clear|sure)\b",
        r"\b(cannot|can't|can not|unable to|not able to)\s+"
        r"(find|locate|answer|help|assist|provide|share|tell|confirm|comment)\b",
        r"\boutside\s+what\s+i\s+can\s+help\b",
        r"\bnot\s+in\s+(my|the)\s+(reference|source|material)",
    )
]
# Note: mentioning the reference material is deliberately *not* a refusal on its
# own. "The reference material mentions that the campus has a retro shop" is a
# correct answer wearing a bad preamble, and treating it as a refusal throws away
# a fact the corpus really did supply. It is handled as machinery below, where
# the framing is stripped and the fact survives.

# Prompt-machinery vocabulary that must never be *spoken*, as opposed to the
# leak markers above which mean the model is reciting its instructions.
#
# "The reference material mentions that the campus has a variety of businesses"
# is grounded and correct, and still wrong to say out loud: the listener has no
# reference material, so the phrase is meaningless at best and sounds evasive at
# worst. Rather than fail closed on an otherwise good answer, these are rewritten
# — and when the sentence was a refusal, replaced with the canonical one so it
# is detected, cached-blocked, and shares a TTS clip.
_MACHINERY = re.compile(
    r"\b(the\s+)?(reference\s+material|provided\s+(text|material|context)|"
    r"(the\s+)?sources?\s+(provided|given|above)|"
    r"(the\s+)?(given|supplied|above)\s+(text|material|context|sources?))\b",
    re.I,
)

# An answer written in the voice of a student rather than the college.
#
# The corpus is full of placement and alumni testimonials, which are first-person
# by nature. Quoted back as an answer they make Sia claim to be a graduate:
# "Thank you for your placement opportunity at M.Phasis Limited. I am grateful to
# the management..." — fluent, genuinely grounded, and completely wrong. The
# intent gate stops the common trigger ("thanks"), but a real question can still
# retrieve a testimonial chunk, so the voice is checked here as well.
#
# Deliberately narrow. Sia says "I" legitimately all the time ("I do not have
# that detail"), so only gratitude and personal-experience phrasing counts.
_TESTIMONIAL_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bi\s+(am|'m|was|feel)\s+(so\s+|very\s+|really\s+|truly\s+|extremely\s+)?"
        r"(grateful|thankful|blessed|honou?red|privileged|proud)\b",
        r"\bi\s+(would\s+like\s+to|want\s+to|wish\s+to)\s+thank\b",
        r"\bthank(s|\s+you)?\s+(to\s+)?(the\s+)?(management|placement|college|"
        r"institution|principal|faculty|department)\b.{0,40}\bfor\s+(the\s+|this\s+|your\s+)",
        r"\bmy\s+(sincere\s+)?(thanks|gratitude|heartfelt)\b",
        r"\bi\s+(will|shall)\s+(use\s+this\s+opportunity|make\s+use\s+of)\b",
    )
]


@dataclass
class AnswerCheck:
    ok: bool
    text: str
    flags: list[str] = field(default_factory=list)


def check_answer(text: str) -> AnswerCheck:
    """Vet a generated answer before it is cached, spoken, or returned."""
    flags: list[str] = []
    clean = re.sub(r"\s+", " ", (text or "")).strip()

    if not clean:
        return AnswerCheck(False, "", ["empty"])

    for pattern in _LEAK_MARKERS:
        if pattern.search(clean):
            # Fail closed: a leaked prompt is worse than a generic fallback.
            return AnswerCheck(False, "", ["prompt_leak"])

    # A testimonial read back as an answer is not repairable — the whole
    # sentence is in the wrong voice — so it is rejected outright and the
    # caller speaks the fallback instead.
    if any(p.search(clean) for p in _TESTIMONIAL_PATTERNS):
        return AnswerCheck(False, "", ["testimonial"])

    if len(clean) > MAX_ANSWER_CHARS:
        cut = clean[:MAX_ANSWER_CHARS]
        # Prefer a sentence boundary so the spoken answer doesn't stop mid-word.
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        clean = (cut[: end + 1] if end > MAX_ANSWER_CHARS * 0.5 else cut).strip()
        flags.append("truncated")

    from .prompts import NO_ANSWER_SENTENCE

    refused = _is_refusal(clean)

    # Talking about the sources is never speakable — the listener has no
    # reference material, so the phrase is meaningless out loud. A *grounded*
    # answer wearing that framing is repaired rather than discarded; the fact it
    # introduces is the answer.
    if _MACHINERY.search(clean):
        flags.append("machinery")
        if not refused:
            clean = _strip_machinery(clean)
            if not clean:
                refused = True

    if refused:
        # Every refusal is spoken with the same words, whatever the model
        # improvised. Three reasons, in order of weight:
        #   - Consistency of voice. "I'm sorry, but I can't assist with that"
        #     reads as a policy refusal; this is only ever a gap in the corpus,
        #     and sounding evasive about it is worse than sounding limited.
        #   - One TTS clip. The canonical sentence is prewarmed at boot, so a
        #     refusal is a disk read instead of ~2s of synthesis.
        #   - Exact matching downstream: `vet()` and the cache both key off it.
        flags.append("no_answer")
        clean = NO_ANSWER_SENTENCE

    return AnswerCheck(True, clean, flags)


def _strip_machinery(text: str) -> str:
    """Remove "according to the reference material"-style framing from an answer.

    Only the framing goes; the fact it introduces is the answer. "The reference
    material mentions that the campus has a retro shop" becomes "The campus has
    a retro shop."
    """
    out = re.sub(
        r"(?i)^(according to|based on|as (stated|mentioned|described) in|from)\s+"
        r"[^,]{0,48}?(reference material|provided (text|material|context)|sources?)\s*,?\s*",
        "",
        text,
    )
    out = re.sub(
        r"(?i)\b(the\s+)?(reference material|provided (text|material|context)|sources?\s+(provided|given|above))\s+"
        r"(mentions?|states?|says?|indicates?|shows?|notes?|suggests?)\s+that\s+",
        "",
        out,
    )
    # Any surviving mention means the sentence was *about* the sources, not a
    # fact with a preamble; there is nothing left worth speaking.
    if _MACHINERY.search(out):
        return ""
    out = out.strip()
    return (out[0].upper() + out[1:]) if out else ""


def _is_refusal(text: str) -> bool:
    """Did the model decline to answer?

    Exact match on the canonical sentence first — that is what the prompt asks
    for and it cannot be missed. The phrase list is only a backstop for when the
    model improvises, which it still sometimes does.
    """
    from .prompts import NO_ANSWER_SENTENCE

    low = text.lower().strip()
    if NO_ANSWER_SENTENCE.lower().rstrip(".") in low:
        return True
    return any(p.search(low) for p in _REFUSAL_PATTERNS)
