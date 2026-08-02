"""Conversational turns that must never reach retrieval.

A voice assistant is talked *to*, not just queried. Roughly one turn in six in
the observed traffic is social — "thanks", "how are you", "what can you do" —
and running those through the RAG pipeline is not merely wasteful, it is the
single worst failure mode this service has:

    "Thanks."      -> read the Principal's welcome message aloud
    "Thank you."   -> "Thank you for your placement opportunity at M.Phasis
                      Limited. I am grateful to the management, placement
                      officers..."

The second one is a student testimonial, quoted verbatim in the first person, so
Sia introduces herself as a placed graduate. Nothing downstream can catch that:
the sentence is fluent, it is genuinely grounded in a real chunk, and the guard
sees a confident answer with no refusal markers. It gets cached and spoken.

The fix belongs here, before retrieval, because the corpus can never answer
"thanks" — there is no right chunk to find. Matching is deliberately narrow:

* the **whole** utterance must be social, never a substring. "Thanks, now tell
  me about placements" is a real question with a courtesy attached, and it goes
  to the knowledge base.
* only short utterances qualify, so a long sentence that happens to open with
  "hi" is not swallowed.

Replies are fixed strings, which means they share TTS cache entries and are
prewarmed at startup — a social turn costs one disk read and no generation.
"""

from __future__ import annotations

import re

# Answers are written to be spoken: short, no lists, no URLs. They also steer
# the conversation back to something the corpus can actually answer, because a
# social turn is usually someone deciding what to ask next.
GREETING_REPLY = (
    "Hello. I am Sia, the assistant for Shasun Jain College for Women. "
    "Ask me about our programmes, campus life, admissions, or placements."
)
THANKS_REPLY = "You are very welcome. Is there anything else you would like to know about the college?"
FAREWELL_REPLY = "Goodbye, and thank you for visiting. Do come back if you have more questions about the college."
IDENTITY_REPLY = (
    "I am Sia, the voice assistant for Shri Shankarlal Sundarbai Shasun Jain College for Women in Chennai. "
    "I answer questions about the college using information from its official website."
)
CAPABILITY_REPLY = (
    "I can tell you about our academic programmes, admissions and eligibility, campus facilities, "
    "and placements. Ask me anything about the college."
)
WELLBEING_REPLY = (
    "I am doing well, thank you for asking. What would you like to know about Shasun Jain College?"
)
ACK_REPLY = "Of course. What else can I tell you about the college?"

# One pattern per intent, anchored end to end. `\W*` tails absorb the trailing
# punctuation and filler that speech-to-text leaves behind ("hi!", "ok then").
_INTENTS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "thanks",
        re.compile(r"^(ok(ay)?\W+)?(thanks?|thank you|thanks a lot|thank u|thx|ty)\W*$", re.I),
        THANKS_REPLY,
    ),
    (
        "farewell",
        re.compile(r"^(ok(ay)?\W+)?(bye|goodbye|good bye|see you|see ya|cya|that('s| is) all)\W*$", re.I),
        FAREWELL_REPLY,
    ),
    (
        "greeting",
        re.compile(
            r"^(hi|hey|hello|hii+|halo|namaste|vanakkam|good\s+(morning|afternoon|evening))"
            r"(\s+(there|sia))?\W*$",
            re.I,
        ),
        GREETING_REPLY,
    ),
    (
        "wellbeing",
        re.compile(r"^how\s+(are|r)\s+(you|u)(\s+doing|\s+today)?\W*$", re.I),
        WELLBEING_REPLY,
    ),
    (
        "identity",
        re.compile(
            r"^(who|what)\s+(are|r)\s+(you|u)(\s+again)?\W*$|^(what('s| is)\s+your\s+name)\W*$"
            r"|^(are\s+you\s+(a\s+)?(bot|robot|human|real|ai|a\.i\.))\W*$",
            re.I,
        ),
        IDENTITY_REPLY,
    ),
    (
        "capability",
        re.compile(
            r"^(what|how)\s+(can|could|do)\s+(you|u)\s+(do|help( me)?( with)?|assist)"
            r"(\s+for\s+me)?\W*$|^(what\s+(can|do)\s+(you|u)\s+know)\W*$"
            r"|^(help|help me)\W*$",
            re.I,
        ),
        CAPABILITY_REPLY,
    ),
    (
        "acknowledgement",
        re.compile(r"^(ok(ay)?|yes|yeah|yep|sure|alright|got it|fine|cool|nice|great)\W*$", re.I),
        ACK_REPLY,
    ),
]

# Above this, treat the turn as a real question even if it starts socially.
# Every canned reply is a fixed string, so a false positive here would answer a
# genuine question with a pleasantry — much worse than a wasted retrieval.
MAX_WORDS = 6


def match(question: str) -> tuple[str, str] | None:
    """``(intent, reply)`` for a purely social turn, or ``None`` to answer normally."""
    text = (question or "").strip()
    if not text or len(text.split()) > MAX_WORDS:
        return None
    for name, pattern, reply in _INTENTS:
        if pattern.match(text):
            return name, reply
    return None


def replies() -> list[str]:
    """Every canned reply, for TTS prewarming."""
    return [reply for _, _, reply in _INTENTS]
