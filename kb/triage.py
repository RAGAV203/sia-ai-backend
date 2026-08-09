"""Cheap, local triage of what a question is *asking for*, before any retrieval.

This exists because of one concrete failure. Asked to "write a python program",
the assistant retrieved the college's "Digital Marketing Proficiency Program"
and "Student Induction Program" pages — a 0.597 similarity, high enough to look
like a real match — and answered about academic programmes. The word "program"
did all the work; nothing in the pipeline ever asked whether *writing code* is
something this assistant does.

Retrieval cannot catch this on its own, and neither can a similarity threshold.
Both operate on topical overlap, and "write a python program" genuinely does
overlap with a page containing the word "Program". The missing step is reading
the question as a **request** rather than as a bag of terms: an imperative to
produce code is out of scope no matter how well its vocabulary matches.

Two layers do that, and this is the cheap one:

* **Here** — a narrow pattern list for the unambiguous cases, costing nothing and
  no latency. It must have *zero* false positives on real college questions, so
  it only fires on constructions no visitor would use to ask about the college.
* **The model** (see ``prompts.py``) — the real judge, which sees the question
  and the retrieved sources together and returns an explicit ``scope`` verdict.

The model is more accurate, so this layer is deliberately not trying to be
complete. It only skips work that is obviously wasted.
"""

from __future__ import annotations

import re

# Requests to *produce* an artifact the college assistant has no business
# producing. Anchored to the start of the question, because that is where an
# imperative lives — "write a poem" is out of scope, while "who wrote the college
# anthem" is a perfectly good question containing the same verb.
_PRODUCE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|would you\s+|i want you to\s+|help me\s+)*"
    r"(?:write|create|generate|compose|draft|build|code|program|implement|design|make)\b"
    r"[^?]{0,40}?\b(?:program|programme|code|script|function|algorithm|app|application|"
    r"website|poem|essay|story|song|joke|email|letter|resume|cv|sql|query|"
    r"python|java|javascript|c\+\+|html|css)\b",
    re.I,
)

# Literal code, or an artifact only a programmer asks for.
#
# Deliberately narrower than the obvious version of this rule. Matching a bare
# "in python" caught "Is there a certificate program in Python?" — a real
# question about a real course — and refusing that is a worse outcome than
# missing an out-of-scope one, because the model catches what slips through
# while nothing recovers a wrongly refused visitor.
#
# "python program" is out for the same reason: it is equally the name of a
# course. Only "python code/script/function/snippet" is unambiguous, since no
# one asks a college whether it offers a "python snippet".
_TECHNICAL = re.compile(
    r"\b(?:(?:python|javascript|java|c\+\+|sql)\s+(?:code|script|function|snippet)"
    r"|def\s+\w+\s*\(|SELECT\s+.+\s+FROM\s)",
    re.I,
)

# General-knowledge and personal-assistant requests that have nothing to do with
# a college but often share vocabulary with one ("what is the capital…").
_GENERAL = re.compile(
    r"^\s*(?:what(?:'s| is) the (?:capital|population|weather|time|date|square root|meaning of life)\b"
    r"|who (?:won|is the president|is the prime minister|invented|discovered)\b"
    r"|give me a recipe|how do i (?:cook|bake|make) \b"
    r"|translate\b|solve\b.{0,20}\bequation|what(?:'s| is)\s+\d+\s*[-+*/x]\s*\d+"
    r"|tell me a (?:joke|story)|do my homework|write my assignment)",
    re.I,
)

_PATTERNS = (("produce", _PRODUCE), ("technical", _TECHNICAL), ("general", _GENERAL))


def out_of_scope(question: str) -> str | None:
    """Name of the rule that puts this question out of scope, or None.

    None means "not obviously out of scope" — emphatically *not* "in scope".
    The model makes that call with the sources in front of it.
    """
    text = " ".join((question or "").split())
    if not text:
        return None
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            return name
    return None


# Words that carry no retrieval signal but dominate a short spoken query. Used
# only to judge whether a question is worth re-querying, never to rewrite it.
_FILLER = frozenset(
    "a an the is are was were do does did can could would should will i you we my our "
    "me your please tell about what which who whom how when where why and or of to for "
    "in on at with from that this it its there here hey hi okay ok so just".split()
)


def content_terms(question: str) -> list[str]:
    """The words in a question that could actually match a document."""
    words = re.findall(r"[a-z0-9][a-z0-9'&+.-]*", (question or "").lower())
    return [w for w in words if w not in _FILLER and len(w) > 1]


def is_underspecified(question: str) -> bool:
    """Too little signal to retrieve well — worth one rewrite attempt.

    "tell me more", "what about fees", "and the timings?" are real things people
    say to a voice assistant, and they retrieve badly not because the corpus
    lacks the answer but because there is almost nothing to match on.
    """
    return len(content_terms(question)) <= 2
