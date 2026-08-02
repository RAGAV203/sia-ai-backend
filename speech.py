"""Sentence segmentation for streaming speech synthesis.

Splitting has to be *stable*, because each sentence becomes a TTS disk-cache
key: if the same answer segments differently on two requests, every clip is a
cache miss. It also has to avoid the abbreviation trap, which this content is
unusually full of — "B.Com", "B.Sc", "M.A.", "Dr.", "10 plus 2" — where naive
splitting on "." shatters a programme name into three unspeakable fragments.
"""

from __future__ import annotations

import re

# Abbreviations whose trailing period never ends a sentence. Matched
# case-insensitively against the token immediately before the period.
_ABBREVIATIONS = {
    "b", "m", "a", "s", "c", "d", "e", "g", "i",     # initials: B.Com, M.A., B.Sc, e.g., i.e.
    "sc", "com", "ba", "bba", "bsc", "msc", "ma", "mba", "mcom", "phd", "ph",
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# A sentence ends at .!? + whitespace, but only when what precedes the period is
# not an abbreviation and what follows looks like a new sentence.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")

MIN_SENTENCE_CHARS = 25   # below this, merge forward — a 3-word clip isn't worth a request
MAX_SENTENCE_CHARS = 320  # above this, synthesis latency defeats the point of streaming

# The opening clip is the only one the listener actually waits on, so it is
# sized for latency while the rest are sized for throughput. It still has to
# cover synthesis of the clip behind it: at RTF ~0.75 a 100-char opener buys
# ~4s of playback against ~2.5s of work, which holds comfortably.
FIRST_SENTENCE_MAX_CHARS = 110


def _ends_with_abbreviation(text: str) -> bool:
    m = re.search(r"([A-Za-z]+)\.[\"')\]]*\s*$", text)
    if not m:
        return False
    return m.group(1).lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split into speakable units, merging fragments and capping long ones."""
    text = " ".join((text or "").split())
    if not text:
        return []

    # Provisional split, then re-join anything that broke on an abbreviation.
    parts: list[str] = []
    for piece in _BOUNDARY.split(text):
        if piece and parts and _ends_with_abbreviation(parts[-1]):
            parts[-1] = f"{parts[-1]} {piece}"
        elif piece:
            parts.append(piece)

    # Merge too-short fragments forward, the first one included.
    #
    # Letting a short opener stand ("Good question.", 1.2s of audio) looks like a
    # latency win but measured worse: playback of clip N has to cover synthesis
    # of clip N+1, and at RTF ~0.26 a 1.2s clip only covers ~4.6s of the next
    # one. Against a 9s follow-up that left a 1.1s hole in the middle of the
    # sentence. Keeping every clip above the floor keeps neighbouring clips
    # within ~4x of each other, which is comfortably inside the RTF budget.
    merged: list[str] = []
    for piece in parts:
        if merged and len(merged[-1]) < MIN_SENTENCE_CHARS:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)

    # Split anything oversized on clause boundaries, with a tighter cap on the
    # opening clip since that is the one the listener waits through.
    out: list[str] = []
    for piece in merged:
        limit = FIRST_SENTENCE_MAX_CHARS if not out else MAX_SENTENCE_CHARS
        while len(piece) > limit:
            window = piece[:limit]
            cut = max(window.rfind(", "), window.rfind("; "), window.rfind(" — "), window.rfind(" and "))
            if cut < MIN_SENTENCE_CHARS:
                cut = window.rfind(" ")
            if cut < MIN_SENTENCE_CHARS:
                break
            out.append(piece[:cut].strip().rstrip(",;"))
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)

    return [s for s in (p.strip() for p in out) if s]
