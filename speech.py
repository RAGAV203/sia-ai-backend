"""Sentence segmentation for streaming speech synthesis.

Splitting has to be *stable*, because each sentence becomes a TTS disk-cache
key: if the same answer segments differently on two requests, every clip is a
cache miss. It also has to avoid the abbreviation trap, which this content is
unusually full of — "B.Com", "B.Sc", "M.A.", "Dr.", "10 plus 2" — where naive
splitting on "." shatters a programme name into three unspeakable fragments.
"""

from __future__ import annotations

import os
import re

# Abbreviations whose trailing period never ends a sentence. Matched
# case-insensitively against the token immediately before the period.
_ABBREVIATIONS = {
    "b", "m", "a", "s", "c", "d", "e", "g", "i",     # initials: B.Com, M.A., B.Sc, e.g., i.e.
    "sc", "com", "ba", "bba", "bsc", "msc", "ma", "mba", "mcom", "phd", "ph",
    "dr", "mr", "mrs", "ms", "prof", "st", "no", "vs", "etc", "approx",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# A sentence ends at .!?, or at a danda, + whitespace — but only when what
# precedes it is not an abbreviation and what follows looks like a new sentence.
#
# `।` (U+0964 devanagari danda) is Hindi's full stop and is what the translator
# emits for one. Without it here, a three-sentence Hindi answer is a single
# 300-character clip: the listener waits through the whole synthesis before
# hearing a word, which is exactly the latency that sentence streaming exists to
# remove. `॥` (double danda) is included for completeness; it is rare in prose.
#
# Tamil and Malay both use the Latin full stop, so they need nothing extra.
_BOUNDARY = re.compile(r"(?<=[.!?।॥])[\"')\]]*\s+")

MIN_SENTENCE_CHARS = 25   # below this, merge forward — a 3-word clip isn't worth a request
MAX_SENTENCE_CHARS = 320  # above this, synthesis latency defeats the point of streaming

# The opening clip is the only one the listener actually waits on, so it is
# sized for latency while the rest are sized for throughput.
#
# It used to be 110, because synthesis was serial and playback of clip N had to
# cover synthesis of clip N+1 — a short opener bought a fast start and then ran
# dry. Sentences are now synthesized several at a time (config.SYNTH_WORKERS),
# so clip 2 is already being rendered while clip 1 plays and the opener no
# longer has to fund it.
#
# Removing that constraint is worth roughly two seconds of silence at the start
# of every answer: a 69-char opener measured 4.2s of audio and 4.8s to
# first sound, where a ~60-char one starts speaking in about half that.
FIRST_SENTENCE_MAX_CHARS = int(os.getenv("TTS_FIRST_SENTENCE_CHARS", "60"))


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
