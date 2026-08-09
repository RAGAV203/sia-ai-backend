"""SIA's knowledge base and answering logic — the single source of truth.

The frontend fetches the greeting + suggestion chips from ``/content`` and gets
answers from ``/ask``; it keeps only a tiny offline fallback. Edit the content
here and redeploy the service — no frontend rebuild required.
"""

from __future__ import annotations

import re
from typing import Dict, List

GREETING = (
    "Hi, I'm Sia, your guide to Shasun Jain College. Ask me anything about our "
    "programs, campus life, or admissions."
)

SUGGESTIONS: List[Dict[str, str]] = [
    {
        "id": "programs",
        "chip": "Academic Programs",
        "question": "What academic programs do you offer?",
        "answer": (
            "Good question. Shri Shankarlal Sundarbai Shasun Jain College for Women "
            "offers a rich variety of programs across Arts, Science, Commerce, and Media "
            "Studies. On the undergraduate side, you will find popular choices like B.Com "
            "in General, Corporate Secretaryship, Accounting and Finance, as well as "
            "Honours. For the sciences, we have B.Sc in Computer Science, Visual "
            "Communication, and Psychology. There is also our B.B.A program. At the "
            "postgraduate level, our M.A. in Journalism and Communication is especially "
            "well-regarded. Every program is designed with a focus on experiential "
            "learning and industry readiness — so you graduate prepared, not just educated."
        ),
    },
    {
        "id": "housing",
        "chip": "Campus Life & Facilities",
        "question": "Tell me about student facilities and campus life.",
        "answer": (
            "I am glad you asked about campus life. Our campus is truly vibrant. You will "
            "find state-of-the-art computer labs, a modern visual communication studio, a "
            "well-stocked library, and fully equipped seminar halls. But it is not just "
            "about facilities. The real magic happens through our student clubs, sports "
            "programs, and cultural festivals. We also have the Shasun Alliance of Young "
            "Entrepreneurs — known as SAYE — which nurtures leadership, creative thinking, "
            "and entrepreneurial spirit. It is a place where you will grow far beyond the "
            "classroom."
        ),
    },
    {
        "id": "admissions",
        "chip": "Admissions & Eligibility",
        "question": "What are the admission requirements and how can I apply?",
        "answer": (
            "Admissions at Shasun Jain College are open to female candidates. The entire "
            "process is handled online through our official portal, making it very "
            "straightforward. Selection is based on academic merit from your higher "
            "secondary examinations — that is, your 10 plus 2 or equivalent — following the "
            "guidelines of the University of Madras. You can register online, upload the "
            "required certificates, and track your application status at every step. If you "
            "need any help along the way, our admissions team is always available to guide you."
        ),
    },
    {
        "id": "careers",
        "chip": "Placement & Career Cell",
        "question": "Do you provide job placement assistance?",
        "answer": (
            "Absolutely. Our Shasun Career Guidance and Placement Cell is one of our "
            "strongest pillars. It actively bridges the gap between academics and the "
            "professional world. Students receive systematic training in soft skills, "
            "aptitude, and interview preparation — so when the time comes, you are ready. "
            "Leading multinational corporations, IT firms, media houses, and financial "
            "institutions recruit directly from our campus every year. Many of our "
            "graduates have gone on to secure outstanding career starts. It is something we "
            "take great pride in."
        ),
    },
]

KEYWORDS: Dict[str, List[str]] = {
    "programs": ["program", "degree", "major", "course", "study"],
    "housing": ["housing", "dorm", "campus", "live", "accommodation", "life"],
    "admissions": ["admission", "apply", "deadline", "requirement", "enroll"],
    "careers": ["job", "placement", "career", "intern", "employ", "hire"],
}

FALLBACK_ANSWER = (
    "That is an excellent question. I appreciate you asking. While I am currently in "
    "preview mode with a curated knowledge base, I would love to connect you with someone "
    "who can give you the most accurate answer. May I help you explore our Academics, "
    "Campus Life, Admissions, or Career services instead?"
)

# Said when the request was never about the college. Kept separate from
# FALLBACK_ANSWER for the same reason the generated path keeps them separate:
# "let me connect you with someone who can answer" is the wrong thing to say to
# someone who asked for a recipe.
OFF_TOPIC_ANSWER = (
    "That is outside what I can help with. I am here to answer questions about "
    "Shasun Jain College. Would you like to know about our programmes, campus life, "
    "admissions, or placements?"
)


def _normalize(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[?.!]+$", "", value)
    return value


def resolve_answer(query: str) -> str:
    """Best curated answer for a question, or the safe fallback.

    This is the last line of the degradation chain, and it used to be a hole in
    the scope defense large enough to drive the whole system through.

    Keyword matching was a bare substring test, so "write a python program"
    matched the keyword "program" and returned the college's full academic
    programmes answer — confidently, and flagged as grounded, without any model
    ever seeing the question. Every guard upstream was bypassed, because this
    path exists precisely for when the upstream is unavailable.

    Two changes close it. Scope is checked first, so an out-of-scope request
    cannot reach the keyword table at all; and the keywords are matched on word
    boundaries, so "program" no longer fires inside "programming" or "programme"
    by accident of spelling.
    """
    q = _normalize(query)
    if not q:
        return FALLBACK_ANSWER

    # An exact match on a suggestion chip is unambiguous and always allowed.
    for s in SUGGESTIONS:
        if _normalize(s["question"]) == q:
            return s["answer"]

    # Anything the cheap triage recognizes as out of scope must not be answered
    # from the college knowledge base, however well its words happen to overlap.
    try:
        from kb.triage import out_of_scope

        if out_of_scope(q):
            return OFF_TOPIC_ANSWER
    except ImportError:
        pass  # knowledge base not installed; keyword matching still works

    for s in SUGGESTIONS:
        for keyword in KEYWORDS.get(s["id"], []):
            if re.search(rf"\b{re.escape(keyword)}\w*\b", q):
                return s["answer"]

    return FALLBACK_ANSWER


def public_suggestions() -> List[Dict[str, str]]:
    """Chips for the UI — question text only, answers stay server-side."""
    return [{"id": s["id"], "chip": s["chip"], "question": s["question"]} for s in SUGGESTIONS]


def spoken_texts() -> List[str]:
    """Every string SIA can ever say, in the order she is likely to say it.

    The knowledge base is a fixed set, so the service pre-synthesizes all of
    these at startup — after that every ``/tts`` call the site makes is a disk
    cache hit and the avatar starts talking immediately.
    """
    texts = [GREETING] + [s["answer"] for s in SUGGESTIONS] + [FALLBACK_ANSWER]

    # The generated-answer path has two fixed strings of its own: the canonical
    # "I can't answer that" sentence the model is told to emit, and the
    # off-contract reply. Both are said often enough — and are constant enough —
    # to be worth prewarming alongside the curated set.
    try:
        from kb.intent import replies as intent_replies
        from kb.prompts import NO_ANSWER_SENTENCE, OFF_CONTRACT_ANSWER

        # Social turns ("thanks", "how are you") are answered from a fixed set
        # too, and are asked often enough to be worth having on disk.
        texts += [NO_ANSWER_SENTENCE, OFF_CONTRACT_ANSWER] + intent_replies()
    except ImportError:
        pass  # knowledge base not installed; the curated set still works

    return texts
