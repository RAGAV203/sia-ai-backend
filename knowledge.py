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


def _normalize(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[?.!]+$", "", value)
    return value


def resolve_answer(query: str) -> str:
    q = _normalize(query)
    if not q:
        return FALLBACK_ANSWER

    for s in SUGGESTIONS:
        if _normalize(s["question"]) == q:
            return s["answer"]

    for s in SUGGESTIONS:
        for keyword in KEYWORDS.get(s["id"], []):
            if keyword in q:
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
    return [GREETING] + [s["answer"] for s in SUGGESTIONS] + [FALLBACK_ANSWER]
