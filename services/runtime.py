"""Process-wide singletons, created once at startup.

The rate limiter and the answer cache are both stateful and both shared by
every request, so they are built in the lifespan hook and read from here rather
than being module-level side effects of an import. That keeps importing a
router free of work, which is what makes the test suite able to import pieces
of the app without booting models.
"""

from __future__ import annotations

rate_limiter = None
answer_cache = None
kb_error: str | None = None


def init(kb_enabled: bool) -> None:
    global rate_limiter, answer_cache, kb_error

    from kb.guard import RateLimiter

    rate_limiter = RateLimiter()
    if not kb_enabled:
        return

    try:
        from kb.answer_cache import AnswerCache
        from kb.index import load as load_index

        index = load_index()
        print(f"[kb] index: {len(index.chunks) if index else 0} chunks")
        answer_cache = AnswerCache()
    except Exception as exc:  # noqa: BLE001 - never block startup on the KB
        kb_error = f"{type(exc).__name__}: {exc}"
        print(f"[kb] disabled: {kb_error}")
