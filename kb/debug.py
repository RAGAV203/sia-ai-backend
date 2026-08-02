"""Verbose tracing for the answering pipeline, behind ``DEBUG=1``.

Off by default and effectively free when off: every call site guards on
``ENABLED`` before building its message, so no formatting or truncation work
happens in production.

What it prints, per question:

  1. the vetted question and how retrieval scored every chunk it considered
  2. the **exact** prompt sent to the model, both system and user turns
  3. the **raw** model output, before the guards touch it
  4. what each guard decided and why the caller got the answer it did

Points 2 and 3 are the ones that matter when an answer looks wrong. A grounded
answer is a function of the retrieved text, and eyeballing that text next to the
reply is the fastest way to tell a retrieval problem (the right page never made
it into the prompt) from a generation problem (it did, and the model still got
it wrong). Those two failures look identical from outside.

Enable with ``DEBUG=1``; add ``DEBUG_FULL_PROMPT=1`` to stop truncating chunk
bodies when you need to see exactly what the model read.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager

ENABLED = os.getenv("DEBUG", "0").lower() in ("1", "true", "yes", "on")
FULL_PROMPT = os.getenv("DEBUG_FULL_PROMPT", "0").lower() in ("1", "true", "yes", "on")
# Mirror the trace to a file as well as the console. Useful on a deployed box
# where scrollback is short or the process output is not easy to reach.
LOG_FILE = os.getenv("DEBUG_LOG_FILE", "").strip()
# stdout by default, so the trace lands on the same stream as every other
# message this service prints ("[llm] ...", "[kb] index: ...", "[prewarm] ...").
# Splitting them across stdout and stderr meant the trace appeared to be missing
# whenever output was redirected or captured by a process manager. Set
# DEBUG_STREAM=stderr to keep it off stdout when something is parsing that.
STREAM = os.getenv("DEBUG_STREAM", "stdout").strip().lower()

_WIDTH = 78
_BODY_LIMIT = 100_000 if FULL_PROMPT else 600

_file = None
if ENABLED and LOG_FILE:
    try:
        _file = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
        _file.write(f"\n\n===== debug session started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        _file.flush()
    except OSError as exc:  # noqa: BLE001 - never let logging break the service
        sys.stderr.write(f"[debug] cannot open {LOG_FILE}: {exc}\n")


def _out(line: str = "") -> None:
    """Print one trace line, surviving any console encoding.

    Goes to stdout by default so it interleaves with the service's own
    ``print()`` output rather than sitting on a separate stream.

    The encode/decode round-trip matters on Windows: the default console is
    cp1252 and scraped page text is full of em dashes and curly quotes, so a
    plain write raises UnicodeEncodeError — crashing the request precisely when
    tracing was turned on to find out why it was failing.
    """
    stream = sys.stderr if STREAM == "stderr" else sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        print(line, file=stream, flush=True)
    except UnicodeEncodeError:
        safe = line.encode(encoding, "replace").decode(encoding, "replace")
        print(safe, file=stream, flush=True)

    if _file is not None:
        try:
            _file.write(line + "\n")
            _file.flush()
        except (OSError, ValueError):
            pass  # disk full or file closed — the console trace still works


def log(message: str) -> None:
    if ENABLED:
        _out(f"  {message}")


def section(title: str) -> None:
    if ENABLED:
        _out(f"\n{'=' * _WIDTH}\n== {title}\n{'=' * _WIDTH}")


def block(title: str, body: str) -> None:
    """Print a labelled block of text, truncated unless DEBUG_FULL_PROMPT=1."""
    if not ENABLED:
        return
    text = body if len(body) <= _BODY_LIMIT else body[:_BODY_LIMIT] + f"\n... [{len(body) - _BODY_LIMIT} more chars]"
    _out(f"\n--- {title} {'-' * max(0, _WIDTH - len(title) - 5)}")
    for line in text.splitlines():
        _out(f"| {line}")
    _out("-" * _WIDTH)


def sources(hits: list) -> None:
    """One line per retrieved chunk: both scores, plus where it came from."""
    if not ENABLED:
        return
    _out(f"\n--- retrieved {len(hits)} chunk(s) " + "-" * 46)
    for i, hit in enumerate(hits, 1):
        sim = hit.get("similarity", 0.0)
        rrf = hit.get("score", 0.0)
        title = (hit.get("title") or "?")[:38]
        body = " ".join((hit.get("text") or "").split())[:88]
        _out(f"| {i}. sim={sim:.3f} rrf={rrf:.4f}  {title}")
        _out(f"|    {body}")
        _out(f"|    {hit.get('url', '')}")
    _out("-" * _WIDTH)


@contextmanager
def timed(label: str):
    """Time a stage and report it, without costing anything when disabled."""
    if not ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        _out(f"  [{label}] {(time.perf_counter() - start) * 1000:.0f} ms")
