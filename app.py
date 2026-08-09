"""SIA voice service — grounded college answering with a voice, behind one small
FastAPI app.

    ask   : retrieve from the scraped college corpus, then answer *only* from
            what was retrieved, on Gemini (kb/answering.py)
    tts   : Kokoro-82M via kokoro-onnx (torch-free ONNX Runtime), cached to disk.
            The voice ``hf_alpha`` is a Hindi female speaker, so English comes
            out in a warm Indian-accented voice — identical on every browser,
            because the client just plays back the bytes returned here.
    stt   : Gemini audio with the college vocabulary biased in, falling back to
            local faster-whisper when the API is unreachable (services/stt.py)

**Where the work happens.** This file is only assembly: configuration lives in
``config.py``, the models live in ``services/``, the endpoints in ``routers/``,
and retrieval and answering in ``kb/``. It used to be one 667-line module that
held all four, which made the interesting parts — chunking, the guards, the
cache thresholds — hard to find among the FastAPI plumbing.

**What is remote and what is not** is a deliberate split rather than a default.
Generation and embeddings are remote because quality dominates there and the
local 1.5B model was measurably worse at the one job that matters (declining
questions the corpus cannot answer). Synthesis stays local because it is free,
CPU-only, and cached, and because a consistent voice is the product.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from config import ALLOW_ORIGINS, KB_ENABLED
from routers import chat, health, voice
from services import prewarm, runtime


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runtime.init(KB_ENABLED)
    prewarm.start(runtime.answer_cache)
    yield


app = FastAPI(title="SIA Voice Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(voice.router)


# Response bodies worth reading back: JSON and text. Audio and the TTS stream are
# logged by size only — dumping WAV bytes into the trace is unreadable, and
# buffering the stream to log it would re-create the very latency /tts/stream
# exists to remove.
_TRACE_BODY_TYPES = ("application/json", "text/", "application/problem+json")
_TRACE_BODY_LIMIT = 2000


@app.middleware("http")
async def _trace_requests(request: Request, call_next):
    """One line per request when DEBUG=1, plus the response it returned.

    The per-model traces cover what each model saw; this covers what the
    *client* asked for and what it got back, which is what you need when the UI
    misbehaves and the question is whether the request even arrived — and, when
    it did, whether the payload was what the UI expected. A 200 carrying
    ``{"text": ""}`` and a 200 carrying a real transcript are the same line
    without the body, and they mean completely different things.
    """
    from kb import debug

    if not debug.ENABLED:
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    client = request.client.host if request.client else "?"
    debug.log(
        f"[http] {request.method} {request.url.path} -> {response.status_code} "
        f"{elapsed:.0f} ms  from {client}"
    )

    ctype = response.headers.get("content-type", "")
    if not any(ctype.startswith(t) for t in _TRACE_BODY_TYPES):
        size = response.headers.get("content-length")
        debug.log(
            f"[http] <- {ctype or 'no content-type'} "
            f"{size + ' bytes' if size else '(streamed)'}"
        )
        return response

    # Draining body_iterator consumes it, so the response has to be rebuilt from
    # the bytes we read — returning the original would send an empty body.
    body = b"".join([chunk async for chunk in response.body_iterator])
    text = body.decode("utf-8", "replace")
    if len(text) > _TRACE_BODY_LIMIT:
        text = f"{text[:_TRACE_BODY_LIMIT]}... [{len(text) - _TRACE_BODY_LIMIT} more chars]"
    debug.log(f"[http] <- {text}")

    return Response(
        content=body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


def _cors_origin(request: Request) -> str:
    origin = request.headers.get("origin", "")
    if "*" in ALLOW_ORIGINS or not ALLOW_ORIGINS:
        return "*"
    return origin if origin in ALLOW_ORIGINS else ALLOW_ORIGINS[0]


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Guarantee CORS headers even on unexpected 500s, so a real server error
    never masquerades as a CORS problem in the browser. (A bare 500 from the
    default handler skips the CORS middleware.)"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {exc}"},
        headers={"Access-Control-Allow-Origin": _cors_origin(request)},
    )
