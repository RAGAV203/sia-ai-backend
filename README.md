# SIA Voice Service (TTS + STT)

A small, self-contained FastAPI service that gives SIA **one consistent
Indian-female voice on every browser/device** and cross-browser speech input —
all open source, no paid APIs.

| Endpoint | Method | In | Out |
|----------|--------|----|-----|
| `/tts`   | POST   | `{ "text": "..." }` (JSON) | `audio/wav` |
| `/tts/stream` | POST | `{ "text": "..." }` (JSON) | length-prefixed WAV clips, one per sentence |
| `/stt`   | POST   | `audio` file (multipart)   | `{ "text": "..." }` |
| `/ask`   | POST   | `{ "question": "..." }` (JSON) | `{ "answer": "...", "grounded": true, "sources": [...] }` |
| `/content`| GET   | –  | `{ "greeting": "...", "suggestions": [...] }` |
| `/health`| GET    | –  | status JSON |
| `/voices`| GET    | –  | list of Kokoro voice ids (confirm `hf_alpha`) |

`/ask` answers open-ended questions from the **scraped college website**, grounded
in retrieved sources and defended against prompt injection — see
[`kb/README.md`](./kb/README.md) for the retrieval design and the security model.
Without an `ANTHROPIC_API_KEY` (or with `KB_ENABLED=0`) it falls back to the
curated answers in [`knowledge.py`](./knowledge.py), whose clips are prewarmed.

`/tts/stream` exists because whole-answer synthesis is the wrong shape for a
voice UI: Kokoro runs below real time, so a long answer means several seconds of
silence first. Streaming per sentence measured **2x faster to first audio** and
stays gapless. The frontend prefers it and falls back to `/tts`.

- **TTS** — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via
  [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) (ONNX Runtime, no
  PyTorch). Voice `hf_alpha` (Hindi female) + `lang=en-us` → warm Indian-accented
  English. Clips are disk-cached, so the fixed set of answers is synthesized once.
- **STT** — [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
  (CTranslate2). Great on Indian-accented English.

### Why answers play instantly

Synthesis is not fast enough to do on demand: a full answer is ~40 s of audio,
and Kokoro on CPU runs at roughly **0.4–0.9× real time**, so generating one
takes 15–40 s. The avatar would sit in "thinking" that whole time.

Since the knowledge base is a *fixed* set of strings, the service instead
**pre-synthesizes every one of them at startup** (`PREWARM=1`, default) in a
background thread. After that each `/tts` the site issues is a disk read:

| | first boot (cold cache) | every request after |
|---|---|---|
| `/tts` per answer | 15–40 s (synthesizing) | **~10 ms** |
| `/stt` per clip | ~4 s (loads Whisper) | **~1 s** |

Progress is visible on `/health` (`prewarm: {done, total, ready}`), so you can
tell "still warming up" apart from "broken". The cache lives in `TTS_CACHE_DIR`
and survives restarts, so a second boot is instant.

This is a **standalone repository** — deploy it to Render on its own; the Vite
frontend only needs its URL via `VITE_API_URL`.

## Run locally

**Docker (matches production, includes espeak-ng/ffmpeg):**

```bash
docker build -t sia-voice .
docker run -p 8000:8000 sia-voice
# → http://localhost:8000/health
```

**Bare Python** — works on Windows/macOS/Linux with no system packages:
`espeak-ng` ships inside `espeakng-loader`, and audio decoding uses PyAV, so
no separate `ffmpeg` install is needed.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py          # fetch model weights once
uvicorn app:app --port 8000
```

The first boot spends a couple of minutes filling the clip cache in the
background — watch `/health` until `prewarm.ready` is `true`. Use `--reload`
only while editing: every reload restarts the prewarm.

Point the frontend at it with `VITE_API_URL=http://localhost:8000` (see the
frontend's `.env.example`).

## Deploy to Render

1. Push this repo to GitHub.
2. Render dashboard → **New + → Blueprint** → select the repo. It reads
   [`render.yaml`](./render.yaml) and creates the `sia-voice` Docker web service.
   (Or **New + → Web Service**, runtime **Docker**, root directory left as the
   repo root.)
3. After it goes live, copy the URL (e.g. `https://sia-voice.onrender.com`) and
   set it as `VITE_API_URL` in your Vercel project, then redeploy the frontend.
4. Set `ALLOW_ORIGINS` to your site's origin (e.g. `https://your-site.vercel.app`).

The Dockerfile bakes the weights into the image, so cold starts don't
re-download them. Render binds `$PORT` automatically.

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `TTS_VOICE` | `hf_alpha` | Any Kokoro voice id (see `/voices`). |
| `TTS_LANG` | `en-us` | `en-us` = correct English + Indian timbre; `hi` = stronger Hindi phonology. |
| `TTS_SPEED` | `0.9` | <1 slower/calmer. |
| `KOKORO_MODEL` | `auto` | `auto` = fastest weights present (fp32 → fp16 → int8). Pin to `kokoro-v1.0.int8.onnx` for low RAM. |
| `ONNX_THREADS` | `min(4, cores)` | Kokoro peaks near 4 threads and *regresses* past it. |
| `PREWARM` | `1` | Pre-synthesize the knowledge base at boot. `0` on a tiny instance. |
| `PREWARM_STT` | `1` | Load Whisper at boot (~200 MB). `0` on a tiny instance. |
| `TTS_CACHE_DIR` | `./tts-cache` | Where synthesized clips are kept. |
| `WHISPER_MODEL` | `base` | `tiny` (lighter) → `small` (more accurate). |
| `WHISPER_COMPUTE` | `int8` | `int8` is smallest/fastest on CPU. |
| `ALLOW_ORIGINS` | `*` | Comma-separated allowed origins. |

**On model choice:** counter-intuitively the **fp32** Kokoro weights are ~2×
*faster* than the int8 ones on CPU (measured RTF 0.42 vs 0.86) — ONNX Runtime's
dynamically-quantized MatMul kernels lose to plain fp32 GEMM here — but cost
~325 MB resident instead of ~92 MB. The Docker image only bakes int8, so
containers stay lean; run `KOKORO_MODEL=kokoro-v1.0.onnx python download_models.py`
to fetch the fast one where RAM allows.

## Troubleshooting

**502 Bad Gateway + a "No 'Access-Control-Allow-Origin' header" (CORS) error in the
browser** — these are almost always the *same* problem, not two. A 502 comes from
Render's proxy (not the app), so it carries no CORS headers, and the browser reports it
as a CORS failure. The app's own errors *do* include CORS headers, so a genuine CORS
error here means the worker crashed. Usual causes:

- **Out of memory (exit 137).** Kokoro + Whisper exceeded the instance RAM. Check the
  Render **Logs** for `Out of memory` / `exit status 137`. Fixes: this repo now defaults
  to the **int8** Kokoro model (`KOKORO_MODEL=kokoro-v1.0.int8.onnx`, ~88 MB); also set
  `WHISPER_MODEL=tiny`, or move to the **Standard (2 GB)** plan.
- **Free-tier cold start.** Free instances spin down when idle; the first request wakes
  them and can 502 for a few seconds. The frontend retries automatically, so it recovers —
  but to avoid it entirely use a plan that stays warm.

Confirm the service directly (bypasses the browser/CORS): `curl -i https://<service>/health`
and `curl -i -X POST https://<service>/tts -H "Content-Type: application/json" -d '{"text":"hello"}' -o out.wav`.

## RAM / cost notes

No GPU needed — both models run on CPU. Real-world behaviour observed:

- **Render Free (512 MB)** — **not recommended.** The shared CPU takes ~60 s to
  cold-load a model (→ 502s past the proxy timeout), the instance spins down when
  idle, and Kokoro + Whisper together **OOM-kill the worker** (502 with an empty
  body, which the browser reports as a CORS error). Fine for a quick demo, not
  for real use.
- **Render Standard (2 GB, always-on)** — **recommended.** Holds both models
  comfortably; with `WARMUP=1` the first request is fast. Use `WHISPER_MODEL=base`
  here for better accuracy.
- **Render Starter ($7, 512 MB, always-on)** — tight but possible: keep
  `WHISPER_MODEL=tiny`, `KOKORO_MODEL=kokoro-v1.0.int8.onnx`, `CPU_THREADS=1`.
  Always-on avoids the cold-start, but RAM is still near the edge if both models
  load at once.

`WARMUP=1` loads both models at boot (only helps on always-on plans). The frontend
also retries through transient 502s and shows a "Preparing voice…" state, and each
`/tts` clip is disk-cached so repeat answers are instant.
