# SIA Voice Service

A small FastAPI service that answers questions about Shasun Jain College,
grounded in the scraped college website, and speaks the answer in **one
consistent Indian-female voice on every browser/device**.

| Endpoint | Method | In | Out |
|----------|--------|----|-----|
| `/tts`   | POST   | `{ "text": "..." }` (JSON) | `audio/wav` |
| `/tts/stream` | POST | `{ "text": "..." }` (JSON) | length-prefixed WAV clips, one per sentence |
| `/stt`   | POST   | `audio` file (multipart)   | `{ "text": "...", "engine": "gemini", "duration_s": 2.4 }` |
| `/ask`   | POST   | `{ "question": "..." }` (JSON) | `{ "answer": "...", "grounded": true, "sources": [...] }` |
| `/content`| GET   | –  | `{ "greeting": "...", "suggestions": [...] }` |
| `/health`| GET    | –  | status JSON |
| `/voices`| GET    | –  | list of Kokoro voice ids (confirm `hf_alpha`) |

## What runs where, and why

Remote and local is a deliberate split, not a default.

| | runs on | why |
|---|---|---|
| **Answering** | Gemini `gemini-3.1-flash-lite` | Quality dominates, and the job that matters most — declining a question the corpus cannot answer — is where the old local 1.5B model was weakest. ~$0.0005/answer, and most questions never reach it. |
| **Embeddings** | Gemini `gemini-embedding-001` (768-d) | Trained for question-to-passage search with an explicit `task_type`. This is what made the off-topic gate work at all (see below). |
| **Speech to text** | Gemini audio, → faster-whisper | The vocabulary is the hard part, not the language. The prompt carries a college glossary so "Shasun" and "B.Com Corporate Secretaryship" survive. Whisper takes over automatically when the API is unreachable. |
| **Speech synthesis** | Kokoro-82M, local | Free per request, CPU-only, disk-cached — and a voice that is identical on every browser is the product. Remote TTS would be strictly worse on all four counts. |

`/ask` is grounded in retrieved sources and defended against prompt injection —
see [`kb/README.md`](./kb/README.md) for the retrieval design and the security
model. Without a `GEMINI_API_KEY` (or with `KB_ENABLED=0`) it falls back to the
curated answers in [`knowledge.py`](./knowledge.py), whose clips are prewarmed.

## Languages

English, Tamil, Hindi and Malay, selected by the client and carried on every
endpoint as `lang`. The registry is [`languages.py`](./languages.py).

**English is the pivot, not merely the default.** A non-English question is
translated to English on the way in, the entire existing pipeline runs on
English exactly as it always has, and the guarded English answer is translated
on the way out:

```
question (ta) ─▶ guard ─▶ intent ─┬─▶ TRANSLATE IN ─▶ [English pipeline] ─▶ TRANSLATE OUT ─▶ answer (ta)
                                  └─▶ canned reply (no model call at all)
```

Answering natively in Tamil would be one API call instead of two, and it breaks
three things at once — which is why it isn't done:

* `kb/guard.py` detects a refusal by matching English verb families. A Tamil
  refusal matches none of them, so it is marked `grounded` and **cached**,
  pinning "I don't know" to a question the corpus can answer.
* `kb/answer_cache.py` compares questions with a local English MiniLM. Tamil
  vectors from it are not worse, they are meaningless.
* `kb/triage.py` reads the question as a *request* using English imperatives. In
  Tamil it sees nothing, and every out-of-scope question reaches the model.

Two things fall out of the pivot that are worth knowing. The answer cache is
**shared across all four languages** — a Tamil visitor asking about fees hits the
entry an English visitor created — and the guards keep their calibration, so
adding a fifth language touches `languages.py` and `services/translate.py`
rather than every pattern list in the tree.

The intent gate runs *before* the translation and matches social turns in their
own script, so "நன்றி" costs no model call at all.

**One voice covers all four.** `hf_alpha` is a Hindi female speaker, and every
phoneme espeak-ng produces for Tamil and Malay falls inside Kokoro's 114-symbol
IPA vocabulary — measured 100% coverage per language. So there is one engine,
one disk cache, no new dependency and no per-request cost. Hindi is native
quality. Tamil and Malay are intelligible and in the right voice, but carry an
Indian-English accent and lose some vowel-length distinctions, because Kokoro
was never *trained* on them. If that ever becomes the limiting factor, the fix
is a second backend behind `services.tts.cached_wav`; nothing above that
function needs to know.

`GET /translit?text=vanakkam&lang=ta` proxies Google Input Tools so a QWERTY
keyboard can type Tamil and Hindi. It is on the keystroke path, so it caches
aggressively, times out fast, and degrades to empty candidates rather than an
error.

## How a question is answered

Retrieve-then-generate is not enough, and one example shows why. Asked to
**"write a python program"**, the assistant used to reply with the college's
academic programmes — because the word "program" matched pages like "Digital
Marketing Proficiency Program" at 0.597 similarity, and nothing downstream ever
asked whether *writing software* is something it does.

No threshold can fix that. Similarity measures topical overlap, and the overlap
is real. So the pipeline asks two questions a person would ask — *is this
something I do?* and *does what I found actually answer it?* — in four stages:

```
question
  │
  ├─ 1. TRIAGE     kb/triage.py — pattern check for obviously out-of-scope
  │                requests. Free, no API call. Zero false positives on 32 real
  │                college questions; catches 15 of 15 out-of-scope ones.
  │                        └─▶ out of scope: refuse, stop here
  │
  ├─ 2. RETRIEVE   hybrid dense + BM25 + RRF, then
  │                  · per-page cap (one page may take 2 of k slots)
  │                  · near-duplicate suppression (cosine > 0.98)
  │                  · relative margin — drop chunks far below the best hit
  │                one adaptive rewrite if the question is too thin to match on
  │                        └─▶ nothing above the floor: refuse, stop here
  │
  ├─ 3. VERIFY     one schema-constrained call returns scope + sufficiency +
  │                answer. The model judges the question against the sources it
  │                was given, and must commit to a verdict as a field.
  │
  └─ 4. ROUTE      scope=out_of_scope  → "that's outside what I can help with"
                   sufficiency=full    → answer from the sources
                   sufficiency=partial → the facts, then what's missing
                   sufficiency=none    → "I don't have that detail" + the office
```

Verification is free: it is the same call that writes the answer, constrained to
a schema, measured at the same ~870 ms as free text. What it buys is that the
judgement is an explicit field rather than something the caller has to infer
from the tone of a sentence.

**The two refusals are different sentences on purpose.** Telling someone who
asked for a Python program to "contact the college office" is nonsense; telling
someone who asked about fees that their question is "outside what I can help
with" turns a gap in the corpus into an apparent brush-off. Only questions that
were about the college get pointed at the college.

| you ask | scope | sufficiency | you get |
|---|---|---|---|
| "write a python program" | out_of_scope | – | outside what I can help with |
| "is there a certificate program in Python?" | college | full | *answers* — there is an add-on Python course |
| "what programs do you offer" | college | full | the actual programme list |
| "what is the fee for B.Com" | college | none | I don't have that detail → college office |
| "who can apply for admission?" | college | partial | women-only, plus what the sources do say |

`/tts/stream` exists because whole-answer synthesis is the wrong shape for a
voice UI: Kokoro runs below real time, so a long answer means several seconds of
silence first. Streaming per sentence gets to first audio far sooner. The
frontend prefers it and falls back to `/tts`.

### Keeping streamed speech continuous

Sentence streaming only works while synthesis outruns playback. It did not, and
the symptom was speech arriving in audible chunks. Three causes, all measured on
one ~19-second answer, three runs per configuration on a busy machine:

| `TTS_WORKERS` | time to first audio | RTF | gaps | worst gap |
|---|---|---|---|---|
| 1 (the old behaviour) | 5.3 s | 1.6 | 2–3 | **4.9–7.2 s** |
| **2** (default) | 5.7 s | 0.90 | 1 | 1.5 s |
| 3 | 7.2 s | 0.83 | **0** | 0 |
| any, warm cache | 0.02 s | – | 0 | 0 |

How many workers it takes to reach zero depends on how loaded the host is — on
an idle machine 2 was already gapless. 2 is the default because it removes the
multi-second stalls that made speech unlistenable and halves worst-case latency,
without a third copy of the weights. Raise it to 3 if RAM allows and cold-cache
answers still sound chunky.

The warm-cache row is the one most users experience: fixed answers are
prewarmed and the semantic cache serves repeats verbatim, so only a genuinely
novel question synthesizes live.

Other measured improvements, independent of worker count:

| | before | after |
|---|---|---|
| silence at each seam | ~135 ms, uneven | 72 ms, uniform |
| repeat answer (cached) | – | 34 ms for the whole stream |

1. **One session could not keep up.** int8 Kokoro runs at RTF 0.95 — no margin,
   so the prewarm thread or a second listener pushed it past 1.0 and playback
   drained faster than clips arrived. Sentences are now synthesized on a pool of
   sessions (`TTS_WORKERS`, default 2). Kokoro is small and cannot use many
   cores in one session, so the thread budget is *split* across sessions rather
   than widening one: 2 workers × 2 threads (RTF 0.55) beat 1 × 4 (0.95) and
   2 × 4 (0.63).

2. **Every clip carried the vocoder's own padding** — ~25 ms before and ~110 ms
   after — which butt-joining turned into dead air at every seam. It is now
   trimmed and replaced by one deliberate, uniform 60 ms pause. A short gap
   between sentences is wanted; an arbitrary one is not.

3. **espeak is not thread-safe.** Parallel synthesis initially produced mangled
   output ("number of lines in input and output must be equal") because Kokoro's
   phonemizer calls into a C library with process-global state — intermittently,
   and returning a *wrong* clip rather than an error. Phonemization is now
   serialized under a lock while the vocoder pass, which is the slow part, still
   runs concurrently.

Two things measured worse and were rejected: rendering the opening clip
exclusively before submitting the rest (first audio 3.4 s instead of 4.7 s, but
**1.3 s of silence** at the first seam), and 3 workers (no gap, but first audio
4.9 s as the thread budget spread too thin).

On a warm cache — the normal case, since the fixed answers are prewarmed and the
semantic cache serves paraphrases verbatim — the whole stream returns in 34 ms.

## Measured behaviour

Scored by `python -m tests.eval_accuracy --answers` on 15 questions verified
against the live site, plus 5 questions the corpus cannot answer.

| | before | after |
|---|---|---|
| retrieval recall | 12/15 (80%) | **14/15 (93%)** |
| answer accuracy | — | **15/15 (100%)** |
| off-topic past the gate | 5/5 leaked | **0/5** |
| off-topic declined by the model | inconsistent | **5/5** |
| corpus actually indexed | 19% | **~100%** |
| answer latency (median) | 6–12 s | **1.1 s** |
| retrieval latency (median) | 6 ms | 420 ms |
| resident memory | ~1.5 GB | ~350 MB |

Two of those deserve a note.

**Retrieval got slower.** Query embedding is now a network round-trip, so
retrieval went from 6 ms to ~420 ms. That is a real regression, paid for
deliberately: it buys the recall and gate numbers above, and it is invisible next
to the 1.1 s answer it precedes. Repeat questions skip it via an in-process LRU,
and the semantic answer cache skips the whole pipeline.

**The off-topic gate became possible.** It used to be a pure cost filter,
because retrieval scores could not separate on- from off-topic on this corpus —
the site really does contain cricket results, so "who won the cricket world cup"
really does match a tournament page. Under the new encoder the bands separate
cleanly, so the gate now rejects most off-topic questions before they cost a
generation, with the model still the backstop:

| encoder | on-topic | off-topic | overlap |
|---|---|---|---|
| BGE (before) | 0.569–0.699 | 0.422–0.583 | yes |
| gemini-embedding-001 | 0.638–0.741 | 0.499–0.567 | **no** |

## Layout

```
app.py          # assembly only: middleware, CORS, router mounting
config.py       # every environment-driven setting in one place
routers/        # HTTP surface — chat.py, voice.py, health.py
services/       # tts.py (Kokoro), stt.py (Gemini + Whisper), prewarm.py, runtime.py
kb/             # corpus, retrieval, answering, guards — see kb/README.md
knowledge.py    # curated fallback answers
speech.py       # sentence segmentation for streaming synthesis
```

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

Full list with commentary in [`.env.example`](./.env.example).

| Var | Default | Notes |
|-----|---------|-------|
| `GEMINI_API_KEY` | – | **Required.** Covers answering, embeddings and STT. `GOOGLE_API_KEY` also accepted. |
| `GEMINI_TEXT_MODEL` | `gemini-3.1-flash-lite` | Cheapest model that answered correctly and declined off-topic. Don't use a `-latest` alias — answer text is the TTS cache key. |
| `GEMINI_AUDIO_MODEL` | `gemini-3.5-flash-lite` | Cheapest audio-input rate. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Changing this invalidates the index *and* the similarity gate. |
| `GEMINI_EMBED_DIM` | `768` | Matryoshka truncation. Vectors arrive unnormalized at <3072 and are renormalized in `kb/embed.py`. |
| `GEMINI_EMBED_RPM` | `90` | Paces an index rebuild under the free tier's 100 texts/min. Raise on a paid key. |
| `RETRIEVAL_MIN_SIMILARITY` | `0.60` | Calibrated to the encoder above; see the table earlier. |
| `RETRIEVAL_TOP_K` | `8` | Was 5 when prefill cost 7 ms/token locally. No longer a constraint. |
| `EMBED_BACKEND` | `gemini` | `local` runs the bundled BGE ONNX encoder instead (no key, lower quality). Requires a rebuild. |
| `STT_BACKEND` | `gemini` | `whisper` skips the API entirely. |
| `TTS_VOICE` | `hf_alpha` | Any Kokoro voice id (see `/voices`). |
| `TTS_LANG` | `en-us` | `en-us` = correct English + Indian timbre; `hi` = stronger Hindi phonology. |
| `TTS_SPEED` | `0.9` | <1 slower/calmer. |
| `KOKORO_MODEL` | `auto` | `auto` = fastest weights present (fp32 → fp16 → int8). Pin to `kokoro-v1.0.int8.onnx` for low RAM. |
| `ONNX_THREADS` | `min(4, cores)` | Kokoro peaks near 4 threads and *regresses* past it. |
| `PREWARM` | `1` | Pre-synthesize the knowledge base at boot. `0` on a tiny instance. |
| `PREWARM_STT` | `0` | Load the Whisper *fallback* at boot (~200 MB). Now off by default — it only serves a path that runs when Gemini is unreachable. |
| `TTS_CACHE_DIR` | `./tts-cache` | Where synthesized clips are kept. |
| `WHISPER_MODEL` | `base` | Fallback engine. `tiny` (lighter) → `small` (more accurate). |
| `WHISPER_COMPUTE` | `int8` | `int8` is smallest/fastest on CPU. |
| `ALLOW_ORIGINS` | `*` | Comma-separated allowed origins. |

### Rebuilding the knowledge base

The index is **committed**, not built during the Docker build — embeddings are a
metered API call now, so building in the image would need a live key and could
fail on a 429. Re-run these locally whenever the site is re-scraped:

```bash
python -m kb.ingest     # crawl -> kb-data/corpus.jsonl  (~40 min; HTML is cached)
python -m kb.index      # embed -> vectors.npy + chunks.jsonl  (~8 min, quota-paced)
```

`kb-data/embed-cache.jsonl` keys every vector by content, so a rebuild after a
chunker tweak only re-embeds what actually changed.

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

No GPU needed. **Not everything is remote** — two models are deliberately local,
and they are what the memory budget is made of:

| | where | why |
|---|---|---|
| answering, embeddings, STT | Gemini | quality dominates; see the table at the top |
| **TTS (Kokoro-82M)** | **local** | free per request, CPU-only, disk-cached, and one identical voice everywhere |
| **answer-cache encoder (MiniLM)** | **local** | its 0.65/0.93 thresholds are calibrated to this model; keeps a cache hit free and offline |
| STT fallback (faster-whisper) | local, optional | only runs when Gemini is unreachable |

Measured resident memory (psutil RSS, int8 Kokoro, `ONNX_THREADS=1`):

| stage | RSS |
|---|---|
| python + fastapi + google-genai SDK | 106 MB |
| + retrieval index (704 chunks × 768d) | 124 MB |
| + MiniLM answer-cache encoder | 185 MB |
| + Kokoro TTS int8 | **323 MB** ← steady state |
| + PyAV (decodes browser audio, primary path) | 336 MB |
| + faster-whisper `tiny` fallback | 447 MB |
| + faster-whisper `base` fallback | 478 MB ← worst case |

For comparison, the old stack was ~1.5 GB, dominated by the 986 MB Qwen GGUF.

### Does it fit Render's free tier (512 MB)?

**It fits, but not with gapless speech. Those are two different questions.**

Everything except synthesis is cheap: the app, the retrieval index and the
answer-cache encoder together are ~185 MB. Synthesis is what costs, and
*continuous* synthesis costs more than one session:

| TTS config | RSS (TTS only) | RTF | gaps |
|---|---|---|---|
| 1 session, arena on | 378 MB | 1.40 | **3** |
| 2 sessions, arena on | 598 MB | 0.96 | 0 |
| 2 sessions, arena off | 336 MB | 1.21 | 2 |
| 3 sessions, arena off | 455 MB | 0.95 | 0 |

A single session cannot keep ahead of playback, and every configuration that can
needs ~450 MB or more for TTS alone. So on a 512 MB instance you must pick:

- **`TTS_WORKERS=1`** — fits, and *most* answers still sound perfect, because
  the fixed answers are prewarmed and the semantic cache serves repeats verbatim
  (34 ms for a whole stream). Only a genuinely novel question synthesizes live,
  and only that one sounds chunky.
- **`TTS_WORKERS=2` + `TTS_MEM_ARENA=0`** — 336 MB, mostly smooth, occasional
  short gap. The middle option.
- **Standard (2 GB)** — `TTS_WORKERS=2`, arena on, always gapless.

Also set **`STT_FALLBACK=0`** on 512 MB: the local Whisper fallback costs
111–155 MB, and an OOM-kill shows up as a 502 with no body, which the browser
then reports as a CORS error. Gemini still transcribes without it.

Three free-tier caveats that are not about memory:

- **It spins down when idle.** The first request after a sleep takes ~60 s to
  wake and can 502 past the proxy timeout. The frontend retries and recovers.
- **The disk is ephemeral.** The TTS cache is lost on every restart, so `PREWARM`
  re-synthesizes the fixed answers each cold start — free, but slow on a shared CPU.
- **Gemini's free tier caps embeddings at 1000/day**, and every *unique* question
  costs one. That is the real ceiling, not RAM. Repeats are free (in-process LRU
  plus the semantic answer cache), and a paid key removes the limit.

`starter` ($7, always-on, 512 MB) avoids the first two and is what `render.yaml`
sets.

Per-request cost is roughly **$0.0005** for a generated answer (~2500 prompt
tokens in, ~30 out) and a fraction of that for a transcription. Most traffic
costs nothing: the suggestion chips are answered at boot, and the semantic answer
cache serves paraphrases verbatim — so the TTS clips are already on disk too.
