# Knowledge base — grounded answering over shasuncollege.edu.in

SIA answers open-ended questions from the college website instead of a fixed
keyword table. This document covers the retrieval design, the security model,
and the latency constraint that shapes both.

```
  scrape             index                  answer
  ──────             ─────                  ──────
  WP REST API   ──►  chunk (800 ch)     ──►  semantic answer cache ──► hit: verbatim
  + HTML scrape      BGE-small ONNX 384d     miss ─┐
  + sanitize         BM25 postings                 ├─► hybrid retrieve (RRF)
                     810 chunks / 1.24 MB          └─► Qwen2.5-1.5B, sources untrusted
                                                        └─► output guard ──► speak
```

## Why there is no vector database

Measured, not assumed:

| | |
|---|---|
| Documents scraped | 364 |
| Corpus | 3.35M chars ≈ **838K tokens** |
| Chunks @ 800 chars | **810** |
| Vectors as float32 (810 × 384) | **1.24 MB** |

At that size, exact search is a single `(N, 384) @ (384,)` matmul — a fraction of
a millisecond, and *exact*. Approximate indexes (HNSW, IVF) start paying off
around 100K+ vectors; here one would be slower to build and only approximate. A
separate vector service would add a process, a network hop, and a backup story
in exchange for nothing.

The index lives in `kb-data/` as `vectors.npy` + `chunks.jsonl`, rebuilds from
the corpus in seconds, and loads into RAM at startup next to the TTS cache.

**Revisit this** above roughly 50K chunks, if retrieval needs per-tenant
filtering, or if several replicas must share one index.

### Retrieval is hybrid, and that matters more than the store

Dense embeddings handle paraphrase — *"where do students live"* retrieves the
hostel page with no shared words. They are weak exactly where a college site is
dense: proper nouns and programme codes (`B.Com Corporate Secretaryship`,
`SAYE`, `NAAC`, `M.A. Journalism`), which get blurred into neighbours.

So both run, and the two rankings merge with **Reciprocal Rank Fusion**
(`1/(60 + rank)`). RRF works on rank position rather than score, so the two
scales never need normalizing against each other, and there is nothing to tune.

Embeddings are `bge-small-en-v1.5`, int8 ONNX, 384-dim, ~34 MB — running on the
ONNX Runtime that Kokoro already loads. No PyTorch, no `sentence-transformers`.
Search is **~4 ms** end to end.

BGE is trained for *question-to-passage* search, which is what this is.
`all-MiniLM-L6-v2` is trained for symmetric sentence similarity and measured
markedly worse here (recall 33% vs 47% on the same corpus) — it compresses short
questions into a narrow band, which is also what made the relevance gate so hard
to place. MiniLM is still loaded, but only for the answer cache, which genuinely
is a symmetric question-to-question comparison.

## Scraping

The site exposes an unauthenticated **WordPress REST API**, and that is the
ingest path. It matters more than convenience: extracting the same pages from
rendered HTML measured **94.9% boilerplate** — every page carries ~39K chars of
identical nav and footer, so a naive scrape would have filled the index with 200
near-identical copies of the menu and buried the actual answers.
`content.rendered` has none of it.

In practice the API covers less than it appears to: **130 of 203 pages return
nothing but their in-page nav widget**, because Elementor keeps page layout in
postmeta rather than in `content.rendered`. Those are detected (`is_real_content`
requires real sentences) and sent to the HTML fallback, along with the
`cool_timeline` post type, which is not exposed over REST at all.

The HTML path selects the content region **structurally** — dropping `nav`,
`header`, `footer`, and any list whose text is almost entirely links — rather
than subtracting text that repeats across pages. Frequency subtraction was tried
first as the primary mechanism and was far too blunt: it cut the NAAC page to 205
characters and the milestones page to zero.

Two details there are easy to get wrong and both cost real content:

- Inactive Elementor tab panels carry `hidden`, but their text is one click away,
  not concealed. Treating them as hidden lost the NAAC accreditation grades.
- Not every page is prose. The NAAC page is 317 characters of short lines
  (`A++ Grade (2023)`) with zero sentences of ten words or more; requiring prose
  on the HTML path silently dropped it, and no retrieval tuning can find a page
  that was never indexed.

```bash
python -m kb.ingest      # -> kb-data/corpus.jsonl, kb-data/quarantine.jsonl
python -m kb.index       # -> kb-data/vectors.npy, kb-data/chunks.jsonl
python -m kb.index --probe "how do I apply"
```

Re-run both after the site changes. Nothing else needs restarting except the
service, which reloads the index at boot.

## The answering model runs locally

Default backend is **Qwen2.5-1.5B-Instruct (Q4_K_M, 986 MB)** on llama.cpp, CPU
only. No API key, no network at request time. `LLM_BACKEND=anthropic` switches to
the Claude API if you ever want the quality ceiling instead.

The model was picked by benchmarking candidates on *this* task — grounded
answering over retrieved college-site chunks — not on general leaderboards:

| model | size | grounded fact | refuses politely | prompt injection |
|---|---|---|---|---|
| **Qwen2.5-1.5B** | 986 MB | correct (`A++`) | yes | **resisted** |
| Llama-3.2-1B | 808 MB | correct | hedged | **replied "OWNED"** ✗ |
| Qwen2.5-0.5B | 398 MB | **wrong** (`A`, a 2013 value) ✗ | yes | resisted |

Both rejections matter. Llama-3.2-1B obeyed an instruction planted inside a
`<source>` block — disqualifying for a system whose corpus is scraped from a CMS.
Qwen2.5-0.5B stayed safe but answered "A grade" when the current accreditation is
**A++**, picking a decade-old value out of the same chunk: confidently wrong,
which is worse than saying "I don't know".

> **Small models resist prompt injection far less reliably than frontier models.**
> That is the real cost of running locally. It does not change the architecture,
> but it does mean the guard layers below carry more weight than they would
> behind a large model — particularly the output check, which is the last thing
> between the model and the speaker.

### Latency, and what it forces

CPU inference here is dominated by **prefill**, ~7 ms per prompt token. That, not
generation, is what caps how much context the retriever may pass:

| chunks | prompt tokens | time to answer |
|---|---|---|
| 2 | 649 | 5.8 s |
| 3 | 920 | 6.4 s |
| 6 | 1803 | 12.2 s |
| 8 | 2182 | 14.8 s |

So the local path retrieves **3 chunks at 700 chars** where the API path would
take 6 at 1000 — `answering.py` sets these per backend. Thread and batch tuning
recovered a little more (5.9 s → 4.8 s at 12 threads / `n_batch=2048`), but not a
different order of magnitude.

#### The KV cache does most of the work

The system prompt is ~457 tokens of *constant* text sitting at the front of every
request — about 3.2 s of prefill, re-paid on each question for no reason.
llama.cpp already keeps the KV cache from the previous call and reuses whatever
prefix the next one shares, so simply never calling `llm.reset()` skips it:

| | time to answer |
|---|---|
| with `reset()` between calls | 3.16 s |
| without (prefix reused) | **0.89 s** — 3.6x |

Two things are needed to actually collect that. `generate()` must not reset (see
the warning in `llm_local.py`), and `warm()` must prime with the **real** system
prompt — priming with any other text loads the weights but leaves the first real
question paying full prefill anyway. Measured after both: 3.5 s once at boot,
then **0.7–1.7 s** per answer.

The `_gen_lock` is what makes this safe. The KV cache is single-sequence shared
state, so two overlapping generations would corrupt each other's prefix.

#### A shorter prompt, and a canonical refusal

The system prompt is written twice. The long version (457 tokens) targets a
frontier model; `LOCAL_SYSTEM_PROMPT` (257 tokens) says the same things in
imperative form with no rationale, because a 1.5B model follows a short concrete
list more reliably than a long one whose later clauses get diluted. Measured on
the same four cases: **equal accuracy, 28% faster** (1.37 s vs 1.76 s median).
The injection paragraph is kept nearly intact even so — it is the one section
where dropping a clause has a security consequence rather than a stylistic one.

The prompt also asks for a **canonical refusal sentence**. Detecting "I couldn't
answer" by matching the model's prose kept failing one wording at a time —
`does not contain` was handled while `does not provide` and the plural `sources
do not contain` were not — and every gap let a refusal be stored as a real answer
and cached, pinning "I don't know" to a question that might work after the next
scrape. A fixed sentence makes the common path an exact match; `guard.py` keeps a
regex *family* (not a string list) as the backstop for when the model
improvises. As a bonus every refusal now shares one TTS cache entry, so its audio
is always already on disk.

Two further things keep cost off the common path entirely:

- **The semantic answer cache** — repeat and reworded questions are served
  verbatim from cache, which also means their audio is already synthesized.
- **Answer prewarm at boot** — the suggestion chips shown on screen are answered
  and synthesized at startup, so tapping one is instant. Same trick the fixed
  knowledge base used, extended to generated answers.

## Measuring accuracy

`tests/eval_accuracy.py` scores 15 questions with ground truth verified against
the live site, plus off-topic negatives:

```bash
python -m tests.eval_accuracy            # retrieval only — fast, no model
python -m tests.eval_accuracy --answers  # + end-to-end answer quality
```

It reports a **corpus ceiling** first — how many questions are answerable with
perfect retrieval. Without that number the two failure modes are
indistinguishable, and tuning the retriever against a page that was never
scraped is wasted work. That distinction drove most of the gains below.

| | recall | ceiling |
|---|---|---|
| over-filtered corpus, MiniLM | 33% | 87% |
| BGE embeddings | 47% | 87% |
| structure-aware scrape (364 docs) | 73% | 93% |
| prose filter fixed (NAAC recovered) | 73% | **100%** |
| chunk size 1000 → 800 | **80%** | 100% |

End to end with the local model: **93% answer accuracy (14/15)** and **5/5
off-topic questions declined**. The single miss answers "is the college
accredited" with its ISO 9001 certification instead of its NAAC grade — a
defensible answer the eval did not expect, not a fabrication.

Two things that measurably did **not** work, recorded so they are not retried
blind: a third BM25 ranker over titles (recall 47% → 40%, and → 33% with `b=0`,
because two-word titles let one shared common term dominate), and every
score-based off-topic gate (see below).

## Debugging an answer

`DEBUG=1` traces the whole pipeline to stderr — retrieval scores per chunk, the
**exact** system and user turns sent to the model, the **raw** output before any
guard touches it, and what each guard decided.

```bash
DEBUG=1 uvicorn app:app --port 8000
DEBUG=1 DEBUG_FULL_PROMPT=1 uvicorn app:app --port 8000   # don't truncate chunks
```

The prompt and raw output are the two that matter. A grounded answer is a
function of the retrieved text, so reading that text beside the reply is the
fastest way to separate a *retrieval* failure (the right page never reached the
prompt) from a *generation* failure (it did, and the model still got it wrong).
From outside, those look identical. Every call site is guarded on the flag, so
the tracing costs nothing when off.

## Security model

**The scraped site is the injection surface.** This is the threat that actually
matters here, and it is not the one people usually defend against. Anyone who
can get text onto that WordPress install — a compromised plugin, a submitted
event listing, an uploaded page — can plant instructions that land in the
model's context as though the operator wrote them. The visitor typing in the
chat box is the *lesser* risk; they can only mislead themselves.

Defense is layered, because no single layer is trustworthy:

| Layer | File | What it does |
|---|---|---|
| **Ingest** | `sanitize.py` | Drops everything a sighted visitor cannot see — HTML comments, `display:none` / `visibility:hidden` / `opacity:0` / offscreen / `aria-hidden` nodes, `<script>`. Strips zero-width and bidi characters. NFKC-normalizes, so fullwidth and math-styled glyphs cannot evade downstream checks. |
| **Quarantine** | `ingest.py` | Pages carrying injection-shaped phrasing are written to `quarantine.jsonl` instead of being indexed, so a compromised page is visible rather than silently trusted. |
| **Prompt** | `prompts.py` | Retrieved text goes in the **user turn**, never the system prompt, wrapped in `<source>` tags with angle brackets neutralized so a chunk cannot forge its own closing tag. The question is placed *after* the sources, so the last instruction the model reads is the operator's. |
| **Request** | `guard.py` | Length cap, Unicode normalization, per-IP rate limit, and refusal of prompt-extraction attempts. |
| **Response** | `guard.py` | Fails closed on any sign of prompt leakage, caps answer length, flags ungrounded replies. |

Underneath all of it, the structural defense: **the model is given no tools.**
Its entire action space is emitting text. A successful injection can produce a
wrong sentence; it can never take an action, read a file, or reach the network.

Sanitizing is treated as raising the cost of an attack, never as proof that a
chunk is safe — which is why the prompt layer still treats every chunk as
hostile.

```bash
python -m tests.test_security          # 53 offline checks
python -m tests.test_security --live   # + the model's own resistance (needs a key)
```

## The latency constraint

Generation is not the bottleneck in this service — **synthesis is.** Kokoro runs
at ~0.4x real time, so a fresh 40-second answer costs ~16s of CPU before the
avatar makes a sound. The TTS disk cache is keyed on exact answer text, which is
what let the fixed knowledge base be prewarmed and served in ~25 ms.

Free-form answers would defeat that outright: two visitors asking the same thing
in different words produce two different strings, so every reply is a TTS cache
miss. Three things hold the line:

1. **`answer_cache.py`** matches questions by *meaning* (cosine ≥ 0.93) and
   replays the previous answer **verbatim** — identical bytes, so the same TTS
   cache key, so instant audio. A hit here is worth much more than the model
   call it saves.
2. **A hard answer cap** (`MAX_ANSWER_CHARS = 700`, and a 2–4 sentence
   instruction) keeps a miss bounded — and is better voice UX regardless.
3. **`POST /tts/stream`** synthesizes sentence by sentence and streams each clip
   as it is ready, so playback starts on sentence one while the rest renders.

Measured on a fully uncached answer (every sentence unique, so nothing could hit
the clip cache):

| | time to first audio |
|---|---|
| `POST /tts` (whole answer) | 8.76 s |
| `POST /tts/stream` | **4.47 s** — 2.0x faster (1.7 s when the vocoder is hot) |

Playback stayed **gapless**: sentence two arrived at 7.15 s while sentence one
played until 9.53 s. Because the real-time factor is well below 1, synthesis
keeps pulling ahead — 35 s of audio was fully rendered in 10 s.

Two details that had to be right for this to work, both found by measuring:

- **Sentence splitting must not break abbreviations.** This content is full of
  `B.Com`, `B.Sc`, `M.A.`, `Dr.` — naive splitting on `.` shatters a programme
  name into unspeakable fragments. It must also be *stable*, since each sentence
  is a cache key.
- **A very short opening clip backfires.** Letting `"Good question."` (1.2 s)
  stand as its own clip looks like a latency win, but playback of clip N has to
  cover synthesis of clip N+1 — against a 9 s follow-up that left a 1.1 s hole
  mid-sentence. Every clip now sits above a floor, and only the *first* gets a
  tighter ceiling, since it is the one the listener actually waits on.

## Configuration

| Var | Default | Notes |
|---|---|---|
| `LLM_BACKEND` | `local` | `local` (llama.cpp, no network) or `anthropic`. |
| `LLM_MODEL_PATH` | `models/llm/qwen2.5-1.5b.gguf` | Any chat-tuned GGUF. |
| `LLM_THREADS` | `min(12, cores)` | |
| `LLM_BATCH` | `2048` | Larger batches speed up prefill, the bottleneck. |
| `LLM_CTX` | `4096` | |
| `RETRIEVAL_TOP_K` | `3` local / `6` api | Chunks passed to the model. |
| `RETRIEVAL_CHUNK_CHARS` | `700` local / `1000` api | Per-chunk truncation. |
| `RETRIEVAL_MIN_SIMILARITY` | `0.45` | Dense cosine floor; below it, say "I don't know". |
| `KB_ENABLED` | `1` | `0` reverts to the keyword knowledge base. |
| `EMBED_MODEL` | `bge` | `bge` / `multiqa` / `minilm`. BGE measured best for question-to-passage search. |
| `DEBUG` | `0` | `1` traces retrieval, the exact prompt, and the raw model output. |
| `DEBUG_FULL_PROMPT` | `0` | `1` stops truncating chunk bodies in the trace. |
| `ANTHROPIC_API_KEY` | — | Only for `LLM_BACKEND=anthropic`. |
| `ANSWER_MODEL` | `claude-opus-5` | Only for `LLM_BACKEND=anthropic`. |

### Resource footprint

| | resident |
|---|---|
| Kokoro TTS (fp32) | ~325 MB |
| Whisper `base` (int8) | ~150 MB |
| Qwen2.5-1.5B (Q4_K_M) | ~1.0 GB |
| MiniLM embeddings (int8) | ~23 MB |
| Retrieval index (~500 chunks) | ~0.8 MB |

Roughly **1.5 GB** all-in, entirely on CPU, with no per-request cost and nothing
leaving the machine. Pin `KOKORO_MODEL=kokoro-v1.0.int8.onnx` to trade ~230 MB
for slower synthesis, or `LLM_BACKEND=anthropic` to drop the 1 GB model
entirely.
