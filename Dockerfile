FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TTS_CACHE_DIR=/tmp/tts-cache

# espeak-ng: grapheme→phoneme for Kokoro. ffmpeg/libsndfile: audio decode/encode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Every dependency now ships a wheel. The custom index and the build-essential
# fallback that used to live here existed only for llama-cpp-python, which built
# from source; generation moved to Gemini, so both are gone and so is the
# toolchain they needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the model weights into the image (survives cold starts): Kokoro + Whisper
# for voice, and the embedding + generation models for the knowledge base.
COPY download_models.py .
RUN python download_models.py

COPY . .

# The retrieval index is built locally and copied in, NOT built here.
#
# It used to be built at image-build time, which worked while embeddings were a
# local ONNX model. They are a metered API call now, so building here would mean
# every image build needed a live key, spent real quota, and could fail on a 429
# — for an artifact that only changes when the corpus does. Two commands, run
# locally whenever the site is re-scraped:
#
#     python -m kb.ingest     # scrape -> kb-data/corpus.jsonl (~40 min)
#     python -m kb.index      # embed  -> vectors.npy + chunks.jsonl
#
# Both outputs are committed, so this image is reproducible without a key.
# Without them the service still boots and answers from the curated keyword
# knowledge base.
RUN if [ -f kb-data/vectors.npy ] && [ -f kb-data/chunks.jsonl ]; then \
        echo "[build] retrieval index present: $(wc -l < kb-data/chunks.jsonl) chunks"; \
    else \
        echo "[build] no prebuilt index - falling back to the keyword knowledge base"; \
    fi

# Render provides $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
