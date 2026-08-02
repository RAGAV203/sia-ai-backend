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

# llama-cpp-python has no wheel on PyPI and builds from source, which needs a
# toolchain. Prefer the maintainer's prebuilt CPU wheels and fall back to a
# source build only if that index has nothing for this platform.
COPY requirements.txt .
RUN pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        -r requirements.txt \
 || (apt-get update \
     && apt-get install -y --no-install-recommends build-essential cmake \
     && pip install -r requirements.txt \
     && apt-get purge -y build-essential cmake && apt-get autoremove -y \
     && rm -rf /var/lib/apt/lists/*)

# Bake the model weights into the image (survives cold starts): Kokoro + Whisper
# for voice, and the embedding + generation models for the knowledge base.
COPY download_models.py .
RUN python download_models.py

COPY . .

# Build the retrieval index at image-build time when a scraped corpus is present.
#
# The scrape itself is deliberately NOT run here: it takes ~40 minutes against
# the live site and would make every image build depend on that site being up.
# Run `python -m kb.ingest` locally, keep the resulting kb-data/corpus.jsonl, and
# it gets baked in here. Without it the service still boots and answers from the
# curated keyword knowledge base.
RUN if [ -f kb-data/corpus.jsonl ]; then \
        python -m kb.index && echo "[build] retrieval index ready"; \
    else \
        echo "[build] no kb-data/corpus.jsonl - falling back to the keyword knowledge base"; \
    fi

# Render provides $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
