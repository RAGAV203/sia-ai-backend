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

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the model weights into the image (survives cold starts).
COPY download_models.py .
RUN python download_models.py

COPY . .

# Render provides $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
