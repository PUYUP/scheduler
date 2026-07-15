# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc g++ libgl1 libglib2.0-0 libgomp1 \
        poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# ─── FINAL IMAGE ─────────────────────────────
# Model TIDAK dibake di sini lagi — dipindah ke shared named volume,
# diisi sekali oleh service `model-init` di docker-compose.yml.
FROM base AS final

WORKDIR /app

# 1. Create user early
RUN groupadd -r atlazer && useradd -r -g atlazer -d /home/atlazer atlazer

# 2. KUNCI UTAMA: Paksa pip memprioritaskan repository versi CPU (Hemat ~2GB+)
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

# 3. Copy file konfigurasi project
COPY pyproject.toml README.md ./ 

# 4. TRIK CACHING: Buat direktori dummy untuk mengelabui setuptools.
# setuptools butuh direktori 'src/atlazer/celery_app' ada saat membaca pyproject.toml.
# Dengan ini, kita bisa menginstal dependensi tanpa harus mencopy seluruh source code (COPY . .) dulu.
RUN mkdir -p src/atlazer/celery_app \
    && touch src/atlazer/__init__.py \
    && touch src/atlazer/celery_app/__init__.py

# 5. Install dependencies dengan BuildKit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install -e . 

# 6. BARU copy seluruh source code asli Anda (ini akan menimpa direktori dummy di atas)
COPY . .

# 7. Siapkan direktori cache HF — akan ditimpa oleh named volume saat runtime,
# tapi tetap dibuat & di-chown di sini supaya kepemilikan awal volume benar
# saat Docker pertama kali inisialisasi volume kosong dari isi image ini.
ENV HF_HOME=/home/atlazer/.cache/huggingface
ENV STANZA_HOME=/home/atlazer/.cache/stanza
ENV ONNX_CACHE_DIR=/app/data/onnx_cache

RUN mkdir -p /app/downloads /app/logs /app/data/onnx_cache /home/atlazer/.cache/huggingface /home/atlazer/.cache/stanza \
    && chown -R atlazer:atlazer /app/downloads /app/logs /app/data/onnx_cache \
    && chown -R atlazer:atlazer /home/atlazer/.cache/huggingface /home/atlazer/.cache/stanza

ENV HOME=/home/atlazer

USER atlazer

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A atlazer.celery_app.main inspect ping -d "atlazer@$HOSTNAME" --timeout 5 || exit 1

CMD celery -A atlazer.celery_app.main worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=default,scrape,process,embed,store,webapi