# ─── BUILD STAGE ─────────────────────────────────────────────────────────────
# Compiler toolchain kept here — never ships to the final image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 1. Copy only the manifest first so pip layer is cached independently of src.
COPY pyproject.toml README.md ./

# 2. Install into an isolated prefix so we can copy just the tree to final.
RUN pip install --upgrade --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install .


# ─── FINAL IMAGE ─────────────────────────────────────────────────────────────
# Only runtime libraries, no compiler, no build cache, no pip cache.
FROM python:3.12-slim AS final

# Runtime system deps for: PyMuPDF, EasyOCR, unstructured, tesseract, pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libmagic1 \
        libjpeg-turbo8 \
        libpng16-16 \
        libtesseract-dev \
        tesseract-ocr \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Pull only the installed packages from builder — no compiler baggage.
COPY --from=builder /install /usr/local

# 4. Copy application source last (so code changes don't bust the dep layer).
COPY . .

# 5. Create unprivileged user BEFORE any filesystem ops that set ownership.
RUN groupadd -r celery && useradd -r -g celery celery \
    && mkdir -p /app/downloads /app/logs \
    && chown -R celery:celery /app

USER celery

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A celery_app.main inspect ping \
        -d "celery@$$HOSTNAME" --timeout 5 || exit 1

CMD ["celery", "-A", "celery_app.main", "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--queues=default,scrape,process,embed"]