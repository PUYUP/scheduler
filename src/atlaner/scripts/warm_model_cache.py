"""
scripts/warm_model_cache.py
─────────────────────────────
One-shot job: download the local embedding model into the HuggingFace
cache directory (HF_HOME), which docker-compose mounts as a shared named
volume across all worker/beat/flower containers.

Run ONCE before any worker starts — see `model-init` service in
docker-compose.yml. Workers themselves never need network access to
HuggingFace Hub; they just read from the already-populated volume.
"""

from __future__ import annotations

import os
import sys

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")


def main() -> None:
    print(f"[warm_model_cache] target model: {MODEL_NAME}")
    print(f"[warm_model_cache] HF_HOME: {os.environ.get('HF_HOME', '(default)')}")

    try:
        SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as exc:  # noqa: BLE001 — this is a startup gate, fail loud
        print(f"[warm_model_cache] FAILED to download '{MODEL_NAME}': {exc}", file=sys.stderr)
        sys.exit(1)

    print("[warm_model_cache] done — model cached and ready for workers.")


if __name__ == "__main__":
    main()
