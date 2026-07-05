"""
scripts/export_onnx.py
─────────────────────────────
Pre-export sentence-transformers model ke ONNX, disimpan ke cache dir
di volume. Didesain untuk dijalankan sebagai init container/init step,
SEBELUM aplikasi utama start.

Idempotent & race-safe:
  - Kalau cache sudah lengkap → langsung exit, tidak export ulang.
  - Export dilakukan ke temp dir dulu, baru di-rename atomic ke path final.
    Ini mencegah aplikasi utama membaca folder cache yang isinya
    setengah jadi (kalau proses export crash di tengah jalan), dan
    aman walau beberapa replica kebetulan export bersamaan (siapa yang
    selesai duluan, itu yang "menang" — hasil akhirnya tetap valid).

Usage:
    python -m scripts.export_onnx
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import structlog

from atlazer.config.settings import settings

log = structlog.get_logger(__name__)


def _cache_path(model_name: str) -> Path:
    return Path(settings.onnx_cache_dir) / model_name.replace("/", "__")


def _is_cache_complete(path: Path) -> bool:
    """Cek minimal file yang wajib ada supaya dianggap 'export selesai'."""
    if not path.exists():
        return False
    required = ["config.json", "onnx"]  # folder 'onnx/' berisi model.onnx
    return all((path / r).exists() for r in required)


def export_model(model_name: str) -> Path:
    final_path = _cache_path(model_name)

    if _is_cache_complete(final_path):
        log.info("export_onnx.cache_hit", path=str(final_path))
        return final_path

    log.info("export_onnx.cache_miss_exporting", model=model_name, target=str(final_path))

    # Export ke temp dir unik dulu, supaya proses lain yang baca final_path
    # tidak pernah lihat folder setengah jadi.
    tmp_path = final_path.parent / f".tmp-{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_name_or_path=model_name,
            backend="onnx",
            device="cpu",  # export tidak perlu GPU
            truncate_dim=settings.truncate_dim,
        )
        model.save_pretrained(str(tmp_path))
    except Exception:
        log.exception("export_onnx.failed", model=model_name)
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    elapsed = time.perf_counter() - t0

    # Atomic rename. Kalau final_path sudah dibuat proses lain barusan
    # (race antar replica), pakai punya mereka saja — buang punya kita.
    try:
        os.rename(tmp_path, final_path)
        log.info("export_onnx.done", path=str(final_path), elapsed_s=round(elapsed, 2))
    except OSError:
        if _is_cache_complete(final_path):
            log.info("export_onnx.lost_race_reusing_existing", path=str(final_path))
            shutil.rmtree(tmp_path, ignore_errors=True)
        else:
            raise

    return final_path


def main() -> None:
    model_name = settings.local_embedding_model
    try:
        export_model(model_name)
    except Exception as e:
        log.error("export_onnx.fatal", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()