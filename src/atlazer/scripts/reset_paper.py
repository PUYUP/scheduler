"""
scripts/reset_paper.py
────────────────────────
CLI kecil untuk keperluan testing/development:
reset status dedup satu atau beberapa paper, tanpa perlu masuk redis-cli manual.

Usage:
  docker compose exec worker python -m scripts.reset_paper 2401.12345 2402.00001
  docker compose exec worker python -m scripts.reset_paper --all --repository arxiv
  docker compose exec worker python -m scripts.reset_paper --show 2401.12345
"""

from __future__ import annotations

import argparse
import sys

from atlazer.utils.dedup import (
    reset_paper,
    is_already_processed,
    count_processed,
    count_queued,
)


def _flush_all(repository: str) -> None:
    """Hapus SEMUA entry dedup untuk repository ini. Dipakai untuk testing saja."""
    from atlazer.utils.dedup import _get_redis  # noqa: internal, testing-only use

    confirm = input(
        f"Ini akan menghapus SEMUA dedup key untuk repository='{repository}'. "
        f"Ketik 'yes' untuk lanjut: "
    )
    if confirm.strip().lower() != "yes":
        print("Dibatalkan.")
        return

    r = _get_redis()
    r.delete(f"atlazer_rag:{repository}:processed")
    r.delete(f"atlazer_rag:{repository}:queued")
    for key in r.scan_iter(match=f"atlazer_rag:{repository}:queued:*"):
        r.delete(key)
    print(f"Selesai. Semua dedup key untuk repository='{repository}' sudah dihapus.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset dedup status untuk testing.")
    parser.add_argument("paper_ids", nargs="*", help="Paper ID yang mau di-reset, mis. 2401.12345")
    parser.add_argument("--repository", default="arxiv", help="Default: arxiv")
    parser.add_argument("--all", action="store_true", help="Flush semua dedup key untuk repository ini")
    parser.add_argument("--show", metavar="PAPER_ID", help="Cek status satu paper tanpa mengubah apa pun")
    args = parser.parse_args()

    if args.show:
        status = "processed/queued" if is_already_processed(args.show, args.repository) else "belum tercatat"
        print(f"{args.show} ({args.repository}): {status}")
        return

    if args.all:
        _flush_all(args.repository)
        return

    if not args.paper_ids:
        print("Tidak ada paper_id diberikan. Pakai --all untuk flush semua, atau --show untuk cek status.")
        sys.exit(1)

    for paper_id in args.paper_ids:
        reset_paper(paper_id, repository=args.repository)
        print(f"Reset: {paper_id} ({args.repository})")

    print(
        f"\nRingkasan {args.repository} — processed: {count_processed(args.repository)}, "
        f"queued: {count_queued(args.repository)}"
    )


if __name__ == "__main__":
    main()