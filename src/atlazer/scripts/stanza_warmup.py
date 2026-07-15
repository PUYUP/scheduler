"""
warmup.py
=========
Pre-download model Stanza (per-bahasa) & model sentence-embedding
SEBELUM aplikasi utama mulai menerima traffic.

Idempotent: kalau model sudah pernah ada di cache (volume Docker),
script ini TIDAK mendownload ulang -- hanya cek lalu langsung selesai.
Jadi aman dipanggil setiap kali container start.

Bahasa dikontrol lewat env var STANZA_LANGS (comma-separated), supaya
bisa nambah bahasa cukup dengan ubah docker-compose.yml + restart,
TANPA rebuild image.

Pemakaian:
    python stanza_warmup.py
    python stanza_warmup.py --lang id --lang en --lang fr
    python stanza_warmup.py --dry-run   # cek konfigurasi tanpa benar-benar download
"""

import argparse
import os
import sys


def _get_langs_from_env(default="id,en"):
    raw = os.environ.get("STANZA_LANGS", default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Warmup model Stanza & sentence-transformers")
    parser.add_argument(
        "--lang", action="append", default=None,
        help="Kode bahasa Stanza, bisa diulang (mis. --lang id --lang en). "
             "Kalau tidak diisi, ambil dari env var STANZA_LANGS."
    )
    parser.add_argument(
        "--embed-model", default=os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3"),
        help="Model sentence-embedding multi-bahasa (untuk mode semantic)."
    )
    parser.add_argument(
        "--skip-embedder", action="store_true",
        help="Lewati download model embedding (kalau cuma pakai mode simple, semantic=False)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Cuma print konfigurasi yang akan dipakai, tanpa benar-benar download apa pun."
    )
    args = parser.parse_args()

    langs = args.lang if args.lang else _get_langs_from_env()

    cache_root = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    hf_home = os.environ.get("HF_HOME", os.path.join(cache_root, "huggingface"))

    print("=== warmup config ===")
    print(f"  Bahasa Stanza     : {langs}")
    print(f"  Model embedding   : {'(dilewati)' if args.skip_embedder else args.embed_model}")
    print(f"  XDG_CACHE_HOME    : {cache_root}")
    print(f"  HF_HOME           : {hf_home}")
    print("======================")

    if args.dry_run:
        print("[dry-run] Tidak ada download yang dilakukan.")
        return 0

    import stanza
    for lang in langs:
        print(f"[warmup] Memastikan model Stanza '{lang}' tersedia di cache...")
        stanza.download(lang, verbose=False)  # skip otomatis kalau sudah ada di cache
        stanza.Pipeline(lang=lang, processors="tokenize", verbose=False)
        print(f"[warmup] Bahasa '{lang}' siap.")

    if not args.skip_embedder:
        print(f"[warmup] Memastikan model embedding '{args.embed_model}' tersedia di cache...")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(args.embed_model)  # skip otomatis kalau sudah ada di cache
        print("[warmup] Model embedding siap.")

    print("[warmup] Selesai. Semua model sudah siap dipakai dari cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
