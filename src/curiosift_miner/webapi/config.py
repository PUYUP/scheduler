import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model embedding yang dipakai (sentence-transformers compatible)
    MODEL_NAME: str = "BAAI/bge-m3"

    # Lokasi cache HuggingFace di dalam container.
    # Di docker-compose.yml, path ini di-mount ke folder .cache di host
    # supaya model tidak perlu didownload ulang & bisa dipakai bareng
    # service lain.
    HF_HOME: str = os.environ.get("HF_HOME", "/root/.cache/huggingface")

    # "cpu" atau "cuda" (kalau container punya akses GPU)
    DEVICE: str = os.environ.get("DEVICE", "cpu")

    # Konfigurasi chunking dokumen (dalam karakter)
    CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", 50))

    # Lokasi penyimpanan index FAISS + metadata (persisted di volume)
    VECTOR_STORE_PATH: str = os.environ.get(
        "VECTOR_STORE_PATH", "/app/data/vector_store"
    )

    DEFAULT_TOP_K: int = int(os.environ.get("DEFAULT_TOP_K", 5))

    class Config:
        env_file = ".env"


settings = Settings()
