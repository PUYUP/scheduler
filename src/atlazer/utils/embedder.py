"""
utils/embedder.py
─────────────────────────────
Provider-agnostic embedding interface.

Usage:
    embedder = get_embedder()
    vectors  = embedder.embed_batch(["text one", "text two"])

Supported providers (set EMBEDDING_PROVIDER in .env):
  • "local"   → sentence-transformers BAAI/bge-m3
  • "onnx"    → sentence-transformers BAAI/bge-m3 ONNX
  • "tei"     → sentence-transformers BAAI/bge-m3 TEI
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Dict, Any, Optional
import requests

import structlog

from atlazer.config.settings import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    model_name: str = ""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors, one per input text."""
        ...

    def embed_one(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def count_tokens(self, texts: List[str]) -> List[int]:
        """
        Return the real tokenizer-based token count for each input text.

        Default implementation raises NotImplementedError; subclasses that
        can access a tokenizer (local, onnx) or a tokenize endpoint (tei)
        should override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement count_tokens()"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Local / offline provider (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model_name = settings.local_embedding_model
        log.info("local_embedder.loading", model=self.model_name)
        self._model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device="cpu",
            truncate_dim=settings.truncate_dim,
        )
        log.info("local_embedder.ready", model=self.model_name)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        # normalize_embeddings=True → cosine similarity == dot product
        vectors = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()
        elapsed = time.perf_counter() - t0
        log.debug(
            "local_embedder.batch",
            count=len(texts),
            model=self.model_name,
            elapsed_s=round(elapsed, 3),
        )
        return vectors

    def count_tokens(self, texts: List[str]) -> List[int]:
        """
        Real token count using the model's own HuggingFace tokenizer,
        respecting the same max_seq_length/truncation the model itself
        applies during encode().
        """
        if not texts:
            return []
        encoded = self._model.tokenizer(
            texts,
            truncation=True,
            max_length=self._model.max_seq_length,
        )
        return [len(ids) for ids in encoded["input_ids"]]


class TEIEmbedder(BaseEmbedder):
    """Embedder yang manggil TEI (Text Embeddings Inference) server via HTTP."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        normalize: bool = True,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.base_url = (base_url or settings.tei_base_url).rstrip("/")
        self.model_name = model_name or settings.local_embedding_model
        self.normalize = normalize
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Request embeddings untuk satu batch teks ke endpoint /embed TEI."""
        payload = {
            "inputs": texts,
            "normalize": self.normalize,
            "truncate": True,
            "dimensions": settings.truncate_dim,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/embed",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                last_exc = e
                log.warning(
                    "tei_embedder.request_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        log.error("tei_embedder.request_failed_final", max_retries=self.max_retries)
        raise RuntimeError(
            f"TEI embed request failed after {self.max_retries} attempts"
        ) from last_exc

    def count_tokens(self, texts: List[str]) -> List[int]:
        """
        Real token count via TEI's own /tokenize endpoint, so it reflects
        exactly what the server used for embedding (same tokenizer,
        same truncation behavior).
        """
        if not texts:
            return []

        payload = {"inputs": texts, "add_special_tokens": True}

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/tokenize",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                # TEI returns a list (per input) of lists of token objects
                return [len(tokens) for tokens in response.json()]
            except requests.exceptions.RequestException as e:
                last_exc = e
                log.warning(
                    "tei_embedder.tokenize_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        log.error("tei_embedder.tokenize_failed_final", max_retries=self.max_retries)
        raise RuntimeError(
            f"TEI tokenize request failed after {self.max_retries} attempts"
        ) from last_exc


class LocalONNXEmbedder(BaseEmbedder):
    """
    Local embedder menggunakan sentence-transformers dengan ONNX Runtime backend.

    Asumsi: model ONNX SUDAH di-export sebelumnya oleh init step
    (scripts/export_onnx.py) ke settings.onnx_cache_dir. Class ini
    HANYA load, tidak pernah export — export saat runtime aplikasi utama
    dihindari supaya startup cepat & konsisten antar replica.
    """

    def __init__(self):
        from pathlib import Path
        from sentence_transformers import SentenceTransformer

        self.model_name = settings.local_embedding_model
        cache_dir = Path(settings.onnx_cache_dir) / self.model_name.replace("/", "__")

        if not cache_dir.exists():
            raise RuntimeError(
                f"ONNX cache tidak ditemukan di {cache_dir}. "
                "Jalankan `python -m scripts.export_onnx` (atau init "
                "container/init step) sebelum start aplikasi."
            )

        provider = getattr(settings, "onnx_provider", "CPUExecutionProvider")
        device = "cuda" if provider.startswith("CUDA") else "cpu"

        log.info("local_onnx_embedder.loading", path=str(cache_dir))
        t0 = time.perf_counter()
        self._model = SentenceTransformer(
            model_name_or_path=str(cache_dir),
            backend="onnx",
            device=device,
            truncate_dim=settings.truncate_dim,
            model_kwargs={"provider": provider},
        )
        log.info(
            "local_onnx_embedder.ready",
            model=self.model_name,
            provider=provider,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        vectors = self._model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        log.debug(
            "local_onnx_embedder.batch",
            count=len(texts),
            elapsed_s=round(time.perf_counter() - t0, 3),
        )
        return vectors

    def count_tokens(self, texts: List[str]) -> List[int]:
        """
        Real token count using the same HuggingFace tokenizer bundled
        with the ONNX-backed SentenceTransformer.
        """
        if not texts:
            return []
        encoded = self._model.tokenizer(
            texts,
            truncation=True,
            max_length=self._model.max_seq_length,
        )
        return [len(ids) for ids in encoded["input_ids"]]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """
    Returns a cached embedder instance (one per worker process).
    Avoids re-loading the model on every task call.
    """
    provider = settings.embedding_provider.lower()

    if provider == "local":
        log.info("embedder.init", provider="local", model=settings.local_embedding_model)
        return LocalEmbedder()

    elif provider == "tei":
        log.info("embedder.init", provider="tei", base_url=settings.tei_base_url)
        return TEIEmbedder()
    
    elif provider == "onnx":
        log.info("embedder.init", provider="onnx", model=settings.local_embedding_model)
        return LocalONNXEmbedder()

    else:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            "Choose 'local', 'tei', or 'onnx'."
        )


def chunks_to_vector(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert chunks to vectors.

    Args:
        chunks: List of chunk dicts. Each dict must contain a "text" key
            (str) whose value will be embedded. Other keys are preserved
            as-is in the output.

    Returns:
        List of chunks (same dicts, copied) with additional keys:
            - "embedding": the embedding vector (List[float]).
            - "embedding_model": name of the model used to generate it.
            - "embedding_dim": dimensionality of the embedding vector.
            - "token_count": real tokenizer-based token count of the
              chunk's text, if the active embedder supports counting
              (see BaseEmbedder.count_tokens). Omitted otherwise.

    Raises:
        KeyError: if any chunk is missing the "text" key.
        Exception: propagates any error raised by the embedder
            (e.g. API/network failures) after logging it.
    """

    log.info("chunks_to_vector.start", chunks=len(chunks))

    embedder = get_embedder()
    batch_size = settings.embedding_batch_size
    embedded_chunks: List[Dict[str, Any]] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        texts = [c["text"] for c in batch]

        t0 = time.perf_counter()
        try:
            vectors = embedder.embed_batch(texts)
        except Exception:
            log.exception(
                "embedder.batch_failed",
                batch_start=batch_start,
                batch_size=len(batch),
            )
            raise

        elapsed = time.perf_counter() - t0
        log.debug(
            "embedder.batch_done",
            batch_start=batch_start,
            batch_size=len(batch),
            elapsed_s=round(elapsed, 2),
        )

        try:
            token_counts = embedder.count_tokens(texts)
        except NotImplementedError:
            token_counts = [None] * len(texts)
        except Exception:
            log.exception(
                "embedder.token_count_failed",
                batch_start=batch_start,
                batch_size=len(batch),
            )
            token_counts = [None] * len(texts)

        for chunk, vector, text, token_count in zip(batch, vectors, texts, token_counts):
            chunk_with_vec = chunk.copy()
            chunk_with_vec["embedding"] = vector
            chunk_with_vec["embedding_model"] = embedder.model_name
            chunk_with_vec["embedding_dim"] = len(vector)
            chunk_with_vec["word_count"] = len(text.split())
            if token_count is not None:
                chunk_with_vec["token_count"] = token_count
            embedded_chunks.append(chunk_with_vec)

    return embedded_chunks
