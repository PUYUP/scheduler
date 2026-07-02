import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Wrapper embedding menggunakan sentence-transformers dengan model BGE-M3.

    Catatan penting BGE-M3:
    - Untuk QUERY, sebaiknya ditambahkan instruction prefix supaya hasil
      retrieval lebih relevan.
    - Untuk DOKUMEN/PASSAGE, tidak perlu instruction prefix.
    - Embedding dinormalisasi (L2 norm) supaya inner product == cosine
      similarity, sehingga bisa dipakai langsung dengan FAISS IndexFlatIP.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, cache_dir: str, device: str = "cpu"):
        # Pastikan sentence-transformers & huggingface_hub membaca cache
        # dari folder yang sama (folder .cache yang di-share via volume).
        os.environ.setdefault("HF_HOME", cache_dir)

        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device=device,
        )
        self.dimension: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        if is_query:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.astype("float32")
