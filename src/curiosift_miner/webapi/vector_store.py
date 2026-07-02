import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np


class VectorStore:
    """
    Vector store sederhana berbasis FAISS (IndexFlatIP untuk cosine
    similarity, karena embedding sudah dinormalisasi).

    Index & metadata di-persist ke disk (folder /app/data via volume)
    supaya data tidak hilang saat container di-restart.
    """

    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_file = f"{index_path}.faiss"
        self.meta_file = f"{index_path}.meta.pkl"

        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "rb") as f:
                self.metadatas: List[Dict[str, Any]] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadatas = []

    @property
    def total_documents(self) -> int:
        return self.index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in texts]
        self.index.add(embeddings.astype("float32"))
        for text, meta in zip(texts, metadatas):
            entry = dict(meta)
            entry["text"] = text
            self.metadatas.append(entry)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = self.metadatas[idx]
            results.append(
                {
                    "text": entry["text"],
                    "score": float(score),
                    "metadata": {k: v for k, v in entry.items() if k != "text"},
                }
            )
        return results

    def save(self) -> None:
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadatas, f)
