import logging
import numpy as np
import hdbscan

from typing import List, Dict, Any
from sklearn.preprocessing import normalize

# Konfigurasi logging standar untuk production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NotesClusteringService")

class NotesClusteringService:
    """
    Layanan untuk mengelompokkan catatan yang mirip/duplikat menggunakan HDBSCAN.
    """
    def __init__(
        self, 
        min_cluster_size: int = 2, 
        min_samples: int = 1, 
        metric: str = 'euclidean'
    ):
        # min_cluster_size = 2: Kita ingin menangkap sekecil-kecilnya 2 catatan yang duplikat.
        # min_samples = 1: Membuat algoritma sangat sensitif terhadap cluster kecil/padat.
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=0.44
        )

    def find_duplicates(
        self, 
        notes_ids: List[str], 
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Mengeksekusi HDBSCAN untuk memisahkan catatan unik dan kelompok duplikat.
        
        Args:
            note_ids (List[str]): List ID unik untuk setiap catatan.
            embeddings (np.ndarray): Matriks embedding berukuran (n_samples, n_features).
            
        Returns:
            Dict: Hasil clustering berisi catatan unik dan kelompok duplikat.
        """
        # 1. Validasi Input
        if not notes_ids or embeddings.size == 0:
            logger.warning("Data input kosong. Mengembalikan hasil kosong.")
            return {"unique": [], "duplicate": {}}
            
        if len(notes_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Dimensi tidak cocok: {len(notes_ids)} note_ids vs {embeddings.shape[0]} embeddings"
            )

        # Jika data kurang dari min_cluster_size, tidak mungkin ada cluster
        if len(notes_ids) < self.min_cluster_size:
            logger.info("Jumlah data terlalu sedikit untuk di-cluster. Semua dianggap unik.")
            return {"unique": notes_ids, "duplicate": {}}

        try:
            # 2. Normalisasi L2 (Best Practice untuk Text Embeddings)
            # Jika embedding (misal dari OpenAI/BERT) menggunakan Cosine Similarity,
            # normalisasi L2 + Euclidean distance secara matematis setara dengan Cosine.
            logger.info("Menormalisasi embedding menggunakan L2 norm...")
            normalized_embeddings = normalize(embeddings, norm='l2')

            # 3. Fitting Model
            logger.info(f"Memulai proses clustering untuk {len(notes_ids)} catatan...")
            labels = self.clusterer.fit_predict(normalized_embeddings)
            
            # 4. Pemrosesan Hasil Labeling
            unique_notes = []
            duplicate_clusters = {}

            for note_id, label in zip(notes_ids, labels):
                if label == -1:
                    # Label -1 adalah noise (tidak masuk cluster mana pun) -> Catatan Unik
                    unique_notes.append(note_id)
                else:
                    # Masuk ke dalam cluster duplikat
                    if label not in duplicate_clusters:
                        duplicate_clusters[label] = []
                    duplicate_clusters[label].append(note_id)

            logger.info(
                f"Clustering selesai: {len(unique_notes)} catatan unik, "
                f"{len(duplicate_clusters)} kelompok duplikat ditemukan."
            )

            return {
                "unique": unique_notes,
                "duplicate": duplicate_clusters
            }

        except Exception as e:
            logger.error(f"Gagal melakukan clustering: {str(e)}", exc_info=True)
            raise
    