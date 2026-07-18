"""
stanza_chunker.py
=====================
Chunking teks panjang (jawaban user) menjadi beberapa bagian sekitar
500-2000 kata, dengan 2 mode:

1. mode simple  (semantic=False) -> hanya berbasis jumlah kata + batas kalimat
   (kalimat tidak pernah terpotong, tapi titik potong antar-chunk murni
   ditentukan oleh hitungan kata).

2. mode semantic (semantic=True) -> topic-aware. Selain menjaga batas kalimat,
   titik potong antar-chunk dipilih pada tempat di mana topik/gagasan memang
   berganti (dideteksi lewat sentence embeddings multi-bahasa), bukan asal
   berhenti begitu jumlah kata terpenuhi.

Instalasi:
    pip install stanza sentence-transformers

Pemakaian singkat:
    from stanza_chunker import chunk_answer

    hasil = chunk_answer(teks, lang="id", semantic=True, download_models=True)
    # hasil -> list[str]
"""

from __future__ import annotations

import numpy as np
import stanza
from stanza.pipeline.multilingual import MultilingualPipeline

_PIPELINE_CACHE: dict = {}
_EMBEDDER_CACHE: dict = {}

DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # ringan & multi-bahasa


# ---------------------------------------------------------------------------
# Segmentasi kalimat (Stanza, multi-bahasa)
# ---------------------------------------------------------------------------

def _get_pipeline(lang: str | None = None, download_models: bool = False):
    cache_key = lang if lang else "__multilingual__"
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    if download_models:
        stanza.download(lang if lang else "multilingual", verbose=False)

    if lang:
        nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize",
            verbose=False,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES
        )
    else:
        nlp = MultilingualPipeline(
            download_method=stanza.DownloadMethod.REUSE_RESOURCES
        )

    _PIPELINE_CACHE[cache_key] = nlp
    return nlp


def _split_sentences(
    text: str,
    lang: str | None = None,
    download_models: bool = False
) -> list[str]:
    nlp = _get_pipeline(lang=lang, download_models=download_models)
    doc = nlp(text)
    return [s.text.strip() for s in doc.sentences if s.text.strip()]


# ---------------------------------------------------------------------------
# Embedding kalimat (untuk mode semantic)
# ---------------------------------------------------------------------------

def _get_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Mode semantic butuh package 'sentence-transformers'.\n"
            "Install dengan: pip install sentence-transformers"
        ) from e
    model = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = model
    return model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Mode 1: simple (word-count based)
# ---------------------------------------------------------------------------

def _group_sentences_by_wordcount(
    sentences: list[str],
    min_words: int,
    max_words: int
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    for sent in sentences:
        words = sent.split()
        n = len(words)

        if n > max_words:
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            for i in range(0, n, max_words):
                chunks.append(" ".join(words[i : i + max_words]))
            continue

        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [sent], n
        else:
            current.append(sent)
            count += n
            if count >= min_words:
                chunks.append(" ".join(current))
                current, count = [], 0

    if current:
        if chunks and count < min_words:
            chunks[-1] = chunks[-1] + " " + " ".join(current)
        else:
            chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Mode 2: semantic / topic-aware
# ---------------------------------------------------------------------------

def _find_chunk_boundaries(
    word_counts: list[int],
    sim_scores: list[float],
    min_words: int,
    max_words: int
) -> list[tuple[int, int]]:
    """
    Fungsi murni (tidak butuh model, mudah di-unit-test): menentukan
    index (start, end) kalimat untuk tiap chunk.

    Algoritma:
    1. Dari `start`, tambah kalimat sampai total kata >= min_words (syarat wajib).
    2. Lanjutkan menambah kalimat (selama total kata masih <= max_words),
       sambil mencatat titik potong dengan sim_scores TERENDAH yang dijumpai
       -> sim_scores rendah antara kalimat i & i+1 berarti topik di situ
       memang berbeda, jadi itu titik potong yang lebih "alami".
    3. Potong di titik terbaik itu (bukan langsung begitu min_words terpenuhi).
    """
    n = len(word_counts)
    boundaries: list[tuple[int, int]] = []
    start = 0

    while start < n:
        cum = 0
        idx = start
        while idx < n and cum < min_words:
            cum += word_counts[idx]
            idx += 1
        idx -= 1  # index kalimat terakhir saat cum >= min_words tercapai

        if idx >= n - 1:
            boundaries.append((start, n - 1))
            break

        best_idx = idx
        best_score = sim_scores[idx] if idx < len(sim_scores) else -1.0

        extend_cum = cum
        j = idx
        while j + 1 < n:
            next_cum = extend_cum + word_counts[j + 1]
            if next_cum > max_words:
                break
            j += 1
            extend_cum = next_cum
            if j < len(sim_scores) and sim_scores[j] < best_score:
                best_score = sim_scores[j]
                best_idx = j

        boundaries.append((start, best_idx))
        start = best_idx + 1

    # Gabungkan sisa chunk terakhir yang terlalu kecil ke chunk sebelumnya
    if len(boundaries) > 1:
        last_start, last_end = boundaries[-1]
        last_word_count = sum(word_counts[last_start : last_end + 1])
        if last_word_count < min_words:
            prev_start, _ = boundaries[-2]
            boundaries[-2] = (prev_start, last_end)
            boundaries.pop()

    return boundaries


def _group_sentences_by_topic(
    sentences: list[str],
    min_words: int,
    max_words: int,
    embed_model_name: str
) -> list[str]:
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [sentences[0]]

    word_counts = [len(s.split()) for s in sentences]

    embedder = _get_embedder(embed_model_name)
    embeddings = embedder.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    sim_scores = [_cosine_sim(embeddings[i], embeddings[i + 1]) for i in range(n - 1)]

    boundaries = _find_chunk_boundaries(word_counts, sim_scores, min_words, max_words)
    return [" ".join(sentences[s : e + 1]) for s, e in boundaries]


# ---------------------------------------------------------------------------
# API utama
# ---------------------------------------------------------------------------

def chunk_answer(
    text: str,
    lang: str | None = None,
    min_words: int = 500,
    max_words: int = 2000,
    download_models: bool = False,
    semantic: bool = False,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> list[str]:
    """
    Membagi teks panjang (jawaban user) menjadi beberapa chunk sekitar
    `min_words`-`max_words` kata, tanpa memotong kalimat di tengah.

    Parameters
    ----------
    text : str
        Teks yang akan dipecah.
    lang : str | None
        Kode bahasa Stanza (mis. 'id', 'en'). None -> auto-deteksi (multilingual).
    min_words, max_words : int
        Rentang target jumlah kata per chunk (default 500-2000).
    download_models : bool
        True -> unduh model Stanza yang dibutuhkan (sekali saja di awal).
    semantic : bool
        False (default) -> mode simple, titik potong murni dari hitungan kata.
        True             -> mode topic-aware, titik potong dipilih di tempat
                             topik memang berganti (butuh sentence-transformers).
    embed_model_name : str
        Model sentence-embedding multi-bahasa yang dipakai saat semantic=True.

    Returns
    -------
    list[str]
        List berisi potongan-potongan teks (chunk).
    """
    if not text or not text.strip():
        return []
    if min_words > max_words:
        raise ValueError("min_words tidak boleh lebih besar dari max_words")

    sentences = _split_sentences(text, lang=lang, download_models=download_models)
    if not sentences:
        return []

    if semantic:
        return _group_sentences_by_topic(sentences, min_words, max_words, embed_model_name)
    return _group_sentences_by_wordcount(sentences, min_words, max_words)
