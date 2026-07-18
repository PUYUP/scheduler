import logging
import json

from typing import Any, Optional, Sequence
from dotenv import load_dotenv
from google import genai

logger = logging.getLogger(__name__)

# Muat variabel dari file .env (kalau ada) ke environment,
# supaya GEMINI_API_KEY bisa diset di .env tanpa export manual.
load_dotenv()

# Inisialisasi klien Gemini.
# Otomatis pakai environment variable GEMINI_API_KEY (dari .env atau shell).
client = genai.Client()

# Prompt untuk batcher: instruksikan Gemini membaca SEMUA chunks dulu,
# baru merangkum satu paper secara utuh (bukan per-chunk).
BATCH_PROMPT = (
    "Berikut adalah isi lengkap sebuah paper akademik, yang dipecah menjadi "
    "beberapa bagian (chunks) berurutan dan digabung jadi satu teks di bawah "
    "ini, dipisahkan dengan marker '--- CHUNK BREAK ---'. Baca seluruh chunks "
    "sampai selesai sebelum menjawab, lalu buat SATU ringkasan utuh untuk "
    "keseluruhan paper (bukan ringkasan per chunk), yang mencakup: latar "
    "belakang/masalah, metode, temuan utama, dan kesimpulan."
)

# Pemisah antar chunk saat digabung jadi satu teks.
CHUNK_SEPARATOR = "\n\n--- CHUNK BREAK ---\n\n"


def get_batch_prompt(language_code: str) -> str:
    """
    Menghasilkan prompt dinamis yang memaksa output ke bahasa target apa pun
    (bisa berupa kode bahasa seperti 'en', 'es', 'ja', atau nama bahasa).
    """
    return (
        "Berikut adalah isi lengkap sebuah paper akademik, yang dipecah menjadi "
        "beberapa bagian (chunks) berurutan dan digabung jadi satu teks di bawah "
        "ini, dipisahkan dengan marker '--- CHUNK BREAK ---'. Baca seluruh chunks "
        "sampai selesai sebelum menjawab, lalu buat SATU ringkasan utuh untuk "
        "keseluruhan paper (bukan ringkasan per chunk) dalam format JSON.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        f"1. Translate and write all the values in the JSON output strictly in the "
        f"language corresponding to this language code/name: '{language_code}'. "
        "Only translate the values, keep the JSON keys strictly as defined in the schema.\n"
        "2. DO NOT use phrases like 'this paper', 'this study', 'the authors', 'artikel ini', "
        "or any equivalent meta-phrases in the target language. Write the summary directly as "
        "factual statements or explanations, completely removing any fluff or context indicating "
        "that this is a summary of an academic paper.\n"
        "3. PRESERVE TECHNICAL JARGON & INDUSTRY TERMS: Do not translate standard technical terms, "
        "academic jargon, widely accepted acronyms, or domain-specific nomenclature (for example: "
        "'machine learning', 'zero-shot learning', 'overfitting', 'framework', etc.) if translating "
        "them would make the text sound awkward, forced, or lose its precise scientific meaning "
        "in the target language. Keep these terms in their original English/technical form."
    )


def _build_inline_request(
    chunks: Sequence[str],
    prompt: str = BATCH_PROMPT,
) -> Any:
    """
    Menyusun satu inline request dari SEMUA chunks milik satu dokumen.

    Semua chunks digabung jadi satu teks (dipisah CHUNK_SEPARATOR) supaya
    Gemini membaca keseluruhan dokumen dalam satu konteks sebelum menjawab.
    Karena ini batch mode dan paper cuma ~30 halaman, ukuran gabungan ini
    masih jauh dari limit context window 1 juta token, jadi aman dikirim
    sebagai teks biasa tanpa perlu upload ke GCS.

    Args:
        chunks: Daftar potongan teks dari satu dokumen, urut sesuai posisi
            aslinya (misal per section atau per beberapa halaman).
        prompt: Instruksi yang ditaruh di depan gabungan chunks.
            Default: BATCH_PROMPT.

    Returns:
        Dict yang merepresentasikan satu request dalam batch.
    """
    combined_text = CHUNK_SEPARATOR.join(chunks)
    combined = f"{prompt}\n\n{combined_text}" if prompt else combined_text
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": combined}],
            }
        ],
        "config": {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "background": {
                        "type": "STRING",
                        "description": "Latar belakang atau masalah dari paper"
                    },
                    "methods": {
                        "type": "STRING",
                        "description": "Metode penelitian yang digunakan"
                    },
                    "results": {
                        "type": "STRING",
                        "description": "Temuan utama dari paper"
                    },
                    "conclusions": {
                        "type": "STRING",
                        "description": "Kesimpulan dari paper"
                    },
                    "limitations": {
                        "type": "STRING",
                        "description": "Keterbatasan, perdebatan, atau kelemahan dari metode maupun hasil penelitian dalam paper"
                    },
                    "future_works": {
                        "type": "STRING",
                        "description": "Potensi improvisasi, pengembangan, atau rekomendasi untuk penelitian selanjutnya"
                    }
                },
                "required": ["background", "methods", "results", "conclusions"]
            }
        }
    }


def create_batch_job(
    documents: Sequence[Sequence[str]],
    model: str = "gemini-3.1-flash-lite",
    display_name: Optional[str] = None,
    language_code: str = 'en',
    user_id: Optional[str] = None,
    paper_id: Optional[str] = None,
    challenge_id: Optional[str] = None,
    challenge_paper_id: Optional[str] = None,
    challenge_paper_summary_id: Optional[str] = None
) -> Any:
    """
    Membuat batch processing job dari chunks langsung (inline), tanpa file/GCS.

    Args:
        documents: Daftar dokumen. Setiap elemen adalah list of chunks milik
            SATU dokumen/paper (misal ["chunk1", "chunk2", "chunk3"]).
            Semua chunks dalam satu dokumen akan digabung jadi satu request,
            sehingga Gemini membaca keseluruhan paper sebelum merangkum.
            Kalau cuma punya satu paper, cukup kirim [[chunk1, chunk2, ...]].
        model: Model yang dipakai untuk batch processing.
        prompt: Instruksi yang digabungkan ke setiap dokumen. Default: BATCH_PROMPT.
        display_name: Nama opsional biar job mudah dikenali di dashboard.

    Returns:
        Batch job object yang berhasil dibuat.
    """
    if not documents:
        raise ValueError("documents tidak boleh kosong")

    prompt = get_batch_prompt(language_code)

    inline_requests = [
        _build_inline_request(chunks, prompt) for chunks in documents
    ]

    config: genai.types.CreateBatchJobConfigDict = {}

    if display_name:
        config["display_name"] = display_name

    if user_id:
        config["webhook_config"] = {
            "uris": ["https://tunnel.atlanize.com/gemini-batch-webhook"],
            "user_metadata": {
                "user_id": user_id,
                "paper_id": paper_id,
                # this will be use for updating challenge paper processing
                "challenge_id": challenge_id,
                "challenge_paper_id": challenge_paper_id,
                "challenge_paper_summary_id": challenge_paper_summary_id
            }
        }

    try:
        job = client.batches.create(
            model=model,
            src=inline_requests,
            config=config or None,
        )
        logger.info(f"Berhasil membuat batch job: {job.name}")
        return job
    except Exception as e:
        logger.error(f"Gagal membuat batch job: {e}")
        raise


def get_batch_job(job_name: str) -> Any:
    """
    Mengambil status dan detail dari batch job yang ada.

    Args:
        job_name: Nama job (contoh: 'batches/123'), diambil dari job.name
            saat create_batch_job dipanggil.
    """
    return client.batches.get(name=job_name)


def list_batch_jobs() -> Any:
    """
    Menampilkan daftar batch jobs yang ada.
    """
    return client.batches.list()


def get_batch_results(job_name: str) -> list[Any]:
    """
    Mengambil hasil dari batch job yang sudah selesai (JOB_STATE_SUCCEEDED)
    dan mengonversi teks JSON-nya menjadi Python dictionary.

    Returns:
        List berisi dictionary hasil ringkasan. Kalau ada request yang gagal 
        atau tidak bisa diparse, elemen tersebut diisi string error-nya.
    """
    job = client.batches.get(name=job_name)
    results: list[Any] = []
    
    if not job.dest or not job.dest.inlined_responses:
        logger.warning(f"Batch job {job_name} belum selesai atau tidak memiliki inlined_responses. Status: {job.state}")
        return results

    for inline_response in job.dest.inlined_responses:
        if inline_response.response:
            response_text = inline_response.response.text
            if not response_text:
                results.append("ERROR: response.text is empty or None")
                continue
            try:
                # Mem-parsing teks JSON langsung jadi dict
                parsed_json = json.loads(response_text)
                results.append(parsed_json)
            except json.JSONDecodeError:
                results.append(f"JSON_ERROR: Gagal melakukan parse -> {response_text}")
        else:
            results.append(f"ERROR: {inline_response.error}")
            
    return results
