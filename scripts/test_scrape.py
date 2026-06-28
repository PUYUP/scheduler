from docling.datamodel.pipeline_options import TesseractCliOcrOptions
from docling.datamodel.pipeline_options import TesseractOcrOptions
from docling.datamodel.pipeline_options import EasyOcrOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import PdfFormatOption
from docling.datamodel.base_models import InputFormat
import json
import os

from pathlib import Path
from datetime import datetime
from celery_app.tasks.scrape import scrape_paper_metadata, download_pdf
from celery_app.tasks.process import parse_pdf, clean_text, chunk_document, _json_to_chunks
from docling.document_converter import DocumentConverter


def save_chunks_to_json(chunks, arxiv_id: str, output_dir: str = "output"):
    """Simpan chunks ke file JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/chunks_{arxiv_id}_{timestamp}.json"

    output = {
        "arxiv_id": arxiv_id,
        "timestamp": timestamp,
        "total_chunks": len(chunks) if isinstance(chunks, list) else None,
        "chunks": chunks,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Chunks saved to: {filename}")
    return filename


def main_x():
    arxiv_id = "2606.20564"

    result = scrape_paper_metadata(arxiv_id=arxiv_id)
    download_result = download_pdf(result)
    parsing = parse_pdf(download_result)
    cleans = clean_text(parsing)
    chunks = chunk_document(cleans)

    print(chunks)

    # Simpan ke JSON
    saved_path = save_chunks_to_json(chunks, arxiv_id=arxiv_id)
    print(f"File tersimpan di: {saved_path}")


def buat_pipeline_options_cpu() -> PdfPipelineOptions:
    """
    Konfigurasi pipeline Docling khusus CPU (tanpa GPU).
    Menonaktifkan model AI berat yang membutuhkan GPU.
    """
    pipeline_options = PdfPipelineOptions()

    # --- Nonaktifkan model yang butuh GPU ---
    pipeline_options.do_table_structure = False   # TableFormer (butuh GPU/berat)
    pipeline_options.do_ocr = False               # OCR (aktifkan jika PDF hasil scan)

    # Gunakan backend ringan berbasis PDF native
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False

    return pipeline_options


def main():
    # path = '/Volumes/SSD1/Private/Curio/indexer/executor/s12929-026-01271-w/auto/s12929-026-01271-w_content_list_v2.json'
    # chunks = _json_to_chunks(Path(path))
    # print(chunks)
    source = "/Volumes/Wgaming/Users/pointilis/Downloads/2512.00565v1.pdf"  # file path or URL

    # --- Konfigurasi pipeline ---
    aktifkan_ocr: bool = True
    engine_ocr: str = "easyocr"
    pipeline_options = buat_pipeline_options_cpu()

    if aktifkan_ocr:
        pipeline_options.do_ocr = True
        if engine_ocr == "easyocr":
            # EasyOCR: otomatis pakai CPU jika GPU tidak tersedia
            pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
        elif engine_ocr == "tesseract":
            pipeline_options.ocr_options = TesseractOcrOptions()
        elif engine_ocr == "tesseract_cli":
            pipeline_options.ocr_options = TesseractCliOcrOptions()
        else:
            raise ValueError(f"engine_ocr tidak dikenal: {engine_ocr}")
        print(f"🔍 OCR    : Aktif ({engine_ocr})")
    else:
        print("🔍 OCR    : Nonaktif (PDF teks native)")

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,  # Backend ringan, tidak butuh GPU
            )
        }
    )
    result = converter.convert(source=source)
    print(result)
    # doc = converter.convert(source).document

    # print(doc.export_to_markdown())


if __name__ == "__main__":
    main()