import json
import os

from pathlib import Path
from datetime import datetime
from celery_app.tasks.scrape import scrape_paper_metadata, download_pdf
from celery_app.tasks.process import parse_pdf, clean_text, chunk_document, _json_to_chunks
from grobid_client.grobid_client import GrobidClient
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


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


def main():
    # arxiv_id = "2606.20564"

    # result = scrape_paper_metadata(arxiv_id=arxiv_id)
    # download_result = download_pdf(result)
    # parsing = parse_pdf(download_result)
    # cleans = clean_text(parsing)
    # chunks = chunk_document(cleans)

    # # Simpan ke JSON
    # saved_path = save_chunks_to_json(chunks, arxiv_id=arxiv_id)
    # print(f"File tersimpan di: {saved_path}")

    client = GrobidClient()
    client.process(
        service="processFulltextDocument",
        input_path="/Volumes/SSD1/Private/Curio/scheduler/input",
        output="/Volumes/SSD1/Private/Curio/scheduler/output",
        n=10,
        json_output=True,
        markdown_output=True,
        segment_sentences=True,
    )

    json_file = "output/2512.00565v1.json"
    with open(json_file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    markdown_file = "output/2512.00565v1.md"
    with open(markdown_file, "r", encoding="utf-8") as f:
        markdown_text = f.read()
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    markdown_chunks = splitter.split_text(markdown_text)
    
    # splitter = RecursiveJsonSplitter(max_chunk_size=512)
    # json_chunks = splitter.split_json(json_data=json_data)
    # saved_path = save_chunks_to_json(markdown_chunks, arxiv_id="2512.00565")
    # print(f"File tersimpan di: {saved_path}")
    for doc in markdown_chunks:
        print(doc)

if __name__ == "__main__":
    main()