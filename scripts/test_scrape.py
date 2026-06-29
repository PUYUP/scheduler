from unittest import result
import json
import os

from pathlib import Path
from datetime import datetime
from itertools import groupby
from celery_app.tasks.scrape import scrape_paper_metadata, download_pdf
from celery_app.tasks.process import parse_pdf, clean_text, chunk_document
from celery_app.tasks.embed import generate_embeddings, store_chunks
from grobid_client.grobid_client import GrobidClient
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings


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
    paper_id = "2606.20564"
    repository = "arxiv"

    result = scrape_paper_metadata(paper_id=paper_id, repository=repository)
    download_result = download_pdf(result)
    parsing = parse_pdf(download_result)
    cleans = clean_text(parsing)
    chunk = chunk_document(cleans)
    embeddings = generate_embeddings(chunk)
    store = store_chunks(embeddings)

    # print(embeddings)
    

    # # Simpan ke JSON
    # saved_path = save_chunks_to_json(chunks, arxiv_id=arxiv_id)
    # print(f"File tersimpan di: {saved_path}")

    # client = GrobidClient()
    # client.process(
    #     service="processFulltextDocument",
    #     input_path="/Volumes/SSD1/Private/Curio/scheduler/input",
    #     output="/Volumes/SSD1/Private/Curio/scheduler/output",
    #     n=10,
    #     json_output=True,
    #     markdown_output=True,
    #     segment_sentences=True,
    # )

    # json_file = "output/2512.00565v1.json"
    # with open(json_file, "r", encoding="utf-8") as f:
    #     json_data = json.load(f)
    
    # # paper metadata
    # title = json_data['biblio'].get('title')
    # authors = json_data['biblio'].get('authors')
    # publication_date = json_data['biblio'].get('publication_date')
    # abstract = " ".join([
    #     item.get("text", "")
    #     for a in json_data["biblio"].get("abstract", [])
    #     for item in a
    #     if item.get("text")
    # ])
    # body_text = json_data.get('body_text', [])
    
    # grouped = {}
    # for item in body_text:
    #     section = item.get('head_section', None)
    #     grouped.setdefault(section, [])
    #     grouped[section].append(item['text'])
    
    # sections = {section: " ".join(texts) for section, texts in grouped.items()}
    # result = []

    # for section, text in sections.items():
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         separators=["\n\n", "\n", ".", " "],
    #         chunk_size=settings.chunk_size_tokens,
    #         chunk_overlap=settings.chunk_overlap_tokens,
    #         length_function=len,
    #         is_separator_regex=False,
    #     )

    #     chunks = text_splitter.split_text(text)
    #     result.append({
    #         "section": section if section is not None else settings.default_section,
    #         "chunks": chunks
    #     })

    # print(result)

    # document = """An intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn’t exceed a specified size limit. Key benefits of length-based splitting:"""
    # texts = text_splitter.split_text(document)
    # print(texts)

    # splitter = RecursiveJsonSplitter(max_chunk_size=512)
    # json_chunks = splitter.split_json(json_data=json_data)
    # saved_path = save_chunks_to_json(markdown_chunks, arxiv_id="2512.00565")
    # print(f"File tersimpan di: {saved_path}")

if __name__ == "__main__":
    main()