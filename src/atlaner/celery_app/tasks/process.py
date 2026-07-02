"""
tasks/process.py
────────────────────────────
Tier-2 tasks (queue: process)
─────────────────────────────────────────────────────────────────────────
Flow (continued from scrape.py):
  download_pdf  →  parse_pdf  →  clean_text  →  chunk_document
                                                      └─► generate_embeddings
─────────────────────────────────────────────────────────────────────────
Design notes:
  • parse_pdf   – layout-aware extraction via PyMuPDF; sections detected
                  by font-size heuristics.
  • clean_text  – remove boilerplate (headers/footers, ref numbers),
                  normalise whitespace.
  • chunk_document – recursive token-aware splitting with metadata
                     inheritance per chunk (paper id, section, page range).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import structlog
import json
from celery import signature
from langchain_text_splitters import RecursiveCharacterTextSplitter

from atlaner.celery_app.main import app
from atlaner.utils.text_cleaner import clean_academic_text
from atlaner.config.settings import settings
from grobid_client.grobid_client import GrobidClient

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 5 — parse_pdf
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlaner.celery_app.tasks.process.parse_pdf",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue="process",
    time_limit=300,
    soft_time_limit=240,
    ignore_result=False,
)
def parse_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts structured text from a downloaded PDF.

    Returns metadata enriched with:
      - `sections`: list of {title, text}
      - `full_text`: concatenated plain text (fallback)
    """
    paper_id     = metadata["paper_id"]
    repository   = metadata["repository"]
    pdf_path_str = metadata.get("local_pdf_path")

    # Skip papers that were flagged during download
    if not pdf_path_str or metadata.get("skip_reason"):
        log.warning(
            "parse_pdf.skip",
            paper_id=paper_id,
            repository=repository,
            reason=metadata.get("skip_reason", "no pdf path"),
        )
        return metadata

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        log.error("parse_pdf.file_missing", paper_id=paper_id, repository=repository, path=str(pdf_path))
        raise self.retry(exc=FileNotFoundError(str(pdf_path)))

    log.info("parse_pdf.start", paper_id=paper_id, repository=repository)

    # Ensure output directory exists
    target_dir = Path(settings.pdf_download_dir) / repository
    out_dir = target_dir / "out"
    target_dir.mkdir(exist_ok=True, parents=True)

    # Run GROBID process (sync)
    log.info("grobid.process.start", paper_id=paper_id, repository=repository, path=str(pdf_path))
    client = GrobidClient(grobid_server=settings.grobid_server_url)

    try:
        client.process(
            service="processFulltextDocument",
            input_path=str(target_dir),
            output=str(out_dir),
            n=1,
            json_output=True,
            segment_sentences=True,
        )
    except Exception as e:
        log.error("grobid.process.failed", paper_id=paper_id, repository=repository, error=str(e))
        raise self.retry(exc=e)

    # Find the output JSON
    fname = pdf_path.stem
    json_path = out_dir / f"{fname}.json"
    json_data = None

    if not json_path.exists():
        log.error("grobid.json_not_found", paper_id=paper_id, repository=repository, expected=str(json_path))
        raise self.retry(exc=FileNotFoundError(f"GROBID JSON missing: {json_path}"))

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # grouping body text
    grouped: Dict[str, List[str]] = {}
    full_text: str = ""
    body_text = json_data.get('body_text', [])

    for item in body_text:
        section = item.get('head_section', None)
        grouped.setdefault(section, [])
        grouped[section].append(item['text'])
        full_text += item['text'] + "\n\n"
    
    sections = { 
        section if section else settings.default_section: " ".join(texts) 
        for section, texts in grouped.items()
    }
    
    log.info(
        "parse_pdf.done",
        paper_id=paper_id,
        repository=repository,
        sections=len(sections),
    )

    metadata["sections"]   = sections
    metadata["full_text"]  = full_text

    log.info(
        "parse_pdf.done",
        paper_id=paper_id,
        repository=repository,
        sections=len(sections),
        chars=len(metadata["full_text"]),
    )

    # Chain to clean_text
    (
        clean_text.s(metadata).set(queue="process")
        | chunk_document.s().set(queue="process")
        | signature(
            "atlaner.celery_app.tasks.embed.generate_embeddings",
            queue="embed",
            immutable=False,
        )
        | signature(
            "atlaner.celery_app.tasks.embed.store_chunks",
            queue="embed",
            immutable=False,
        )
    ).apply_async()

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4b — clean_text
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlaner.celery_app.tasks.process.clean_text",
    bind=True,
    max_retries=2,
    queue="process",
    ignore_result=False,
)
def clean_text(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans raw extracted text in every section:
      - strip running headers/footers (page numbers, journal name)
      - normalise unicode & whitespace
      - remove reference section (not useful for RAG body search)
      - remove figure/table captions inline markers (Figure 1, Table 2…)
    """
    paper_id     = metadata["paper_id"]
    repository   = metadata["repository"]
    log.info("clean_text.start", paper_id=paper_id, repository=repository)

    cleaned_sections = {}
    for section, text in metadata.get("sections", {}).items():
        cleaned_section = clean_academic_text(section)
        cleaned_text = clean_academic_text(text)
        cleaned_sections.setdefault(cleaned_section, [])
        cleaned_sections[cleaned_section].append(cleaned_text)

    # Also clean the full_text field
    metadata["sections"]  = {section: " ".join(texts) for section, texts in cleaned_sections.items()}
    metadata["full_text"] = clean_academic_text(metadata.get("full_text", ""))

    log.info(
        "clean_text.done",
        paper_id=paper_id,
        repository=repository,
        sections_kept=len(cleaned_sections),
    )
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 5 — chunk_document
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="atlaner.celery_app.tasks.process.chunk_document",
    bind=True,
    max_retries=2,
    queue="process",
    ignore_result=False,
)
def chunk_document(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Splits each section into RAG-ready chunks.

    Each chunk is a dict:
    {
        "chunk_id":      "<paper_id>_<section>_<chunk_idx>",
        "paper_id":      str,
        "repository":    str,
        "title":         str,         # paper title
        "section":       str,         # heading of section
        "text":          str,         # chunk body
        "authors":       list[str],
        "categories":    list[str],
        "published":     str,         # ISO date
        "token_count":   int,         # approximate
    }

    Returns metadata with `chunks` key added.
    """
    paper_id     = metadata["paper_id"]
    repository   = metadata["repository"]
    log.info("chunk_document.start", paper_id=paper_id, repository=repository)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens * 4,     # ~4 chars per token
        chunk_overlap=settings.chunk_overlap_tokens * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Dict[str, Any]] = []

    for sec_idx, (section, raw_text) in enumerate(metadata.get("sections", {}).items()):
        if len(raw_text) < settings.min_chunk_chars:
            continue

        split_texts = splitter.split_text(raw_text)

        for chunk_idx, chunk_text in enumerate(split_texts):
            if len(chunk_text) < settings.min_chunk_chars:
                continue

            chunk: Dict[str, Any] = {
                "chunk_id":    f"{paper_id}_{sec_idx}_{chunk_idx}",
                "paper_id":    paper_id,
                "repository":  repository,
                "title":       metadata.get("title", ""),
                "section":     section,
                "text":        chunk_text,
                "authors":     metadata.get("authors", []),
                "categories":  metadata.get("categories", []),
                "published":   metadata.get("published", ""),
                "doi":         metadata.get("doi", ""),
                # Approximate token count (chars / 4)
                "token_count": len(chunk_text) // 4,
            }
            chunks.append(chunk)

    log.info(
        "chunk_document.done",
        paper_id=paper_id,
        repository=repository,
        chunks=len(chunks),
        avg_tokens=int(sum(c["token_count"] for c in chunks) / max(len(chunks), 1)),
    )

    metadata["chunks"] = chunks
    return metadata
