# Author: Youssef Aitbouddroub
"""Embed PDF documents for the RAG index.

Pipeline:
    PDF -> Markdown (PDF2MarkdownProcessor)
        -> heading- and token-aware chunks (hybrid splitter)
        -> Ollama embeddings (batched, with retries)
        -> one JSONL file per document

Usage:
    python embed_documents.py --input documents/pdfs --output documents/embeddings
    python embed_documents.py --input paper.pdf --output out/ --overwrite

Each output line is a JSON object:
    {"source", "chunk_index", "page", "heading", "text", "token_count", "embedding"}
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import ollama
import pymupdf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.PDF2MarkdownProcessor import PDF2MarkdownProcessor  
from markdown_splitter import load_tokenizer as _load_tokenizer, split_markdown_by_headings_with_token_limit  

load_dotenv()
log = logging.getLogger("embed_documents")

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "google/gemma-3-4b-it")
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "512"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "3"))
# Chunks are capped at TOKEN_LIMIT, so the default 4096 context only wastes KV-cache RAM.
# 2x headroom covers tokenizer differences between Gemma (splitter) and Qwen (embedder).
# Use the same value as retrieval.py: a different num_ctx makes Ollama reload the model.
# Ollama picks GPU or CPU on its own; OLLAMA_NUM_GPU (layers to offload, 0 = CPU only)
# overrides it. Keep it equal to retrieval.py's, for the same reload reason as num_ctx.
GPU_OPTIONS = {"num_gpu": int(os.environ["OLLAMA_NUM_GPU"])} if os.getenv("OLLAMA_NUM_GPU") else {}
EMBED_OPTIONS = {"num_ctx": int(os.getenv("EMBED_NUM_CTX", str(2 * TOKEN_LIMIT))), **GPU_OPTIONS}
HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class ChunkRecord:
    source: str
    chunk_index: int
    page: int
    heading: Optional[str]
    text: str
    token_count: int
    embedding: list[float]


# ---------------------------------------------------------------------------
# Setup and checks
# ---------------------------------------------------------------------------

def load_tokenizer():
    return _load_tokenizer(TOKENIZER_NAME, token=os.getenv("HG_ACCESS_TOKEN"))


def ensure_embedding_model(model: str) -> None:
    """Fail fast if Ollama is unreachable or the model is not pulled."""
    try:
        ollama.show(model)
    except ollama.ResponseError as exc:
        raise SystemExit(f"Ollama model '{model}' is not available: {exc}. Run: ollama pull {model}")
    except ConnectionError as exc:
        raise SystemExit(f"Cannot reach Ollama: {exc}")


def document_stats(pdf_path: Path) -> dict:
    with pymupdf.open(pdf_path) as doc:
        return {
            "page_count": doc.page_count,
            "title": doc.metadata.get("title") or "",
            "needs_pass": doc.needs_pass,
        }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def extract_heading(text: str) -> Optional[str]:
    headings = [
        match.group(1).strip()
        for line in text.splitlines()
        if (match := HEADING_PATTERN.match(line))
    ]
    return headings[-1] if headings else None


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts, retrying transient Ollama failures."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.embed(model=EMBED_MODEL, input=list(texts), options=EMBED_OPTIONS)
            embeddings = response["embeddings"]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
                )
            return embeddings
        except (ollama.ResponseError, ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            log.warning("Embedding attempt %d/%d failed (%s); retrying in %ds",
                        attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover




def embed_document(pdf_path: Path, converter: PDF2MarkdownProcessor, tokenizer, token_limit: int = TOKEN_LIMIT) -> list[ChunkRecord]:
    page_chunks = converter.convert(pdf_path, page_chunks=True)
    if not isinstance(page_chunks, list):
        raise RuntimeError(f"Expected page chunks for {pdf_path.name}")
    log.info("Converted %s to %d page chunks", pdf_path.name, len(page_chunks))

    chunks: list[tuple[int, Optional[str], str]] = []
    for page_chunk in page_chunks:
        page = int(page_chunk["page"])
        heading = page_chunk.get("heading")
        page_markdown = str(page_chunk["text"])
        page_parts = split_markdown_by_headings_with_token_limit(
            page_markdown, tokenizer, token_limit
        )
        chunks.extend(
            (page, extract_heading(text) or heading, text)
            for text in page_parts
            if text.strip()
        )

    if not chunks:
        raise ValueError(f"No chunks produced for {pdf_path.name}")
    log.info("Split %s into %d chunks (limit %d tokens)", pdf_path.name, len(chunks), token_limit)

    records: list[ChunkRecord] = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_records = chunks[start:start + EMBED_BATCH_SIZE]
        batch = [text for _, _, text in batch_records]
        vectors = embed_batch(batch)
        for offset, ((page, heading, text), vector) in enumerate(zip(batch_records, vectors)):
            records.append(
                ChunkRecord(
                    source=pdf_path.name,
                    chunk_index=start + offset,
                    page=page,
                    heading=heading,
                    text=text,
                    token_count=count_tokens(text, tokenizer),
                    embedding=vector,
                )
            )

    dimensions = {len(record.embedding) for record in records}
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent embedding dimensions for {pdf_path.name}: {dimensions}")
    log.info("Embedded %d chunks (dimension %d)", len(records), dimensions.pop())
    return records



# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: Sequence[ChunkRecord]) -> None:
    """Write records atomically so a crash never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    return []


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed PDF documents into JSONL files.")
    parser.add_argument("--input", type=Path, required=True, help="a PDF file or a directory of PDFs")
    parser.add_argument("--output", type=Path, required=True, help="directory for JSONL output")
    parser.add_argument("--overwrite", action="store_true", help="re-embed documents that already have output")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pdfs = collect_pdfs(args.input)
    if not pdfs:
        log.error("No PDF files found at %s", args.input)
        return 1

    ensure_embedding_model(EMBED_MODEL)
    tokenizer = load_tokenizer()
    converter = PDF2MarkdownProcessor()

    failures = 0
    for pdf in pdfs:
        target = args.output / f"{pdf.stem}.jsonl"
        if target.exists() and not args.overwrite:
            log.info("Skipping %s (output exists, use --overwrite to re-embed)", pdf.name)
            continue
        try:
            stats = document_stats(pdf)
            log.info("Processing %s (%d pages)", pdf.name, stats["page_count"])
            records = embed_document(pdf, converter, tokenizer)
            write_jsonl(target, records)
            log.info("Wrote %d records to %s", len(records), target)
        except Exception:
            failures += 1
            log.exception("Failed to embed %s", pdf)

    # Ingestion is a batch job: free the embedder's RAM now instead of after Ollama's idle timeout.
    with contextlib.suppress(ollama.ResponseError, ConnectionError):
        ollama.generate(model=EMBED_MODEL, keep_alive=0)

    log.info("Finished: %d document(s), %d failed", len(pdfs), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())