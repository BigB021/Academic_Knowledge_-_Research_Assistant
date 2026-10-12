# Author: Youssef Aitbouddroub
# This script uses a hybrid approach to split markdown text into chunks based on headings, paragraphs, sentences, and token limits. 
# It ensures that each chunk does not exceed a specified token limit while preserving the structure of the markdown content.
# This approach is chosen after a series of experiments comparing different chunking strategies, including splitting by headings, paragraphs, sentences, and raw token windows.
# Check the notebook '2-text_splitting_strategy_benchmark.ipynb'

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.PDF2MarkdownProcessor import PDF2MarkdownProcessor

HEADING_START = re.compile(r"(?m)^(?=#{1,6}\s)")  # zero-width: keeps the '#' markers
HEADING_ONLY = re.compile(r"#{1,6}\s[^\n]*")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
SEPARATOR = "\n\n"


def _count(text, tokenizer):
    return len(tokenizer.encode(text, add_special_tokens=False))


def _hard_split(text, tokenizer, limit):
    """Last resort: cut by raw token windows. May split mid-word."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = (tokenizer.decode(ids[i:i + limit]).strip() for i in range(0, len(ids), limit))
    return [p for p in pieces if p]


def _split_to_limit(text, tokenizer, limit):
    """Break text into pieces that each fit within `limit` tokens.

    Tries paragraphs first, then sentences, then token windows.
    """
    if _count(text, tokenizer) <= limit:
        return [text]

    pieces = []
    for paragraph in PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _count(paragraph, tokenizer) <= limit:
            pieces.append(paragraph)
            continue
        for sentence in SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if _count(sentence, tokenizer) <= limit:
                pieces.append(sentence)
            else:
                pieces.extend(_hard_split(sentence, tokenizer, limit))
    return pieces


def _glue_orphan_headings(pieces, tokenizer, limit):
    """A heading left alone on a piece travels with the text that follows it."""
    result, pending = [], None
    for piece in pieces:
        if pending is not None:
            piece = f"{pending}{SEPARATOR}{piece}"
            pending = None
        if HEADING_ONLY.fullmatch(piece):
            pending = piece
        else:
            result.extend(_split_to_limit(piece, tokenizer, limit))
    if pending is not None:
        result.append(pending)
    return result


def _pack(pieces, tokenizer, limit):
    """Greedily join pieces while the *joined* text stays within the limit."""
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}{SEPARATOR}{piece}" if current else piece
        if _count(candidate, tokenizer) <= limit:
            current = candidate
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def split_markdown_by_headings_with_token_limit(markdown_text, tokenizer, token_limit=500):
    """Split markdown into chunks of at most `token_limit` tokens.

    Sections start at every markdown heading (# to ######). Sections that fit
    are kept whole; oversized ones are split by paragraph, then sentence, then
    token window. Pieces are then packed back together up to the limit.

    Every returned chunk is verified against the limit as a string, using
    tokenizer.encode(..., add_special_tokens=False).
    """
    if token_limit < 1:
        raise ValueError("token_limit must be >= 1")

    sections = [s.strip() for s in HEADING_START.split(markdown_text) if s.strip()]

    pieces = []
    for section in sections:
        pieces.extend(_split_to_limit(section, tokenizer, token_limit))

    pieces = _glue_orphan_headings(pieces, tokenizer, token_limit)
    return _pack(pieces, tokenizer, token_limit)

def token_length_function(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
# Checks------------------------------------------------------------------

def check_chunks(chunks, tokenizer, token_limit):
    """Print each chunk's size and assert none exceeds the limit."""
    for i, chunk in enumerate(chunks, start=1):
        n_chars = len(chunk)
        n_tokens = _count(chunk, tokenizer)
        print(f"chunk {i}: {n_chars} chars, {n_tokens} tokens")
        assert n_tokens <= token_limit, f"chunk {i} has {n_tokens} tokens (limit {token_limit})"
    print(f"OK: {len(chunks)} chunk(s), all within {token_limit} tokens")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from dotenv import load_dotenv
    import os

    load_dotenv()

    HF_TOKEN = os.getenv("HG_ACCESS_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-4b-it",
        token=HF_TOKEN
    )

    token_limit = 100

    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "documents", "pdfs")
    pdf_path = os.path.join(pdf_dir, "2410.01574v4.pdf")
    converter = PDF2MarkdownProcessor()
    document = converter.convert(pdf_path)

    chunks = split_markdown_by_headings_with_token_limit(document, tokenizer, token_limit)
    check_chunks(chunks, tokenizer, token_limit)


    print("\n--- chunks ---")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[{i}] {chunk!r}")