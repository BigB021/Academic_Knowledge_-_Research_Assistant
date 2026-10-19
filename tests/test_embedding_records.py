"""Standalone tests for page-aware embedding record creation.

No Ollama model is used; the embedding call is replaced with deterministic
vectors. Run from the repository root:
    python tests/test_embedding_records.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.academic_research_assistant import embed_documents  # noqa: E402


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeConverter:
    def convert(self, pdf_path, *, page_chunks):
        self.called_with_page_chunks = page_chunks
        return [
            {"page": 4, "heading": "Results", "text": "## Results\n\nAUROC improved."},
            {"page": 5, "heading": "Results", "text": "Additional evidence."},
        ]


class EmbeddingRecordTests(unittest.TestCase):
    @patch.object(embed_documents, "embed_batch", side_effect=lambda texts: [[float(i)] for i, _ in enumerate(texts)])
    @patch.object(
        embed_documents,
        "split_markdown_by_headings_with_token_limit",
        side_effect=lambda text, tokenizer, token_limit: [text],
    )
    def test_records_keep_page_and_heading_metadata(self, _split, _embed):
        records = embed_documents.embed_document(
            Path("paper.pdf"), FakeConverter(), FakeTokenizer(), token_limit=32
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].page, 4)
        self.assertEqual(records[0].heading, "Results")
        self.assertEqual(records[1].page, 5)
        self.assertEqual(records[1].heading, "Results")
        self.assertEqual(records[0].chunk_index, 0)
        self.assertEqual(records[1].chunk_index, 1)


if __name__ == "__main__":
    unittest.main()
