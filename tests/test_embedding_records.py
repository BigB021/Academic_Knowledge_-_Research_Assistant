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


class FakeOfficeConverter:
    def convert(self, document_path, *, units):
        self.called_with_units = units
        return [
            {
                "text": "## Sheet: Data\n\n| Name |\n| --- |\n| value |",
                "metadata": {"unit_index": 3, "heading": "Data"},
            }
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

    @patch.object(embed_documents, "embed_batch", return_value=[[1.0, 0.0]])
    @patch.object(
        embed_documents,
        "split_markdown_by_headings_with_token_limit",
        side_effect=lambda text, tokenizer, token_limit: [text],
    )
    def test_office_units_become_indexable_metadata(self, _split, _embed):
        converter = FakeOfficeConverter()
        records = embed_documents.embed_document(
            Path("workbook.xlsx"), converter, FakeTokenizer(), token_limit=32
        )

        self.assertTrue(converter.called_with_units)
        self.assertEqual(records[0].source, "workbook.xlsx")
        self.assertEqual(records[0].page, 3)
        self.assertEqual(records[0].heading, "Data")


if __name__ == "__main__":
    unittest.main()
