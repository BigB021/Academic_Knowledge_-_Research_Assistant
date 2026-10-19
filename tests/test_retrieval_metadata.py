"""Standalone tests for vector-index metadata propagation.

The Chroma collection is replaced with a fake collection, so no database or
embedding model is required.
Run from the repository root:
    python tests/test_retrieval_metadata.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.academic_research_assistant.retrieve_documents import VectorIndex  # noqa: E402


class FakeCollection:
    def __init__(self):
        self.metadata = None

    def count(self):
        return 1

    def upsert(self, **kwargs):
        self.metadata = kwargs["metadatas"][0]


class RetrievalMetadataTests(unittest.TestCase):
    def test_upsert_persists_page_and_heading(self):
        index = object.__new__(VectorIndex)
        collection = FakeCollection()
        index._collection = collection

        index.upsert_records(
            [{
                "source": "paper.pdf",
                "chunk_index": 7,
                "page": 10,
                "heading": "Attack Comparison",
                "text": "AUROC comparison.",
                "embedding": [1.0, 0.0],
            }]
        )

        self.assertEqual(collection.metadata["page"], 10)
        self.assertEqual(collection.metadata["heading"], "Attack Comparison")


if __name__ == "__main__":
    unittest.main()
