"""Standalone tests for prompt context and UI-ready citation resolution.

No LLM call is made. Run from the repository root:
    python tests/test_citations_and_prompt.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.academic_research_assistant.rag_pipeline import (  # noqa: E402
    format_context,
    resolve_citations,
)
from src.academic_research_assistant.retrieve_documents import Chunk, RetrievedChunk  # noqa: E402


def retrieved(chunk_index, text, page, heading, source="paper.pdf"):
    return RetrievedChunk(
        chunk=Chunk(source, chunk_index, text, page=page, heading=heading),
        distance=0.2,
    )


class CitationAndPromptTests(unittest.TestCase):
    def test_context_contains_internal_id_and_real_location(self):
        chunks = [retrieved(8, "The result is 80.5.", 10, "Attack Comparison")]

        context = format_context(chunks)

        self.assertIn("[1]", context)
        self.assertIn("paper.pdf", context)
        self.assertIn("page 10", context)
        self.assertIn("section Attack Comparison", context)

    def test_resolves_internal_id_to_structured_source_and_removes_marker(self):
        chunks = [retrieved(8, "The result is 80.5.", 10, "Attack Comparison")]

        answer, citations = resolve_citations("The result is 80.5 [1].", chunks)

        self.assertEqual(answer, "The result is 80.5.")
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].document, "paper.pdf")
        self.assertEqual(citations[0].page, 10)
        self.assertEqual(citations[0].section, "Attack Comparison")
        self.assertEqual(citations[0].chunk_index, 8)
        self.assertEqual(citations[0].passage, "The result is 80.5.")

    def test_invalid_internal_ids_are_not_exposed_as_citations(self):
        answer, citations = resolve_citations("Unsupported claim [99].", [])

        self.assertEqual(answer, "Unsupported claim.")
        self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
