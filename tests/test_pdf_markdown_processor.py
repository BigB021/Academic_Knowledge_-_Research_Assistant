"""Standalone tests for PDF page metadata enrichment.

These tests do not open a PDF or call the Markdown conversion service.
Run from the repository root:
    python tests/test_pdf_markdown_processor.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.PDF2MarkdownProcessor import PDF2MarkdownProcessor  # noqa: E402


class PdfMarkdownMetadataTests(unittest.TestCase):
    def test_assigns_one_based_page_and_heading_metadata(self):
        pages = [
            {"text": "# Introduction\n\nOverview."},
            {"text": "Continued discussion."},
            {"text": "## Method\n\nExperiment."},
        ]

        enriched = PDF2MarkdownProcessor._add_page_metadata(pages)

        self.assertEqual([page["page"] for page in enriched], [1, 2, 3])
        self.assertEqual(enriched[0]["heading"], "Introduction")
        self.assertEqual(enriched[1]["heading"], "Introduction")
        self.assertEqual(enriched[2]["heading"], "Method")

    def test_rejects_malformed_page_chunks(self):
        with self.assertRaises(Exception):
            PDF2MarkdownProcessor._add_page_metadata([{"metadata": {}}])


if __name__ == "__main__":
    unittest.main()
