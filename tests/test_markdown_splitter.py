"""Standalone tests for heading-aware Markdown chunking.

Run from the repository root:
    python tests/test_markdown_splitter.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.academic_research_assistant.markdown_splitter import (  # noqa: E402
    FastTokenizer,
    split_markdown_by_headings_with_token_limit,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)


class MarkdownSplitterTests(unittest.TestCase):
    def test_preserves_headings_and_respects_limit(self):
        markdown = "# Methods\n\n" + ("token " * 20) + "\n\n## Results\n\n" + ("value " * 20)
        tokenizer = CharacterTokenizer()

        chunks = split_markdown_by_headings_with_token_limit(markdown, tokenizer, token_limit=80)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(chunk.startswith("# Methods") for chunk in chunks))
        self.assertTrue(any("## Results" in chunk for chunk in chunks))
        self.assertTrue(all(len(tokenizer.encode(chunk, add_special_tokens=False)) <= 80 for chunk in chunks))

    def test_glues_orphan_heading_to_following_content(self):
        markdown = "# Introduction\n\nShort content."
        chunks = split_markdown_by_headings_with_token_limit(markdown, CharacterTokenizer(), token_limit=100)

        self.assertEqual(len(chunks), 1)
        self.assertIn("# Introduction", chunks[0])
        self.assertIn("Short content.", chunks[0])


if __name__ == "__main__":
    unittest.main()
