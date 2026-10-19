"""Optional end-to-end prompt inference test using the configured local LLM.

This is intentionally separate from deterministic tests because it requires
Ollama and the configured generation model:
    python tests/test_prompt_inference.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.academic_research_assistant.rag_pipeline import (  # noqa: E402
    GENERATION_MODEL,
    generate_answer,
)
from src.academic_research_assistant.retrieve_documents import Chunk, RetrievedChunk  # noqa: E402


class PromptInferenceTests(unittest.TestCase):
    def test_local_model_answers_with_an_internal_citation(self):
        chunks = [
            RetrievedChunk(
                Chunk("paper.pdf", 8, "The detector achieved an AUROC of 80.5.", 10, "Results"),
                distance=0.1,
            )
        ]

        answer = generate_answer("What AUROC did the detector achieve?", chunks)

        self.assertTrue(answer.strip())
        self.assertIn("[1]", answer)


if __name__ == "__main__":
    print(f"Using local generation model: {GENERATION_MODEL}")
    unittest.main()
