"""Standalone tests for retrieval scoring and gate calibration.

Run from the repository root:
    python tests/test_retrieval_metrics.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.academic_research_assistant.retrieve_documents import TestCase, score_retrieval  # noqa: E402
from src.academic_research_assistant.rag_pipeline import GateThresholds  # noqa: E402


class RetrievalMetricTests(unittest.TestCase):
    def test_scores_hit_recall_and_mrr(self):
        case = TestCase("question", frozenset({3, 5}), source="paper.pdf")
        scores = score_retrieval([("paper.pdf", 8), ("paper.pdf", 3), ("paper.pdf", 5)], case, k=3)

        self.assertEqual(scores["hit@1"], 0.0)
        self.assertEqual(scores["hit@3"], 1.0)
        self.assertEqual(scores["recall@3"], 1.0)
        self.assertEqual(scores["mrr"], 0.5)

    def test_threshold_order_is_validated(self):
        with self.assertRaises(ValueError):
            GateThresholds(threshold=0.7, reject_below=0.8, accept_at=0.9)


if __name__ == "__main__":
    unittest.main()
