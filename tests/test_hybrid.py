import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from academic_research_assistant import retrieve_documents as retrieval
from academic_research_assistant.rag_pipeline import build_retriever


RECORDS = [
    {"source": "paper.pdf", "chunk_index": 0, "text": "A general methods discussion.", "embedding": [1.0, 0.0]},
    {"source": "paper.pdf", "chunk_index": 1, "text": "DRCT-CLIP achieves 95.1 AUROC.", "embedding": [0.0, 1.0]},
    {"source": "paper.pdf", "chunk_index": 2, "text": "The experiment evaluates robustness.", "embedding": [0.7, 0.7]},
    {"source": "paper.pdf", "chunk_index": 3, "text": "Ablation results and limitations.", "embedding": [0.6, 0.8]},
    {"source": "paper.pdf", "chunk_index": 4, "text": "Training details and datasets.", "embedding": [0.8, 0.6]},
]


class FakeDenseIndex:
    def __init__(self):
        self.items = {
            (record["source"], record["chunk_index"]): retrieval.RetrievedChunk(
                retrieval.Chunk(record["source"], record["chunk_index"], record["text"]),
                distance=0.1 + record["chunk_index"],
            )
            for record in RECORDS
        }
        self.order = [0, 2, 3, 4]

    def search(self, embedding, n):
        return [self.items[("paper.pdf", index)] for index in self.order[:n]]


def test_tokenizer_preserves_technical_tokens_and_removes_stopwords():
    tokens = retrieval.tokenize("The DRCT-CLIP score was 5.1 in the paper.")
    assert "drct-clip" in tokens
    assert "5.1" in tokens
    assert "the" not in tokens
    assert retrieval.tokenize(RECORDS[1]["text"]) == retrieval.tokenize(RECORDS[1]["text"])


def test_bm25_exact_rare_term_and_persistence(tmp_path):
    sparse = retrieval.SparseIndex(tmp_path)
    sparse.upsert_records(RECORDS)
    results = sparse.search("DRCT-CLIP", n=3)
    assert results[0][0] == ("paper.pdf", 1)

    reloaded = retrieval.SparseIndex(tmp_path)
    assert reloaded.search("DRCT-CLIP", n=3) == results
    assert len(reloaded) == len(RECORDS)
    sparse.upsert_records(RECORDS)
    assert len(sparse) == len(RECORDS)


def test_rrf_ordering_ties_and_single_ranking():
    fused = retrieval.reciprocal_rank_fusion(
        [[("a", 0), ("b", 0)], [("b", 0), ("a", 0)]], k=60
    )
    assert fused[0][0] == ("a", 0)
    assert fused[0][1] == fused[1][1]
    assert [key for key, _ in retrieval.reciprocal_rank_fusion([[("b", 0), ("a", 0)]])] == [
        ("b", 0), ("a", 0)
    ]


def test_hybrid_retriever_populates_channel_ranks_and_sparse_only(monkeypatch, tmp_path):
    sparse = retrieval.SparseIndex(tmp_path)
    sparse.upsert_records(RECORDS)
    dense = FakeDenseIndex()
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [1.0, 0.0])

    result = retrieval.HybridRetriever(dense, sparse, candidates=3).retrieve("DRCT-CLIP", k=3)

    assert len(result.chunks) == 3
    assert len(result.chunk_keys) == len(set(result.chunk_keys))
    assert all(item.channel_ranks for item in result.chunks)
    assert any(item.distance is None for item in result.chunks)


def test_hybrid_rejects_k_larger_than_candidates(tmp_path):
    sparse = retrieval.SparseIndex(tmp_path)
    with pytest.raises(ValueError):
        retrieval.HybridRetriever(FakeDenseIndex(), sparse, candidates=2).retrieve("x", k=3)


def test_evaluate_accepts_hybrid_retriever(monkeypatch, tmp_path):
    sparse = retrieval.SparseIndex(tmp_path)
    sparse.upsert_records(RECORDS)
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [1.0, 0.0])
    retriever = retrieval.HybridRetriever(FakeDenseIndex(), sparse, candidates=3)
    cases = [retrieval.TestCase("DRCT-CLIP", frozenset({1}), source="paper.pdf")]

    rows = retrieval.evaluate(retriever, cases, k=3)

    assert len(rows) == 1
    assert rows[0]["strategy"] == "hybrid"


def test_hybrid_reranked_scores_fused_pool(monkeypatch, tmp_path):
    sparse = retrieval.SparseIndex(tmp_path)
    sparse.upsert_records(RECORDS)
    monkeypatch.setattr(retrieval, "embed_query", lambda question: [1.0, 0.0])

    class FakeReranker:
        def score(self, question, documents):
            return [float(len(document)) for document in documents]

    result = retrieval.HybridRerankedRetriever(
        FakeDenseIndex(), sparse, FakeReranker(), candidates=10
    ).retrieve("DRCT-CLIP", k=3)

    assert len(result.chunks) == 3
    assert all(item.rerank_score is not None for item in result.chunks)


def test_rag_builder_syncs_sparse_index_before_hybrid_retrieval(tmp_path):
    sparse = retrieval.SparseIndex(tmp_path)
    dense = FakeDenseIndex()
    dense.keys = set(dense.items)
    dense.records = lambda: RECORDS

    retriever = build_retriever(dense, sparse, candidates=10)

    assert retriever.name == "hybrid"
    assert sparse.keys == dense.keys
