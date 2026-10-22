# Hybrid retrieval

The retrieval module keeps a local BM25 index beside the persistent Chroma
store. The default sparse index path is `vector_store/bm25.pkl`, matching the
default Chroma path.

## Sparse tokenization

`tokenize()` lowercases text, preserves intra-token hyphens and dots when both
sides are alphanumeric (for example `DRCT-CLIP`, `Qwen3-Reranker`, and `5.1`),
removes a small English stopword list, and does not stem. The same function is
used for both indexed chunks and queries.

## Index lifecycle

`SparseIndex.upsert_records()` is idempotent by `(source, chunk_index)`. It
serializes tokenized documents and chunk metadata atomically to `bm25.pkl`.
The CLI updates the sparse index whenever JSONL files are ingested. On startup,
it compares sparse and Chroma chunk keys and rebuilds the sparse index from
Chroma if they differ.

## Fusion

`HybridRetriever` obtains dense candidates from Chroma and BM25 candidates from
the sparse index, then fuses their ranked keys with Reciprocal Rank Fusion.
The RRF constant is `60`, the conventional value that keeps the contribution
of either rank list smooth rather than letting the first few ranks dominate.
`HybridRerankedRetriever` applies the existing cross-encoder reranker after
fusion.

## Evaluation

Run all strategies:

```powershell
python -m academic_research_assistant.retrieve_documents --ingest documents\embeddings --eval benchmarks\data\rag_eval_set.json
```

Run a focused strategy:

```powershell
python -m academic_research_assistant.retrieve_documents --strategies hybrid --eval benchmarks\data\rag_eval_set.json
```

The `--strategies` option accepts comma-separated names: `baseline`, `rewrite`,
`baseline+rerank`, `hybrid`, and `hybrid+rerank`.
