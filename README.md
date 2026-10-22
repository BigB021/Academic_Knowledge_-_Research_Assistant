# Academic Knowledge & Research Assistant

A local-first Retrieval-Augmented Generation (RAG) system for asking
grounded questions about academic papers, course material, lecture notes,
technical reports, and other research documents.

The project converts documents into structured chunks, embeds those chunks
locally, retrieves relevant passages with dense, sparse, or hybrid search, and
uses a local language model to generate an answer with source citations. An
answer gate can abstain before generation when the retrieved evidence is too
weak.

## Features

- Local document processing and inference through [Ollama](https://ollama.com/).
- PDF, CSV, XLSX, XLSM, DOCX, and PPTX ingestion.
- Heading-aware and token-limited Markdown chunking.
- Persistent Chroma vector storage for dense semantic retrieval.
- Persistent dependency-free BM25 storage for sparse lexical retrieval.
- Hybrid dense + sparse retrieval using Reciprocal Rank Fusion (RRF).
- Optional query rewriting with a local LLM.
- Optional Qwen3 cross-encoder reranking.
- Retrieval evaluation with Hit@1, Hit@k, Recall@k, MRR, and latency.
- Retrieval-gated answer generation with calibrated abstention thresholds.
- Citation resolution from temporary context IDs to source, page, section, and
  chunk metadata.
- Tests for document processing, chunking, metadata, retrieval, prompts, and
  citations.

## Architecture

```text
Documents
   |
   v
Document conversion and page/unit extraction
   |
   v
Heading-aware Markdown splitting (<= 512 tokenizer tokens)
   |
   v
Local embedding model
   |
   +--> JSONL embedding records
   |        |
   |        v
   |    Chroma vector index
   |
   +--> BM25 sparse index (bm25.pkl)

Question
   |
   v
Selected retriever
   |
   +--> Dense semantic retrieval
   +--> Sparse BM25 retrieval
   +--> Hybrid RRF fusion
   +--> Optional cross-encoder reranking
   |
   v
Retrieved chunks
   |
   v
AnswerGate
   |
   +--> Abstain without generation
   |
   +--> Generate grounded answer
             |
             v
       Resolve [1], [2], ... citations
```

## Requirements

- Python 3.12 or newer.
- Windows, macOS, or Linux.
- [Ollama](https://ollama.com/download) for local embeddings and generation.
- Enough RAM and disk space for the selected local models. CPU-only operation
  is supported but may be slow.
- Git, if cloning the repository.

The default models are:

| Component | Default model | Runtime |
| --- | --- | --- |
| Embeddings | `qwen3-embedding:0.6b` | Ollama |
| Query rewriting | `gemma3:4b` | Ollama |
| Answer generation | `gemma3:4b` | Ollama |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` | Hugging Face Transformers |

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/BigB021/Academic_Knowledge_-_Research_Assistant.git
Set-Location Academic_Knowledge_-_Research_Assistant
```

### 2. Create the Python environment

Using `uv`:

```powershell
uv sync
```

Alternatively, create a virtual environment and install the project with your
preferred Python package manager:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### 3. Start Ollama and download the Ollama models

Make sure the Ollama application or server is running, then pull the models:

```powershell
ollama pull qwen3-embedding:0.6b
ollama pull gemma3:4b
```

The Hugging Face reranker is loaded only when a reranked strategy is used. If
the model is not already cached locally, Transformers may need network access
to download it. Set `RERANK_DEVICE=cpu` for CPU-only inference or use
`RERANK_DEVICE=cuda` on a supported CUDA system.

### 4. Configure the environment

The application reads configuration from environment variables. A local
`.env` file can be used for development, but it must never be committed.
Start from the variable names below and provide your own values:

```dotenv
EMBED_MODEL=qwen3-embedding:0.6b
REWRITE_MODEL=gemma3:4b
GENERATION_MODEL=gemma3:4b
RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
TOKENIZER_NAME=google/gemma-3-4b-it
TOKEN_LIMIT=512
COLLECTION_NAME=documents
CHROMA_PATH=src/academic_research_assistant/vector_store
```

Useful optional settings include:

| Variable | Purpose |
| --- | --- |
| `OLLAMA_HOST` | Ollama server URL; defaults to `http://127.0.0.1:11434`. |
| `OLLAMA_NUM_GPU` | GPU layers to offload; `0` forces CPU. |
| `EMBED_BATCH_SIZE` | Number of chunks embedded per request. |
| `EMBED_NUM_CTX` | Embedding context size. |
| `REWRITE_NUM_CTX` | Query-rewrite context size. |
| `RERANK_DEVICE` | `auto`, `cpu`, `cuda`, `cuda:1`, or `mps`. |
| `RERANK_BATCH_SIZE` | Reranker batch size. |
| `RERANK_MAX_LENGTH` | Maximum reranker input length. |
| `OLLAMA_MAX_RETRIES` | Retry count for transient Ollama failures. |
| `GATE_CONFIG` | Path to calibrated answer-gate thresholds. |
| `GENERATION_NUM_CTX` | Generation context size. |

Do not put Hugging Face or other credentials in documentation or source
control. If a private model requires authentication, provide its token through
your local environment.

## Ingest documents

Ingestion is a two-step process:

1. Convert and split supported documents, then create embedding JSONL files.
2. Load those JSONL files into Chroma and the sparse BM25 index.

### Step 1: create embeddings

For a directory of documents:

```powershell
uv run python -m academic_research_assistant.embed_documents `
  --input documents `
  --output documents\embeddings
```

For one document:

```powershell
uv run python -m academic_research_assistant.embed_documents `
  --input documents\pdfs\paper.pdf `
  --output documents\embeddings
```

Existing JSONL output is skipped by default. Rebuild it with:

```powershell
uv run python -m academic_research_assistant.embed_documents `
  --input documents `
  --output documents\embeddings `
  --overwrite
```

Each JSONL record contains the source, chunk index, text, page or unit
metadata, token count, and embedding vector.

### Step 2: ingest the embedding records

Ingest a directory of JSONL files:

```powershell
uv run python -m academic_research_assistant.retrieve_documents `
  --ingest documents\embeddings
```

This updates the persistent Chroma collection and the BM25 index. Re-ingestion
is idempotent by `(source, chunk_index)`.

The default stores are:

- Dense Chroma store:
  `src/academic_research_assistant/vector_store`
- Sparse BM25 index:
  `src/academic_research_assistant/vector_store\bm25.pkl`

If the sparse and dense key sets differ, the retrieval CLI rebuilds the sparse
index from Chroma records.

## Retrieval strategies

All strategies return the same `RetrievalResult` structure.

### `baseline`

Embeds the original question and retrieves the nearest Chroma chunks. This is
the simplest and usually fastest semantic baseline.

### `rewrite`

Uses the rewrite model to turn the question into a precise search query, then
performs dense retrieval with that rewritten query. It can help with vague,
conversational, or underspecified questions, but adds an LLM call and latency.

### `baseline+rerank`

Retrieves a larger dense candidate pool and scores each candidate with the
Qwen3 cross-encoder. It can improve precision, but reranking is substantially
more expensive than vector search, especially on CPU.

### `hybrid`

Runs dense retrieval and BM25 retrieval in parallel conceptually, then fuses
their ranked chunk keys with Reciprocal Rank Fusion. This is useful for
academic queries containing exact acronyms, model names, equations, numbers,
or technical terms that lexical matching can identify more reliably than
semantic embeddings alone.

### `hybrid+rerank`

Fuses dense and sparse candidates first, then reranks the combined candidate
pool. This is the most quality-oriented strategy and usually the most
expensive.

The CLI does not automatically choose a different strategy for each question.
Use `--strategies` to select the strategies to run. In programmatic RAG usage,
[`build_retriever`](src/academic_research_assistant/rag_pipeline.py) chooses
dense, reranked, hybrid, or hybrid-reranked retrieval based on which index and
reranker objects are provided.

## Ask a question from the command line

Run every available retrieval strategy:

```powershell
uv run python -m academic_research_assistant.retrieve_documents `
  --ask "What is the main contribution of this paper?"
```

Run only hybrid retrieval:

```powershell
uv run python -m academic_research_assistant.retrieve_documents `
  --strategies hybrid `
  --ask "Which model achieves the highest AUROC?"
```

Run hybrid retrieval with reranking:

```powershell
uv run python -m academic_research_assistant.retrieve_documents `
  --strategies hybrid+rerank `
  --candidates 15 `
  --ask "How does the ablation study explain the performance difference?"
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--ingest PATH` | Ingest one JSONL file or every JSONL file in a directory. |
| `--ask QUESTION` | Print retrieved chunks for a question. |
| `--eval PATH` | Evaluate retrieval against a JSON test set. |
| `--strategies NAMES` | Comma-separated strategy names or `all`. |
| `-k N` | Number of final chunks to return; default is `3`. |
| `--candidates N` | Candidate-pool size for reranking and hybrid retrieval. |
| `-v` | Enable informational logging. |

## End-to-end RAG usage

The retrieval CLI is useful for inspection and benchmarking. For grounded
answers, use the RAG pipeline:

```python
from academic_research_assistant.rag_pipeline import (
    AnswerGate,
    answer_question,
    build_retriever,
)
from academic_research_assistant.retrieve_documents import SparseIndex, VectorIndex

index = VectorIndex()
sparse = SparseIndex()

# Uses hybrid dense + BM25 retrieval and synchronizes the sparse index if needed.
retriever = build_retriever(index, sparse_index=sparse)

gate = AnswerGate.from_config()
result = answer_question(
    "What is the paper's main contribution?",
    retriever,
    gate,
    k=3,
)

print(result.answer)
for citation in result.citations:
    print(citation.document, citation.page, citation.chunk_index)
```

To use dense retrieval only:

```python
retriever = build_retriever(index)
```

To use hybrid retrieval with the cross-encoder:

```python
from academic_research_assistant.retrieve_documents import Reranker

reranker = Reranker()
retriever = build_retriever(index, sparse_index=sparse, reranker=reranker)
```

### AnswerGate behavior

The gate protects the generation model from weak retrieval results:

- Without a gate reranker, the top chunk similarity is compared with one
  calibrated threshold.
- With a gate reranker:
  - scores at or above `accept_at` are accepted immediately;
  - scores below `reject_below` abstain immediately;
  - borderline results are reranked and accepted only when the best reranker
    score reaches `rerank_min`.

When the gate rejects, the generation model is not called and the pipeline
returns:

```text
I cannot answer this from the provided documents.
```

An accepted request is still instructed to abstain if the context is
insufficient. The final answer is checked for common model-written refusal
phrases as well.

## Calibrate and evaluate the answer gate

The default evaluation set is
`benchmarks\data\rag_eval_set.json`.

Calibrate and save thresholds:

```powershell
uv run python -m academic_research_assistant.evaluate_rag `
  --calibrate
```

Calibrate with borderline reranking:

```powershell
uv run python -m academic_research_assistant.evaluate_rag `
  --calibrate `
  --rerank
```

Run retrieval and answer evaluation:

```powershell
uv run python -m academic_research_assistant.evaluate_rag `
  --rerank `
  --generate
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--cases PATH` | Evaluation JSON file. |
| `--gate-config PATH` | Threshold configuration to load or save. |
| `--calibrate` | Fit thresholds and write the gate configuration. |
| `--rerank` | Rerank borderline gate decisions. |
| `--generate` | Generate and judge answers; this is slower. |
| `--margin FLOAT` | Widen the calibrated borderline band. |
| `--out PATH` | Output per-question JSON results. |
| `-k N` | Number of retrieved chunks. |

Evaluation results are written to `benchmarks\logs` by default.

## Test suite

Run the deterministic tests with `pytest`:

```powershell
uv run pytest
```

The suite covers:

- Markdown splitting and token limits.
- PDF and Office document metadata.
- Embedding-record structure.
- Chroma metadata propagation.
- Sparse BM25 and hybrid retrieval.
- Retrieval metrics.
- Prompt formatting and citation resolution.

The model-dependent prompt test requires a running Ollama server and the
configured generation model:

```powershell
uv run pytest tests\test_prompt_inference.py
```

## Repository layout

```text
src/academic_research_assistant/
  embed_documents.py       Document conversion, chunking, and embeddings
  markdown_splitter.py     Heading-aware token-limited splitting
  retrieve_documents.py    Chroma, BM25, retrieval strategies, and CLI
  rag_pipeline.py          Answer gate, generation, and citation handling
  evaluate_rag.py          Calibration and end-to-end evaluation
  answer_gate.json         Saved calibrated gate thresholds
  vector_store/             Persistent Chroma data

tests/                      Automated tests
documents/                  Example/source documents and embedding records
benchmarks/                 Evaluation sets, notebooks, plots, and logs
utils/                      PDF and Office conversion helpers
HYBRID.md                   BM25 and hybrid retrieval notes
PLAN.md                     Project design and research plan
```

## Privacy and security

The intended deployment is local-first: documents, embeddings, retrieval, and
generation remain on the local machine when Ollama and local models are used.
However:

- Do not commit `.env` files or model credentials.
- Review model and server configuration before pointing `OLLAMA_HOST` at a
  remote machine.
- Hugging Face model downloads may use the network unless models are already
  cached and offline mode is configured.
- Keep generated indexes and private documents out of source control unless
  they are intentionally shareable.

## Known limitations

- CPU-only generation and reranking can be slow.
- Retrieval quality depends on chunking, embedding-model quality, and the
  selected strategy.
- The default CLI runs selected strategies for comparison; it does not learn a
  per-question strategy router.
- The answer gate requires calibrated thresholds in
  `src/academic_research_assistant/answer_gate.json` unless thresholds are
  provided directly in code.
- The package entry point is currently a minimal placeholder; the supported
  workflows are the module commands documented above.

## License

No license file is currently included. Add an appropriate license before
redistributing the project.
