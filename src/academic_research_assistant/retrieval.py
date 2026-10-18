"""Retrieval strategies for the RAG index.

Three independent strategies, all returning the same RetrievalResult:

    BaselineRetriever           embed the raw question, return the top-k chunks
    RewriteRetriever            LLM rewrites the question into one search query,
                                then return the top-k chunks for that query
    RerankedBaselineRetriever   baseline top-N candidates re-scored by
                                Qwen3-Reranker (Hugging Face, in-process), return the top-k

Library usage
    from retrieval_strategies import VectorIndex, BaselineRetriever, RewriteRetriever
    from retrieval_strategies import RerankedBaselineRetriever, Reranker

    index = VectorIndex()
    index.add_jsonl(Path("documents/embeddings/paper.jsonl"))

    result = BaselineRetriever(index).retrieve("What does X do?", k=5)
    for item in result.chunks:
        print(item.chunk.chunk_index, item.distance, item.chunk.text[:80])

CLI usage
    python retrieval_strategies.py --ingest documents/embeddings --ask "What does X do?"
    python retrieval_strategies.py --ingest documents/embeddings --eval cases.json

    cases.json: [{"question": "...", "gold": [13, 15], "type": "multi-part"}, ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, TypeVar

import chromadb
import ollama
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
log = logging.getLogger("retrieval_strategies")

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gemma3:4b")
# A Hugging Face model id, not an Ollama tag: the Ollama GGUF of Qwen3-Reranker returns
# uniform logits (garbage), so the reranker runs in-process with transformers instead.
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).resolve().parent / "vector_store"))
MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

MIN_CANDIDATES = 10
MAX_CANDIDATES = 20

# Context sizes are set per model: Ollama's default 4096 reserves KV-cache RAM these
# prompts never use. Keep each model's num_ctx constant (and EMBED_NUM_CTX equal to the
# one in embed_documents.py): a request with a different num_ctx forces a model reload.
#   embedder  chunks are <= 512 Gemma tokens          -> 1024 (2x tokenizer headroom)
#   rewriter  ~100-token prompt + a short JSON answer -> 512
EMBED_OPTIONS = {"num_ctx": int(os.getenv("EMBED_NUM_CTX", "1024"))}
LLM_OPTIONS = {
    "temperature": 0.3,
    "seed": 67,  # fixed seed so results are reproducible
    "num_ctx": int(os.getenv("REWRITE_NUM_CTX", "512")),
    "num_predict": 128,  # a search query never needs more; stops runaway generations
}
# Reranker input: ~90-token instruction + query + a chunk of <= 512 Gemma tokens.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "1024"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "4"))

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    distance: Optional[float]               # vector distance (lower is closer)
    rerank_score: Optional[float] = None    # P(yes) from the reranker (higher is better)


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    question: str
    search_query: str                       # the text that was actually embedded
    chunks: list[RetrievedChunk]
    latency_s: float

    @property
    def chunk_indices(self) -> list[int]:
        return [item.chunk.chunk_index for item in self.chunks]


class Retriever(Protocol):
    name: str
    models: tuple[str, ...]  # Ollama models this strategy keeps loaded

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult: ...


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _retry(call: Callable[[], T], what: str) -> T:
    """Retry transient Ollama failures with exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except (ollama.ResponseError, ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            log.warning("%s failed (attempt %d/%d): %s; retrying in %ds",
                        what, attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def embed_query(text: str) -> list[float]:
    response = _retry(
        lambda: ollama.embed(model=EMBED_MODEL, input=text, options=EMBED_OPTIONS),
        "query embedding",
    )
    return response["embeddings"][0]


def unload_models(*models: str) -> None:
    """Free models in Ollama now instead of after its idle timeout (default: all of them)."""
    for model in models or (EMBED_MODEL, REWRITE_MODEL):
        try:
            ollama.generate(model=model, keep_alive=0)
        except (ollama.ResponseError, ConnectionError) as exc:
            log.warning("Could not unload %s: %s", model, exc)


# ---------------------------------------------------------------------------
# Vector index (ChromaDB)
# ---------------------------------------------------------------------------

class VectorIndex:
    """Persistent ChromaDB collection holding embedded chunks."""

    def __init__(self, path: str | Path = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        client = chromadb.PersistentClient(path=str(path))
        self._collection = client.get_or_create_collection(name=collection_name)

    def __len__(self) -> int:
        return self._collection.count()

    def add_jsonl(self, path: Path) -> int:
        """Load records written by embed_documents.py. Re-adding is idempotent."""
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        return self.upsert_records(records)

    def upsert_records(self, records: Sequence[dict], batch_size: int = 256) -> int:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            self._collection.upsert(
                ids=[f"{r['source']}-{r['chunk_index']}" for r in batch],
                documents=[r["text"] for r in batch],
                metadatas=[{"source": r["source"], "chunk_index": r["chunk_index"]} for r in batch],
                embeddings=[r["embedding"] for r in batch],
            )
        return len(records)

    def search(self, embedding: list[float], n: int) -> list[RetrievedChunk]:
        n = min(n, len(self))
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk=Chunk(source=meta["source"], chunk_index=meta["chunk_index"], text=doc),
                distance=dist,
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


# ---------------------------------------------------------------------------
# Strategy 1: baseline
# ---------------------------------------------------------------------------

class BaselineRetriever:
    """Embed the raw question and return the nearest chunks."""

    name = "baseline"
    models = (EMBED_MODEL,)

    def __init__(self, index: VectorIndex):
        self.index = index

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        start = time.perf_counter()
        chunks = self.index.search(embed_query(question), k)
        return RetrievalResult(
            strategy=self.name,
            question=question,
            search_query=question,
            chunks=chunks,
            latency_s=time.perf_counter() - start,
        )


# ---------------------------------------------------------------------------
# Strategy 2: query rewriting
# ---------------------------------------------------------------------------

REWRITE_PROMPT = """You are an expert search query optimizer. Your task is to rewrite the user's input into a precise, self-contained search query for a vector database. Remove filler words, infer missing context from any previous turns, and use domain-specific terminology.
Return only the rewritten query as JSON: {{"query": "..."}}. No explanation, no markdown.

User Query: "{user_query}"

Rewritten Search Query:
"""


class GeneralizedQuery(BaseModel):
    query: str


def rewrite_query(question: str) -> str:
    """Rewrite the question into one search query. Falls back to the raw question."""
    response = _retry(
        lambda: ollama.chat(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(user_query=question)}],
            format=GeneralizedQuery.model_json_schema(),  # constrain output to {"query": ...}
            options=LLM_OPTIONS,
            think=False,
        ),
        "query rewrite",
    )
    rewritten = GeneralizedQuery.model_validate_json(response.message.content).query
    rewritten = rewritten.strip().strip('"*').strip()
    if not rewritten:
        log.warning("Empty rewrite for %r; using the raw question", question)
        return question
    return rewritten


class RewriteRetriever:
    """Rewrite the question with an LLM, then search with the rewritten query."""

    name = "rewrite"
    models = (REWRITE_MODEL, EMBED_MODEL)

    def __init__(self, index: VectorIndex):
        self.index = index

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        start = time.perf_counter()
        rewritten = rewrite_query(question)
        chunks = self.index.search(embed_query(rewritten), k)
        return RetrievalResult(
            strategy=self.name,
            question=question,
            search_query=rewritten,
            chunks=chunks,
            latency_s=time.perf_counter() - start,
        )


# ---------------------------------------------------------------------------
# Strategy 3: baseline candidates + reranker
# ---------------------------------------------------------------------------

RERANK_INSTRUCTION = (
    "Given a question about an academic paper, judge whether the document "
    "contains the information needed to answer it."
)
RERANK_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)

# Might not even consider it (Very slow on CPU inference)
class Reranker:
    """Scores (query, document) pairs with Qwen3-Reranker via Hugging Face transformers.

    The model answers "yes" or "no"; the relevance score is P(yes) from a softmax
    over the "yes" and "no" logits at the last position. Documents are scored in
    batches, in-process, so no Ollama runner is involved.

    The model (~2.4 GB in float32) is loaded on first use and freed by close().
    """

    def __init__(self, model: str = RERANK_MODEL, instruction: str = RERANK_INSTRUCTION,
                 max_length: int = RERANK_MAX_LENGTH, batch_size: int = RERANK_BATCH_SIZE):
        self.model_name = model
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        # Reranking is the slowest step on CPU; repeated eval runs in the same
        # process (e.g. a notebook) reuse earlier scores.
        self._scores: dict[tuple[str, str], float] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imported here so the other strategies never pay for importing torch.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = os.getenv("HG_ACCESS_TOKEN")
        start = time.perf_counter()
        # Left padding keeps every sequence's last token at position -1.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.float32, token=token
        ).eval()
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix_ids = self._tokenizer.encode(
            f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n", add_special_tokens=False
        )
        self._suffix_ids = self._tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        )
        log.info("Loaded reranker %s in %.1fs", self.model_name, time.perf_counter() - start)

    def close(self) -> None:
        """Free the model's memory; it is reloaded on the next score() call."""
        self._model = self._tokenizer = None
        import gc
        gc.collect()

    def _score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        import torch

        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]
        budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
        encoded = self._tokenizer(pairs, padding=False, truncation="longest_first",
                                  return_attention_mask=False, max_length=budget)
        encoded["input_ids"] = [self._prefix_ids + ids + self._suffix_ids for ids in encoded["input_ids"]]
        inputs = self._tokenizer.pad(encoded, padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**inputs).logits[:, -1, :]
        yes_no = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
        return torch.softmax(yes_no.float(), dim=1)[:, 1].tolist()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        missing = list(dict.fromkeys(doc for doc in documents if (query, doc) not in self._scores))
        if missing:
            self._load()
            # Batch documents of similar length together to minimise padding.
            missing.sort(key=len)
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                for doc, score in zip(batch, self._score_batch(query, batch)):
                    self._scores[(query, doc)] = score
        return [self._scores[(query, doc)] for doc in documents]


class RerankedBaselineRetriever:
    """Baseline candidates re-scored by the reranker; the top-k by rerank score are returned."""

    name = "baseline+rerank"
    models = (EMBED_MODEL,)  # the reranker runs in-process, not in Ollama

    def __init__(self, index: VectorIndex, reranker: Reranker, candidates: int = 15):
        if not MIN_CANDIDATES <= candidates <= MAX_CANDIDATES:
            raise ValueError(f"candidates must be between {MIN_CANDIDATES} and {MAX_CANDIDATES}")
        self.index = index
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        if k > self.candidates:
            raise ValueError(f"k ({k}) cannot exceed candidates ({self.candidates})")
        start = time.perf_counter()
        pool = self.index.search(embed_query(question), self.candidates)
        scores = self.reranker.score(question, [item.chunk.text for item in pool])
        ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)[:k]
        chunks = [
            RetrievedChunk(chunk=item.chunk, distance=item.distance, rerank_score=score)
            for item, score in ranked
        ]
        return RetrievalResult(
            strategy=self.name,
            question=question,
            search_query=question,
            chunks=chunks,
            latency_s=time.perf_counter() - start,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    question: str
    gold: frozenset[int]  # gold chunk indices; assumes a single source document
    qtype: str = "lookup"


def score_retrieval(retrieved: Sequence[int], gold: frozenset[int]) -> dict[str, float]:
    first_hit = next((rank for rank, idx in enumerate(retrieved) if idx in gold), None)
    return {
        "hit@1": float(bool(retrieved) and retrieved[0] in gold),
        "hit@5": float(first_hit is not None),
        "recall@5": len(gold & set(retrieved)) / len(gold),
        "mrr": 0.0 if first_hit is None else 1.0 / (first_hit + 1),
    }


def evaluate(retriever: Retriever, cases: Sequence[TestCase], k: int = 5) -> list[dict]:
    rows = []
    for case in cases:
        result = retriever.retrieve(case.question, k=k)
        rows.append({
            "strategy": retriever.name,
            "question": case.question,
            "type": case.qtype,
            "gold": sorted(case.gold),
            "retrieved": result.chunk_indices,
            "latency_s": result.latency_s,
            **score_retrieval(result.chunk_indices, case.gold),
        })
    return rows


def load_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        TestCase(question=item["question"], gold=frozenset(item["gold"]), qtype=item.get("type", "lookup"))
        for item in raw
    ]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(result: RetrievalResult) -> None:
    print(f"\n=== {result.strategy} ({result.latency_s:.2f}s) ===")
    if result.search_query != result.question:
        print(f"search query: {result.search_query}")
    for rank, item in enumerate(result.chunks, start=1):
        scores = []
        if item.distance is not None:
            scores.append(f"distance={item.distance:.4f}")
        if item.rerank_score is not None:
            scores.append(f"rerank={item.rerank_score:.4f}")
        print(f"#{rank} chunk {item.chunk.chunk_index} ({', '.join(scores)})")
        print(f"   {item.chunk.text[:150]!r}")


def _print_eval(rows: list[dict]) -> None:
    import statistics

    by_strategy: dict[str, list[dict]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    print(f"\n{'strategy':<20}{'hit@1':>8}{'hit@5':>8}{'recall@5':>10}{'mrr':>8}{'latency':>10}")
    for name, group in by_strategy.items():
        print(
            f"{name:<20}"
            f"{statistics.mean(r['hit@1'] for r in group):>8.3f}"
            f"{statistics.mean(r['hit@5'] for r in group):>8.3f}"
            f"{statistics.mean(r['recall@5'] for r in group):>10.3f}"
            f"{statistics.mean(r['mrr'] for r in group):>8.3f}"
            f"{statistics.mean(r['latency_s'] for r in group):>9.2f}s"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval strategies over embedded documents.")
    parser.add_argument("--ingest", type=Path, help="a JSONL file or a directory of JSONL files")
    parser.add_argument("--ask", help="question to run through every strategy")
    parser.add_argument("--eval", type=Path, help="JSON file of test cases to score every strategy")
    parser.add_argument("-k", type=int, default=5, help="chunks returned per strategy")
    parser.add_argument("--candidates", type=int, default=15,
                        help=f"baseline candidates re-scored by the reranker ({MIN_CANDIDATES}-{MAX_CANDIDATES})")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    index = VectorIndex()
    if args.ingest:
        files = [args.ingest] if args.ingest.is_file() else sorted(args.ingest.glob("*.jsonl"))
        for path in files:
            print(f"Indexed {index.add_jsonl(path)} chunk(s) from {path.name}")
    if len(index) == 0:
        print("error: the index is empty; pass --ingest with embedded JSONL files")
        return 1

    strategies: list[Retriever] = [
        BaselineRetriever(index),
        RewriteRetriever(index),
        RerankedBaselineRetriever(index, Reranker(), candidates=args.candidates),
    ]

    cases = load_cases(args.eval) if args.eval else []
    rows: list[dict] = []
    # Run one strategy at a time and unload each model once no later strategy needs it,
    # so only the current strategy's models stay resident.
    for position, strategy in enumerate(strategies):
        if args.ask:
            _print_result(strategy.retrieve(args.ask, k=args.k))
        if cases:
            rows.extend(evaluate(strategy, cases, k=args.k))
        still_needed = {model for later in strategies[position + 1:] for model in later.models}
        unload_models(*(model for model in strategy.models if model not in still_needed))

    if rows:
        _print_eval(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())