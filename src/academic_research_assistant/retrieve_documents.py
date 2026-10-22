"""Retrieval strategies for the RAG index.

Four independent strategies, all returning the same RetrievalResult:

    BaselineRetriever           embed the raw question, return the top-k chunks
    RewriteRetriever            LLM rewrites the question into one search query,
                                then return the top-k chunks for that query
    RerankedBaselineRetriever   baseline top-N candidates re-scored by
                                Qwen3-Reranker (Hugging Face, in-process), return the top-k
    SparseRetriever             BM25 sparse retrieval, return the top-k chunks

Library usage
    from retrieval_strategies import VectorIndex, BaselineRetriever, RewriteRetriever
    from retrieval_strategies import RerankedBaselineRetriever, Reranker

    index = VectorIndex()
    index.add_jsonl(Path("documents/embeddings/paper.jsonl"))

    result = BaselineRetriever(index).retrieve("What does X do?", k=3)
    for item in result.chunks:
        print(item.chunk.chunk_index, item.distance, item.chunk.text[:80])

CLI usage
    python retrieval_strategies.py --ingest documents/embeddings --ask "What does X do?"
    python retrieval_strategies.py --ingest documents/embeddings --eval cases.json

    cases.json: [{"question": "...", "gold": [13, 15], "type": "multi-part"}, ...]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import pickle
import re
import tempfile
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

# Context sizes are set per model: Ollama's default 4096 reserves KV-cache RAM thes prompts never use. 
# Keep each model's num_ctx constant (and EMBED_NUM_CTX equal to the one in embed_documents.py): 
# a request with a different num_ctx forces a model reload.
#   embedder  chunks are <= 512 Gemma tokens          -> 1024 (2x tokenizer headroom)
#   rewriter  ~100-token prompt + a short JSON answer -> 512
# Ollama picks GPU or CPU on its own (CUDA, ROCm or Metal when available). OLLAMA_NUM_GPU
# overrides how many layers it offloads to the GPU: 0 forces CPU, 999 offloads everything.
# Unset leaves Ollama's automatic choice.
GPU_OPTIONS = {"num_gpu": int(os.environ["OLLAMA_NUM_GPU"])} if os.getenv("OLLAMA_NUM_GPU") else {}
EMBED_OPTIONS = {"num_ctx": int(os.getenv("EMBED_NUM_CTX", "1024")), **GPU_OPTIONS}
LLM_OPTIONS = {
    "temperature": 0.3,
    "seed": 67,  # fixed seed so results are reproducible
    "num_ctx": int(os.getenv("REWRITE_NUM_CTX", "512")),
    "num_predict": 128,  # a search query never needs more; stops runaway generations
    **GPU_OPTIONS,
}
# Reranker input: ~90-token instruction + query + a chunk of <= 512 Gemma tokens.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "1024"))
# "auto" picks cuda, then mps (Apple silicon), then cpu; or name one, e.g. "cuda:1".
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto")
# Unset: 16 on a GPU, 4 on CPU (larger CPU batches only add padding work).
RERANK_BATCH_SIZE = int(os.environ["RERANK_BATCH_SIZE"]) if os.getenv("RERANK_BATCH_SIZE") else None

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_index: int
    text: str
    page: Optional[int] = None
    heading: Optional[str] = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    distance: Optional[float]               # vector distance (lower is closer)
    rerank_score: Optional[float] = None    # P(yes) from the reranker (higher is better)
    channel_ranks: Optional[dict[str, int | None]] = None

    @property
    def similarity(self) -> Optional[float]:
        """Cosine similarity in [-1, 1]. The collection uses Chroma's default squared-L2
        space and the embedder returns unit vectors, so ||a - b||^2 = 2 - 2 cos(a, b)."""
        return None if self.distance is None else 1.0 - self.distance / 2.0


@dataclass(frozen=True)
class RetrievalResult:
    strategy: str
    question: str
    search_query: str                       # the text that was actually embedded
    chunks: list[RetrievedChunk]
    latency_s: float
    channel_ranks: Optional[dict[tuple[str, int], dict[str, int | None]]] = None

    @property
    def chunk_indices(self) -> list[int]:
        return [item.chunk.chunk_index for item in self.chunks]

    @property
    def chunk_keys(self) -> list[tuple[str, int]]:
        """(source, chunk_index) pairs: chunk indices alone collide across documents."""
        return [(item.chunk.source, item.chunk.chunk_index) for item in self.chunks]

    @property
    def top_similarity(self) -> float:
        """Similarity of the closest chunk; -1.0 when nothing was retrieved."""
        return max((item.similarity for item in self.chunks if item.similarity is not None), default=-1.0)


class Retriever(Protocol):
    name: str
    models: tuple[str, ...]  # Ollama models this strategy keeps loaded

    def retrieve(self, question: str, k: int = 3) -> RetrievalResult: ...


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
# Vector index (ChromaDB) Dense Retrieval
# ---------------------------------------------------------------------------

class VectorIndex:
    """Persistent ChromaDB collection holding embedded chunks."""

    def __init__(self, path: str | Path = CHROMA_PATH, collection_name: str = COLLECTION_NAME):
        client = chromadb.PersistentClient(path=str(path))
        self._collection = client.get_or_create_collection(name=collection_name)

    def __len__(self) -> int:
        return self._collection.count()

    @property
    def keys(self) -> set[tuple[str, int]]:
        result = self._collection.get(include=["metadatas"])
        return {
            (meta["source"], int(meta["chunk_index"]))
            for meta in result["metadatas"]
        }

    def records(self) -> list[dict]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "text": text,
                "page": meta.get("page"),
                "heading": meta.get("heading"),
            }
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

    def add_jsonl(self, path: Path) -> int:
        """Load records written by embed_documents.py. Re-adding is idempotent."""
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        return self.upsert_records(records)

    def upsert_records(self, records: Sequence[dict], batch_size: int = 256) -> int:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            metadatas = []
            for record in batch:
                metadata = {
                    "source": record["source"],
                    "chunk_index": record["chunk_index"],
                }
                if record.get("page") is not None:
                    metadata["page"] = record["page"]
                if record.get("heading"):
                    metadata["heading"] = record["heading"]
                metadatas.append(metadata)
            self._collection.upsert(
                ids=[f"{r['source']}-{r['chunk_index']}" for r in batch],
                documents=[r["text"] for r in batch],
                metadatas=metadatas,
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
                chunk=Chunk(
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    text=doc,
                    page=meta.get("page"),
                    heading=meta.get("heading"),
                ),
                distance=dist,
            )
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]


# ---------------------------------------------------------------------------
# Sparse BM25 retrieval
# ---------------------------------------------------------------------------

SPARSE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "with",
})
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize technical text while preserving intra-token hyphens and dots."""
    return [
        token for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in SPARSE_STOPWORDS
    ]


class SparseIndex:
    """Persistent, dependency-free BM25 index parallel to a VectorIndex."""

    def __init__(self, path: str | Path = CHROMA_PATH):
        self.path = Path(path) / "bm25.pkl"
        self._records: dict[tuple[str, int], Chunk] = {}
        self._tokenized: dict[tuple[str, int], list[str]] = {}
        self._document_frequency: dict[str, int] = {}
        self._average_length = 0.0
        self._load()

    def __len__(self) -> int:
        return len(self._records)

    @property
    def keys(self) -> set[tuple[str, int]]:
        return set(self._records)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as fh:
            state = pickle.load(fh)
        self._records = {
            tuple(key): Chunk(**value) for key, value in state["records"].items()
        }
        self._tokenized = {
            tuple(key): list(tokens) for key, tokens in state["tokenized"].items()
        }
        self._document_frequency = state["document_frequency"]
        self._average_length = state["average_length"]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "records": {
                key: {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "page": chunk.page,
                    "heading": chunk.heading,
                }
                for key, chunk in self._records.items()
            },
            "tokenized": self._tokenized,
            "document_frequency": self._document_frequency,
            "average_length": self._average_length,
        }
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".bm25.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)
            raise

    def upsert_records(self, records: Sequence[dict]) -> int:
        for record in records:
            key = (record["source"], int(record["chunk_index"]))
            chunk = Chunk(
                source=key[0],
                chunk_index=key[1],
                text=record["text"],
                page=record.get("page"),
                heading=record.get("heading"),
            )
            self._records[key] = chunk
            self._tokenized[key] = tokenize(chunk.text)
        self._rebuild_statistics()
        self._save()
        return len(records)

    def replace_records(self, records: Sequence[dict]) -> int:
        self._records.clear()
        self._tokenized.clear()
        return self.upsert_records(records)

    def _rebuild_statistics(self) -> None:
        self._document_frequency = {
            token: sum(token in tokens for tokens in self._tokenized.values())
            for tokens in self._tokenized.values()
            for token in set(tokens)
        }
        self._average_length = (
            sum(map(len, self._tokenized.values())) / len(self._tokenized)
            if self._tokenized else 0.0
        )

    def add_jsonl(self, path: Path) -> int:
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        return self.upsert_records(records)

    def search(self, query_text: str, n: int = 10) -> list[tuple[tuple[str, int], float]]:
        query_tokens = tokenize(query_text)
        if not query_tokens or not self._records or not self._average_length:
            return []
        document_count = len(self._tokenized)
        scores: list[tuple[tuple[str, int], float]] = []
        k1, b = 1.5, 0.75
        for key, tokens in self._tokenized.items():
            length = len(tokens)
            frequencies = {token: tokens.count(token) for token in set(query_tokens)}
            score = 0.0
            for token, frequency in frequencies.items():
                if not frequency:
                    continue
                df = self._document_frequency[token]
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                score += idf * frequency * (k1 + 1) / (
                    frequency + k1 * (1 - b + b * length / self._average_length)
                )
            if score > 0:
                scores.append((key, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:n]

    def get(self, key: tuple[str, int]) -> Optional[Chunk]:
        return self._records.get(key)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, int]]], k: int = 60
) -> list[tuple[tuple[str, int], float]]:
    """Fuse ranked key lists with deterministic first-ranking tie breaking."""
    scores: dict[tuple[str, int], float] = {}
    first_seen: dict[tuple[str, int], int] = {}
    order = 0
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            if key not in first_seen:
                first_seen[key] = order
                order += 1
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))


# ---------------------------------------------------------------------------
# Strategy 1: baseline
# ---------------------------------------------------------------------------

class BaselineRetriever:
    """Embed the raw question and return the nearest chunks."""

    name = "baseline"
    models = (EMBED_MODEL,)

    def __init__(self, index: VectorIndex):
        self.index = index

    def retrieve(self, question: str, k: int = 3) -> RetrievalResult:
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

    def retrieve(self, question: str, k: int = 3) -> RetrievalResult:
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
    Only recommended for GPU inference with mid to high-end GPUs.

    The model answers "yes" or "no"; the relevance score is P(yes) from a softmax
    over the "yes" and "no" logits at the last position. Documents are scored in
    batches, in-process, so no Ollama runner is involved.

    Runs on a GPU in half precision when one is available (CUDA or Apple MPS),
    otherwise on CPU in float32. The model (~1.2 GB in half precision, ~2.4 GB in
    float32) is loaded on first use and freed by close().
    """

    def __init__(self, model: str = RERANK_MODEL, instruction: str = RERANK_INSTRUCTION,
                 max_length: int = RERANK_MAX_LENGTH, batch_size: Optional[int] = RERANK_BATCH_SIZE,
                 device: str = RERANK_DEVICE):
        self.model_name = model
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
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

        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        kind = torch.device(self.device).type
        if kind == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif kind == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32  # half precision is slower than float32 on most CPUs
        if self.batch_size is None:
            self.batch_size = 4 if kind == "cpu" else 16

        token = os.getenv("HG_ACCESS_TOKEN")
        start = time.perf_counter()
        # Left padding keeps every sequence's last token at position -1.
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype, token=token
        ).to(self.device).eval()
        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix_ids = self._tokenizer.encode(
            f"<|im_start|>system\n{RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n", add_special_tokens=False
        )
        self._suffix_ids = self._tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        )
        log.info("Loaded reranker %s on %s (%s, batch %d) in %.1fs", self.model_name, self.device,
                 str(dtype).removeprefix("torch."), self.batch_size, time.perf_counter() - start)

    def close(self) -> None:
        """Free the model's memory; it is reloaded on the next score() call."""
        if self._model is None:
            return
        self._model = self._tokenizer = None
        import gc

        import torch
        gc.collect()
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()
        elif torch.device(self.device).type == "mps":
            torch.mps.empty_cache()

    def _score_batch(self, query: str, documents: Sequence[str]) -> list[float]:
        import torch

        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]
        budget = self.max_length - len(self._prefix_ids) - len(self._suffix_ids)
        encoded = self._tokenizer(pairs, padding=False, truncation="longest_first",
                                  return_attention_mask=False, max_length=budget)
        encoded["input_ids"] = [self._prefix_ids + ids + self._suffix_ids for ids in encoded["input_ids"]]
        inputs = self._tokenizer.pad(encoded, padding=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits[:, -1, :]
        yes_no = torch.stack([logits[:, self._no_id], logits[:, self._yes_id]], dim=1)
        return torch.softmax(yes_no.float(), dim=1)[:, 1].cpu().tolist()

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

    def retrieve(self, question: str, k: int = 3) -> RetrievalResult:
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


def _hybrid_candidates(
    index: VectorIndex,
    sparse: SparseIndex,
    question: str,
    candidates: int,
    rrf_k: int,
) -> tuple[list[tuple[tuple[str, int], float]], dict[tuple[str, int], RetrievedChunk], dict[tuple[str, int], dict[str, int | None]]]:
    dense = index.search(embed_query(question), candidates)
    sparse_results = sparse.search(question, candidates)
    dense_by_key = {(
        item.chunk.source, item.chunk.chunk_index
    ): item for item in dense}
    sparse_by_key = {key: sparse.get(key) for key, _ in sparse_results}
    dense_keys = list(dense_by_key)
    sparse_keys = [key for key, _ in sparse_results]
    fused = reciprocal_rank_fusion((dense_keys, sparse_keys), k=rrf_k)
    channel_ranks = {
        key: {
            "dense": dense_keys.index(key) + 1 if key in dense_by_key else None,
            "sparse": sparse_keys.index(key) + 1 if key in sparse_keys else None,
        }
        for key, _ in fused
    }
    chunks: dict[tuple[str, int], RetrievedChunk] = {}
    for key, _ in fused:
        dense_item = dense_by_key.get(key)
        chunk = dense_item.chunk if dense_item else sparse_by_key.get(key)
        if chunk is None:
            continue
        chunks[key] = RetrievedChunk(
            chunk=chunk,
            distance=dense_item.distance if dense_item else None,
            channel_ranks=channel_ranks[key],
        )
    return fused, chunks, channel_ranks


class HybridRetriever:
    """Fuse dense Chroma candidates and sparse BM25 candidates with RRF."""

    name = "hybrid"
    models = (EMBED_MODEL,)

    def __init__(self, index: VectorIndex, sparse: SparseIndex, candidates: int = 20, rrf_k: int = 60):
        self.index = index
        self.sparse = sparse
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int = 3) -> RetrievalResult:
        if k > self.candidates:
            raise ValueError(f"k ({k}) cannot exceed candidates ({self.candidates})")
        start = time.perf_counter()
        fused, chunks_by_key, channel_ranks = _hybrid_candidates(
            self.index, self.sparse, question, self.candidates, self.rrf_k
        )
        chunks = [chunks_by_key[key] for key, _ in fused[:k] if key in chunks_by_key]
        return RetrievalResult(
            strategy=self.name,
            question=question,
            search_query=question,
            chunks=chunks,
            latency_s=time.perf_counter() - start,
            channel_ranks={key: channel_ranks[key] for key, _ in fused[:k]},
        )


class HybridRerankedRetriever:
    """Fuse dense and sparse candidates, then score the fused pool."""

    name = "hybrid+rerank"
    models = (EMBED_MODEL,)

    def __init__(
        self, index: VectorIndex, sparse: SparseIndex, reranker: Reranker,
        candidates: int = 15, rrf_k: int = 60,
    ):
        if not MIN_CANDIDATES <= candidates <= MAX_CANDIDATES:
            raise ValueError(f"candidates must be between {MIN_CANDIDATES} and {MAX_CANDIDATES}")
        self.index = index
        self.sparse = sparse
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, question: str, k: int = 3) -> RetrievalResult:
        if k > self.candidates:
            raise ValueError(f"k ({k}) cannot exceed candidates ({self.candidates})")
        start = time.perf_counter()
        fused, chunks_by_key, channel_ranks = _hybrid_candidates(
            self.index, self.sparse, question, self.candidates, self.rrf_k
        )
        pool = [chunks_by_key[key] for key, _ in fused[:self.candidates] if key in chunks_by_key]
        scores = self.reranker.score(question, [item.chunk.text for item in pool])
        ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)[:k]
        chunks = [
            RetrievedChunk(
                chunk=item.chunk,
                distance=item.distance,
                rerank_score=score,
                channel_ranks=item.channel_ranks,
            )
            for item, score in ranked
        ]
        return RetrievalResult(
            strategy=self.name,
            question=question,
            search_query=question,
            chunks=chunks,
            latency_s=time.perf_counter() - start,
            channel_ranks=channel_ranks,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestCase:
    question: str
    gold: frozenset[int]             # gold chunk indices; empty when the corpus cannot answer
    qtype: str = "lookup"
    source: Optional[str] = None     # document holding the gold chunks; None matches any source
    answer: Optional[str] = None     # reference answer, used to judge generated answers

    @property
    def answerable(self) -> bool:
        return bool(self.gold)

    def is_gold(self, key: tuple[str, int]) -> bool:
        source, index = key
        return index in self.gold and (self.source is None or source == self.source)


def score_retrieval(retrieved: Sequence[tuple[str, int]], case: TestCase, k: int = 3) -> dict[str, float]:
    """Hit@1, Hit@k, Recall@k and MRR for one answerable case (retrieved is in rank order)."""
    retrieved = list(retrieved)[:k]
    hits = [case.is_gold(key) for key in retrieved]
    first_hit = next((rank for rank, hit in enumerate(hits) if hit), None)
    return {
        "hit@1": float(bool(hits) and hits[0]),
        f"hit@{k}": float(first_hit is not None),
        f"recall@{k}": len({key[1] for key, hit in zip(retrieved, hits) if hit}) / len(case.gold),
        "mrr": 0.0 if first_hit is None else 1.0 / (first_hit + 1),
    }


def evaluate(retriever: Retriever, cases: Sequence[TestCase], k: int = 3) -> list[dict]:
    rows = []
    for case in cases:
        if not case.answerable:
            continue  # no gold chunks to score against
        result = retriever.retrieve(case.question, k=k)
        rows.append({
            "strategy": retriever.name,
            "question": case.question,
            "type": case.qtype,
            "gold": sorted(case.gold),
            "retrieved": result.chunk_indices,
            "latency_s": result.latency_s,
            **score_retrieval(result.chunk_keys, case, k),
        })
    return rows


def load_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        TestCase(
            question=item["question"],
            gold=frozenset(item.get("gold", [])),
            qtype=item.get("type", "lookup"),
            source=item.get("source"),
            answer=item.get("answer"),
        )
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
            scores.append(f"similarity={item.similarity:.4f}")
        if item.rerank_score is not None:
            scores.append(f"rerank={item.rerank_score:.4f}")
        if item.channel_ranks:
            scores.append(
                "ranks=" + ",".join(
                    f"{channel}:{rank if rank is not None else '-'}"
                    for channel, rank in item.channel_ranks.items()
                )
            )
        print(f"#{rank} chunk {item.chunk.chunk_index} ({', '.join(scores)})")
        print(f"   {item.chunk.text[:150]!r}")



def _print_eval(rows: list[dict], k: int = 3) -> None:
    import statistics

    by_strategy: dict[str, list[dict]] = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    print(f"\n{'strategy':<20}{'hit@1':>8}{f'hit@{k}':>8}{f'recall@{k}':>10}{'mrr':>8}{'latency':>10}")
    for name, group in by_strategy.items():
        print(
            f"{name:<20}"
            f"{statistics.mean(r['hit@1'] for r in group):>8.3f}"
            f"{statistics.mean(r[f'hit@{k}'] for r in group):>8.3f}"
            f"{statistics.mean(r[f'recall@{k}'] for r in group):>10.3f}"
            f"{statistics.mean(r['mrr'] for r in group):>8.3f}"
            f"{statistics.mean(r['latency_s'] for r in group):>9.2f}s"
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval strategies over embedded documents.")
    parser.add_argument("--ingest", type=Path, help="a JSONL file or a directory of JSONL files")
    parser.add_argument("--ask", help="question to run through every strategy")
    parser.add_argument("--eval", type=Path, help="JSON file of test cases to score every strategy")
    parser.add_argument("-k", type=int, default=3, help="chunks returned per strategy")
    parser.add_argument("--candidates", type=int, default=15,
                        help=f"baseline candidates re-scored by the reranker ({MIN_CANDIDATES}-{MAX_CANDIDATES})")
    parser.add_argument(
        "--strategies", default="all",
        help="comma-separated strategy names, or 'all' (default)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    index = VectorIndex()
    sparse = SparseIndex()
    if args.ingest:
        files = [args.ingest] if args.ingest.is_file() else sorted(args.ingest.glob("*.jsonl"))
        for path in files:
            print(f"Indexed {index.add_jsonl(path)} chunk(s) from {path.name}")
            sparse.add_jsonl(path)
    if len(index) == 0:
        print("error: the index is empty; pass --ingest with embedded JSONL files")
        return 1
    if sparse.keys != index.keys:
        log.warning("Sparse and dense indexes differ; rebuilding sparse index from Chroma")
        sparse.replace_records(index.records())
    if sparse.keys != index.keys:
        raise RuntimeError("Sparse and dense indexes are out of sync after ingestion")

    reranker = Reranker()
    strategies: list[Retriever] = [
        BaselineRetriever(index),
        RewriteRetriever(index),
        RerankedBaselineRetriever(index, reranker, candidates=args.candidates),
        HybridRetriever(index, sparse, candidates=args.candidates),
        HybridRerankedRetriever(index, sparse, reranker, candidates=args.candidates),
    ]
    selected = [name.strip() for name in args.strategies.split(",") if name.strip()]
    available = {strategy.name: strategy for strategy in strategies}
    unknown = set(selected) - set(available) if selected != ["all"] else set()
    if unknown:
        parser.error(f"unknown strategy: {', '.join(sorted(unknown))}")
    if selected == ["all"]:
        strategies = strategies
    else:
        strategies = [available[name] for name in selected]

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

    reranker.close()
    if rows:
        _print_eval(rows, args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())