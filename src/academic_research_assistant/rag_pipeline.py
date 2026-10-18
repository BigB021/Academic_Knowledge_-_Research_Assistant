"""Retrieval-gated answer generation.

    question -> retriever (top-k chunks) -> AnswerGate -> LLM answer, or abstain

The gate looks at the cosine similarity of the closest chunk:

    similarity >= accept_at                 answer
    similarity <  reject_below              abstain without calling the LLM
    in between (borderline)                 rerank the top chunks; answer only if the best
                                            reranker P(yes) >= rerank_min
    no reranker given                       one cut: answer if similarity >= threshold

Thresholds are calibrated by evaluate_rag.py on questions the corpus can and cannot
answer, and saved to GATE_CONFIG. The gate is a cheap first filter: the generation
prompt still tells the LLM to abstain when the context is insufficient.

Library usage
    from academic_research_assistant.rag_pipeline import AnswerGate, answer_question

    gate = AnswerGate.from_config(reranker=Reranker())   # reranker optional
    result = answer_question("What does X do?", BaselineRetriever(index), gate)
    print(result.decision.reason, result.answer)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import ollama
from dotenv import load_dotenv

from academic_research_assistant.retrieve_documents import (
    GPU_OPTIONS,
    Reranker,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    _retry,
)

load_dotenv()
log = logging.getLogger("rag_pipeline")

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma3:4b")
GATE_CONFIG = Path(os.getenv("GATE_CONFIG", str(Path(__file__).resolve().parent / "answer_gate.json")))
# 5 chunks of <= 512 Gemma tokens + the instructions fit in 4096. The judge in
# evaluate_rag.py uses the same model and num_ctx, so switching between them never
# reloads the model.
GENERATION_OPTIONS = {
    "temperature": 0.0,
    "seed": 67,
    "num_ctx": int(os.getenv("GENERATION_NUM_CTX", "4096")),
    "num_predict": 512,
    **GPU_OPTIONS,
}

ABSTAIN_MESSAGE = "I cannot answer this from the provided documents."

ANSWER_PROMPT = """You are a research assistant for a corpus of academic papers.
Answer the question using ONLY the context below. Do not use outside knowledge.
If the context does not contain the information needed to answer, reply with exactly this sentence and nothing else:
{abstain}

### CONTEXT:
{context}

### QUESTION:
{question}

### ANSWER:
"""

# Refusals the model writes in its own words instead of the exact sentence.
_ABSTAIN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\W*(i\s+)?(cannot|can't|can not|am unable to|am not able to)\s+answer",
        r"^\W*(the\s+)?(provided\s+)?(context|documents?|passages?)\s+(is|are|do(es)?)\s+not\s+"
        r"(contain|provide|include|mention|sufficient|enough)",
        r"^\W*(the\s+)?(provided\s+)?(context|documents?)\s+(is|are)\s+insufficient",
        r"^\W*there\s+is\s+(no|not enough)\s+information",
    )
]


def is_abstention(answer: str) -> bool:
    text = answer.strip()
    return ABSTAIN_MESSAGE.lower().rstrip(".") in text.lower() or any(p.search(text) for p in _ABSTAIN_PATTERNS)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateThresholds:
    threshold: float          # single cut on top-1 similarity, used when there is no reranker
    reject_below: float       # with a reranker: abstain below this without reranking
    accept_at: float          # with a reranker: answer at or above this without reranking
    rerank_min: float = 0.5   # borderline band: answer if the best reranker P(yes) >= this

    def __post_init__(self):
        if not self.reject_below <= self.threshold <= self.accept_at:
            raise ValueError(f"expected reject_below <= threshold <= accept_at, got {self}")

    def save(self, path: Path = GATE_CONFIG, **calibration: object) -> None:
        """Write the thresholds, plus any calibration details worth keeping next to them."""
        path.write_text(json.dumps({**asdict(self), "calibration": calibration}, indent=2) + "\n",
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path = GATE_CONFIG) -> GateThresholds:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; calibrate it first: python -m academic_research_assistant.evaluate_rag --calibrate"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{name: raw[name] for name in ("threshold", "reject_below", "accept_at", "rerank_min")})


@dataclass(frozen=True)
class GateDecision:
    answer: bool
    reason: str                       # accept | reject | rerank-accept | rerank-reject
    top_similarity: float
    top_rerank: Optional[float]       # best reranker P(yes); None when the reranker did not run
    chunks: list[RetrievedChunk]      # context for the LLM, in reranked order when reranked


class AnswerGate:
    def __init__(self, thresholds: GateThresholds, reranker: Optional[Reranker] = None):
        self.thresholds = thresholds
        self.reranker = reranker

    @classmethod
    def from_config(cls, path: Path = GATE_CONFIG, reranker: Optional[Reranker] = None) -> AnswerGate:
        return cls(GateThresholds.load(path), reranker)

    def is_borderline(self, similarity: float) -> bool:
        return self.thresholds.reject_below <= similarity < self.thresholds.accept_at

    def decide(self, question: str, result: RetrievalResult) -> GateDecision:
        t = self.thresholds
        top = result.top_similarity
        if self.reranker is None:
            ok = top >= t.threshold
            return GateDecision(ok, "accept" if ok else "reject", top, None, result.chunks)
        if top >= t.accept_at:
            return GateDecision(True, "accept", top, None, result.chunks)
        if top < t.reject_below:
            return GateDecision(False, "reject", top, None, result.chunks)

        scores = self.reranker.score(question, [item.chunk.text for item in result.chunks])
        reranked = sorted(
            (RetrievedChunk(item.chunk, item.distance, score) for item, score in zip(result.chunks, scores)),
            key=lambda item: item.rerank_score,
            reverse=True,
        )
        best = reranked[0].rerank_score if reranked else 0.0
        ok = best >= t.rerank_min
        return GateDecision(ok, "rerank-accept" if ok else "rerank-reject", top, best, reranked)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{rank}] {item.chunk.text}" for rank, item in enumerate(chunks, start=1))


def generate_answer(question: str, chunks: Sequence[RetrievedChunk], model: str = GENERATION_MODEL) -> str:
    prompt = ANSWER_PROMPT.format(abstain=ABSTAIN_MESSAGE, context=format_context(chunks), question=question)
    response = _retry(
        lambda: ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=GENERATION_OPTIONS,
            think=False,
        ),
        "answer generation",
    )
    return response.message.content.strip()


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    retrieval: RetrievalResult
    decision: GateDecision
    answer: str
    abstained: bool       # the gate refused, or the LLM said the context is insufficient
    generated: bool       # False when the gate abstained before calling the LLM
    latency_s: float


def answer_question(question: str, retriever: Retriever, gate: AnswerGate, k: int = 5,
                    retrieval: Optional[RetrievalResult] = None) -> RAGAnswer:
    """Retrieve (unless a retrieval result is passed in), gate, then generate."""
    start = time.perf_counter()
    retrieval = retrieval or retriever.retrieve(question, k=k)
    decision = gate.decide(question, retrieval)
    if not decision.answer:
        log.info("Abstaining (%s, similarity %.3f)", decision.reason, decision.top_similarity)
        return RAGAnswer(question, retrieval, decision, ABSTAIN_MESSAGE, abstained=True, generated=False,
                         latency_s=time.perf_counter() - start)
    answer = generate_answer(question, decision.chunks)
    return RAGAnswer(question, retrieval, decision, answer, abstained=is_abstention(answer), generated=True,
                     latency_s=time.perf_counter() - start)
