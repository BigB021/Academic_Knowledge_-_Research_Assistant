"""Calibrate the answer gate and evaluate the RAG pipeline end to end.

    Retrieval    Hit@1, Recall@k and MRR on the answerable questions
    Gate         answer / abstain decisions from top-1 similarity (+ reranker when borderline)
    Generation   answer correctness and unsupported-answer rate, graded by an LLM judge

Test set (benchmarks/data/rag_eval_set.json), one object per question:
    answerable               {"question", "source", "gold": [chunk indices], "type", "answer"}
    out-of-corpus            {"question", "type": "out-of-corpus"}            no relevant chunk exists
    in-corpus-unanswerable   {"question", "type": "in-corpus-unanswerable"}   on topic, answer absent

The gate is calibrated on answerable vs out-of-corpus questions only. In-corpus-unanswerable
questions retrieve on-topic chunks by design, so no similarity cut can separate them; they
measure whether the LLM itself abstains.

Usage (from the project root)
    python -m academic_research_assistant.evaluate_rag --calibrate                # fit + save the gate
    python -m academic_research_assistant.evaluate_rag --calibrate --rerank       # also fit rerank_min
    python -m academic_research_assistant.evaluate_rag --rerank --generate        # full evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Literal, Optional, Sequence

import ollama
from pydantic import BaseModel

from academic_research_assistant.rag_pipeline import (
    GATE_CONFIG,
    GENERATION_MODEL,
    GENERATION_OPTIONS,
    AnswerGate,
    GateThresholds,
    answer_question,
    format_context,
)
from academic_research_assistant.retrieve_documents import (
    EMBED_MODEL,
    BaselineRetriever,
    Reranker,
    RetrievalResult,
    TestCase,
    VectorIndex,
    _retry,
    load_cases,
    score_retrieval,
    unload_models,
)

log = logging.getLogger("evaluate_rag")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = PROJECT_ROOT / "benchmarks" / "data" / "rag_eval_set.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "benchmarks" / "logs"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GENERATION_MODEL)
NEGATIVE_TYPE = "out-of-corpus"   # gate calibration negatives


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def best_cut(pos: Sequence[float], neg: Sequence[float]) -> tuple[float, float]:
    """The cut maximizing balanced accuracy, where score >= cut predicts positive.

    Candidates are midpoints between neighboring observed scores, so the chosen cut keeps
    a margin on both sides; among tied cuts the middle one is taken. Returns
    (cut, balanced accuracy).
    """
    values = sorted(set(pos) | set(neg))
    candidates = [values[0] - 1e-6] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [values[-1] + 1e-6]

    def balanced_accuracy(cut: float) -> float:
        tpr = sum(p >= cut for p in pos) / len(pos)
        tnr = sum(n < cut for n in neg) / len(neg)
        return (tpr + tnr) / 2

    scored = [(balanced_accuracy(cut), cut) for cut in candidates]
    best = max(score for score, _ in scored)
    tied = [cut for score, cut in scored if score == best]
    return tied[len(tied) // 2], best


def calibrate(pos: Sequence[float], neg: Sequence[float], margin: float,
              rerank_pos: Optional[Sequence[float]] = None,
              rerank_neg: Optional[Sequence[float]] = None) -> tuple[GateThresholds, dict]:
    """Fit the gate on top-1 similarities of answerable (pos) and out-of-corpus (neg) questions.

    The borderline band spans the region where the two classes are not cleanly separated:
    from the lowest answerable score to the highest out-of-corpus score when they overlap,
    or the gap between them when they do not, widened by `margin` on both sides.
    rerank_pos / rerank_neg are the best reranker scores of the questions inside that band
    whose retrieved chunks do / do not include a gold chunk (fit on all questions when the
    band holds too few of either class).
    """
    if not pos or not neg:
        raise ValueError("calibration needs answerable and out-of-corpus questions")
    threshold, balanced_acc = best_cut(pos, neg)
    low, high = sorted((min(pos), max(neg)))
    reject_below = min(low - margin, threshold)
    accept_at = max(high + margin, threshold)

    rerank_min, rerank_acc = 0.5, None
    if rerank_pos and rerank_neg:
        rerank_min, rerank_acc = best_cut(rerank_pos, rerank_neg)
    thresholds = GateThresholds(threshold=round(threshold, 4), reject_below=round(reject_below, 4),
                                accept_at=round(accept_at, 4), rerank_min=round(rerank_min, 4))
    info = {
        "n_answerable": len(pos),
        "n_out_of_corpus": len(neg),
        "threshold_balanced_accuracy": round(balanced_acc, 4),
        "rerank_balanced_accuracy": None if rerank_acc is None else round(rerank_acc, 4),
        "margin": margin,
        "answerable_similarity": _summary(pos),
        "out_of_corpus_similarity": _summary(neg),
    }
    return thresholds, info


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {"min": round(min(values), 4), "median": round(statistics.median(values), 4),
            "max": round(max(values), 4)}


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are grading the output of a question-answering system over academic papers.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CONTEXT GIVEN TO THE SYSTEM:
{context}

SYSTEM ANSWER:
{answer}

Grade the system answer and return JSON with three fields:
- "abstained": true if the system answer declines to answer or says the context does not contain the information, false otherwise.
- "correctness": compare with the reference answer. "correct" if it states the key facts of the reference, "partial" if it states only some of them, "incorrect" if it states none or contradicts the reference. If there is no reference answer, use "incorrect".
- "supported": true if every factual claim in the system answer is stated in, or directly follows from, the context given to the system. false if any claim is not in that context.
"""


class JudgeVerdict(BaseModel):
    abstained: bool
    correctness: Literal["correct", "partial", "incorrect"]
    supported: bool


def judge(question: str, reference: Optional[str], context: str, answer: str) -> JudgeVerdict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference or "(none: the corpus cannot answer this)",
                                 context=context, answer=answer)
    response = _retry(
        lambda: ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=JudgeVerdict.model_json_schema(),
            options={**GENERATION_OPTIONS, "num_predict": 128},
            think=False,
        ),
        "judge",
    )
    return JudgeVerdict.model_validate_json(response.message.content)


CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(cases: Sequence[TestCase], retriever: BaselineRetriever, k: int, reranker: Optional[Reranker],
             thresholds: Optional[GateThresholds], margin: float, generate: bool) -> tuple[list[dict], GateThresholds, Optional[dict]]:
    results: list[RetrievalResult] = []
    for number, case in enumerate(cases, start=1):
        results.append(retriever.retrieve(case.question, k=k))
        log.info("retrieved %d/%d", number, len(cases))

    rows = []
    for case, result in zip(cases, results):
        row = {
            "question": case.question,
            "type": case.qtype,
            "answerable": case.answerable,
            "gold": sorted(case.gold),
            "retrieved": [f"{source}#{index}" for source, index in result.chunk_keys],
            "top_similarity": round(result.top_similarity, 4),
        }
        if case.answerable:
            row.update(score_retrieval(result.chunk_keys, case, k))
        rows.append(row)

    info = None
    if thresholds is None:
        calib = [(case, result) for case, result in zip(cases, results)
                 if case.answerable or case.qtype == NEGATIVE_TYPE]
        pos = [r.top_similarity for c, r in calib if c.answerable]
        neg = [r.top_similarity for c, r in calib if not c.answerable]
        thresholds, info = calibrate(pos, neg, margin)
        if reranker is not None:
            # Score every calibration question, then fit rerank_min on the borderline ones.
            # The reranker judges the retrieved chunks, so its positives are answerable
            # questions whose top-k holds a gold chunk; an answerable question whose
            # retrieval missed should be refused too.
            band = AnswerGate(thresholds).is_borderline
            best = {}
            for number, (case, result) in enumerate(calib, start=1):
                best[case.question] = max(reranker.score(case.question, [i.chunk.text for i in result.chunks]))
                log.info("reranked %d/%d", number, len(calib))
            has_gold = {c.question: any(c.is_gold(key) for key in r.chunk_keys) for c, r in calib}
            in_band = [(c, r) for c, r in calib if band(r.top_similarity)]
            fit_on = in_band if (sum(has_gold[c.question] for c, _ in in_band) >= 2
                                 and sum(not has_gold[c.question] for c, _ in in_band) >= 2) else calib
            thresholds, info = calibrate(
                pos, neg, margin,
                rerank_pos=[best[c.question] for c, _ in fit_on if has_gold[c.question]],
                rerank_neg=[best[c.question] for c, _ in fit_on if not has_gold[c.question]],
            )
            info["rerank_fit_on"] = "borderline" if fit_on is in_band else "all"

    gate = AnswerGate(thresholds, reranker)
    for number, (case, result, row) in enumerate(zip(cases, results, rows), start=1):
        if generate:
            rag = answer_question(case.question, retriever, gate, k=k, retrieval=result)
            decision = rag.decision
            row.update({"answer": rag.answer, "generated": rag.generated})
            abstained = rag.abstained
            if rag.generated and not rag.abstained:
                verdict = judge(case.question, case.answer, format_context(decision.chunks), rag.answer)
                row.update(verdict.model_dump())
                abstained = verdict.abstained
            row["final_abstained"] = abstained
            log.info("answered %d/%d", number, len(cases))
        else:
            decision = gate.decide(case.question, result)
        row.update({
            "gate": decision.reason,
            "gate_answer": decision.answer,
            "top_rerank": None if decision.top_rerank is None else round(decision.top_rerank, 4),
        })
    return rows, thresholds, info


def _rate(values: Sequence[float]) -> str:
    return f"{statistics.mean(values):.3f} ({sum(values):g}/{len(values)})" if values else "n/a"


def report(rows: Sequence[dict], k: int, thresholds: GateThresholds, info: Optional[dict],
           reranker: bool, generate: bool) -> None:
    answerable = [r for r in rows if r["answerable"]]
    print(f"\n=== Retrieval (baseline, k={k}, {len(answerable)} answerable questions) ===")
    print(f"{'type':<14}{'n':>4}{'hit@1':>8}{f'recall@{k}':>11}{'mrr':>8}")
    groups = {t: [r for r in answerable if r["type"] == t] for t in dict.fromkeys(r["type"] for r in answerable)}
    for name, group in [*groups.items(), ("all", answerable)]:
        print(f"{name:<14}{len(group):>4}"
              f"{statistics.mean(r['hit@1'] for r in group):>8.3f}"
              f"{statistics.mean(r[f'recall@{k}'] for r in group):>11.3f}"
              f"{statistics.mean(r['mrr'] for r in group):>8.3f}")

    print("\n=== Gate ===")
    for qtype in dict.fromkeys(r["type"] if not r["answerable"] else "answerable" for r in rows):
        sims = [r["top_similarity"] for r in rows if (r["type"] if not r["answerable"] else "answerable") == qtype]
        s = _summary(sims)
        print(f"top-1 similarity  {qtype:<24} n={len(sims):<3} min={s['min']:.3f} "
              f"median={s['median']:.3f} max={s['max']:.3f}")
    mode = (f"band [{thresholds.reject_below:.3f}, {thresholds.accept_at:.3f}) -> rerank, "
            f"answer if P(yes) >= {thresholds.rerank_min:.3f}") if reranker else "no reranker"
    print(f"threshold={thresholds.threshold:.3f}  ({mode})")
    if info:
        print(f"calibrated in-sample: balanced accuracy {info['threshold_balanced_accuracy']:.3f} "
              f"at the threshold on {info['n_answerable']} answerable / {info['n_out_of_corpus']} out-of-corpus")
    print(f"answerable answered:            {_rate([float(r['gate_answer']) for r in answerable])}")
    for qtype in ("out-of-corpus", "in-corpus-unanswerable"):
        group = [r for r in rows if r["type"] == qtype]
        if group:
            print(f"{qtype + ' abstained:':<32}{_rate([float(not r['gate_answer']) for r in group])}")
    reranked = [r for r in rows if r["gate"].startswith("rerank")]
    print(f"borderline (reranked):          {len(reranked)}/{len(rows)}")
    for r in sorted(reranked, key=lambda r: r["top_similarity"]):
        print(f"   sim={r['top_similarity']:.3f} rerank={r['top_rerank']:.3f} {r['gate']:<14}"
              f"{'answerable' if r['answerable'] else r['type']:<24}{r['question'][:60]}")

    if not generate:
        return
    print("\n=== Generation ===")
    correctness = [0.0 if r["final_abstained"] else CORRECTNESS_SCORE[r["correctness"]] for r in answerable]
    strict = [float(not r["final_abstained"] and r["correctness"] == "correct") for r in answerable]
    print(f"answer correctness (partial = 0.5, abstention = 0): {_rate(correctness)}")
    print(f"answer correctness (strict):                        {_rate(strict)}")
    print(f"false abstention rate (answerable):                 "
          f"{_rate([float(r['final_abstained']) for r in answerable])}")
    for qtype in ("out-of-corpus", "in-corpus-unanswerable"):
        group = [r for r in rows if r["type"] == qtype]
        if group:
            print(f"{'abstention rate (' + qtype + '):':<52}{_rate([float(r['final_abstained']) for r in group])}")
    answered = [r for r in rows if not r["final_abstained"]]
    print(f"unsupported-answer rate (of answered):              "
          f"{_rate([float(not r['supported']) for r in answered])}")
    for r in answered:
        if not r["supported"] or not r["answerable"]:
            tag = "unsupported" if not r["supported"] else "answered-unanswerable"
            print(f"   [{tag}] {r['question'][:70]}\n      -> {r['answer'][:160]!r}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the answer gate and evaluate retrieval + generation.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--calibrate", action="store_true", help=f"fit the gate and save it to {GATE_CONFIG.name}")
    parser.add_argument("--gate-config", type=Path, default=GATE_CONFIG)
    parser.add_argument("--rerank", action="store_true", help="rerank borderline questions (Qwen3-Reranker)")
    parser.add_argument("--generate", action="store_true", help="generate and judge answers (slow)")
    parser.add_argument("--margin", type=float, default=0.02, help="widening of the borderline band")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--out", type=Path, help="per-question results (JSON); default: benchmarks/logs/")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    index = VectorIndex()
    if len(index) == 0:
        print("error: the index is empty; ingest documents first (retrieve_documents.py --ingest)")
        return 1
    cases = load_cases(args.cases)
    reranker = Reranker() if args.rerank else None
    thresholds = None if args.calibrate else GateThresholds.load(args.gate_config)

    start = time.perf_counter()
    try:
        rows, thresholds, info = evaluate(cases, BaselineRetriever(index), args.k, reranker, thresholds,
                                          args.margin, args.generate)
    finally:
        unload_models(EMBED_MODEL, *([GENERATION_MODEL, JUDGE_MODEL] if args.generate else []))
        if reranker is not None:
            reranker.close()

    if args.calibrate:
        thresholds.save(args.gate_config, **info, k=args.k, cases=args.cases.name, reranker=bool(reranker))
        print(f"Saved gate thresholds to {args.gate_config}")
    report(rows, args.k, thresholds, info, reranker is not None, args.generate)

    out = args.out or DEFAULT_OUT_DIR / f"rag_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"thresholds": thresholds.__dict__, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nPer-question results: {out}  ({time.perf_counter() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
