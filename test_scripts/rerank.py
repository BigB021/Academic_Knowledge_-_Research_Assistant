"""Proof of concept: score (query, document) pairs with Qwen3-Reranker-0.6B.

The Ollama build (dengcao/Qwen3-Reranker-0.6B:Q8_0) cannot be used: its output
layer is broken, so every token gets the same probability and even a plain
completion returns ",,,,". Here the official Hugging Face model runs in-process.

Qwen3-Reranker is a causal LM that answers "yes" or "no". The relevance score
is P(yes) from a softmax over the "yes" and "no" logits at the last position.
All documents are scored in one forward pass instead of one request each.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HG_ACCESS_TOKEN")

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MAX_LENGTH = 2048
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query = "What is the capital of China?"
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]


class Qwen3Reranker:
    def __init__(self, model_name=MODEL_NAME, instruction=INSTRUCTION):
        self.instruction = instruction
        # Left padding keeps every sequence's last token at position -1.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def _encode(self, query, docs):
        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in docs]
        budget = MAX_LENGTH - len(self.prefix_ids) - len(self.suffix_ids)
        encoded = self.tokenizer(pairs, padding=False, truncation="longest_first",
                                 return_attention_mask=False, max_length=budget)
        encoded["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in encoded["input_ids"]]
        return self.tokenizer.pad(encoded, padding=True, return_tensors="pt")

    @torch.inference_mode()
    def score(self, query, docs, batch_size=8):
        """Return P(yes) for each document, in input order."""
        scores = []
        for start in range(0, len(docs), batch_size):
            inputs = self._encode(query, docs[start:start + batch_size])
            logits = self.model(**inputs).logits[:, -1, :]
            yes_no = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1)
            scores.extend(torch.softmax(yes_no, dim=1)[:, 1].tolist())
        return scores


def rerank(reranker, query, documents, top_k=5):
    scores = reranker.score(query, documents)
    scored = [{"text": doc, "rerank_score": score} for doc, score in zip(documents, scores)]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    start = time.perf_counter()
    reranker = Qwen3Reranker()
    print(f"Loaded {MODEL_NAME} in {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    reranked_docs = rerank(reranker, query, documents)
    print(f"Scored {len(documents)} documents in {time.perf_counter() - start:.2f}s")
    for doc in reranked_docs:
        print(f"Score: {doc['rerank_score']:.4f}, Document: {doc['text']}")
