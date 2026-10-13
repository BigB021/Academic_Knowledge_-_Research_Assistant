import sys
from pathlib import Path
import ollama
import os
from transformers import AutoTokenizer
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.PDF2MarkdownProcessor import PDF2MarkdownProcessor
from .markdown_splitter import split_markdown_by_headings_with_token_limit



load_dotenv()

HF_TOKEN = os.getenv("HG_ACCESS_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-it",
    token=HF_TOKEN
)

token_limit = 100

pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "documents", "pdfs")
pdf_path = os.path.join(pdf_dir, "Face_Anti_Spoofing_with_Swin_Tiny__Transfer_Learning__Class_Imbalance_Mitigation__and_Adversarial_Robustness_on_LCC_FASD__1_.pdf")
converter = PDF2MarkdownProcessor()
stats = converter.get_document_statistics(pdf_path)
print(f"[INFO] Document Statistics: {stats}")
print(f"[INFO] Converting PDF to Markdown: {pdf_path}")
document = converter.convert(pdf_path)
print(f"[+] Conversion complete. Markdown length: {len(document)} characters")
print(f"[INFO] Splitting Markdown into chunks with token limit: {token_limit}")
chunks = split_markdown_by_headings_with_token_limit(document, tokenizer, token_limit)
print(f"[+] Total Chunks Created: {len(chunks)}")
vector_embeddings = []
for i, chunk in enumerate(chunks, start=1):
    # Generate embedding for each chunk using Ollama
    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=chunk
    )
    
    # Extract the embedding vector array
    embedding = response['embeddings'][0]
    
    vector_embeddings.append(embedding)
    
    print(f"Chunk {i}:")
    print(f"Embedding Vector Length: {len(embedding)}")
    print(f"Sample values: {embedding[:5]}")
    print("-" * 50)
