# RAG module tests

Each script tests one major stage independently:

| Script | Coverage | External model required |
| --- | --- | --- |
| `test_markdown_splitter.py` | Heading-aware chunking and token limits | No |
| `test_pdf_markdown_processor.py` | Page and heading metadata | No |
| `test_embedding_records.py` | Page-aware embedding records | No |
| `test_retrieval_metadata.py` | Chroma metadata propagation | No |
| `test_retrieval_metrics.py` | Retrieval metrics and gate validation | No |
| `test_citations_and_prompt.py` | Internal IDs, prompt context, resolved citations | No |
| `test_prompt_inference.py` | Actual local LLM prompt inference | Ollama |

Run an individual test with:

```powershell
python tests\test_citations_and_prompt.py
```

Run all deterministic tests:

```powershell
Get-ChildItem tests\test_*.py |
  Where-Object { $_.Name -ne "test_prompt_inference.py" } |
  ForEach-Object { python $_.FullName }
```

Run the model-dependent test only after Ollama is running and the configured
generation model is available:

```powershell
python tests\test_prompt_inference.py
```
