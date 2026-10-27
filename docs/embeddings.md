# AI/ML embeddings and search

Paper text and vectors are kept in separate SQLite databases:

| File | Contents |
| --- | --- |
| `data/ai-ml-papers.sqlite3` | Titles, abstracts, citation metadata, proceedings evidence and collection progress |
| `data/ai-ml-openai-embeddings.sqlite3` | OpenAI vectors, paper IDs, text hashes and model settings |

The vector database contains no paper text. Search checks IDs and text hashes
against the paper database and omits stale or missing matches. The bundled
index contains all 81,066 active papers and uses `text-embedding-3-large` with
3,072 dimensions and FP16 vector storage.

## OpenAI setup

```bash
uv sync --extra openai
```

Set `OPENAI_API_KEY` in `.env`. API mode sends paper titles/abstracts during a
build and the query during search to OpenAI; ranking uses the locally stored
vectors. For a standalone environment without uv or Torch, install
`requirements-api-embeddings.txt` and replace `uv run --extra openai` with
`python` in the examples.

```bash
uv run --extra openai embed_papers.py search \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3 \
  --query "Learning from limited labels" --top-k 50
```

Use `--query-file target.txt` for a title plus abstract, and `--json` for
structured results. Both title-only and abstract queries are embedded using
the model and dimensions stored in the index. No local model is loaded.

## Estimate, build and resume

```bash
uv run --extra openai embed_papers.py estimate \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3
uv run --extra openai embed_papers.py build \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3 \
  --model text-embedding-3-large --batch-size 128
uv run embed_papers.py status \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3
```

`estimate` counts actual chunk tokens locally. With `--index`, it estimates only
pending or changed papers. Its standard input-price defaults were checked on
2026-10-27; use `--price-per-million` for updated/account-specific pricing.
See [OpenAI's model documentation](https://developers.openai.com/api/docs/models/text-embedding-3-large).
A tokenizer vocabulary may download on first use, but estimation makes no
embedding API calls. `build`, `search`, and an OpenAI `benchmark` make paid
requests.

Add `--limit 100` for a pilot, then rerun without it to finish. The index commits
completed batches; interruptions resume saved progress. SDK retries handle
transient errors, although an uncommitted batch may need to be requested again.
Changed title/abstract text is re-embedded, while metadata-only edits reuse its
vectors. The paper database is read-only during embedding generation.

OpenAI's large model defaults to 3072 dimensions and its small model to 1536.
Specify `--dimensions` when creating a new index to request fewer. Documents
are chunked without truncation using `cl100k_base`, with an 8191-token chunk
limit by default. Requests are split at the 2048-input and 300,000-total-token
limits. Returned indices, dimensions and vector values are checked before
storage. See the [embedding API reference](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

Queries and documents must use the same provider, model, dimensions and chunk
settings. The CLI rejects incompatible resumes. OpenAI exposes a model ID,
rather than an immutable weights commit; service-side model changes cannot be
frozen like a local Hugging Face revision.

## Optional local model

Install `requirements-client.txt` for a Mac/local environment or
`requirements-server.txt` for an existing CUDA environment. The local backend
uses `Qwen/Qwen3-Embedding-4B`. Keep its vectors separate from the OpenAI index:

```bash
python embed_papers.py build --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-qwen-embeddings.sqlite3 \
  --model Qwen/Qwen3-Embedding-4B --device mps --batch-size 1 --limit 100
python embed_papers.py search --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-qwen-embeddings.sqlite3 \
  --query-file target.txt --device mps --top-k 50
```

Use `--device cuda` on a CUDA GPU or `--device cpu` on a CPU. Device selection
is automatic when omitted. Automatic inference precision is FP16 on Apple MPS,
BF16 when supported on CUDA, and FP32 on CPU. Increase the batch size only as
memory permits. The first local run downloads model weights; its revision and
prompts are saved in the index.

For a GPU server, copy a verified paper archive from `package_database.py pack`,
restore it on the server, build the Qwen index, then package and bring back only
the vector database. Keep the same paper snapshot and model settings on both
machines. The [README](../README.md) explains LFS snapshot restoration.
