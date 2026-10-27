# AI/ML Reference Finder

Find papers related to a title, abstract, or research question across **ICLR,
ICML, NeurIPS, AAAI, and COLM**. The repository includes a searchable paper
corpus and a matching OpenAI embedding index, so you can start searching
without collecting papers or generating their embeddings again.

The bundled snapshot contains **81,066 papers with abstracts** from the
2017–2026 collection window. Its index uses **`text-embedding-3-large`, 3,072
dimensions**, with vectors stored locally in SQLite. Current-year proceedings
coverage is provisional; these counts describe the collected corpus.

## Quick start

Install [Git LFS](https://git-lfs.com/) and [uv](https://docs.astral.sh/uv/), then:

```bash
git lfs install
git clone https://github.com/yuxuan18/reference_finder_aiml.git
cd reference_finder_aiml
git lfs pull
uv sync --extra openai
```

Restore the two verified database snapshots. The commands refuse to overwrite
existing files:

```bash
uv run package_database.py unpack \
  --archive snapshots/ai-ml-papers.sqlite3.xz \
  --output data/ai-ml-papers.sqlite3
uv run package_database.py unpack \
  --archive snapshots/ai-ml-openai-embeddings.sqlite3.xz \
  --output data/ai-ml-openai-embeddings.sqlite3
```

Create `.env` from `.env.example` and set `OPENAI_API_KEY`. Credentials and
personal query files are ignored by Git. Project commands load `.env`
automatically; exported environment variables take precedence.

Search with just a title or a short research question:

```bash
uv run --extra openai embed_papers.py search \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3 \
  --query "Efficient reasoning in large language models" \
  --top-k 50
```

For a longer query, save your title and optional abstract in `target.txt` and
replace `--query "..."` with `--query-file target.txt`. Add `--json` to save
ranked results with abstracts and citation metadata.

Each search embeds only your query through OpenAI, then ranks the stored paper
vectors locally by cosine similarity. It automatically uses the model and
dimensions recorded in the index. No GPU or local model download is needed for
OpenAI search. Review the returned papers before citing them.

Check that a paper/index pair is complete and current without an API call:

```bash
uv run embed_papers.py status \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3
```

`snapshots/manifest.json` records archive hashes, corpus coverage, embedding
settings, and snapshot validation. Keep the two databases together: paper IDs
and text hashes link the vectors to the correct paper text.

## Collect or update papers

The default window is the current year and the preceding nine years. COLM
starts in 2024. You can select only the five supported conferences:

```bash
uv run build_corpus.py \
  --db data/ai-ml-papers.sqlite3 \
  --output data/ai-ml-corpus \
  --venues iclr icml neurips aaai colm
```

| Conference | Official roster and abstract sources |
| --- | --- |
| ICLR | ICLR proceedings and OpenReview |
| ICML | Proceedings of Machine Learning Research (PMLR) |
| NeurIPS | NeurIPS proceedings, including its published main research tracks |
| AAAI | AAAI proceedings, with main and special research tracks selected by section |
| COLM | Accepted-paper lists and OpenReview |

DBLP supplies an independent roster for the first four conferences, but no
abstracts. OpenAlex and Semantic Scholar can provide additional abstract
metadata. Workshops and confirmed ancillary programs are excluded from the
active corpus. Known withdrawals and retractions retain their source records
but are excluded from active search. Uncertain publisher removals carry review
notes instead of being treated as confirmed retractions.

Add `--start-year 2024 --end-year 2025` for a smaller range, or `--sources pmlr`
with `--venues icml` for a particular source. Successful source scans are
skipped on reruns, and downloaded pages are cached. `--max-requests 100` bounds
requests; `--refresh` revisits source lists. Unavailable or incomplete source
coverage is reported explicitly and does not mean a conference had zero papers.

OpenReview may require `OPENREVIEW_USERNAME` and `OPENREVIEW_PASSWORD`, or
`OPENREVIEW_TOKEN`, in `.env`. The collector reports access failures rather
than bypassing verification. Optional `OPENALEX_API_KEY` and
`SEMANTIC_SCHOLAR_API_KEY` configure the enrichment services.

Repair missing abstracts and apply reviewed duplicate/scope decisions:

```bash
uv run recover_abstracts.py \
  --db data/ai-ml-papers.sqlite3 --output data/ai-ml-corpus
```

This backs up the database, preserves source evidence and completed source
scans, and writes a recovery audit. `--dry-run` previews without applying paper
changes. Reviewed identity links, publication-status decisions, and verified
source text live in `reference_finder/roster_aliases.json`, `scope_reviews.json`,
and `abstract_snapshots.json`.

Regenerate the HTML inventory, paper CSV, coverage reports, and exclusion
ledger without collecting again:

```bash
uv run build_corpus.py --db data/ai-ml-papers.sqlite3 \
  --output data/ai-ml-corpus --report-only
```

## Build or refresh embeddings

The bundled OpenAI index is already complete. Run a build after adding papers
or changing their text; completed, unchanged papers are reused.

Estimate pending token usage locally before making embedding requests:

```bash
uv run --extra openai embed_papers.py estimate \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3
```

Build or resume the index:

```bash
uv run --extra openai embed_papers.py build \
  --db data/ai-ml-papers.sqlite3 \
  --index data/ai-ml-openai-embeddings.sqlite3 \
  --model text-embedding-3-large --batch-size 128
```

Use `--limit 100` for a pilot. OpenAI builds and searches make API calls;
`estimate` and `status` do not. Different models or dimensions require separate
indexes. A local Qwen backend is also available. See
[embedding details](docs/embeddings.md) for local/GPU setup, dimensions,
chunking, storage, and resume behavior.

## Data and development

Large compressed SQLite snapshots are tracked with **Git LFS**. Working
`data/` files, caches, backups, credentials, and personal queries stay local.
The snapshots contain the AI/ML corpus and matching vectors; reports can be
regenerated from the restored paper database.

To package updated databases, choose new archive paths, verify them, and update
`snapshots/manifest.json` before committing the archives:

```bash
uv run package_database.py pack --db data/ai-ml-papers.sqlite3 \
  --output snapshots/ai-ml-papers-next.sqlite3.xz
uv run package_database.py pack --db data/ai-ml-openai-embeddings.sqlite3 \
  --output snapshots/ai-ml-openai-embeddings-next.sqlite3.xz --page-size 1024
```

The project uses Python 3.12 with uv. Collection, snapshot handling, and API
search can also be installed through their `requirements-*.txt` files in a
separate Python environment.

Run the offline test suite:

```bash
uv run --extra openai python -m unittest discover -s tests
```

Tests mock external services and do not require paid API calls. Optional local
model runtime tests run when their dependencies are installed.
