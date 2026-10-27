#!/usr/bin/env python3
"""Embed and search papers using separate paper and vector databases."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3
import sys

from reference_finder.embeddings import (
    DEFAULT_INDEX, DEFAULT_MODEL, DEFAULT_PAPER_DB, DEFAULT_VECTOR_FORMAT, benchmark, build_index, encoder_config, export_corpus,
    index_status, linked_papers, load_config, read_only, same_database, search_index,
)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def model_options(parser, query=False):
    parser.add_argument("--provider", choices=("local", "openai"),
                        help="default: infer from model/index; otherwise local")
    parser.add_argument("--dimensions", type=positive, help="OpenAI vector dimensions; default is the model's full size")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto",
                        help="auto chooses CUDA, then Apple MPS, otherwise CPU")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    if not query:
        parser.add_argument("--model", help="default: %s (local), text-embedding-3-large (OpenAI)" % DEFAULT_MODEL)
        parser.add_argument("--revision", help="Hugging Face revision; resolved and saved as an immutable commit")
        parser.add_argument("--max-tokens", type=positive, help="chunk tokens; default 2048 local / 8191 OpenAI")
        parser.add_argument("--overlap", type=int, help="overlapping content tokens per chunk, default 64")
        parser.add_argument("--batch-size", type=positive, default=16)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="create a consistent, compact corpus snapshot (no model required)")
    export.add_argument("--db", default=DEFAULT_PAPER_DB)
    export.add_argument("--output", required=True)
    build = sub.add_parser("build", help="embed every paper, resuming committed progress")
    build.add_argument("--db", default=DEFAULT_PAPER_DB)
    build.add_argument("--index", default=DEFAULT_INDEX, help="vector database; contains no paper text")
    build.add_argument("--limit", type=positive, help="maximum pending papers this run; omit to finish")
    build.add_argument("--storage-dtype", choices=("float16", "float32"),
                       help="saved vector precision; default float16 for new indexes, or the saved format when resuming")
    model_options(build)
    bench = sub.add_parser("benchmark", help="warm up and time a random sample without writing an index")
    bench.add_argument("--db", default=DEFAULT_PAPER_DB)
    bench.add_argument("--limit", type=positive, default=500)
    model_options(bench)
    estimate = sub.add_parser("estimate", help="count OpenAI tokens and estimate input cost without API calls")
    estimate.add_argument("--db", default=DEFAULT_PAPER_DB)
    estimate.add_argument("--index", help="optional OpenAI index; estimate only missing/changed papers")
    estimate.add_argument("--limit", type=positive)
    estimate.add_argument("--price-per-million", type=float, help="override the standard input price in USD")
    model_options(estimate)
    estimate.set_defaults(provider="openai")
    status = sub.add_parser("status", help="report saved coverage and settings (no model required)")
    status.add_argument("--db", default=DEFAULT_PAPER_DB)
    status.add_argument("--index", default=DEFAULT_INDEX)
    convert = sub.add_parser("convert", help="copy an existing index to float16 storage (no model required)")
    convert.add_argument("--db", default=DEFAULT_PAPER_DB)
    convert.add_argument("--index", default=DEFAULT_INDEX)
    convert.add_argument("--output", required=True, help="new output file; the input is never overwritten")
    search = sub.add_parser("search", help="rank indexed papers using an abstract or natural-language query")
    search.add_argument("--db", default=DEFAULT_PAPER_DB)
    search.add_argument("--index", default=DEFAULT_INDEX)
    query = search.add_mutually_exclusive_group(required=True)
    query.add_argument("--query")
    query.add_argument("--query-file", type=Path, help="UTF-8 plain text, not PDF")
    search.add_argument("--top-k", type=positive, default=25)
    search.add_argument("--abstracts-only", action="store_true", help="exclude explicitly marked title-only records")
    search.add_argument("--json", action="store_true", help="write machine-readable results to stdout")
    model_options(search, query=True)
    return p


def make_encoder(args, existing=None, offline=False):
    storage_dtype = getattr(args, "storage_dtype", None)
    vector_format = storage_dtype + "-le" if storage_dtype else DEFAULT_VECTOR_FORMAT
    if existing:
        if storage_dtype and vector_format != existing["vector_format"]:
            raise ValueError("Storage precision differs from the saved index; use convert or a new --index path")
        vector_format = existing["vector_format"]
    if sys.version_info < (3, 9):
        raise ValueError(
            "Embedding inference requires Python 3.9+; "
            "running %s at %s. Create a new environment with Python 3.9+ and install "
            "requirements-client.txt (local), requirements-server.txt (GPU), "
            "or requirements-api-embeddings.txt (OpenAI). "
            % (sys.version.split()[0], sys.executable)
        )
    requested_model = getattr(args, "model", None)
    inferred = "openai" if requested_model and requested_model.startswith("text-embedding-") else None
    saved_provider = existing.get("provider", "local") if existing else None
    provider = getattr(args, "provider", None) or inferred or saved_provider or "local"
    if existing and provider != saved_provider:
        raise ValueError("Embedding provider differs from the index; use a new --index path")
    if provider == "openai":
        return make_openai_encoder(args, existing, vector_format, offline)
    if getattr(args, "dimensions", None) is not None:
        raise ValueError("--dimensions is supported only for OpenAI embeddings")
    if inferred:
        raise ValueError("OpenAI embedding models require --provider openai")
    if offline:
        raise ValueError("Cost estimation currently supports only --provider openai")
    from reference_finder.embedding_backend import SentenceEncoder, resolve_revision
    # Preserve the local default and legacy mismatch protection for builds.
    # Search always follows the model recorded in its index.
    model_id = requested_model or (existing["model_id"] if existing and args.command == "search" else DEFAULT_MODEL)
    max_tokens = getattr(args, "max_tokens", None) or (existing["max_tokens"] if existing else 2048)
    overlap = getattr(args, "overlap", None)
    if overlap is None:
        overlap = existing["overlap"] if existing else 64
    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap must be nonnegative and smaller than max-tokens")
    if existing and (model_id != existing["model_id"] or max_tokens != existing["max_tokens"] or overlap != existing["overlap"]):
        raise ValueError("Model/chunk settings differ from the saved index; use the matching saved model and chunk settings")
    requested = getattr(args, "revision", None)
    revision = resolve_revision(model_id, requested) if requested or not existing else existing["revision"]
    if existing and revision != existing["revision"]:
        raise ValueError("Revision differs from the saved index; use a new --index path")
    logging.info("Loading %s @ %s", model_id, revision)
    encoder = SentenceEncoder(model_id, revision, max_tokens=max_tokens, device=args.device, dtype=args.dtype)
    version = existing["format_version"] if existing else 3
    config = encoder_config(encoder, max_tokens, overlap, vector_format=vector_format, format_version=version)
    if existing and config != existing:
        raise ValueError("Loaded encoder differs from saved index settings (prompts/dimensions); use matching dependencies")
    logging.info("Using device=%s dtype=%s dimension=%d", encoder.device, encoder.dtype, encoder.dimension)
    return encoder, config


def make_openai_encoder(args, existing, vector_format, offline=False):
    from reference_finder.openai_embedding_backend import API_REVISION, DEFAULT_OPENAI_MODEL, MODEL_DIMENSIONS, OpenAIEncoder
    if args.device != "auto" or args.dtype != "auto" or getattr(args, "revision", None):
        raise ValueError("OpenAI embeddings do not use --device, --dtype or Hugging Face --revision")
    model = getattr(args, "model", None) or (existing["model_id"] if existing else DEFAULT_OPENAI_MODEL)
    dimension = getattr(args, "dimensions", None) or (existing["dimension"] if existing else MODEL_DIMENSIONS.get(model))
    max_tokens = getattr(args, "max_tokens", None) or (existing["max_tokens"] if existing else 8191)
    overlap = getattr(args, "overlap", None)
    if overlap is None:
        overlap = existing["overlap"] if existing else 64
    if not 0 <= overlap < max_tokens:
        raise ValueError("overlap must be nonnegative and smaller than max-tokens")
    if existing and (model != existing["model_id"] or dimension != existing["dimension"]
                     or max_tokens != existing["max_tokens"] or overlap != existing["overlap"]
                     or existing["revision"] != API_REVISION):
        raise ValueError("OpenAI model/dimensions/chunk settings differ from the index; use a new --index path")
    from reference_finder.config import load_environment
    load_environment()
    encoder = OpenAIEncoder(model, dimensions=dimension, max_tokens=max_tokens, offline=offline)
    config = encoder_config(encoder, max_tokens, overlap, vector_format,
                            format_version=existing["format_version"] if existing else 3)
    if existing and config != existing:
        raise ValueError("OpenAI encoder differs from saved index settings; use the matching configuration")
    logging.info("Using OpenAI model=%s dimensions=%d", model, encoder.dimension)
    return encoder, config


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    encoder = None
    try:
        if args.command == "export":
            output = export_corpus(args.db, args.output)
        elif args.command == "status":
            output = index_status(args.index, paper_db=args.db)
        elif args.command == "convert":
            from reference_finder.embedding_conversion import convert_index
            output = convert_index(args.index, args.output, paper_db=args.db)
        elif args.command == "estimate":
            from reference_finder.embeddings import read_corpus
            from reference_finder.openai_embedding_backend import estimate_cost
            read_corpus(args.db)
            existing = load_config(args.index) if args.index and Path(args.index).expanduser().exists() else None
            encoder, config = make_encoder(args, existing, offline=True)
            completed = set()
            if existing:
                if existing["format_version"] != 3:
                    raise ValueError("Pending-cost estimates require a linked format-3 OpenAI index")
                with closing(read_only(args.index)) as conn:
                    _, _, completed = linked_papers(conn, args.db)
                    contiguous = {r[0] for r in conn.execute("""SELECT paper_id FROM chunks GROUP BY paper_id
                        HAVING MIN(chunk_index)=0 AND MAX(chunk_index)=COUNT(*)-1""")}
                    completed.intersection_update(contiguous)
            output = estimate_cost(args.db, encoder, config["overlap"], args.limit,
                                   completed, args.price_per_million)
        elif args.command == "benchmark":
            # Validate input before downloading/loading model weights.
            from reference_finder.embeddings import read_corpus
            read_corpus(args.db)
            encoder, config = make_encoder(args)
            output = benchmark(args.db, encoder, config["overlap"], args.limit, args.batch_size)
        elif args.command == "build":
            from reference_finder.embeddings import read_corpus
            if same_database(args.db, args.index):
                raise ValueError("Paper database and embedding index must be different files")
            read_corpus(args.db)
            existing = load_config(args.index) if Path(args.index).expanduser().exists() else None
            encoder, config = make_encoder(args, existing)
            output = build_index(args.db, args.index, encoder, config, args.batch_size, args.limit)
        else:
            query = args.query_file.expanduser().read_text(encoding="utf-8") if args.query_file else args.query
            if not query.strip():
                raise ValueError("Query is empty")
            status = index_status(args.index, paper_db=args.db)
            if not status["embedded"]:
                if status.get("stale") or status.get("missing_papers"):
                    raise ValueError("No embeddings match this paper database; check --db and --index, or run build to refresh changed papers")
                raise ValueError("Index has no completed papers; run build first")
            if status["pending"]:
                logging.warning("Partial index: %d/%d papers embedded", status["embedded"], status["papers"])
            encoder, _ = make_encoder(args, status["config"])
            matches = search_index(args.index, encoder, query, args.top_k, args.abstracts_only, paper_db=args.db)
            if not args.json:
                for i, result in enumerate(matches, 1):
                    marker = "abstract" if result["has_abstract"] else "TITLE ONLY"
                    print("%2d. %.4f  [%s; id=%s; %s] %s" % (i, result["score"],
                          result["publication_year"] or "?", result["paper_id"], marker, result["title"]))
                    url = "https://doi.org/" + result["doi"] if result["doi"] else result["url"]
                    if url:
                        print("    " + url)
                return 0
            output = {"db": str(args.db), "index": str(args.index), "coverage": status, "matches": matches}
        if getattr(encoder, "provider", None) == "openai" and args.command != "estimate":
            output["api_usage"] = dict(batches=encoder.batches, input_tokens=encoder.input_tokens)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted. Previously committed batches are saved; repeat build to resume.")
        return 130
    except (ValueError, OSError, sqlite3.Error, ImportError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
