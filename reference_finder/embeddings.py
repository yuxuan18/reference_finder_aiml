"""Portable corpus snapshots and resumable, model-pinned embedding indexes.

This module deliberately does not import a model framework. Its persistence and
coverage logic can be exercised without downloading weights or owning a GPU.
"""
from __future__ import annotations

from array import array
from contextlib import closing
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import tempfile
import time


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_PAPER_DB = "data/papers.sqlite3"
DEFAULT_INDEX = "data/embeddings.sqlite3"
STORAGE_LAYOUT = "linked-papers-v1"
DEFAULT_VECTOR_FORMAT = "float16-le"
FORMAT_VERSION = 3
VECTOR_FORMATS = {"float16-le": (2, "e", 2), "float32-le": (1, "f", 4)}
LOG = logging.getLogger(__name__)
PAPER_FIELDS = ("id", "title", "abstract", "publication_year", "doi", "url", "authors_json")
PAPER_COLUMNS = """id INTEGER PRIMARY KEY, title TEXT NOT NULL, abstract TEXT,
    publication_year INTEGER, doi TEXT, url TEXT, authors_json TEXT NOT NULL,
    venues_json TEXT NOT NULL"""


def read_only(path):
    path = Path(path).expanduser().resolve()
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def same_database(left, right):
    left, right = Path(left).expanduser().resolve(), Path(right).expanduser().resolve()
    return left == right or (left.exists() and right.exists() and os.path.samefile(left, right))


def read_corpus(path):
    """Read active papers from one snapshot, including API-only entries."""
    with closing(read_only(path)) as conn:
        conn.execute("BEGIN")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
        if not {"id", "title", "abstract"} <= columns:
            raise ValueError("Corpus must contain a papers table with id, title and abstract")
        selected = [name if name in columns else "NULL AS " + name for name in PAPER_FIELDS]
        selected.append("venues_json" if "venues_json" in columns else "'[]' AS venues_json")
        active_views = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name='active_corpus_papers' AND type='view'").fetchone())
        source = "active_corpus_papers" if active_views else "papers"
        rows = [dict(r) for r in conn.execute("SELECT " + ", ".join(selected) + " FROM " + source + " ORDER BY id")]
        venues = {}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_venues'").fetchone():
            memberships = "active_paper_venues" if active_views else "paper_venues"
            for r in conn.execute("SELECT DISTINCT paper_id, venue, year FROM " + memberships + " ORDER BY paper_id,venue,year"):
                venues.setdefault(r[0], []).append({"venue": r[1], "year": r[2]})
        for row in rows:
            row["title"] = (row["title"] or "").strip()
            row["abstract"] = (row["abstract"] or "").strip() or None
            row["authors_json"] = row["authors_json"] or "[]"
            if row["id"] in venues:
                row["venues_json"] = json.dumps(venues[row["id"]], ensure_ascii=False)
            row["venues_json"] = row["venues_json"] or "[]"
            if not paper_text(row):
                raise ValueError("Paper %s has neither a title nor an abstract" % row["id"])
        return rows


def paper_text(paper):
    return ((paper["title"] or "").strip() + "\n\n" + (paper["abstract"] or "").strip()).strip()


def text_hash(paper):
    return hashlib.sha256(paper_text(paper).encode("utf-8")).hexdigest()


def export_corpus(source, output):
    """Export only paper metadata/text; never copy a live SQLite main file."""
    rows = read_corpus(source)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Export already exists; choose a new output path: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".corpus-", suffix=".sqlite3", dir=output.parent)
    os.close(handle)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            conn.execute("CREATE TABLE papers (" + PAPER_COLUMNS + ")")
            names = PAPER_FIELDS + ("venues_json",)
            conn.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?)",
                             [tuple(row[k] for k in names) for row in rows])
            conn.commit()
        # Linking in the same directory publishes atomically without overwriting.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {"output": str(output), "papers": len(rows),
            "abstracts": sum(bool(row["abstract"]) for row in rows)}


def load_config(path, required=True):
    with closing(read_only(path)) as conn:
        try:
            row = conn.execute("SELECT value FROM embedding_meta WHERE key='config'").fetchone()
        except sqlite3.OperationalError as exc:
            if not required and not conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_meta'").fetchone():
                return None
            raise ValueError("Not an embedding index: %s" % path) from exc
    if row is None:
        if not required:
            return None
        raise ValueError("Database has no embeddings; run build first")
    config = json.loads(row[0])
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Unsupported embedding index format")
    vector_format = config.get("vector_format")
    if (not isinstance(vector_format, str) or vector_format not in VECTOR_FORMATS or
            type(config.get("format_version")) is not int or
            config.get("format_version") not in (VECTOR_FORMATS[vector_format][0], FORMAT_VERSION)):
        raise ValueError("Unsupported embedding index format/version pair")
    if config["format_version"] == FORMAT_VERSION and config.get("storage_layout") != STORAGE_LAYOUT:
        raise ValueError("Unsupported embedding index layout; format 3 requires linked-papers-v1")
    dimension = config.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("Invalid embedding index dimension")


def encoder_config(encoder, max_tokens, overlap, vector_format=DEFAULT_VECTOR_FORMAT,
                   format_version=FORMAT_VERSION):
    if vector_format not in VECTOR_FORMATS:
        raise ValueError("Unsupported vector format: %s" % vector_format)
    if format_version not in (VECTOR_FORMATS[vector_format][0], FORMAT_VERSION):
        raise ValueError("Unsupported embedding index format/version pair")
    config = dict(format_version=format_version, model_id=encoder.model_id,
                revision=encoder.revision, dimension=encoder.dimension,
                max_tokens=max_tokens, overlap=overlap,
                document_prompt=encoder.document_prompt, query_prompt=encoder.query_prompt,
                text_format="title-newline-newline-abstract-v1", normalization="l2",
                vector_format=vector_format, aggregation="max-chunk")
    if format_version == FORMAT_VERSION:
        config["storage_layout"] = STORAGE_LAYOUT
    if getattr(encoder, "provider", None) == "openai":
        config.update(provider="openai", api="openai-v1-embeddings", tokenizer=encoder.tokenizer_name)
    return config


def open_index(path, config):
    validate_config(config)
    path = Path(path).expanduser()
    if path.exists():
        previous = load_config(path)
        if previous is not None and previous != config:
            changed = sorted(k for k in set(previous) | set(config) if previous.get(k) != config.get(k))
            raise ValueError("Index settings differ (%s); use a new --index path" % ", ".join(changed))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # DELETE journaling leaves one portable file when each transaction finishes.
    conn.execute("PRAGMA journal_mode=DELETE")
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS embedding_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        conn.execute("INSERT OR IGNORE INTO embedding_meta VALUES ('config',?)", (json.dumps(config, sort_keys=True),))
        if config["format_version"] == FORMAT_VERSION:
            conn.execute("""CREATE TABLE IF NOT EXISTS paper_state (
                paper_id INTEGER PRIMARY KEY, source_hash TEXT NOT NULL,
                embedded INTEGER NOT NULL DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
                paper_id INTEGER NOT NULL REFERENCES paper_state(paper_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL, token_count INTEGER NOT NULL,
                vector BLOB NOT NULL, PRIMARY KEY(paper_id,chunk_index))""")
            return conn
        conn.execute("CREATE TABLE IF NOT EXISTS papers (" + PAPER_COLUMNS + ", source_hash TEXT NOT NULL, embedded INTEGER NOT NULL DEFAULT 0)")
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
            paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL,
            vector BLOB NOT NULL, PRIMARY KEY(paper_id,chunk_index))""")
    return conn


def sync_papers(conn, rows):
    """Invalidate changed text, refresh metadata, and remove absent source IDs."""
    old = {r["id"]: (r["source_hash"], r["embedded"]) for r in conn.execute("SELECT id,source_hash,embedded FROM papers")}
    ids = {p["id"] for p in rows}
    with conn:
        conn.executemany("DELETE FROM papers WHERE id=?", [(i,) for i in old.keys() - ids])
        for p in rows:
            digest = text_hash(p)
            same = p["id"] in old and old[p["id"]][0] == digest
            if not same:
                conn.execute("DELETE FROM chunks WHERE paper_id=?", (p["id"],))
            values = tuple(p[k] for k in PAPER_FIELDS + ("venues_json",))
            conn.execute("""INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,abstract=excluded.abstract,
                publication_year=excluded.publication_year,doi=excluded.doi,url=excluded.url,
                authors_json=excluded.authors_json,venues_json=excluded.venues_json,
                source_hash=excluded.source_hash,embedded=excluded.embedded""",
                         values + (digest, old[p["id"]][1] if same else 0))


def sync_paper_state(conn, rows):
    """Store only IDs and text fingerprints in the linked vector database."""
    old = {r["paper_id"]: (r["source_hash"], r["embedded"])
           for r in conn.execute("SELECT paper_id,source_hash,embedded FROM paper_state")}
    complete_ids = {r[0] for r in conn.execute("""SELECT paper_id FROM chunks GROUP BY paper_id
        HAVING MIN(chunk_index)=0 AND MAX(chunk_index)=COUNT(*)-1""")}
    ids = {p["id"] for p in rows}
    with conn:
        conn.executemany("DELETE FROM paper_state WHERE paper_id=?", [(i,) for i in old.keys() - ids])
        for paper in rows:
            digest = text_hash(paper)
            same = paper["id"] in old and old[paper["id"]][0] == digest
            if not same:
                conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper["id"],))
            conn.execute("""INSERT INTO paper_state VALUES (?,?,?) ON CONFLICT(paper_id)
                DO UPDATE SET source_hash=excluded.source_hash,embedded=excluded.embedded""",
                         (paper["id"], digest, old[paper["id"]][1] if same and paper["id"] in complete_ids else 0))


def linked_papers(conn, paper_db):
    """Match vectors to a consistent external paper snapshot using ID and text."""
    if paper_db is None:
        raise ValueError("This linked index needs its paper database; pass --db papers.sqlite3")
    try:
        papers = {paper["id"]: paper for paper in read_corpus(paper_db)}
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise ValueError("Cannot read the linked paper database %s; pass the matching --db file: %s" %
                         (paper_db, exc)) from exc
    states = {r["paper_id"]: dict(r) for r in conn.execute("SELECT * FROM paper_state")}
    valid = {paper_id for paper_id, state in states.items()
             if state["embedded"] and paper_id in papers and
             state["source_hash"] == text_hash(papers[paper_id])}
    return papers, states, valid


def pack_vector(vector, dimension, vector_format=DEFAULT_VECTOR_FORMAT):
    """Normalize before storage; half precision changes storage, not inference."""
    if vector_format not in VECTOR_FORMATS:
        raise ValueError("Unsupported vector format: %s" % vector_format)
    values = [float(v) for v in vector]
    if not values or len(values) != dimension or not all(math.isfinite(v) for v in values):
        raise ValueError("Model returned an invalid embedding dimension or non-finite values")
    scale = max(abs(v) for v in values)
    if scale == 0:
        raise ValueError("Model returned a zero embedding")
    # Scaling first keeps normalization safe even for extreme finite inputs.
    scaled = [v / scale for v in values]
    length = math.hypot(*scaled)
    code = VECTOR_FORMATS[vector_format][1]
    return struct.pack("<%d%s" % (dimension, code), *(v / length for v in scaled))


def unpack_vector(blob, vector_format=DEFAULT_VECTOR_FORMAT):
    """Decode portable little-endian blobs into float32 values without NumPy."""
    if vector_format not in VECTOR_FORMATS:
        raise ValueError("Unsupported vector format: %s" % vector_format)
    _, code, item_size = VECTOR_FORMATS[vector_format]
    if not blob or len(blob) % item_size:
        raise ValueError("Corrupt index: malformed embedding blob")
    values = array("f", struct.unpack("<%d%s" % (len(blob) // item_size, code), blob))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Corrupt index: non-finite embedding values")
    if not any(values):
        raise ValueError("Corrupt index: zero embedding")
    return values


def index_status(path, paper_db=None):
    config = load_config(path)
    with closing(read_only(path)) as conn:
        conn.execute("BEGIN")
        if config["format_version"] == FORMAT_VERSION:
            papers, states, valid = linked_papers(conn, paper_db)
            chunk_counts = dict(conn.execute("SELECT paper_id,COUNT(*) FROM chunks GROUP BY paper_id"))
            valid.intersection_update(chunk_counts)
            result = dict(papers=len(papers), abstracts=sum(bool(p["abstract"]) for p in papers.values()),
                          embedded=len(valid), embedded_title_only=sum(not papers[i]["abstract"] for i in valid),
                          chunks=sum(chunk_counts[i] for i in valid), stored_chunks=sum(chunk_counts.values()),
                          stale=sum(i in papers and row["source_hash"] != text_hash(papers[i]) for i, row in states.items()),
                          missing_papers=len(states.keys() - papers.keys()))
        else:
            row = conn.execute("""SELECT COUNT(*) AS papers, COALESCE(SUM(abstract IS NOT NULL),0) AS abstracts,
                COALESCE(SUM(embedded),0) AS embedded,
                COALESCE(SUM(embedded=1 AND abstract IS NULL),0) AS embedded_title_only FROM papers""").fetchone()
            result = dict(row)
            result["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    result["pending"] = result["papers"] - result["embedded"]
    result["title_only"] = result["papers"] - result["abstracts"]
    result["config"] = config
    return result


def build_index(source, destination, encoder, config, batch_size=32, limit=None):
    linked = config["format_version"] == FORMAT_VERSION
    if same_database(source, destination):
        raise ValueError("Paper database and embedding index must be different files")
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and limit must be positive")
    rows = read_corpus(source)
    with closing(open_index(destination, config)) as conn:
        if linked:
            sync_paper_state(conn, rows)
            pending_ids = {r[0] for r in conn.execute("SELECT paper_id FROM paper_state WHERE embedded=0")}
            pending = [p for p in rows if p["id"] in pending_ids]
        else:
            sync_papers(conn, rows)
            pending = [dict(r) for r in conn.execute("SELECT * FROM papers WHERE embedded=0 ORDER BY id")]
        selected = pending if limit is None else pending[:limit]
        LOG.info("%d papers in corpus, %d pending; embedding %d this run", len(rows), len(pending), len(selected))
        started = time.monotonic()
        done = 0
        for offset in range(0, len(selected), batch_size):
            group = selected[offset:offset + batch_size]
            prepared = [(p, encoder.document_chunks(paper_text(p), overlap=config["overlap"])) for p in group]
            flat = [chunk for _, chunks in prepared for chunk in chunks]
            if any(not chunks for _, chunks in prepared):
                raise ValueError("Tokenizer produced no chunks for a nonempty paper")
            vectors = encoder.encode_documents([c["text"] for c in flat], batch_size=batch_size)
            if len(vectors) != len(flat):
                raise ValueError("Model returned the wrong number of embeddings")
            blobs = [pack_vector(v, config["dimension"], config["vector_format"]) for v in vectors]
            current = None
            if linked:
                with closing(read_only(source)) as paper_conn:
                    placeholders = ",".join("?" for _ in group)
                    current = {r["id"]: text_hash(r) for r in paper_conn.execute(
                        "SELECT id,title,abstract FROM papers WHERE id IN (" + placeholders + ")",
                        [p["id"] for p in group])}
            position = 0
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                for paper, chunks in prepared:
                    digest = text_hash(paper)
                    if linked:
                        state = conn.execute("SELECT source_hash FROM paper_state WHERE paper_id=?", (paper["id"],)).fetchone()
                        if current.get(paper["id"]) != digest or state is None or state[0] != digest:
                            position += len(chunks)
                            LOG.warning("Paper %s changed during inference; leaving it pending", paper["id"])
                            continue
                    conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper["id"],))
                    for i, chunk in enumerate(chunks):
                        values = (paper["id"], i, chunk["token_count"], blobs[position])
                        if linked:
                            conn.execute("INSERT INTO chunks VALUES (?,?,?,?)", values)
                        else:
                            conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                                         values[:2] + (chunk["text"],) + values[2:])
                        position += 1
                    if linked:
                        conn.execute("UPDATE paper_state SET embedded=1 WHERE paper_id=?", (paper["id"],))
                    else:
                        conn.execute("UPDATE papers SET embedded=1 WHERE id=?", (paper["id"],))
            done += len(group)
            elapsed = max(time.monotonic() - started, 0.001)
            LOG.info("Processed %d/%d papers (%.1f papers/s; %.1f min remaining this run)",
                     done, len(selected), done / elapsed, (len(selected) - done) * elapsed / done / 60)
    return index_status(destination, paper_db=source if linked else None)


def benchmark(source, encoder, overlap=64, limit=500, batch_size=32):
    rows = read_corpus(source)
    if not rows or limit < 1 or batch_size < 1:
        raise ValueError("Benchmark needs a nonempty corpus and positive limit/batch size")
    sample = random.Random(0).sample(rows, min(limit, len(rows)))
    warmup = encoder.document_chunks(paper_text(sample[0]), overlap=overlap)
    encoder.encode_documents([warmup[0]["text"]], batch_size=1)
    encoder.synchronize()
    started = time.monotonic()
    chunks = [c for p in sample for c in encoder.document_chunks(paper_text(p), overlap=overlap)]
    encoder.encode_documents([c["text"] for c in chunks], batch_size=batch_size)
    encoder.synchronize()
    seconds = max(time.monotonic() - started, 1e-9)
    return dict(papers=len(sample), chunks=len(chunks), title_only=sum(not p["abstract"] for p in sample),
                tokens=sum(c["token_count"] for c in chunks), seconds=seconds,
                papers_per_second=len(sample) / seconds,
                projected_corpus_minutes=seconds * len(rows) / len(sample) / 60,
                corpus_papers=len(rows), device=encoder.device, dtype=encoder.dtype,
                model_id=encoder.model_id, revision=encoder.revision,
                note="Projection from a deterministic random sample after warmup; excludes model loading and index writes.")


def search_index(path, encoder, query, top_k=25, abstracts_only=False, paper_db=None):
    if not query.strip() or top_k < 1:
        raise ValueError("Query must be nonempty and top-k must be positive")
    config = load_config(path)
    actual = encoder_config(encoder, config["max_tokens"], config["overlap"], config["vector_format"],
                            format_version=config["format_version"])
    if actual != config:
        raise ValueError("Query encoder does not match the index configuration")
    with closing(read_only(path)) as conn:
        conn.execute("BEGIN")
        linked = config["format_version"] == FORMAT_VERSION
        if linked:
            papers, states, valid = linked_papers(conn, paper_db)
            candidates = [r for r in conn.execute("SELECT paper_id,chunk_index,vector FROM chunks ORDER BY paper_id,chunk_index")
                          if r["paper_id"] in valid and (not abstracts_only or papers[r["paper_id"]]["abstract"])]
            if not valid and any(state["embedded"] for state in states.values()):
                raise ValueError("No saved embeddings match this paper database. Pass the matching --db file or run build to refresh changed papers")
        else:
            candidates = conn.execute("""SELECT c.paper_id,c.chunk_index,c.text,c.vector FROM chunks c
                JOIN papers p ON p.id=c.paper_id WHERE p.embedded=1""" +
                (" AND p.abstract IS NOT NULL" if abstracts_only else "")).fetchall()
        vector = encoder.encode_queries([query], batch_size=1)[0]
        # Queries retain float32 precision even when the stored corpus uses float16.
        packed = pack_vector(vector, config["dimension"], "float32-le")
        try:
            import numpy as np
        except ImportError:
            np = None
        q = np.frombuffer(packed, dtype="<f4") if np is not None else unpack_vector(packed, "float32-le")
        item_size = VECTOR_FORMATS[config["vector_format"]][2]
        for row in candidates:
            if len(row["vector"]) != config["dimension"] * item_size:
                raise ValueError("Corrupt index: vector dimension mismatch")
        if np is not None and candidates:
            matrix = np.vstack([np.frombuffer(r["vector"], dtype="<f%d" % item_size)
                                for r in candidates]).astype(np.float32, copy=False)
            if not np.isfinite(matrix).all():
                raise ValueError("Corrupt index: non-finite embedding values")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("Corrupt index: zero embedding or invalid norm")
            # Quantization perturbs unit length; restore it for actual cosine.
            matrix /= norms[:, None]
            q = q / np.linalg.norm(q)
            scores = matrix @ q
        else:
            scores = []
            query_norm = math.hypot(*q)
            for row in candidates:
                values = unpack_vector(row["vector"], config["vector_format"])
                scores.append(sum(a * b for a, b in zip(values, q)) /
                              (math.hypot(*values) * query_norm))
        best = {}
        for row, score in zip(candidates, scores):
            score = float(score)
            if row["paper_id"] not in best or score > best[row["paper_id"]]["score"]:
                best[row["paper_id"]] = dict(paper_id=row["paper_id"], score=score,
                    chunk_index=row["chunk_index"])
                if not linked:
                    best[row["paper_id"]]["matching_text"] = row["text"]
        ranked = sorted(best.values(), key=lambda r: (-r["score"], r["paper_id"]))[:top_k]
        for result in ranked:
            p = papers[result["paper_id"]] if linked else dict(conn.execute(
                "SELECT * FROM papers WHERE id=?", (result["paper_id"],)).fetchone())
            result.update({k: p[k] for k in ("title", "abstract", "publication_year", "doi", "url")})
            result["has_abstract"] = bool(p["abstract"])
            result["authors"] = json.loads(p["authors_json"])
            result["venues"] = json.loads(p["venues_json"])
    return ranked
