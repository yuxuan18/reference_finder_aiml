"""Vector-only migration removes text and rejects stale or incomplete indexes."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

from reference_finder.embedding_migration import strip_embedding_text
from reference_finder.embeddings import index_status, pack_vector, search_index, text_hash


COLLECTION_SCHEMA = """
    PRAGMA user_version=2;
    CREATE TABLE papers (
        id INTEGER PRIMARY KEY, doi TEXT UNIQUE, title TEXT NOT NULL,
        normalized_title TEXT NOT NULL, publication_year INTEGER, publication_date TEXT,
        authors_json TEXT NOT NULL DEFAULT '[]', abstract TEXT, abstract_source TEXT,
        url TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE source_records (
        provider TEXT NOT NULL, external_id TEXT NOT NULL,
        paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        dblp_key TEXT, fetched_at TEXT NOT NULL, PRIMARY KEY(provider,external_id)
    );
    CREATE TABLE paper_venues (
        paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        venue TEXT NOT NULL, year INTEGER NOT NULL, provider TEXT NOT NULL,
        PRIMARY KEY(paper_id,venue,year,provider)
    );
    CREATE TABLE checkpoints (
        job TEXT PRIMARY KEY, cursor TEXT, done INTEGER NOT NULL DEFAULT 0,
        fetched INTEGER NOT NULL DEFAULT 0, accepted INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );
"""

INDEX_SCHEMA = """
    CREATE TABLE papers (
        id INTEGER PRIMARY KEY, title TEXT NOT NULL, abstract TEXT,
        source_hash TEXT NOT NULL, embedded INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE chunks (
        paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL,
        vector BLOB NOT NULL, PRIMARY KEY(paper_id,chunk_index)
    );
    CREATE TABLE embedding_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


class EmbeddingMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.collection = self.root / "collection ü #%.sqlite3"
        self.index = self.root / "legacy.sqlite3"
        self.output = self.root / "vectors.sqlite3"
        self.config = dict(format_version=2, vector_format="float16-le", dimension=3,
                           model_id="test/model", revision="a" * 40, document_prompt="", query_prompt="query: ",
                           max_tokens=2048, overlap=64, normalization="l2", aggregation="max-chunk",
                           text_format="title-newline-newline-abstract-v1")
        self.papers = [
            (1, "Paper one — 证明", "Abstract one"),
            (2, "Second paper without an abstract", None),
            (3, "Pending indexed paper", "Pending abstract"),
            (4, "New collection paper", "Not indexed yet"),
        ]
        with closing(sqlite3.connect(self.collection)) as connection:
            connection.executescript(COLLECTION_SCHEMA)
            connection.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                (paper_id, None, title, title.lower(), 2026, None, '["An Author"]', abstract,
                 "semantic_scholar" if abstract else None, "https://example.org/paper", "2026-01-01", "2026-10-11")
                for paper_id, title, abstract in self.papers
            ])
            connection.execute("INSERT INTO source_records VALUES ('dblp','conf/iclr/Test26',1,NULL,'2026-10-11')")
            connection.execute("INSERT INTO paper_venues VALUES (1,'iclr',2026,'dblp')")
            connection.execute("INSERT INTO checkpoints VALUES ('resume','cursor',0,50,4,'2026-10-11')")
            connection.commit()
        with closing(sqlite3.connect(self.index)) as connection:
            connection.executescript(INDEX_SCHEMA)
            connection.executemany("INSERT INTO papers VALUES (?,?,?,?,?)", [
                (paper_id, title, abstract, text_hash(dict(title=title, abstract=abstract)), int(paper_id != 3))
                for paper_id, title, abstract in self.papers[:3]
            ])
            text = self.papers[0][1] + "\n\n" + self.papers[0][2]
            second = self.papers[1][1]
            connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", [
                (1, 0, text, 20, pack_vector([1, 2, 3], 3)),
                (2, 0, second[:15], 10, pack_vector([3, 1, 2], 3)),
                (2, 1, second[10:], 12, pack_vector([2, 3, 1], 3)),
            ])
            connection.executemany("INSERT INTO embedding_meta VALUES (?,?)", [
                ("config", json.dumps(self.config)), ("additional", "preserved metadata"),
            ])
            connection.commit()

    def rows(self, path, sql):
        with closing(sqlite3.connect(path)) as connection:
            return connection.execute(sql).fetchall()

    def mutate(self, path, sql, parameters=()):
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(sql, parameters)
            connection.commit()

    def assert_failure_preserves_inputs(self, message):
        originals = [path.read_bytes() for path in (self.collection, self.index)]
        with self.assertRaisesRegex(ValueError, message):
            strip_embedding_text(self.collection, self.index, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual([path.read_bytes() for path in (self.collection, self.index)], originals)
        self.assertFalse(list(self.root.glob(".sqlite-package-*")))

    def test_paper_text_is_absent_and_vector_bytes_are_unchanged(self):
        before = [path.read_bytes() for path in (self.collection, self.index)]
        result = strip_embedding_text(self.collection, self.index, self.output)
        self.assertEqual([path.read_bytes() for path in (self.collection, self.index)], before)
        tables = self.rows(self.output, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        self.assertEqual(tables, [("chunks",), ("embedding_meta",), ("paper_state",)])
        state = self.rows(self.output, "SELECT * FROM paper_state ORDER BY paper_id")
        self.assertEqual(state, [(paper_id, text_hash(dict(title=title, abstract=abstract)), int(paper_id < 3))
                                 for paper_id, title, abstract in self.papers])
        self.assertEqual(self.rows(self.index, "SELECT paper_id,chunk_index,token_count,vector FROM chunks ORDER BY 1,2"),
                         self.rows(self.output, "SELECT * FROM chunks ORDER BY 1,2"))
        columns = [row[1] for row in self.rows(self.output, "PRAGMA table_info(chunks)")]
        self.assertEqual(columns, ["paper_id", "chunk_index", "token_count", "vector"])
        self.assertEqual(self.rows(self.output, "SELECT key FROM embedding_meta"), [("config",)])
        content = self.output.read_bytes()
        for _, title, abstract in self.papers:
            self.assertNotIn(title.encode("utf-8"), content)
            if abstract:
                self.assertNotIn(abstract.encode("utf-8"), content)
        saved = json.loads(self.rows(self.output, "SELECT value FROM embedding_meta WHERE key='config'")[0][0])
        self.assertEqual(saved, dict(self.config, format_version=3, storage_layout="linked-papers-v1"))
        self.assertEqual(result["papers"], 4)
        self.assertEqual(result["abstracts"], 3)
        self.assertEqual(result["embedded"], 2)
        self.assertEqual(result["pending"], 2)
        self.assertEqual(result["chunks"], 3)
        self.assertEqual(result["vector_bytes"], 18)
        self.assertEqual(result["output_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(self.rows(self.output, "PRAGMA integrity_check"), [("ok",)])
        self.assertEqual(self.rows(self.output, "PRAGMA journal_mode"), [("delete",)])
        self.assertEqual(self.rows(self.output, "PRAGMA page_size"), [(1024,)])
        self.assertFalse(Path(str(self.output) + "-wal").exists())

    def test_float32_index_retains_exact_vector_bytes_and_format(self):
        self.config.update(format_version=1, vector_format="float32-le")
        self.mutate(self.index, "UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(self.config),))
        blob = pack_vector([1, 2, 3], 3, "float32-le")
        self.mutate(self.index, "UPDATE chunks SET vector=?", (blob,))
        result = strip_embedding_text(self.collection, self.index, self.output)
        self.assertEqual(result["vector_bytes"], 36)
        self.assertEqual(result["config"]["vector_format"], "float32-le")
        self.assertEqual(self.rows(self.output, "SELECT vector FROM chunks"), [(blob,)] * 3)

    def test_migrated_index_search_joins_current_papers_and_excludes_changed_text(self):
        class QueryEncoder:
            model_id = "test/model"
            revision = "a" * 40
            dimension = 3
            document_prompt = ""
            query_prompt = "query: "

            def encode_queries(self, texts, batch_size=1):
                return [[1, 0, 0] for _ in texts]

        strip_embedding_text(self.collection, self.index, self.output)
        status = index_status(self.output, paper_db=self.collection)
        self.assertEqual(status["embedded"], 2)
        self.assertEqual(status["pending"], 2)
        found = search_index(self.output, QueryEncoder(), "example query", paper_db=self.collection)
        self.assertEqual([row["paper_id"] for row in found], [2, 1])
        self.assertEqual(found[0]["title"], self.papers[1][1])
        self.assertNotIn("matching_text", found[0])
        self.mutate(self.collection, "UPDATE papers SET title='Changed after migration' WHERE id=2")
        found = search_index(self.output, QueryEncoder(), "example query", paper_db=self.collection)
        self.assertEqual([row["paper_id"] for row in found], [1])

    def test_both_committed_wal_snapshots_included_without_changing_sources(self):
        with closing(sqlite3.connect(self.collection)) as source_writer, closing(sqlite3.connect(self.index)) as index_writer:
            for writer in (source_writer, index_writer):
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute("PRAGMA wal_autocheckpoint=0")
            source_writer.execute("UPDATE papers SET title='New collection title in WAL' WHERE id=4")
            source_writer.commit()
            config = dict(self.config, query_prompt="query prompt committed in WAL: ")
            index_writer.execute("UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(config),))
            index_writer.commit()
            paths = [self.collection, Path(str(self.collection) + "-wal"), self.index, Path(str(self.index) + "-wal")]
            before = [path.read_bytes() for path in paths]
            strip_embedding_text(self.collection, self.index, self.output)
            self.assertEqual([path.read_bytes() for path in paths], before)
        self.assertEqual(self.rows(self.output, "SELECT source_hash FROM paper_state WHERE paper_id=4"),
                         [(text_hash(dict(title="New collection title in WAL", abstract="Not indexed yet")),)])
        saved = json.loads(self.rows(self.output, "SELECT value FROM embedding_meta WHERE key='config'")[0][0])
        self.assertEqual(saved["query_prompt"], "query prompt committed in WAL: ")

    def test_arbitrary_legacy_metadata_does_not_retain_a_hidden_text_copy(self):
        self.mutate(self.index, "INSERT INTO embedding_meta VALUES ('duplicate_abstract',?)", (self.papers[0][2],))
        strip_embedding_text(self.collection, self.index, self.output)
        self.assertNotIn(self.papers[0][2].encode("utf-8"), self.output.read_bytes())
        self.assertEqual(self.rows(self.output, "SELECT key FROM embedding_meta"), [("config",)])

    def test_refuses_existing_output_or_either_source_path(self):
        self.output.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            strip_embedding_text(self.collection, self.index, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep")
        for path in (self.collection, self.index):
            with self.subTest(path=path), self.assertRaises((ValueError, FileExistsError)):
                strip_embedding_text(self.collection, self.index, path)

    def test_changed_collection_text_is_rejected(self):
        self.mutate(self.collection, "UPDATE papers SET abstract='Changed source text' WHERE id=1")
        self.assert_failure_preserves_inputs("Text hash mismatch for paper 1")

    def test_stale_saved_source_hash_is_rejected(self):
        self.mutate(self.index, "UPDATE papers SET source_hash='stale' WHERE id=1")
        self.assert_failure_preserves_inputs("Text hash mismatch for paper 1")

    def test_missing_collection_paper_is_rejected(self):
        self.mutate(self.collection, "DELETE FROM papers WHERE id=2")
        self.assert_failure_preserves_inputs("paper 2 is absent")

    def test_completed_paper_without_vectors_is_rejected(self):
        self.mutate(self.index, "DELETE FROM chunks WHERE paper_id=1")
        self.assert_failure_preserves_inputs("Incomplete embedding state for paper 1")

    def test_pending_paper_with_vectors_is_rejected(self):
        self.mutate(self.index, "UPDATE papers SET embedded=0 WHERE id=1")
        self.assert_failure_preserves_inputs("Incomplete embedding state for paper 1")

    def test_missing_first_chunk_is_rejected(self):
        self.mutate(self.index, "DELETE FROM chunks WHERE paper_id=2 AND chunk_index=0")
        self.assert_failure_preserves_inputs("Noncontiguous chunks for paper 2")

    def test_missing_final_chunk_is_rejected_even_when_indexes_are_contiguous(self):
        self.mutate(self.index, "DELETE FROM chunks WHERE paper_id=2 AND chunk_index=1")
        self.assert_failure_preserves_inputs("source text tail is missing")

    def test_unrelated_chunk_text_is_rejected(self):
        self.mutate(self.index, "UPDATE chunks SET text='Unrelated words' WHERE paper_id=1")
        self.assert_failure_preserves_inputs("Chunk text does not cover paper 1")

    def test_orphan_chunk_is_rejected(self):
        self.mutate(self.index, "UPDATE chunks SET paper_id=999 WHERE paper_id=1")
        self.assert_failure_preserves_inputs("chunks referring to missing papers")

    def test_invalid_token_count_is_rejected(self):
        self.mutate(self.index, "UPDATE chunks SET token_count=2049 WHERE paper_id=1")
        self.assert_failure_preserves_inputs("Invalid token count for paper 1")

    def test_invalid_vectors_are_rejected_without_publishing(self):
        for blob in (b"bad", struct.pack("<eee", 0, 0, 0), struct.pack("<eee", float("nan"), 1, 2),
                     struct.pack("<ee", 1, 2)):
            with self.subTest(blob=blob):
                self.mutate(self.index, "UPDATE chunks SET vector=? WHERE paper_id=1", (blob,))
                self.assert_failure_preserves_inputs("Invalid vector for paper 1")

    def test_unsupported_model_or_text_configuration_is_rejected(self):
        for key, value in (("revision", "main"), ("normalization", "none"),
                           ("text_format", "abstract-only"), ("aggregation", "mean"), ("max_tokens", 0)):
            with self.subTest(key=key):
                config = dict(self.config, **{key: value})
                self.mutate(self.index, "UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(config),))
                self.assert_failure_preserves_inputs("[Cc]onfiguration")

    def test_linked_index_input_is_rejected_as_already_migrated(self):
        strip_embedding_text(self.collection, self.index, self.output)
        destination = self.root / "another.sqlite3"
        with self.assertRaisesRegex(ValueError, "legacy embedding index"):
            strip_embedding_text(self.collection, self.output, destination)
        self.assertFalse(destination.exists())

    def test_wrong_database_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "paper database"):
            strip_embedding_text(self.index, self.collection, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_runs_without_optional_model_or_api_dependencies(self):
        command = [sys.executable, "-S", str(Path(__file__).resolve().parents[1] / "strip_embedding_text.py"),
                   "--db", str(self.collection), "--index", str(self.index), "--output", str(self.output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["config"]["format_version"], 3)


if __name__ == "__main__":
    unittest.main()
