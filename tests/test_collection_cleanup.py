"""Safe collection compaction preserves canonical papers and collection progress."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reference_finder.collection_cleanup import prune_collection


LEGACY_SCHEMA = """
    PRAGMA user_version=1;
    CREATE TABLE papers (
        id INTEGER PRIMARY KEY, doi TEXT UNIQUE, title TEXT NOT NULL,
        normalized_title TEXT NOT NULL, publication_year INTEGER, publication_date TEXT,
        authors_json TEXT NOT NULL DEFAULT '[]', abstract TEXT, abstract_source TEXT,
        url TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE source_records (
        provider TEXT NOT NULL, external_id TEXT NOT NULL,
        paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        raw_json TEXT NOT NULL, fetched_at TEXT NOT NULL,
        PRIMARY KEY(provider, external_id)
    );
    CREATE INDEX source_records_paper ON source_records(paper_id);
    CREATE TABLE paper_venues (
        paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
        venue TEXT NOT NULL, year INTEGER NOT NULL, provider TEXT NOT NULL,
        PRIMARY KEY(paper_id, venue, year, provider)
    );
    CREATE TABLE checkpoints (
        job TEXT PRIMARY KEY, cursor TEXT, done INTEGER NOT NULL DEFAULT 0,
        fetched INTEGER NOT NULL DEFAULT 0, accepted INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE roster_entries (
        provider TEXT NOT NULL, external_id TEXT NOT NULL, venue TEXT NOT NULL,
        year INTEGER NOT NULL, paper_id INTEGER NOT NULL REFERENCES papers(id),
        title TEXT NOT NULL, doi TEXT, url TEXT, authors_json TEXT NOT NULL,
        section TEXT, raw_json TEXT NOT NULL,
        PRIMARY KEY(provider,external_id,venue,year)
    );
    CREATE INDEX roster_entries_paper ON roster_entries(paper_id);
"""


class CollectionCleanupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "collection ü #%.sqlite3"
        self.output = self.root / "pruned.sqlite3"
        self.source_payload = json.dumps({
            "externalIds": {"DBLP": "conf/iclr/Example26"},
            "abstract": "Redundant source data. " * 10000,
        })
        self.roster_payload = json.dumps({
            "title": "Original title", "abstract": "Redundant abstract" * 1000,
            "roster_review": {"action": "include", "note": "Confirmed paper",
                              "evidence_urls": ["https://example.org/proceedings"]},
            "roster_match": {"method": "dblp-key", "dblp_key": "conf/iclr/Example26",
                             "reason": "Exact key match"},
        })
        with closing(sqlite3.connect(self.source)) as connection:
            connection.executescript(LEGACY_SCHEMA)
            connection.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                (1, "10.1234/test", "Canonical paper — 证明", "canonical paper", 2026, None,
                 '["An Author"]', "Canonical abstract", "semantic_scholar", "https://example.org/paper",
                 "2026-01-01", "2026-10-11"),
                (2, None, "Title-only paper", "title only paper", 2025, None, "[]", None,
                 None, None, "2026-01-01", "2026-10-11"),
            ])
            connection.execute("INSERT INTO source_records VALUES (?,?,?,?,?)",
                               ("semantic_scholar", "s2-id", 1, self.source_payload, "2026-10-11"))
            connection.executemany("INSERT INTO paper_venues VALUES (?,?,?,?)", [
                (1, "iclr", 2026, "semantic_scholar"), (1, "iclr", 2026, "dblp"),
                (2, "iclr", 2025, "dblp"),
            ])
            connection.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?)",
                               ('{"provider":"openalex"}', "resume-here", 0, 100, 2, "2026-10-11"))
            connection.execute("INSERT INTO roster_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                               ("dblp", "conf/iclr/Example26", "iclr", 2026, 1, "Roster title", "10.1234/test",
                                "https://example.org/paper", '["An Author"]', "Research papers",
                                self.roster_payload))
            connection.commit()

    def rows(self, path, sql):
        with closing(sqlite3.connect(path)) as connection:
            return connection.execute(sql).fetchall()

    def test_raw_payloads_removed_but_papers_links_reviews_and_progress_preserved(self):
        original = self.source.read_bytes()
        expected = {table: self.rows(self.source, "SELECT * FROM " + table + " ORDER BY 1,2")
                    for table in ("papers", "paper_venues", "checkpoints")}
        result = prune_collection(self.source, self.output)
        self.assertEqual(self.source.read_bytes(), original)
        for table, rows in expected.items():
            self.assertEqual(self.rows(self.output, "SELECT * FROM " + table + " ORDER BY 1,2"), rows)
        for table in ("source_records", "roster_entries"):
            columns = self.rows(self.output, "PRAGMA table_info(" + table + ")")
            self.assertNotIn("raw_json", [row[1] for row in columns])
        self.assertEqual(self.rows(self.output, "SELECT * FROM source_records"), [
            ("semantic_scholar", "s2-id", 1, "conf/iclr/Example26", "2026-10-11")])
        review = json.loads(self.rows(self.output, "SELECT review_json FROM roster_entries")[0][0])
        raw_review = json.loads(self.roster_payload)
        self.assertEqual(review, {key: raw_review[key] for key in ("roster_review", "roster_match")})
        self.assertEqual(result["papers"], 2)
        self.assertEqual(result["abstracts"], 1)
        self.assertEqual(result["coverage"], [
            {"venue": "iclr", "year": 2025, "papers": 1, "abstracts": 0},
            {"venue": "iclr", "year": 2026, "papers": 1, "abstracts": 1},
        ])
        self.assertEqual(result["removed_raw_json_bytes"],
                         len(self.source_payload.encode()) + len(self.roster_payload.encode()))
        self.assertEqual(result["schema_version"], 2)
        self.assertLess(result["output_bytes"], result["original_snapshot_bytes"])
        self.assertEqual(self.rows(self.output, "PRAGMA freelist_count"), [(0,)])
        self.assertEqual(self.rows(self.output, "PRAGMA journal_mode"), [("delete",)])
        self.assertFalse(Path(str(self.output) + "-wal").exists())
        self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])

    def test_live_wal_included_without_changing_source_or_wal(self):
        with closing(sqlite3.connect(self.source)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("UPDATE papers SET abstract='New abstract committed in WAL' WHERE id=2")
            writer.commit()
            before = self.source.read_bytes()
            wal = Path(str(self.source) + "-wal")
            wal_before = wal.read_bytes()
            result = prune_collection(self.source, self.output)
            self.assertGreater(result["source_wal_bytes"], 0)
            self.assertEqual(result["abstracts"], 2)
            self.assertEqual(self.rows(self.output, "SELECT abstract FROM papers WHERE id=2"),
                             [("New abstract committed in WAL",)])
            self.assertEqual(self.source.read_bytes(), before)
            self.assertEqual(wal.read_bytes(), wal_before)

    def test_existing_output_and_same_path_never_overwritten(self):
        self.output.write_bytes(b"keep this")
        with self.assertRaises(FileExistsError):
            prune_collection(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep this")
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different files"):
            prune_collection(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), before)

    def test_corpus_or_embedding_input_rejected_before_database_initializer(self):
        for filename, extra in (("corpus.sqlite3", ""),
                                ("index.sqlite3", "CREATE TABLE embedding_meta (key TEXT, value TEXT);")):
            with self.subTest(filename=filename):
                path = self.root / filename
                with closing(sqlite3.connect(path)) as connection:
                    connection.executescript("CREATE TABLE papers (id INTEGER, title TEXT, abstract TEXT);" + extra)
                before = path.read_bytes()
                with patch("reference_finder.collection_cleanup.Database") as initializer:
                    with self.assertRaisesRegex(ValueError, "collection database"):
                        prune_collection(path, self.output)
                    initializer.assert_not_called()
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])

    def test_legacy_v0_and_already_pruned_collection_supported(self):
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute("PRAGMA user_version=0")
        prune_collection(self.source, self.output)
        second = self.root / "pruned-again.sqlite3"
        result = prune_collection(self.output, second)
        self.assertEqual(result["removed_raw_json_bytes"], 0)
        self.assertEqual(self.rows(self.output, "SELECT * FROM papers"), self.rows(second, "SELECT * FROM papers"))

    def test_invalid_foreign_key_never_published(self):
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute("INSERT INTO paper_venues VALUES (999,'iclr',2026,'dblp')")
            connection.commit()
        before = self.source.read_bytes()
        with self.assertRaises((ValueError, sqlite3.IntegrityError)):
            prune_collection(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])

    def test_cli_works_without_site_packages_or_model_dependencies(self):
        command = [sys.executable, "-S", str(Path(__file__).resolve().parents[1] / "package_database.py"),
                   "prune-collection", "--db", str(self.source), "--output", str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
