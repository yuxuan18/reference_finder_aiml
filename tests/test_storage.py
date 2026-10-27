import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from reference_finder.storage import Database, normalize_doi, reconstruct_abstract


def openalex_record(identifier="https://openalex.org/W1", doi=None, title="A learning method", author="Alice Example", abstract=None):
    return {
        "id": identifier, "doi": doi, "display_name": title,
        "publication_year": 2024, "publication_date": "2024-08-01",
        "authorships": [{"author": {"display_name": author}}] if author else [],
        "abstract_inverted_index": abstract,
    }


def scholar_record(identifier="s2-1", doi=None, title="A learning method", author="Alice Example", abstract=None):
    return {
        "paperId": identifier, "externalIds": {"DOI": doi} if doi else {},
        "title": title, "year": 2024, "publicationDate": "2024-08-01",
        "authors": [{"name": author}] if author else [], "abstract": abstract,
    }


class NormalizationTests(unittest.TestCase):
    def test_abstract_reconstructs_positions_and_repeated_tokens(self):
        self.assertEqual(
            reconstruct_abstract({"protocol.": [6], "We": [0, 3], "a": [5], "propose": [4], "study": [1], "learning.": [2]}),
            "We study learning. We propose a protocol.",
        )

    def test_missing_abstract_and_invalid_offsets(self):
        self.assertIsNone(reconstruct_abstract(None))
        self.assertIsNone(reconstruct_abstract({}))
        self.assertEqual(reconstruct_abstract({"valid": [2], "invalid": [-1, "0", 1.5], "ignored": None}), "valid")

    def test_doi_normalization(self):
        for value in ("10.1145/ABC.123", " https://doi.org/10.1145/ABC.123 ", "HTTP://DX.DOI.ORG/10.1145/ABC.123", "doi: 10.1145/ABC.123", "https://doi.org/10.1145%2FABC.123"):
            with self.subTest(value=value):
                self.assertEqual(normalize_doi(value), "10.1145/abc.123")
        for value in (None, "", "not-a-doi", "https://example.org/10.1145/abc", "10.1145/has spaces"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_doi(value))


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)

    def paper(self, paper_id):
        return self.db.conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()

    def test_doi_deduplication_retains_both_provider_records(self):
        oa = openalex_record(doi="https://doi.org/10.1145/ABC")
        s2 = scholar_record(doi="10.1145/abc", title="A learning method: revised metadata")
        first = self.db.upsert("openalex", oa, "icml", 2024)
        second = self.db.upsert("semantic_scholar", s2, "icml", 2024)
        self.assertEqual(first, second)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        records = self.db.conn.execute("SELECT provider, external_id FROM source_records ORDER BY provider").fetchall()
        self.assertEqual([tuple(row) for row in records], [("openalex", oa["id"]), ("semantic_scholar", s2["paperId"])])
        self.assertEqual(self.db.coverage("icml", 2024)["papers"], 1)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM paper_venues").fetchone()[0], 2)

    def test_missing_doi_merges_matching_title_and_first_author(self):
        first = self.db.upsert("openalex", openalex_record(title="A Learning Method!", author="ALICE EXAMPLE"), "iclr", 2024)
        second = self.db.upsert("semantic_scholar", scholar_record(title="A learning method", author="Alice Example"), "iclr", 2024)
        self.assertEqual(first, second)

    def test_title_fallback_rejects_different_or_missing_authors(self):
        first = self.db.upsert("openalex", openalex_record(), "iclr", 2024)
        different = self.db.upsert("semantic_scholar", scholar_record(identifier="different", author="Bob Example"), "iclr", 2024)
        missing = self.db.upsert("semantic_scholar", scholar_record(identifier="missing", author=None), "iclr", 2024)
        self.assertEqual(len({first, different, missing}), 3)

    def test_title_fallback_requires_same_venue_and_year(self):
        first = self.db.upsert("openalex", openalex_record(), "icml", 2024)
        different_venue = self.db.upsert("semantic_scholar", scholar_record(identifier="venue"), "sp", 2024)
        different_year = self.db.upsert("semantic_scholar", scholar_record(identifier="year"), "icml", 2023)
        self.assertEqual(len({first, different_venue, different_year}), 3)

    def test_title_fallback_rejects_blank_first_author_names(self):
        oa = openalex_record()
        oa["authorships"] = [{"author": {}}]
        s2 = scholar_record()
        s2["authors"] = [{"name": "   "}]
        first = self.db.upsert("openalex", oa, "iclr", 2024)
        second = self.db.upsert("semantic_scholar", s2, "iclr", 2024)
        self.assertNotEqual(first, second)

    def test_title_fallback_rejects_ambiguous_candidates(self):
        first = self.db.upsert("openalex", openalex_record(identifier="W1", doi="10.1145/first"), "icml", 2024)
        second = self.db.upsert("openalex", openalex_record(identifier="W2", doi="10.1145/second"), "icml", 2024)
        ambiguous = self.db.upsert("semantic_scholar", scholar_record(), "icml", 2024)
        self.assertEqual(len({first, second, ambiguous}), 3)

    def test_same_title_different_dois_do_not_merge(self):
        first = self.db.upsert("openalex", openalex_record(doi="10.1145/first"), "icml", 2024)
        second = self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/second"), "icml", 2024)
        self.assertNotEqual(first, second)

    def test_null_abstract_never_erases_existing_text(self):
        first = self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract={"Original": [0], "abstract.": [1]}), "icml", 2024)
        self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/abc", abstract=None), "icml", 2024)
        self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract=None), "icml", 2024)
        self.assertEqual(self.paper(first)["abstract"], "Original abstract.")
        self.assertEqual(self.paper(first)["abstract_source"], "openalex")

    def test_semantic_scholar_abstract_is_preferred(self):
        first = self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract={"OpenAlex": [0]}), "icml", 2024)
        self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/abc", abstract="  Semantic Scholar text.  "), "icml", 2024)
        self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract={"Later": [0], "OpenAlex": [1]}), "icml", 2024)
        self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/abc", abstract="  "), "icml", 2024)
        self.assertEqual(self.paper(first)["abstract"], "Semantic Scholar text.")
        self.assertEqual(self.paper(first)["abstract_source"], "semantic_scholar")

    def test_later_doi_bridges_records_and_preserves_memberships_and_provenance(self):
        first = self.db.upsert("openalex", openalex_record(title="Original title", abstract={"OpenAlex": [0]}), "icml", 2024)
        second = self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/bridge", title="Different indexed title", abstract="Preferred abstract"), "iclr", 2023)
        self.assertNotEqual(first, second)
        bridge = openalex_record(title="Original title", doi="10.1145/bridge")
        merged = self.db.upsert("openalex", bridge, "icml", 2024)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(self.paper(merged)["doi"], "10.1145/bridge")
        self.assertEqual(self.paper(merged)["abstract"], "Preferred abstract")
        self.assertEqual(self.paper(merged)["abstract_source"], "semantic_scholar")
        memberships = self.db.conn.execute("SELECT venue, year, provider FROM paper_venues WHERE paper_id=? ORDER BY venue", (merged,)).fetchall()
        self.assertEqual([tuple(row) for row in memberships], [("iclr", 2023, "semantic_scholar"), ("icml", 2024, "openalex")])
        records = self.db.conn.execute("SELECT provider, paper_id FROM source_records ORDER BY provider").fetchall()
        self.assertEqual([(row["provider"], row["paper_id"]) for row in records], [("openalex", merged), ("semantic_scholar", merged)])
        self.assertEqual(self.db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_source_response_bodies_are_not_retained(self):
        record = scholar_record(abstract="The canonical abstract.")
        record["externalIds"]["DBLP"] = "conf/icml/Example24"
        record["unused_response"] = "Do not retain this response payload."
        paper_id = self.db.upsert("semantic_scholar", record, "icml", 2024)
        source = dict(self.db.conn.execute("SELECT * FROM source_records").fetchone())
        self.assertEqual(set(source), {"provider", "external_id", "paper_id", "dblp_key", "fetched_at"})
        self.assertEqual(source["dblp_key"], "conf/icml/Example24")
        self.assertNotIn(record["unused_response"], json.dumps(source))
        self.assertNotIn(record["abstract"], json.dumps(source))
        self.assertEqual(self.paper(paper_id)["abstract"], record["abstract"])

    def test_page_data_and_checkpoint_roll_back_together(self):
        job = '{"provider":"openalex","venue":"icml","year":2024}'
        with self.db.conn:
            first = self.db.upsert("openalex", openalex_record(), "icml", 2024)
            self.db.save_checkpoint(job, "first-cursor", False, 10, 1)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with self.db.conn:
                self.db.upsert("semantic_scholar", scholar_record(identifier="new", title="A different paper"), "icml", 2024)
                self.db.save_checkpoint(job, "second-cursor", True, 20, 1)
                raise RuntimeError("interrupted")
        self.assertEqual(self.db.conn.execute("SELECT id FROM papers").fetchall()[0][0], first)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM source_records").fetchone()[0], 1)
        checkpoint = self.db.checkpoint(job)
        self.assertEqual((checkpoint["cursor"], checkpoint["done"], checkpoint["fetched"], checkpoint["accepted"]), ("first-cursor", 0, 10, 1))
        with self.db.conn:
            self.db.save_checkpoint(job, None, True, 20, 2)
        checkpoint = self.db.checkpoint(job)
        self.assertEqual((checkpoint["cursor"], checkpoint["done"], checkpoint["fetched"], checkpoint["accepted"]), (None, 1, 30, 3))


class DatabaseMigrationTests(unittest.TestCase):
    """Fixtures use the old schema independently of Database's current schema."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "collection.sqlite3"

    def legacy_database(self, version=1, invalid_roster_json=False):
        connection = sqlite3.connect(str(self.path))
        connection.executescript("""
            PRAGMA foreign_keys=ON;
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
                PRIMARY KEY(provider,external_id)
            );
            CREATE INDEX source_records_paper ON source_records(paper_id);
            CREATE TABLE roster_entries (
                provider TEXT NOT NULL, external_id TEXT NOT NULL, venue TEXT NOT NULL,
                year INTEGER NOT NULL, paper_id INTEGER NOT NULL REFERENCES papers(id),
                title TEXT NOT NULL, doi TEXT, url TEXT, authors_json TEXT NOT NULL,
                section TEXT, raw_json TEXT NOT NULL,
                PRIMARY KEY(provider,external_id,venue,year)
            );
            CREATE INDEX roster_entries_paper ON roster_entries(paper_id);
            CREATE TABLE checkpoints (
                job TEXT PRIMARY KEY, cursor TEXT, done INTEGER NOT NULL DEFAULT 0,
                fetched INTEGER NOT NULL DEFAULT 0, accepted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE paper_venues (
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                venue TEXT NOT NULL, year INTEGER NOT NULL, provider TEXT NOT NULL,
                PRIMARY KEY(paper_id,venue,year,provider)
            );
        """)
        self.review = {
            "roster_review": {"action": "annotate", "note": "Keep pending confirmation.",
                              "evidence_urls": ["https://proceedings.example/evidence"]},
            "roster_match": {"method": "reviewed_alias", "dblp_key": "conf/icml/Example24", "reason": "Changed title."},
        }
        with connection:
            connection.execute("PRAGMA user_version=" + str(version))
            connection.execute("INSERT INTO papers VALUES (41,NULL,?,?,?,?,?,?,?,?,?,?)", (
                "Preserved canonical title", "preserved canonical title", 2024, "2024-08-01", '["Ada Researcher"]',
                "Canonical abstract.", "semantic_scholar", "https://paper.example", "created", "updated"))
            connection.execute("INSERT INTO source_records VALUES (?,?,?,?,?)", (
                "semantic_scholar", "s2-original", 41,
                json.dumps({"externalIds": {"DBLP": "conf/icml/Example24"}, "abstract": "Duplicate source abstract."}), "fetched"))
            connection.execute("INSERT INTO roster_entries VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                "pmlr", "original-roster", "icml", 2024, 41, "Original proceedings title", None,
                "https://roster.example", '["Ada Researcher"]', "Research papers",
                "broken-json" if invalid_roster_json else json.dumps(dict(self.review, abstract="Duplicate roster abstract."))))
            connection.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?)", ("retained-job", "resume-cursor", 0, 10, 8, "checkpoint-time"))
            connection.execute("INSERT INTO paper_venues VALUES (?,?,?,?)", (41, "icml", 2024, "semantic_scholar"))
        before = {table: connection.execute("SELECT * FROM " + table).fetchall()
                  for table in ("papers", "checkpoints", "paper_venues")}
        connection.close()
        return before

    def test_legacy_migration_preserves_papers_links_reviews_and_checkpoints(self):
        from reference_finder.roster import RosterStore

        for version in (0, 1):
            with self.subTest(version=version):
                if self.path.exists():
                    self.path.unlink()
                before = self.legacy_database(version)
                db = Database(self.path)
                try:
                    self.assertEqual(db.conn.execute("PRAGMA user_version").fetchone()[0], 2)
                    for table, rows in before.items():
                        self.assertEqual([tuple(row) for row in db.conn.execute("SELECT * FROM " + table)], rows)
                    source = dict(db.conn.execute("SELECT * FROM source_records").fetchone())
                    self.assertEqual(source, {"provider": "semantic_scholar", "external_id": "s2-original", "paper_id": 41,
                                              "dblp_key": "conf/icml/Example24", "fetched_at": "fetched"})
                    roster = dict(db.conn.execute("SELECT * FROM roster_entries").fetchone())
                    self.assertNotIn("raw_json", roster)
                    self.assertEqual(json.loads(roster["review_json"]), self.review)
                    self.assertEqual(db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
                    store = RosterStore(db)
                    paper = next(store.papers(["icml"], 2024, 2024))
                    self.assertEqual(paper["evidence"][0]["review_note"], "Keep pending confirmation.")
                    self.assertEqual(paper["evidence"][0]["review_evidence_urls"], self.review["roster_review"]["evidence_urls"])
                    # DBLP keys must still bridge a differently named API record
                    # from a different publication/event year after migration.
                    store.save("dblp", "icml", 2025, {"status": "complete", "records": [{
                        "external_id": "conf/icml/Example24", "title": "A later title", "year": 2025,
                    }]})
                    self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
                    linked = db.conn.execute("SELECT id,abstract FROM papers").fetchone()
                    self.assertEqual(tuple(linked), (41, "Canonical abstract."))
                    snapshot = list(db.conn.iterdump())
                finally:
                    db.conn.close()
                reopened = Database(self.path)
                try:
                    self.assertEqual(list(reopened.conn.iterdump()), snapshot)
                finally:
                    reopened.conn.close()

    def test_unversioned_legacy_collection_without_roster_tables_migrates(self):
        before = self.legacy_database(version=0)
        with sqlite3.connect(str(self.path)) as connection:
            connection.execute("DROP TABLE roster_entries")
        db = Database(self.path)
        try:
            self.assertEqual(db.conn.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual([tuple(row) for row in db.conn.execute("SELECT * FROM papers")], before["papers"])
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM roster_entries").fetchone()[0], 0)
            self.assertEqual(db.conn.execute("SELECT dblp_key FROM source_records").fetchone()[0], "conf/icml/Example24")
            self.assertEqual(db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        finally:
            db.conn.close()

    def test_failed_migration_rolls_back_both_tables_and_schema_version(self):
        before = self.legacy_database(invalid_roster_json=True)
        with self.assertRaisesRegex(ValueError, "invalid JSON in roster_entries"):
            Database(self.path)
        with sqlite3.connect(str(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            for table, rows in before.items():
                self.assertEqual(connection.execute("SELECT * FROM " + table).fetchall(), rows)
            for table in ("source_records", "roster_entries"):
                columns = [row[1] for row in connection.execute("PRAGMA table_info(" + table + ")")]
                self.assertIn("raw_json", columns)
            self.assertEqual(connection.execute("SELECT name FROM sqlite_master WHERE name LIKE '%migration_v2%'").fetchall(), [])
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
