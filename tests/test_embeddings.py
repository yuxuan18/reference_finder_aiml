"""Persistence and retrieval checks using a deterministic, framework-free encoder."""
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

from reference_finder.embeddings import (
    DEFAULT_VECTOR_FORMAT, benchmark, build_index, encoder_config, export_corpus, index_status,
    load_config, pack_vector, read_corpus, search_index, unpack_vector,
)


class FakeEncoder:
    model_id = "test/tiny-encoder"
    revision = "a" * 40
    dimension = 2
    document_prompt = "passage: "
    query_prompt = "query: "
    device = "cpu"
    dtype = "float32"

    def __init__(self, vectors=None, failure=None):
        self.vectors = vectors or (lambda text: [1.0, 1.0])
        self.failure = failure
        self.document_batches = []
        self.query_batches = []
        self.synchronizations = 0

    def document_chunks(self, text, overlap=64):
        return [{"text": part, "token_count": len(part.split())}
                for part in text.split("||")]

    def encode_documents(self, texts, batch_size=32):
        self.document_batches.append(list(texts))
        vectors = [self.vectors(text) for text in texts]
        if self.failure:
            return self.failure(len(self.document_batches), vectors)
        return vectors

    def encode_queries(self, texts, batch_size=1):
        self.query_batches.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def synchronize(self):
        self.synchronizations += 1


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.sqlite3"
        self.index = self.root / "embeddings.sqlite3"
        with closing(sqlite3.connect(self.source)) as conn:
            conn.executescript("""
                CREATE TABLE papers (
                    id INTEGER PRIMARY KEY, title TEXT, abstract TEXT,
                    publication_year INTEGER, doi TEXT, url TEXT, authors_json TEXT
                );
                CREATE TABLE paper_venues (
                    paper_id INTEGER, venue TEXT, year INTEGER, provider TEXT
                );
                CREATE TABLE private_provider_payload (raw_json TEXT);
            """)
            conn.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?)", [
                (1, "Paper one", "First abstract", 2020, "10.1/one",
                 "https://example.test/one", '["Alice", "Bob"]'),
                (2, "Paper two", None, 2021, None,
                 "https://example.test/two", '["Carol"]'),
                (3, "Paper three", "Third abstract", 2022, "10.1/three", None, "[]"),
            ])
            conn.executemany("INSERT INTO paper_venues VALUES (?,?,?,?)", [
                (1, "icml", 2020, "openalex"),
                (1, "icml", 2020, "semantic_scholar"),
                (2, "icml", 2021, "openalex"),
            ])
            conn.execute("INSERT INTO private_provider_payload VALUES ('omit this')")
            conn.commit()
        self.encoder = FakeEncoder()
        self.config = encoder_config(self.encoder, max_tokens=2048, overlap=64, format_version=2)

    def modify_source(self, sql, values=()):
        with closing(sqlite3.connect(self.source)) as conn:
            conn.execute(sql, values)
            conn.commit()

    def build(self, encoder=None, **kwargs):
        return build_index(self.source, self.index, encoder or self.encoder,
                           self.config, **kwargs)

    def chunks(self):
        with closing(sqlite3.connect(self.index)) as conn:
            return conn.execute("SELECT paper_id,chunk_index,text FROM chunks ORDER BY paper_id,chunk_index").fetchall()

    def test_export_keeps_all_papers_titles_abstracts_and_metadata(self):
        destination = self.root / "transfer" / "corpus.sqlite3"
        result = export_corpus(self.source, destination)
        self.assertEqual(result["papers"], 3)
        self.assertEqual(result["abstracts"], 2)
        rows = read_corpus(destination)
        self.assertEqual(rows, read_corpus(self.source))
        self.assertIsNone(rows[1]["abstract"])
        self.assertEqual(json.loads(rows[0]["authors_json"]), ["Alice", "Bob"])
        self.assertEqual(json.loads(rows[0]["venues_json"]), [{"venue": "icml", "year": 2020}])
        # Paper three has no venue evidence but remains in the exported corpus.
        self.assertEqual(json.loads(rows[2]["venues_json"]), [])
        with closing(sqlite3.connect(destination)) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        self.assertEqual(tables, [("papers",)])

    def test_export_reads_committed_wal_and_excludes_uncommitted_changes(self):
        writer = sqlite3.connect(self.source)
        self.addCleanup(writer.close)
        self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("UPDATE papers SET abstract='Committed WAL abstract' WHERE id=1")
        writer.commit()
        self.assertTrue(Path(str(self.source) + "-wal").exists())
        writer.execute("UPDATE papers SET abstract='Uncommitted abstract' WHERE id=1")
        destination = self.root / "wal-snapshot.sqlite3"
        export_corpus(self.source, destination)
        self.assertEqual(read_corpus(destination)[0]["abstract"], "Committed WAL abstract")
        writer.rollback()

    def test_export_never_overwrites_existing_destination(self):
        destination = self.root / "keep.txt"
        destination.write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            export_corpus(self.source, destination)
        self.assertEqual(destination.read_text(encoding="utf-8"), "preserve me")

    def test_whitespace_abstract_is_counted_as_title_only(self):
        self.modify_source("UPDATE papers SET abstract='  \n  ' WHERE id=3")
        status = self.build()
        self.assertEqual(status["abstracts"], 1)
        self.assertEqual(status["title_only"], 2)
        self.assertEqual(status["embedded_title_only"], 2)

    def test_resume_skips_completed_and_unchanged_papers(self):
        first = self.build(limit=1, batch_size=2)
        self.assertEqual((first["embedded"], first["pending"]), (1, 2))
        self.assertEqual(load_config(self.index), self.config)
        second = self.build(batch_size=2)
        self.assertEqual((second["embedded"], second["pending"]), (3, 0))
        self.assertEqual(sum(map(len, self.encoder.document_batches)), 3)
        saved_chunks = self.chunks()
        third = self.build()
        self.assertEqual(third["embedded"], 3)
        self.assertEqual(sum(map(len, self.encoder.document_batches)), 3)
        self.assertEqual(self.chunks(), saved_chunks)

    def test_changed_text_reembeds_metadata_refreshes_and_deleted_papers_disappear(self):
        self.build()
        self.modify_source("UPDATE papers SET abstract='Changed abstract||New tail' WHERE id=1")
        self.modify_source("UPDATE papers SET doi='10.1/updated' WHERE id=2")
        self.modify_source("DELETE FROM papers WHERE id=3")
        resumed = FakeEncoder()
        status = self.build(encoder=resumed)
        self.assertEqual((status["papers"], status["embedded"], status["chunks"]), (2, 2, 3))
        self.assertEqual(sum(map(len, resumed.document_batches)), 2)
        self.assertIn("Changed abstract", resumed.document_batches[0][0])
        self.assertEqual({r[0] for r in self.chunks()}, {1, 2})
        self.assertFalse(any("First abstract" in r[2] for r in self.chunks()))
        results = search_index(self.index, resumed, "query")
        self.assertEqual(next(r["doi"] for r in results if r["paper_id"] == 2), "10.1/updated")

    def test_failed_reembedding_removes_stale_vectors(self):
        self.build()
        self.modify_source("UPDATE papers SET abstract='New unembedded text' WHERE id=1")
        broken = FakeEncoder(vectors=lambda text: [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "zero embedding"):
            self.build(encoder=broken)
        status = index_status(self.index)
        self.assertEqual((status["embedded"], status["pending"]), (2, 1))
        self.assertEqual({r[0] for r in self.chunks()}, {2, 3})
        self.assertEqual({r["paper_id"] for r in search_index(self.index, self.encoder, "query")}, {2, 3})

    def test_resume_rejects_changed_model_revision_prompts_or_chunk_settings(self):
        self.build(limit=1)
        alternatives = {
            "model_id": "test/different-model", "revision": "b" * 40,
            "dimension": 3, "document_prompt": "different: ",
            "query_prompt": "different: ", "max_tokens": 1024, "overlap": 16,
        }
        for field, value in alternatives.items():
            with self.subTest(field=field):
                config = dict(self.config, **{field: value})
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    build_index(self.source, self.index, self.encoder, config)
                self.assertEqual(index_status(self.index)["embedded"], 1)

    def test_search_uses_best_chunk_not_average_and_can_exclude_title_only(self):
        self.modify_source("UPDATE papers SET abstract='weak||strong' WHERE id=1")

        def vectors(text):
            if text == "strong":
                return [1.0, 0.0]
            if "weak" in text:
                return [0.0, 1.0]
            if "Paper two" in text:
                return [0.9, math.sqrt(0.19)]
            return [0.8, 0.6]

        encoder = FakeEncoder(vectors=vectors)
        self.build(encoder=encoder)
        matches = search_index(self.index, encoder, "target", top_k=3)
        self.assertEqual([m["paper_id"] for m in matches], [1, 2, 3])
        self.assertEqual(matches[0]["chunk_index"], 1)
        self.assertEqual(matches[0]["matching_text"], "strong")
        self.assertAlmostEqual(matches[0]["score"], 1.0)
        self.assertFalse(matches[1]["has_abstract"])
        abstracts = search_index(self.index, encoder, "target", abstracts_only=True)
        self.assertEqual([m["paper_id"] for m in abstracts], [1, 3])
        self.assertEqual(search_index(self.index, encoder, "target", top_k=1)[0]["paper_id"], 1)

    def test_search_rejects_incompatible_query_encoder(self):
        self.build()
        changed = FakeEncoder()
        changed.revision = "b" * 40
        with self.assertRaisesRegex(ValueError, "does not match"):
            search_index(self.index, changed, "target")
        self.assertEqual(changed.query_batches, [])

    def test_vector_validation_and_normalization(self):
        self.assertEqual(DEFAULT_VECTOR_FORMAT, "float16-le")
        self.assertEqual(len(pack_vector([3.0, 4.0], 2)), 4)
        self.assertEqual(pack_vector([3.0, 4.0], 2), struct.pack("<ee", 0.6, 0.8))
        normalized = list(unpack_vector(pack_vector([3.0, 4.0], 2)))
        self.assertAlmostEqual(normalized[0], 0.6, delta=0.00025)
        self.assertAlmostEqual(normalized[1], 0.8, delta=0.00025)
        full = pack_vector([3.0, 4.0], 2, "float32-le")
        self.assertEqual(len(full), 8)
        self.assertEqual(full, struct.pack("<ff", 0.6, 0.8))
        self.assertAlmostEqual(unpack_vector(full, "float32-le")[0], 0.6)
        for values in ([1.0], [1.0, 2.0, 3.0], [float("nan"), 0],
                       [float("inf"), 1], [0.0, 0.0]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    pack_vector(values, 2)

    def test_vector_normalization_handles_extreme_finite_values(self):
        for vector_format in ("float16-le", "float32-le"):
            for scale in (1e-310, 1e308):
                with self.subTest(vector_format=vector_format, scale=scale):
                    # Unscaled sum-of-squares underflows or overflows here.
                    values = unpack_vector(pack_vector([scale, scale], 2, vector_format), vector_format)
                    self.assertAlmostEqual(values[0], math.sqrt(0.5), delta=0.00025)
                    self.assertEqual(values[0], values[1])

    def test_unpack_rejects_malformed_nonfinite_and_zero_vectors(self):
        for vector_format, code in (("float16-le", "e"), ("float32-le", "f")):
            for blob in (b"", b"\x00", struct.pack("<2" + code, math.nan, 1),
                         struct.pack("<2" + code, math.inf, 1),
                         struct.pack("<2" + code, 0, 0)):
                with self.subTest(vector_format=vector_format, blob=blob):
                    with self.assertRaisesRegex(ValueError, "Corrupt index"):
                        unpack_vector(blob, vector_format)
        with self.assertRaisesRegex(ValueError, "Unsupported vector format"):
            pack_vector([1, 0], 2, "unknown")
        with self.assertRaisesRegex(ValueError, "Unsupported vector format"):
            unpack_vector(b"\x00\x3c", "unknown")

    def test_new_index_uses_version_two_half_sized_blobs(self):
        self.build(limit=1)
        self.assertEqual(self.config["format_version"], 2)
        self.assertEqual(self.config["vector_format"], "float16-le")
        self.build()
        with closing(sqlite3.connect(self.index)) as conn:
            sizes = conn.execute("SELECT DISTINCT length(vector) FROM chunks").fetchall()
        self.assertEqual(sizes, [(2 * self.encoder.dimension,)])

    def test_legacy_float32_index_remains_readable_resumable_and_searchable(self):
        self.config = encoder_config(self.encoder, 2048, 64, "float32-le", format_version=1)
        self.build(limit=1)
        self.assertEqual(load_config(self.index)["format_version"], 1)
        self.build()
        with closing(sqlite3.connect(self.index)) as conn:
            self.assertEqual(conn.execute("SELECT DISTINCT length(vector) FROM chunks").fetchall(), [(8,)])
            # Handwritten legacy bytes also check the reader independently of packing.
            conn.execute("UPDATE chunks SET vector=? WHERE paper_id=1", (struct.pack("<ff", 1, 0),))
            conn.commit()
        for without_numpy in (False, True):
            with self.subTest(without_numpy=without_numpy), patch.dict(
                    "sys.modules", {"numpy": None} if without_numpy else {}):
                result = search_index(self.index, self.encoder, "target")
                self.assertEqual(result[0]["paper_id"], 1)
                self.assertAlmostEqual(result[0]["score"], 1)
                self.assertAlmostEqual(result[1]["score"], math.sqrt(0.5), places=6)
        incompatible = encoder_config(self.encoder, 2048, 64, format_version=2)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            build_index(self.source, self.index, self.encoder, incompatible)

    def test_load_rejects_unknown_or_mismatched_storage_formats(self):
        self.build()
        for version, vector_format in ((1, "float16-le"), (2, "float32-le"),
                                       (4, "float16-le"), (2, "unknown")):
            with self.subTest(version=version, vector_format=vector_format):
                with closing(sqlite3.connect(self.index)) as conn:
                    invalid = dict(self.config, format_version=version, vector_format=vector_format)
                    conn.execute("UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(invalid),))
                    conn.commit()
                with self.assertRaisesRegex(ValueError, "Unsupported embedding index format"):
                    load_config(self.index)

    def test_search_renormalizes_half_vectors_for_actual_cosine(self):
        self.build(encoder=FakeEncoder(vectors=lambda text: [3, 4]))
        stored = unpack_vector(pack_vector([3, 4], 2))
        expected = stored[0] / math.hypot(*stored)
        for without_numpy in (False, True):
            with self.subTest(without_numpy=without_numpy), patch.dict(
                    "sys.modules", {"numpy": None} if without_numpy else {}):
                score = search_index(self.index, self.encoder, "target")[0]["score"]
                self.assertAlmostEqual(score, expected, places=6)
                self.assertGreater(abs(score - stored[0]), 0.00001)

    def test_search_retains_float32_query_precision_with_half_storage(self):
        self.build(encoder=FakeEncoder(vectors=lambda text: [0, 1]))
        vector = [1.0, 0.000123456]
        expected = vector[1] / math.hypot(*vector)
        with patch.object(self.encoder, "encode_queries", return_value=[vector]):
            for without_numpy in (False, True):
                with self.subTest(without_numpy=without_numpy), patch.dict(
                        "sys.modules", {"numpy": None} if without_numpy else {}):
                    score = search_index(self.index, self.encoder, "target")[0]["score"]
                    self.assertAlmostEqual(score, expected, places=10)

    def test_search_rejects_corrupt_stored_vectors_with_and_without_numpy(self):
        self.build()
        for blob in (b"\x00", struct.pack("<ee", math.nan, 1),
                     struct.pack("<ee", math.inf, 1), struct.pack("<ee", 0, 0)):
            with closing(sqlite3.connect(self.index)) as conn:
                conn.execute("UPDATE chunks SET vector=? WHERE paper_id=1", (blob,))
                conn.commit()
            for without_numpy in (False, True):
                with self.subTest(blob=blob, without_numpy=without_numpy), patch.dict(
                        "sys.modules", {"numpy": None} if without_numpy else {}):
                    with self.assertRaisesRegex(ValueError, "Corrupt index"):
                        search_index(self.index, self.encoder, "target")

    def test_failed_batch_preserves_earlier_commits_and_never_commits_partial_paper(self):
        self.modify_source("UPDATE papers SET abstract='start||tail' WHERE id=2")

        def failure(call, vectors):
            if call == 2:
                vectors[-1] = [float("nan"), 1.0]
            return vectors

        broken = FakeEncoder(failure=failure)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.build(encoder=broken, batch_size=1)
        status = index_status(self.index)
        self.assertEqual((status["embedded"], status["pending"], status["chunks"]), (1, 2, 1))
        self.assertEqual({r[0] for r in self.chunks()}, {1})
        resumed = FakeEncoder()
        finished = self.build(encoder=resumed, batch_size=2)
        self.assertEqual((finished["embedded"], finished["chunks"]), (3, 4))
        self.assertEqual(sum(map(len, resumed.document_batches)), 3)
        self.assertFalse(any("Paper one" in text for batch in resumed.document_batches for text in batch))

    def test_wrong_embedding_count_leaves_whole_batch_pending(self):
        broken = FakeEncoder(failure=lambda call, vectors: vectors[:-1])
        with self.assertRaisesRegex(ValueError, "wrong number"):
            self.build(encoder=broken, batch_size=3)
        status = index_status(self.index)
        self.assertEqual((status["embedded"], status["chunks"], status["pending"]), (0, 0, 3))

    def test_benchmark_samples_without_creating_an_index(self):
        before = set(self.root.iterdir())
        result = benchmark(self.source, self.encoder, limit=2, batch_size=2)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(result["papers"], 2)
        self.assertEqual(result["corpus_papers"], 3)
        self.assertGreater(result["papers_per_second"], 0)
        self.assertEqual(len(self.encoder.document_batches), 2)
        self.assertEqual(len(self.encoder.document_batches[0]), 1)
        self.assertEqual(self.encoder.synchronizations, 2)

    def test_corpus_and_index_cannot_be_the_same_file(self):
        with self.assertRaisesRegex(ValueError, "different files"):
            build_index(self.source, self.source, self.encoder, self.config)
        self.assertEqual(len(read_corpus(self.source)), 3)


class LinkedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        import hashlib
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.papers = self.root / "papers.sqlite3"
        self.index = self.root / "embeddings.sqlite3"
        with closing(sqlite3.connect(self.papers)) as conn:
            conn.execute("""CREATE TABLE papers (id INTEGER PRIMARY KEY,title TEXT,abstract TEXT,
                publication_year INTEGER,doi TEXT,url TEXT,authors_json TEXT)""")
            conn.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?)", [
                (1, "Résumé 安全研究", "weak||strong", 2024, "10.1234/one", None, '["Alice"]'),
                (2, "Title only", None, 2024, None, None, "[]"),
                (3, "Third paper", "Another abstract", 2024, None, None, "[]"),
            ])
            conn.execute("CREATE TABLE paper_venues (paper_id INTEGER,venue TEXT,year INTEGER)")
            conn.execute("INSERT INTO paper_venues VALUES (1,'icml',2024)")
            conn.commit()
        self.original_hash = hashlib.sha256(self.papers.read_bytes()).hexdigest()
        self.encoder = FakeEncoder(vectors=lambda text: [1, 0] if text == "strong" else [0, 1])
        self.config = encoder_config(self.encoder, 2048, 64)

    def build(self, encoder=None, **kwargs):
        return build_index(self.papers, self.index, encoder or self.encoder, self.config, **kwargs)

    def status(self):
        return index_status(self.index, paper_db=self.papers)

    def search(self, **kwargs):
        return search_index(self.index, self.encoder, "target", paper_db=self.papers, **kwargs)

    def rows(self, sql):
        with closing(sqlite3.connect(self.index)) as conn:
            return conn.execute(sql).fetchall()

    def update(self, sql):
        with closing(sqlite3.connect(self.papers)) as conn:
            conn.execute(sql)
            conn.commit()

    def test_index_contains_no_paper_text_and_source_is_unchanged(self):
        import hashlib
        status = self.build()
        self.assertEqual(status["config"]["storage_layout"], "linked-papers-v1")
        self.assertEqual((status["embedded"], status["chunks"]), (3, 4))
        self.assertEqual(hashlib.sha256(self.papers.read_bytes()).hexdigest(), self.original_hash)
        self.assertEqual(self.rows("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"),
                         [("chunks",), ("embedding_meta",), ("paper_state",)])
        self.assertEqual([r[1] for r in self.rows("PRAGMA table_info(chunks)")],
                         ["paper_id", "chunk_index", "token_count", "vector"])
        self.assertEqual([r[1] for r in self.rows("PRAGMA table_info(paper_state)")],
                         ["paper_id", "source_hash", "embedded"])
        self.assertNotIn("Résumé".encode(), self.index.read_bytes())
        for without_numpy in (False, True):
            with patch.dict("sys.modules", {"numpy": None} if without_numpy else {}):
                matches = self.search()
            self.assertEqual(matches[0]["title"], "Résumé 安全研究")
            self.assertEqual(matches[0]["chunk_index"], 1)
            self.assertEqual(matches[0]["abstract"], "weak||strong")
            self.assertAlmostEqual(matches[0]["score"], 1)
            self.assertEqual(matches[0]["venues"], [{"venue": "icml", "year": 2024}])
            self.assertNotIn("matching_text", matches[0])

    def test_resume_and_changed_text_reembed_only_pending(self):
        self.build(limit=1)
        self.assertEqual(self.status()["pending"], 2)
        self.build()
        self.build()
        self.assertEqual(len(self.encoder.document_batches), 2)
        self.update("UPDATE papers SET abstract='Changed abstract' WHERE id=1")
        self.assertEqual((self.status()["pending"], self.status()["stale"]), (1, 1))
        self.assertNotIn(1, [p["paper_id"] for p in self.search()])
        self.build()
        self.assertEqual(self.encoder.document_batches[-1], ["Résumé 安全研究\n\nChanged abstract"])
        self.assertEqual(self.status()["pending"], 0)

    def test_metadata_changes_and_row_reordering_preserve_vectors(self):
        self.build()
        blobs = self.rows("SELECT paper_id,chunk_index,vector FROM chunks ORDER BY paper_id,chunk_index")
        self.update("UPDATE papers SET doi='10.1234/new' WHERE id=1")
        self.update("VACUUM")
        self.build()
        self.assertEqual(len(self.encoder.document_batches), 1)
        self.assertEqual(self.search()[0]["doi"], "10.1234/new")
        self.assertEqual(self.rows("SELECT paper_id,chunk_index,vector FROM chunks ORDER BY paper_id,chunk_index"), blobs)

    def test_deleted_and_new_ids_are_reconciled_without_returning_stale_papers(self):
        self.build()
        self.update("DELETE FROM papers WHERE id=1")
        self.update("INSERT INTO papers VALUES (4,'New paper','New abstract',2025,NULL,NULL,'[]')")
        status = self.status()
        self.assertEqual((status["papers"], status["embedded"], status["pending"], status["missing_papers"]), (3, 2, 1, 1))
        self.assertNotIn(1, [p["paper_id"] for p in self.search()])
        self.build()
        self.assertEqual(self.rows("SELECT paper_id FROM paper_state ORDER BY paper_id"), [(2,), (3,), (4,)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM chunks WHERE paper_id=1"), [(0,)])

    def test_wrong_paper_database_or_reassigned_ids_never_return_wrong_vectors(self):
        self.build()
        self.update("UPDATE papers SET id=id+10")
        status = self.status()
        self.assertEqual((status["embedded"], status["pending"], status["missing_papers"]), (0, 3, 3))
        with self.assertRaisesRegex(ValueError, "matching --db"):
            self.search()
        self.assertEqual(self.encoder.query_batches, [])
        with self.assertRaisesRegex(ValueError, "needs its paper database"):
            index_status(self.index)
        with self.assertRaisesRegex(ValueError, "matching --db"):
            index_status(self.index, paper_db=self.root / "missing.sqlite3")

    def test_moving_both_files_and_minimal_metadata_snapshot_is_supported(self):
        self.build()
        before = self.search()
        moved = self.root / "moved"
        moved.mkdir()
        self.papers = self.papers.rename(moved / "renamed-papers.sqlite3")
        self.index = self.index.rename(moved / "renamed-vectors.sqlite3")
        self.assertEqual(self.search(), before)
        exported = self.root / "exported-papers.sqlite3"
        export_corpus(self.papers, exported)
        self.assertEqual(search_index(self.index, self.encoder, "target", paper_db=exported), before)
        self.assertNotIn(str(self.root), json.dumps(load_config(self.index)))

    def test_paper_changes_during_inference_leave_it_pending_and_unsearchable(self):
        original_encode = self.encoder.encode_documents
        def encode_and_update(texts, batch_size=32):
            output = original_encode(texts, batch_size)
            self.update("UPDATE papers SET abstract='New text during inference' WHERE id=1")
            return output
        with patch.object(self.encoder, "encode_documents", side_effect=encode_and_update):
            status = self.build()
        self.assertEqual((status["embedded"], status["pending"]), (2, 1))
        self.assertEqual(self.rows("SELECT * FROM chunks WHERE paper_id=1"), [])
        self.build()
        self.assertEqual(self.status()["pending"], 0)

    def test_failed_batch_atomicity_and_resume(self):
        def fail_second(call, vectors):
            if call == 2:
                vectors[-1] = [float("nan"), 1]
            return vectors
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.build(FakeEncoder(failure=fail_second), batch_size=1)
        self.assertEqual(self.status()["embedded"], 1)
        self.build()
        self.assertEqual(self.status()["pending"], 0)

    def test_missing_chunks_are_pending_not_complete(self):
        self.build()
        with closing(sqlite3.connect(self.index)) as conn:
            conn.execute("DELETE FROM chunks WHERE paper_id=1")
            conn.commit()
        self.assertEqual((self.status()["embedded"], self.status()["pending"]), (2, 1))
        self.assertNotIn(1, [p["paper_id"] for p in self.search()])
        self.build()
        self.assertEqual(self.status()["pending"], 0)
        self.assertEqual(len(self.encoder.document_batches), 2)

    def test_float32_conversion_keeps_linked_layout_and_no_text(self):
        from reference_finder.embedding_conversion import convert_index
        self.config = encoder_config(self.encoder, 2048, 64, "float32-le")
        self.build()
        destination = self.index.with_name("converted.sqlite3")
        result = convert_index(self.index, destination, paper_db=self.papers)
        self.assertEqual(result["config"]["format_version"], 3)
        self.assertEqual(result["config"]["storage_layout"], "linked-papers-v1")
        self.assertEqual(result["config"]["vector_format"], "float16-le")
        self.assertEqual(result["original_vector_bytes"], result["vector_bytes"] * 2)
        self.assertEqual(self.search(), search_index(destination, self.encoder, "target", paper_db=self.papers))

    def test_format_three_without_layout_is_rejected(self):
        self.build()
        with closing(sqlite3.connect(self.index)) as conn:
            config = dict(self.config)
            del config["storage_layout"]
            conn.execute("UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(config),))
            conn.commit()
        with self.assertRaisesRegex(ValueError, "requires linked-papers-v1"):
            load_config(self.index)

    def test_same_file_is_rejected_without_changing_papers(self):
        with self.assertRaisesRegex(ValueError, "different files"):
            build_index(self.papers, self.papers, self.encoder, self.config)


    def test_hardlinked_source_and_existing_paper_destination_are_rejected(self):
        import os
        import hashlib
        alias = self.root / "alias.sqlite3"
        os.link(self.papers, alias)
        with self.assertRaisesRegex(ValueError, "different files"):
            build_index(self.papers, alias, self.encoder, self.config)
        other_papers = self.root / "other-papers.sqlite3"
        other_papers.write_bytes(self.papers.read_bytes())
        with self.assertRaisesRegex(ValueError, "Not an embedding index"):
            build_index(self.papers, other_papers, self.encoder, self.config)
        self.assertEqual(hashlib.sha256(other_papers.read_bytes()).hexdigest(), self.original_hash)


if __name__ == "__main__":
    unittest.main()
