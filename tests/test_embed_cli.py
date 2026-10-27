"""Exercise CLI safety and checkpoint selection without loading model weights."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import embed_papers
from reference_finder.embeddings import encoder_config


class EmbeddingCliTests(unittest.TestCase):
    def test_export_works_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sqlite3"
            destination = Path(directory) / "snapshot.sqlite3"
            with sqlite3.connect(source) as conn:
                conn.execute("CREATE TABLE papers (id INTEGER, title TEXT, abstract TEXT)")
                conn.execute("INSERT INTO papers VALUES (1, 'A title', NULL)")
            output = io.StringIO()
            with patch.object(embed_papers, "make_encoder") as load, contextlib.redirect_stdout(output):
                result = embed_papers.main(["export", "--db", str(source), "--output", str(destination)])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.getvalue())["papers"], 1)
            load.assert_not_called()

    def test_missing_corpus_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(embed_papers, "make_encoder") as load, self.assertLogs(level="ERROR"):
                result = embed_papers.main(["build", "--db", str(Path(directory) / "missing.sqlite3")])
            self.assertEqual(result, 1)
            load.assert_not_called()

    def test_missing_explicit_same_database_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "corpus.sqlite3")
            with patch.object(embed_papers, "make_encoder") as load, self.assertLogs(level="ERROR"):
                result = embed_papers.main(["build", "--db", db, "--index", db])
            self.assertEqual(result, 1)
            load.assert_not_called()

    def test_resume_reuses_immutable_revision_without_resolving_main(self):
        encoder = SimpleNamespace(
            model_id="test/model", revision="a" * 40, dimension=2,
            document_prompt="passage: ", query_prompt="query: ",
            device="cpu", dtype="float32",
        )
        config = encoder_config(encoder, 1024, 32)
        args = embed_papers.parser().parse_args(["build", "--model", "test/model"])
        with patch.object(embed_papers.sys, "version_info", (3, 11)), \
                patch("reference_finder.embedding_backend.resolve_revision") as resolve, \
                patch("reference_finder.embedding_backend.SentenceEncoder", return_value=encoder) as load:
            _, actual = embed_papers.make_encoder(args, config)
        self.assertEqual(actual, config)
        resolve.assert_not_called()
        load.assert_called_once_with("test/model", "a" * 40, max_tokens=1024, device="auto", dtype="auto")

    def test_changed_model_is_rejected_without_network_or_model_load(self):
        encoder = SimpleNamespace(model_id="test/model", revision="a" * 40, dimension=2,
                                  document_prompt="passage: ", query_prompt="query: ")
        config = encoder_config(encoder, 1024, 32)
        args = embed_papers.parser().parse_args(["build", "--model", "different/model"])
        with patch.object(embed_papers.sys, "version_info", (3, 11)), \
                patch("reference_finder.embedding_backend.resolve_revision") as resolve, \
                patch("reference_finder.embedding_backend.SentenceEncoder") as load:
            with self.assertRaisesRegex(ValueError, "settings differ"):
                embed_papers.make_encoder(args, config)
        resolve.assert_not_called()
        load.assert_not_called()

    def test_qwen_profile_loads_on_python_396(self):
        encoder = SimpleNamespace(
            model_id=embed_papers.DEFAULT_MODEL, revision="a" * 40, dimension=2560,
            document_prompt="", query_prompt="Instruct: Retrieve related papers.\nQuery:",
            device="mps", dtype="float16",
        )
        args = embed_papers.parser().parse_args(["build"])
        with patch.object(embed_papers.sys, "version_info", (3, 9, 6)), \
                patch("reference_finder.embedding_backend.resolve_revision", return_value="a" * 40), \
                patch("reference_finder.embedding_backend.SentenceEncoder", return_value=encoder) as load:
            _, config = embed_papers.make_encoder(args)
        self.assertEqual(config["model_id"], "Qwen/Qwen3-Embedding-4B")
        self.assertEqual(config["document_prompt"], "")
        load.assert_called_once()

    def test_new_default_does_not_resume_a_previous_model_index(self):
        encoder = SimpleNamespace(
            model_id="example/previous-model", revision="a" * 40,
            dimension=2048, document_prompt="passage: ", query_prompt="query: ",
        )
        args = embed_papers.parser().parse_args(["build"])
        config = encoder_config(encoder, 2048, 64)
        with patch("reference_finder.embedding_backend.SentenceEncoder") as load:
            with self.assertRaisesRegex(ValueError, "settings differ"):
                embed_papers.make_encoder(args, config)
        load.assert_not_called()

    def test_build_search_and_status_use_separate_paper_and_vector_files(self):
        for argv in (["build"], ["status"], ["search", "--query", "abstract"]):
            with self.subTest(argv=argv):
                args = embed_papers.parser().parse_args(argv)
                self.assertEqual(args.db, "data/papers.sqlite3")
                self.assertEqual(args.index, "data/embeddings.sqlite3")
                args = embed_papers.parser().parse_args(argv + ["--db", "papers.sqlite3", "--index", "vectors.sqlite3"])
                self.assertEqual(args.db, "papers.sqlite3")
                self.assertEqual(args.index, "vectors.sqlite3")

    def test_legacy_build_resume_and_search_keep_float32_storage(self):
        encoder = SimpleNamespace(
            model_id=embed_papers.DEFAULT_MODEL, revision="a" * 40, dimension=2,
            document_prompt="", query_prompt="query: ", device="cpu", dtype="float32",
        )
        config = encoder_config(encoder, 1024, 32, vector_format="float32-le", format_version=1)
        for argv in (["build"], ["search", "--query", "abstract"]):
            with self.subTest(argv=argv):
                args = embed_papers.parser().parse_args(argv)
                with patch("reference_finder.embedding_backend.resolve_revision") as resolve, \
                        patch("reference_finder.embedding_backend.SentenceEncoder", return_value=encoder):
                    _, actual = embed_papers.make_encoder(args, config)
                self.assertEqual(actual, config)
                self.assertEqual(actual["format_version"], 1)
                resolve.assert_not_called()

    def test_explicit_storage_mismatch_is_rejected_before_model_load(self):
        encoder = SimpleNamespace(model_id=embed_papers.DEFAULT_MODEL, revision="a" * 40,
                                  dimension=2, document_prompt="", query_prompt="query: ")
        config = encoder_config(encoder, 1024, 32, vector_format="float32-le", format_version=1)
        args = embed_papers.parser().parse_args(["build", "--storage-dtype", "float16"])
        with patch("reference_finder.embedding_backend.resolve_revision") as resolve, \
                patch("reference_finder.embedding_backend.SentenceEncoder") as load:
            with self.assertRaisesRegex(ValueError, "Storage precision differs"):
                embed_papers.make_encoder(args, config)
        resolve.assert_not_called()
        load.assert_not_called()

    def test_storage_dtype_is_independent_of_model_dtype(self):
        encoder = SimpleNamespace(model_id=embed_papers.DEFAULT_MODEL, revision="a" * 40,
                                  dimension=2, document_prompt="", query_prompt="query: ",
                                  device="cpu", dtype="float32")
        for storage_dtype in (None, "float16", "float32"):
            argv = ["build", "--dtype", "float32"]
            if storage_dtype:
                argv += ["--storage-dtype", storage_dtype]
            with self.subTest(storage_dtype=storage_dtype):
                args = embed_papers.parser().parse_args(argv)
                with patch("reference_finder.embedding_backend.resolve_revision", return_value="a" * 40), \
                        patch("reference_finder.embedding_backend.SentenceEncoder", return_value=encoder) as load:
                    _, config = embed_papers.make_encoder(args)
                self.assertEqual(config["vector_format"], (storage_dtype or "float16") + "-le")
                self.assertEqual(load.call_args.kwargs["dtype"], "float32")


    def test_search_with_missing_index_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "missing.sqlite3"
            with patch.object(embed_papers, "make_encoder") as load, self.assertLogs(level="ERROR"):
                result = embed_papers.main(["search", "--index", str(index), "--query", "target"])
            self.assertEqual(result, 1)
            load.assert_not_called()

    def test_build_links_separate_files_and_search_rejects_wrong_papers_before_loading_model(self):
        from test_embeddings import FakeEncoder
        encoder = FakeEncoder()
        config = encoder_config(encoder, 2048, 64)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "papers.sqlite3"
            index = Path(directory) / "embeddings.sqlite3"
            with sqlite3.connect(source) as conn:
                conn.execute("CREATE TABLE papers (id INTEGER,title TEXT,abstract TEXT)")
                conn.execute("INSERT INTO papers VALUES (1,'A paper','An abstract')")
            output = io.StringIO()
            with patch.object(embed_papers, "make_encoder", return_value=(encoder, config)), \
                    contextlib.redirect_stdout(output):
                result = embed_papers.main(["build", "--db", str(source), "--index", str(index)])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.getvalue())["embedded"], 1)
            with sqlite3.connect(source) as conn:
                conn.execute("UPDATE papers SET abstract='Different content'")
            with patch.object(embed_papers, "make_encoder") as load, self.assertLogs(level="ERROR"):
                result = embed_papers.main(["search", "--db", str(source), "--index", str(index), "--query", "target"])
            self.assertEqual(result, 1)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
