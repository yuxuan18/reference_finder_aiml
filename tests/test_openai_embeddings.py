"""OpenAI adapter and full index lifecycle, with no paid API requests."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import embed_papers
from reference_finder.embeddings import build_index, encoder_config, index_status, load_config, search_index
from reference_finder.openai_embedding_backend import (
    API_REVISION, DEFAULT_OPENAI_MODEL, OpenAIEncoder, estimate_cost,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=True, truncation=False):
        return list(range(len(text)))


def response(texts, dimensions=2, reverse=True):
    data = [SimpleNamespace(index=i, embedding=[3.0, 4.0] if "apple" in text else [4.0, 3.0])
            for i, text in enumerate(texts)]
    return SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=list(reversed(data)) if reverse else data,
                           usage=SimpleNamespace(total_tokens=sum(len(t) for t in texts)))


def client():
    return SimpleNamespace(embeddings=SimpleNamespace(create=Mock(side_effect=lambda **kw: response(kw["input"]))))


def encoder(api=None, **kwargs):
    return OpenAIEncoder(client=api or client(), tokenizer=Tokenizer(), dimensions=2, **kwargs)


class OpenAIBackendTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("openai") and importlib.util.find_spec("httpx"), "optional OpenAI SDK not installed")
    def test_real_sdk_request_serialization_uses_mock_http_only(self):
        import httpx
        from openai import OpenAI

        requests = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(str(request.url), "https://api.openai.com/v1/embeddings")
            self.assertEqual(payload["encoding_format"], "float")
            self.assertEqual(payload["dimensions"], 2)
            return httpx.Response(200, json=dict(model=payload["model"], object="list", data=[
                dict(object="embedding", index=i, embedding=[3.0, 4.0])
                for i in reversed(range(len(payload["input"])))], usage=dict(prompt_tokens=4, total_tokens=4)))

        with OpenAI(api_key="test-only-placeholder", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle))) as api:
            vectors = encoder(api).encode_documents(["first", "second"])
        self.assertEqual(vectors, [[0.6, 0.8], [0.6, 0.8]])
        self.assertEqual(requests[0]["input"], ["first", "second"])

    def test_plain_input_dimensions_response_order_and_normalization(self):
        api = client()
        model = encoder(api)
        vectors = model.encode_documents(["apple", "banana"], batch_size=2)
        self.assertEqual(vectors, [[0.6, 0.8], [0.8, 0.6]])
        request = api.embeddings.create.call_args.kwargs
        self.assertEqual(request["input"], ["apple", "banana"])
        self.assertEqual(request["model"], "text-embedding-3-large")
        self.assertEqual(request["dimensions"], 2)
        self.assertEqual(request["encoding_format"], "float")
        self.assertEqual(model.input_tokens, 11)
        self.assertEqual(model.document_prompt, model.query_prompt)

    def test_request_partition_respects_input_and_total_token_limits(self):
        api = client()
        model = encoder(api)
        with patch("reference_finder.openai_embedding_backend.MAX_REQUEST_TOKENS", 5), \
                patch("reference_finder.openai_embedding_backend.MAX_REQUEST_INPUTS", 2):
            vectors = model.encode_documents(["aa", "bb", "cc", "dd", "eee"], batch_size=100)
        batches = [call.kwargs["input"] for call in api.embeddings.create.call_args_list]
        self.assertEqual(batches, [["aa", "bb"], ["cc", "dd"], ["eee"]])
        self.assertEqual(len(vectors), 5)
        # Even a large allowed input count must split at the token budget.
        api.embeddings.create.reset_mock()
        with patch("reference_finder.openai_embedding_backend.MAX_REQUEST_TOKENS", 5):
            model.encode_documents(["aaa", "bbb", "ccc"], batch_size=100)
        self.assertEqual(api.embeddings.create.call_count, 3)

    def test_rejects_invalid_inputs_before_any_api_call(self):
        api = client()
        model = encoder(api, max_tokens=8)
        for texts in ([""], ["   "], [None], ["x" * 9]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                model.encode_documents(texts)
        with self.assertRaises(TypeError):
            model.encode_documents("single string")
        self.assertEqual(model.encode_documents([]), [])
        api.embeddings.create.assert_not_called()

    def test_response_validation_rejects_wrong_vectors_indices_and_model(self):
        invalid = [
            SimpleNamespace(model="other-model", data=[], usage=None),
            SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=None, usage=None),
            SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=[SimpleNamespace(index=0, embedding=[0, 0])], usage=None),
            SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=[SimpleNamespace(index=0, embedding=[math.nan, 1])], usage=None),
            SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=[SimpleNamespace(index=0, embedding=[1])], usage=None),
            SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=[SimpleNamespace(index=True, embedding=[1, 1])], usage=None),
            SimpleNamespace(model=DEFAULT_OPENAI_MODEL, data=[SimpleNamespace(index=0, embedding=None)], usage=None),
        ]
        for payload in invalid:
            api = client()
            api.embeddings.create.side_effect = None
            api.embeddings.create.return_value = payload
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                encoder(api).encode_queries(["query"])
        api = client()
        api.embeddings.create.side_effect = lambda **kw: SimpleNamespace(model=DEFAULT_OPENAI_MODEL,
            data=[SimpleNamespace(index=0, embedding=[1, 1]), SimpleNamespace(index=0, embedding=[1, 1])])
        with self.assertRaisesRegex(ValueError, "indices"):
            encoder(api).encode_documents(["one", "two"])

    def test_error_redaction_never_echoes_key_or_private_input(self):
        class Failure(Exception):
            status_code = 429
        api = client()
        api.embeddings.create.side_effect = Failure("sk-test-secret and confidential-query")
        with self.assertRaises(RuntimeError) as caught:
            encoder(api).encode_queries(["confidential-query"])
        self.assertIn("HTTP 429", str(caught.exception))
        self.assertNotIn("sk-test-secret", str(caught.exception))
        self.assertNotIn("confidential-query", str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)

    def test_offline_mode_requires_neither_credentials_nor_sdk(self):
        with patch.dict(sys.modules, {"openai": None, "torch": None, "sentence_transformers": None}), \
                patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            model = OpenAIEncoder(offline=True, tokenizer=Tokenizer(), max_tokens=5)
            self.assertEqual(model.dimension, 3072)
            text = "Unicode 🙂 tail"
            chunks = model.document_chunks(text, overlap=0)
            self.assertEqual("".join(c["text"] for c in chunks), text)
            self.assertTrue(all(c["token_count"] <= 5 for c in chunks))
            with self.assertRaisesRegex(RuntimeError, "Token-counting"):
                model.encode_queries(["hi"])

    def test_missing_key_and_invalid_model_settings_fail_before_tokenizer_load(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                patch("reference_finder.openai_embedding_backend.OpenAITokenizer") as tokenizer:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIEncoder()
            tokenizer.assert_not_called()
        for kwargs in (dict(model_id="unknown"), dict(dimensions=3073), dict(dimensions=True),
                       dict(max_tokens=8193), dict(max_tokens=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OpenAIEncoder(offline=True, tokenizer=Tokenizer(), **kwargs)


class OpenAIIndexTests(unittest.TestCase):
    def setUp(self):
        # CLI unit tests must not load the user's real .env into later tests.
        environment = patch("reference_finder.config.load_environment")
        environment.start()
        self.addCleanup(environment.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "papers.sqlite3"
        self.index = Path(temp.name) / "openai.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE papers (id INTEGER,title TEXT,abstract TEXT)")
            conn.executemany("INSERT INTO papers VALUES (?,?,?)", [(1, "apple", "fruit"), (2, "banana", "fruit"), (3, "cherry", "fruit")])
        self.api = client()
        self.encoder = encoder(self.api)
        self.config = encoder_config(self.encoder, 8191, 0)

    def test_resumable_build_search_and_stale_text_use_the_same_provider(self):
        first = build_index(self.db, self.index, self.encoder, self.config, batch_size=1, limit=1)
        self.assertEqual(first["embedded"], 1)
        self.api.embeddings.create.side_effect = [response(["banana\n\nfruit"]), RuntimeError("temporary")]
        with self.assertRaises(RuntimeError):
            build_index(self.db, self.index, self.encoder, self.config, batch_size=1)
        self.assertEqual(index_status(self.index, self.db)["embedded"], 2)
        self.api.embeddings.create.side_effect = lambda **kw: response(kw["input"])
        self.api.embeddings.create.reset_mock()
        final = build_index(self.db, self.index, self.encoder, self.config, batch_size=1)
        self.assertEqual(final["embedded"], 3)
        self.assertEqual(self.api.embeddings.create.call_count, 1)
        self.assertEqual(load_config(self.index)["provider"], "openai")
        self.assertEqual(load_config(self.index)["dimension"], 2)
        matches = search_index(self.index, self.encoder, "apple", paper_db=self.db)
        self.assertEqual(matches[0]["paper_id"], 1)
        self.assertEqual(self.api.embeddings.create.call_args.kwargs["input"], ["apple"])
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE papers SET abstract='changed' WHERE id=2")
        self.api.embeddings.create.reset_mock()
        build_index(self.db, self.index, self.encoder, self.config)
        self.assertEqual(self.api.embeddings.create.call_count, 1)
        self.assertEqual(self.api.embeddings.create.call_args.kwargs["input"], ["banana\n\nchanged"])

    def test_cost_estimate_does_not_call_api_or_modify_files(self):
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        model = OpenAIEncoder(offline=True, tokenizer=Tokenizer(), dimensions=2)
        estimated = estimate_cost(self.db, model, overlap=0, completed={1}, limit=1)
        self.assertEqual(estimated["pending_papers"], 2)
        self.assertEqual(estimated["selected_papers"], 1)
        self.assertEqual(estimated["input_tokens"], len("banana\n\nfruit"))
        self.assertEqual(estimated["api_requests"], 0)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)
        self.assertFalse(self.index.exists())

    def test_cli_search_and_resume_automatically_select_saved_api_model(self):
        for argv in (["search", "--query", "test"], ["build"]):
            args = embed_papers.parser().parse_args(argv)
            with patch("reference_finder.openai_embedding_backend.OpenAIEncoder", return_value=self.encoder), \
                    patch("reference_finder.embedding_backend.resolve_revision") as hub:
                _, config = embed_papers.make_encoder(args, self.config)
            self.assertEqual(config, self.config)
            hub.assert_not_called()
        for flags in (["--dimensions", "3"], ["--model", "text-embedding-3-small"], ["--provider", "local"]):
            args = embed_papers.parser().parse_args(["build"] + flags)
            with patch("reference_finder.openai_embedding_backend.OpenAIEncoder") as load:
                with self.assertRaisesRegex(ValueError, "differs|differ"):
                    embed_papers.make_encoder(args, self.config)
                load.assert_not_called()

    def test_model_name_infers_openai_without_loading_local_frameworks(self):
        args = embed_papers.parser().parse_args(["build", "--model", "text-embedding-3-large", "--dimensions", "2", "--overlap", "0"])
        with patch("reference_finder.openai_embedding_backend.OpenAIEncoder", return_value=self.encoder), \
                patch("reference_finder.embedding_backend.SentenceEncoder") as local:
            _, config = embed_papers.make_encoder(args)
        self.assertEqual(config["provider"], "openai")
        local.assert_not_called()

    def test_estimate_cli_skips_completed_inputs_without_authentication(self):
        build_index(self.db, self.index, self.encoder, self.config, limit=1)
        offline = OpenAIEncoder(offline=True, tokenizer=Tokenizer(), dimensions=2)
        output = io.StringIO()
        before = hashlib.sha256(self.index.read_bytes()).hexdigest()
        with patch.object(embed_papers, "make_encoder", return_value=(offline, self.config)), contextlib.redirect_stdout(output):
            result = embed_papers.main(["estimate", "--db", str(self.db), "--index", str(self.index)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["selected_papers"], 2)
        self.assertEqual(hashlib.sha256(self.index.read_bytes()).hexdigest(), before)

    def test_api_config_never_contains_credentials(self):
        self.assertEqual(self.config["revision"], API_REVISION)
        self.assertNotIn("api_key", self.config)
        self.assertEqual(self.config["tokenizer"], "cl100k_base")
        other = dict(self.config, model_id="Qwen/Qwen3-Embedding-4B", provider="local")
        args = embed_papers.parser().parse_args(["build", "--provider", "openai"])
        with patch("reference_finder.openai_embedding_backend.OpenAIEncoder") as load:
            with self.assertRaisesRegex(ValueError, "provider differs"):
                embed_papers.make_encoder(args, other)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
