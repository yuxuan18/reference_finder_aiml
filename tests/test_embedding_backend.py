"""Check lossless preparation and model safeguards without ML dependencies."""
import contextlib
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from reference_finder.embedding_backend import SentenceEncoder, chunk_text, resolve_revision


SHA = "abcdef0123456789" * 2 + "abcdef01"


class CharacterTokenizer:
    """One token per Unicode character, plus two special tokens."""
    model_max_length = 262144

    def encode(self, text, add_special_tokens=True, truncation=False):
        if truncation:
            raise AssertionError("Preparation must never request truncation")
        return list(text) + (["BOS", "EOS"] if add_special_tokens else [])

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ByteTokenizerWithoutOffsets(CharacterTokenizer):
    """Models multi-token Unicode characters and a slow HF tokenizer."""
    def encode(self, text, add_special_tokens=True, truncation=False):
        if truncation:
            raise AssertionError("Preparation must never request truncation")
        return list(text.encode("utf-8")) + ([256, 257] if add_special_tokens else [])

    def __call__(self, text, **kwargs):
        raise NotImplementedError("This tokenizer does not support offsets")


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def assert_budgets(self, chunks, tokenizer, prompt, budget):
        for chunk in chunks:
            actual = len(tokenizer.encode(prompt + chunk["text"]))
            self.assertEqual(chunk["token_count"], actual)
            self.assertLessEqual(actual, budget)
            self.assertTrue(chunk["text"])

    def test_short_abstract_is_unchanged(self):
        text = "  A proof of privacy.\n"
        chunks = chunk_text(self.tokenizer, text, "passage: ", 64, 3)
        self.assertEqual(chunks, [{"text": text, "token_count": len(text) + 11}])

    def test_long_original_text_is_preserved_including_whitespace_unicode_and_tail(self):
        text = "  Title: café 🙂\n" + "αβ graph learning\t" * 80 + "\nTAIL 漢字"
        chunks = chunk_text(self.tokenizer, text, "passage: ", 100, 0)
        self.assertGreater(len(chunks), 10)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        self.assertTrue(chunks[-1]["text"].endswith("TAIL 漢字"))
        self.assert_budgets(chunks, self.tokenizer, "passage: ", 100)

    def test_overlap_preserves_all_content_without_extra_prefix(self):
        text = "".join(chr(0x400 + i) for i in range(113))
        chunks = chunk_text(self.tokenizer, text, "doc: ", 30, 5)
        reconstructed = chunks[0]["text"]
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous["text"][-5:], current["text"][:5])
            reconstructed += current["text"][5:]
        self.assertEqual(reconstructed, text)
        self.assert_budgets(chunks, self.tokenizer, "doc: ", 30)

    def test_final_short_tail_is_not_discarded(self):
        chunks = chunk_text(self.tokenizer, "a" * 16 + "END", "p:", 12, 0)
        self.assertEqual([chunk["text"] for chunk in chunks], ["a" * 8, "a" * 8, "END"])
        self.assertEqual(chunks[-1]["token_count"], 7)

    def test_prompt_and_special_tokens_reduce_available_space(self):
        chunks = chunk_text(self.tokenizer, "123456789", "query:", 12, 0)
        self.assertEqual([chunk["text"] for chunk in chunks], ["1234", "5678", "9"])
        self.assert_budgets(chunks, self.tokenizer, "query:", 12)

    def test_no_offsets_uses_original_unicode_not_decoded_partial_tokens(self):
        tokenizer = ByteTokenizerWithoutOffsets()
        text = "🙂é漢字\n" * 17 + "🙂END"
        chunks = chunk_text(tokenizer, text, "p:", 17, 0)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        self.assert_budgets(chunks, tokenizer, "p:", 17)

    def test_large_offset_span_falls_back_to_checked_character_boundaries(self):
        tokenizer = CharacterTokenizer()
        with patch.object(CharacterTokenizer, "__call__", return_value={"offset_mapping": [(0, 100)]}):
            chunks = chunk_text(tokenizer, "x" * 100, "", 20, 0)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), "x" * 100)
        self.assert_budgets(chunks, tokenizer, "", 20)

    def test_invalid_budgets_and_overlaps_are_rejected(self):
        for budget in (0, -1, True, 10.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                chunk_text(self.tokenizer, "text", "", budget, 0)
        for overlap in (-1, True, 2.5, 6, 100):
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                chunk_text(self.tokenizer, "text", "p:", 10, overlap)
        with self.assertRaisesRegex(ValueError, "no room"):
            chunk_text(self.tokenizer, "text", "p:", 4, 0)
        with self.assertRaisesRegex(ValueError, "one text character"):
            chunk_text(ByteTokenizerWithoutOffsets(), "🙂", "", 3, 0)

    def test_empty_text_and_type_validation(self):
        self.assertEqual(chunk_text(self.tokenizer, "", "", 12, 0), [])
        for text, prompt in ((None, ""), ("text", None)):
            with self.subTest(text=text, prompt=prompt), self.assertRaises(TypeError):
                chunk_text(self.tokenizer, text, prompt, 12, 0)


class RevisionTests(unittest.TestCase):
    def test_pinned_sha_is_used_without_hub_dependency_or_network(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            self.assertEqual(resolve_revision("some/model", SHA.upper()), SHA)

    def test_branch_is_resolved_to_immutable_commit(self):
        api = Mock()
        api.model_info.return_value.sha = SHA.upper()
        module = types.ModuleType("huggingface_hub")
        module.HfApi = Mock(return_value=api)
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            self.assertEqual(resolve_revision("some/model", "release-v1"), SHA)
        api.model_info.assert_called_once_with("some/model", revision="release-v1")

    def test_noncommit_hub_response_cannot_be_saved_as_revision(self):
        for value in (None, "main", "12345", "z" * 40):
            module = types.ModuleType("huggingface_hub")
            module.HfApi = Mock()
            module.HfApi.return_value.model_info.return_value.sha = value
            with self.subTest(value=value), patch.dict(sys.modules, {"huggingface_hub": module}):
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    resolve_revision("some/model")


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock()
        self.model.max_seq_length = 32768
        self.model.tokenizer = CharacterTokenizer()
        self.model.prompts = {"query": "query: ", "passage": "passage: "}
        self.model.get_sentence_embedding_dimension.return_value = 2048
        self.config = types.SimpleNamespace(max_position_embeddings=262144, use_cache=True)
        self.model.__getitem__.return_value = types.SimpleNamespace(
            auto_model=types.SimpleNamespace(config=self.config)
        )
        self.torch = types.ModuleType("torch")
        self.torch.float32 = "mock-float32"
        self.torch.float16 = "mock-float16"
        self.torch.bfloat16 = "mock-bfloat16"
        self.torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=Mock(return_value=False))
        )
        self.torch.cuda = types.SimpleNamespace(
            is_available=Mock(return_value=False),
            is_bf16_supported=Mock(return_value=True),
            device=Mock(side_effect=lambda device: contextlib.nullcontext()),
            current_device=Mock(return_value=0),
            device_count=Mock(return_value=1),
            OutOfMemoryError=type("FakeCUDAOutOfMemoryError", (RuntimeError,), {}),
            empty_cache=Mock(),
        )
        self.torch.inference_mode = contextlib.nullcontext
        self.np = types.ModuleType("numpy")
        self.st = types.ModuleType("sentence_transformers")
        self.st.SentenceTransformer = Mock(return_value=self.model)
        patcher = patch.dict(sys.modules, {
            "torch": self.torch, "numpy": self.np, "sentence_transformers": self.st,
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_encoder(self, **kwargs):
        return SentenceEncoder("some/model", revision=SHA, **kwargs)

    def test_embedding_limit_wins_over_larger_underlying_model_context(self):
        with self.assertRaisesRegex(ValueError, "32768"):
            self.make_encoder(max_tokens=32769)
        self.assertEqual(self.model.max_seq_length, 32768)

    def test_exact_embedding_limit_is_allowed_and_revision_and_prompts_saved(self):
        encoder = self.make_encoder(max_tokens=32768)
        self.assertEqual(encoder.revision, SHA)
        self.assertEqual(encoder.dimension, 2048)
        self.assertEqual(encoder.device, "cpu")
        self.assertEqual(encoder.dtype, "float32")
        self.assertEqual(encoder.document_prompt, "passage: ")
        self.assertEqual(encoder.query_prompt, "query: ")
        self.st.SentenceTransformer.assert_called_once_with(
            "some/model", revision=SHA, device="cpu", trust_remote_code=False,
            model_kwargs={
                "dtype": "mock-float32", "attn_implementation": "sdpa",
                "device_map": {"": "cpu"},
            },
            tokenizer_kwargs={"padding_side": "left"},
        )

    def test_auto_device_prefers_mps_and_half_precision_on_mac(self):
        self.torch.backends.mps.is_available.return_value = True
        encoder = self.make_encoder()
        self.assertEqual((encoder.device, encoder.dtype), ("mps", "float16"))
        kwargs = self.st.SentenceTransformer.call_args.kwargs
        self.assertEqual(kwargs["device"], "mps")
        self.assertEqual(kwargs["model_kwargs"]["dtype"], "mock-float16")
        self.assertEqual(kwargs["model_kwargs"]["device_map"], {"": "mps"})

    def test_auto_device_prefers_cuda_over_mps_and_uses_bfloat16(self):
        self.torch.cuda.is_available.return_value = True
        self.torch.backends.mps.is_available.return_value = True
        encoder = self.make_encoder()
        self.assertEqual((encoder.device, encoder.dtype), ("cuda", "bfloat16"))
        kwargs = self.st.SentenceTransformer.call_args.kwargs
        self.assertEqual(kwargs["model_kwargs"]["dtype"], "mock-bfloat16")
        self.assertEqual(kwargs["model_kwargs"]["device_map"], {"": "cuda"})
        self.torch.cuda.device.assert_called_once_with("cuda")

    def test_generation_cache_is_disabled_for_embedding_forward_passes(self):
        self.assertTrue(self.config.use_cache)
        self.make_encoder()
        self.assertFalse(self.config.use_cache)

    def test_models_without_a_generation_cache_setting_are_supported(self):
        del self.config.use_cache
        self.make_encoder()
        self.assertFalse(hasattr(self.config, "use_cache"))

    def test_cuda_without_bfloat16_support_defaults_to_float16(self):
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.is_bf16_supported.return_value = False
        encoder = self.make_encoder()
        self.assertEqual(encoder.dtype, "float16")

    def test_explicit_cpu_remains_available_even_when_mps_is_available(self):
        self.torch.backends.mps.is_available.return_value = True
        with self.assertLogs("reference_finder.embedding_backend", level="WARNING") as logs:
            encoder = SentenceEncoder("Qwen/Qwen3-Embedding-4B", revision=SHA, device="cpu")
        self.assertEqual((encoder.device, encoder.dtype), ("cpu", "float32"))
        self.assertIn("16 GB", " ".join(logs.output))

    def test_explicit_device_or_precision_is_not_silently_changed(self):
        with self.assertRaisesRegex(ValueError, "MPS was requested"):
            self.make_encoder(device="mps")
        self.torch.backends.mps.is_available.return_value = True
        encoder = self.make_encoder(device="mps", dtype="float32")
        self.assertEqual((encoder.device, encoder.dtype), ("mps", "float32"))

    def test_lower_tokenizer_and_underlying_limits_are_also_respected(self):
        self.model.tokenizer.model_max_length = 128
        with self.assertRaisesRegex(ValueError, "128"):
            self.make_encoder(max_tokens=129)
        self.model.tokenizer.model_max_length = 10**30  # HF unspecified sentinel
        self.config.max_position_embeddings = 256
        with self.assertRaisesRegex(ValueError, "256"):
            self.make_encoder(max_tokens=257)

    def test_queries_and_documents_raise_before_silent_truncation(self):
        encoder = self.make_encoder(max_tokens=32)
        # The raw texts fit; the actual prompted encodings do not.
        with self.assertRaisesRegex(ValueError, "Query 0 uses 33 tokens"):
            encoder.encode_queries(["x" * 24])
        with self.assertRaisesRegex(ValueError, "Document 0 uses 33 tokens"):
            encoder.encode_documents(["x" * 22])
        self.model.encode_query.assert_not_called()
        self.model.encode_document.assert_not_called()

    def test_document_prompt_is_budgeted_once_during_chunking(self):
        encoder = self.make_encoder(max_tokens=32)
        text = "x" * 60
        chunks = encoder.document_chunks(text, overlap=0)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        self.assertEqual([len(chunk["text"]) for chunk in chunks], [21, 21, 18])

    def test_encoding_passes_raw_text_and_explicit_prompt_once(self):
        encoder = self.make_encoder(max_tokens=32)
        for name, method, prompt in (
            ("query", encoder.encode_queries, "query: "),
            ("document", encoder.encode_documents, "passage: "),
        ):
            mock = getattr(self.model, "encode_" + name)
            mock.side_effect = RuntimeError("stop before tensor conversion")
            with self.subTest(kind=name), self.assertRaisesRegex(RuntimeError, "stop before"):
                method(["an abstract"])
            args, kwargs = mock.call_args
            self.assertEqual(args, (["an abstract"],))
            self.assertEqual(kwargs["prompt"], prompt)
            self.assertTrue(kwargs["normalize_embeddings"])
            self.assertTrue(kwargs["convert_to_tensor"])

    def test_qwen_instruction_is_applied_only_to_queries(self):
        instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        self.model.prompts = {"query": instruction, "document": ""}
        self.model.default_prompt_name = "query"
        encoder = self.make_encoder(max_tokens=256)
        self.assertEqual(encoder.query_prompt, instruction)
        self.assertEqual(encoder.document_prompt, "")
        text = "Zero knowledge proofs for social network graphs"
        chunks = encoder.document_chunks(text, overlap=0)
        self.assertEqual(chunks, [{"text": text, "token_count": len(text) + 2}])
        for name, method, expected_prompt in (
            ("query", encoder.encode_queries, instruction),
            ("document", encoder.encode_documents, ""),
        ):
            mock = getattr(self.model, "encode_" + name)
            mock.side_effect = RuntimeError("stop before tensor conversion")
            with self.subTest(kind=name), self.assertRaisesRegex(RuntimeError, "stop before"):
                method([text])
            self.assertEqual(mock.call_args.args, ([text],))
            self.assertEqual(mock.call_args.kwargs["prompt"], expected_prompt)

    def test_explicit_empty_document_prompt_is_not_replaced_by_passage_prompt(self):
        self.model.prompts = {"document": "", "passage": "do not use this"}
        encoder = self.make_encoder()
        self.assertEqual(encoder.document_prompt, "")

    def test_cuda_oom_retries_are_bounded_and_reduce_batch_size(self):
        encoder = self.make_encoder(max_tokens=32)
        encoder.device = "cuda"
        error = self.torch.cuda.OutOfMemoryError
        self.model.encode_document.side_effect = error("simulated memory exhaustion")
        with self.assertLogs("reference_finder.embedding_backend", level="WARNING"):
            with self.assertRaises(error):
                encoder.encode_documents(["abstract"] * 8, batch_size=8)
        self.assertEqual(
            [call.kwargs["batch_size"] for call in self.model.encode_document.call_args_list],
            [8, 4, 2, 1],
        )
        self.assertEqual(self.torch.cuda.empty_cache.call_count, 3)


if __name__ == "__main__":
    unittest.main()
