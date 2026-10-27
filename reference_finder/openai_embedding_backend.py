"""OpenAI embedding adapter; no local model weights or GPU runtime required."""
from __future__ import annotations

import math
import os

from .embedding_backend import chunk_text
from .embeddings import paper_text, read_corpus

DEFAULT_OPENAI_MODEL = "text-embedding-3-large"
MODEL_DIMENSIONS = {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536}
# Standard endpoint input pricing, verified 2026-10-27. The estimate command
# accepts an override; prices and account-specific discounts can change.
PRICE_PER_MILLION = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}
API_REVISION = "openai-api-model-id-v1"
MAX_INPUT_TOKENS = 8192
MAX_REQUEST_TOKENS = 300_000
MAX_REQUEST_INPUTS = 2048


class OpenAITokenizer:
    """Adapt tiktoken to the checked, Unicode-preserving chunk splitter."""
    name = "cl100k_base"

    def __init__(self):
        try:
            import tiktoken
        except ImportError:
            raise RuntimeError("Install requirements-api-embeddings.txt for OpenAI embeddings.") from None
        self.encoding = tiktoken.get_encoding(self.name)

    def encode(self, text, add_special_tokens=True, truncation=False):
        # Literal special-token spellings in papers are ordinary input text.
        return self.encoding.encode_ordinary(text)


class OpenAIEncoder:
    provider = "openai"
    revision = API_REVISION
    tokenizer_name = "cl100k_base"
    document_prompt = ""
    query_prompt = ""
    device = "api"
    dtype = "float32"

    def __init__(self, model_id=DEFAULT_OPENAI_MODEL, dimensions=None, max_tokens=8191,
                 api_key=None, client=None, tokenizer=None, offline=False):
        if model_id not in MODEL_DIMENSIONS:
            raise ValueError("Supported OpenAI models: " + ", ".join(MODEL_DIMENSIONS))
        dimension = MODEL_DIMENSIONS[model_id] if dimensions is None else dimensions
        if type(dimension) is not int or not 1 <= dimension <= MODEL_DIMENSIONS[model_id]:
            raise ValueError("dimensions must be between 1 and %d" % MODEL_DIMENSIONS[model_id])
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_INPUT_TOKENS:
            raise ValueError("OpenAI max_tokens must be between 1 and %d" % MAX_INPUT_TOKENS)
        self.model_id, self.dimension, self.max_tokens = model_id, dimension, max_tokens
        self.batches, self.input_tokens = 0, 0
        if client is None and not offline:
            key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise ValueError("Set OPENAI_API_KEY in .env or the environment before using OpenAI embeddings.")
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("Install requirements-api-embeddings.txt for OpenAI embeddings.") from None
            client = OpenAI(api_key=key, base_url="https://api.openai.com/v1", timeout=60, max_retries=5)
        self.client = client
        self.tokenizer = tokenizer if tokenizer is not None else OpenAITokenizer()

    def document_chunks(self, text, overlap=64):
        return chunk_text(self.tokenizer, text, "", self.max_tokens, overlap)

    def _encode(self, texts, batch_size):
        if isinstance(texts, str):
            raise TypeError("Pass a sequence of texts, not a single string")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        texts = list(texts)
        counts = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Embedding inputs must be nonempty strings")
            count = len(self.tokenizer.encode(text))
            if not 1 <= count <= self.max_tokens:
                raise ValueError("Input exceeds the token limit; split it explicitly before encoding")
            counts.append(count)
        if not texts:
            return []
        if self.client is None:
            raise RuntimeError("Token-counting mode cannot make embedding API requests")
        vectors, start = [], 0
        while start < len(texts):
            end, tokens = start, 0
            while (end < len(texts) and end - start < min(batch_size, MAX_REQUEST_INPUTS)
                   and tokens + counts[end] <= MAX_REQUEST_TOKENS):
                tokens += counts[end]
                end += 1
            if end == start:
                raise ValueError("An input exceeds the request token budget")
            self.batches += 1
            try:
                response = self.client.embeddings.create(model=self.model_id, input=texts[start:end],
                    dimensions=self.dimension, encoding_format="float")
            except Exception as exc:
                # SDK exceptions can contain request text or credential details.
                # Report only a numeric status, keeping provider payloads private.
                status = getattr(exc, "status_code", None)
                detail = " (HTTP %d)" % status if type(status) is int else ""
                raise RuntimeError("OpenAI embedding request failed%s. Check API access/quota and rerun; committed batches resume." % detail) from None
            if response.model != self.model_id:
                raise ValueError("OpenAI returned a different embedding model")
            data = response.data
            if not isinstance(data, list) or len(data) != end - start:
                raise ValueError("OpenAI returned the wrong number of embeddings")
            ordered = {}
            for item in data:
                if type(item.index) is not int or not 0 <= item.index < len(data) or item.index in ordered:
                    raise ValueError("OpenAI returned invalid or duplicate embedding indices")
                values = item.embedding
                if not isinstance(values, list) or len(values) != self.dimension or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError("OpenAI returned an invalid embedding vector")
                scale = max(abs(v) for v in values)
                if not scale:
                    raise ValueError("OpenAI returned a zero embedding vector")
                scaled = [v / scale for v in values]
                norm = math.hypot(*scaled)
                ordered[item.index] = [v / norm for v in scaled]
            vectors.extend(ordered[i] for i in range(len(data)))
            usage = getattr(getattr(response, "usage", None), "total_tokens", None)
            self.input_tokens += usage if type(usage) is int and usage >= 0 else tokens
            start = end
        return vectors

    def encode_documents(self, texts, batch_size=128):
        return self._encode(texts, batch_size)

    def encode_queries(self, texts, batch_size=1):
        return self._encode(texts, batch_size)

    def synchronize(self):
        pass


def estimate_cost(source, encoder, overlap=64, limit=None, completed=None, price_per_million=None):
    """Count actual chunk tokens locally; never invoke the embeddings API."""
    price = PRICE_PER_MILLION[encoder.model_id] if price_per_million is None else price_per_million
    if not isinstance(price, (int, float)) or not math.isfinite(price) or price < 0:
        raise ValueError("price-per-million must be nonnegative and finite")
    rows = read_corpus(source)
    completed = completed or set()
    selected = [p for p in rows if p["id"] not in completed]
    pending = len(selected)
    if limit is not None:
        selected = selected[:limit]
    tokens = chunks = 0
    for paper in selected:
        parts = encoder.document_chunks(paper_text(paper), overlap)
        tokens += sum(c["token_count"] for c in parts)
        chunks += len(parts)
    return dict(provider="openai", model_id=encoder.model_id, dimension=encoder.dimension,
                corpus_papers=len(rows), pending_papers=pending, selected_papers=len(selected),
                chunks=chunks, input_tokens=tokens, price_per_million_usd=price,
                estimated_cost_usd=round(tokens * price / 1_000_000, 6), api_requests=0,
                note="Local token estimate at the supplied standard endpoint price; excludes retries and future price changes.")
