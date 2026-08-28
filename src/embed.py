"""Thin client for the Foundry Local embedding endpoint.

Foundry Local exposes an OpenAI-compatible HTTP API, but the daemon's port is
assigned dynamically and a model must be explicitly loaded into VRAM before it
will serve requests -- both handled here, following the same discovery
sequence proven in scripts/verify_foundry.py. Everything else in this module
exists to make embedding ~27k chunks safe to run unattended: batched calls,
retries on transient failures, and a hard assertion that every vector returned
actually has config.EMBEDDING_DIM elements, since a silently wrong dimension
would corrupt the vector store in a way nothing downstream would catch.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass

from openai import OpenAI

from . import config


class FoundryUnavailable(Exception):
    """The Foundry Local daemon, CLI, or configured model could not be reached."""


class EmbeddingDimensionMismatch(Exception):
    """A returned vector's length does not match config.EMBEDDING_DIM.

    Never retried: a dimension mismatch means the wrong model is loaded or
    config.EMBEDDING_DIM is stale, not a transient server hiccup, and retrying
    would just fail identically while masking the real problem.
    """


def discover_endpoint() -> str:
    """Ask the Foundry CLI where its daemon is listening -- the port is dynamic."""
    exe = shutil.which("foundry")
    if exe is None:
        raise FoundryUnavailable("`foundry` CLI not found on PATH. Is Foundry Local installed?")

    proc = subprocess.run(
        [exe, "server", "status", "-o", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if proc.returncode != 0:
        raise FoundryUnavailable(f"`foundry server status` failed: {proc.stderr.strip()}")

    try:
        status = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FoundryUnavailable(f"could not parse `foundry server status` output: {exc}") from exc

    urls = status.get("webUrls") or []
    if not urls:
        raise FoundryUnavailable(f"Foundry daemon reported no web URLs: {status}")
    return urls[0].rstrip("/") + "/v1"


def ensure_loaded(alias: str) -> None:
    """Models must be resident in VRAM before the OpenAI endpoint will serve them."""
    exe = shutil.which("foundry")
    if exe is None:
        raise FoundryUnavailable("`foundry` CLI not found on PATH.")
    proc = subprocess.run(
        [exe, "model", "load", alias],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if proc.returncode != 0:
        raise FoundryUnavailable(f"Failed to load {alias!r}: {proc.stderr.strip()}")


def resolve_model_id(client: OpenAI, alias: str) -> str:
    """Map a short alias to the concrete hardware-specific id (qwen3-embedding-0.6b -> ...-cuda-gpu)."""
    ids = [m.id for m in client.models.list().data]
    for model_id in ids:
        if model_id == alias or model_id.startswith(alias + "-"):
            return model_id
    raise FoundryUnavailable(f"No model matching alias {alias!r}. Available: {ids}")


@dataclass
class EmbeddingClient:
    """A loaded Foundry Local embedding model, reused across calls."""

    client: OpenAI
    model_id: str
    dimension: int

    @classmethod
    def connect(cls, alias: str | None = None) -> "EmbeddingClient":
        """Discover the daemon, load the model, and resolve its concrete id.

        `dimension` is always config.EMBEDDING_DIM -- the one place that value
        is allowed to originate; embed_batch() then asserts every response
        against it, so a model swap that changes dimension is caught, not
        silently accepted.
        """
        alias = alias or config.EMBEDDING_MODEL
        endpoint = discover_endpoint()
        client = OpenAI(base_url=endpoint, api_key="not-needed")
        ensure_loaded(alias)
        model_id = resolve_model_id(client, alias)
        return cls(client=client, model_id=model_id, dimension=config.EMBEDDING_DIM)

    def embed_batch(
        self, texts: list[str], retries: int | None = None
    ) -> list[list[float]]:
        """Embed one batch, in order. Retries transient failures; never retries a dimension mismatch."""
        if not texts:
            return []
        max_retries = retries if retries is not None else config.EMBED_MAX_RETRIES
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(model=self.model_id, input=texts)
            except Exception as exc:  # noqa: BLE001 - network/server errors are all transient here
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

            vectors = [d.embedding for d in response.data]
            for i, vector in enumerate(vectors):
                if len(vector) != self.dimension:
                    raise EmbeddingDimensionMismatch(
                        f"model {self.model_id!r} returned a {len(vector)}-dim vector for "
                        f"input {i}, but config.EMBEDDING_DIM is {self.dimension}. "
                        "The loaded model does not match config -- fix config or reload "
                        "the intended model rather than proceeding with mismatched vectors."
                    )
            return vectors

        raise FoundryUnavailable(
            f"embedding batch of {len(texts)} failed after {max_retries} attempts: {last_exc}"
        ) from last_exc

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        on_batch: "callable[[int, int], None] | None" = None,
    ) -> list[list[float]]:
        """Embed many texts in batches, preserving order. `on_batch(done, total)` fires after each batch."""
        size = batch_size if batch_size is not None else config.EMBED_BATCH_SIZE
        out: list[list[float]] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            out.extend(self.embed_batch(batch))
            if on_batch:
                on_batch(len(out), len(texts))
        return out
