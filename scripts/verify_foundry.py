"""Standalone smoke test for Foundry Local: one Turkish chat completion, one embedding.

Not part of src/ — this only proves the models load and respond, and records the
exact embedding dimension the vector store schema will need.

Usage:
    .venv\\Scripts\\python.exe scripts\\verify_foundry.py
"""

import json
import re
import shutil
import subprocess
import sys

from openai import OpenAI

CHAT_ALIAS = "qwen3-4b"
EMBED_ALIAS = "qwen3-embedding-0.6b"

TURKISH_PROMPT = (
    "EPDK'nin elektrik piyasasındaki temel görevi nedir? "
    "İki cümleyle, resmi bir dille açıklayın."
)
TURKISH_EMBED_INPUT = (
    "Elektrik piyasasında dağıtım lisansı sahibi tüzel kişilerin "
    "yükümlülükleri EPDK tarafından düzenlenir."
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Qwen3 emits <think>...</think> reasoning before its answer; keep only the answer."""
    return THINK_BLOCK.sub("", text).strip()


def discover_endpoint() -> str:
    """Ask the Foundry CLI where its daemon is listening — the port is assigned dynamically."""
    exe = shutil.which("foundry")
    if exe is None:
        raise RuntimeError("`foundry` CLI not found on PATH. Is Foundry Local installed?")

    proc = subprocess.run(
        [exe, "server", "status", "-o", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"`foundry server status` failed: {proc.stderr.strip()}")

    status = json.loads(proc.stdout)
    urls = status.get("webUrls") or []
    if not urls:
        raise RuntimeError(f"Foundry daemon reported no web URLs: {status}")
    return urls[0].rstrip("/") + "/v1"


def ensure_loaded(alias: str) -> None:
    """Models must be resident in VRAM before the OpenAI endpoint will serve them."""
    exe = shutil.which("foundry")
    proc = subprocess.run(
        [exe, "model", "load", alias],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to load {alias!r}: {proc.stderr.strip()}")


def resolve_model_id(client: OpenAI, alias: str) -> str:
    """Map a short alias to the concrete hardware-specific id (e.g. qwen3-4b -> qwen3-4b-cuda-gpu)."""
    ids = [m.id for m in client.models.list().data]
    for model_id in ids:
        if model_id == alias or model_id.startswith(alias + "-"):
            return model_id
    raise RuntimeError(f"No model matching alias {alias!r}. Available: {ids}")


def main() -> int:
    print("=" * 70)
    print("FOUNDRY LOCAL VERIFICATION")
    print("=" * 70)

    endpoint = discover_endpoint()
    print(f"Endpoint : {endpoint}")
    client = OpenAI(base_url=endpoint, api_key="not-needed")

    # ---------- 1. Chat ----------
    ensure_loaded(CHAT_ALIAS)
    chat_id = resolve_model_id(client, CHAT_ALIAS)
    print(f"Chat model : {chat_id}")
    print("-" * 70)
    print(f"Prompt: {TURKISH_PROMPT}\n")

    completion = client.chat.completions.create(
        model=chat_id,
        messages=[{"role": "user", "content": TURKISH_PROMPT}],
        max_completion_tokens=512,
        temperature=0.7,
    )
    print(f"Response:\n{strip_thinking(completion.choices[0].message.content or '')}\n")

    # ---------- 2. Embedding ----------
    ensure_loaded(EMBED_ALIAS)
    embed_id = resolve_model_id(client, EMBED_ALIAS)
    print("-" * 70)
    print(f"Embedding model : {embed_id}")
    print(f"Input: {TURKISH_EMBED_INPUT}\n")

    vector = client.embeddings.create(model=embed_id, input=TURKISH_EMBED_INPUT).data[0].embedding

    print(f">>> EMBEDDING DIMENSION: {len(vector)}")
    print(f"First 8 values: {[round(v, 6) for v in vector[:8]]}")

    print("=" * 70)
    print(f"OK - chat and embedding both responded. Dimension = {len(vector)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
