"""Step 0: validate config.TOKENS_PER_WORD against the real embedding-model tokenizer.

src/chunk.py estimates chunk size with a crude words * TOKENS_PER_WORD guess
(config.TOKENS_PER_WORD = 2.0), chosen because loading the real tokenizer per
chunk during chunking would be needless overhead. This script checks that
guess against ground truth: the actual tokenizer.json for the embedding model
Step 1 verified (qwen3-embedding-0.6b, see scripts/verify_foundry.py), applied
to every chunk the real ~587-document corpus produces.

The tokenizer file is read directly from Foundry Local's local model cache
(~/.foundry/cache/models/**/tokenizer.json) via the `tokenizers` package
(HuggingFace's Rust-backed fast tokenizer) -- no server round trip needed, and
it's fast enough (~1ms/chunk) to tokenize the whole corpus rather than only a
sample.

Usage:
    .venv\\Scripts\\python.exe scripts\\validate_tokenizer.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.chunk import chunk_document  # noqa: E402
from src.extract import _force_utf8_stdout, extract_corpus  # noqa: E402
from src.titles import extract_title  # noqa: E402

SAMPLE_TARGET = 50


def find_tokenizer() -> Path:
    """Locate the embedding model's tokenizer.json in Foundry Local's cache."""
    cache_root = Path.home() / ".foundry" / "cache" / "models"
    if not cache_root.exists():
        raise SystemExit(f"Foundry Local model cache not found at {cache_root}")

    def norm(s: str) -> str:
        return s.replace("-", "").replace("_", "").lower()

    alias_key = norm(config.EMBEDDING_MODEL)
    candidates = [
        p for p in cache_root.glob("**/tokenizer.json") if alias_key in norm(str(p))
    ]
    if not candidates:
        raise SystemExit(
            f"No tokenizer.json found matching EMBEDDING_MODEL={config.EMBEDDING_MODEL!r} "
            f"under {cache_root}. Has the model been downloaded (`foundry model load "
            f"{config.EMBEDDING_MODEL}`)?"
        )
    if len(candidates) > 1:
        print(f"warning: {len(candidates)} tokenizer.json candidates matched, using the first:")
        for c in candidates:
            print(f"    {c}")
    return candidates[0]


def context_length_for(tokenizer_path: Path) -> int | None:
    """Read the model's real context window from genai_config.json next to the tokenizer."""
    genai_config_path = tokenizer_path.parent / "genai_config.json"
    if not genai_config_path.exists():
        return None
    data = json.loads(genai_config_path.read_text(encoding="utf-8"))
    return data.get("model", {}).get("context_length")


def stratified_sample(chunks_with_counts: list[tuple], target: int) -> list[tuple]:
    """Pick chunks spanning every chunking strategy and a range of lengths within each."""
    by_strategy: dict[str, list[tuple]] = {}
    for chunk, n_real in chunks_with_counts:
        by_strategy.setdefault(chunk.strategy, []).append((chunk, n_real))

    per_bucket = max(1, target // max(1, len(by_strategy)))
    sample: list[tuple] = []
    for items in by_strategy.values():
        items.sort(key=lambda pair: pair[1])
        step = max(1, len(items) // per_bucket)
        sample.extend(items[::step][:per_bucket])
    return sample


def main() -> int:
    from tokenizers import Tokenizer

    _force_utf8_stdout()

    tokenizer_path = find_tokenizer()
    print(f"Tokenizer         : {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    context_length = context_length_for(tokenizer_path)
    print(f"Model context     : {context_length} tokens (genai_config.json)")
    print(f"Configured        : CHUNK_SIZE={config.CHUNK_SIZE}  TOKENS_PER_WORD={config.TOKENS_PER_WORD}")

    print("\nExtracting + chunking the real corpus (data/) ...")
    run = extract_corpus(Path("data"))

    all_chunks = []
    for doc in run.docs:
        info = extract_title(doc)
        all_chunks.extend(chunk_document(doc, info))
    print(f"{len(all_chunks)} chunks produced across {len(run.docs)} documents.")

    print("\nTokenizing every chunk with the real tokenizer ...")
    results = []
    for chunk in all_chunks:
        real_tokens = len(tokenizer.encode(chunk.text).ids)
        results.append((chunk, real_tokens))

    real_counts = [n for _, n in results]
    ratios = [n / len(c.text.split()) for c, n in results if c.text.split()]

    mean_ratio = statistics.mean(ratios)
    median_ratio = statistics.median(ratios)

    print("\n" + "=" * 72)
    print("TOKENIZER VALIDATION REPORT")
    print("=" * 72)
    print(f"\nChunks analyzed          : {len(results)}")
    print(f"Real tokens/word (mean)  : {mean_ratio:.3f}")
    print(f"Real tokens/word (median): {median_ratio:.3f}")
    print(f"Configured TOKENS_PER_WORD estimate: {config.TOKENS_PER_WORD}")
    print(
        f"  -> estimate is {'an OVER-estimate' if config.TOKENS_PER_WORD > mean_ratio else 'an UNDER-estimate'} "
        f"of the real ratio by {abs(config.TOKENS_PER_WORD - mean_ratio) / mean_ratio * 100:.1f}%"
    )

    over_target = [n for n in real_counts if n > config.CHUNK_SIZE]
    over_by_half = [n for n in real_counts if n > config.CHUNK_SIZE * 1.5]
    over_context = [n for n in real_counts if context_length and n > context_length]

    print(f"\nCHUNK_SIZE target        : {config.CHUNK_SIZE} tokens")
    print(f"  max real tokens in any chunk        : {max(real_counts)}")
    print(
        f"  chunks over target                  : {len(over_target)}/{len(real_counts)} "
        f"({100 * len(over_target) / len(real_counts):.1f}%)"
    )
    print(
        f"  chunks over target by >50%           : {len(over_by_half)}/{len(real_counts)} "
        f"({100 * len(over_by_half) / len(real_counts):.2f}%)"
    )
    if context_length:
        print(f"  chunks exceeding model context ({context_length}): {len(over_context)}")

    print(f"\n-- Representative sample (~{SAMPLE_TARGET} chunks across strategies/lengths) --")
    sample = stratified_sample(results, SAMPLE_TARGET)
    print(f"{'strategy':16} {'words':>6} {'estimate':>9} {'real':>6} {'real/estimate':>14}")
    for chunk, n_real in sample:
        n_words = len(chunk.text.split())
        ratio = n_real / max(1, chunk.token_estimate)
        print(f"{chunk.strategy:16} {n_words:6} {chunk.token_estimate:9} {n_real:6} {ratio:14.2f}")

    print("\n" + "=" * 72)
    if over_context:
        verdict = (
            f"RE-CHUNKING NEEDED: {len(over_context)} chunk(s) exceed the model's "
            f"real {context_length}-token context limit."
        )
    elif len(over_by_half) / len(real_counts) > 0.05:
        verdict = (
            "RE-CHUNKING RECOMMENDED: over 5% of chunks real-exceed CHUNK_SIZE by "
            "more than 50%, which is more than imprecision."
        )
    else:
        verdict = (
            "NO RE-CHUNKING NEEDED: the word-count estimate is imprecise (see ratio "
            f"above) but every chunk stays far under the model's {context_length}-token "
            "context window, with only a small share even exceeding the CHUNK_SIZE target. "
            "Leave chunking as-is."
        )
    print(verdict)
    print("=" * 72)
    print(
        f"\nFor config.py MEASURED_TOKENS_PER_WORD (recorded by hand, not auto-applied):\n"
        f"    MEASURED_TOKENS_PER_WORD = {mean_ratio:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
