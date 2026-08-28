"""Chat client for Foundry Local, plus real-tokenizer counting for context budgeting.

Deliberately knows nothing about RAG: this module is transport and token
arithmetic only. Prompt construction, source labelling and citation assembly
live in src/answer.py, so the rules about what may be shown to the model are
stated in exactly one place rather than spread across the call stack.

Three things here are not obvious and are the reason the module exists:

1. Discovery is shared with src/embed.py. The daemon's port is dynamic and a
   model must be resident in VRAM before the OpenAI endpoint will serve it --
   already solved there, so it is imported rather than reimplemented.

2. Qwen3 emits a <think>...</think> block before its answer. With `/no_think`
   in the system prompt the block comes back empty, but it always comes back,
   so it is always stripped and always paid for out of the completion budget.

3. Token counting uses qwen3-4b's REAL tokenizer, read from Foundry Local's
   model cache. Context budgeting that guessed token counts would be pointless
   on this machine, where the usable window is a few thousand tokens (see
   config.CHAT_EFFECTIVE_CONTEXT) and overshooting is a hard CUDA OOM rather
   than a graceful refusal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import config
from .embed import FoundryUnavailable, discover_endpoint, ensure_loaded, resolve_model_id

Message = dict[str, str]


class ContextExhausted(Exception):
    """The prompt was too large for the server to serve.

    Raised for both honest context-length refusals and the CUDA OOM this
    machine actually produces (see config.CHAT_EFFECTIVE_CONTEXT). Callers are
    expected to catch this and surface a message to the user rather than let it
    escape -- src/answer.py does exactly that.
    """


class GenerationFailed(Exception):
    """The chat model could not be reached or returned nothing usable."""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Substrings that identify a server-side failure caused by prompt size rather
# than by anything wrong with the request. The CUDA variant is what this GPU
# raises; the others are what a larger machine would raise instead.
_CONTEXT_ERROR_MARKERS = (
    "out of memory",
    "cudamallocarray",
    "context_length",
    "context length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
)


def strip_thinking(text: str) -> str:
    """Remove Qwen3's <think> reasoning block, keeping only the answer.

    Three cases, because all three occur with this model:

    1. A properly closed block -- removed.
    2. An UNCLOSED tag with nothing but whitespace after it before real text.
       This is a stray marker, not reasoning: the model opened the block, wrote
       nothing in it, and forgot the closing tag, so everything after it is the
       answer. Observed with multi-turn prompts, where the raw completion was
       "<think>\\n\\n**Cevap:** ... (KAYNAK 1)" -- discarding that would throw
       away a complete, correctly cited answer.
    3. An unclosed tag with actual reasoning after it -- generation was cut off
       mid-thought, and only the text before the tag is safe to show. Returning
       raw reasoning to a user is worse than returning nothing.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think>" not in cleaned:
        return cleaned.strip()

    before, after = cleaned.split("<think>", 1)
    # Case 2: the block itself is empty, so what follows is the answer, not
    # reasoning. Anything on the far side of a blank line is past the marker.
    paragraph_break = re.match(r"\s*\n\s*\n", after)
    if not before.strip() and paragraph_break:
        return after[paragraph_break.end():].strip()
    return before.strip()


def find_tokenizer(alias: str) -> Path | None:
    """Locate a model's tokenizer.json in Foundry Local's cache, or None.

    Same cache layout scripts/validate_tokenizer.py relies on. Returns None
    rather than raising: a missing tokenizer degrades budgeting to an estimate,
    which TokenCounter reports honestly instead of crashing the pipeline.
    """
    cache_root = Path.home() / ".foundry" / "cache" / "models"
    if not cache_root.exists():
        return None

    def norm(s: str) -> str:
        return s.replace("-", "").replace("_", "").lower()

    key = norm(alias)
    candidates = sorted(
        p for p in cache_root.glob("**/tokenizer.json") if key in norm(str(p))
    )
    return candidates[0] if candidates else None


@dataclass
class TokenCounter:
    """Counts tokens for context budgeting, with the real tokenizer when available.

    `exact` records which mode is in use. It is surfaced in the CLI rather than
    hidden, because a budget computed from an estimate is only as trustworthy as
    the estimate -- and on this hardware the cost of under-counting is a crashed
    request, not a slightly long prompt.
    """

    tokenizer: Any | None = None
    exact: bool = False
    source: str = "estimate (words x config.MEASURED_TOKENS_PER_WORD)"

    @classmethod
    def load(cls, alias: str | None = None) -> "TokenCounter":
        """Load the model's real tokenizer if found; else fall back to the estimate."""
        alias = alias or config.CHAT_MODEL
        path = find_tokenizer(alias)
        if path is None:
            return cls()
        try:
            from tokenizers import Tokenizer
        except ImportError:
            return cls()
        try:
            return cls(tokenizer=Tokenizer.from_file(str(path)), exact=True, source=str(path))
        except Exception:  # noqa: BLE001 - a corrupt tokenizer must not be fatal
            return cls()

    def count(self, text: str) -> int:
        """Token count for one string, exact if the real tokenizer loaded, else estimated."""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        # Deliberately rounds up: over-counting shrinks the prompt, under-counting
        # crashes the server.
        return int(len(text.split()) * config.MEASURED_TOKENS_PER_WORD) + 1

    def count_messages(self, messages: list[Message]) -> int:
        """Token cost of a whole message list, including chat-template framing."""
        return sum(
            self.count(m.get("content", "")) + config.TOKENS_PER_MESSAGE for m in messages
        )


def _is_context_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS)


@dataclass
class ChatClient:
    """A loaded Foundry Local chat model, reused across calls.

    Connect once and keep it: `connect()` shells out to the Foundry CLI twice
    and loads 2.6 GB into VRAM, which is not something to repeat per question.
    """

    client: OpenAI
    model_id: str
    counter: TokenCounter = field(default_factory=TokenCounter)
    calls: int = 0

    @classmethod
    def connect(cls, alias: str | None = None) -> "ChatClient":
        """Discover the Foundry Local daemon, load the model, and load its tokenizer."""
        alias = alias or config.CHAT_MODEL
        endpoint = discover_endpoint()
        client = OpenAI(base_url=endpoint, api_key="not-needed")
        ensure_loaded(alias)
        model_id = resolve_model_id(client, alias)
        return cls(client=client, model_id=model_id, counter=TokenCounter.load(alias))

    @property
    def context_budget(self) -> int:
        """Prompt tokens available: the usable window less the answer and the margin.

        Uses config.CHAT_EFFECTIVE_CONTEXT, the measured VRAM ceiling, not the
        model's declared CHAT_CONTEXT_WINDOW -- budgeting against the declared
        window OOMs this GPU.
        """
        return (
            config.CHAT_EFFECTIVE_CONTEXT
            - config.CHAT_MAX_COMPLETION_TOKENS
            - config.CONTEXT_SAFETY_MARGIN
        )

    def count_messages(self, messages: list[Message]) -> int:
        """Token cost of a whole message list, via this client's counter."""
        return self.counter.count_messages(messages)

    def complete(
        self,
        messages: list[Message],
        max_completion_tokens: int | None = None,
    ) -> str:
        """One deterministic completion, with the thinking block stripped.

        Raises ContextExhausted if the server rejects the prompt on size, and
        GenerationFailed for anything else. Never retried: temperature is 0, so
        a retry is the identical request and would fail identically.
        """
        limit = (
            config.CHAT_MAX_COMPLETION_TOKENS
            if max_completion_tokens is None
            else max_completion_tokens
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_completion_tokens=limit,
                temperature=config.CHAT_TEMPERATURE,
                frequency_penalty=config.CHAT_FREQUENCY_PENALTY,
            )
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised narrowly
            if _is_context_error(exc):
                raise ContextExhausted(
                    f"the chat server refused a prompt of ~"
                    f"{self.count_messages(messages)} tokens "
                    f"(budget {self.context_budget}): {exc}"
                ) from exc
            raise GenerationFailed(f"chat completion failed: {exc}") from exc

        self.calls += 1
        choices = completion.choices or []
        raw = (choices[0].message.content or "") if choices else ""
        return strip_thinking(raw)


__all__ = [
    "ChatClient",
    "ContextExhausted",
    "FoundryUnavailable",
    "GenerationFailed",
    "Message",
    "TokenCounter",
    "find_tokenizer",
    "strip_thinking",
]
