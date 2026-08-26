"""Multi-turn conversation: history, follow-up detection, and query rewriting.

Retrieval is stateless -- retrieve() embeds the string it is given and nothing
else. So "peki ya ikinci fıkrası?" retrieves essentially noise: the words that
carry the topic are in the *previous* turn, not in the question. Something has to
put them back before the query reaches the retriever, and that is this module.

The rewrite is a decision, not an unconditional transform. A fresh question must
go to retrieve() untouched -- rewriting one that needs no rewriting can only
distort it, dragging in vocabulary from an unrelated earlier topic and pulling
retrieval toward the wrong documents. So classification and rewriting happen in a
single call to the same local chat model: it either reports the question stands
alone, or returns the standalone version.

Both the original and the rewritten query are logged and carried on the returned
answer. When retrieval quality looks wrong later, the rewrite is the first thing
to check and the hardest thing to reconstruct after the fact.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from . import config
from .answer import Answerer, GeneratedAnswer
from .lexical import fold_diacritics, tokenize
from .llm import Message

logger = logging.getLogger(__name__)

# The sentinel the rewriter returns for a question that already stands alone.
# A word, not a JSON blob: a 4B model asked for structured output at
# temperature 0 will still occasionally wrap it in prose, and one bare token is
# the easiest thing to recognise when it does.
FRESH_SENTINEL = "YENİ"

REWRITE_SYSTEM_PROMPT = """Görevin, bir sohbetteki son soruyu tek başına anlaşılır hale getirmek.

SORUYU CEVAPLAMA. Sadece yeniden yaz.

- Son soru kendi başına anlaşılıyorsa, yani önceki turlara ihtiyaç duymadan \
cevaplanabiliyorsa: sadece YENİ yaz. Başka hiçbir şey yazma.
- Son soru önceki turlara atıf yapıyorsa (örnek: "peki ya ikinci fıkrası?", \
"bu süre ne kadar?", "istisnası var mı?"): önceki turlardan gereken konuyu \
sorunun içine yerleştir ve tek başına anlaşılır TEK bir soru yaz.

Kurallar:
- Yalnızca tek satır yaz. Açıklama, gerekçe, başlık veya tırnak ekleme.
- Çıktın ya YENİ kelimesi ya da soru işaretiyle biten tek bir soru olmalı.
- Yeni bilgi uydurma. Yalnızca önceki turlarda geçen konuyu kullan.

/no_think"""

# Trailing instruction, for the same measured reason src.answer repeats its
# format rules next to the question: this model ignores an instruction stated
# only in a system prompt. Observed failure without it -- asked to rewrite
# "peki bu faaliyet için ayrıca izin alınması gerekir mi?", the model replied
# "evet, elektrik enerjisi ithalat ve ihracat faaliyetlerine girişmek için ..."
# It answered the question instead of rewriting it, because the transcript it
# was shown is a SORU/CEVAP sequence and continuing that pattern means answering.
REWRITE_INSTRUCTION = (
    "Yukarıdaki son soruyu, önceki turlara bakmadan anlaşılacak tek bir soru "
    "hâline getir. Soruyu cevaplama. Zaten tek başına anlaşılıyorsa sadece "
    "YENİ yaz."
)

# Two rewrite examples, both anaphoric, and deliberately NO fresh-topic example.
#
# The fresh/follow-up decision is made in code by looks_like_follow_up() before
# the model is consulted at all, so by the time these examples are in play the
# question is already known to refer backwards and the only job left is the
# rewrite. Demonstrating the YENİ branch here actively hurt: asked to rewrite
# "peki bu faaliyet için ayrıca izin alınması gerekir mi?" -- plainly anaphoric
# ("bu faaliyet") -- the model answered YENİ, having pattern-matched the example
# on "looks like a grammatical sentence" rather than on whether it refers back.
#
# The second example is that exact shape: a grammatically complete question whose
# subject is a bare "bu ...". The first is the elliptical shape.
_REWRITE_EXAMPLE_HISTORY = (
    "[Önceki tur]\n"
    "Kullanıcı: Önlisans süresi ne kadardır?\n"
    "Asistan: Önlisans süresi en çok otuz altı aydır."
)

REWRITE_FEW_SHOT: list[Message] = [
    {
        "role": "user",
        "content": (
            f"{_REWRITE_EXAMPLE_HISTORY}\n\n"
            "SON SORU: peki uzatılabilir mi?\n\n"
            f"{REWRITE_INSTRUCTION}\n\n/no_think"
        ),
    },
    {"role": "assistant", "content": "Önlisans süresi uzatılabilir mi?"},
    {
        "role": "user",
        "content": (
            f"{_REWRITE_EXAMPLE_HISTORY}\n\n"
            "SON SORU: peki bu süre için ayrıca başvuru yapılması gerekir mi?\n\n"
            f"{REWRITE_INSTRUCTION}\n\n/no_think"
        ),
    },
    {
        "role": "assistant",
        "content": "Önlisans süresi için ayrıca başvuru yapılması gerekir mi?",
    },
]

# Words that make a question depend on what came before it. Turkish marks this
# with demonstratives and discourse particles, all of which are cheap to spot --
# so the fresh-vs-follow-up decision does not need a model at all, and asking a
# 4B model to make it produced errors in both directions. Splitting the work this
# way leaves the model only the rewrite, which it does well.
#
# Stored diacritic-folded and matched against folded input, reusing the same
# tokenizer the BM25 index uses, so "peki" and "pekı" or an ASCII-typed "bu sure"
# behave identically.
_FOLLOW_UP_MARKERS = frozenset(
    {
        "peki", "bu", "bunun", "buna", "bunda", "bundan", "bunlar", "bunlari",
        "su", "sunun", "o", "onun", "ona", "onda", "ondan", "onlar",
        "ayni", "soz", "konusu", "ayrica", "istisnasi", "istisna",
        "digeri", "oteki", "yukaridaki", "bahsedilen", "belirtilen",
    }
)

# A question with this few CONTENT words cannot be carrying its own topic
# ("Süresi ne kadar?" -> {süresi}), so it is treated as dependent even with no
# marker. Counted after stopword removal, which is why the floor is so low: two
# content words is already "Kimler başvurabilir?", while an ordinary standalone
# question like "Dağıtım bağlantı bedeli nasıl hesaplanır?" still has four.
_CONTENT_WORD_FLOOR = 2

# Marker detection must NOT use lexical.tokenize(): that drops stopwords, and
# "bu", "şu", "o" and "ayrıca" are all in its stopword list -- precisely the
# markers that matter most here. They are stopwords for BM25 for a good reason
# (they discriminate nothing between documents) and load-bearing for this
# decision for an equally good one (they are exactly what makes a question
# depend on its predecessor). So this splits raw words, folding diacritics for
# the same ASCII-insensitivity, and only the content-word COUNT comes from
# tokenize().
_WORD_RE = re.compile(r"[a-z0-9]+")


def _folded_words(text: str) -> list[str]:
    return _WORD_RE.findall(fold_diacritics(text).lower())

# Stale KAYNAK labels from earlier turns must not survive into a later prompt:
# the labels are re-assigned per retrieval, so turn 2's "KAYNAK 1" names a
# different chunk than turn 1's did. Leaving them in invites the model to cite a
# number that means something else now.
_LABEL_MENTION = re.compile(r"\[?\s*KAYNAK\s*\d+(?:\s*(?:,|ve|ile)\s*\d+)*\s*\]?", re.IGNORECASE)
_COLLAPSE_SPACE = re.compile(r"\s{2,}")
_LEADING_NOISE = re.compile(
    r"^(?:\s*(?:SON\s*SORU|YENİ\s*SORU|SORU|Yeniden\s*yaz[ıi]lm[ıi]ş\s*soru|Cevap)"
    r"\s*[:\-]\s*)+",
    re.IGNORECASE,
)

# An answer, not a rewrite. A rewrite that opens with "evet"/"hayır" is the model
# having answered the question -- observed in practice -- and such a string sent
# to the retriever describes a conclusion rather than asking for the provision.
_ANSWER_SHAPED = re.compile(r"^\s*(?:evet|hayır|hayir|elbette|maalesef)\b", re.IGNORECASE)


@dataclass
class Turn:
    """One completed exchange, kept for context on later turns."""

    question: str
    rewritten_query: str
    answer: str
    was_follow_up: bool


def _strip_labels(text: str) -> str:
    return _COLLAPSE_SPACE.sub(" ", _LABEL_MENTION.sub("", text)).strip()


def looks_like_follow_up(question: str) -> bool:
    """Does this question depend on a previous turn? Decided lexically, no model.

    Two signals, both deliberately cheap:
      * a demonstrative or discourse marker ("bu", "peki", "aynı", "söz konusu")
      * too few content words to be carrying a topic at all

    Being wrong in the two directions costs very different amounts, and the
    thresholds are set accordingly. A missed follow-up retrieves on a vague
    question and usually lands in NOT_FOUND -- visible, and the user can rephrase.
    A false positive sends a perfectly good standalone question through a rewrite
    that can only contaminate it with the previous topic, which is invisible and
    corrupts retrieval silently. So the markers are specific rather than broad.
    """
    words = _folded_words(question)
    if not words:
        return False
    if any(word in _FOLLOW_UP_MARKERS for word in words):
        return True
    return len(tokenize(question)) <= _CONTENT_WORD_FLOOR


def _clean_rewrite(raw: str) -> str:
    """Pull one question out of the rewriter's output, tolerating prose around it."""
    line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    line = _LEADING_NOISE.sub("", line).strip()
    return line.strip("\"'“”‘’ ").strip()


@dataclass
class Session:
    """A bounded conversation over one Answerer.

    Keeps at most config.SESSION_MAX_TURNS completed exchanges. The cap is a
    context-budget consequence, not a guess about what users want: on this
    hardware the whole prompt has to fit in ~2,800 tokens alongside the
    retrieved chunks (see config.CHAT_EFFECTIVE_CONTEXT), and history competes
    with the sources that actually ground the answer.
    """

    answerer: Answerer
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = config.SESSION_MAX_TURNS
    # Tokens any single remembered answer may contribute. History is
    # conversational context, not evidence, so unlike a retrieved chunk it may
    # be truncated -- a shortened recap still orients the model, whereas half a
    # legal provision would silently mislead it.
    history_answer_tokens: int = 120

    # -- history ------------------------------------------------------------

    def history_text(self) -> str | None:
        """Prior turns as one text block, label-stripped and length-capped.

        Text rather than alternating chat messages, because with real
        user/assistant turns this model stops honouring the /no_think switch and
        buries the answer in an unclosed <think> tag -- see build_messages().
        Returns None when there is nothing to recap, so the caller adds nothing
        to the prompt rather than an empty header.
        """
        if not self.turns:
            return None
        lines = ["[Önceki tur]"]
        for turn in self.turns:
            lines.append(f"Kullanıcı: {turn.rewritten_query}")
            lines.append(f"Asistan: {self._condense(turn.answer)}")
        return "\n".join(lines)

    def _condense(self, answer: str) -> str:
        text = _strip_labels(answer)
        counter = self.answerer.client.counter
        if counter.count(text) <= self.history_answer_tokens:
            return text
        # Truncate on a sentence boundary so the recap does not end mid-clause.
        kept: list[str] = []
        used = 0
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            cost = counter.count(sentence)
            if used + cost > self.history_answer_tokens:
                break
            kept.append(sentence)
            used += cost
        return " ".join(kept) if kept else text[:400].rstrip() + " …"

    # -- rewriting ----------------------------------------------------------

    def rewrite(self, question: str) -> tuple[str, bool]:
        """Decide fresh-vs-follow-up and return (query_for_retrieval, was_follow_up).

        Two fast paths skip the model entirely. With no prior turns there is
        nothing a rewrite could resolve. And a question carrying no dependence on
        the conversation (looks_like_follow_up) is already standalone, so
        rewriting it could only import vocabulary from an unrelated earlier topic
        -- the model is not asked, which is both faster and safer than asking and
        hoping for YENİ.
        """
        if not self.turns:
            return question, False

        if not looks_like_follow_up(question):
            logger.info("rewrite: no dependence markers, querying as typed | %r", question)
            return question, False

        # Labelled "Kullanıcı/Asistan" rather than "SORU/CEVAP": the latter reads
        # as a question-answering pattern the model completes by answering.
        transcript = "\n".join(
            f"Kullanıcı: {turn.rewritten_query}\nAsistan: {self._condense(turn.answer)}"
            for turn in self.turns
        )
        messages: list[Message] = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            *REWRITE_FEW_SHOT,
            {
                "role": "user",
                "content": (
                    f"[Önceki tur]\n{transcript}\n\n"
                    f"SON SORU: {question}\n\n"
                    f"{REWRITE_INSTRUCTION}\n\n/no_think"
                ),
            },
        ]

        try:
            raw = self.answerer.client.complete(
                messages, max_completion_tokens=config.REWRITE_MAX_COMPLETION_TOKENS
            )
        except Exception as exc:  # noqa: BLE001 - a failed rewrite must not lose the question
            logger.warning("rewrite call failed for %r: %s -- using question as typed", question, exc)
            return question, False

        candidate = _clean_rewrite(raw)
        folded = candidate.replace("I", "İ").upper().rstrip(".!? ")
        if not candidate or folded.startswith(FRESH_SENTINEL) or folded.startswith("YENI"):
            logger.info("rewrite: fresh topic, querying as typed | %r", question)
            return question, False

        # Two ways the rewriter fails without erroring, both of which would feed
        # retrieval something worse than the original: it explains what it would
        # do instead of doing it, or it pads the rewrite into a paragraph. A
        # rewrite is asked to be a single question, so anything that is not
        # shaped like one is not trusted -- falling back costs a slightly weaker
        # retrieval, while accepting commentary poisons it outright.
        if "?" not in candidate or _ANSWER_SHAPED.match(candidate):
            logger.warning(
                "rewrite was not a question, using question as typed | %r -> %r",
                question, candidate[:120],
            )
            return question, False
        if len(candidate) > max(240, len(question) * 6):
            logger.warning(
                "rewrite looked like commentary (%d chars), using question as typed | %r -> %r",
                len(candidate), question, candidate[:120],
            )
            return question, False

        logger.info("rewrite: follow-up | %r -> %r", question, candidate)
        return candidate, True

    # -- asking -------------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> GeneratedAnswer:
        """Rewrite if needed, retrieve, gate, generate, and record the turn."""
        query, was_follow_up = self.rewrite(question)

        # Logged as a pair, always, follow-up or not: reconstructing after the
        # fact which string actually reached the retriever is otherwise guesswork.
        logger.info(
            "retrieval query | original=%r | used=%r | follow_up=%s",
            question, query, was_follow_up,
        )

        answer = self.answerer.answer(query, top_k=top_k, history_text=self.history_text())
        # The question the user asked is what they should see echoed; `query` is
        # what retrieval actually ran on, so both are carried.
        answer.question = question
        answer.rewritten_query = query

        self.turns.append(
            Turn(
                question=question,
                rewritten_query=query,
                answer=answer.text,
                was_follow_up=was_follow_up,
            )
        )
        # Evict oldest first; the cap counts completed exchanges.
        if len(self.turns) > self.max_turns:
            del self.turns[: len(self.turns) - self.max_turns]
        return answer


__all__ = ["FRESH_SENTINEL", "REWRITE_SYSTEM_PROMPT", "Session", "Turn"]
