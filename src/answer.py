"""Generation on top of fused retrieval: gated, source-labelled, cited in code.

The design rule this module exists to enforce: **the model never sees a citation
and never writes one.**

Retrieved chunks reach it as KAYNAK 1..N and nothing else. No document title, no
article reference, no page number, no Resmî Gazete reference, no date -- and not
just as omitted metadata fields, but scrubbed out of the chunk *body* too, since
mevzuat text is full of "MADDE 9 - (Değişik:RG-24/2/2017-29989)" and a model that
can read that can repeat it. The model answers and references labels; code maps
the labels it used back to real metadata and builds the citation list.

That inversion is the whole point. A citation assembled in code from the row the
chunk came from cannot be wrong about which provision it names. A citation
written by a 4B model reading legal text can be, and would be wrong in the most
expensive way available -- a confident, well-formatted, fabricated article
reference in a regulatory answer. tests/test_answer.py asserts the invariant
directly against the bytes sent to the server rather than trusting this
docstring.

A label the model references that was never supplied is dropped and counted as a
hallucinated reference, not silently ignored: it is the signal that the prompt
contract is being violated, and it is worth seeing.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from . import config
from .llm import ChatClient, ContextExhausted, Message
from .retrieval import GateDecision, RetrievalResult, Retriever, gate

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

# Rule 3 is the load-bearing one and is stated twice on purpose (as a
# prohibition, then as the reason): a 4B model asked to answer legal questions
# will otherwise reach for the citation format it saw in training. Rule 5
# explains the redaction marker so the model treats it as absent information
# rather than as something to reconstruct.
SYSTEM_PROMPT = """Sen Türkiye elektrik piyasası mevzuatı üzerine çalışan bir asistansın.

KURALLAR:
1. Yalnızca aşağıda KAYNAK olarak verilen metinlere dayanarak cevap ver. Kendi \
hafızandaki bilgilere, tahminlerine veya genel kültürüne dayanarak cevap verme.
2. Cevabını Türkçe yaz. Sorunun üslubuna uy: soru resmî bir dille sorulduysa \
resmî, günlük bir dille sorulduysa sade bir Türkçe kullan.
3. Mevzuat adı, kanun/yönetmelik/tebliğ ismi, madde numarası, fıkra numarası \
dışında bir numara, Resmî Gazete atfı, sayı veya tarih YAZMA. Bu bilgiler sana \
verilmemiştir; yazarsan uydurmuş olursun.
4. Kullandığın her bilginin hemen ardından kaynağını (KAYNAK 1) biçiminde \
belirt. Sadece sana verilen etiket numaralarını kullan, başka numara uydurma.
5. Metinlerde […] işaretini görürsen, orada kasıtlı olarak çıkarılmış bir bilgi \
vardır. Ne olduğunu tahmin etme ve o kısma atıf yapma.
6. Verilen kaynaklar soruyu cevaplamaya yetmiyorsa bunu açıkça söyle. Eksik \
bilgiyi tamamlamaya çalışma.

/no_think"""

# Repeated at the END of the user turn, immediately after the question.
#
# Not redundancy for its own sake: with the formatting rules stated only in the
# system prompt -- thousands of tokens of legal text earlier in the context --
# this model ignored them completely and answered in markdown headings and
# numbered lists, citing nothing. Moving the same constraints next to the
# question cut a 900-token structured essay to a 110-token paragraph that cited
# its source, on the same query and the same retrieved chunks. Recency wins over
# position in the system prompt for a 4B model, so the constraints live in both
# places and this is the copy that does the work.
ANSWER_INSTRUCTION = (
    "Yukarıdaki kaynaklara dayanarak tek bir düz paragraf yaz. En fazla 4 cümle. "
    "Başlık, madde imi, numaralı liste, kalın yazı ve \"Cevap:\" etiketi kullanma. "
    "Kullandığın her bilginin ardından (KAYNAK 1) biçiminde kaynağını yaz."
)

# Returned instead of calling the model when the gate says NOT_FOUND. Fixed
# text, not generated: the one thing a model must never be asked to do is
# improvise around "I have no source for this".
NOT_FOUND_MESSAGE = (
    "Sorunuzla ilgili bir hüküm, elimdeki elektrik piyasası mevzuatı "
    "derlemesinde bulunamadı.\n\n"
    "Şunları denemenizi öneririm:\n"
    "- Soruyu mevzuatta geçen terimlerle yeniden yazın "
    "(örnek: \"abonelik\" yerine \"bağlantı anlaşması\").\n"
    "- Sorunuzu daha belirgin hale getirin: hangi piyasa faaliyeti, hangi "
    "lisans türü veya hangi taraf hakkında olduğunu belirtin.\n"
    "- Sorunuz doğal gaz, LPG veya petrol piyasasıyla ilgiliyse: bu derleme "
    "yalnızca elektrik piyasası mevzuatını kapsıyor."
)

CONTEXT_EXHAUSTED_MESSAGE = (
    "Sorunuza ilişkin bulunan mevzuat metinleri, modelin tek seferde "
    "işleyebileceği uzunluğu aştığı için cevap üretilemedi.\n\n"
    "Soruyu daha dar kapsamlı sorarsanız (tek bir konu veya tek bir lisans "
    "türü hakkında) cevap üretilebilir."
)

GENERATION_FAILED_MESSAGE = (
    "Cevap üretilirken dil modeline erişilemedi. Foundry Local sunucusunun "
    "çalıştığını doğrulayıp soruyu tekrar deneyin."
)

EMPTY_ANSWER_MESSAGE = (
    "Model, ilgili mevzuat metinlerini buldu ancak okunabilir bir cevap "
    "üretmeden ayrılan cevap uzunluğunu doldurdu.\n\n"
    "Soruyu daha kısa ve tek konuya odaklı biçimde tekrar sorarsanız cevap "
    "üretilebilir."
)

# --------------------------------------------------------------------------
# Context scrubbing -- removing every citable identifier from the chunk body
# --------------------------------------------------------------------------

REDACTED = "[…]"

# Turkish mevzuat-type words, with a trailing \w* to absorb the case suffixes
# Turkish attaches to them ("Yönetmeliğin", "Kanununun", "Tebliğinde").
_TYPE_WORDS_UPPER = (
    r"KANUN|YÖNETMELİK|YÖNETMELİĞ|TEBLİĞ|TEBLIG|KARARNAME|KARAR|GENELGE|"
    r"TALİMAT|ESASLAR|USUL|KHK"
)
_TYPE_WORDS_MIXED = (
    r"Kanun|Yönetmelik|Yönetmeliğ|Tebliğ|Teblig|Kararname|Karar|Genelge|"
    r"Talimat|Esaslar|Usul"
)

# Order matters: the most specific patterns run first, so that a Resmî Gazete
# parenthetical is consumed whole rather than being half-eaten by the date rule.
_SCRUB_PATTERNS: tuple[re.Pattern[str], ...] = (
    # (Değişik:RG-24/2/2017-29989), (Ek:RG-...), (Mülga:RG-...)
    re.compile(
        r"\(\s*(?:Değişik|Degisik|Ek|Mülga|Mulga|Yeniden\s+düzenleme|İptal)\s*:?\s*"
        r"[^()]*?RG[^()]*\)",
        re.IGNORECASE,
    ),
    # Bare Resmî Gazete references, with or without the RG- prefix.
    re.compile(r"RG\s*-\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}\s*-\s*\d+"),
    re.compile(r"Resm[îi]\s*Gazete(?:'?\w*)?", re.IGNORECASE),
    # Article markers: MADDE 6, GEÇİCİ MADDE 12/A, EK MADDE 1, Madde 3-
    re.compile(
        r"(?:(?:GEÇİCİ|GECICI|EK|Geçici|Gecici|Ek)\s+)?"
        r"(?:MADDE|Madde|madde)\s*\d+\s*(?:/\s*[A-Za-zÇĞİÖŞÜçğıöşü])?",
    ),
    # Article cross-references: "3 üncü maddesinin", "geçici 5 inci maddesinde"
    re.compile(
        r"(?:(?:geçici|gecici|ek)\s+)?\d+\s*"
        r"(?:inci|ıncı|nci|ncı|uncu|üncü|ünci)\s+madde\w*",
        re.IGNORECASE,
    ),
    # Written-ordinal article cross-references: "birinci maddesinde"
    re.compile(
        r"(?:birinci|ikinci|üçüncü|dördüncü|beşinci|altıncı|yedinci|sekizinci|"
        r"dokuzuncu|onuncu)\s+madde\w*",
        re.IGNORECASE,
    ),
    # Law/decision numbers: "6446 sayılı", "10543 sayılı Kurul Kararı"
    re.compile(r"\d{3,6}\s*say[ıi]l[ıi]\s*\w*", re.IGNORECASE),
    # The same identifiers in masthead field form: "Kanun No. : 4628",
    # "Karar No: 10543", "Sayı : 32415". Not caught by the "sayılı" rule above,
    # and a law number identifies a document just as precisely as its title.
    re.compile(
        r"(?:Kanun|Karar|Kararname|Yönetmelik|Tebli[ğg]|Say[ıi])\s*(?:No|Numaras[ıi])"
        r"\s*\.?\s*:?\s*\d+",
        re.IGNORECASE,
    ),
    # Uppercase mevzuat titles: ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ
    re.compile(
        r"(?:[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9'’]*\s+){1,12}"
        rf"(?:{_TYPE_WORDS_UPPER})\w*"
    ),
    # Mixed-case named mevzuat: "Doğal Gaz Piyasası Kanununun", "Aynı Yönetmeliğin"
    re.compile(
        r"(?:[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+){1,8}"
        rf"(?:{_TYPE_WORDS_MIXED})\w*"
    ),
    # A bare qualified mevzuat-type word still names a document by reference.
    re.compile(rf"(?:Bu|Aynı|Ayni|İlgili|Ilgili|Söz\s+konusu)\s+(?:{_TYPE_WORDS_MIXED})\w*"),
    # Dates: 24/2/2017, 1.1.2026, and "18 Nisan 2001"
    re.compile(r"\d{1,2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{2,4}"),
    re.compile(
        r"\d{1,2}\s+(?:Ocak|Şubat|Subat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|"
        r"Ağustos|Agustos|Eylül|Eylul|Ekim|Kasım|Kasim|Aralık|Aralik)\s+\d{4}",
        re.IGNORECASE,
    ),
)

_COLLAPSE_REDACTIONS = re.compile(r"(?:\[…\]\s*){2,}")
_COLLAPSE_SPACE = re.compile(r"[ \t]{2,}")

# A title line the model adds despite being told not to: "**Serbest Tüketici
# Nasıl Değiştirir?**" or "### Cevap". Stripped as presentation only -- it is
# removed when it is a heading with no sentence in it and something follows, so
# no substantive content and no citation can be lost this way. Deliberately
# narrow: the answer's prose is never rewritten, only an empty header dropped.
_LEADING_HEADING = re.compile(r"^\s*(?:#{1,6}\s*)?\*{0,2}[^\n.!?]{0,120}\*{0,2}\s*$")


def strip_leading_heading(answer: str) -> str:
    """Drop a leading empty heading line the model added despite being told not to."""
    lines = answer.split("\n")
    if len(lines) < 2:
        return answer.strip()
    first = lines[0].strip()
    is_heading = first.startswith("#") or (
        first.startswith("**") and first.endswith("**") and len(first) > 4
    )
    if is_heading and _LEADING_HEADING.match(first) and "KAYNAK" not in first:
        return "\n".join(lines[1:]).strip()
    return answer.strip()


def scrub_context(text: str, extra_terms: tuple[str, ...] = ()) -> str:
    """Strip every citable identifier from chunk text before the model sees it.

    Removes mevzuat titles, article markers and cross-references, Resmî Gazete
    references, law numbers and dates, replacing each with `[…]`. What survives
    is the substantive provision -- the part that actually answers the question.

    `extra_terms` are literal strings scrubbed in addition to the patterns; the
    caller passes the retrieved chunks' real `document_title` values so that a
    title the generic patterns happen not to match is still removed. Longest
    first, so a title is not left half-scrubbed by one of its own substrings.

    This is one half of the citation invariant; the other half is that metadata
    fields are simply never rendered. Both are asserted in tests/test_answer.py.
    """
    scrubbed = text
    for term in sorted({t for t in extra_terms if t and len(t) > 3}, key=len, reverse=True):
        scrubbed = re.sub(re.escape(term), REDACTED, scrubbed, flags=re.IGNORECASE)
    for pattern in _SCRUB_PATTERNS:
        scrubbed = pattern.sub(REDACTED, scrubbed)
    scrubbed = _COLLAPSE_REDACTIONS.sub(REDACTED + " ", scrubbed)
    scrubbed = _COLLAPSE_SPACE.sub(" ", scrubbed)
    return scrubbed.strip()


# --------------------------------------------------------------------------
# Source blocks and citations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceBlock:
    """One retrieved chunk as the model will see it: a label and scrubbed text."""

    label: int
    result: RetrievalResult
    text: str
    tokens: int

    def render(self) -> str:
        """The exact text sent to the model for this source: a KAYNAK-labelled block."""
        return f"KAYNAK {self.label}:\n{self.text}"


@dataclass(frozen=True)
class Citation:
    """A user-facing citation, assembled in code from the store's own metadata."""

    label: int
    document_title: str | None
    article_ref: str | None
    page_start: int | None
    page_end: int | None
    source_path: str
    chunk_id: int
    quality_flag: str | None

    @classmethod
    def from_block(cls, block: SourceBlock) -> "Citation":
        """Build a citation from the store's own metadata, never from model output."""
        r = block.result
        return cls(
            label=block.label,
            document_title=r.document_title,
            article_ref=r.article_ref,
            page_start=r.page_start,
            page_end=r.page_end,
            source_path=r.source_path,
            chunk_id=r.chunk_id,
            quality_flag=r.quality_flag,
        )

    def render(self) -> str:
        """Human-readable citation line: [KAYNAK n] title / article / pages."""
        parts = [self.document_title or self.source_path]
        if self.article_ref:
            parts.append(self.article_ref)
        if self.page_start is not None:
            parts.append(
                f"s. {self.page_start}"
                if self.page_end in (None, self.page_start)
                else f"s. {self.page_start}-{self.page_end}"
            )
        return f"[KAYNAK {self.label}] " + " / ".join(parts)


@dataclass
class GeneratedAnswer:
    """Everything a caller (CLI now, UI later) needs about one answered question."""

    question: str
    decision: GateDecision
    confidence: float
    text: str
    citations: list[Citation] = field(default_factory=list)
    results: list[RetrievalResult] = field(default_factory=list)
    # True when the gate said ANSWER_WEAK: generated normally, but the caller
    # should show a low-confidence warning.
    low_confidence: bool = False
    generated: bool = False
    chunks_retrieved: int = 0
    chunks_dropped: int = 0
    hallucinated_references: int = 0
    hallucinated_labels: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    rewritten_query: str | None = None

    @property
    def cited_labels(self) -> list[int]:
        """KAYNAK labels the model actually cited, in citation order."""
        return [c.label for c in self.citations]


# --------------------------------------------------------------------------
# Label parsing
# --------------------------------------------------------------------------

# Matches "KAYNAK 1", "KAYNAK 1 ve 2", "KAYNAK 1, 2 ve 3", "[KAYNAK 4]".
_LABEL_RUN = re.compile(
    r"KAYNAK\s*((?:\d+\s*(?:,|ve|ile|/|&|-)\s*)*\d+)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d+")


def parse_labels(answer: str) -> list[int]:
    """Extract the KAYNAK numbers the model referenced, in order of first appearance.

    Handles the enumerations Turkish makes natural ("KAYNAK 1 ve 2") rather than
    only the single-label form, because a label the parser misses would be
    counted as an uncited source and quietly drop a real citation.
    """
    seen: list[int] = []
    for match in _LABEL_RUN.finditer(answer):
        for number in _NUMBER.findall(match.group(1)):
            label = int(number)
            if label not in seen:
                seen.append(label)
    return seen


# --------------------------------------------------------------------------
# Context budgeting
# --------------------------------------------------------------------------


def build_source_blocks(
    results: list[RetrievalResult],
    client: ChatClient,
    available_tokens: int,
) -> tuple[list[SourceBlock], int]:
    """Fit rank-ordered chunks into `available_tokens`, dropping the lowest-ranked.

    Chunks are never truncated. A chunk is either included whole or dropped --
    half a provision is worse than no provision, because the missing half is
    invisible to both the model and the reader. Keeps the longest rank-ordered
    prefix that fits, so what survives is always the best-ranked material.

    Returns (blocks, dropped_count).
    """
    blocks: list[SourceBlock] = []
    used = 0
    for position, result in enumerate(results, start=1):
        text = scrub_context(
            result.text,
            extra_terms=(result.document_title or "", result.article_ref or ""),
        )
        block = SourceBlock(label=position, result=result, text=text, tokens=0)
        cost = client.counter.count(block.render()) + config.TOKENS_PER_MESSAGE
        if used + cost > available_tokens:
            break
        blocks.append(SourceBlock(label=position, result=result, text=text, tokens=cost))
        used += cost
    return blocks, len(results) - len(blocks)


# Qwen3's thinking soft-switch. It is a PER-TURN switch parsed from the current
# user turn, so it has to end the user message -- putting it only in the system
# prompt works on short prompts and is silently ignored on a full RAG prompt,
# where the model then spends its entire completion budget on an unterminated
# <think> block and returns nothing usable. Measured, not guessed: with the
# marker in the system prompt only, a 2,699-token prompt produced 900 completion
# tokens of English reasoning with no closing tag and an empty answer.
#
# The OpenAI-style alternative, extra_body={"chat_template_kwargs":
# {"enable_thinking": False}}, is NOT honoured by this Foundry Local build --
# it was tried and the model kept thinking.
NO_THINK = "/no_think"

# One worked example, because a 4B model learns a format by demonstration and
# only partly by instruction. With rules alone this model kept emitting "**Cevap:**"
# headers, markdown bullet lists and numbered sections that SYSTEM_PROMPT
# explicitly forbids, and -- more damaging -- reproduced the KAYNAK blocks
# verbatim instead of citing them, which yields an answer with no citations at
# all. The example demonstrates all four things the rules ask for at once: plain
# prose, inline (KAYNAK n) attribution, brevity, and a […] span left alone.
#
# It costs ~130 prompt tokens on a ~3,000-token budget. That is roughly one
# retrieved chunk, and it buys the citations the entire module exists to produce.
# The invented example text is generic licensing language, deliberately not
# copied from the corpus, so it cannot be mistaken for a retrieved source.
FEW_SHOT: list[Message] = [
    {
        "role": "user",
        "content": (
            "KAYNAK 1:\n"
            "[…] (1) Lisans süresi en az on yıl, en çok kırk dokuz yıl olarak "
            "belirlenir.\n\n"
            "KAYNAK 2:\n"
            "[…] (2) Lisans süresi, lisans sahibinin talebi üzerine uzatılabilir.\n\n"
            "SORU: Lisans süresi ne kadardır?\n\n"
            f"{NO_THINK}"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Lisans süresi en az on yıl, en çok kırk dokuz yıl olarak belirlenir "
            "(KAYNAK 1). Bu süre, lisans sahibinin talebi üzerine uzatılabilir "
            "(KAYNAK 2)."
        ),
    },
]


def preamble_messages() -> list[Message]:
    """Everything in the prompt except the retrieved sources and the question.

    Exists so that budgeting and prompt construction cannot disagree: the
    Answerer subtracts exactly this from the context budget, and build_messages
    sends exactly this. Computing the overhead from a different list than the one
    actually sent is how a budget silently starts under-counting.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}, *FEW_SHOT]


def build_messages(
    question: str,
    blocks: list[SourceBlock],
    history_text: str | None = None,
) -> list[Message]:
    """Assemble the exact message list sent to the server.

    Nothing but the preamble, the KAYNAK blocks, the conversation recap and the
    question goes in. This function is the single place a prompt is constructed,
    which is what makes the citation invariant testable: a test can call it and
    inspect every byte the model would receive.

    Conversation history arrives as TEXT inside the user turn, not as alternating
    user/assistant messages. That is a measured requirement of this model, not a
    stylistic choice: with history as real chat messages, Qwen3 stops honouring
    the /no_think switch and emits an unclosed <think> tag with the answer inside
    it. The identical prompt with history folded into the user turn returns a
    properly closed empty block and a clean answer, and it was reproducible in
    both directions.
    """
    parts = [block.render() for block in blocks]
    if history_text:
        parts.append(history_text)
    parts.append(question_block(question))
    return [*preamble_messages(), {"role": "user", "content": "\n\n".join(parts)}]


def question_block(question: str) -> str:
    """The tail of the user turn: question, format reminder, thinking switch.

    Shared with the budget calculation for the same reason preamble_messages()
    is -- so the cost that is subtracted is the cost that is actually sent.
    """
    return f"SORU: {question}\n\n{ANSWER_INSTRUCTION}\n\n{NO_THINK}"


# --------------------------------------------------------------------------
# Answerer
# --------------------------------------------------------------------------


@dataclass
class Answerer:
    """Retrieval + gate + generation + citation assembly.

    Holds a Retriever and a ChatClient, both expensive to construct (a 111 MB
    embedding matrix, a BM25 index over 27k chunks, and 2.6 GB of VRAM), so one
    instance is built at startup and reused.
    """

    retriever: Retriever
    client: ChatClient

    @classmethod
    def open(cls, db_path: str | None = None) -> "Answerer":
        """Connect the retriever and chat client. Call once and reuse the result."""
        return cls(retriever=Retriever.open(db_path), client=ChatClient.connect())

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        history_text: str | None = None,
    ) -> GeneratedAnswer:
        """Answer one question end to end.

        The gate decides whether the model is called at all:
          NOT_FOUND    -> fixed refusal, model never invoked
          ANSWER_WEAK  -> generated, low_confidence set for the caller
          ANSWER       -> generated
        """
        timings: dict[str, float] = {}

        started = time.perf_counter()
        results, retrieval_timings = self.retriever.retrieve_fused_timed(question, top_k)
        confidence = self.retriever.confidence(question, results)
        decision = gate(confidence)
        timings["retrieve"] = time.perf_counter() - started
        timings.update({f"retrieve_{k}": v for k, v in retrieval_timings.items()})

        answer = GeneratedAnswer(
            question=question,
            decision=decision,
            confidence=confidence,
            text="",
            results=results,
            low_confidence=decision == "ANSWER_WEAK",
            chunks_retrieved=len(results),
            timings=timings,
        )

        # NOT_FOUND: the model is never called. Nothing it could say would be
        # grounded, so there is no version of this worth spending a call on.
        if decision == "NOT_FOUND":
            answer.text = NOT_FOUND_MESSAGE
            answer.chunks_dropped = len(results)
            timings["total"] = time.perf_counter() - started
            return answer

        budget_started = time.perf_counter()
        overhead = self.client.counter.count_messages(preamble_messages())
        question_cost = (
            self.client.counter.count(question_block(question)) + config.TOKENS_PER_MESSAGE
        )
        if history_text:
            question_cost += self.client.counter.count(history_text)
        available = self.client.context_budget - overhead - question_cost
        blocks, dropped = build_source_blocks(results, self.client, available)
        answer.chunks_dropped = dropped
        timings["budget"] = time.perf_counter() - budget_started

        if not blocks:
            # Even the top-ranked chunk does not fit. Refusing is the honest
            # outcome; generating from no context would be ungrounded.
            logger.warning(
                "context budget %d tokens could not fit even the top-ranked chunk "
                "for %r; refusing rather than generating ungrounded",
                available, question[:60],
            )
            answer.text = CONTEXT_EXHAUSTED_MESSAGE
            timings["total"] = time.perf_counter() - started
            return answer

        messages = build_messages(question, blocks, history_text)
        answer.prompt_tokens = self.client.count_messages(messages)

        generate_started = time.perf_counter()
        try:
            raw = self.client.complete(messages)
        except ContextExhausted as exc:
            # Budgeting is meant to prevent this; if it happens anyway the
            # budget is wrong, so say so loudly and still answer the user in
            # Turkish rather than crashing.
            logger.error(
                "server rejected a prompt of ~%d tokens that budgeting accepted "
                "(budget %d): %s",
                answer.prompt_tokens, self.client.context_budget, exc,
            )
            answer.text = CONTEXT_EXHAUSTED_MESSAGE
            timings["generate"] = time.perf_counter() - generate_started
            timings["total"] = time.perf_counter() - started
            return answer
        except Exception as exc:  # noqa: BLE001 - user gets Turkish, log gets detail
            logger.error("generation failed for %r: %s", question[:60], exc)
            answer.text = GENERATION_FAILED_MESSAGE
            timings["generate"] = time.perf_counter() - generate_started
            timings["total"] = time.perf_counter() - started
            return answer
        timings["generate"] = time.perf_counter() - generate_started

        # An empty completion is a real, observed failure mode, not a hypothetical:
        # when the thinking soft-switch is not honoured the model spends the whole
        # completion budget inside an unterminated <think> block, and stripping it
        # correctly leaves nothing. Showing the user a blank answer under a
        # confident ANSWER decision would be the worst available outcome, so this
        # is surfaced instead.
        if not raw.strip():
            logger.error(
                "model returned an empty answer for %r (prompt ~%d tokens, "
                "completion cap %d) -- most likely the whole budget went into an "
                "unterminated <think> block",
                question[:60], answer.prompt_tokens, config.CHAT_MAX_COMPLETION_TOKENS,
            )
            answer.text = EMPTY_ANSWER_MESSAGE
            timings["total"] = time.perf_counter() - started
            return answer

        answer.generated = True
        answer.text = strip_leading_heading(raw)
        by_label = {block.label: block for block in blocks}
        for label in parse_labels(raw):
            block = by_label.get(label)
            if block is None:
                answer.hallucinated_labels.append(label)
                continue
            answer.citations.append(Citation.from_block(block))

        answer.hallucinated_references = len(answer.hallucinated_labels)
        if answer.hallucinated_references:
            # Not swallowed: a reference to a label that was never supplied means
            # the model invented a source, which is exactly the failure the whole
            # labelling scheme exists to make visible.
            logger.warning(
                "model referenced %d unsupplied KAYNAK label(s) %s for %r "
                "(supplied 1..%d) -- dropped from citations",
                answer.hallucinated_references, answer.hallucinated_labels,
                question[:60], len(blocks),
            )

        timings["total"] = time.perf_counter() - started
        return answer

    def close(self) -> None:
        """Release the underlying SQLite connection."""
        self.retriever.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_answer(answer: GeneratedAnswer, show_sources: bool = False) -> str:
    """Render one answer as the CLI's human-readable report block."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"SORU     : {answer.question}")
    if answer.rewritten_query and answer.rewritten_query != answer.question:
        lines.append(f"YENİDEN  : {answer.rewritten_query}   (takip sorusu olarak yeniden yazıldı)")
    lines.append(
        f"KARAR    : {answer.decision}  (güven {answer.confidence:.5f}; "
        f"eşik {config.FUSION_THRESHOLD}, taban {config.FUSION_FLOOR})"
    )
    lines.append(
        f"ÜRETİM   : {'evet' if answer.generated else 'HAYIR - model çağrılmadı'}"
        + (f"  |  prompt ~{answer.prompt_tokens} token" if answer.generated else "")
    )
    if answer.low_confidence:
        lines.append(
            "UYARI    : Düşük güven (ANSWER_WEAK). Cevap üretildi, ancak "
            "dayanağın soruyla ilgisi zayıf olabilir."
        )
    if answer.chunks_dropped:
        lines.append(
            f"BÜTÇE    : {answer.chunks_dropped}/{answer.chunks_retrieved} parça "
            f"bağlam bütçesine sığmadığı için düşürüldü (en düşük sıralı olanlar)."
        )
    if answer.hallucinated_references:
        lines.append(
            f"UYARI    : Model verilmeyen {answer.hallucinated_references} etikete "
            f"atıf yaptı {answer.hallucinated_labels} - atıflar listeden düşürüldü."
        )
    lines.append("=" * 72)
    lines.append("")
    lines.append(answer.text)
    lines.append("")

    if answer.citations:
        lines.append("-" * 72)
        lines.append("KAYNAKÇA")
        lines.append("-" * 72)
        for citation in answer.citations:
            lines.append("  " + citation.render())
            if citation.quality_flag:
                lines.append(f"      (çıkarım uyarısı: {citation.quality_flag})")
            lines.append(f"      {citation.source_path}")
    elif answer.generated:
        lines.append("-" * 72)
        lines.append("KAYNAKÇA : model hiçbir KAYNAK etiketine atıf yapmadı.")

    if show_sources:
        lines.append("")
        lines.append("-" * 72)
        lines.append("MODELE GÖNDERİLEN BAĞLAM (temizlenmiş)")
        lines.append("-" * 72)
        for result in answer.results:
            scrubbed = scrub_context(
                result.text,
                extra_terms=(result.document_title or "", result.article_ref or ""),
            )
            lines.append(f"  {' '.join(scrubbed.split())[:300]}")

    timings = answer.timings
    if timings:
        lines.append("")
        lines.append("-" * 72)
        lines.append(
            "SÜRE     : "
            f"retrieve {timings.get('retrieve', 0) * 1000:.0f} ms"
            f" (dense {timings.get('retrieve_dense', 0) * 1000:.0f} ms || "
            f"bm25 {timings.get('retrieve_lexical', 0) * 1000:.0f} ms)"
            f" -> bütçe {timings.get('budget', 0) * 1000:.1f} ms"
            f" -> üretim {timings.get('generate', 0) * 1000:.0f} ms"
            f" = toplam {timings.get('total', 0) * 1000:.0f} ms"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: answer one question (and optional follow-ups) and print the report."""
    import argparse

    from .extract import _force_utf8_stdout
    from .llm import FoundryUnavailable

    _force_utf8_stdout()
    parser = argparse.ArgumentParser(
        prog="python -m src.answer",
        description="Full RAG pipeline: retrieve -> gate -> generate -> cited answer.",
    )
    parser.add_argument("question", help="the question to answer")
    parser.add_argument(
        "--then", action="append", default=[], metavar="SORU",
        help="a follow-up question, asked in the same session (repeatable)",
    )
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--db", default=None, help=f"SQLite path (default: {config.DB_PATH})")
    parser.add_argument(
        "--show-context", action="store_true",
        help="print the scrubbed context actually sent to the model",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the rewrite and hallucinated-reference log lines",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
        # The OpenAI client logs one INFO line per HTTP call, which buries the
        # rewrite and hallucinated-reference lines this flag exists to show.
        for noisy in ("httpx", "httpcore", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        answerer = Answerer.open(args.db)
    except FoundryUnavailable as exc:
        print(f"FATAL: {exc}")
        return 3

    retriever = answerer.retriever
    print(
        f"Vektörler : {len(retriever):,} aktif parça "
        f"({retriever.matrix.nbytes / 1e6:.1f} MB) {retriever.load_seconds:.2f}s"
    )
    print(
        f"BM25      : {len(retriever.bm25):,} belge, "
        f"{retriever.bm25.vocabulary_size:,} terim, {retriever.bm25_load_seconds:.2f}s"
    )
    print(f"Gömme     : {retriever.embedder.model_id} (dim {retriever.embedder.dimension})")
    print(f"Sohbet    : {answerer.client.model_id} (temperature {config.CHAT_TEMPERATURE})")
    print(
        f"Bağlam    : {config.CHAT_EFFECTIVE_CONTEXT} token ölçülen tavan "
        f"(model beyanı {config.CHAT_CONTEXT_WINDOW}); "
        f"prompt bütçesi {answerer.client.context_budget}"
    )
    print(
        f"Sayaç     : {'gerçek tokenizer' if answerer.client.counter.exact else 'TAHMİN'}"
        f" ({answerer.client.counter.source})"
    )
    print()

    # Multi-turn only when follow-ups were actually asked, so a single question
    # never pays for the rewrite round trip.
    if args.then:
        from .session import Session

        session = Session(answerer=answerer)
        for question in [args.question, *args.then]:
            print(format_answer(session.ask(question, top_k=args.top_k), args.show_context))
            print()
    else:
        print(format_answer(answerer.answer(args.question, args.top_k), args.show_context))
        print()

    answerer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
