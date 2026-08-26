"""Tests for generation: the citation invariant, gate integration, and budgeting.

The load-bearing test in this file is
TestCitationInvariant::test_no_citable_identifier_reaches_the_model. Everything
src/answer.py claims about citations rests on one property -- that the model is
physically unable to write a real citation because it was never shown one -- and
that property is either asserted against the bytes sent to the server or it is
just a comment. So it is asserted against the bytes.

No Foundry Local dependency: the chat client is stubbed and the corpus is
hand-built chunks with realistic Turkish mevzuat text (article markers, Resmî
Gazete references, dates, masthead fields), because those are exactly the strings
that must not survive scrubbing.
"""
import numpy as np
import pytest

from src import config, store
from src.answer import (
    ANSWER_INSTRUCTION,
    FEW_SHOT,
    NO_THINK,
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    Answerer,
    build_messages,
    build_source_blocks,
    parse_labels,
    preamble_messages,
    question_block,
    scrub_context,
    strip_leading_heading,
)
from src.chunk import ArticleRef
from src.llm import TokenCounter, strip_thinking
from src.retrieval import Retriever
from src.lexical import BM25Index

from tests.test_retrieval import FakeEmbedder, make_chunk, unit

# --------------------------------------------------------------------------
# Stub chat client
# --------------------------------------------------------------------------


class StubChatClient:
    """Mirrors the ChatClient surface src.answer actually uses, and records calls.

    `messages_seen` keeps every message list passed to complete(), which is what
    the citation-invariant test inspects. Token counting runs in TokenCounter's
    estimate mode so the tests stay hermetic -- they must not depend on Foundry
    Local's model cache being present.
    """

    model_id = "stub-chat"

    def __init__(self, responses=None, default="Cevap yok.", budget=None):
        self.counter = TokenCounter()
        self.messages_seen: list[list[dict]] = []
        self.responses = list(responses or [])
        self.default = default
        self.calls = 0
        self._budget = budget

    @property
    def context_budget(self) -> int:
        if self._budget is not None:
            return self._budget
        return (
            config.CHAT_EFFECTIVE_CONTEXT
            - config.CHAT_MAX_COMPLETION_TOKENS
            - config.CONTEXT_SAFETY_MARGIN
        )

    def count_messages(self, messages) -> int:
        return self.counter.count_messages(messages)

    def complete(self, messages, max_completion_tokens=None) -> str:
        self.messages_seen.append(messages)
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return self.default

    @property
    def last_prompt_text(self) -> str:
        """Every byte of the most recent call, flattened -- what the model saw."""
        return "\n".join(m.get("content", "") for m in self.messages_seen[-1])

    @property
    def last_context_text(self) -> str:
        """The same, minus the fixed system prompt.

        SYSTEM_PROMPT is authored text containing no chunk-derived data, and it
        necessarily names the identifier classes it forbids ("Resmî Gazete atfı
        ... YAZMA"). Assertions about those class names therefore target the
        context, while assertions about a specific chunk's real metadata still
        target the whole prompt -- that must appear nowhere at all.
        """
        return "\n".join(
            m.get("content", "")
            for m in self.messages_seen[-1]
            if m.get("role") != "system"
        )

    @property
    def last_sources_text(self) -> str:
        """Only the final user turn: the retrieved sources plus the question.

        Label-presence checks must use this rather than the whole prompt. Both
        SYSTEM_PROMPT and the one-shot example legitimately contain the strings
        "KAYNAK 1" and "KAYNAK 2" -- as the format to follow, and as a worked
        demonstration -- so searching the full prompt for a label always matches
        and would make every budgeting assertion vacuous.
        """
        return self.messages_seen[-1][-1].get("content", "")


# Realistic mevzuat text: each carries the identifiers that must not leak.
MEVZUAT_CHUNKS = [
    {
        "text": (
            "MADDE 9 – (Değişik:RG-24/2/2017-29989) (1) Önlisansın süresi, önlisans "
            "başvurusuna konu üretim tesisi projesinin kaynak türü ve kurulu gücüne "
            "bağlı olarak, mücbir sebep hâlleri hariç, otuz altı ayı geçmemek üzere "
            "Kurul kararı ile belirlenir."
        ),
        "article": ArticleRef("MADDE", "9"),
        "title": "ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ",
        "path": "mevzuat/raw/Yonetmelikler/lisans.docx",
        "page_start": 4,
        "page_end": 4,
    },
    {
        "text": (
            "MADDE 6 — (1) Dağıtım bağlantı tarifesi; dağıtım bağlantı bedeli ile "
            "tarifenin uygulanmasına ilişkin usul ve esaslardan oluşur. (2) Dağıtım "
            "bağlantı bedeli, şebeke yatırım maliyetlerini içermez."
        ),
        "article": ArticleRef("MADDE", "6"),
        "title": "ELEKTRİK PİYASASI TARİFELER YÖNETMELİĞİ",
        "path": "mevzuat/raw/Yonetmelikler/tarifeler.doc",
        "page_start": 2,
        "page_end": 3,
    },
    {
        "text": (
            "GEÇİCİ MADDE 3 – 18/4/2001 tarihli ve 4646 sayılı Doğal Gaz Piyasası "
            "Kanununun 12 nci maddesinin birinci fıkrası uyarınca, 6446 sayılı Kanun "
            "kapsamındaki tüzel kişiler Resmî Gazete'de yayımlanan usullere uyar."
        ),
        "article": ArticleRef("GEÇİCİ MADDE", "3"),
        "title": "ELEKTRİK PİYASASI KANUNU",
        "path": "mevzuat/raw/Kanunlar/6446.doc",
        "page_start": 11,
        "page_end": 11,
    },
]


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "answer.db")
    yield c
    c.close()


@pytest.fixture
def populated(conn):
    """Three realistic chunks, all embedded at unit(0) so dense ranking is a tie
    broken by id -- retrieval order is therefore insertion order, which the
    budgeting tests rely on."""
    for i, spec in enumerate(MEVZUAT_CHUNKS):
        store.insert_chunks(
            conn,
            source_path=spec["path"],
            file_sha256=f"hash-{i}",
            document_title=spec["title"],
            chunks=[
                make_chunk(
                    index=0,
                    text=spec["text"],
                    article=spec["article"],
                    page_start=spec["page_start"],
                    page_end=spec["page_end"],
                )
            ],
            embeddings=[unit(0)],
            quality_flag=None,
        )
    return conn


def make_answerer(conn, client=None, confidence=None, embedder=None):
    """An Answerer over `conn` with a stubbed chat client and, optionally, a
    forced confidence so a specific gate branch can be exercised."""
    from src import retrieval as retrieval_module

    matrix, ids = store.fetch_active_embeddings(conn)
    matrix = retrieval_module._l2_normalize(np.ascontiguousarray(matrix, dtype=np.float32))
    retriever = Retriever(
        conn=conn,
        embedder=embedder or FakeEmbedder({"q": unit(0)}),
        matrix=matrix,
        ids=ids,
        load_seconds=0.0,
        bm25=BM25Index.from_connection(conn),
    )
    if confidence is not None:
        retriever.confidence = lambda question, results: confidence  # type: ignore[assignment]
    return Answerer(retriever=retriever, client=client or StubChatClient())


def budget_admitting(conn, question: str, keep: int, top_k: int = 3) -> int:
    """The exact context budget that admits the first `keep` retrieved chunks.

    Derived rather than hardcoded: a budget literal would silently stop meaning
    what it says the moment SYSTEM_PROMPT is reworded or a chunk's text changes,
    and the test would keep passing while measuring something else. Computed as
    the fixed overhead plus the cost of exactly the chunks meant to survive --
    since every block costs a positive number of tokens, that admits `keep` and
    excludes `keep + 1`.
    """
    probe = StubChatClient()
    answerer = make_answerer(conn, probe, confidence=0.9)
    results = answerer.retriever.retrieve(question, top_k=top_k)
    counter = probe.counter

    # Same overhead the Answerer computes, from the same function, so that adding
    # to the preamble (a few-shot example, say) cannot make these budgets mean
    # something other than what they say.
    overhead = counter.count_messages(preamble_messages())
    overhead += (
        counter.count(question_block(question)) + config.TOKENS_PER_MESSAGE
    )

    costs = []
    for position, retrieved in enumerate(results, start=1):
        scrubbed = scrub_context(
            retrieved.text,
            extra_terms=(retrieved.document_title or "", retrieved.article_ref or ""),
        )
        costs.append(
            counter.count(f"KAYNAK {position}:\n{scrubbed}") + config.TOKENS_PER_MESSAGE
        )
    return overhead + sum(costs[:keep])


# --------------------------------------------------------------------------
# The citation invariant
# --------------------------------------------------------------------------


class TestCitationInvariant:
    """Nothing citable may reach the model. Asserted against the sent bytes."""

    def test_no_citable_identifier_reaches_the_model(self, populated):
        client = StubChatClient(responses=["KAYNAK 1 uyarınca süre otuz altı aydır."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Önlisans süresi nedir?", top_k=3)

        assert client.calls == 1, "the model must actually have been called"
        prompt = client.last_prompt_text

        for retrieved in result.results:
            if retrieved.article_ref:
                assert retrieved.article_ref not in prompt, (
                    f"article_ref {retrieved.article_ref!r} leaked into the prompt"
                )
            if retrieved.document_title:
                assert retrieved.document_title not in prompt, (
                    f"document_title {retrieved.document_title!r} leaked into the prompt"
                )
            assert retrieved.source_path not in prompt
            if retrieved.page_start is not None:
                assert f"s. {retrieved.page_start}" not in prompt

    @pytest.mark.parametrize(
        "forbidden",
        [
            "RG-24/2/2017-29989",
            "Resmî Gazete",
            "MADDE 9",
            "MADDE 6",
            "GEÇİCİ MADDE 3",
            "24/2/2017",
            "18/4/2001",
            "6446 sayılı",
            "4646 sayılı",
            "ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ",
            "ELEKTRİK PİYASASI TARİFELER YÖNETMELİĞİ",
            "ELEKTRİK PİYASASI KANUNU",
            "Doğal Gaz Piyasası Kanununun",
            "12 nci maddesinin",
        ],
        ids=lambda s: s[:28],
    )
    def test_specific_identifier_classes_are_absent(self, populated, forbidden):
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        answerer.answer("Önlisans süresi nedir?", top_k=3)
        assert forbidden not in client.last_context_text

    def test_the_substantive_provision_still_survives(self, populated):
        """Guards the test above from passing trivially.

        A scrubber that deleted everything would satisfy every absence
        assertion. The content that answers the question must still be there.
        """
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        answerer.answer("Önlisans süresi nedir?", top_k=3)
        prompt = client.last_prompt_text
        assert "otuz altı ayı geçmemek üzere" in prompt
        assert "mücbir sebep" in prompt
        assert "şebeke yatırım maliyetlerini içermez" in prompt

    def test_sources_appear_only_under_opaque_labels(self, populated):
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        answerer.answer("Önlisans süresi nedir?", top_k=3)
        context = client.last_sources_text
        for label in ("KAYNAK 1", "KAYNAK 2", "KAYNAK 3"):
            assert label in context

    def test_system_prompt_forbids_writing_citations(self, populated):
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        answerer.answer("Önlisans süresi nedir?", top_k=1)
        system = client.messages_seen[-1][0]
        assert system["role"] == "system"
        assert system["content"] == SYSTEM_PROMPT
        # The prohibition is load-bearing; assert it is actually stated.
        assert "YAZMA" in system["content"]
        assert "KAYNAK" in system["content"]

    def test_history_cannot_smuggle_metadata_back_in(self, populated):
        """A caller-supplied history is still subject to nothing else being added."""
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        answerer.answer(
            "Önlisans süresi nedir?",
            top_k=2,
            history_text="[Önceki tur]\nKullanıcı: Daha önce ne sormuştum?",
        )
        prompt = client.last_prompt_text
        assert "ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ" not in prompt


class TestScrubContext:
    @pytest.mark.parametrize(
        "source, gone",
        [
            ("MADDE 6 — (1) Hüküm budur.", "MADDE 6"),
            ("GEÇİCİ MADDE 12/A – Hüküm.", "GEÇİCİ MADDE 12/A"),
            ("EK MADDE 1 - Hüküm.", "EK MADDE 1"),
            ("(Değişik:RG-24/2/2017-29989) Hüküm.", "RG-24/2/2017-29989"),
            ("Resmî Gazete'de yayımlanır.", "Resmî Gazete"),
            ("3/3/2001 tarihinde yürürlüğe girer.", "3/3/2001"),
            ("6446 sayılı Kanun uyarınca.", "6446 sayılı"),
            ("Kanun No. : 4628 olarak kabul edildi.", "Kanun No. : 4628"),
            ("12 nci maddesinin birinci fıkrası.", "12 nci maddesinin"),
            ("ELEKTRİK PİYASASI KANUNU hükümleri.", "ELEKTRİK PİYASASI KANUNU"),
            ("Doğal Gaz Piyasası Kanununun kapsamı.", "Doğal Gaz Piyasası Kanununun"),
            ("Bu Yönetmeliğin amacı.", "Bu Yönetmeliğin"),
            ("18 Nisan 2001 tarihli karar.", "18 Nisan 2001"),
        ],
        ids=lambda s: s[:24],
    )
    def test_identifier_is_removed(self, source, gone):
        assert gone not in scrub_context(source)

    def test_substantive_text_is_preserved(self):
        scrubbed = scrub_context(
            "MADDE 9 – (1) Önlisansın süresi otuz altı ayı geçemez."
        )
        assert "Önlisansın süresi otuz altı ayı geçemez" in scrubbed

    def test_extra_terms_scrub_titles_the_patterns_would_miss(self):
        odd_title = "4. Tarife Uygulama Dönemi Kalite Parametreleri"
        text = f"{odd_title} kapsamında hüküm uygulanır."
        assert odd_title not in scrub_context(text, extra_terms=(odd_title,))
        assert "kapsamında hüküm uygulanır" in scrub_context(text, extra_terms=(odd_title,))

    def test_very_short_extra_terms_are_ignored(self):
        """A 1-3 character 'title' would scrub half the alphabet out of the text."""
        text = "Dağıtım bedeli hesaplanır."
        assert scrub_context(text, extra_terms=("a", "de")) == text

    def test_fikra_references_survive_because_they_are_not_article_numbers(self):
        """"ikinci fıkrası" is what a follow-up question asks about; scrubbing it
        would make the retrieved provision unanswerable."""
        scrubbed = scrub_context("(2) İkinci fıkrada belirtilen süre uygulanır.")
        assert "İkinci fıkrada belirtilen süre" in scrubbed

    def test_runs_of_redactions_collapse(self):
        scrubbed = scrub_context("MADDE 1 MADDE 2 MADDE 3 hüküm.")
        assert "hüküm" in scrubbed
        assert "[…] […] […]" not in scrubbed

    def test_scrubbing_is_idempotent(self):
        once = scrub_context("MADDE 9 – (Değişik:RG-24/2/2017-29989) (1) Süre.")
        assert scrub_context(once) == once


# --------------------------------------------------------------------------
# Label parsing and citation mapping
# --------------------------------------------------------------------------


class TestParseLabels:
    @pytest.mark.parametrize(
        "answer, expected",
        [
            ("KAYNAK 1 uyarınca.", [1]),
            ("[KAYNAK 2] uyarınca.", [2]),
            ("kaynak 3 diyor.", [3]),
            ("KAYNAK 1 ve 2 birlikte.", [1, 2]),
            ("KAYNAK 1, 2 ve 3 uyarınca.", [1, 2, 3]),
            ("KAYNAK 2 ile KAYNAK 1.", [2, 1]),
            ("KAYNAK 1 ve KAYNAK 1 tekrar.", [1]),
            ("Hiçbir atıf yok.", []),
            ("", []),
        ],
        ids=lambda s: str(s)[:26],
    )
    def test_labels_are_extracted_in_first_appearance_order(self, answer, expected):
        assert parse_labels(answer) == expected


class TestCitationMapping:
    def test_labels_map_back_to_real_metadata(self, populated):
        client = StubChatClient(responses=["KAYNAK 2 uyarınca bağlantı bedeli hesaplanır."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Bağlantı bedeli nedir?", top_k=3)

        assert len(result.citations) == 1
        citation = result.citations[0]
        source = result.results[1]  # KAYNAK 2 is the second-ranked chunk
        assert citation.label == 2
        assert citation.document_title == source.document_title
        assert citation.article_ref == source.article_ref
        assert citation.page_start == source.page_start
        assert citation.page_end == source.page_end
        assert citation.source_path == source.source_path
        assert citation.chunk_id == source.chunk_id

    def test_citation_renders_real_identifiers_the_model_never_saw(self, populated):
        client = StubChatClient(responses=["KAYNAK 1 uyarınca."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Önlisans süresi?", top_k=3)
        rendered = result.citations[0].render()
        assert "KAYNAK 1" in rendered
        assert result.results[0].document_title in rendered
        assert result.results[0].article_ref in rendered
        # ... and that title is provably absent from what the model was shown.
        assert result.results[0].document_title not in client.last_prompt_text

    def test_multiple_labels_produce_citations_in_order_of_use(self, populated):
        client = StubChatClient(responses=["KAYNAK 3 ve KAYNAK 1 birlikte değerlendirilir."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert result.cited_labels == [3, 1]

    def test_uncited_sources_produce_no_citation(self, populated):
        client = StubChatClient(responses=["Kaynaklarda bu bilgi yok."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert result.citations == []
        assert result.hallucinated_references == 0

    def test_unsupplied_label_is_dropped_and_counted(self, populated):
        """The model cites KAYNAK 7 when only 1..3 exist."""
        client = StubChatClient(responses=["KAYNAK 7 uyarınca böyledir."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert result.citations == []
        assert result.hallucinated_references == 1
        assert result.hallucinated_labels == [7]

    def test_real_and_hallucinated_labels_are_separated(self, populated):
        client = StubChatClient(responses=["KAYNAK 1 ve KAYNAK 9 uyarınca."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert result.cited_labels == [1]
        assert result.hallucinated_labels == [9]
        assert result.hallucinated_references == 1

    def test_hallucinated_reference_is_logged_not_swallowed(self, populated, caplog):
        client = StubChatClient(responses=["KAYNAK 42 uyarınca."])
        answerer = make_answerer(populated, client, confidence=0.9)
        with caplog.at_level("WARNING", logger="src.answer"):
            answerer.answer("Soru?", top_k=3)
        assert any("unsupplied KAYNAK" in r.getMessage() for r in caplog.records)

    def test_a_label_beyond_the_surviving_blocks_counts_as_hallucinated(self, populated):
        """Budget dropped chunk 3, so KAYNAK 3 was never supplied even though
        three chunks were retrieved."""
        client = StubChatClient(
            responses=["KAYNAK 3 uyarınca."],
            budget=budget_admitting(populated, "Soru?", keep=2),
        )
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert result.chunks_dropped == 1
        assert result.hallucinated_labels == [3]


# --------------------------------------------------------------------------
# Gate integration
# --------------------------------------------------------------------------


class TestGateIntegration:
    def test_not_found_never_calls_the_model(self, populated):
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.01)
        result = answerer.answer("Deniz balıkçılığı av yasağı?", top_k=3)

        assert result.decision == "NOT_FOUND"
        assert client.calls == 0
        assert client.messages_seen == []
        assert result.generated is False
        assert result.text == NOT_FOUND_MESSAGE
        assert result.citations == []

    def test_not_found_message_is_turkish_and_suggests_rephrasing(self):
        assert "bulunamadı" in NOT_FOUND_MESSAGE
        assert "yeniden yazın" in NOT_FOUND_MESSAGE
        assert "doğal gaz" in NOT_FOUND_MESSAGE.lower()

    def test_answer_weak_generates_and_sets_the_flag(self, populated):
        midband = (config.FUSION_THRESHOLD + config.FUSION_FLOOR) / 2
        client = StubChatClient(responses=["KAYNAK 1 uyarınca."])
        answerer = make_answerer(populated, client, confidence=midband)
        result = answerer.answer("Soru?", top_k=3)

        assert result.decision == "ANSWER_WEAK"
        assert result.low_confidence is True
        assert result.generated is True
        assert client.calls == 1
        assert result.citations

    def test_answer_generates_without_the_flag(self, populated):
        client = StubChatClient(responses=["KAYNAK 1 uyarınca."])
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)

        assert result.decision == "ANSWER"
        assert result.low_confidence is False
        assert result.generated is True
        assert client.calls == 1

    def test_exact_floor_is_weak_and_exact_threshold_is_answer(self, populated):
        weak = make_answerer(populated, StubChatClient(), confidence=config.FUSION_FLOOR)
        assert weak.answer("Soru?", top_k=1).low_confidence is True
        strong = make_answerer(
            populated, StubChatClient(), confidence=config.FUSION_THRESHOLD
        )
        assert strong.answer("Soru?", top_k=1).low_confidence is False

    def test_generation_failure_surfaces_turkish_not_a_traceback(self, populated):
        class Exploding(StubChatClient):
            def complete(self, messages, max_completion_tokens=None):
                raise RuntimeError("boom")

        answerer = make_answerer(populated, Exploding(), confidence=0.9)
        result = answerer.answer("Soru?", top_k=1)
        assert result.generated is False
        assert "Foundry Local" in result.text
        assert "dil modeline erişilemedi" in result.text

    def test_context_exhaustion_from_the_server_is_caught(self, populated):
        from src.llm import ContextExhausted

        class Exhausted(StubChatClient):
            def complete(self, messages, max_completion_tokens=None):
                raise ContextExhausted("out of memory")

        answerer = make_answerer(populated, Exhausted(), confidence=0.9)
        result = answerer.answer("Soru?", top_k=1)
        assert result.generated is False
        assert "uzunluğu aştığı için" in result.text


# --------------------------------------------------------------------------
# Context budgeting
# --------------------------------------------------------------------------


class TestContextBudget:
    def test_all_chunks_fit_when_the_budget_is_generous(self, populated):
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert result.chunks_retrieved == 3
        assert result.chunks_dropped == 0
        assert "KAYNAK 3" in client.last_sources_text

    def test_lowest_ranked_chunks_are_dropped_first(self, populated):
        client = StubChatClient(budget=budget_admitting(populated, "Soru?", keep=2))
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)

        assert result.chunks_dropped > 0
        context = client.last_sources_text
        assert "KAYNAK 1" in context, "the best-ranked chunk must always survive"
        # Whatever was dropped, the surviving labels are a prefix of 1..N.
        surviving = [n for n in (1, 2, 3) if f"KAYNAK {n}" in context]
        assert surviving == list(range(1, len(surviving) + 1))
        assert len(surviving) == 3 - result.chunks_dropped

    def test_no_chunk_is_ever_truncated(self, populated):
        """A dropped chunk is invisible; a truncated one is silently misleading."""
        client = StubChatClient(budget=budget_admitting(populated, "Soru?", keep=2))
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        prompt = client.last_prompt_text

        for retrieved in result.results:
            scrubbed = scrub_context(
                retrieved.text,
                extra_terms=(retrieved.document_title or "", retrieved.article_ref or ""),
            )
            # Either the whole scrubbed text is present, or none of its tail is.
            if scrubbed[:40] in prompt:
                assert scrubbed in prompt, "a chunk was included but truncated"

    def test_budget_counts_the_system_prompt_and_history(self, populated):
        """More history must leave room for fewer chunks, not silently overflow."""
        generous = budget_admitting(populated, "Soru?", keep=3)

        bare = StubChatClient(budget=generous)
        without = make_answerer(populated, bare, confidence=0.9).answer("Soru?", top_k=3)
        assert without.chunks_dropped == 0

        loaded = StubChatClient(budget=generous)
        with_history = make_answerer(populated, loaded, confidence=0.9).answer(
            "Soru?",
            top_k=3,
            history_text="önceki soru " * 40,
        )
        assert with_history.chunks_dropped > without.chunks_dropped

    def test_refuses_rather_than_generating_with_no_context(self, populated):
        client = StubChatClient(budget=budget_admitting(populated, "Soru?", keep=0))
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        assert client.calls == 0
        assert result.generated is False
        assert result.chunks_dropped == 3
        assert "uzunluğu aştığı için" in result.text

    @pytest.mark.parametrize("keep", [1, 2, 3])
    def test_dropped_count_matches_what_was_sent(self, populated, keep):
        client = StubChatClient(budget=budget_admitting(populated, "Soru?", keep=keep))
        answerer = make_answerer(populated, client, confidence=0.9)
        result = answerer.answer("Soru?", top_k=3)
        supplied = [n for n in (1, 2, 3) if f"KAYNAK {n}" in client.last_sources_text]
        assert supplied == list(range(1, keep + 1))
        assert result.chunks_dropped == 3 - keep

    def test_not_found_counts_every_chunk_as_dropped(self, populated):
        answerer = make_answerer(populated, StubChatClient(), confidence=0.01)
        result = answerer.answer("Soru?", top_k=3)
        assert result.chunks_dropped == result.chunks_retrieved

    def test_build_source_blocks_keeps_a_prefix(self, populated):
        client = StubChatClient()
        answerer = make_answerer(populated, client, confidence=0.9)
        results = answerer.retriever.retrieve("Soru?", top_k=3)
        blocks, dropped = build_source_blocks(results, client, available_tokens=10_000)
        assert [b.label for b in blocks] == [1, 2, 3]
        assert dropped == 0

        tight, dropped_tight = build_source_blocks(results, client, available_tokens=90)
        assert [b.label for b in tight] == list(range(1, len(tight) + 1))
        assert dropped_tight == 3 - len(tight)


# A conversation recap in the shape Session.history_text() produces.
RECAP = "[Önceki tur]\nKullanıcı: önceki soru\nAsistan: önceki cevap"


class TestBuildMessages:
    def test_shape_is_system_fewshot_then_one_user_turn(self):
        messages = build_messages("Soru?", [], history_text=RECAP)
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1:3] == FEW_SHOT

    def test_history_is_folded_into_the_user_turn_not_sent_as_chat_turns(self):
        """Measured requirement: as real assistant turns, this model stops
        honouring /no_think and buries the answer in an unclosed <think> tag."""
        messages = build_messages("Soru?", [], history_text=RECAP)
        # Exactly one user message after the worked example -- no history turns.
        assert sum(1 for m in messages[3:] if m["role"] == "user") == 1
        tail = messages[-1]["content"]
        assert "[Önceki tur]" in tail
        assert tail.index("[Önceki tur]") < tail.index("SORU: Soru?")

    def test_history_is_omitted_entirely_when_absent(self):
        messages = build_messages("Soru?", [])
        assert messages[-1]["content"] == question_block("Soru?")

    def test_question_is_present_even_with_no_sources(self):
        messages = build_messages("Soru?", [])
        assert messages[-1]["content"] == question_block("Soru?")
        assert "SORU: Soru?" in messages[-1]["content"]

    def test_the_thinking_switch_ends_the_user_turn(self):
        """Qwen3 parses /no_think per turn; in the system prompt alone it is
        ignored on a full-length RAG prompt."""
        messages = build_messages("Soru?", [])
        assert messages[-1]["content"].rstrip().endswith(NO_THINK)

    def test_preamble_is_what_build_messages_actually_sends(self):
        """Budgeting subtracts preamble_messages(); if the two ever diverge the
        budget silently under-counts and the server OOMs."""
        messages = build_messages("Soru?", [], history_text="önceki")
        assert messages[:-1] == preamble_messages()

    def test_format_constraints_are_repeated_next_to_the_question(self):
        """Stated only in the system prompt they were measurably ignored; the
        copy that actually changes the model's output is this one."""
        messages = build_messages("Soru?", [])
        tail = messages[-1]["content"]
        assert ANSWER_INSTRUCTION in tail
        # It must come after the question, not before it.
        assert tail.index("SORU: Soru?") < tail.index(ANSWER_INSTRUCTION)
        assert "(KAYNAK 1)" in ANSWER_INSTRUCTION

    def test_the_example_demonstrates_inline_citation(self):
        """The one-shot exists to teach the (KAYNAK n) form; assert it does."""
        assert "(KAYNAK 1)" in FEW_SHOT[1]["content"]
        assert "(KAYNAK 2)" in FEW_SHOT[1]["content"]
        # ... in plain prose, with none of the formatting the rules forbid.
        assert "**" not in FEW_SHOT[1]["content"]
        assert "Cevap:" not in FEW_SHOT[1]["content"]


# --------------------------------------------------------------------------
# llm.py helpers
# --------------------------------------------------------------------------


class TestStripLeadingHeading:
    """Presentation-only cleanup: it must never remove content or a citation."""

    def test_a_bold_title_line_is_dropped(self):
        assert strip_leading_heading("**Serbest Tüketici**\n\nCevap metni.") == "Cevap metni."

    def test_a_markdown_heading_is_dropped(self):
        assert strip_leading_heading("### Cevap\n\nMetin burada.") == "Metin burada."

    def test_a_single_line_answer_is_never_touched(self):
        assert strip_leading_heading("**Tek satır cevap.**") == "**Tek satır cevap.**"

    def test_a_real_first_sentence_is_kept(self):
        answer = "Önlisans süresi otuz altı aydır (KAYNAK 1).\nİkinci cümle."
        assert strip_leading_heading(answer) == answer

    def test_a_heading_carrying_a_citation_is_kept(self):
        """If a label is in there, dropping the line would drop a citation."""
        answer = "**KAYNAK 1 uyarınca**\n\nDevam eden metin."
        assert strip_leading_heading(answer) == answer

    def test_plain_prose_is_unchanged(self):
        answer = "Birinci cümle. İkinci cümle."
        assert strip_leading_heading(answer) == answer


class TestStripThinking:
    def test_think_block_is_removed(self):
        assert strip_thinking("<think>uzun düşünce</think>\n\nCevap.") == "Cevap."

    def test_empty_think_block_from_no_think_is_removed(self):
        assert strip_thinking("<think>\n\n</think>\n\nCevap.") == "Cevap."

    def test_unterminated_think_block_yields_no_reasoning_leak(self):
        # Generation cut off mid-thought by the completion budget.
        assert strip_thinking("Kısmi cevap.<think>yarım düşünce") == "Kısmi cevap."

    def test_an_unclosed_but_empty_block_keeps_the_answer(self):
        """The observed multi-turn artifact: an opened-and-forgotten tag with the
        real answer after it. Discarding this threw away a cited answer."""
        raw = "<think>\n\n**Cevap:**\nSüre otuz altı aydır. (KAYNAK 1)"
        assert strip_thinking(raw) == "**Cevap:**\nSüre otuz altı aydır. (KAYNAK 1)"

    def test_an_unclosed_block_with_real_reasoning_is_still_discarded(self):
        """No blank line after the tag means the model was actually thinking;
        showing that to a user is worse than showing nothing."""
        raw = "<think>\nOkay, the user is asking about the maximum duration"
        assert strip_thinking(raw) == ""

    def test_text_without_a_block_is_untouched(self):
        assert strip_thinking("  Cevap.  ") == "Cevap."


class TestTokenCounter:
    def test_estimate_mode_is_deterministic_and_positive(self):
        counter = TokenCounter()
        assert counter.exact is False
        assert counter.count("bir iki üç") == counter.count("bir iki üç") > 0

    def test_message_framing_is_included(self):
        counter = TokenCounter()
        messages = [{"role": "user", "content": "bir iki üç"}]
        assert counter.count_messages(messages) == (
            counter.count("bir iki üç") + config.TOKENS_PER_MESSAGE
        )

    def test_empty_text_costs_nothing_but_framing(self):
        counter = TokenCounter()
        assert counter.count_messages([{"role": "user", "content": ""}]) >= (
            config.TOKENS_PER_MESSAGE
        )


class TestConfiguredContextBudget:
    def test_effective_context_is_below_the_declared_window(self):
        """The measured VRAM ceiling on this hardware, not the model's claim."""
        assert config.CHAT_EFFECTIVE_CONTEXT < config.CHAT_CONTEXT_WINDOW

    def test_prompt_budget_leaves_room_for_the_answer_and_a_margin(self):
        budget = (
            config.CHAT_EFFECTIVE_CONTEXT
            - config.CHAT_MAX_COMPLETION_TOKENS
            - config.CONTEXT_SAFETY_MARGIN
        )
        assert budget > 0
        assert budget < config.CHAT_EFFECTIVE_CONTEXT

    def test_temperature_is_pinned_at_zero(self):
        """Regulatory answers must be reproducible; sampling variance is a defect."""
        assert config.CHAT_TEMPERATURE == 0.0
