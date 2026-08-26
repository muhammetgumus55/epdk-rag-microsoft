"""Tests for multi-turn sessions: history eviction, follow-up detection, rewriting.

The question these tests exist to answer is narrow and load-bearing: does the
string that reaches retrieve() actually stand on its own? "peki ya ikinci
fıkrası?" embeds to nothing useful -- every word that carries the topic is in the
previous turn. If the rewrite silently no-ops, retrieval degrades in a way that
looks like an embedding problem and is not one, which is exactly why
Session.ask() logs the original and the rewritten query as a pair.

The chat client is stubbed, so what is verified here is the plumbing and the
decision logic: that a follow-up is rewritten before retrieval, that a fresh
question is not, that failures fall back to the question as typed, and that
history is bounded. Whether the local model writes a *good* rewrite is checked
against the real model in the CLI, not here.
"""
import pytest

from src import config
from src.session import (
    FRESH_SENTINEL,
    REWRITE_FEW_SHOT,
    REWRITE_INSTRUCTION,
    REWRITE_SYSTEM_PROMPT,
    Session,
    Turn,
    looks_like_follow_up,
)

from tests.test_answer import MEVZUAT_CHUNKS, StubChatClient, make_answerer
from tests.test_retrieval import make_chunk, unit
from src import store


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "session.db")
    yield c
    c.close()


@pytest.fixture
def populated(conn):
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


def make_session(conn, responses=None, confidence=0.9, **kwargs):
    """A Session whose chat client returns `responses` in order.

    Responses are consumed by both the rewrite call and the generation call, in
    whatever order Session.ask() makes them -- which is itself part of what the
    tests below pin down.
    """
    client = StubChatClient(responses=responses or [])
    answerer = make_answerer(conn, client, confidence=confidence)
    return Session(answerer=answerer, **kwargs), client


def spy_on_retrieval(session) -> list[str]:
    """Record every query string that actually reaches the retriever."""
    queries: list[str] = []
    original = session.answerer.retriever.retrieve_fused_timed

    def wrapper(query, top_k=None):
        queries.append(query)
        return original(query, top_k)

    session.answerer.retriever.retrieve_fused_timed = wrapper  # type: ignore[assignment]
    return queries


# --------------------------------------------------------------------------
# The follow-up case this module exists for
# --------------------------------------------------------------------------

FOLLOW_UP = "peki ya ikinci fıkrası?"
REWRITTEN = "Önlisans süresine ilişkin hükmün ikinci fıkrası nedir?"


class TestFollowUpRewriting:
    def test_a_meaningless_followup_is_rewritten_before_retrieval(self, populated):
        """The core case: "peki ya ikinci fıkrası?" cannot be retrieved on.

        Standalone it contains no topic at all -- only a discourse particle
        ("peki ya") and a structural reference ("ikinci fıkrası"). What reaches
        the retriever must be the resolved question instead.
        """
        session, client = make_session(
            populated,
            responses=[
                "KAYNAK 1 uyarınca önlisans süresi otuz altı ayı geçemez.",  # turn 1
                REWRITTEN,                                                    # rewrite
                "KAYNAK 1 uyarınca ikinci fıkra süre uzatımını düzenler.",   # turn 2
            ],
        )
        queries = spy_on_retrieval(session)

        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask(FOLLOW_UP)

        # Turn 1 retrieved the question as typed; turn 2 retrieved the rewrite.
        assert queries[0] == "Önlisans süresi ne kadardır?"
        assert queries[1] == REWRITTEN
        assert queries[1] != FOLLOW_UP

        # The user still sees what they asked; the rewrite is recorded alongside.
        assert answer.question == FOLLOW_UP
        assert answer.rewritten_query == REWRITTEN

    def test_the_rewritten_query_is_self_contained(self, populated):
        """Self-contained means: carries its own topic, needs no prior turn.

        Asserted structurally rather than by comparing to a fixed string -- the
        properties that make a query retrievable are that the topic words are
        present and the dangling references are gone.
        """
        session, _ = make_session(
            populated,
            responses=[
                "KAYNAK 1 uyarınca önlisans süresi otuz altı ayı geçemez.",
                REWRITTEN,
                "KAYNAK 1 uyarınca.",
            ],
        )
        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask(FOLLOW_UP)
        query = answer.rewritten_query

        # The topic is present in the rewrite and absent from the raw follow-up.
        assert "önlisans" in query.lower()
        assert "önlisans" not in FOLLOW_UP.lower()
        # The discourse particle that made it dependent is gone.
        assert "peki" not in query.lower()
        # It is still a question, and it is longer than the fragment it replaced.
        assert query.rstrip().endswith("?")
        assert len(query) > len(FOLLOW_UP)

    def test_the_rewriter_is_shown_the_prior_turns(self, populated):
        session, client = make_session(
            populated,
            responses=[
                "KAYNAK 1 uyarınca önlisans süresi otuz altı ayı geçemez.",
                REWRITTEN,
                "KAYNAK 1 uyarınca.",
            ],
        )
        session.ask("Önlisans süresi ne kadardır?")
        session.ask(FOLLOW_UP)

        # messages_seen[1] is the rewrite call (0 is turn 1's generation).
        rewrite_call = client.messages_seen[1]
        assert rewrite_call[0]["content"] == REWRITE_SYSTEM_PROMPT
        assert rewrite_call[1:-1] == REWRITE_FEW_SHOT
        payload = rewrite_call[-1]["content"]
        assert "Önlisans süresi ne kadardır?" in payload
        assert "otuz altı" in payload
        assert f"SON SORU: {FOLLOW_UP}" in payload
        # The instruction comes after the question, where this model attends to it.
        assert payload.index("SON SORU:") < payload.index(REWRITE_INSTRUCTION)

    def test_the_examples_only_ever_demonstrate_rewriting(self):
        """The fresh/follow-up call is made in code before the model is asked, so
        the examples must not teach the YENİ branch -- demonstrating it made the
        model answer YENİ to a plainly anaphoric question."""
        replies = [m["content"] for m in REWRITE_FEW_SHOT if m["role"] == "assistant"]
        assert replies, "there must be worked examples"
        assert FRESH_SENTINEL not in replies
        assert all(r.rstrip().endswith("?") for r in replies)

    def test_one_example_covers_the_grammatically_complete_anaphor(self):
        """The failing real case was "peki bu <noun> ... gerekir mi?" -- a full
        sentence whose subject is a bare demonstrative."""
        prompts = [m["content"] for m in REWRITE_FEW_SHOT if m["role"] == "user"]
        assert any("peki bu " in p for p in prompts)

    def test_the_rewriter_is_told_not_to_answer(self):
        assert "CEVAPLAMA" in REWRITE_SYSTEM_PROMPT
        assert "cevaplama" in REWRITE_INSTRUCTION.lower()

    def test_fresh_topic_sentinel_leaves_the_question_untouched(self, populated):
        session, _ = make_session(
            populated,
            responses=[
                "KAYNAK 1 uyarınca.",
                FRESH_SENTINEL,          # rewriter says: stands alone
                "KAYNAK 2 uyarınca.",
            ],
        )
        queries = spy_on_retrieval(session)
        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask("Dağıtım bağlantı bedeli nasıl hesaplanır?")

        assert queries[1] == "Dağıtım bağlantı bedeli nasıl hesaplanır?"
        assert answer.rewritten_query == "Dağıtım bağlantı bedeli nasıl hesaplanır?"

    @pytest.mark.parametrize("sentinel", ["YENİ", "YENI", "yeni", "Yeni.", "YENİ SORU"])
    def test_sentinel_is_recognised_in_the_forms_a_model_actually_emits(
        self, populated, sentinel
    ):
        session, _ = make_session(
            populated, responses=["KAYNAK 1 uyarınca.", sentinel, "KAYNAK 1 uyarınca."]
        )
        session.ask("İlk soru nedir?")
        answer = session.ask("Bağımsız ikinci soru nedir?")
        assert answer.rewritten_query == "Bağımsız ikinci soru nedir?"

    def test_the_first_question_never_costs_a_rewrite_call(self, populated):
        """With no prior turns there is nothing to resolve."""
        session, client = make_session(populated, responses=["KAYNAK 1 uyarınca."])
        answer = session.ask("Önlisans süresi ne kadardır?")
        assert client.calls == 1  # generation only
        assert answer.rewritten_query == "Önlisans süresi ne kadardır?"
        assert session.turns[0].was_follow_up is False

    def test_rewrite_output_is_cleaned_of_prefixes_and_quotes(self, populated):
        session, _ = make_session(
            populated,
            responses=[
                "KAYNAK 1 uyarınca.",
                'SORU: "Önlisans süresinin ikinci fıkrası nedir?"',
                "KAYNAK 1 uyarınca.",
            ],
        )
        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask(FOLLOW_UP)
        assert answer.rewritten_query == "Önlisans süresinin ikinci fıkrası nedir?"

    def test_a_failed_rewrite_falls_back_to_the_question_as_typed(self, populated):
        class Failing(StubChatClient):
            def complete(self, messages, max_completion_tokens=None):
                if messages[0]["content"] == REWRITE_SYSTEM_PROMPT:
                    raise RuntimeError("rewrite exploded")
                return super().complete(messages, max_completion_tokens)

        client = Failing(default="KAYNAK 1 uyarınca.")
        session = Session(answerer=make_answerer(populated, client, confidence=0.9))
        queries = spy_on_retrieval(session)

        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask(FOLLOW_UP)

        # The question is not lost when the rewrite fails.
        assert queries[1] == FOLLOW_UP
        assert answer.rewritten_query == FOLLOW_UP

    def test_a_commentary_shaped_rewrite_is_rejected(self, populated):
        """A model that explains itself instead of rewriting must not poison retrieval."""
        commentary = (
            "Bu soru önceki turdaki önlisans süresi konusuna atıf yapıyor, bu nedenle "
            "sorunun yeniden yazılması gerekir ve aşağıdaki gibi ifade edilebilir; "
            "ancak bağlamın tamamını korumak için ek açıklama da eklenmiştir."
        )
        session, _ = make_session(
            populated,
            responses=["KAYNAK 1 uyarınca.", commentary, "KAYNAK 1 uyarınca."],
        )
        queries = spy_on_retrieval(session)
        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask(FOLLOW_UP)
        assert queries[1] == FOLLOW_UP
        assert answer.rewritten_query == FOLLOW_UP

    def test_an_empty_rewrite_falls_back(self, populated):
        session, _ = make_session(
            populated, responses=["KAYNAK 1 uyarınca.", "   ", "KAYNAK 1 uyarınca."]
        )
        session.ask("Önlisans süresi ne kadardır?")
        answer = session.ask(FOLLOW_UP)
        assert answer.rewritten_query == FOLLOW_UP

    def test_both_queries_are_logged(self, populated, caplog):
        session, _ = make_session(
            populated,
            responses=["KAYNAK 1 uyarınca.", REWRITTEN, "KAYNAK 1 uyarınca."],
        )
        session.ask("Önlisans süresi ne kadardır?")
        with caplog.at_level("INFO", logger="src.session"):
            session.ask(FOLLOW_UP)
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert FOLLOW_UP in logged, "the original question must be recoverable from logs"
        assert REWRITTEN in logged, "so must the query retrieval actually ran on"
        assert "follow_up=True" in logged


# --------------------------------------------------------------------------
# History bounds
# --------------------------------------------------------------------------


class TestFollowUpDetection:
    """The fresh-vs-follow-up decision, made lexically so no model can fumble it."""

    @pytest.mark.parametrize(
        "question",
        [
            "peki ya ikinci fıkrası?",
            "peki bu faaliyet için ayrıca izin alınması gerekir mi?",
            "bu süre ne kadar?",
            "bunun istisnası var mı?",
            "aynı kural burada da geçerli mi?",
            "söz konusu bedel nasıl hesaplanır?",
            "onun için de geçerli mi?",
            "Süresi ne kadar?",
            "Kimler başvurabilir?",
        ],
        ids=lambda s: s[:34],
    )
    def test_dependent_questions_are_detected(self, question):
        assert looks_like_follow_up(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "Elektrik enerjisi ithalat ve ihracat faaliyeti için hangi lisans gereklidir?",
            "Dağıtım bağlantı bedeli nasıl hesaplanır?",
            "Yan hizmetler kapsamında primer frekans kontrol hizmeti nasıl tedarik edilir?",
            "Serbest tüketici tedarikçisini nasıl değiştirir?",
            "Önlisans süresi en fazla ne kadar olabilir?",
        ],
        ids=lambda s: s[:34],
    )
    def test_standalone_questions_are_not_flagged(self, question):
        assert looks_like_follow_up(question) is False

    def test_detection_is_diacritic_insensitive(self):
        """ASCII-typed input must behave identically -- the same property the
        BM25 tokenizer provides, reused here rather than reimplemented."""
        assert looks_like_follow_up("peki bu sure ne kadar?") is True
        assert looks_like_follow_up("peki bu süre ne kadar?") is True

    def test_an_empty_question_is_not_a_follow_up(self):
        assert looks_like_follow_up("") is False
        assert looks_like_follow_up("???") is False

    def test_a_standalone_question_never_reaches_the_rewriter(self, populated):
        """The saving is not only latency: a rewrite of a question that needs
        none can only import the previous topic into it."""
        session, client = make_session(populated, responses=[])
        session.ask("Önlisans süresi ne kadardır?")
        calls_after_first = client.calls
        session.ask("Yan hizmetler kapsamında primer frekans kontrol nasıl tedarik edilir?")
        # Exactly one more call: generation. No rewrite call was made.
        assert client.calls == calls_after_first + 1

    def test_a_dependent_question_does_reach_the_rewriter(self, populated):
        session, client = make_session(populated, responses=[])
        session.ask("Önlisans süresi ne kadardır?")
        calls_after_first = client.calls
        session.ask("peki bu süre uzatılabilir mi?")
        # Two more calls: the rewrite, then generation.
        assert client.calls == calls_after_first + 2


class TestTurnEviction:
    def test_at_most_three_turns_are_kept(self, populated):
        session, _ = make_session(populated, responses=[])
        # Every call after the first also spends a rewrite response; the stub's
        # default covers both, and FRESH_SENTINEL is not returned so each is
        # treated as a follow-up -- irrelevant to eviction, which counts turns.
        for i in range(6):
            session.ask(f"Soru {i} nedir?")
        assert len(session.turns) == config.SESSION_MAX_TURNS == 3

    def test_the_oldest_turns_are_the_ones_evicted(self, populated):
        session, _ = make_session(populated, responses=[])
        for i in range(5):
            session.ask(f"Soru {i} nedir?")
        assert [t.question for t in session.turns] == [
            "Soru 2 nedir?", "Soru 3 nedir?", "Soru 4 nedir?",
        ]

    def test_the_cap_is_configurable_and_respected(self, populated):
        session, _ = make_session(populated, responses=[], max_turns=2)
        for i in range(4):
            session.ask(f"Soru {i} nedir?")
        assert [t.question for t in session.turns] == ["Soru 2 nedir?", "Soru 3 nedir?"]

    def test_history_recaps_each_turn_as_labelled_text(self, populated):
        session, _ = make_session(populated, responses=[])
        session.ask("Birinci soru nedir?")
        session.ask("İkinci soru nedir?")
        recap = session.history_text()
        assert recap.startswith("[Önceki tur]")
        assert recap.count("Kullanıcı:") == 2
        assert recap.count("Asistan:") == 2

    def test_history_uses_the_rewritten_query_not_the_raw_followup(self, populated):
        session, _ = make_session(
            populated,
            responses=["KAYNAK 1 uyarınca.", REWRITTEN, "KAYNAK 1 uyarınca."],
        )
        session.ask("Önlisans süresi ne kadardır?")
        session.ask(FOLLOW_UP)
        recap = session.history_text()
        assert f"Kullanıcı: {REWRITTEN}" in recap
        assert FOLLOW_UP not in recap

    def test_history_is_none_before_the_first_turn(self, populated):
        """None, not an empty string: the prompt must gain nothing at all."""
        session, _ = make_session(populated)
        assert session.history_text() is None


class TestHistoryHygiene:
    def test_stale_kaynak_labels_are_stripped_from_history(self, populated):
        """Labels are re-assigned per retrieval: turn 2's KAYNAK 1 is a different
        chunk than turn 1's. Carrying the old label forward invites a citation
        that means something else now."""
        session, client = make_session(
            populated,
            responses=[
                "KAYNAK 1 ve KAYNAK 2 uyarınca süre otuz altı aydır.",
                FRESH_SENTINEL,
                "KAYNAK 1 uyarınca.",
            ],
        )
        session.ask("Önlisans süresi ne kadardır?")
        session.ask("Bağlantı bedeli nedir?")

        recap = session.history_text()
        assert "Asistan:" in recap, "history should carry the prior answer"
        assert "KAYNAK" not in recap
        # The substance of the prior answer survives the stripping.
        assert "otuz altı" in recap

    def test_a_long_history_answer_is_condensed(self, populated):
        session, _ = make_session(populated, responses=[])
        long_answer = " ".join(f"Bu {i}. cümledir ve uzun bir metnin parçasıdır." for i in range(80))
        session.turns.append(
            Turn(question="s", rewritten_query="s", answer=long_answer, was_follow_up=False)
        )
        condensed = session._condense(long_answer)
        counter = session.answerer.client.counter
        assert counter.count(condensed) <= session.history_answer_tokens
        assert len(condensed) < len(long_answer)

    def test_condensing_cuts_on_a_sentence_boundary(self, populated):
        session, _ = make_session(populated, responses=[])
        text = " ".join(f"Cümle {i} burada bitiyor." for i in range(60))
        condensed = session._condense(text)
        assert condensed.endswith(".") or condensed.endswith("…")

    def test_a_short_answer_is_left_alone(self, populated):
        session, _ = make_session(populated, responses=[])
        assert session._condense("Kısa cevap.") == "Kısa cevap."


class TestSessionAnswerPassthrough:
    def test_gate_decisions_still_reach_the_caller(self, populated):
        session, client = make_session(populated, responses=[], confidence=0.01)
        answer = session.ask("Alakasız bir soru nedir?")
        assert answer.decision == "NOT_FOUND"
        assert client.calls == 0, "NOT_FOUND must skip generation inside a session too"

    def test_low_confidence_flag_survives_a_session(self, populated):
        midband = (config.FUSION_THRESHOLD + config.FUSION_FLOOR) / 2
        session, _ = make_session(populated, responses=[], confidence=midband)
        assert session.ask("Soru nedir?").low_confidence is True

    def test_citations_survive_a_session(self, populated):
        session, _ = make_session(populated, responses=["KAYNAK 2 uyarınca."])
        answer = session.ask("Bağlantı bedeli nedir?")
        assert answer.cited_labels == [2]
        # Asserted positionally: which document ranks second is a retrieval
        # outcome, while KAYNAK 2 resolving to the second-ranked chunk's real
        # metadata is the mapping property under test.
        assert answer.citations[0].document_title == answer.results[1].document_title
        assert answer.citations[0].chunk_id == answer.results[1].chunk_id

    def test_a_not_found_turn_is_still_recorded_in_history(self, populated):
        session, _ = make_session(populated, responses=[], confidence=0.01)
        session.ask("Alakasız soru nedir?")
        assert len(session.turns) == 1
