"""Tests for Reciprocal Rank Fusion and the three failure classes it is measured against.

Two kinds of test live here:

1. TestReciprocalRankFusion -- pure ordering tests on a synthetic ranked list,
   independent of any corpus. reciprocal_rank_fusion() is a rank-position
   function; its correctness does not depend on real embeddings or BM25 data.

2. The regression classes -- pin the gate DECISION for specific measured
   questions to the fusion-confidence scores scripts/calibrate_gate.py actually
   produced. They do not re-run retrieval themselves -- that would make the test
   suite depend on a live Foundry Local server and the full corpus -- they pin
   gate() against the measured scores, the same pattern TestGateBoundaries in
   test_retrieval.py already uses. If a future change to fusion_confidence(),
   the BM25 tokenizer, or the calibrated cutoffs shifts one of these decisions,
   the test that catches it is here, not a person rereading a config.py comment.

   TestDomainMismatchRegression   neighbouring energy domains (doğal gaz, petrol)
   TestDiacriticFlipRegression    as-typed vs ASCII-folded agreement
   TestOutOfScopeQuestionRegression / TestOutOfDomainSweep
                                  questions from unrelated bodies of law

All scores are from the 2026-08-29 calibration run against the filtered
26,441-chunk corpus, with FUSION_THRESHOLD=0.32979 / FUSION_FLOOR=0.18804 as
recorded in config.py. Both inputs changed on that date -- the corpus lost 606
out-of-scope chunks and the cutoffs were rederived on a 21-question negative
class -- so scores here were re-measured rather than carried over from the
2026-08-26 run.
"""
import pytest

from src import config
from src.retrieval import gate, reciprocal_rank_fusion

# --------------------------------------------------------------------------
# Reciprocal Rank Fusion: pure ordering, synthetic ranked lists
# --------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_agreement_between_rankers_beats_a_single_ranker_first_place(self):
        # doc 2 is #2 in both lists; doc 1 is #1 in only one. RRF should still
        # prefer the doc both rankers vouch for once it's this close to the top.
        dense = [1, 2, 3]
        lexical = [4, 2, 5]
        fused = reciprocal_rank_fusion([dense, lexical])
        assert fused[0][0] == 2

    def test_top_of_both_rankings_wins_outright(self):
        dense = [1, 2, 3]
        lexical = [1, 4, 5]
        fused = reciprocal_rank_fusion([dense, lexical])
        assert fused[0][0] == 1

    def test_score_is_sum_of_reciprocal_ranks(self):
        fused = dict(reciprocal_rank_fusion([[1, 2], [2, 1]], k=10))
        # doc 1: rank 1 in list A (1/11), rank 2 in list B (1/12)
        # doc 2: rank 2 in list A (1/12), rank 1 in list B (1/11)
        expected = 1 / 11 + 1 / 12
        assert fused[1] == pytest.approx(expected)
        assert fused[2] == pytest.approx(expected)
        assert fused[1] == pytest.approx(fused[2])  # symmetric by construction

    def test_a_doc_missing_from_one_ranker_only_scores_from_the_other(self):
        fused = dict(reciprocal_rank_fusion([[1, 2], [3]], k=10))
        assert fused[1] == pytest.approx(1 / 11)
        assert 3 in fused and 2 in fused
        assert fused[3] == pytest.approx(1 / 11)

    def test_larger_k_flattens_the_influence_of_rank_one(self):
        # With small k, being #1 vs #2 in one list matters a lot relative to
        # showing up in the other list at all; with large k it matters less.
        rankings = [[1, 2, 3], [4, 5, 6]]
        small_k = dict(reciprocal_rank_fusion(rankings, k=1))
        large_k = dict(reciprocal_rank_fusion(rankings, k=1000))
        gap_small = small_k[1] - small_k[2]
        gap_large = large_k[1] - large_k[2]
        assert gap_small > gap_large > 0

    def test_default_k_comes_from_config(self):
        rankings = [[1, 2], [2, 1]]
        default = dict(reciprocal_rank_fusion(rankings))
        explicit = dict(reciprocal_rank_fusion(rankings, k=config.RRF_K))
        assert default == explicit

    def test_ties_are_broken_by_id_for_determinism(self):
        # Two docs that never co-occur end up with identical RRF scores;
        # the sort must still be deterministic rather than dict-order-dependent.
        fused = reciprocal_rank_fusion([[5, 9], [9, 5]])
        scores = {doc_id: score for doc_id, score in fused}
        assert scores[5] == scores[9]
        tied = [doc_id for doc_id, score in fused if score == scores[5]]
        assert tied == sorted(tied)

    def test_results_are_sorted_descending_by_score(self):
        fused = reciprocal_rank_fusion([[1, 2, 3, 4], [4, 3, 2, 1]])
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_empty_rankings_produce_no_results(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_a_single_ranking_reduces_to_its_own_order(self):
        fused = reciprocal_rank_fusion([[7, 3, 9]])
        assert [doc_id for doc_id, _ in fused] == [7, 3, 9]

    def test_raw_score_magnitude_in_the_input_is_irrelevant(self):
        # reciprocal_rank_fusion() takes plain id lists -- there is no score
        # channel to smuggle magnitude through. Only position can matter.
        by_position = reciprocal_rank_fusion([[1, 2, 3]])
        assert [doc_id for doc_id, _ in by_position] == [1, 2, 3]

    def test_three_rankers_fuse_the_same_way_as_two(self):
        fused = reciprocal_rank_fusion([[1, 2], [1, 2], [2, 1]])
        scores = dict(fused)
        # doc 1: ranks 1,1,2 vs doc 2: ranks 2,2,1 -- doc 1 has two firsts.
        assert scores[1] > scores[2]


# --------------------------------------------------------------------------
# Failure class 1: domain-vocabulary mismatch (Step 4 -> Step 5)
# --------------------------------------------------------------------------


class TestDomainMismatchRegression:
    def test_dogal_gaz_question_still_answers_wrong_known_limitation(self):
        """Known limitation, NOT a target to silently start passing.

        "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl
        hesaplanır?" retrieves the *electricity* Dağıtım Bağlantı Bedelleri
        chunk at fusion confidence 0.50657 -- comfortably above THRESHOLD --
        because "doğal" and "gaz" both genuinely occur in this electricity-only
        corpus (the Electricity Market Law references natural gas), so IDF
        coverage stays high (0.796) even though the question is about a
        neighbouring energy domain this corpus does not cover. See config.py's
        FUSION_THRESHOLD comment ("STILL UNFIXED") for the full analysis --
        fixing this needs a query-side domain gate or a reranker, not a cutoff
        change. The 2026-08-29 corpus scope filter did NOT help here, and could
        not have: the retrieved chunk is genuine electricity law, the same shape
        of problem as kıdem tazminatı. This test exists to CATCH a future
        accidental fix (or regression), not to assert the bug is fine forever:
        if this ever flips, update the comment and the config.py known-limitation
        note together, don't just delete the test.
        """
        assert gate(0.50657) == "ANSWER"

    def test_rafinerici_question_improved_but_is_not_rejected_outright(self):
        """The related domain-mismatch question DID improve under fusion.

        Step 4 dense-only: 0.5536 -> ANSWER_WEAK. Fused: 0.21310, still
        ANSWER_WEAK (not the ANSWER a raw cosine gave partial credit for), but
        also not below FLOOR -- BM25 coverage for "rafinerici / ulusal petrol
        stoku" against an electricity corpus is low (0.385) but not zero.
        """
        assert gate(0.21310) == "ANSWER_WEAK"


# --------------------------------------------------------------------------
# Failure class 2: diacritic sensitivity (Step 4's 8 flipped questions)
# --------------------------------------------------------------------------
#
# Each entry is (question, fused_typed, decision, fused_folded, decision).
# Values are the exact fusion-confidence scores from the 2026-08-29
# calibrate_gate.py rerun -- re-measured, not carried over, because BOTH inputs
# changed since the 2026-08-26 run: the corpus lost 606 out-of-scope chunks
# (docs/decisions/2026-08-29-omnibus-scope-filter.md) and the cutoffs were
# recalibrated on a 21-question negative class.
#
# 6 of these 8 now agree between spellings, down from 7. The two that regressed
# straddle the raised FUSION_THRESHOLD (0.23963 -> 0.32979) rather than having
# moved themselves, and both now disagree as ANSWER vs ANSWER_WEAK -- both
# spellings still produce an answer, where Step 4's disagreements crossed into
# NOT_FOUND. Meanwhile "Akaryakıt bayilik lisansı" -- the one that still
# disagreed in Step 5 -- now agrees, as NOT_FOUND both ways.

DIACRITIC_FLIP_CASES = [
    (
        "Gün öncesi piyasasında teklif verme ve eşleştirme süreci nasıl işler?",
        0.41094, "ANSWER", 0.35339, "ANSWER",
    ),
    (
        "Lisanssız elektrik üretiminde çatı tipi güneş enerjisi santralleri "
        "için kurulu güç sınırı nedir?",
        0.24805, "ANSWER_WEAK", 0.25037, "ANSWER_WEAK",
    ),
    (
        "Dağıtım şirketinin tüketiciye planlı kesinti öncesinde bildirim "
        "yapma yükümlülüğü nedir?",
        0.23959, "ANSWER_WEAK", 0.40313, "ANSWER",
    ),
    (
        "Bağlantı anlaşması hangi hallerde sona erer veya feshedilir?",
        0.51015, "ANSWER", 0.29426, "ANSWER_WEAK",
    ),
    (
        "Sayaçların okunması ve tüketim değerlerinin belirlenmesine ilişkin "
        "usul ve esaslar nelerdir?",
        0.36537, "ANSWER", 0.33440, "ANSWER",
    ),
    (
        "Dağıtım tarifesinin düzenlenmesinde gelir tavanı nasıl belirlenir?",
        0.36438, "ANSWER", 0.38245, "ANSWER",
    ),
    (
        # Same question as the domain-mismatch test above, this time checked
        # for typed/folded agreement rather than for the ANSWER-vs-correct
        # question. Both facts are true of it at once.
        "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl "
        "hesaplanır?",
        0.50657, "ANSWER", 0.46844, "ANSWER",
    ),
    (
        "Akaryakıt bayilik lisansı için aranan asgari sermaye şartı nedir?",
        0.17950, "NOT_FOUND", 0.16116, "NOT_FOUND",
    ),
]


class TestDiacriticFlipRegression:
    @pytest.mark.parametrize(
        "question, fused_typed, decision_typed, fused_folded, decision_folded",
        DIACRITIC_FLIP_CASES,
        ids=[c[0][:40] for c in DIACRITIC_FLIP_CASES],
    )
    def test_pinned_to_current_measured_decision(
        self, question, fused_typed, decision_typed, fused_folded, decision_folded
    ):
        assert gate(fused_typed) == decision_typed
        assert gate(fused_folded) == decision_folded

    def test_six_of_eight_now_agree_between_spellings(self):
        agreeing = sum(
            1
            for _, fused_typed, dt, fused_folded, df in DIACRITIC_FLIP_CASES
            if gate(fused_typed) == gate(fused_folded)
        )
        assert agreeing == 6

    def test_akaryakit_bayilik_now_agrees_as_not_found(self):
        """Step 5's one remaining disagreement is resolved by the new cutoffs."""
        question, fused_typed, _, fused_folded, _ = DIACRITIC_FLIP_CASES[-1]
        assert question.startswith("Akaryakıt bayilik lisansı")
        assert gate(fused_typed) == gate(fused_folded) == "NOT_FOUND"

    def test_remaining_disagreements_never_cross_into_refusal(self):
        """The two regressions cost confidence, not answers.

        Raising FUSION_THRESHOLD made two questions straddle it. Both spellings
        still answer -- ANSWER vs ANSWER_WEAK -- whereas Step 4's diacritic
        disagreements crossed ANSWER -> NOT_FOUND and lost the answer outright.
        If a future change turns one of these into a NOT_FOUND on either
        spelling, that is a real regression and this test is what catches it.
        """
        for question, typed, _, folded, _ in DIACRITIC_FLIP_CASES:
            decisions = {gate(typed), gate(folded)}
            if len(decisions) == 1:
                continue
            assert "NOT_FOUND" not in decisions, question


# --------------------------------------------------------------------------
# Failure class 3: out-of-scope questions answered from the corpus (2026-08-29)
# --------------------------------------------------------------------------
#
# Three questions with no connection to electricity regulation were answered
# from real chunks of the live corpus. They had two different causes, and the
# tests below pin both the fixes and the one that is NOT fixed:
#
#   trafik cezası / vergi levhası -- retrieved omnibus-act articles amending the
#     Criminal Procedure, Enforcement, Trade Union and Tax Procedure codes.
#     A corpus-scope defect, fixed by excluding those articles from the index
#     (docs/decisions/2026-08-29-omnibus-scope-filter.md). tests/test_scope.py
#     asserts the mechanism; these assert the resulting gate decision.
#
#   kıdem tazminatı -- retrieved genuine electricity law (the Kalite
#     Yönetmeliği's kesinti-tazminatı formulas). Not a corpus defect, and not
#     fixable by a cutoff either. See
#     docs/decisions/2026-08-29-kidem-tazminati-gate-limit.md.
#
# Scores are from the 2026-08-29 calibrate_gate.py run against the filtered
# 26,441-chunk corpus.

REAL_FAILURE_CASES = [
    ("Trafik cezası itiraz süresi nedir?", 0.23971, 0.15201, "NOT_FOUND"),
    ("Vergi levhası nereye asılır?", 0.10336, 0.10845, "NOT_FOUND"),
    ("Kıdem tazminatı nasıl hesaplanır?", 0.29203, 0.31580, "ANSWER_WEAK"),
]


class TestOutOfScopeQuestionRegression:
    @pytest.mark.parametrize(
        "question, before, after, expected",
        REAL_FAILURE_CASES,
        ids=[c[0][:32] for c in REAL_FAILURE_CASES],
    )
    def test_pinned_to_current_measured_decision(self, question, before, after, expected):
        assert gate(after) == expected

    def test_none_of_the_three_is_answered_with_full_confidence(self):
        """The floor this change was made to clear: none may reach ANSWER."""
        for question, _before, after, _expected in REAL_FAILURE_CASES:
            assert gate(after) != "ANSWER", question

    @pytest.mark.parametrize(
        "question, before, after, expected",
        [c for c in REAL_FAILURE_CASES if c[0].startswith(("Trafik", "Vergi"))],
        ids=["trafik", "vergi"],
    )
    def test_scope_leak_failures_are_refused_outright(
        self, question, before, after, expected
    ):
        """These two were caused by the corpus and are fully fixed by it."""
        assert gate(after) == "NOT_FOUND"

    def test_kidem_tazminati_is_a_known_limitation_not_a_fix(self):
        """Known limitation, NOT a target to silently start passing.

        It retrieves correct electricity law about how outage compensation is
        calculated, so no corpus filter reaches it, and no cutoff separates it:
        a FLOOR above 0.31580 would refuse 3 of 15 (20%) genuinely answerable
        questions. Pinned as ANSWER_WEAK -- answered, with low_confidence set.
        If this ever becomes NOT_FOUND, check WHY before celebrating: the likely
        cause is a FLOOR that is now refusing real questions. Update the config
        comment and the decision record together, don't just edit the test.
        """
        assert gate(0.31580) == "ANSWER_WEAK"
        # The three answerable questions a floor above it would take with it.
        for answerable in (0.18804, 0.23964, 0.24807):
            assert gate(answerable) != "NOT_FOUND"


class TestOutOfDomainSweep:
    """The wider class the three failures were symptoms of.

    12 of these 19 gate to NOT_FOUND. The 7 that do not are all questions whose
    non-domain-specific vocabulary ("süre", "hesaplanır", "itiraz", "belirlenir")
    is common in regulatory Turkish; none reaches ANSWER. Pinned so that a
    change which pushes any of them INTO ANSWER fails loudly.
    """

    # (question fragment, measured fusion confidence)
    OUT_OF_DOMAIN = [
        ("kıdem tazminatı", 0.31580),
        ("toplu iş sözleşmesi", 0.28518),
        ("yıllık ücretli izin", 0.26760),
        ("sürücü belgesi", 0.23986),
        ("konut kira sözleşmesi", 0.22738),
        ("katma değer vergisi beyannamesi", 0.20110),
        ("LPG sorumlu müdür", 0.19375),
        ("gelir vergisi tarifesi", 0.17987),
        ("akaryakıt bayilik sermaye", 0.17950),
        ("devlet memuru disiplin cezası", 0.17046),
        ("memur aylık katsayısı", 0.16941),
        ("trafik cezası itiraz", 0.15201),
        ("vergi levhası", 0.10845),
        ("alkollü araç kullanma cezası", 0.10715),
        ("anlaşmalı boşanma", 0.08218),
        ("nafaka miktarı", 0.06514),
        ("kasten yaralama cezası", 0.05841),
        ("tutukluluk süresi", 0.05641),
        ("deniz balıkçılığı av yasağı", 0.06990),
    ]

    @pytest.mark.parametrize("question, score", OUT_OF_DOMAIN, ids=[q for q, _ in OUT_OF_DOMAIN])
    def test_no_out_of_domain_question_reaches_answer(self, question, score):
        assert gate(score) != "ANSWER", question

    def test_questions_sharing_no_vocabulary_are_refused_comfortably(self):
        """Criminal and family law share no terms with the corpus, and it shows.

        This is the control: it demonstrates the residual leakage is caused by
        vocabulary overlap rather than by a gate that fails on everything.
        """
        for question, score in self.OUT_OF_DOMAIN:
            if question in ("kasten yaralama cezası", "tutukluluk süresi", "nafaka miktarı"):
                assert gate(score) == "NOT_FOUND", question
                assert score < config.FUSION_FLOOR / 2, question

    def test_twelve_of_nineteen_are_refused_outright(self):
        """Matches the OUT-OF-DOMAIN SWEEP figure calibrate_gate.py reports.

        The other 7 reach ANSWER_WEAK, never ANSWER. Raising this number is the
        goal of any future query-side domain gate or reranker; lowering it
        without changing the cutoffs deliberately is a regression.
        """
        refused = sum(1 for _, score in self.OUT_OF_DOMAIN if gate(score) == "NOT_FOUND")
        assert (refused, len(self.OUT_OF_DOMAIN)) == (12, 19)
