"""Tests for Reciprocal Rank Fusion and the two Step 4 failure classes it exists
to fix.

Two kinds of test live here:

1. TestReciprocalRankFusion -- pure ordering tests on a synthetic ranked list,
   independent of any corpus. reciprocal_rank_fusion() is a rank-position
   function; its correctness does not depend on real embeddings or BM25 data.

2. TestDomainMismatchRegression / TestDiacriticFlipRegression -- pin the gate
   DECISION for the specific questions Step 4 and Step 5 measured, to the
   fusion-confidence scores scripts/calibrate_gate.py actually produced against
   the real 27,047-chunk corpus (rerun 2026-08-26, matching the values recorded
   in config.py). These do not re-run retrieval themselves -- that would make
   the test suite depend on a live Foundry Local server and the full corpus --
   they pin gate() against the measured scores, the same pattern
   TestGateBoundaries in test_retrieval.py already uses. If a future change to
   fusion_confidence(), the BM25 tokenizer, or the calibrated cutoffs shifts one
   of these decisions, the test that catches it is here, not a person rereading
   a config.py comment.
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
#
# Fusion-confidence scores below are exactly what scripts/calibrate_gate.py
# measured against the real 27,047-chunk corpus (2026-08-26 rerun, matching
# the FUSION_THRESHOLD=0.23963 / FUSION_FLOOR=0.1871 recorded in config.py).


class TestDomainMismatchRegression:
    def test_dogal_gaz_question_still_answers_wrong_known_limitation(self):
        """Known limitation, NOT a target to silently start passing.

        "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl
        hesaplanır?" retrieves the *electricity* Dağıtım Bağlantı Bedelleri
        chunk at fusion confidence 0.50644 -- comfortably above THRESHOLD --
        because "doğal" and "gaz" both genuinely occur in this electricity-only
        corpus (the Electricity Market Law references natural gas), so IDF
        coverage stays high (0.796) even though the question is about a
        neighbouring energy domain this corpus does not cover. See config.py's
        FUSION_THRESHOLD comment ("STILL UNFIXED") for the full analysis --
        fixing this needs document-level domain filtering or a reranker, not a
        cutoff change. This test exists to CATCH a future accidental fix (or
        regression) here, not to assert the bug is fine forever: if this ever
        flips, update the comment and the config.py known-limitation note
        together, don't just delete the test.
        """
        assert gate(0.50644) == "ANSWER"

    def test_rafinerici_question_improved_but_is_not_rejected_outright(self):
        """The related domain-mismatch question DID improve under fusion.

        Step 4 dense-only: 0.5536 -> ANSWER_WEAK. Step 5 fused: 0.21310, still
        ANSWER_WEAK (not the ANSWER a raw cosine gave partial credit for), but
        also not below FLOOR -- BM25 coverage for "rafinerici / ulusal petrol
        stoku" against an electricity corpus is low (0.385) but not zero.
        """
        assert gate(0.21310) == "ANSWER_WEAK"


# --------------------------------------------------------------------------
# Failure class 2: diacritic sensitivity (Step 4's 8 flipped questions)
# --------------------------------------------------------------------------
#
# Each entry is (question, fused_typed, fused_folded). Values are the exact
# fusion-confidence scores from the 2026-08-26 calibrate_gate.py rerun.
# 7 of these 8 now agree between spellings; "Akaryakıt bayilik lisansı..." is
# the one that still disagrees (ANSWER_WEAK typed vs NOT_FOUND folded) --
# pinned as still-disagreeing, not as a regression to silently tolerate.

DIACRITIC_FLIP_CASES = [
    (
        "Gün öncesi piyasasında teklif verme ve eşleştirme süreci nasıl işler?",
        0.41196, "ANSWER", 0.35427, "ANSWER",
    ),
    (
        "Lisanssız elektrik üretiminde çatı tipi güneş enerjisi santralleri "
        "için kurulu güç sınırı nedir?",
        0.24814, "ANSWER", 0.25065, "ANSWER",
    ),
    (
        "Dağıtım şirketinin tüketiciye planlı kesinti öncesinde bildirim "
        "yapma yükümlülüğü nedir?",
        0.23963, "ANSWER", 0.40318, "ANSWER",
    ),
    (
        "Bağlantı anlaşması hangi hallerde sona erer veya feshedilir?",
        0.51015, "ANSWER", 0.29426, "ANSWER",
    ),
    (
        "Sayaçların okunması ve tüketim değerlerinin belirlenmesine ilişkin "
        "usul ve esaslar nelerdir?",
        0.36552, "ANSWER", 0.33454, "ANSWER",
    ),
    (
        "Dağıtım tarifesinin düzenlenmesinde gelir tavanı nasıl belirlenir?",
        0.36407, "ANSWER", 0.38212, "ANSWER",
    ),
    (
        # Same question as the domain-mismatch test above, this time checked
        # for typed/folded agreement rather than for the ANSWER-vs-correct
        # question. Both facts are true of it at once.
        "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl "
        "hesaplanır?",
        0.50644, "ANSWER", 0.46844, "ANSWER",
    ),
    (
        "Akaryakıt bayilik lisansı için aranan asgari sermaye şartı nedir?",
        0.18723, "ANSWER_WEAK", 0.16810, "NOT_FOUND",
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

    def test_seven_of_eight_now_agree_between_spellings(self):
        agreeing = sum(
            1
            for _, fused_typed, dt, fused_folded, df in DIACRITIC_FLIP_CASES
            if gate(fused_typed) == gate(fused_folded)
        )
        assert agreeing == 7

    def test_the_one_remaining_disagreement_is_akaryakit_bayilik(self):
        """Named explicitly so a future fix to it doesn't go unnoticed either."""
        question, fused_typed, _, fused_folded, _ = DIACRITIC_FLIP_CASES[-1]
        assert question.startswith("Akaryakıt bayilik lisansı")
        assert gate(fused_typed) != gate(fused_folded)
