"""Step 5: calibrate the confidence gate on the FUSED (dense + BM25) signal.

Step 4 calibrated a dense-only gate and, in doing so, measured two failure
modes that no choice of cosine cutoff could fix:

  1. Domain-vocabulary mismatch -- "Doğal gaz dağıtım şirketlerinin abone
     bağlantı bedeli" retrieved the *electricity* Dağıtım Bağlantı Bedelleri
     Tebliği at cosine 0.6405, above the ANSWER threshold.
  2. Diacritic sensitivity -- ASCII-folding the questions moved scores by up to
     -0.2062 and flipped 8 of 21 gate decisions, two from ANSWER to NOT_FOUND.

This script re-runs the same 15 answerable + 6 not-answerable questions through
the hybrid retriever, derives fresh cutoffs on the fused score, and reports
before/after on both failure classes specifically -- an aggregate improvement
that left the domain-mismatch case untouched would not be the win we wanted.

Every question is run TWICE, as typed and ASCII-folded. That probe is
permanent: diacritic handling is now a property the pipeline is supposed to
have, so a future change that regresses it must show up here rather than being
rediscovered by a user.

Usage:
    .venv\\Scripts\\python.exe scripts\\calibrate_gate.py
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.embed import FoundryUnavailable  # noqa: E402
from src.extract import _force_utf8_stdout  # noqa: E402
from src.lexical import fold_diacritics  # noqa: E402
from src.retrieval import Retriever, gate_dense, idf_coverage  # noqa: E402

# (question, expected_answerable)
#
# Unchanged from Step 4 on purpose: the point of this run is to compare against
# those numbers, which requires the same questions.
QUESTIONS: list[tuple[str, bool]] = [
    # -- expected answerable from the indexed electricity mevzuat ------------
    ("Elektrik piyasasında üretim lisansı almak için başvuru sahibinin sağlaması "
     "gereken şartlar nelerdir?", True),
    ("Gün öncesi piyasasında teklif verme ve eşleştirme süreci nasıl işler?", True),
    ("Lisanssız elektrik üretiminde çatı tipi güneş enerjisi santralleri için "
     "kurulu güç sınırı nedir?", True),
    ("Yenilenebilir enerji kaynak belgesi (YEK belgesi) nasıl alınır?", True),
    ("Dağıtım şirketinin tüketiciye planlı kesinti öncesinde bildirim yapma "
     "yükümlülüğü nedir?", True),
    ("Bağlantı anlaşması hangi hallerde sona erer veya feshedilir?", True),
    ("Yan hizmetler kapsamında primer frekans kontrol hizmeti nasıl tedarik edilir?", True),
    ("Serbest tüketici kimdir ve tedarikçisini değiştirme hakkını nasıl kullanır?", True),
    ("Kapasite mekanizması kapsamında yapılacak ödemeler nasıl hesaplanır?", True),
    ("Elektrik enerjisi ithalat ve ihracat faaliyeti için hangi lisans gereklidir?", True),
    ("Sayaçların okunması ve tüketim değerlerinin belirlenmesine ilişkin usul ve "
     "esaslar nelerdir?", True),
    ("Dağıtım tarifesinin düzenlenmesinde gelir tavanı nasıl belirlenir?", True),
    ("Piyasa işletmecisine verilecek teminatların türleri ve tutarı nasıl hesaplanır?", True),
    ("Lisans sahiplerine uygulanacak idari para cezaları nelerdir?", True),
    ("İletim sistemi işletmecisinin arz güvenliğine ilişkin yükümlülükleri nelerdir?", True),
    # -- expected NOT answerable: plausible-sounding but outside this corpus --
    ("Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl hesaplanır?", False),
    ("LPG otogaz istasyonlarında sorumlu müdür bulundurma zorunluluğu nedir?", False),
    ("Akaryakıt bayilik lisansı için aranan asgari sermaye şartı nedir?", False),
    ("Rafinerici lisansı sahiplerinin ulusal petrol stoku tutma yükümlülüğü nedir?", False),
    # -- expected NOT answerable: unrelated to energy regulation entirely -----
    ("Konut kira sözleşmesinin feshi için ihtarname süresi ne kadardır?", False),
    ("Deniz balıkçılığında av yasağı dönemleri hangi aylardır?", False),
    # -- the three real failures reported 2026-08-29 -------------------------
    # Answered from the live corpus by a user. Pinned here permanently: the
    # first two were caused by omnibus scope leakage and are fixed by
    # docs/decisions/2026-08-29-omnibus-scope-filter.md; the third is NOT, and
    # its unfixability is the point of measuring it here every run.
    ("Kıdem tazminatı nasıl hesaplanır?", False),
    ("Trafik cezası itiraz süresi nedir?", False),
    ("Vergi levhası nereye asılır?", False),
    # -- expected NOT answerable: legal domains with ZERO conceptual overlap --
    #
    # The pre-existing negatives were all energy-adjacent (doğal gaz, LPG,
    # petrol) plus two strays, which measured only the hardest case. These
    # measure the case the corpus leak actually broke: questions sharing no
    # vocabulary and no concepts with electricity regulation, where NOT_FOUND
    # should be easy and any ANSWER is a scope defect rather than a close call.
    #
    # İş hukuku (labour)
    ("İşçinin yıllık ücretli izin süresi kaç gündür?", False),
    ("Toplu iş sözleşmesi en fazla kaç yıl süreyle yapılabilir?", False),
    # Vergi hukuku (tax)
    ("Katma değer vergisi beyannamesi hangi tarihe kadar verilir?", False),
    ("Gelir vergisi tarifesindeki dilimler nasıl belirlenir?", False),
    # Trafik (traffic)
    ("Sürücü belgesi kaç yılda bir yenilenir?", False),
    ("Alkollü araç kullanmanın idari para cezası ne kadardır?", False),
    # Ceza hukuku (criminal)
    ("Kasten yaralama suçunun cezası nedir?", False),
    ("Tutukluluk süresinin azami sınırı ne kadardır?", False),
    # Aile hukuku (family)
    ("Anlaşmalı boşanma davası nasıl açılır?", False),
    ("Nafaka miktarı neye göre belirlenir?", False),
    # Memur hukuku (civil servants)
    ("Devlet memuruna verilen disiplin cezasına itiraz süresi nedir?", False),
    ("Memur aylık katsayısı hangi usulle belirlenir?", False),
]

# The three failures reported 2026-08-29, with the fusion confidence each
# scored against the UNFILTERED corpus. Pinned so the report can show
# before/after rather than asserting an improvement.
REAL_FAILURES = {
    "Kıdem tazminatı nasıl hesaplanır?": 0.29203,
    "Trafik cezası itiraz süresi nedir?": 0.23971,
    "Vergi levhası nereye asılır?": 0.10336,
}

# The two domain-mismatch questions Step 4 singled out, with their dense
# cosines. "doğal gaz" scored 0.6405 -> ANSWER (wrong); the rafinerici question
# scored 0.5536 -> ANSWER_WEAK on an electricity licensing chunk.
DOMAIN_MISMATCH = {
    "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl hesaplanır?": 0.6405,
    "Rafinerici lisansı sahiplerinin ulusal petrol stoku tutma yükümlülüğü nedir?": 0.5536,
}

# The 8 questions whose gate decision changed under ASCII folding in Step 4,
# with their (as-typed, folded) dense cosines. Fusion is supposed to make each
# pair agree; this is the list the regression tests pin.
DIACRITIC_FLIPS = {
    "Gün öncesi piyasasında teklif verme ve eşleştirme süreci nasıl işler?": (0.6294, 0.5412),
    "Lisanssız elektrik üretiminde çatı tipi güneş enerjisi santralleri için "
    "kurulu güç sınırı nedir?": (0.6700, 0.5878),
    "Dağıtım şirketinin tüketiciye planlı kesinti öncesinde bildirim yapma "
    "yükümlülüğü nedir?": (0.6552, 0.5662),
    "Bağlantı anlaşması hangi hallerde sona erer veya feshedilir?": (0.5361, 0.3299),
    "Sayaçların okunması ve tüketim değerlerinin belirlenmesine ilişkin usul ve "
    "esaslar nelerdir?": (0.6565, 0.4931),
    "Dağıtım tarifesinin düzenlenmesinde gelir tavanı nasıl belirlenir?": (0.6144, 0.5992),
    "Doğal gaz dağıtım şirketlerinin abone bağlantı bedeli nasıl hesaplanır?": (0.6405, 0.5303),
    "Akaryakıt bayilik lisansı için aranan asgari sermaye şartı nedir?": (0.5438, 0.4911),
}


@dataclass
class Probe:
    """One question measured both ways, on both the fused and the dense path."""

    question: str
    answerable: bool
    fused: float
    fused_folded: float
    dense: float
    dense_folded: float
    citation: str
    coverage: float


def measure(retriever: Retriever, question: str, answerable: bool) -> Probe:
    """Run one question every way we need to compare: fused and dense, typed and folded.

    `fused` is the CONFIDENCE score (retrieval.fusion_confidence), not the RRF
    score. The first hybrid calibration run thresholded raw RRF and the classes
    overlapped worse than dense-only did -- RRF encodes rank, not certainty.
    See scripts/eval_gate_signals.py for the six candidates that were compared.
    """
    folded = fold_diacritics(question)

    fused = retriever.retrieve(question, top_k=config.TOP_K)
    fused_folded = retriever.retrieve(folded, top_k=config.TOP_K)
    dense = retriever.retrieve_dense(question, top_k=1)
    dense_folded = retriever.retrieve_dense(folded, top_k=1)

    return Probe(
        question=question,
        answerable=answerable,
        fused=retriever.confidence(question, fused),
        fused_folded=retriever.confidence(folded, fused_folded),
        dense=dense[0].score if dense else 0.0,
        dense_folded=dense_folded[0].score if dense_folded else 0.0,
        citation=fused[0].citation() if fused else "(nothing retrieved)",
        coverage=idf_coverage(retriever.bm25, question, [r.text for r in fused]),
    )


def recommend(pos: list[float], neg: list[float]) -> tuple[float, float, str]:
    """Derive (threshold, floor) from where the two score distributions separate.

    Same derivation as Step 4 so the two runs stay comparable -- only the signal
    being thresholded has changed.

    Clean case: both cutoffs go inside the gap between the classes, at 25% and
    75% of it. Overlapping case: the floor is anchored at the lowest answerable
    score (never refuse something we can actually answer) and the threshold at
    the observed cutoff maximizing Youden's J.
    """
    pos_min, neg_max = min(pos), max(neg)
    if pos_min > neg_max:
        gap = pos_min - neg_max
        floor = math.floor((neg_max + 0.25 * gap) * 100000) / 100000
        threshold = math.floor((neg_max + 0.75 * gap) * 100000) / 100000
        note = (
            f"CLEAN SEPARATION: lowest answerable {pos_min:.5f} > highest "
            f"not-answerable {neg_max:.5f} (gap {gap:.5f}).\n"
            f"  Both cutoffs sit inside that gap, at 25% and 75%. Every "
            f"answerable question lands at or above the threshold (ANSWER) and "
            f"every not-answerable one below the floor (NOT_FOUND)."
        )
        return threshold, floor, note

    best_cut, best_j, best_tpr, best_fpr = pos_min, -1.0, 0.0, 0.0
    for cut in sorted(set(pos + neg)):
        tpr = sum(1 for s in pos if s >= cut) / len(pos)
        fpr = sum(1 for s in neg if s >= cut) / len(neg)
        if tpr - fpr > best_j:
            best_cut, best_j, best_tpr, best_fpr = cut, tpr - fpr, tpr, fpr

    threshold = math.floor(best_cut * 100000) / 100000
    floor = math.floor(pos_min * 100000) / 100000
    rejected = sum(1 for s in neg if s < floor)
    note = (
        f"OVERLAP: lowest answerable {pos_min:.5f} <= highest not-answerable "
        f"{neg_max:.5f}, so no single cutoff separates the classes.\n"
        f"  threshold from the best observed separation point {best_cut:.5f} "
        f"(Youden's J={best_j:.3f}: {best_tpr:.0%} of answerable at/above it, "
        f"{best_fpr:.0%} of not-answerable).\n"
        f"  floor anchored at the lowest answerable score so nothing answerable "
        f"is refused; it rejects {rejected}/{len(neg)} not-answerable questions "
        f"outright and leaves the rest in ANSWER_WEAK."
    )
    return threshold, floor, note


def _decision(score: float, threshold: float, floor: float) -> str:
    if score >= threshold:
        return "ANSWER"
    return "ANSWER_WEAK" if score >= floor else "NOT_FOUND"


def main() -> int:
    _force_utf8_stdout()

    try:
        retriever = Retriever.open()
    except FoundryUnavailable as exc:
        print(f"FATAL: {exc}")
        return 3

    print("=" * 100)
    print("CONFIDENCE GATE CALIBRATION -- HYBRID (dense + BM25, RRF)")
    print("=" * 100)
    print(f"Corpus vectors  : {len(retriever):,} active chunks "
          f"({retriever.matrix.nbytes / 1e6:.1f} MB) loaded in {retriever.load_seconds:.2f}s")
    print(f"BM25 index      : {len(retriever.bm25):,} docs, "
          f"{retriever.bm25.vocabulary_size:,} terms, built in "
          f"{retriever.bm25_load_seconds:.2f}s")
    print(f"Embedding model : {retriever.embedder.model_id} (dim {retriever.embedder.dimension})")
    print(f"Fusion          : RRF k={config.RRF_K}, "
          f"candidate depth {config.FUSION_CANDIDATES}, top_k {config.TOP_K}")
    print(f"Questions       : {sum(1 for _, a in QUESTIONS if a)} answerable, "
          f"{sum(1 for _, a in QUESTIONS if not a)} not answerable, each run twice "
          f"(as typed + ASCII-folded)")
    print()

    # Burn the one-off embedding warm-up so it does not distort the latency figures.
    warm_started = time.perf_counter()
    retriever.embed_query("elektrik piyasası")
    warmup = time.perf_counter() - warm_started

    probes: list[Probe] = []
    latencies: list[float] = []
    for question, answerable in QUESTIONS:
        started = time.perf_counter()
        probes.append(measure(retriever, question, answerable))
        latencies.append(time.perf_counter() - started)

    pos = [p.fused for p in probes if p.answerable]
    neg = [p.fused for p in probes if not p.answerable]
    threshold, floor, note = recommend(pos, neg)

    header = (f"{'exp':<5} {'fused':>9} {'folded':>9} {'delta':>9}  {'dense':>7} "
              f"{'cover':>6}  {'decision':<12} question")
    print(header)
    print("-" * len(header))
    for p in probes:
        print(
            f"{'YES' if p.answerable else 'NO':<5} {p.fused:>9.4f} {p.fused_folded:>9.4f} "
            f"{p.fused_folded - p.fused:>+9.5f}  {p.dense:>7.4f} {p.coverage:>6.3f}  "
            f"{_decision(p.fused, threshold, floor):<12} {p.question[:42]}"
        )

    print()
    print("-" * 100)
    print("SCORE DISTRIBUTIONS (fusion confidence = dense_top1 x idf_coverage)")
    print("-" * 100)
    for label, values in (("answerable    ", pos), ("not answerable", neg)):
        print(f"  {label}: n={len(values):<3} min={min(values):.5f} "
              f"median={statistics.median(values):.5f} "
              f"mean={statistics.mean(values):.5f} max={max(values):.5f}")

    # ---------------------------------------------------------------- failure class 1
    print()
    print("=" * 100)
    print("FAILURE CLASS 1 -- DOMAIN-VOCABULARY MISMATCH (before -> after)")
    print("=" * 100)
    for p in probes:
        if p.question not in DOMAIN_MISMATCH:
            continue
        before_score = DOMAIN_MISMATCH[p.question]
        before = gate_dense(before_score)
        after = _decision(p.fused, threshold, floor)
        if after == "NOT_FOUND":
            verdict = "FIXED"
        elif after == "ANSWER":
            verdict = "STILL WRONG"
        else:
            verdict = "improved (no longer ANSWER)"
        print(f"\n  {p.question}")
        print(f"    Step 4 dense : {before_score:.4f} -> {before}")
        print(f"    Step 5 fused : {p.fused:.5f} -> {after}    [{verdict}]")
        print(f"    IDF coverage of query terms in top-{config.TOP_K}: {p.coverage:.3f}")
        print(f"    top-1 now    : {p.citation[:86]}")

    # ---------------------------------------------------------------- failure class 2
    print()
    print("=" * 100)
    print("FAILURE CLASS 2 -- DIACRITIC FLIPS (do as-typed and ASCII-folded now agree?)")
    print("=" * 100)
    print("  Step 4: 8 of 21 questions changed gate decision when ASCII-folded.")
    print()
    agree = 0
    for p in probes:
        if p.question not in DIACRITIC_FLIPS:
            continue
        d_typed, d_folded = DIACRITIC_FLIPS[p.question]
        now_typed = _decision(p.fused, threshold, floor)
        now_folded = _decision(p.fused_folded, threshold, floor)
        matched = now_typed == now_folded
        agree += matched
        print(f"  [{'OK ' if matched else 'NO '}] {p.question[:68]}")
        print(f"         Step 4 dense : {d_typed:.4f} / {d_folded:.4f} -> "
              f"{gate_dense(d_typed)} / {gate_dense(d_folded)}")
        print(f"         Step 5 fused : {p.fused:.5f} / {p.fused_folded:.5f} -> "
              f"{now_typed} / {now_folded}")
    print(f"\n  {agree}/{len(DIACRITIC_FLIPS)} of Step 4's flipped questions now agree "
          f"between spellings.")

    total_flips = sum(
        1 for p in probes
        if _decision(p.fused, threshold, floor) != _decision(p.fused_folded, threshold, floor)
    )
    print(f"  Across ALL {len(probes)} questions: {total_flips} still change decision when "
          f"folded (Step 4 dense-only: 8).")
    deltas = [p.fused_folded - p.fused for p in probes]
    print(f"  Fused score change under folding: mean {statistics.mean(deltas):+.5f}, "
          f"worst {min(deltas):+.5f}, best {max(deltas):+.5f}")

    # ---------------------------------------------------------------- failure class 3
    print()
    print("=" * 100)
    print("FAILURE CLASS 3 -- THE THREE REAL FAILURES (2026-08-29)")
    print("=" * 100)
    print("  Scores before are against the UNFILTERED corpus, i.e. before")
    print("  docs/decisions/2026-08-29-omnibus-scope-filter.md was applied.")
    for p in probes:
        if p.question not in REAL_FAILURES:
            continue
        before_score = REAL_FAILURES[p.question]
        before = _decision(before_score, config.FUSION_THRESHOLD, config.FUSION_FLOOR)
        after = _decision(p.fused, threshold, floor)
        verdict = "FIXED" if after == "NOT_FOUND" else (
            "STILL WRONG" if after == "ANSWER" else "improved (no longer ANSWER)"
        )
        print(f"\n  {p.question}")
        print(f"    before (unfiltered corpus, old cutoffs): {before_score:.5f} -> {before}")
        print(f"    after  (filtered corpus, new cutoffs)  : {p.fused:.5f} -> {after}"
              f"    [{verdict}]")
        print(f"    IDF coverage: {p.coverage:.3f}   top-1 now: {p.citation[:74]}")

    # ------------------------------------------------- out-of-domain sweep
    print()
    print("=" * 100)
    print("OUT-OF-DOMAIN SWEEP -- legal domains with no overlap with electricity")
    print("=" * 100)
    off_domain = [p for p in probes if not p.answerable and p.question not in DOMAIN_MISMATCH]
    refused = [p for p in off_domain if _decision(p.fused, threshold, floor) == "NOT_FOUND"]
    print(f"  {len(refused)}/{len(off_domain)} correctly gated to NOT_FOUND.")
    leaking = [p for p in off_domain if _decision(p.fused, threshold, floor) != "NOT_FOUND"]
    if leaking:
        print("\n  Still not refused:")
        for p in sorted(leaking, key=lambda x: -x.fused):
            print(f"    {p.fused:.5f}  {_decision(p.fused, threshold, floor):<12} {p.question}")
            print(f"    {'':13}top-1: {p.citation[:78]}")
    else:
        print("  None leaking.")

    # ---------------------------------------------------------------- overall accuracy
    print()
    print("=" * 100)
    print("OVERALL GATE ACCURACY -- dense-only baseline vs hybrid")
    print("=" * 100)
    print("  'correct' = an answerable question is not refused, and a")
    print("  not-answerable question is refused (NOT_FOUND).")
    print()

    def accuracy(typed_scores, folded_scores, decide) -> tuple[int, int]:
        typed = sum(
            1 for p, s in zip(probes, typed_scores)
            if (decide(s) != "NOT_FOUND") == p.answerable
        )
        folded = sum(
            1 for p, s in zip(probes, folded_scores)
            if (decide(s) != "NOT_FOUND") == p.answerable
        )
        return typed, folded

    dense_typed, dense_folded_acc = accuracy(
        [p.dense for p in probes], [p.dense_folded for p in probes], gate_dense
    )
    fused_typed, fused_folded_acc = accuracy(
        [p.fused for p in probes], [p.fused_folded for p in probes],
        lambda s: _decision(s, threshold, floor),
    )
    n = len(probes)
    print(f"  {'':<24} {'as typed':>12} {'ASCII-folded':>14}")
    print(f"  {'dense-only (Step 4)':<24} {dense_typed:>7}/{n}     {dense_folded_acc:>7}/{n}")
    print(f"  {'hybrid (Step 5)':<24} {fused_typed:>7}/{n}     {fused_folded_acc:>7}/{n}")

    # ---------------------------------------------------------------- latency
    print()
    print("-" * 100)
    print("LATENCY")
    print("-" * 100)
    print(f"  dense matrix load (startup) : {retriever.load_seconds:.2f} s")
    print(f"  BM25 index build (startup)  : {retriever.bm25_load_seconds:.2f} s")
    print(f"  embed warm-up (once)        : {warmup * 1000:.0f} ms (excluded below)")
    print(f"  full probe per question     : {statistics.mean(latencies) * 1000:.0f} ms avg "
          f"(5 retrievals each: fused x2, dense x2, lexical)")

    sample = "Elektrik piyasasında üretim lisansı almak için gereken şartlar nelerdir?"
    _, timings = retriever.retrieve_fused_timed(sample, top_k=config.TOP_K)
    print(f"  one fused query             : dense {timings['dense'] * 1000:.0f} ms || "
          f"bm25 {timings['lexical'] * 1000:.0f} ms (concurrent), "
          f"fuse {timings['fuse'] * 1000:.1f} ms, "
          f"hydrate {timings['hydrate'] * 1000:.1f} ms, "
          f"total {timings['total'] * 1000:.0f} ms")

    # ---------------------------------------------------------------- recommendation
    print()
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print(f"  {note}")
    print()
    print(f"  FUSION_THRESHOLD = {threshold}   # >= this -> ANSWER")
    print(f"  FUSION_FLOOR     = {floor}   # >= this -> ANSWER_WEAK, below -> NOT_FOUND")
    print()
    print(f"  (current config: threshold={config.FUSION_THRESHOLD}, "
          f"floor={config.FUSION_FLOOR})")
    print("  These are fusion-CONFIDENCE scores (not RRF scores). They REPLACE")
    print("  the dense-only SIMILARITY_* cutoffs as what gate() thresholds.")
    print()

    retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
