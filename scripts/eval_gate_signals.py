"""Which signal should the hybrid gate threshold? Measure several, pick by separation.

The first hybrid calibration run exposed a problem: the raw RRF score is a good
ranker but a useless confidence signal. RRF scores depend only on rank
position, so a top-1 result scores ~1/(k+1) doubled when both rankers agree and
~1/(k+1) when only one does -- roughly 0.0328 or 0.0164 regardless of whether
the corpus actually contains the answer. Measured over the 21 calibration
questions the classes overlapped worse than dense-only did.

So the gate needs a signal that preserves magnitude while still benefiting from
fusion. This script evaluates the candidates side by side on the same 15
answerable + 6 not-answerable questions, scoring each by Youden's J at its best
cutoff, and reports how each one handles the two named failure modes.

Nothing here is used at runtime -- it exists to justify the choice recorded in
src/retrieval.py and the cutoffs in config.py.

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_gate_signals.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.extract import _force_utf8_stdout  # noqa: E402
from src.lexical import fold_diacritics, tokenize  # noqa: E402
from src.retrieval import Retriever  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_gate import DIACRITIC_FLIPS, DOMAIN_MISMATCH, QUESTIONS  # noqa: E402


def signals(retriever: Retriever, query: str) -> dict[str, float]:
    """Every candidate gate signal for one query, computed from one retrieval."""
    fused = retriever.retrieve(query, top_k=config.TOP_K)
    if not fused:
        return {name: 0.0 for name in SIGNALS}

    top = fused[0]
    bm25 = retriever.bm25
    n = len(bm25.ids)
    max_idf = math.log(1.0 + (n + 0.5) / 0.5)

    terms = set(tokenize(query))
    total_idf = 0.0
    for term in terms:
        df = len(bm25.postings.get(term, ()))
        total_idf += max_idf if df == 0 else bm25.idf(term)

    # Dense cosine of whichever chunk fusion put first. Falls back to the best
    # dense score in the candidate set when fusion's top-1 came from BM25 alone.
    dense_top = top.dense_score
    if dense_top is None:
        dense_top = max((r.dense_score or 0.0) for r in fused)

    # Best raw BM25 score among the fused results, normalized by how much
    # information the query carried in the first place. "How much of what the
    # user asked did the best chunk actually explain."
    lexical_top = max((r.lexical_score or 0.0) for r in fused)
    lexical_norm = lexical_top / total_idf if total_idf else 0.0

    # IDF-weighted share of query terms appearing anywhere in the top-k texts.
    union = set()
    for result in fused:
        union |= set(tokenize(result.text))
    covered = 0.0
    for term in terms:
        df = len(bm25.postings.get(term, ()))
        weight = max_idf if df == 0 else bm25.idf(term)
        if term in union:
            covered += weight
    coverage = covered / total_idf if total_idf else 0.0

    return {
        "dense_top1": dense_top,
        "lexical_norm": lexical_norm,
        "coverage_topk": coverage,
        "dense_x_coverage": dense_top * coverage,
        "dense_x_lexnorm": dense_top * lexical_norm,
        "mean_dense_cov": (dense_top + coverage) / 2,
    }


SIGNALS = (
    "dense_top1",
    "lexical_norm",
    "coverage_topk",
    "dense_x_coverage",
    "dense_x_lexnorm",
    "mean_dense_cov",
)


def best_separation(pos: list[float], neg: list[float]) -> tuple[float, float, float, float]:
    """Best Youden's J over observed cutoffs. Returns (J, cutoff, tpr, fpr)."""
    best = (-1.0, 0.0, 0.0, 0.0)
    for cut in sorted(set(pos + neg)):
        tpr = sum(1 for s in pos if s >= cut) / len(pos)
        fpr = sum(1 for s in neg if s >= cut) / len(neg)
        if tpr - fpr > best[0]:
            best = (tpr - fpr, cut, tpr, fpr)
    return best


def main() -> int:
    _force_utf8_stdout()
    retriever = Retriever.open()
    retriever.embed_query("elektrik piyasası")

    print(f"Corpus {len(retriever):,} chunks | BM25 {retriever.bm25.vocabulary_size:,} terms")
    print(f"Evaluating {len(SIGNALS)} candidate gate signals on "
          f"{sum(1 for _, a in QUESTIONS if a)} answerable + "
          f"{sum(1 for _, a in QUESTIONS if not a)} not-answerable questions,\n"
          f"each also ASCII-folded.\n")

    measured: list[tuple[str, bool, dict[str, float], dict[str, float]]] = []
    for question, answerable in QUESTIONS:
        typed = signals(retriever, question)
        folded = signals(retriever, fold_diacritics(question))
        measured.append((question, answerable, typed, folded))

    print("=" * 100)
    print("SEPARATION BY SIGNAL (higher J is better; J=1.0 means perfectly separable)")
    print("=" * 100)
    print(f"{'signal':<20} {'J':>6} {'cutoff':>9} {'TPR':>7} {'FPR':>7}   "
          f"{'fold-stable':>11}   verdict")
    print("-" * 100)

    results = {}
    for name in SIGNALS:
        pos = [t[name] for _, a, t, _ in measured if a]
        neg = [t[name] for _, a, t, _ in measured if not a]
        j, cut, tpr, fpr = best_separation(pos, neg)
        # How many questions keep the same side of the cutoff when ASCII-folded.
        stable = sum(
            1 for _, _, t, f in measured if (t[name] >= cut) == (f[name] >= cut)
        )
        results[name] = (j, cut, tpr, fpr, stable)
        verdict = "clean" if j >= 0.999 else ("usable" if j >= 0.75 else "weak")
        print(f"{name:<20} {j:>6.3f} {cut:>9.4f} {tpr:>7.0%} {fpr:>7.0%}   "
              f"{stable:>6}/{len(measured)}   {verdict}")

    best_name = max(results, key=lambda k: (results[k][0], results[k][4]))
    print(f"\n  best: {best_name} (J={results[best_name][0]:.3f}, "
          f"fold-stable {results[best_name][4]}/{len(measured)})")

    print()
    print("=" * 100)
    print(f"PER-QUESTION VALUES FOR THE TWO STRONGEST SIGNALS")
    print("=" * 100)
    ranked = sorted(results, key=lambda k: -results[k][0])[:2]
    print(f"{'exp':<5} " + "".join(f"{n:>20}" for n in ranked) + "  question")
    for question, answerable, typed, folded in measured:
        cells = "".join(f"{typed[n]:>11.4f}/{folded[n]:<8.4f}" for n in ranked)
        print(f"{'YES' if answerable else 'NO':<5} {cells}  {question[:40]}")

    print()
    print("=" * 100)
    print("BEHAVIOUR ON THE TWO NAMED FAILURE MODES")
    print("=" * 100)
    for name in ranked:
        j, cut, _, _, stable = results[name]
        print(f"\n  -- {name} (cutoff {cut:.4f}) --")
        for question, answerable, typed, folded in measured:
            if question in DOMAIN_MISMATCH:
                verdict = "REJECTED (good)" if typed[name] < cut else "accepted (BAD)"
                print(f"     domain-mismatch  {typed[name]:.4f}  {verdict:<16} "
                      f"{question[:46]}")
        flips = sum(
            1 for question, _, typed, folded in measured
            if question in DIACRITIC_FLIPS and (typed[name] >= cut) != (folded[name] >= cut)
        )
        print(f"     diacritic flips still disagreeing: {flips}/{len(DIACRITIC_FLIPS)}")

    retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
