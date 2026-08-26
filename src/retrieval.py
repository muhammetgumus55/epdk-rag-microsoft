"""Hybrid retrieval: dense embeddings fused with BM25, behind a confidence gate.

The whole active embedding matrix is loaded into RAM once at startup and kept
there: 27k x 1024 float32 is ~111 MB, which is cheap next to paying SQLite a
BLOB decode for every vector on every query. SQLite is still consulted per
query, but only for the handful of chunks that actually made top_k -- metadata
lookups by primary key, not a scan. The BM25 index (src/lexical.py) is built
into memory alongside it.

Vectors are L2-normalized at load time, so dense scoring is a single
matrix-vector product and cosines sit in [-1, 1]. BM25 scores are unbounded,
so the two are combined by Reciprocal Rank Fusion, which uses only rank
positions and never has to reconcile the two scales.

Three retrieval paths exist and all stay callable:

    retrieve()          fused dense + BM25 -- the production path, gated by gate()
    retrieve_dense()    Step 4's dense-only baseline, gated by gate_dense()
    retrieve_lexical()  BM25 only, mostly for diagnosis

The baseline is kept on purpose: "hybrid is better" is a measured claim (see
scripts/calibrate_gate.py), not an assumption, and it needs something to be
better *than*.
"""
from __future__ import annotations

import math
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import config, store
from .embed import EmbeddingClient
from .lexical import BM25Index, tokenize

GateDecision = Literal["ANSWER", "ANSWER_WEAK", "NOT_FOUND"]


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved chunk: its text, its score, and every citation field the store holds.

    The field set mirrors store.fetch_chunk_metadata() exactly (minus the raw
    embedding BLOB). Nothing is dropped between the DB row and this object --
    a citation needs document_title + article_ref + page range + source_path,
    and quality_flag is what tells a caller the underlying extraction was
    suspect, so all of them travel together.
    """

    chunk_id: int
    score: float
    text: str
    article_ref: str | None
    document_title: str | None
    page_start: int | None
    page_end: int | None
    quality_flag: str | None
    source_path: str
    chunk_index: int
    file_sha256: str
    # Provenance of a fused result: which ranker found it, and where. None on a
    # given side means that ranker did not surface this chunk at all -- which
    # is itself the signal (a chunk with no lexical rank matched no query term).
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None

    @classmethod
    def from_metadata(cls, meta: dict, score: float, **provenance) -> "RetrievalResult":
        return cls(
            chunk_id=meta["id"],
            score=score,
            **provenance,
            text=meta["text"],
            article_ref=meta["article_ref"],
            document_title=meta["document_title"],
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            quality_flag=meta["quality_flag"],
            source_path=meta["source_path"],
            chunk_index=meta["chunk_index"],
            file_sha256=meta["file_sha256"],
        )

    def citation(self) -> str:
        """Human-readable citation line, degrading gracefully when fields are NULL."""
        parts = [self.document_title or self.source_path]
        if self.article_ref:
            parts.append(self.article_ref)
        if self.page_start is not None:
            pages = (
                f"s. {self.page_start}"
                if self.page_end in (None, self.page_start)
                else f"s. {self.page_start}-{self.page_end}"
            )
            parts.append(pages)
        return " / ".join(parts)


# --------------------------------------------------------------------------
# Confidence gate
# --------------------------------------------------------------------------


def _classify(score: float, threshold: float | None, floor: float | None, names: str) -> GateDecision:
    if threshold is None or floor is None:
        raise ValueError(
            f"config.{names} are unset. Run scripts/calibrate_gate.py against "
            "the real corpus and record the derived values in config.py."
        )
    if score >= threshold:
        return "ANSWER"
    if score >= floor:
        return "ANSWER_WEAK"
    return "NOT_FOUND"


def gate(score: float) -> GateDecision:
    """Classify a fusion CONFIDENCE score into an answering decision.

    This is the gate the system actually uses. Its cutoffs are
    config.FUSION_THRESHOLD / FUSION_FLOOR, calibrated on the output of
    fusion_confidence() -- NOT on raw RRF scores (which are not a confidence
    signal at all, see fusion_confidence's docstring) and NOT on the dense-only
    SIMILARITY_* values from Step 4, which live on a different scale.

    Feed it Retriever.confidence(), or just call Retriever.answer().

    Boundaries are inclusive downward: a score exactly equal to the threshold
    is ANSWER, and one exactly equal to the floor is ANSWER_WEAK.
    """
    return _classify(
        score, config.FUSION_THRESHOLD, config.FUSION_FLOOR,
        "FUSION_THRESHOLD / FUSION_FLOOR",
    )


def gate_dense(score: float) -> GateDecision:
    """The Step 4 dense-only gate, on raw cosine, kept as the hybrid baseline.

    Retained so calibrate_gate.py can report before/after against the same
    questions. Production answering goes through gate().
    """
    return _classify(
        score, config.SIMILARITY_THRESHOLD, config.SIMILARITY_FLOOR,
        "SIMILARITY_THRESHOLD / SIMILARITY_FLOOR",
    )


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int | None = None
) -> list[tuple[int, float]]:
    """Fuse several ranked id lists into one, by sum of 1/(k + rank).

    RRF deliberately ignores each ranker's raw scores and uses only positions,
    which is what makes it safe here: BM25 scores are unbounded sums of IDF
    terms and cosine scores sit in [-1, 1], so no fixed weighting of the two
    would be stable across queries. Rank is the one comparable quantity.

    Ranks are 1-based. A document missing from a ranking simply contributes
    nothing for that ranker rather than being penalised explicitly.
    """
    k = config.RRF_K if k is None else k
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Ties broken by id for determinism -- without it, equal-scoring chunks
    # would reorder between runs and the calibration would not reproduce.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


# --------------------------------------------------------------------------
# Fusion confidence -- the signal the gate actually thresholds
# --------------------------------------------------------------------------


def idf_coverage(bm25: BM25Index, query: str, texts: list[str]) -> float:
    """IDF-weighted share of the query's terms that appear in the retrieved texts.

    Answers "how much of what the user actually asked did we find?", weighted so
    that rare, discriminating words count for far more than common ones. A term
    absent from the corpus entirely is weighted at the maximum IDF and can never
    be covered, so a query built around vocabulary this corpus does not have
    scores near zero however fluent the rest of it looks.

    Returns a value in [0, 1].
    """
    terms = set(tokenize(query))
    if not terms or not bm25.ids:
        return 0.0
    max_idf = math.log(1.0 + (len(bm25.ids) + 0.5) / 0.5)

    found = set()
    for text in texts:
        found |= set(tokenize(text))

    covered = total = 0.0
    for term in terms:
        df = len(bm25.postings.get(term, ()))
        weight = max_idf if df == 0 else bm25.idf(term)
        total += weight
        if term in found:
            covered += weight
    return covered / total if total else 0.0


def fusion_confidence(dense_score: float, coverage: float) -> float:
    """The gate signal: dense semantic similarity scaled by lexical coverage.

    NOT the RRF score. RRF is an excellent ranker and a useless confidence
    measure -- it is computed from rank positions alone, so a top-1 result
    scores ~2/(k+1) whenever both rankers agree and ~1/(k+1) when only one
    does, almost regardless of whether the corpus contains the answer. Measured
    over the 21 calibration questions the RRF score separated the answerable
    from the not-answerable classes *worse* than raw cosine did (see
    scripts/eval_gate_signals.py).

    This product keeps both halves of the hybrid honest. Dense similarity
    alone cannot tell that "doğal gaz" is the one word that matters; lexical
    coverage alone rewards keyword overlap without understanding. Multiplying
    means a result must be both semantically close AND actually contain the
    terms asked about. Chosen by measurement, not taste: of six candidate
    signals it scored the best separation (Youden's J 0.767 vs 0.500 for
    dense-only) and was the most stable under ASCII folding.
    """
    return dense_score * coverage


# --------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize, leaving all-zero rows as zeros rather than producing NaN."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@dataclass
class Retriever:
    conn: sqlite3.Connection
    embedder: EmbeddingClient
    matrix: np.ndarray  # (N, EMBEDDING_DIM) float32, L2-normalized
    ids: list[int]
    load_seconds: float = 0.0
    bm25: BM25Index | None = None
    bm25_load_seconds: float = 0.0

    @classmethod
    def open(
        cls,
        db_path: str | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> "Retriever":
        """Connect, load every active embedding into memory, normalize, and time it."""
        started = time.perf_counter()
        conn = store.connect(db_path)
        matrix, ids = store.fetch_active_embeddings(conn)
        matrix = _l2_normalize(np.ascontiguousarray(matrix, dtype=np.float32))
        load_seconds = time.perf_counter() - started

        bm25_started = time.perf_counter()
        bm25 = BM25Index.from_connection(conn)
        bm25_load_seconds = time.perf_counter() - bm25_started

        embedder = embedder if embedder is not None else EmbeddingClient.connect()
        if embedder.dimension != config.EMBEDDING_DIM:
            raise store.DimensionMismatch(
                f"embedding client reports dimension {embedder.dimension} but "
                f"config.EMBEDDING_DIM is {config.EMBEDDING_DIM}"
            )
        if matrix.shape[0] and matrix.shape[1] != config.EMBEDDING_DIM:
            raise store.DimensionMismatch(
                f"stored matrix is {matrix.shape[1]}-dim, config says {config.EMBEDDING_DIM}"
            )
        return cls(
            conn=conn, embedder=embedder, matrix=matrix, ids=ids,
            load_seconds=load_seconds, bm25=bm25, bm25_load_seconds=bm25_load_seconds,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query with the same model/dimension the corpus was built with."""
        vector = np.asarray(self.embedder.embed_batch([query])[0], dtype=np.float32)
        if vector.shape[0] != config.EMBEDDING_DIM:
            raise store.DimensionMismatch(
                f"query embedding has {vector.shape[0]} dims, "
                f"config.EMBEDDING_DIM is {config.EMBEDDING_DIM}"
            )
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0 else vector / norm

    def search_vector(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Rank the in-memory matrix against an already-normalized query vector.

        Returns (chunk_id, cosine score) pairs, highest first. argpartition
        keeps this O(N) rather than sorting all 27k scores for a top-5.
        """
        if self.matrix.shape[0] == 0 or top_k <= 0:
            return []
        scores = self.matrix @ query_vector
        k = min(top_k, scores.shape[0])
        candidates = np.argpartition(-scores, k - 1)[:k]
        candidates = candidates[np.argsort(-scores[candidates])]
        return [(self.ids[i], float(scores[i])) for i in candidates]

    def _hydrate(self, hits: list[tuple[int, float]], **per_id) -> list[RetrievalResult]:
        """Turn (chunk_id, score) pairs into full results, one SQLite lookup each."""
        results: list[RetrievalResult] = []
        for chunk_id, score in hits:
            meta = store.fetch_chunk_metadata(self.conn, chunk_id)
            if meta is None:  # only reachable if the row vanished mid-session
                continue
            provenance = {key: mapping.get(chunk_id) for key, mapping in per_id.items()}
            results.append(RetrievalResult.from_metadata(meta, score, **provenance))
        return results

    # -- individual rankers --------------------------------------------------

    def retrieve_dense(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Dense-only retrieval -- the Step 4 baseline that hybrid must beat.

        Kept callable deliberately: "hybrid is better" is a claim that needs a
        baseline to be measured against, not an assumption.
        """
        k = config.TOP_K if top_k is None else top_k
        hits = self.search_vector(self.embed_query(query), k)
        return self._hydrate(hits, dense_score=dict(hits))

    def retrieve_lexical(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """BM25-only retrieval. Empty when no query term appears in the corpus."""
        k = config.TOP_K if top_k is None else top_k
        if self.bm25 is None:
            return []
        hits = self.bm25.search(query, k)
        return self._hydrate(hits, lexical_score=dict(hits))

    # -- fusion --------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Fused dense + BM25 retrieval. This is the production path."""
        results, _ = self.retrieve_fused_timed(query, top_k)
        return results

    def retrieve_fused_timed(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[RetrievalResult], dict[str, float]]:
        """retrieve(), plus a timing breakdown for latency reporting.

        The two rankers run concurrently: the dense side is dominated by a
        blocking HTTP call to Foundry Local (~30 ms) which releases the GIL, so
        BM25's CPU work overlaps it almost entirely instead of adding to it.
        """
        k = config.TOP_K if top_k is None else top_k
        depth = max(config.FUSION_CANDIDATES, k)

        timings: dict[str, float] = {}
        started = time.perf_counter()

        dense_hits: list[tuple[int, float]] = []
        lexical_hits: list[tuple[int, float]] = []

        def run_dense():
            t = time.perf_counter()
            hits = self.search_vector(self.embed_query(query), depth)
            timings["dense"] = time.perf_counter() - t
            return hits

        def run_lexical():
            t = time.perf_counter()
            hits = self.bm25.search(query, depth) if self.bm25 is not None else []
            timings["lexical"] = time.perf_counter() - t
            return hits

        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future = pool.submit(run_dense)
            lexical_future = pool.submit(run_lexical)
            dense_hits = dense_future.result()
            lexical_hits = lexical_future.result()

        timings["retrieval"] = time.perf_counter() - started

        fuse_started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in dense_hits], [cid for cid, _ in lexical_hits]]
        )[:k]
        timings["fuse"] = time.perf_counter() - fuse_started

        hydrate_started = time.perf_counter()
        results = self._hydrate(
            fused,
            dense_score=dict(dense_hits),
            lexical_score=dict(lexical_hits),
            dense_rank={cid: i for i, (cid, _) in enumerate(dense_hits, start=1)},
            lexical_rank={cid: i for i, (cid, _) in enumerate(lexical_hits, start=1)},
        )
        timings["hydrate"] = time.perf_counter() - hydrate_started
        timings["total"] = time.perf_counter() - started
        return results, timings

    def confidence(self, query: str, results: list[RetrievalResult]) -> float:
        """The gate signal for a completed retrieval. See fusion_confidence()."""
        if not results or self.bm25 is None:
            return 0.0
        dense_top = results[0].dense_score
        if dense_top is None:
            # Fusion's #1 came from BM25 alone; use the best dense score present.
            dense_top = max((r.dense_score or 0.0) for r in results)
        coverage = idf_coverage(self.bm25, query, [r.text for r in results])
        return fusion_confidence(dense_top, coverage)

    def answer(
        self, query: str, top_k: int | None = None
    ) -> tuple[GateDecision, float, list[RetrievalResult]]:
        """End to end: fused retrieval, confidence, gate decision.

        The one call a caller (generation, UI) should need -- it is not possible
        to get results without also getting the decision about whether they
        should be trusted.
        """
        results = self.retrieve(query, top_k)
        score = self.confidence(query, results)
        return gate(score), score, results

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def format_results(results: list[RetrievalResult], text_chars: int = 400) -> str:
    lines: list[str] = []
    for rank, r in enumerate(results, start=1):
        lines.append(f"\n[{rank}] score={r.score:.5f}  {r.citation()}")
        provenance = []
        if r.dense_rank is not None:
            provenance.append(f"dense #{r.dense_rank} ({r.dense_score:.4f})")
        else:
            provenance.append("dense --")
        if r.lexical_rank is not None:
            provenance.append(f"bm25 #{r.lexical_rank} ({r.lexical_score:.2f})")
        else:
            provenance.append("bm25 --")
        lines.append(f"    ranked : {', '.join(provenance)}")
        lines.append(f"    source : {r.source_path}  (chunk {r.chunk_index})")
        if r.quality_flag:
            lines.append(f"    quality: {r.quality_flag}")
        snippet = " ".join(r.text.split())
        if len(snippet) > text_chars:
            snippet = snippet[:text_chars] + " ..."
        lines.append(f"    {snippet}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .embed import FoundryUnavailable
    from .extract import _force_utf8_stdout

    _force_utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m src.retrieval")
    parser.add_argument("query", help="the question to retrieve for")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--db", default=None, help=f"SQLite path (default: {config.DB_PATH})")
    parser.add_argument("--chars", type=int, default=400, help="chunk text preview length")
    parser.add_argument(
        "--dense-only", action="store_true",
        help="use the Step 4 dense-only path and its SIMILARITY_* cutoffs (baseline)",
    )
    args = parser.parse_args(argv)

    try:
        retriever = Retriever.open(args.db)
    except FoundryUnavailable as exc:
        print(f"FATAL: {exc}")
        return 3

    print(
        f"Loaded {len(retriever):,} active chunk vectors "
        f"({retriever.matrix.nbytes / 1e6:.1f} MB) in {retriever.load_seconds:.2f}s"
    )
    print(
        f"BM25 index: {len(retriever.bm25):,} docs, "
        f"{retriever.bm25.vocabulary_size:,} terms in {retriever.bm25_load_seconds:.2f}s"
    )
    print(f"Embedding model: {retriever.embedder.model_id} (dim {retriever.embedder.dimension})")

    if args.dense_only:
        results = retriever.retrieve_dense(args.query, args.top_k)
        timings = {}
        top_score = results[0].score if results else 0.0
        decision = gate_dense(top_score)
        cutoffs = f"threshold {config.SIMILARITY_THRESHOLD}, floor {config.SIMILARITY_FLOOR}"
        mode = "DENSE-ONLY (Step 4 baseline)"
    else:
        results, timings = retriever.retrieve_fused_timed(args.query, args.top_k)
        top_score = retriever.confidence(args.query, results)
        decision = gate(top_score)
        cutoffs = f"threshold {config.FUSION_THRESHOLD}, floor {config.FUSION_FLOOR}"
        mode = f"HYBRID (RRF k={config.RRF_K}, depth {config.FUSION_CANDIDATES})"

    print("\n" + "=" * 72)
    print(f"QUERY   : {args.query}")
    print(f"MODE    : {mode}")
    print(f"DECISION: {decision}  (top score {top_score:.5f}; {cutoffs})")
    if timings:
        print(
            f"LATENCY : dense {timings['dense'] * 1000:.0f} ms || "
            f"bm25 {timings['lexical'] * 1000:.0f} ms (concurrent) "
            f"-> fuse {timings['fuse'] * 1000:.1f} ms "
            f"-> total {timings['total'] * 1000:.0f} ms"
        )
    print("=" * 72)
    print(format_results(results, args.chars))
    print()
    retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
