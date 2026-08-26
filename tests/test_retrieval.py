"""Tests for dense retrieval, the confidence gate, and the metadata contract.

No Foundry Local dependency: the embedding client is faked, and the corpus is a
handful of hand-built vectors whose cosine ordering is known by construction.
The point of these tests is the plumbing -- ranking order, gate boundaries, and
that nothing is dropped between a SQLite row and a RetrievalResult -- not
embedding quality, which is what scripts/calibrate_gate.py measures instead.
"""
import numpy as np
import pytest

from src import config, retrieval, store
from src.chunk import ArticleRef, Chunk
from src.lexical import BM25Index
from src.retrieval import RetrievalResult, Retriever, gate, gate_dense


class FakeEmbedder:
    """Returns a vector chosen per query string, so ranking is fully determined."""

    def __init__(self, responses: dict[str, np.ndarray] | None = None):
        self.dimension = config.EMBEDDING_DIM
        self.model_id = "fake-embedder"
        self.responses = responses or {}
        self.calls: list[list[str]] = []

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            if text in self.responses:
                out.append(list(self.responses[text]))
            else:
                out.append(list(unit(0)))
        return out


def unit(axis: int, dim: int | None = None) -> np.ndarray:
    """A one-hot basis vector -- cosine against another basis vector is 0 or 1."""
    vector = np.zeros(dim or config.EMBEDDING_DIM, dtype=np.float32)
    vector[axis] = 1.0
    return vector


def blend(a: int, b: int, wa: float, wb: float) -> np.ndarray:
    vector = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    vector[a] = wa
    vector[b] = wb
    return vector


def make_chunk(index=0, text="MADDE 1 - Test hükmü.", article=None, page_start=None, page_end=None):
    return Chunk(
        doc_id="deadbeef-test",
        text=text,
        strategy="article",
        index=index,
        article=article,
        document_title="TEST KANUNU",
        page_start=page_start,
        page_end=page_end,
    )


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "test.db")
    yield c
    c.close()


def build_retriever(conn, embedder=None) -> Retriever:
    """A Retriever over whatever `conn` already holds, bypassing Foundry discovery.

    The BM25 index is built from the same rows, so fused paths work too.
    """
    matrix, ids = store.fetch_active_embeddings(conn)
    matrix = retrieval._l2_normalize(np.ascontiguousarray(matrix, dtype=np.float32))
    return Retriever(
        conn=conn,
        embedder=embedder or FakeEmbedder(),
        matrix=matrix,
        ids=ids,
        load_seconds=0.0,
        bm25=BM25Index.from_connection(conn),
    )


# --------------------------------------------------------------------------
# Cosine ranking
# --------------------------------------------------------------------------


class TestCosineRanking:
    """A small synthetic set whose correct ordering is known before we run anything."""

    @pytest.fixture
    def ranked_conn(self, conn):
        # Cosine against unit(0) is, by construction: 1.0, 0.8, 0.6, 0.0.
        vectors = [
            ("perfect", unit(0)),
            ("strong", blend(0, 1, 0.8, 0.6)),
            ("weak", blend(0, 1, 0.6, 0.8)),
            ("orthogonal", unit(1)),
        ]
        for i, (name, vector) in enumerate(vectors):
            store.insert_chunks(
                conn,
                source_path=f"docs/{name}.doc",
                file_sha256=f"hash-{name}",
                document_title=name.upper(),
                chunks=[make_chunk(index=0, text=f"text for {name}")],
                embeddings=[vector],
                quality_flag=None,
            )
        return conn

    def test_results_are_ordered_by_descending_similarity(self, ranked_conn):
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        results = r.retrieve_dense("q", top_k=4)
        assert [x.document_title for x in results] == ["PERFECT", "STRONG", "WEAK", "ORTHOGONAL"]
        scores = [x.score for x in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_true_cosines_not_raw_dot_products(self, ranked_conn):
        # Stored vectors have norms 1.0 and 1.0 (0.8,0.6 blends); an unnormalized
        # dot product would only coincide with cosine by luck, so check values.
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        results = r.retrieve_dense("q", top_k=4)
        assert results[0].score == pytest.approx(1.0, abs=1e-6)
        assert results[1].score == pytest.approx(0.8, abs=1e-6)
        assert results[2].score == pytest.approx(0.6, abs=1e-6)
        assert results[3].score == pytest.approx(0.0, abs=1e-6)

    def test_query_vector_magnitude_does_not_change_ranking_or_scores(self, ranked_conn):
        # A 100x longer query vector must produce identical cosines.
        big = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0) * 100.0}))
        small = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        assert [x.score for x in big.retrieve_dense("q", top_k=4)] == pytest.approx(
            [x.score for x in small.retrieve_dense("q", top_k=4)], abs=1e-6
        )

    def test_top_k_limits_result_count_and_keeps_the_best(self, ranked_conn):
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        results = r.retrieve_dense("q", top_k=2)
        assert len(results) == 2
        assert [x.document_title for x in results] == ["PERFECT", "STRONG"]

    def test_top_k_larger_than_corpus_returns_everything_without_error(self, ranked_conn):
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        assert len(r.retrieve_dense("q", top_k=99)) == 4

    def test_top_k_zero_returns_nothing(self, ranked_conn):
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        assert r.retrieve_dense("q", top_k=0) == []

    def test_empty_corpus_returns_no_results_rather_than_raising(self, conn):
        r = build_retriever(conn, FakeEmbedder({"q": unit(0)}))
        assert r.matrix.shape == (0, config.EMBEDDING_DIM)
        assert r.retrieve_dense("q", top_k=5) == []

    def test_inactive_chunks_are_never_retrieved(self, ranked_conn):
        store.mark_document_inactive(ranked_conn, "docs/perfect.doc")
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        results = r.retrieve_dense("q", top_k=4)
        assert [x.document_title for x in results] == ["STRONG", "WEAK", "ORTHOGONAL"]

    def test_ties_do_not_drop_or_duplicate_results(self, conn):
        for i in range(3):
            store.insert_chunks(
                conn,
                source_path=f"docs/tie{i}.doc",
                file_sha256=f"hash-tie{i}",
                document_title=f"TIE{i}",
                chunks=[make_chunk(index=0)],
                embeddings=[unit(0)],
                quality_flag=None,
            )
        r = build_retriever(conn, FakeEmbedder({"q": unit(0)}))
        results = r.retrieve_dense("q", top_k=3)
        assert len(results) == 3
        assert len({x.chunk_id for x in results}) == 3
        assert all(x.score == pytest.approx(1.0, abs=1e-6) for x in results)

    def test_zero_vector_in_corpus_yields_zero_score_not_nan(self, conn):
        store.insert_chunks(
            conn,
            source_path="docs/zero.doc", file_sha256="hash-zero", document_title="ZERO",
            chunks=[make_chunk()],
            embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)],
            quality_flag=None,
        )
        r = build_retriever(conn, FakeEmbedder({"q": unit(0)}))
        results = r.retrieve_dense("q", top_k=1)
        assert not np.isnan(results[0].score)
        assert results[0].score == pytest.approx(0.0)

    def test_only_top_candidates_are_looked_up_in_sqlite(self, ranked_conn, monkeypatch):
        # The whole design premise: vectors come from RAM, SQLite is touched
        # only for the chunks that actually made top_k.
        calls = []
        real = store.fetch_chunk_metadata
        monkeypatch.setattr(
            store, "fetch_chunk_metadata",
            lambda c, cid: (calls.append(cid), real(c, cid))[1],
        )
        r = build_retriever(ranked_conn, FakeEmbedder({"q": unit(0)}))
        r.retrieve_dense("q", top_k=2)
        assert len(calls) == 2


# --------------------------------------------------------------------------
# Confidence gate boundaries
# --------------------------------------------------------------------------


class TestGateBoundaries:
    """Exact-boundary behavior of the production gate, pinned against config.

    gate() thresholds the FUSION confidence signal; gate_dense() is the Step 4
    dense-only baseline on raw cosine. Each is tested against its own cutoffs --
    the two scales are not interchangeable.
    """

    def test_exactly_at_threshold_is_answer(self):
        assert gate(config.FUSION_THRESHOLD) == "ANSWER"

    def test_just_above_threshold_is_answer(self):
        assert gate(config.FUSION_THRESHOLD + 1e-9) == "ANSWER"

    def test_just_below_threshold_is_answer_weak(self):
        assert gate(config.FUSION_THRESHOLD - 1e-6) == "ANSWER_WEAK"

    def test_exactly_at_floor_is_answer_weak(self):
        assert gate(config.FUSION_FLOOR) == "ANSWER_WEAK"

    def test_just_above_floor_is_answer_weak(self):
        assert gate(config.FUSION_FLOOR + 1e-9) == "ANSWER_WEAK"

    def test_just_below_floor_is_not_found(self):
        assert gate(config.FUSION_FLOOR - 1e-6) == "NOT_FOUND"

    def test_midband_is_answer_weak(self):
        mid = (config.FUSION_THRESHOLD + config.FUSION_FLOOR) / 2
        assert gate(mid) == "ANSWER_WEAK"

    def test_extremes(self):
        assert gate(1.0) == "ANSWER"
        assert gate(0.0) == "NOT_FOUND"
        assert gate(-1.0) == "NOT_FOUND"

    def test_configured_values_are_ordered_and_calibrated(self):
        assert config.FUSION_FLOOR is not None
        assert config.FUSION_THRESHOLD is not None
        assert config.FUSION_FLOOR < config.FUSION_THRESHOLD

    def test_gate_refuses_to_guess_when_config_is_uncalibrated(self, monkeypatch):
        monkeypatch.setattr(config, "FUSION_THRESHOLD", None)
        with pytest.raises(ValueError, match="calibrate_gate"):
            gate(0.9)

    def test_dense_baseline_gate_uses_its_own_cutoffs(self):
        assert gate_dense(config.SIMILARITY_THRESHOLD) == "ANSWER"
        assert gate_dense(config.SIMILARITY_FLOOR) == "ANSWER_WEAK"
        assert gate_dense(config.SIMILARITY_FLOOR - 1e-6) == "NOT_FOUND"

    def test_the_two_gates_are_on_different_scales(self):
        """A fused score fed to gate_dense() would be a category error -- prove
        the cutoffs actually differ so nobody "simplifies" them into one."""
        assert config.FUSION_THRESHOLD != config.SIMILARITY_THRESHOLD
        assert gate(config.FUSION_THRESHOLD) == "ANSWER"
        assert gate_dense(config.FUSION_THRESHOLD) == "NOT_FOUND"

    def test_all_three_decisions_are_reachable_with_configured_values(self, monkeypatch):
        monkeypatch.setattr(config, "FUSION_THRESHOLD", 0.7)
        monkeypatch.setattr(config, "FUSION_FLOOR", 0.4)
        assert gate(0.75) == "ANSWER"
        assert gate(0.55) == "ANSWER_WEAK"
        assert gate(0.30) == "NOT_FOUND"


# --------------------------------------------------------------------------
# Metadata contract: store.py -> RetrievalResult
# --------------------------------------------------------------------------


class TestMetadataContract:
    """Every field the chunks table holds must survive into RetrievalResult unchanged."""

    FIELDS = (
        "source_path", "file_sha256", "document_title", "chunk_index",
        "text", "article_ref", "page_start", "page_end", "quality_flag",
    )

    @pytest.fixture
    def populated(self, conn):
        chunk = make_chunk(
            index=7,
            text="MADDE 12 – (1) Önlisans süresi yirmi dört ayı geçemez.",
            article=ArticleRef("GEÇİCİ MADDE", "12/A"),
            page_start=3,
            page_end=5,
        )
        store.insert_chunks(
            conn,
            source_path="mevzuat/raw/Yonetmelikler/lisans.doc",
            file_sha256="0f1e2d3c",
            document_title="ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ",
            chunks=[chunk],
            embeddings=[unit(0)],
            quality_flag="rg-missing: no Resmî Gazete reference found",
        )
        return conn

    def test_every_stored_field_reaches_the_result_unchanged(self, populated):
        r = build_retriever(populated, FakeEmbedder({"q": unit(0)}))
        result = r.retrieve_dense("q", top_k=1)[0]
        meta = store.fetch_chunk_metadata(populated, result.chunk_id)
        for field in self.FIELDS:
            assert getattr(result, field) == meta[field], f"{field} was altered in transit"

    def test_result_exposes_every_metadata_field_the_store_returns(self, populated):
        """Guards against store.py gaining a column that RetrievalResult silently drops."""
        r = build_retriever(populated, FakeEmbedder({"q": unit(0)}))
        result = r.retrieve_dense("q", top_k=1)[0]
        meta = store.fetch_chunk_metadata(populated, result.chunk_id)
        # `id` is renamed to chunk_id; `active` is implied (only active rows load).
        expected = set(meta) - {"id", "active", "embedded_at"}
        assert expected <= set(vars(result))

    def test_turkish_text_and_article_ref_survive_verbatim(self, populated):
        r = build_retriever(populated, FakeEmbedder({"q": unit(0)}))
        result = r.retrieve_dense("q", top_k=1)[0]
        assert result.text == "MADDE 12 – (1) Önlisans süresi yirmi dört ayı geçemez."
        assert result.article_ref == "GEÇİCİ MADDE 12/A"
        assert result.document_title == "ELEKTRİK PİYASASI LİSANS YÖNETMELİĞİ"

    def test_quality_flag_is_not_dropped(self, populated):
        # A caller must be able to see that the underlying extraction was suspect.
        r = build_retriever(populated, FakeEmbedder({"q": unit(0)}))
        assert r.retrieve_dense("q", top_k=1)[0].quality_flag == (
            "rg-missing: no Resmî Gazete reference found"
        )

    def test_page_range_and_chunk_index_survive(self, populated):
        r = build_retriever(populated, FakeEmbedder({"q": unit(0)}))
        result = r.retrieve_dense("q", top_k=1)[0]
        assert (result.page_start, result.page_end, result.chunk_index) == (3, 5, 7)

    def test_nulls_stay_null_rather_than_becoming_empty_strings(self, conn):
        store.insert_chunks(
            conn,
            source_path="p.doc", file_sha256="h", document_title=None,
            chunks=[make_chunk()], embeddings=[unit(0)], quality_flag=None,
        )
        r = build_retriever(conn, FakeEmbedder({"q": unit(0)}))
        result = r.retrieve_dense("q", top_k=1)[0]
        assert result.document_title is None
        assert result.article_ref is None
        assert result.page_start is None
        assert result.quality_flag is None

    def test_score_is_a_plain_float_not_a_numpy_scalar(self, populated):
        r = build_retriever(populated, FakeEmbedder({"q": unit(0)}))
        assert type(r.retrieve_dense("q", top_k=1)[0].score) is float


class TestCitationRendering:
    def _result(self, **overrides):
        base = dict(
            chunk_id=1, score=0.7, text="t", article_ref="MADDE 6",
            document_title="ELEKTRİK PİYASASI KANUNU", page_start=None, page_end=None,
            quality_flag=None, source_path="mevzuat/raw/x.doc", chunk_index=0,
            file_sha256="h",
        )
        base.update(overrides)
        return RetrievalResult(**base)

    def test_title_and_article(self):
        assert self._result().citation() == "ELEKTRİK PİYASASI KANUNU / MADDE 6"

    def test_page_range_included_when_present(self):
        cite = self._result(page_start=3, page_end=5).citation()
        assert cite.endswith("s. 3-5")

    def test_single_page_not_rendered_as_a_range(self):
        assert self._result(page_start=4, page_end=4).citation().endswith("s. 4")
        assert self._result(page_start=4, page_end=None).citation().endswith("s. 4")

    def test_falls_back_to_source_path_when_title_is_null(self):
        cite = self._result(document_title=None, article_ref=None).citation()
        assert cite == "mevzuat/raw/x.doc"


class TestQueryEmbedding:
    def test_query_is_normalized_before_search(self, conn):
        r = build_retriever(conn, FakeEmbedder({"q": unit(0) * 42.0}))
        assert np.linalg.norm(r.embed_query("q")) == pytest.approx(1.0, abs=1e-6)

    def test_wrong_dimension_query_embedding_raises(self, conn):
        embedder = FakeEmbedder({"q": np.ones(config.EMBEDDING_DIM + 3, dtype=np.float32)})
        r = build_retriever(conn, embedder)
        with pytest.raises(store.DimensionMismatch):
            r.embed_query("q")

    def test_query_text_is_passed_through_verbatim(self, conn):
        embedder = FakeEmbedder({"Önlisans süresi?": unit(0)})
        r = build_retriever(conn, embedder)
        r.embed_query("Önlisans süresi?")
        assert embedder.calls == [["Önlisans süresi?"]]
