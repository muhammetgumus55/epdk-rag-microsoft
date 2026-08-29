"""Tests for the SQLite chunk/embedding store."""
import threading

import numpy as np
import pytest

from src import config, store
from src.chunk import ArticleRef, Chunk


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


class TestSchemaAndRoundTrip:
    def test_insert_and_read_back_chunk_content_and_vector(self, conn):
        rng = np.random.default_rng(0)
        vector = rng.random(config.EMBEDDING_DIM, dtype=np.float32)
        chunk = make_chunk(article=ArticleRef("MADDE", "1"))

        inserted = store.insert_chunks(
            conn,
            source_path="Kanunlar/test.doc",
            file_sha256="abc123",
            document_title="TEST KANUNU",
            chunks=[chunk],
            embeddings=[vector],
            quality_flag=None,
        )
        assert inserted == 1

        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape == (1, config.EMBEDDING_DIM)
        np.testing.assert_array_equal(matrix[0], vector)

        meta = store.fetch_chunk_metadata(conn, ids[0])
        assert meta["text"] == chunk.text
        assert meta["document_title"] == "TEST KANUNU"
        assert meta["article_ref"] == "MADDE 1"
        assert meta["file_sha256"] == "abc123"
        assert meta["source_path"] == "Kanunlar/test.doc"
        assert meta["active"] == 1

    def test_vector_round_trips_exactly(self, conn):
        # float32 must survive serialize/deserialize bit-for-bit, not just "close".
        rng = np.random.default_rng(1)
        vector = rng.standard_normal(config.EMBEDDING_DIM).astype(np.float32)
        store.insert_chunks(
            conn,
            source_path="p", file_sha256="h1", document_title=None,
            chunks=[make_chunk()], embeddings=[vector], quality_flag=None,
        )
        matrix, _ = store.fetch_active_embeddings(conn)
        assert (matrix[0] == vector).all()

    def test_missing_chunk_id_returns_none(self, conn):
        assert store.fetch_chunk_metadata(conn, 99999) is None

    def test_nullable_fields_stay_null(self, conn):
        store.insert_chunks(
            conn,
            source_path="p", file_sha256="h2", document_title=None,
            chunks=[make_chunk()], embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)],
            quality_flag=None,
        )
        matrix, ids = store.fetch_active_embeddings(conn)
        meta = store.fetch_chunk_metadata(conn, ids[0])
        assert meta["document_title"] is None
        assert meta["article_ref"] is None
        assert meta["page_start"] is None
        assert meta["quality_flag"] is None

    def test_quality_flag_persisted_when_present(self, conn):
        store.insert_chunks(
            conn,
            source_path="p", file_sha256="h3", document_title="X",
            chunks=[make_chunk()], embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)],
            quality_flag="near-empty: only 80 chars extracted",
        )
        matrix, ids = store.fetch_active_embeddings(conn)
        meta = store.fetch_chunk_metadata(conn, ids[0])
        assert meta["quality_flag"] == "near-empty: only 80 chars extracted"

    def test_page_range_persisted(self, conn):
        store.insert_chunks(
            conn,
            source_path="p", file_sha256="h4", document_title="X",
            chunks=[make_chunk(page_start=3, page_end=5)],
            embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)],
            quality_flag=None,
        )
        matrix, ids = store.fetch_active_embeddings(conn)
        meta = store.fetch_chunk_metadata(conn, ids[0])
        assert meta["page_start"] == 3
        assert meta["page_end"] == 5


class TestActiveInactiveFiltering:
    def test_marking_document_inactive_excludes_it_from_active_embeddings(self, conn):
        v1 = np.ones(config.EMBEDDING_DIM, dtype=np.float32)
        v2 = np.full(config.EMBEDDING_DIM, 2.0, dtype=np.float32)
        store.insert_chunks(
            conn, source_path="doc-a", file_sha256="hash-a", document_title="A",
            chunks=[make_chunk(index=0)], embeddings=[v1], quality_flag=None,
        )
        store.insert_chunks(
            conn, source_path="doc-b", file_sha256="hash-b", document_title="B",
            chunks=[make_chunk(index=0)], embeddings=[v2], quality_flag=None,
        )

        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape[0] == 2

        affected = store.mark_document_inactive(conn, "doc-a")
        assert affected == 1

        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape[0] == 1
        np.testing.assert_array_equal(matrix[0], v2)

        meta = store.fetch_chunk_metadata(conn, ids[0])
        assert meta["source_path"] == "doc-b"

    def test_has_active_chunks_reflects_inactive_state(self, conn):
        store.insert_chunks(
            conn, source_path="doc-a", file_sha256="hash-a", document_title="A",
            chunks=[make_chunk()], embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)],
            quality_flag=None,
        )
        assert store.has_active_chunks(conn, "hash-a") is True
        store.mark_document_inactive(conn, "doc-a")
        assert store.has_active_chunks(conn, "hash-a") is False

    def test_reingest_with_changed_hash_retires_old_version(self, conn):
        # Same source_path, content changed -> old hash's chunks go inactive.
        old_vec = np.ones(config.EMBEDDING_DIM, dtype=np.float32)
        new_vec = np.full(config.EMBEDDING_DIM, 9.0, dtype=np.float32)
        store.insert_chunks(
            conn, source_path="doc-a", file_sha256="hash-v1", document_title="A v1",
            chunks=[make_chunk()], embeddings=[old_vec], quality_flag=None,
        )
        prior = store.active_hash_for_path(conn, "doc-a")
        assert prior == "hash-v1"
        store.mark_document_inactive(conn, "doc-a")
        store.insert_chunks(
            conn, source_path="doc-a", file_sha256="hash-v2", document_title="A v2",
            chunks=[make_chunk()], embeddings=[new_vec], quality_flag=None,
        )

        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape[0] == 1
        np.testing.assert_array_equal(matrix[0], new_vec)
        assert store.active_hash_for_path(conn, "doc-a") == "hash-v2"

    def test_empty_store_returns_empty_matrix_with_correct_width(self, conn):
        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape == (0, config.EMBEDDING_DIM)
        assert ids == []


class TestResumability:
    def test_existing_chunk_indices_reflects_what_is_stored(self, conn):
        assert store.existing_chunk_indices(conn, "hash-x") == set()
        store.insert_chunks(
            conn, source_path="p", file_sha256="hash-x", document_title="X",
            chunks=[make_chunk(index=0), make_chunk(index=1)],
            embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)] * 2,
            quality_flag=None,
        )
        assert store.existing_chunk_indices(conn, "hash-x") == {0, 1}

    def test_reinserting_same_index_is_ignored_not_duplicated(self, conn):
        # UNIQUE(file_sha256, chunk_index) must make a crash-then-rerun safe.
        chunk = make_chunk(index=0)
        v = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        store.insert_chunks(
            conn, source_path="p", file_sha256="hash-y", document_title="X",
            chunks=[chunk], embeddings=[v], quality_flag=None,
        )
        second = store.insert_chunks(
            conn, source_path="p", file_sha256="hash-y", document_title="X",
            chunks=[chunk], embeddings=[v], quality_flag=None,
        )
        assert second == 0
        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape[0] == 1

    def test_partial_resume_only_inserts_missing_indices(self, conn):
        v = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
        # Simulate a crash after chunk 0 succeeded but before chunk 1 did.
        store.insert_chunks(
            conn, source_path="p", file_sha256="hash-z", document_title="X",
            chunks=[make_chunk(index=0)], embeddings=[v], quality_flag=None,
        )
        done = store.existing_chunk_indices(conn, "hash-z")
        all_chunks = [make_chunk(index=0), make_chunk(index=1)]
        todo = [c for c in all_chunks if c.index not in done]
        assert [c.index for c in todo] == [1]


class TestDimensionMismatch:
    def test_stored_dimension_mismatch_raises_not_silently_proceeds(self, conn):
        # Insert a vector with the WRONG dimension by bypassing insert_chunks'
        # normal path (which always uses config.EMBEDDING_DIM-sized input in
        # practice) to simulate a store built under a different config.
        wrong_dim_vector = np.zeros(config.EMBEDDING_DIM + 7, dtype=np.float32)
        conn.execute(
            """
            INSERT INTO chunks
                (source_path, file_sha256, document_title, chunk_index, text,
                 article_ref, page_start, page_end, quality_flag, embedding, embedded_at, active)
            VALUES ('p', 'bad-hash', 'X', 0, 'text', NULL, NULL, NULL, NULL, ?, '2024-01-01', 1)
            """,
            (wrong_dim_vector.tobytes(),),
        )
        conn.commit()

        with pytest.raises(store.DimensionMismatch):
            store.fetch_active_embeddings(conn)


class TestIngestDocumentLogic:
    """Exercises ingest_document's resume/re-ingest branching with a fake embedder."""

    class FakeEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_batch(self, texts):
            self.calls += 1
            return [np.full(config.EMBEDDING_DIM, len(t), dtype=np.float32) for t in texts]

    def _fake_doc(self, tmp_path, text, name="fake.docx"):
        from src.extract import ExtractedDoc, Page

        path = tmp_path / name
        path.write_text("placeholder", encoding="utf-8")
        doc = ExtractedDoc(
            path=path,
            original_filename=name,
            doc_id="fakehash-fake",
            file_sha256="fakehash",
            detected_type="docx",
            page_number_note="docx has no fixed page model",
        )
        doc.pages = [Page(number=None, text=text)]
        return doc

    def test_second_ingest_of_unchanged_document_embeds_nothing(self, conn, tmp_path):
        doc = self._fake_doc(tmp_path, "MADDE 1 - Birinci hüküm.\nMADDE 2 - İkinci hüküm.")
        embedder = self.FakeEmbedder()

        first = store.ingest_document(conn, embedder, doc, tmp_path)
        assert first.embedded == 2
        assert first.skipped == 0
        calls_after_first = embedder.calls

        second = store.ingest_document(conn, embedder, doc, tmp_path)
        assert second.embedded == 0
        assert second.skipped == 2
        assert embedder.calls == calls_after_first  # no new API calls at all

    def test_changed_content_retires_old_chunks_and_embeds_new(self, conn, tmp_path):
        doc_v1 = self._fake_doc(tmp_path, "MADDE 1 - Eski hüküm.")
        doc_v1.file_sha256 = "hash-v1"
        doc_v1.doc_id = "hash-v1-fake"
        embedder = self.FakeEmbedder()
        store.ingest_document(conn, embedder, doc_v1, tmp_path)

        doc_v2 = self._fake_doc(tmp_path, "MADDE 1 - Yeni hüküm.\nMADDE 2 - Ek hüküm.")
        doc_v2.file_sha256 = "hash-v2"
        doc_v2.doc_id = "hash-v2-fake"
        result = store.ingest_document(conn, embedder, doc_v2, tmp_path)
        assert result.embedded == 2
        assert result.skipped == 0

        matrix, ids = store.fetch_active_embeddings(conn)
        assert matrix.shape[0] == 2  # only v2's chunks are active
        assert store.has_active_chunks(conn, "hash-v1") is False


class TestCrossThreadConnection:
    """A connection from store.connect() must be usable from any thread.

    app.py holds one connection (via st.cache_resource) across Streamlit's
    per-rerun threads, serialized by its own lock rather than sqlite3's
    thread-affinity check -- see store.connect()'s check_same_thread=False.
    Without it, a read issued from a thread other than the one that opened
    the connection raises "SQLite objects created in a thread can only be
    used in that same thread."
    """

    def test_read_from_a_different_thread_than_the_connection_was_opened_on(self, conn):
        store.insert_chunks(
            conn, source_path="p", file_sha256="hash-cross-thread", document_title="X",
            chunks=[make_chunk()], embeddings=[np.zeros(config.EMBEDDING_DIM, dtype=np.float32)],
            quality_flag=None,
        )

        result: dict = {}

        def read_from_other_thread():
            try:
                matrix, ids = store.fetch_active_embeddings(conn)
                result["matrix"] = matrix
                result["ids"] = ids
            except Exception as exc:  # noqa: BLE001 - captured to fail the test with detail
                result["error"] = exc

        worker = threading.Thread(target=read_from_other_thread)
        worker.start()
        worker.join()

        assert "error" not in result, f"cross-thread read raised: {result.get('error')}"
        assert result["matrix"].shape == (1, config.EMBEDDING_DIM)

    def test_close_from_a_different_thread_than_the_connection_was_opened_on(self, tmp_path):
        # Regression for the exact crash found in manual QA: app.py's old
        # _rebind_connection() closed the previous connection from whatever
        # thread the current Streamlit rerun happened to be on, which is
        # never guaranteed to be the thread that opened it.
        c = store.connect(tmp_path / "close-cross-thread.db")
        errors: list[Exception] = []

        def close_from_other_thread():
            try:
                c.close()
            except Exception as exc:  # noqa: BLE001 - captured to fail the test with detail
                errors.append(exc)

        worker = threading.Thread(target=close_from_other_thread)
        worker.start()
        worker.join()

        assert not errors, f"cross-thread close raised: {errors}"
