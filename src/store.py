"""SQLite-backed chunk + embedding store, and the embed -> store CLI.

Document identity across re-ingests is `source_path` (the file's path relative
to the corpus root), which is stable even though EPDK filenames are opaque --
it is NOT `file_sha256`, which is a content hash and therefore changes whenever
EPDK republishes an updated version of the same document. Re-ingesting a
document whose content changed retires its old chunks (active=0) and inserts
the new version as active=1; nothing ever answers queries with a mix of both.

Resumability works at chunk granularity, not document granularity: each batch
of embedded chunks is inserted and committed immediately, and a rerun after a
crash queries which (file_sha256, chunk_index) pairs are already active before
calling the embedding API again, so only genuinely missing chunks are
re-embedded. Extraction itself (LibreOffice conversion, PDF parsing) is not
cached to disk and does re-run in full on a rerun -- it is comparatively fast
next to ~27k embedding calls, and the definition of "resumable" here is about
not repeating the expensive, rate-limited part.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config
from .chunk import Chunk, chunk_document
from .embed import EmbeddingClient
from .extract import ExtractedDoc, _force_utf8_stdout, extract_corpus
from .titles import extract_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path   TEXT NOT NULL,
    file_sha256   TEXT NOT NULL,
    document_title TEXT,
    chunk_index   INTEGER NOT NULL,
    text          TEXT NOT NULL,
    article_ref   TEXT,
    page_start    INTEGER,
    page_end      INTEGER,
    quality_flag  TEXT,
    embedding     BLOB NOT NULL,
    embedded_at   TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    UNIQUE (file_sha256, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash    ON chunks(file_sha256);
CREATE INDEX IF NOT EXISTS idx_chunks_active       ON chunks(active);
"""


class DimensionMismatch(Exception):
    """A vector stored on disk does not have config.EMBEDDING_DIM elements."""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path if db_path is not None else config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    # WAL: readers (retrieval) are never blocked by an in-progress ingest, and
    # committed batches survive a crash mid-run -- both matter for resumability.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _serialize(vector: list[float] | np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _deserialize(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# --------------------------------------------------------------------------
# Resumability / re-ingest identity
# --------------------------------------------------------------------------


def active_hash_for_path(conn: sqlite3.Connection, source_path: str) -> str | None:
    """The file_sha256 currently active for a document, or None if never ingested."""
    row = conn.execute(
        "SELECT DISTINCT file_sha256 FROM chunks WHERE source_path = ? AND active = 1 LIMIT 1",
        (source_path,),
    ).fetchone()
    return row[0] if row else None


def has_active_chunks(conn: sqlite3.Connection, file_sha256: str) -> bool:
    """Whether this exact file content already has at least one active chunk stored."""
    row = conn.execute(
        "SELECT 1 FROM chunks WHERE file_sha256 = ? AND active = 1 LIMIT 1", (file_sha256,)
    ).fetchone()
    return row is not None


def existing_chunk_indices(conn: sqlite3.Connection, file_sha256: str) -> set[int]:
    """Which chunk_index values are already active for this content -- the resume set."""
    rows = conn.execute(
        "SELECT chunk_index FROM chunks WHERE file_sha256 = ? AND active = 1", (file_sha256,)
    )
    return {r[0] for r in rows}


def mark_document_inactive(conn: sqlite3.Connection, source_path: str) -> int:
    """Retire every active chunk for a document (its content changed). Returns rows affected."""
    cur = conn.execute(
        "UPDATE chunks SET active = 0 WHERE source_path = ? AND active = 1", (source_path,)
    )
    conn.commit()
    return cur.rowcount


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def insert_chunks(
    conn: sqlite3.Connection,
    *,
    source_path: str,
    file_sha256: str,
    document_title: str | None,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    quality_flag: str | None,
) -> int:
    """Insert a batch of already-embedded chunks and commit. Returns rows inserted.

    UNIQUE(file_sha256, chunk_index) makes this safe to call again with
    overlapping chunks after a crash: INSERT OR IGNORE silently skips any
    (hash, index) pair already stored rather than erroring or duplicating.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            source_path,
            file_sha256,
            document_title,
            chunk.index,
            chunk.text,
            str(chunk.article) if chunk.article else None,
            chunk.page_start,
            chunk.page_end,
            quality_flag,
            _serialize(vector),
            now,
        )
        for chunk, vector in zip(chunks, embeddings)
    ]
    cur = conn.executemany(
        """
        INSERT OR IGNORE INTO chunks
            (source_path, file_sha256, document_title, chunk_index, text,
             article_ref, page_start, page_end, quality_flag, embedding, embedded_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        rows,
    )
    conn.commit()
    return cur.rowcount


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def fetch_active_embeddings(conn: sqlite3.Connection) -> tuple[np.ndarray, list[int]]:
    """All active embeddings as one (N, EMBEDDING_DIM) float32 matrix, plus parallel chunk ids.

    Raises DimensionMismatch rather than returning a matrix that would silently
    corrupt every downstream similarity computation.
    """
    rows = conn.execute("SELECT id, embedding FROM chunks WHERE active = 1 ORDER BY id").fetchall()
    if not rows:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32), []

    ids: list[int] = []
    vectors: list[np.ndarray] = []
    for row_id, blob in rows:
        vector = _deserialize(blob)
        if vector.shape[0] != config.EMBEDDING_DIM:
            raise DimensionMismatch(
                f"chunk id {row_id}: stored embedding has {vector.shape[0]} dims, "
                f"but config.EMBEDDING_DIM is {config.EMBEDDING_DIM}. The store was "
                "built with a different embedding model than is currently configured."
            )
        ids.append(row_id)
        vectors.append(vector)
    return np.vstack(vectors), ids


def fetch_chunk_metadata(conn: sqlite3.Connection, chunk_id: int) -> dict | None:
    """All columns for one chunk except the raw embedding BLOB, keyed by id."""
    row = conn.execute(
        """
        SELECT id, source_path, file_sha256, document_title, chunk_index, text,
               article_ref, page_start, page_end, quality_flag, embedded_at, active
        FROM chunks WHERE id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    columns = (
        "id", "source_path", "file_sha256", "document_title", "chunk_index", "text",
        "article_ref", "page_start", "page_end", "quality_flag", "embedded_at", "active",
    )
    return dict(zip(columns, row))


# --------------------------------------------------------------------------
# CLI: extract -> chunk -> embed -> store
# --------------------------------------------------------------------------


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _quality_flag_for(doc: ExtractedDoc, title_flags: list[str]) -> str | None:
    combined = [*doc.flags, *title_flags]
    return "; ".join(combined) if combined else None


def ingest_document(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    doc: ExtractedDoc,
    root: Path,
) -> tuple[int, int]:
    """Embed and store one document's chunks. Returns (embedded, skipped) chunk counts."""
    info = extract_title(doc)
    chunks = chunk_document(doc, info)
    if not chunks:
        return 0, 0

    source_path = _rel(doc.path, root)
    quality_flag = _quality_flag_for(doc, info.flags)

    prior_hash = active_hash_for_path(conn, source_path)
    if prior_hash is not None and prior_hash != doc.file_sha256:
        # Content changed since the last ingest: retire the old version outright
        # so a query never mixes chunks from two versions of the same law.
        mark_document_inactive(conn, source_path)
        done = set()
    else:
        done = existing_chunk_indices(conn, doc.file_sha256)

    todo = [c for c in chunks if c.index not in done]
    if not todo:
        return 0, len(chunks)

    embedded = 0
    size = config.EMBED_BATCH_SIZE
    for start in range(0, len(todo), size):
        batch = todo[start : start + size]
        vectors = embedder.embed_batch([c.text for c in batch])
        insert_chunks(
            conn,
            source_path=source_path,
            file_sha256=doc.file_sha256,
            document_title=info.title,
            chunks=batch,
            embeddings=vectors,
            quality_flag=quality_flag,
        )
        embedded += len(batch)
    return embedded, len(chunks) - len(todo)


def run_ingest(root: Path, db_path: str | Path | None = None, verbose: bool = True) -> dict:
    """Full extract -> chunk -> embed -> store pass. Returns summary counters."""
    started = time.monotonic()
    conn = connect(db_path)
    embedder = EmbeddingClient.connect()

    if verbose:
        print(f"Embedding model : {embedder.model_id}")
        print(f"Embedding dim   : {embedder.dimension}")
        print(f"Database        : {config.DB_PATH if db_path is None else db_path}")
        print(f"\nScanning {root.resolve()} ...")

    run = extract_corpus(root, verbose=verbose)

    docs_processed = 0
    chunks_embedded = 0
    chunks_skipped = 0
    for i, doc in enumerate(run.docs, start=1):
        embedded, skipped = ingest_document(conn, embedder, doc, root)
        chunks_embedded += embedded
        chunks_skipped += skipped
        docs_processed += 1
        if verbose and (embedded or i % 25 == 0 or i == len(run.docs)):
            print(
                f"  [{i}/{len(run.docs)}] {doc.doc_id}  "
                f"embedded={embedded} skipped={skipped} "
                f"(total embedded={chunks_embedded}, skipped={chunks_skipped})",
                flush=True,
            )

    conn.close()
    elapsed = time.monotonic() - started
    return {
        "documents_processed": docs_processed,
        "chunks_embedded": chunks_embedded,
        "chunks_skipped": chunks_skipped,
        "embedding_dim": embedder.dimension,
        "elapsed_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .embed import FoundryUnavailable

    _force_utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m src.store")
    parser.add_argument("--root", default="data", help="corpus root (default: data)")
    parser.add_argument("--db", default=None, help=f"SQLite path (default: {config.DB_PATH})")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"error: corpus root {root} does not exist")
        return 2

    try:
        summary = run_ingest(root, db_path=args.db)
    except FoundryUnavailable as exc:
        print(f"\nFATAL: {exc}")
        return 3

    print("\n" + "=" * 72)
    print("EMBED + STORE REPORT")
    print("=" * 72)
    print(f"Documents processed : {summary['documents_processed']}")
    print(f"Chunks embedded     : {summary['chunks_embedded']}")
    print(f"Chunks skipped      : {summary['chunks_skipped']} (already active + unchanged)")
    print(f"Embedding dimension : {summary['embedding_dim']}")
    print(f"Elapsed             : {summary['elapsed_seconds']:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
