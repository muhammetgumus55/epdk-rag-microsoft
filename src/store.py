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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config, scope
from .chunk import Chunk, chunk_document
from .embed import EmbeddingClient
from .extract import ExtractedDoc, _force_utf8_stdout, extract_corpus
from .titles import extract_title

# `embedding` and `embedded_at` are NULLable, and that is the point of the
# out-of-scope design rather than an oversight: a chunk excluded by src.scope is
# still stored in full, so the corpus keeps complete provenance of what the
# source documents contain, but it is never sent to the embedding model and has
# no vector. `indexable = 0` is therefore a stronger guarantee than a query-time
# filter -- there is nothing to retrieve, not merely something that scores badly.
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
    embedding     BLOB,
    embedded_at   TEXT,
    active        INTEGER NOT NULL DEFAULT 1,
    indexable     INTEGER NOT NULL DEFAULT 1,
    scope_label   TEXT,
    UNIQUE (file_sha256, chunk_index)
);
"""

# Created after _migrate(), never inside SCHEMA: on a store built before the
# scope work the `indexable` column does not exist yet, and SCHEMA runs first,
# so indexing it there fails the whole script before the migration can add it.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_chunks_source_path ON chunks(source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_file_hash    ON chunks(file_sha256);
CREATE INDEX IF NOT EXISTS idx_chunks_active       ON chunks(active);
CREATE INDEX IF NOT EXISTS idx_chunks_indexable    ON chunks(indexable);
"""

# Columns added after the first corpus was built. ALTER TABLE ADD COLUMN is
# enough for these two -- both have defaults and neither changes an existing
# constraint.
_ADDED_COLUMNS = (
    ("indexable", "INTEGER NOT NULL DEFAULT 1"),
    ("scope_label", "TEXT"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an existing store up to the current schema, in place.

    Two changes, and they need different treatment. The new columns are a plain
    ALTER TABLE. Relaxing `embedding`/`embedded_at` from NOT NULL to NULLable is
    not expressible as an ALTER in SQLite at all, so it needs the standard
    table-rebuild dance -- done inside one transaction so a crash mid-migration
    leaves the old table intact rather than a half-copied one.
    """
    columns = {row[1]: row for row in conn.execute("PRAGMA table_info(chunks)")}
    if not columns:  # fresh database; SCHEMA already created it correctly
        return

    for name, definition in _ADDED_COLUMNS:
        if name not in columns:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {definition}")
            conn.commit()

    # notnull is column 3 of PRAGMA table_info. Only rebuild if it still applies.
    if not columns["embedding"][3]:
        return

    conn.executescript(
        """
        BEGIN;
        CREATE TABLE chunks_migrated (
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
            embedding     BLOB,
            embedded_at   TEXT,
            active        INTEGER NOT NULL DEFAULT 1,
            indexable     INTEGER NOT NULL DEFAULT 1,
            scope_label   TEXT,
            UNIQUE (file_sha256, chunk_index)
        );
        INSERT INTO chunks_migrated
            SELECT id, source_path, file_sha256, document_title, chunk_index, text,
                   article_ref, page_start, page_end, quality_flag, embedding,
                   embedded_at, active, indexable, scope_label
            FROM chunks;
        DROP TABLE chunks;
        ALTER TABLE chunks_migrated RENAME TO chunks;
        COMMIT;
        """
    )
    conn.commit()


class DimensionMismatch(Exception):
    """A vector stored on disk does not have config.EMBEDDING_DIM elements."""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the chunk store and ensure its schema exists."""
    path = Path(db_path if db_path is not None else config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: callers that share one long-lived connection
    # across threads (e.g. app.py's Streamlit UI, where every rerun executes
    # on its own OS thread) are responsible for their own serialization --
    # app.py does this via GENERATION_LOCK. sqlite3's own thread-affinity
    # check has no knowledge of that lock and would reject the same safe
    # access pattern outright.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    # WAL: readers (retrieval) are never blocked by an in-progress ingest, and
    # committed batches survive a crash mid-run -- both matter for resumability.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)
    conn.executescript(INDEXES)
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
    """Which chunk_index values are already active for this content -- the resume set.

    Deliberately counts non-indexable chunks as done. They are stored and final;
    a resumed run must not treat "has no embedding" as "still needs embedding"
    and send excluded text to the model on every rerun.
    """
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


def delete_document_chunks(conn: sqlite3.Connection, source_path: str) -> int:
    """Remove a document's chunks outright. Returns rows deleted.

    Used by the reprocess path, where marking inactive is not enough and would
    silently do the wrong thing. UNIQUE(file_sha256, chunk_index) does not
    consider `active`, so when a document is rebuilt from UNCHANGED source
    bytes -- exactly the reprocess case, since only our scope rules moved --
    every re-inserted row collides with the retired one and INSERT OR IGNORE
    drops it, leaving the document with no active chunks at all.

    Retiring is right when content changed: the old rows keep a distinct
    file_sha256 and stay as history. Here the old rows are the same text under a
    stale policy, so they are litter, not provenance.
    """
    cur = conn.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
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
    embeddings: list[list[float] | None],
    quality_flag: str | None,
    scope_labels: list[str | None] | None = None,
) -> int:
    """Insert a batch of chunks and commit. Returns rows inserted.

    An embedding of None marks a chunk excluded by src.scope: it is stored in
    full for provenance, with a NULL vector and indexable = 0, and can never be
    retrieved. `scope_labels` carries why, or None for chunks in single-subject
    documents that were never classified at all.

    UNIQUE(file_sha256, chunk_index) makes this safe to call again with
    overlapping chunks after a crash: INSERT OR IGNORE silently skips any
    (hash, index) pair already stored rather than erroring or duplicating.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    labels = scope_labels if scope_labels is not None else [None] * len(chunks)
    if len(labels) != len(chunks):
        raise ValueError(f"{len(chunks)} chunks but {len(labels)} scope labels")
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
            None if vector is None else _serialize(vector),
            None if vector is None else now,
            0 if vector is None else 1,
            label,
        )
        for chunk, vector, label in zip(chunks, embeddings, labels)
    ]
    cur = conn.executemany(
        """
        INSERT OR IGNORE INTO chunks
            (source_path, file_sha256, document_title, chunk_index, text,
             article_ref, page_start, page_end, quality_flag, embedding, embedded_at,
             indexable, scope_label, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        rows,
    )
    conn.commit()
    return cur.rowcount


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def fetch_active_embeddings(conn: sqlite3.Connection) -> tuple[np.ndarray, list[int]]:
    """All active, in-scope embeddings as one (N, EMBEDDING_DIM) float32 matrix, plus ids.

    `indexable = 1` is what keeps out-of-scope text out of retrieval. Those rows
    have no embedding at all, so this is not a filter that could be forgotten
    somewhere else in the stack -- there is no vector for them to match against.

    Raises DimensionMismatch rather than returning a matrix that would silently
    corrupt every downstream similarity computation.
    """
    rows = conn.execute(
        "SELECT id, embedding FROM chunks WHERE active = 1 AND indexable = 1 ORDER BY id"
    ).fetchall()
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


@dataclass
class IngestResult:
    """What one document's ingest did, split by scope disposition."""

    embedded: int = 0
    skipped: int = 0        # already active from a previous run
    excluded: int = 0       # stored non-indexable: out of scope
    disposition: str = "IN_SCOPE"


def ingest_document(
    conn: sqlite3.Connection,
    embedder: EmbeddingClient,
    doc: ExtractedDoc,
    root: Path,
) -> IngestResult:
    """Scope, embed and store one document's chunks.

    Out-of-scope chunks (src.scope) are stored with a NULL embedding and
    indexable = 0 -- kept for provenance, never sent to the embedding model,
    never retrievable.
    """
    info = extract_title(doc)
    chunks = chunk_document(doc, info)
    if not chunks:
        return IngestResult()

    source_path = _rel(doc.path, root)
    quality_flag = _quality_flag_for(doc, info.flags)

    decision, verdicts = scope.scope_chunks(
        info.title, [(str(c.article) if c.article else None, c.text) for c in chunks]
    )

    prior_hash = active_hash_for_path(conn, source_path)
    if prior_hash is not None and prior_hash != doc.file_sha256:
        # Content changed since the last ingest: retire the old version outright
        # so a query never mixes chunks from two versions of the same law.
        mark_document_inactive(conn, source_path)
        done = set()
    else:
        done = existing_chunk_indices(conn, doc.file_sha256)

    pending = [(c, v) for c, v in zip(chunks, verdicts) if c.index not in done]
    result = IngestResult(skipped=len(chunks) - len(pending), disposition=decision.disposition)
    if not pending:
        return result

    size = config.EMBED_BATCH_SIZE
    for start in range(0, len(pending), size):
        batch = pending[start : start + size]
        # Only in-scope chunks are embedded. The excluded ones are interleaved
        # with them in document order, so their vectors are filled back in as
        # None rather than the batch being split and reassembled.
        wanted = [c for c, v in batch if v is None or v.indexable]
        vectors = embedder.embed_batch([c.text for c in wanted]) if wanted else []
        by_index = {c.index: vec for c, vec in zip(wanted, vectors)}
        insert_chunks(
            conn,
            source_path=source_path,
            file_sha256=doc.file_sha256,
            document_title=info.title,
            chunks=[c for c, _ in batch],
            embeddings=[by_index.get(c.index) for c, _ in batch],
            quality_flag=quality_flag,
            scope_labels=[None if v is None else v.label for _, v in batch],
        )
        result.embedded += len(wanted)
        result.excluded += len(batch) - len(wanted)
    return result


def run_ingest(
    root: Path,
    db_path: str | Path | None = None,
    verbose: bool = True,
    only: list[Path] | None = None,
) -> dict:
    """Full extract -> scope -> chunk -> embed -> store pass. Returns summary counters.

    `only` restricts the run to an explicit list of files and RETIRES each of
    them first, which is what makes a partial reprocess possible at all. The
    normal resume path deliberately skips any document whose file_sha256 is
    unchanged, so a policy change -- new scope rules, an added manual exclusion --
    is invisible to it: the bytes on disk are identical, only our judgement about
    them moved. Retiring first forces those documents to be rebuilt from source
    while every other document in the store is left completely alone.
    """
    started = time.monotonic()
    conn = connect(db_path)
    embedder = EmbeddingClient.connect()

    if verbose:
        print(f"Embedding model : {embedder.model_id}")
        print(f"Embedding dim   : {embedder.dimension}")
        print(f"Database        : {config.DB_PATH if db_path is None else db_path}")

    if only is not None:
        if verbose:
            print(f"\nReprocessing {len(only)} document(s); the rest of the store is untouched.")
        removed = 0
        for path in only:
            removed += delete_document_chunks(conn, _rel(path, root))
        if verbose:
            print(f"Removed {removed} existing chunk(s) from those documents.")
        run = extract_corpus(root, verbose=verbose, paths=only)
    else:
        if verbose:
            print(f"\nScanning {root.resolve()} ...")
        run = extract_corpus(root, verbose=verbose)

    docs_processed = 0
    totals = IngestResult()
    dispositions: dict[str, int] = {}
    for i, doc in enumerate(run.docs, start=1):
        result = ingest_document(conn, embedder, doc, root)
        totals.embedded += result.embedded
        totals.skipped += result.skipped
        totals.excluded += result.excluded
        dispositions[result.disposition] = dispositions.get(result.disposition, 0) + 1
        docs_processed += 1
        if verbose and (result.embedded or result.excluded or i % 25 == 0 or i == len(run.docs)):
            print(
                f"  [{i}/{len(run.docs)}] {doc.doc_id}  [{result.disposition}]  "
                f"embedded={result.embedded} excluded={result.excluded} "
                f"skipped={result.skipped}",
                flush=True,
            )

    conn.close()
    elapsed = time.monotonic() - started
    return {
        "documents_processed": docs_processed,
        "chunks_embedded": totals.embedded,
        "chunks_skipped": totals.skipped,
        "chunks_excluded": totals.excluded,
        "dispositions": dispositions,
        "embedding_dim": embedder.dimension,
        "elapsed_seconds": elapsed,
    }


def scoped_out_documents(root: Path, verbose: bool = True) -> list[Path]:
    """Every file under `root` that src.scope would filter or exclude outright.

    Extraction-only: it reads and classifies but writes nothing, so it is safe to
    run before deciding to reprocess. This is what --reprocess-scope uses to work
    out which documents need rebuilding, so the set comes from the same
    classifier that will run during ingest rather than a hand-maintained list
    that could drift from it.
    """
    run = extract_corpus(root, verbose=verbose)
    selected: list[Path] = []
    for doc in run.docs:
        info = extract_title(doc)
        chunks = chunk_document(doc, info)
        if not chunks:
            continue
        opening = "\n".join(c.text for c in chunks[:2])
        if scope.document_scope(info.title, opening).disposition != "IN_SCOPE":
            selected.append(doc.path)
    return selected


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the full extract -> chunk -> embed -> store pipeline."""
    import argparse

    from .embed import FoundryUnavailable

    _force_utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m src.store")
    parser.add_argument("--root", default="data", help="corpus root (default: data)")
    parser.add_argument("--db", default=None, help=f"SQLite path (default: {config.DB_PATH})")
    parser.add_argument(
        "--only", nargs="+", metavar="PATH", default=None,
        help="reprocess just these files: retire their stored chunks and rebuild "
             "them from source. Every other document is left untouched.",
    )
    parser.add_argument(
        "--reprocess-scope", action="store_true",
        help="reprocess exactly the documents src.scope filters or excludes "
             "(omnibus acts + manual exclusions). Use after changing scope rules.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="with --reprocess-scope, list the documents that would be rebuilt and stop.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"error: corpus root {root} does not exist")
        return 2
    if args.only and args.reprocess_scope:
        print("error: --only and --reprocess-scope are mutually exclusive")
        return 2

    only: list[Path] | None = None
    if args.reprocess_scope:
        only = scoped_out_documents(root)
        print(f"\n{len(only)} document(s) are out of scope or need article filtering:")
        for path in only:
            print(f"  {_rel(path, root)}")
        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return 0
    elif args.only:
        only = [Path(p) for p in args.only]
        missing = [p for p in only if not p.exists()]
        if missing:
            print(f"error: not found: {', '.join(str(p) for p in missing)}")
            return 2

    try:
        summary = run_ingest(root, db_path=args.db, only=only)
    except FoundryUnavailable as exc:
        print(f"\nFATAL: {exc}")
        return 3

    print("\n" + "=" * 72)
    print("EMBED + STORE REPORT")
    print("=" * 72)
    print(f"Documents processed : {summary['documents_processed']}")
    print(f"  by disposition    : {summary['dispositions']}")
    print(f"Chunks embedded     : {summary['chunks_embedded']}")
    print(f"Chunks excluded     : {summary['chunks_excluded']} (stored, not indexable)")
    print(f"Chunks skipped      : {summary['chunks_skipped']} (already active + unchanged)")
    print(f"Embedding dimension : {summary['embedding_dim']}")
    print(f"Elapsed             : {summary['elapsed_seconds']:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
