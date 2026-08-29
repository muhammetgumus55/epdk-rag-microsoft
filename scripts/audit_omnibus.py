"""Scope audit: how much of the indexed corpus is not actually about electricity.

Reads the live chunk store and reports, per omnibus document, how many of its
stored chunks src.scope classifies as electricity-relevant, off-domain, or
undecidable. Read-only -- it changes nothing, and is the evidence behind
docs/decisions/2026-08-29-omnibus-scope-filter.md.

    python -m scripts.audit_omnibus                 # summary
    python -m scripts.audit_omnibus --samples 3     # + sample excluded text per doc
    python -m scripts.audit_omnibus --ambiguous     # + every AMBIGUOUS chunk, for review
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict

from src import config, scope
from src.extract import _force_utf8_stdout


def document_rows(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    """Every active document as (source_path, document_title)."""
    return conn.execute(
        "SELECT source_path, MAX(document_title) FROM chunks WHERE active = 1 "
        "GROUP BY source_path ORDER BY source_path"
    ).fetchall()


def opening_text(conn: sqlite3.Connection, source_path: str) -> str:
    """The document's first two chunks -- where an omnibus act names itself."""
    rows = conn.execute(
        "SELECT text FROM chunks WHERE active = 1 AND source_path = ? "
        "ORDER BY chunk_index LIMIT 2",
        (source_path,),
    ).fetchall()
    return "\n".join(r[0] for r in rows)


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(prog="python -m scripts.audit_omnibus")
    parser.add_argument("--db", default=config.DB_PATH)
    parser.add_argument("--samples", type=int, default=0,
                        help="print N sample OFF_DOMAIN chunk excerpts per document")
    parser.add_argument("--ambiguous", action="store_true",
                        help="print every AMBIGUOUS chunk in full, for human review")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    docs = document_rows(conn)
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks WHERE active = 1").fetchone()[0]

    omnibus: list[tuple[str, str | None]] = []
    excluded_docs: list[tuple[str, str | None, str]] = []
    for source_path, title in docs:
        decision = scope.document_scope(title, opening_text(conn, source_path))
        if decision.disposition == "OMNIBUS":
            omnibus.append((source_path, title))
        elif decision.disposition == "EXCLUDED":
            excluded_docs.append((source_path, title, decision.reason))

    print("=" * 100)
    print("CORPUS SCOPE AUDIT")
    print("=" * 100)
    print(f"Active documents : {len(docs)}")
    print(f"Active chunks    : {total_chunks:,}")
    print(f"Omnibus documents: {len(omnibus)}")
    print(f"Manually excluded documents: {len(excluded_docs)}")

    excluded_whole = 0
    for source_path, title, reason in excluded_docs:
        n = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE active = 1 AND source_path = ?", (source_path,)
        ).fetchone()[0]
        excluded_whole += n
        print(f"\n  [{n} chunks] {source_path}")
        print(f"      title : {title}")
        print(f"      reason: {reason}")

    per_doc: dict[str, Counter] = {}
    ambiguous_rows: list[tuple[str, int, str | None, str]] = []
    samples: dict[str, list[tuple[int, str | None, str, str]]] = defaultdict(list)
    grand = Counter()
    omnibus_chunks = 0

    for source_path, _title in omnibus:
        rows = conn.execute(
            "SELECT id, chunk_index, article_ref, text FROM chunks "
            "WHERE active = 1 AND source_path = ? ORDER BY chunk_index",
            (source_path,),
        ).fetchall()
        verdicts = scope.classify_chunks([(r[2], r[3]) for r in rows])
        counts: Counter = Counter()
        for (chunk_id, _idx, article_ref, text), verdict in zip(rows, verdicts):
            counts[verdict.label] += 1
            grand[verdict.label] += 1
            omnibus_chunks += 1
            if verdict.label == "AMBIGUOUS":
                ambiguous_rows.append((source_path, chunk_id, article_ref, text))
            elif verdict.label == "OFF_DOMAIN" and len(samples[source_path]) < args.samples:
                samples[source_path].append(
                    (chunk_id, article_ref, verdict.reason, " ".join(text.split())[:300])
                )
        per_doc[source_path] = counts

    print(f"\nChunks in omnibus documents: {omnibus_chunks:,} "
          f"({omnibus_chunks / total_chunks:.1%} of the corpus)")
    print()
    print(f"  ELECTRICITY (keep)   : {grand['ELECTRICITY']:,}")
    print(f"  OFF_DOMAIN  (exclude): {grand['OFF_DOMAIN']:,}")
    print(f"  AMBIGUOUS   (review) : {grand['AMBIGUOUS']:,}")
    if omnibus_chunks:
        print(f"\n  Off-domain share of omnibus chunks : {grand['OFF_DOMAIN'] / omnibus_chunks:.1%}")

    total_excluded = grand["OFF_DOMAIN"] + excluded_whole
    print("\n" + "-" * 100)
    print(f"TOTAL NON-INDEXABLE: {total_excluded:,} chunks "
          f"({total_excluded / total_chunks:.2%} of the corpus)")
    print(f"  from omnibus article filtering : {grand['OFF_DOMAIN']:,}")
    print(f"  from manual document exclusions: {excluded_whole:,}")

    print("\n" + "=" * 100)
    print("PER-DOCUMENT BREAKDOWN  (sorted by off-domain chunk count)")
    print("=" * 100)
    print(f"{'ELEC':>5} {'OFF':>5} {'AMB':>5}  document")
    order = sorted(omnibus, key=lambda d: -per_doc[d[0]]["OFF_DOMAIN"])
    for source_path, title in order:
        c = per_doc[source_path]
        print(f"{c['ELECTRICITY']:5d} {c['OFF_DOMAIN']:5d} {c['AMBIGUOUS']:5d}  {source_path}")
        print(f"{'':18}title: {title}")
        for chunk_id, article_ref, reason, excerpt in samples.get(source_path, []):
            print(f"{'':18}  - id={chunk_id} {article_ref or '(no article)'} :: {reason}")
            print(f"{'':18}    {excerpt}")

    if args.ambiguous:
        print("\n" + "=" * 100)
        print(f"AMBIGUOUS CHUNKS ({len(ambiguous_rows)}) -- need a human decision")
        print("=" * 100)
        for source_path, chunk_id, article_ref, text in ambiguous_rows:
            print(f"\n--- id={chunk_id} {article_ref or '(no article)'}  {source_path}")
            print(" ".join(text.split())[:700])

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
