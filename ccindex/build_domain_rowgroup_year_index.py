#!/usr/bin/env python3
"""Build a per-year rowgroup slice index DB from per-collection DBs.

This avoids opening many per-collection DBs at query time by aggregating
cc_domain_rowgroups into one DB per year.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import duckdb


def _iter_collection_dbs(collection_dir: Path, year: str) -> List[Path]:
    out: List[Path] = []
    if not collection_dir.exists():
        return out
    for p in sorted(collection_dir.glob(f"CC-MAIN-{year}-*.duckdb")):
        if p.is_file():
            out.append(p)
    return out


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cc_domain_rowgroups (
            collection TEXT,
            source_path TEXT,
            parquet_relpath TEXT,
            row_group INTEGER,
            dom_rg_row_start BIGINT,
            dom_rg_row_end BIGINT,
            host_rev TEXT
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_ccdr_host_rev ON cc_domain_rowgroups(host_rev)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ccdr_collection ON cc_domain_rowgroups(collection)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ccdr_coll_host ON cc_domain_rowgroups(collection, host_rev)")


def _copy_collection(con: duckdb.DuckDBPyConnection, db_path: Path, collection: str) -> int:
    con.execute(f"ATTACH '{db_path}' AS src")
    try:
        rows = con.execute("SELECT COUNT(*) FROM src.cc_domain_rowgroups").fetchone()[0]
        con.execute(
            """
            INSERT INTO cc_domain_rowgroups
            SELECT ?, source_path, parquet_relpath, row_group, dom_rg_row_start, dom_rg_row_end, host_rev
            FROM src.cc_domain_rowgroups
            """,
            [collection],
        )
    except Exception:
        rows = con.execute("SELECT COUNT(*) FROM src.cc_domain_rowgroups").fetchone()[0]
        con.execute(
            """
            INSERT INTO cc_domain_rowgroups
            SELECT ?, source_path, parquet_relpath, row_group, dom_rg_row_start, dom_rg_row_end, host_rev
            FROM src.cc_domain_rowgroups
            """,
            [collection],
        )
    finally:
        con.execute("DETACH src")
    return int(rows or 0)


def _existing_collections(con: duckdb.DuckDBPyConnection) -> set[str]:
    try:
        con.execute("SELECT 1 FROM cc_domain_rowgroups LIMIT 1").fetchone()
    except Exception:
        return set()
    try:
        rows = con.execute("SELECT DISTINCT collection FROM cc_domain_rowgroups").fetchall()
        return {str(r[0]) for r in rows if r and r[0] is not None}
    except Exception:
        return set()


def build_year_index(
    collection_dir: Path,
    year: str,
    output_db: Path,
    *,
    overwrite: bool,
    resume: bool,
    memory_limit: str | None,
) -> None:
    if output_db.exists() and overwrite and not resume:
        output_db.unlink()
    output_db.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(output_db))
    try:
        con.execute("PRAGMA threads=4")
        if memory_limit:
            try:
                con.execute(f"PRAGMA memory_limit='{memory_limit}'")
            except Exception:
                pass
        _ensure_schema(con)

        total = 0
        files = _iter_collection_dbs(collection_dir, year)
        if not files:
            raise SystemExit(f"No collection DBs found for year {year} in {collection_dir}")

        existing = _existing_collections(con) if resume else set()
        if existing:
            print(f"resume enabled: {len(existing)} collections already in output", flush=True)

        start_time = time.time()
        last_log = start_time
        print(
            f"year_start {year} collections={len(files)} output_db={output_db}",
            flush=True,
        )

        for idx, db_path in enumerate(files, 1):
            collection = db_path.stem.replace(".domain_rowgroups", "")
            if resume and collection in existing:
                print(f"collection_skip {collection} (already present)", flush=True)
                continue
            print(f"collection_start {collection}", flush=True)
            t0 = time.time()
            rows = _copy_collection(con, db_path, collection)
            total += rows
            dt = time.time() - t0
            print(f"[{idx}/{len(files)}] {collection}: {rows} rows in {dt:.2f}s", flush=True)

            now = time.time()
            if (now - last_log) >= 60:
                rate = total / max(1.0, (now - start_time))
                print(
                    f"year_progress {year} copied={idx}/{len(files)} rows={total} "
                    f"elapsed_s={int(now - start_time)} rate={rate:.1f}/s",
                    flush=True,
                )
                last_log = now

        con.execute("ANALYZE")
        elapsed = time.time() - start_time
        print(
            f"year_done {year} rows={total} elapsed_s={int(elapsed)} output_db={output_db}",
            flush=True,
        )
    finally:
        con.close()


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a per-year rowgroup index DB")
    ap.add_argument("--year", required=True, help="Year, e.g. 2025")
    ap.add_argument(
        "--collection-dir",
        default="/storage/ccindex_duckdb/cc_domain_rowgroups_by_collection",
        help="Directory with per-collection rowgroup DBs",
    )
    ap.add_argument(
        "--output-db",
        default=None,
        help="Output DB path (default: /storage/ccindex_duckdb/cc_domain_rowgroups_by_year/cc_domain_rowgroups_<year>.duckdb)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output DB if it exists")
    ap.add_argument("--resume", action="store_true", help="Resume if output DB already has data")
    ap.add_argument(
        "--memory-limit",
        default=(os.environ.get("CC_ROWGROUP_YEAR_BUILD_MEMORY_LIMIT") or ""),
        help="DuckDB memory limit (e.g., 8GB).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    year = str(args.year).strip()
    if not year.isdigit():
        raise SystemExit("Year must be numeric")

    collection_dir = Path(args.collection_dir).expanduser().resolve()
    if args.output_db:
        output_db = Path(args.output_db).expanduser().resolve()
    else:
        output_db = Path(
            f"/storage/ccindex_duckdb/cc_domain_rowgroups_by_year/cc_domain_rowgroups_{year}.duckdb"
        ).expanduser().resolve()

    mem_limit = str(args.memory_limit).strip() or None
    build_year_index(
        collection_dir,
        year,
        output_db,
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        memory_limit=mem_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
