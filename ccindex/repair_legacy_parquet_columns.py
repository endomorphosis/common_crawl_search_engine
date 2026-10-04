#!/usr/bin/env python3
"""Repair legacy parquet shards missing collection/shard_file columns."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pyarrow.parquet as pq

import duckdb


def collection_year(collection: str) -> Optional[str]:
    """Extract the year from a CC collection name like CC-MAIN-2024-18."""
    try:
        parts = str(collection).split("-")
        if len(parts) >= 3 and parts[0] == "CC" and parts[1] == "MAIN":
            y = parts[2]
            return y if y.isdigit() and len(y) == 4 else None
        # Also accept the canonical form without splitting weirdness.
        if str(collection).startswith("CC-MAIN-"):
            y = str(collection)[len("CC-MAIN-") : len("CC-MAIN-") + 4]
            return y if y.isdigit() else None
        return None
    except Exception:
        return None

REQUIRED_COLS = ("collection", "shard_file")
DEFAULT_COMPRESSION = (os.environ.get("CC_PARQUET_COMPRESSION") or "zstd").strip().lower()
DEFAULT_ROW_GROUP_SIZE = int(os.environ.get("CC_SORT_ROW_GROUP_SIZE") or "71680")


def _collection_dirs(parquet_root: Path) -> List[Path]:
    out: List[Path] = []
    base = parquet_root / "cc_pointers_by_collection"
    if not base.exists():
        return out
    for year_dir in sorted(base.glob("[0-9][0-9][0-9][0-9]")):
        if not year_dir.is_dir():
            continue
        for coll_dir in sorted(year_dir.iterdir()):
            if coll_dir.is_dir():
                out.append(coll_dir)
    return out


def _parquet_missing_cols(pq_path: Path) -> bool:
    try:
        pf = pq.ParquetFile(str(pq_path))
        names = set(pf.schema.names)
        return not set(REQUIRED_COLS).issubset(names)
    except Exception:
        return False


def _quote_ident(name: str) -> str:
    # DuckDB uses standard SQL identifier quoting.
    return '"' + str(name).replace('"', '""') + '"'


def _repair_file(
    pq_path: Path,
    collection: str,
    *,
    overwrite: bool,
    compression: str,
    row_group_size: int,
    temp_dir: Optional[Path],
) -> bool:
    """Rewrite pq_path to ensure REQUIRED_COLS exist, preserving row order.

    Uses DuckDB COPY for streaming rewrite (safer on memory than loading Arrow tables).
    """

    try:
        pf = pq.ParquetFile(str(pq_path))
        existing_cols = list(pf.schema_arrow.names)
        missing = [c for c in REQUIRED_COLS if c not in set(existing_cols)]
        if not missing:
            return True
    except Exception:
        return False

    select_cols = ", ".join(_quote_ident(c) for c in existing_cols)
    extras: List[str] = []
    if "collection" in missing:
        extras.append(f"'{collection}' AS collection")
    if "shard_file" in missing:
        extras.append(f"'{pq_path.name}' AS shard_file")

    select_sql = f"SELECT {select_cols}" + (", " + ", ".join(extras) if extras else "") + " FROM read_parquet(?)"

    tmp = pq_path.with_suffix(pq_path.suffix + ".repair")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    try:
        con = duckdb.connect(database=":memory:")
        con.execute("SET preserve_insertion_order=true")
        con.execute("SET threads=1")
        if temp_dir is not None:
            con.execute(f"SET temp_directory='{str(temp_dir)}'")

        copy_opts = [
            "FORMAT PARQUET",
            f"COMPRESSION {str(compression).upper()}",
        ]
        try:
            rgs = int(row_group_size)
            if rgs > 0:
                copy_opts.append(f"ROW_GROUP_SIZE {rgs}")
        except Exception:
            pass

        copy_sql = f"COPY ({select_sql}) TO '{str(tmp).replace("'", "''")}' ({', '.join(copy_opts)})"
        con.execute(copy_sql, [str(pq_path)])
        con.close()
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False

    if overwrite:
        try:
            tmp.replace(pq_path)
        except Exception:
            return False

    return True


def _repair_job(job: Tuple[str, str, bool, str, int, Optional[str]]) -> bool:
    pq_path_s, collection_s, overwrite, compression, row_group_size, temp_dir_s = job
    temp_dir = Path(temp_dir_s).expanduser().resolve() if temp_dir_s else None
    return _repair_file(
        Path(pq_path_s),
        collection_s,
        overwrite=overwrite,
        compression=compression,
        row_group_size=row_group_size,
        temp_dir=temp_dir,
    )


def _iter_targets(parquet_root: Path, collections: Optional[List[str]]) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if collections:
        for c in collections:
            y = collection_year(c)
            if not y:
                raise SystemExit(f"Invalid collection name: {c}")
            out.append((c, parquet_root / "cc_pointers_by_collection" / y / c))
        return out

    for coll_dir in _collection_dirs(parquet_root):
        out.append((coll_dir.name, coll_dir))
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repair parquet shards missing collection/shard_file")
    ap.add_argument(
        "--parquet-root",
        default="/storage/ccindex_parquet",
        help="Root containing cc_pointers_by_collection",
    )
    ap.add_argument("--collections", default="", help="Comma-separated list of collections")
    ap.add_argument("--dry-run", action="store_true", help="Report files needing repair")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite parquet files in place")
    ap.add_argument(
        "--compression",
        default=DEFAULT_COMPRESSION,
        help="Parquet compression to use when rewriting (default: zstd)",
    )
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=DEFAULT_ROW_GROUP_SIZE,
        help="Row group size in rows for rewrites (default: env CC_SORT_ROW_GROUP_SIZE else 71680; use 0 for DuckDB default)",
    )
    ap.add_argument(
        "--temp-dir",
        default=None,
        help="Optional DuckDB temp spill directory (default: none)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for per-file rewrites (default: 1)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    parquet_root = Path(args.parquet_root).expanduser().resolve()
    collections = [c.strip() for c in str(args.collections).split(",") if c.strip()]
    temp_dir = Path(args.temp_dir).expanduser().resolve() if args.temp_dir else None
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)

    targets = _iter_targets(parquet_root, collections or None)
    if not targets:
        print("No parquet collections found.", flush=True)
        return 0

    repaired = 0
    scanned = 0
    to_repair: List[Tuple[Path, str]] = []
    t0 = time.time()
    last_log = t0
    print(
        f"starting repair parquet_root={parquet_root} collections={len(targets)} dry_run={args.dry_run}",
        flush=True,
    )
    for collection, coll_dir in targets:
        if not coll_dir.exists():
            continue
        print(f"collection_start {collection}", flush=True)
        for pq_path in sorted(coll_dir.glob("cdx-*.parquet")):
            scanned += 1
            if _parquet_missing_cols(pq_path):
                if args.dry_run:
                    print(f"missing_cols: {pq_path}", flush=True)
                else:
                    to_repair.append((pq_path, collection))

            now = time.time()
            if scanned % 200 == 0 or (now - last_log) >= 30:
                rate = scanned / max(1.0, (now - t0))
                print(
                    f"progress scanned={scanned} queued={len(to_repair)} repaired={repaired} "
                    f"elapsed_s={int(now - t0)} rate={rate:.1f}/s",
                    flush=True,
                )
                last_log = now
        print(f"collection_done {collection}", flush=True)

    if args.dry_run:
        elapsed = time.time() - t0
        print(
            f"done scanned={scanned} queued={len(to_repair)} repaired={repaired} elapsed_s={int(elapsed)}",
            flush=True,
        )
        return 0

    workers = max(1, int(args.workers))
    if workers == 1:
        for pq_path, collection in to_repair:
            ok = _repair_file(
                pq_path,
                collection,
                overwrite=bool(args.overwrite),
                compression=str(args.compression),
                row_group_size=int(args.row_group_size),
                temp_dir=temp_dir,
            )
            if ok:
                repaired += 1
    else:
        jobs: List[Tuple[str, str, bool, str, int, Optional[str]]] = [
            (
                str(p),
                c,
                bool(args.overwrite),
                str(args.compression),
                int(args.row_group_size),
                str(temp_dir) if temp_dir is not None else None,
            )
            for (p, c) in to_repair
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_repair_job, j) for j in jobs]
            for i, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    if bool(fut.result()):
                        repaired += 1
                except Exception:
                    pass

                now = time.time()
                if i % 50 == 0 or (now - last_log) >= 30:
                    rate = (scanned + i) / max(1.0, (now - t0))
                    print(
                        f"repair_progress repaired={repaired}/{len(to_repair)} "
                        f"completed={i}/{len(to_repair)} elapsed_s={int(now - t0)} rate={rate:.1f}/s",
                        flush=True,
                    )
                    last_log = now

    elapsed = time.time() - t0
    print(
        f"done scanned={scanned} queued={len(to_repair)} repaired={repaired} elapsed_s={int(elapsed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Allow piping to head/grep without noisy tracebacks.
        try:
            sys.stdout.close()
        except Exception:
            pass
        raise SystemExit(0)
