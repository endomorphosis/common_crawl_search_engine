#!/usr/bin/env python3
"""Prototype: build a domain→row-group pointer index from already-sorted Parquet shards.

Goal
----
Create a DuckDB index that maps each domain (host_rev) to the Parquet row group(s)
that contain its rows, plus the exact row ranges for that domain within each row
group.

This enables an execution strategy like:
1) Lookup domain -> (parquet file, row_group id, row offsets)
2) Read only those row groups via PyArrow (pf.read_row_group)
3) Slice to the domain's row-range and apply any additional filtering

Notes
-----
- This script assumes shards are already sorted by (host_rev, url, ts) and that
  all rows for a given host_rev form contiguous runs.
- It does *not* build per-URL pointers.
- It intentionally stores per-(host_rev, row_group) segments. If a domain spans
  multiple row groups, it will have multiple rows in the index.

Example
-------
  python -m common_crawl_search_engine.ccindex.build_domain_rowgroup_index \
    --parquet-root /storage/ccindex_parquet/cc_pointers_by_collection/2024/CC-MAIN-2024-10 \
    --output-db /storage/ccindex_duckdb/proto_domain_rowgroups/CC-MAIN-2024-10.domain_rowgroups.duckdb \
    --batch-size 1 \
    --max-files 2
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

from concurrent.futures import ProcessPoolExecutor, as_completed
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def _worker_scan_to_ipc(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: scan one parquet file and write its segments to an Arrow IPC file.

    Must be a top-level function so it can be pickled by multiprocessing.
    """

    pq_path = Path(job["parquet_path"]).resolve()
    pq_root = Path(job["parquet_root"]).resolve()
    tmp_dir = Path(job["temp_dir"]).resolve()
    max_segments = job.get("max_segments_per_file")

    try:
        segs = _segment_rowgroup_host_revs(
            parquet_path=pq_path,
            parquet_root=pq_root,
            max_segments_per_file=(int(max_segments) if max_segments is not None else None),
        )
    except Exception as e:
        return {"ok": False, "parquet_path": str(pq_path), "error": str(e)}

    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ipc_path = tmp_dir / f"rowgroups_{pq_path.name}_{os.getpid()}_{uuid4().hex}.arrow"
        tbl = _segments_to_arrow(segs) if segs else pa.Table.from_batches([], schema=SEGMENT_SCHEMA)
        with ipc_path.open("wb") as f:
            with pa.ipc.new_file(f, tbl.schema) as writer:
                writer.write_table(tbl)
        return {
            "ok": True,
            "parquet_path": str(pq_path),
            "ipc_path": str(ipc_path),
            "segments": int(len(segs)),
        }
    except Exception as e:
        return {"ok": False, "parquet_path": str(pq_path), "error": f"IPC write failed: {e}"}


@dataclass
class DomainRowGroupSegment:
    source_path: str
    collection: Optional[str]
    year: Optional[int]
    shard_file: str
    parquet_relpath: str
    host_rev: str
    row_group: int
    rg_row_start: int
    rg_row_end: int
    dom_row_start: int
    dom_row_end: int
    dom_rg_row_start: int
    dom_rg_row_end: int


SEGMENT_SCHEMA = pa.schema(
    [
        ("source_path", pa.string()),
        ("collection", pa.string()),
        ("year", pa.int32()),
        ("shard_file", pa.string()),
        ("parquet_relpath", pa.string()),
        ("host_rev", pa.string()),
        ("row_group", pa.int32()),
        ("rg_row_start", pa.int64()),
        ("rg_row_end", pa.int64()),
        ("dom_row_start", pa.int64()),
        ("dom_row_end", pa.int64()),
        ("dom_rg_row_start", pa.int32()),
        ("dom_rg_row_end", pa.int32()),
    ]
)


def _iter_candidate_parquet_files(parquet_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for p in parquet_root.rglob("*.parquet"):
        try:
            if not p.is_file():
                continue
            rel = p.relative_to(parquet_root)
            if any(part.startswith(".") for part in rel.parts[:-1]):
                continue
            candidates.append(p)
        except Exception:
            continue

    # Prefer sorted shards when present.
    sorted_candidates = [p for p in candidates if p.name.endswith(".sorted.parquet")]
    return sorted(sorted_candidates if sorted_candidates else candidates)


def _parse_collection_year_from_path(p: Path) -> tuple[Optional[str], Optional[int]]:
    collection = None
    year: Optional[int] = None
    for part in p.parts:
        if part.startswith("CC-MAIN-"):
            collection = part
            try:
                year = int(part.split("-")[2])
            except Exception:
                year = None
            break
    return collection, year


def _segment_rowgroup_host_revs(
    *,
    parquet_path: Path,
    parquet_root: Path,
    max_segments_per_file: Optional[int] = None,
) -> List[DomainRowGroupSegment]:
    """Scan a Parquet file row group-by-row group and emit per-domain segments.

    Returns a list of segments (one per contiguous host_rev run within each row group).
    """

    pf = pq.ParquetFile(parquet_path)
    md = pf.metadata
    if md is None or int(md.num_row_groups or 0) <= 0:
        return []

    try:
        parquet_relpath = parquet_path.relative_to(parquet_root).as_posix()
    except Exception:
        parquet_relpath = str(parquet_path)

    collection, year = _parse_collection_year_from_path(parquet_path)
    shard_file = parquet_path.name
    src = str(parquet_path)

    segments: List[DomainRowGroupSegment] = []

    global_row_cursor = 0
    for rg_idx in range(int(md.num_row_groups)):
        rg = md.row_group(rg_idx)
        n = int(rg.num_rows or 0)
        rg_row_start = int(global_row_cursor)
        rg_row_end = int(global_row_cursor + n)

        # Read only the host_rev column for this row group.
        # This is the key: later, you can use this index to read only row groups.
        try:
            tbl = pf.read_row_group(rg_idx, columns=["host_rev"])
        except Exception as e:
            raise RuntimeError(f"Failed to read row group {rg_idx} host_rev from {parquet_path}: {e}")

        if tbl.num_rows != n:
            # Parquet metadata mismatch shouldn't happen, but keep going safely.
            n = int(tbl.num_rows)
            rg_row_end = int(rg_row_start + n)

        col = tbl.column(0)  # host_rev

        prev: Optional[str] = None
        run_start_global: Optional[int] = None
        run_start_in_rg: Optional[int] = None

        in_rg = 0
        for chunk in col.chunks:
            # Convert chunk to Python values. This is a prototype; optimize later if needed.
            vals = chunk.to_pylist()
            for v in vals:
                cur = None
                if v is not None:
                    # normalize to str
                    try:
                        cur = str(v)
                    except Exception:
                        cur = None

                if cur != prev:
                    if prev is not None and run_start_global is not None and run_start_in_rg is not None:
                        seg = DomainRowGroupSegment(
                            source_path=src,
                            collection=collection,
                            year=year,
                            shard_file=shard_file,
                            parquet_relpath=parquet_relpath,
                            host_rev=prev,
                            row_group=int(rg_idx),
                            rg_row_start=rg_row_start,
                            rg_row_end=rg_row_end,
                            dom_row_start=int(run_start_global),
                            dom_row_end=int(rg_row_start + in_rg),
                            dom_rg_row_start=int(run_start_in_rg),
                            dom_rg_row_end=int(in_rg),
                        )
                        segments.append(seg)
                        if max_segments_per_file is not None and len(segments) >= int(max_segments_per_file):
                            return segments

                    # Start a new run.
                    if cur is None:
                        prev = None
                        run_start_global = None
                        run_start_in_rg = None
                    else:
                        prev = cur
                        run_start_global = int(rg_row_start + in_rg)
                        run_start_in_rg = int(in_rg)

                in_rg += 1

        # finalize last run in this row group
        if prev is not None and run_start_global is not None and run_start_in_rg is not None:
            seg = DomainRowGroupSegment(
                source_path=src,
                collection=collection,
                year=year,
                shard_file=shard_file,
                parquet_relpath=parquet_relpath,
                host_rev=prev,
                row_group=int(rg_idx),
                rg_row_start=rg_row_start,
                rg_row_end=rg_row_end,
                dom_row_start=int(run_start_global),
                dom_row_end=int(rg_row_start + in_rg),
                dom_rg_row_start=int(run_start_in_rg),
                dom_rg_row_end=int(in_rg),
            )
            segments.append(seg)

        global_row_cursor = rg_row_end

        if max_segments_per_file is not None and len(segments) >= int(max_segments_per_file):
            return segments

    return segments


def _segments_to_arrow(rows: Sequence[DomainRowGroupSegment]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "source_path": [r.source_path for r in rows],
            "collection": [r.collection for r in rows],
            "year": [r.year for r in rows],
            "shard_file": [r.shard_file for r in rows],
            "parquet_relpath": [r.parquet_relpath for r in rows],
            "host_rev": [r.host_rev for r in rows],
            "row_group": [r.row_group for r in rows],
            "rg_row_start": [r.rg_row_start for r in rows],
            "rg_row_end": [r.rg_row_end for r in rows],
            "dom_row_start": [r.dom_row_start for r in rows],
            "dom_row_end": [r.dom_row_end for r in rows],
            "dom_rg_row_start": [r.dom_rg_row_start for r in rows],
            "dom_rg_row_end": [r.dom_rg_row_end for r in rows],
        },
        schema=SEGMENT_SCHEMA,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build prototype domain->row-group pointer index")
    ap.add_argument("--parquet-root", required=True, help="Root directory of parquet files")
    ap.add_argument("--output-db", required=True, help="Output DuckDB file")
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for reading/scanning parquet files (default: 1). DuckDB writes remain single-threaded.",
    )
    ap.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help=(
            "Temporary directory for intermediate Arrow IPC files when --workers>1. "
            "Default: create a temp directory next to the output DB."
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Commit every N files (default: 1; safest for prototyping)",
    )
    ap.add_argument("--max-files", type=int, default=None, help="Only process up to N parquet files")
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help=(
            "Restrict processing to specific shard(s). Accepts base names like "
            "'cdx-00257', 'cdx-00257.gz', 'cdx-00257.gz.parquet'. Can be repeated."
        ),
    )
    ap.add_argument(
        "--max-segments-per-file",
        type=int,
        default=None,
        help="Stop after emitting N segments per parquet file (debug/prototype)",
    )
    args = ap.parse_args()

    parquet_root = Path(args.parquet_root).expanduser().resolve()
    output_db = Path(args.output_db).expanduser().resolve()
    workers = max(1, int(args.workers or 1))

    if not parquet_root.exists():
        print(f"❌ parquet root not found: {parquet_root}", file=sys.stderr)
        return 2

    files = _iter_candidate_parquet_files(parquet_root)
    if args.only:
        only_raw = {str(x).strip() for x in args.only if str(x).strip()}

        def _matches_only(p: Path) -> bool:
            name = p.name
            stem = name
            for suf in (".gz.sorted.parquet", ".gz.parquet", ".sorted.parquet", ".parquet", ".gz"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            candidates = {
                stem,
                stem + ".gz",
                stem + ".gz.parquet",
                stem + ".gz.sorted.parquet",
                name,
            }
            return bool(candidates & only_raw)

        files = [p for p in files if _matches_only(p)]

    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]

    print(f"Parquet root: {parquet_root}")
    print(f"Output DB:    {output_db}")
    print(f"Files:        {len(files)}")
    print(f"Workers:      {workers}")
    print()

    output_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(output_db))
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS cc_domain_rowgroups (
                source_path VARCHAR,
                collection VARCHAR,
                year INTEGER,
                shard_file VARCHAR,
                parquet_relpath VARCHAR,
                host_rev VARCHAR,
                row_group INTEGER,
                rg_row_start BIGINT,
                rg_row_end BIGINT,
                dom_row_start BIGINT,
                dom_row_end BIGINT,
                dom_rg_row_start INTEGER,
                dom_rg_row_end INTEGER
            )
            """
        )

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS cc_indexed_parquet_files (
                parquet_path VARCHAR PRIMARY KEY,
                size_bytes BIGINT,
                mtime_ns BIGINT,
                indexed_at TIMESTAMP
            )
            """
        )

        # If the set of Parquet shards changes (deletions/renames, or switching
        # from unsorted to sorted candidates), remove stale rows so this index
        # remains correct across incremental reruns.
        try:
            target_paths = [str(p) for p in files]
            con.execute(
                "CREATE TEMP TABLE IF NOT EXISTS cc_target_parquet_files (parquet_path VARCHAR PRIMARY KEY)"
            )
            con.execute("DELETE FROM cc_target_parquet_files")
            if target_paths:
                con.executemany(
                    "INSERT INTO cc_target_parquet_files (parquet_path) VALUES (?)",
                    [(p,) for p in target_paths],
                )

            stale_count = con.execute(
                """
                SELECT COUNT(*)
                FROM cc_indexed_parquet_files
                WHERE parquet_path NOT IN (SELECT parquet_path FROM cc_target_parquet_files)
                """
            ).fetchone()[0]
            if int(stale_count or 0) > 0:
                print(
                    f"Pruning {int(stale_count):,} stale indexed parquet file(s) (no longer present under parquet-root)"
                )
                con.execute(
                    """
                    DELETE FROM cc_domain_rowgroups
                    WHERE source_path IN (
                        SELECT parquet_path FROM cc_indexed_parquet_files
                        WHERE parquet_path NOT IN (SELECT parquet_path FROM cc_target_parquet_files)
                    )
                    """
                )
                con.execute(
                    """
                    DELETE FROM cc_indexed_parquet_files
                    WHERE parquet_path NOT IN (SELECT parquet_path FROM cc_target_parquet_files)
                    """
                )
                con.commit()
        except Exception as e:
            print(f"Warning: failed to prune stale indexed parquet files: {e}", file=sys.stderr)

        commit_every = max(1, int(args.batch_size or 1))
        did_files = 0
        skipped_files = 0
        failed_files = 0
        total_segments = 0

        def _fingerprint(p: Path) -> tuple[int, int]:
            try:
                st = p.stat()
                return int(st.st_size), int(st.st_mtime_ns)
            except Exception:
                return -1, -1

        # Decide which files need work up-front (single-threaded DB reads).
        to_process: List[tuple[int, Path, int, int]] = []
        for idx, pq_file in enumerate(files, 1):
            size_bytes, mtime_ns = _fingerprint(pq_file)
            pq_path_str = str(pq_file)
            existing = con.execute(
                "SELECT size_bytes, mtime_ns FROM cc_indexed_parquet_files WHERE parquet_path = ?",
                [pq_path_str],
            ).fetchone()
            if existing and int(existing[0]) == size_bytes and int(existing[1]) == mtime_ns:
                skipped_files += 1
                continue
            to_process.append((idx, pq_file, size_bytes, mtime_ns))

        print(f"Plan: {len(files)} files total; {skipped_files} unchanged; {len(to_process)} to process")

        t0 = time.time()

        max_segments_per_file = int(args.max_segments_per_file) if args.max_segments_per_file is not None else None

        # Choose temp dir
        if workers > 1:
            if args.temp_dir:
                base_tmp = Path(args.temp_dir).expanduser().resolve()
            else:
                base_tmp = (output_db.parent / ".tmp_domain_rowgroup_index").resolve()
            base_tmp.mkdir(parents=True, exist_ok=True)
            # unique run dir to avoid clashes
            run_tmp = base_tmp / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
            run_tmp.mkdir(parents=True, exist_ok=True)
        else:
            run_tmp = None

        if workers == 1:
            # Preserve original single-threaded behavior.
            for idx, pq_file, size_bytes, mtime_ns in to_process:
                pq_path_str = str(pq_file)
                print(f"[{idx}/{len(files)}] Indexing rowgroup segments for {pq_file.name}...")

                try:
                    segs = _segment_rowgroup_host_revs(
                        parquet_path=pq_file,
                        parquet_root=parquet_root,
                        max_segments_per_file=max_segments_per_file,
                    )
                except Exception as e:
                    failed_files += 1
                    print(f"  ❌ failed: {e}", file=sys.stderr)
                    continue

                # Per-file idempotency (only after success so we don't lose data on failure).
                con.execute("DELETE FROM cc_domain_rowgroups WHERE source_path = ?", [pq_path_str])

                if segs:
                    tbl = _segments_to_arrow(segs)
                    con.register("_cc_domain_rowgroups", tbl)
                    con.execute("INSERT INTO cc_domain_rowgroups SELECT * FROM _cc_domain_rowgroups")
                    con.unregister("_cc_domain_rowgroups")
                    total_segments += int(len(segs))

                con.execute("DELETE FROM cc_indexed_parquet_files WHERE parquet_path = ?", [pq_path_str])
                con.execute(
                    "INSERT INTO cc_indexed_parquet_files (parquet_path, size_bytes, mtime_ns, indexed_at) VALUES (?, ?, ?, now())",
                    [pq_path_str, size_bytes, mtime_ns],
                )

                did_files += 1
                if (did_files % commit_every) == 0:
                    con.commit()

                dt = time.time() - t0
                rate = (did_files / dt) if dt > 0 else 0.0
                print(
                    f"  segments: {len(segs):,} (total={total_segments:,})  files_done={did_files}  rate={rate:.2f} files/s"
                )
        else:
            assert run_tmp is not None

            jobs: List[Dict[str, Any]] = []
            for idx, pq_file, size_bytes, mtime_ns in to_process:
                jobs.append(
                    {
                        "idx": int(idx),
                        "parquet_path": str(pq_file),
                        "parquet_root": str(parquet_root),
                        "temp_dir": str(run_tmp),
                        "max_segments_per_file": max_segments_per_file,
                        "size_bytes": int(size_bytes),
                        "mtime_ns": int(mtime_ns),
                    }
                )

            print(f"Spawning {workers} worker(s); IPC temp dir: {run_tmp}")

            # Submit all scans; write results into DuckDB as they complete.
            # Note: DuckDB writes remain in this (main) process.
            with ProcessPoolExecutor(max_workers=workers) as ex:
                fut_to_job = {ex.submit(_worker_scan_to_ipc, j): j for j in jobs}
                for fut in as_completed(fut_to_job):
                    j = fut_to_job[fut]
                    idx = int(j["idx"])
                    pq_path_str = str(j["parquet_path"])
                    pq_name = Path(pq_path_str).name

                    try:
                        res = fut.result()
                    except Exception as e:
                        failed_files += 1
                        print(f"[{idx}/{len(files)}] ❌ {pq_name}: worker crashed: {e}", file=sys.stderr)
                        continue

                    if not res.get("ok"):
                        failed_files += 1
                        print(
                            f"[{idx}/{len(files)}] ❌ {pq_name}: {res.get('error')}",
                            file=sys.stderr,
                        )
                        continue

                    ipc_path = Path(str(res.get("ipc_path"))).resolve()
                    seg_count = int(res.get("segments") or 0)

                    # Per-file idempotency (only after scan success).
                    con.execute("DELETE FROM cc_domain_rowgroups WHERE source_path = ?", [pq_path_str])

                    try:
                        with ipc_path.open("rb") as f:
                            reader = pa.ipc.open_file(f)
                            tbl = reader.read_all()
                        if tbl.num_rows:
                            con.register("_cc_domain_rowgroups", tbl)
                            con.execute("INSERT INTO cc_domain_rowgroups SELECT * FROM _cc_domain_rowgroups")
                            con.unregister("_cc_domain_rowgroups")
                            total_segments += int(seg_count)

                        con.execute("DELETE FROM cc_indexed_parquet_files WHERE parquet_path = ?", [pq_path_str])
                        con.execute(
                            "INSERT INTO cc_indexed_parquet_files (parquet_path, size_bytes, mtime_ns, indexed_at) VALUES (?, ?, ?, now())",
                            [pq_path_str, int(j["size_bytes"]), int(j["mtime_ns"])],
                        )

                        did_files += 1
                        if (did_files % commit_every) == 0:
                            con.commit()

                        dt = time.time() - t0
                        rate = (did_files / dt) if dt > 0 else 0.0
                        print(
                            f"[{idx}/{len(files)}] ✅ {pq_name}: segments={seg_count:,} total_segments={total_segments:,} "
                            f"files_done={did_files} rate={rate:.2f} files/s"
                        )
                    finally:
                        try:
                            ipc_path.unlink(missing_ok=True)
                        except Exception:
                            pass

            # Best-effort cleanup of empty temp dir.
            try:
                if run_tmp.exists():
                    for p in run_tmp.glob("*"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    run_tmp.rmdir()
            except Exception:
                pass

        con.commit()

        print("\nCreating indexes...")
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_cc_domain_rowgroups_host_rev ON cc_domain_rowgroups(host_rev)",
            "CREATE INDEX IF NOT EXISTS idx_cc_domain_rowgroups_collection ON cc_domain_rowgroups(collection)",
            "CREATE INDEX IF NOT EXISTS idx_cc_domain_rowgroups_year ON cc_domain_rowgroups(year)",
            "CREATE INDEX IF NOT EXISTS idx_cc_domain_rowgroups_parquet_rg ON cc_domain_rowgroups(parquet_relpath, row_group)",
        ]:
            try:
                con.execute(stmt)
            except Exception:
                pass

        print("\nDone")
        print(f"  processed: {did_files:,} files (skipped unchanged: {skipped_files:,}, failed: {failed_files:,})")
        print(f"  segments:  {total_segments:,}")
        try:
            print(f"  db size:   {output_db.stat().st_size / (1024**3):.3f} GB")
        except Exception:
            pass

    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
