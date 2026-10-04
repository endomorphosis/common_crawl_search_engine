#!/usr/bin/env python3
"""Audit CC parquet shards for compression, schema invariants, row-group sizing, and sortedness signals.

This is a *metadata-first* scanner intended for large corpora. It avoids reading full
files unless you explicitly enable deeper sortedness checks.

What it checks per shard:
- Presence of provenance columns: collection, shard_file
- Compression codecs (from parquet metadata)
- Row group sizing:
  - median row-group rows vs expected (default: 71680)
  - median compressed row-group size vs expected MB (default: 4.0)
- Sortedness signal (cheap): uses host_rev min/max statistics when present

Outputs:
- JSONL results (one record per file)
- Summary JSON with per-collection aggregates and a list of files needing action

Resume:
- Uses a tiny sqlite DB keyed by absolute path, and skips files that are unchanged
  (same size + mtime).

Example:
  python src/common_crawl_search_engine/ccindex/audit_parquet_shards.py \
    --parquet-root /storage/ccindex_parquet/cc_pointers_by_collection \
    --state-db state/audit_parquet_shards.sqlite \
    --out-json state/audit_parquet_shards.jsonl \
    --summary-json state/audit_parquet_shards_summary.json \
    --require-codec ZSTD \
    --expected-row-group-rows 71680 \
    --expected-row-group-mb 4
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq


DEFAULT_REQUIRED_COLS = ("collection", "shard_file")


@dataclass(frozen=True)
class AuditConfig:
    required_cols: Tuple[str, ...]
    require_codec: Optional[str]
    expected_row_group_rows: int
    row_group_rows_tolerance: float
    expected_row_group_mb: float
    row_group_mb_tolerance: float
    check_sorted_stats: bool


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    try:
        return float(statistics.median(values))
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _codec_names_from_row_group(rg: pq.RowGroupMetaData) -> List[str]:
    codecs: List[str] = []
    try:
        for i in range(rg.num_columns):
            col = rg.column(i)
            codec = getattr(col, "compression", None)
            if codec is None:
                continue
            codecs.append(str(codec).upper())
    except Exception:
        return []
    # Keep order stable but unique
    out: List[str] = []
    seen = set()
    for c in codecs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _row_group_compressed_bytes(rg: pq.RowGroupMetaData) -> Optional[int]:
    try:
        total = 0
        for i in range(rg.num_columns):
            col = rg.column(i)
            n = getattr(col, "total_compressed_size", None)
            if n is None:
                return None
            total += int(n)
        return int(total)
    except Exception:
        return None


def _host_rev_min_max_stats(pf: pq.ParquetFile) -> Optional[List[Tuple[Optional[str], Optional[str]]]]:
    """Return per-rowgroup (min,max) statistics for host_rev if present.

    If any row group lacks stats, returns None.
    """

    try:
        md = pf.metadata
        schema = pf.schema_arrow
        if "host_rev" not in schema.names:
            return None

        # Find column index by name within parquet schema.
        # Parquet schema may include nested fields; ccindex schema is flat.
        parquet_schema = pf.schema
        col_idx = None
        for i in range(parquet_schema.num_columns):
            if parquet_schema.column(i).name == "host_rev":
                col_idx = i
                break
        if col_idx is None:
            return None

        out: List[Tuple[Optional[str], Optional[str]]] = []
        for rg_i in range(md.num_row_groups):
            rg = md.row_group(rg_i)
            col = rg.column(col_idx)
            st = getattr(col, "statistics", None)
            if st is None:
                return None
            mn = getattr(st, "min", None)
            mx = getattr(st, "max", None)
            # Arrow may hand back bytes for some encodings.
            if isinstance(mn, (bytes, bytearray)):
                try:
                    mn = mn.decode("utf-8", errors="replace")
                except Exception:
                    mn = None
            if isinstance(mx, (bytes, bytearray)):
                try:
                    mx = mx.decode("utf-8", errors="replace")
                except Exception:
                    mx = None
            out.append((mn if mn is None else str(mn), mx if mx is None else str(mx)))
        return out
    except Exception:
        return None


def _sorted_stats_signal(host_rev_stats: List[Tuple[Optional[str], Optional[str]]]) -> Tuple[Optional[bool], str]:
    """Cheap sortedness signal using row-group min/max stats.

    If for every adjacent row group i->i+1 we have max_i <= min_{i+1}, we consider
    it "likely sorted". This does not prove within-rowgroup ordering.
    """

    if not host_rev_stats:
        return None, "no_stats"

    for i in range(len(host_rev_stats) - 1):
        _min_i, max_i = host_rev_stats[i]
        min_j, _max_j = host_rev_stats[i + 1]
        if max_i is None or min_j is None:
            return None, "stats_missing_values"
        if str(max_i) > str(min_j):
            return False, f"boundary_violation_rg{i}_to_rg{i+1}"

    return True, "boundary_ok"


def _needs_rewrite(
    *,
    codecs: List[str],
    cfg: AuditConfig,
    missing_cols: List[str],
    median_rg_rows: Optional[float],
    median_rg_mb: Optional[float],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if missing_cols:
        reasons.append("missing_required_columns")

    if cfg.require_codec:
        want = cfg.require_codec.upper()
        # Treat mixed codecs as needing action.
        if not codecs:
            reasons.append("unknown_codec")
        elif any(c != want for c in codecs):
            reasons.append("codec_mismatch")

    if median_rg_rows is not None and cfg.expected_row_group_rows > 0:
        lo = cfg.expected_row_group_rows * (1.0 - cfg.row_group_rows_tolerance)
        hi = cfg.expected_row_group_rows * (1.0 + cfg.row_group_rows_tolerance)
        if not (lo <= median_rg_rows <= hi):
            reasons.append("row_group_rows_mismatch")

    if median_rg_mb is not None and cfg.expected_row_group_mb > 0:
        lo = cfg.expected_row_group_mb * (1.0 - cfg.row_group_mb_tolerance)
        hi = cfg.expected_row_group_mb * (1.0 + cfg.row_group_mb_tolerance)
        if not (lo <= median_rg_mb <= hi):
            reasons.append("row_group_mb_mismatch")

    return (len(reasons) > 0), reasons


class AuditState:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(db_path))
        self._init()

    def _init(self) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scanned (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                scanned_at TEXT NOT NULL,
                status_json TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scanned_at ON scanned(scanned_at)")
        self.con.commit()

    def is_fresh(self, path: Path) -> bool:
        try:
            st = path.stat()
        except Exception:
            return False
        cur = self.con.cursor()
        row = cur.execute(
            "SELECT size, mtime FROM scanned WHERE path = ?",
            (str(path),),
        ).fetchone()
        if not row:
            return False
        size, mtime = row
        return int(size) == int(st.st_size) and float(mtime) == float(st.st_mtime)

    def upsert(self, path: Path, status: Dict[str, Any]) -> None:
        try:
            st = path.stat()
            size = int(st.st_size)
            mtime = float(st.st_mtime)
        except Exception:
            size = 0
            mtime = 0.0
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO scanned(path, size, mtime, scanned_at, status_json)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size=excluded.size,
                mtime=excluded.mtime,
                scanned_at=excluded.scanned_at,
                status_json=excluded.status_json
            """,
            (str(path), size, mtime, _utc_ts(), json.dumps(status, sort_keys=True)),
        )

    def commit(self) -> None:
        self.con.commit()

    def close(self) -> None:
        try:
            self.con.commit()
        finally:
            self.con.close()


def _iter_parquet_files(parquet_root: Path) -> Iterable[Path]:
    # Only CC shards; avoid temp/marker files.
    # We include both sorted and unsorted for auditing.
    for p in parquet_root.rglob("*.parquet"):
        if not p.is_file():
            continue
        # Ignore hidden/work directories produced by sort/repair tooling.
        # These often contain zero-byte or partial parquet temp files.
        parts = p.parts
        if any(
            part.startswith(".cc_sort_work_")
            or part in {".duckdb_sort_tmp", ".duckdb"}
            for part in parts
        ):
            continue
        name = p.name
        if name.endswith(".sorting.part") or name.endswith(".sorted"):
            continue
        yield p


def _infer_collection_and_shard_file(path: Path) -> Tuple[Optional[str], Optional[str]]:
    # Expected layout: <root>/<year>/<collection>/<cdx-....parquet>
    try:
        shard_file = path.name
        collection = path.parent.name
        if collection.startswith("CC-MAIN-"):
            return collection, shard_file
        return None, shard_file
    except Exception:
        return None, None


def audit_one(path: Path, cfg: AuditConfig) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "path": str(path),
        "ts": _utc_ts(),
    }

    try:
        st = path.stat()
        rec["size_bytes"] = int(st.st_size)
        rec["mtime"] = float(st.st_mtime)
    except Exception:
        rec["size_bytes"] = None
        rec["mtime"] = None

    collection_guess, shard_guess = _infer_collection_and_shard_file(path)
    rec["collection_guess"] = collection_guess
    rec["shard_file_guess"] = shard_guess

    try:
        pf = pq.ParquetFile(str(path))
        md = pf.metadata
        rec["num_rows"] = int(md.num_rows)
        rec["num_row_groups"] = int(md.num_row_groups)
        cols = list(pf.schema_arrow.names)
        rec["columns"] = cols

        missing_cols = [c for c in cfg.required_cols if c not in set(cols)]
        rec["missing_required_columns"] = missing_cols

        codecs: List[str] = []
        if md.num_row_groups > 0:
            codecs = _codec_names_from_row_group(md.row_group(0))
        rec["codecs_rg0"] = codecs

        rg_rows: List[float] = []
        rg_mb: List[float] = []

        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            rg_rows.append(float(rg.num_rows))
            b = _row_group_compressed_bytes(rg)
            if b is not None:
                rg_mb.append(float(b) / (1024.0 * 1024.0))

        rec["median_row_group_rows"] = _median(rg_rows)
        rec["median_row_group_compressed_mb"] = _median(rg_mb)

        sorted_signal = None
        sorted_reason = "disabled"
        if cfg.check_sorted_stats:
            host_stats = _host_rev_min_max_stats(pf)
            if host_stats is None:
                sorted_signal, sorted_reason = None, "no_host_rev_stats"
            else:
                sorted_signal, sorted_reason = _sorted_stats_signal(host_stats)
        rec["sorted_stats_signal"] = sorted_signal
        rec["sorted_stats_reason"] = sorted_reason

        needs, reasons = _needs_rewrite(
            codecs=codecs,
            cfg=cfg,
            missing_cols=missing_cols,
            median_rg_rows=_safe_float(rec["median_row_group_rows"]),
            median_rg_mb=_safe_float(rec["median_row_group_compressed_mb"]),
        )
        rec["needs_action"] = bool(needs)
        rec["needs_action_reasons"] = reasons

        return rec

    except Exception as e:
        rec["error"] = str(e)
        rec["needs_action"] = True
        rec["needs_action_reasons"] = ["read_error"]
        return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit parquet shards for codec/sort/rowgroup/schema issues")
    ap.add_argument("--parquet-root", required=True, type=Path, help="Root containing parquet shards")
    ap.add_argument(
        "--state-db",
        type=Path,
        default=Path("state/audit_parquet_shards.sqlite"),
        help="SQLite state DB for resume (default: state/audit_parquet_shards.sqlite)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("state/audit_parquet_shards.jsonl"),
        help="Append JSONL results here (default: state/audit_parquet_shards.jsonl)",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=Path("state/audit_parquet_shards_summary.json"),
        help="Write a summary JSON here (default: state/audit_parquet_shards_summary.json)",
    )
    ap.add_argument(
        "--required-col",
        action="append",
        default=None,
        help="Repeatable: required column (default: collection and shard_file)",
    )
    ap.add_argument(
        "--require-codec",
        type=str,
        default=None,
        help="If set, flag shards whose codecs differ (e.g. ZSTD)",
    )
    ap.add_argument(
        "--expected-row-group-rows",
        type=int,
        default=int(os.environ.get("CC_SORT_ROW_GROUP_SIZE", "71680")),
        help="Expected row-group size in rows (default: env CC_SORT_ROW_GROUP_SIZE else 71680)",
    )
    ap.add_argument(
        "--row-group-rows-tolerance",
        type=float,
        default=0.02,
        help="Tolerance ratio for row-group rows check (default: 0.02 == ±2%)",
    )
    ap.add_argument(
        "--expected-row-group-mb",
        type=float,
        default=float(os.environ.get("CC_SORT_ROW_GROUP_TARGET_MB", "4")),
        help="Expected compressed row-group MB (default: env CC_SORT_ROW_GROUP_TARGET_MB else 4)",
    )
    ap.add_argument(
        "--row-group-mb-tolerance",
        type=float,
        default=0.35,
        help="Tolerance ratio for compressed row-group size check (default: 0.35 == ±35%)",
    )
    ap.add_argument(
        "--check-sorted-stats",
        action="store_true",
        default=False,
        help="Enable cheap sortedness signal using host_rev min/max statistics",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after auditing N files (for pilot runs)",
    )
    ap.add_argument(
        "--flush-every",
        type=int,
        default=200,
        help="Commit state/flush output every N audited files (default: 200)",
    )

    args = ap.parse_args(argv)

    parquet_root = args.parquet_root.expanduser().resolve()
    if not parquet_root.exists():
        print(f"❌ parquet-root not found: {parquet_root}", file=sys.stderr)
        return 1

    required_cols = tuple(args.required_col) if args.required_col else DEFAULT_REQUIRED_COLS

    cfg = AuditConfig(
        required_cols=required_cols,
        require_codec=(args.require_codec.upper() if args.require_codec else None),
        expected_row_group_rows=int(args.expected_row_group_rows),
        row_group_rows_tolerance=float(args.row_group_rows_tolerance),
        expected_row_group_mb=float(args.expected_row_group_mb),
        row_group_mb_tolerance=float(args.row_group_mb_tolerance),
        check_sorted_stats=bool(args.check_sorted_stats),
    )

    state = AuditState(args.state_db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    out_f = open(args.out_json, "a", encoding="utf-8")

    started = time.time()
    seen = 0
    audited = 0
    skipped = 0
    errors = 0

    per_collection: Dict[str, Dict[str, int]] = {}
    needs_action_files: List[str] = []

    def _bump(collection: str, key: str) -> None:
        d = per_collection.setdefault(collection, {})
        d[key] = int(d.get(key, 0)) + 1

    try:
        for p in _iter_parquet_files(parquet_root):
            seen += 1

            if args.max_files is not None and audited >= int(args.max_files):
                break

            if state.is_fresh(p):
                skipped += 1
                continue

            rec = audit_one(p, cfg)
            audited += 1

            collection = rec.get("collection_guess") or "unknown"
            _bump(collection, "audited")

            if rec.get("error"):
                errors += 1
                _bump(collection, "errors")

            if rec.get("needs_action"):
                needs_action_files.append(str(p))
                _bump(collection, "needs_action")

            out_f.write(json.dumps(rec, sort_keys=True) + "\n")
            state.upsert(p, rec)

            if audited % int(args.flush_every) == 0:
                out_f.flush()
                state.commit()
                elapsed = time.time() - started
                rate = audited / max(1e-9, elapsed)
                print(
                    f"progress seen={seen} audited={audited} skipped={skipped} errors={errors} "
                    f"rate={rate:.2f} files/s last={p.name}",
                    flush=True,
                )

        out_f.flush()
        state.commit()

    finally:
        try:
            out_f.close()
        except Exception:
            pass
        state.close()

    elapsed = time.time() - started
    summary = {
        "ts": _utc_ts(),
        "parquet_root": str(parquet_root),
        "config": {
            "required_cols": list(cfg.required_cols),
            "require_codec": cfg.require_codec,
            "expected_row_group_rows": cfg.expected_row_group_rows,
            "row_group_rows_tolerance": cfg.row_group_rows_tolerance,
            "expected_row_group_mb": cfg.expected_row_group_mb,
            "row_group_mb_tolerance": cfg.row_group_mb_tolerance,
            "check_sorted_stats": cfg.check_sorted_stats,
        },
        "counters": {
            "seen": seen,
            "audited": audited,
            "skipped_fresh": skipped,
            "errors": errors,
            "elapsed_seconds": elapsed,
            "files_per_second": (audited / elapsed) if elapsed > 0 else None,
            "needs_action": len(needs_action_files),
        },
        "per_collection": per_collection,
        "needs_action_files": needs_action_files,
    }

    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print(f"seen={seen} audited={audited} skipped_fresh={skipped} errors={errors} needs_action={len(needs_action_files)}")
    print(f"summary_json={args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
