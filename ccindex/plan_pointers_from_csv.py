#!/usr/bin/env python3
"""Build a unified Common Crawl pointer list from a CSV of municipal URLs.

This reads a CSV (like data/us_towns_and_counties_urls.csv), extracts all hostnames
from the `source_url` column, then runs ccindex domain search for each hostname.

Output is JSONL (one record per CC pointer). This is intended to be fed into
`plan_slices_from_pointers.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from common_crawl_search_engine.ccindex import api


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for --out-parquet (install pyarrow)"
        ) from e
    return pa, pq


@dataclass(frozen=True)
class CsvRow:
    gnis: str
    place_name: str
    state_code: str
    source_url: str
    status: str


def _iter_csv_rows(csv_path: Path) -> Iterator[CsvRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not isinstance(row, dict):
                continue
            yield CsvRow(
                gnis=str(row.get("gnis") or "").strip(),
                place_name=str(row.get("place_name") or "").strip(),
                state_code=str(row.get("state_code") or "").strip(),
                source_url=str(row.get("source_url") or "").strip(),
                status=str(row.get("status") or "").strip(),
            )


def _split_source_urls(raw: str) -> List[str]:
    # CSV contains comma-separated URLs in a single cell.
    # We treat commas as separators and keep simple; these URLs should not contain commas.
    parts = [p.strip() for p in str(raw or "").split(",")]
    return [p for p in parts if p]


def _url_to_domain(url_or_host: str) -> Optional[str]:
    s = str(url_or_host or "").strip()
    if not s:
        return None

    # Handle bare hosts like "nyc.gov".
    if "://" not in s:
        s2 = "http://" + s
    else:
        s2 = s
    try:
        u = urlparse(s2)
        host = (u.hostname or "").strip()
    except Exception:
        host = ""
    if not host:
        # Fall back to ccindex's normalize_domain best-effort.
        host = api.normalize_domain(s)
    host = api.normalize_domain(host)
    return host or None


def build_domains_from_csv(csv_path: Path) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Return (domains, meta_by_domain).

    meta_by_domain is best-effort, used only for enriching pointer JSONL.
    If multiple CSV rows map to the same domain, the first row wins.
    """

    meta_by: Dict[str, Dict[str, str]] = {}
    domains: List[str] = []
    seen: set[str] = set()

    for row in _iter_csv_rows(csv_path):
        for u in _split_source_urls(row.source_url):
            dom = _url_to_domain(u)
            if not dom:
                continue
            if dom not in meta_by:
                meta_by[dom] = {
                    "gnis": row.gnis,
                    "place_name": row.place_name,
                    "state_code": row.state_code,
                    "source_url": row.source_url,
                    "status": row.status,
                }
            if dom in seen:
                continue
            seen.add(dom)
            domains.append(dom)
    return domains, meta_by


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _parquet_schema(pa) -> object:
    # Keep types stable for downstream tools.
    return pa.schema(
        [
            pa.field("domain", pa.string()),
            pa.field("url", pa.string()),
            pa.field("collection", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("mime", pa.string()),
            pa.field("status", pa.int32()),
            pa.field("warc_filename", pa.string()),
            pa.field("warc_offset", pa.int64()),
            pa.field("warc_length", pa.int64()),
            pa.field("gnis", pa.string()),
            pa.field("place_name", pa.string()),
            pa.field("state_code", pa.string()),
        ]
    )


def _rows_to_table(pa, schema: object, rows: List[Dict[str, object]]):
    # Normalize and coerce values into the schema.
    cols: Dict[str, List[object]] = {name: [] for name in schema.names}
    for r in rows:
        cols["domain"].append(r.get("domain"))
        cols["url"].append(r.get("url"))
        cols["collection"].append(r.get("collection"))
        cols["timestamp"].append(r.get("timestamp"))
        cols["mime"].append(r.get("mime"))

        st = r.get("status")
        try:
            cols["status"].append(int(st) if st is not None else None)
        except Exception:
            cols["status"].append(None)

        cols["warc_filename"].append(r.get("warc_filename"))

        off = r.get("warc_offset")
        try:
            cols["warc_offset"].append(int(off) if off is not None else None)
        except Exception:
            cols["warc_offset"].append(None)

        ln = r.get("warc_length")
        try:
            cols["warc_length"].append(int(ln) if ln is not None else None)
        except Exception:
            cols["warc_length"].append(None)

        cols["gnis"].append(r.get("gnis"))
        cols["place_name"].append(r.get("place_name"))
        cols["state_code"].append(r.get("state_code"))

    arrays = [pa.array(cols[name], type=schema.field(name).type) for name in schema.names]
    return pa.Table.from_arrays(arrays, names=schema.names)


def _default_cache_root() -> Path:
    env = str(__import__("os").environ.get("CCINDEX_CACHE_ROOT") or "").strip()
    if env:
        return Path(env)
    return Path("datasets") / "CCINDEX_WARC_CACHE_DIR"


def _run_dir(cache_root: Path, run_id: str) -> Path:
    return Path(cache_root) / "slice_indexes" / str(run_id)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build ccindex pointer JSONL from a municipal URL CSV")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV (must include source_url column)")
    ap.add_argument("--cache-root", type=Path, default=None, help="Cache root (default: datasets/CCINDEX_WARC_CACHE_DIR)")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for slice index outputs (default: UTC timestamp)",
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Output JSONL path (default: <cache-root>/slice_indexes/<run-id>/pointers.jsonl)",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Output Parquet path (zstd) (default: <cache-root>/slice_indexes/<run-id>/pointers.parquet). If set, JSONL is not written.",
    )

    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--master-db",
        type=Path,
        default=Path("/storage/ccindex_duckdb/cc_pointers_master/cc_master_index.duckdb"),
        help="Master meta-index DuckDB",
    )
    src.add_argument("--year-db", type=Path, default=None, help="Year meta-index DuckDB")
    src.add_argument("--collection-db", type=Path, default=None, help="Single collection DuckDB")

    ap.add_argument("--year", type=str, default=None, help="Restrict to a year (only used with --master-db)")
    ap.add_argument(
        "--parquet-root",
        type=Path,
        default=Path("/storage/ccindex_parquet"),
        help="Parquet root",
    )
    ap.add_argument(
        "--max-parquet-files",
        type=int,
        default=0,
        help="Max parquet shard files to scan per collection (<=0 means no cap)",
    )
    ap.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Max pointers to emit per domain (<=0 means no cap)",
    )
    ap.add_argument(
        "--per-parquet-limit",
        type=int,
        default=0,
        help="Max pointers to read per parquet shard (<=0 means no cap)",
    )
    ap.add_argument("--domain-workers", type=int, default=12, help="Parallelism for per-domain index search")
    ap.add_argument("--domains-limit", type=int, default=None, help="For testing: only process first N domains")
    ap.add_argument("--emit-stats", action="store_true", default=True)
    ap.add_argument("--no-emit-stats", dest="emit_stats", action="store_false")

    args = ap.parse_args(list(argv) if argv is not None else None)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    cache_root = (Path(args.cache_root).expanduser().resolve() if args.cache_root is not None else _default_cache_root().resolve())
    run_id = str(args.run_id).strip() if args.run_id is not None else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = _run_dir(cache_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = (
        Path(args.out_parquet).expanduser().resolve()
        if args.out_parquet is not None
        else None
    )
    out_jsonl = None
    if out_parquet is None:
        out_jsonl = (
            Path(args.out_jsonl).expanduser().resolve()
            if args.out_jsonl is not None
            else (run_dir / "pointers.jsonl").resolve()
        )
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl.write_text("", encoding="utf-8")
    else:
        if str(out_parquet).strip() == "":
            raise SystemExit("--out-parquet must not be empty")
        if out_parquet.suffix.lower() != ".parquet":
            out_parquet = out_parquet.with_suffix(out_parquet.suffix + ".parquet")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        # Truncate if exists.
        out_parquet.unlink(missing_ok=True)

    domains, meta_by = build_domains_from_csv(csv_path)
    if args.domains_limit is not None:
        domains = domains[: max(0, int(args.domains_limit))]

    if not domains:
        sys.stderr.write("No domains found in CSV\n")
        return 2

    meta_path = run_dir / "meta.json"
    try:
        meta_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "stage": "pointers-from-csv",
                    "csv": str(csv_path),
                    "cache_root": str(cache_root),
                    "run_id": str(run_id),
                    "pointers_jsonl": (str(out_jsonl) if out_jsonl is not None else None),
                    "pointers_parquet": (str(out_parquet) if out_parquet is not None else None),
                    "pointers_parquet_inprogress": (
                        (str(Path(str(out_parquet) + ".inprogress")) if out_parquet is not None else None)
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (cache_root / "slice_indexes" / "LATEST.txt").write_text(str(run_id) + "\n", encoding="utf-8")
    except Exception:
        pass

    out_desc = str(out_parquet) if out_parquet is not None else str(out_jsonl)
    sys.stderr.write(f"domains={len(domains)} out={out_desc} run_dir={run_dir} ranges_dir={cache_root / 'ranges'}\n")

    lock = __import__("threading").Lock()
    started = time.time()
    completed = 0
    total_emitted = 0
    failed = 0

    parquet_lock = __import__("threading").Lock()
    parquet_writer = None
    parquet_schema = None
    out_parquet_tmp = None
    if out_parquet is not None:
        out_parquet_tmp = Path(str(out_parquet) + ".inprogress")
        try:
            out_parquet_tmp.unlink()
        except FileNotFoundError:
            pass

        pa, pq = _require_pyarrow()
        parquet_schema = _parquet_schema(pa)
        parquet_writer = pq.ParquetWriter(
            str(out_parquet_tmp),
            parquet_schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )

    def _search_one(dom: str) -> Tuple[str, int, str]:
        nonlocal total_emitted

        meta = meta_by.get(dom) or {}
        stats: Dict[str, object] = {}
        emitted = 0
        batch: List[Dict[str, object]] = []

        def flush() -> None:
            nonlocal emitted, total_emitted
            if not batch:
                return
            if out_parquet is not None:
                pa, _pq = _require_pyarrow()
                assert parquet_schema is not None
                tbl = _rows_to_table(pa, parquet_schema, batch)
                with parquet_lock:
                    assert parquet_writer is not None
                    parquet_writer.write_table(tbl)
                with lock:
                    total_emitted += len(batch)
            else:
                assert out_jsonl is not None
                with lock:
                    _write_jsonl(out_jsonl, batch)
                    total_emitted += len(batch)
            emitted += len(batch)
            batch.clear()

        for r in api.iter_domain_records_via_meta_indexes(
            dom,
            parquet_root=Path(args.parquet_root),
            master_db=Path(args.master_db) if args.master_db is not None else None,
            year_db=Path(args.year_db) if args.year_db is not None else None,
            collection_db=Path(args.collection_db) if args.collection_db is not None else None,
            year=str(args.year) if args.year is not None else None,
            max_parquet_files=int(args.max_parquet_files),
            max_matches=int(args.max_matches),
            per_parquet_limit=int(args.per_parquet_limit),
            stats_out=stats,
        ):
            if not isinstance(r, dict):
                continue
            batch.append(
                {
                    "domain": dom,
                    "url": r.get("url"),
                    "collection": r.get("collection"),
                    "timestamp": r.get("timestamp"),
                    "mime": r.get("mime"),
                    "status": r.get("status"),
                    "warc_filename": r.get("warc_filename"),
                    "warc_offset": r.get("warc_offset"),
                    "warc_length": r.get("warc_length"),
                    # Best-effort enrichment from the CSV.
                    "gnis": meta.get("gnis"),
                    "place_name": meta.get("place_name"),
                    "state_code": meta.get("state_code"),
                }
            )
            if len(batch) >= 2000:
                flush()

        flush()
        return dom, emitted, str(stats.get("meta_source") or "")

    workers = max(1, int(args.domain_workers or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_search_one, d) for d in domains]
        for fut in as_completed(futs):
            try:
                dom, emitted, meta_source = fut.result()
            except Exception as e:
                failed += 1
                completed += 1
                if args.emit_stats:
                    sys.stderr.write(
                        f"done={completed}/{len(domains)} domain=? emitted=0 error={type(e).__name__}: {e} "
                        f"elapsed_s={time.time() - started:.1f}\n"
                    )
                continue

            completed += 1
            if args.emit_stats:
                sys.stderr.write(
                    f"done={completed}/{len(domains)} domain={dom} emitted={emitted} "
                    f"meta_source={meta_source} elapsed_s={time.time() - started:.1f}\n"
                )

    if parquet_writer is not None:
        try:
            parquet_writer.close()
        except Exception:
            pass

        if failed == 0 and out_parquet_tmp is not None:
            try:
                __import__("os").replace(str(out_parquet_tmp), str(out_parquet))
            except Exception:
                pass

    sys.stderr.write(
        f"ok={(1 if failed==0 else 0)} domains={len(domains)} completed={completed} failed={failed} "
        f"pointers={total_emitted} elapsed_s={time.time() - started:.1f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
