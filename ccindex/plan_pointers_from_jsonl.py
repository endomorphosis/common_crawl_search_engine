#!/usr/bin/env python3
"""Build a unified Common Crawl pointer list from a JSONL of seed URLs/domains.

This is analogous to `plan_pointers_from_csv.py`, but for JSONL inputs like
`artifacts/state_agencies_all.jsonl`.

Input: JSONL where each line is a dict. We extract domains from, in order:
  - --domain-field (default: "domain")
  - --host-field (default: "host")
  - --url-field (default: "agency_url")

Output:
    - pointers.parquet (zstd) by default under:
            <cache-root>/slice_indexes/<run-id>/pointers.parquet

Schema note:
    By default, this writes the same "legacy" pointers schema used by the municipal
    planner so downstream tooling can treat these runs uniformly:

        domain (string)
        url (string)               # CC index URL for the record
        collection (string)
        timestamp (string)
        mime (string)
        status (int32)
        warc_filename (string)
        warc_offset (int64)
        warc_length (int64)
        gnis (string)              # will be null for state agencies
        place_name (string)        # will be null for state agencies
        state_code (string)        # will be null for state agencies

    To preserve *all input URLs per domain* (e.g. all agency_url values), we also
    write a sidecar JSONL mapping domains to their input URLs.

The Parquet file always includes the required CC pointer fields:
  warc_filename, warc_offset, warc_length

Additional best-effort enrichment columns are included for convenience.
Downstream slice planning only needs the warc_* fields.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from common_crawl_search_engine.ccindex import api


def _require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Parquet output (install pyarrow)") from e
    return pa, pq


def _default_cache_root() -> Path:
    env = str(__import__("os").environ.get("CCINDEX_CACHE_ROOT") or "").strip()
    if env:
        return Path(env)
    return Path("datasets") / "CCINDEX_WARC_CACHE_DIR"


def _run_dir(cache_root: Path, run_id: str) -> Path:
    return Path(cache_root) / "slice_indexes" / str(run_id)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _url_to_domain(url_or_host: str) -> Optional[str]:
    s = str(url_or_host or "").strip()
    if not s:
        return None

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
        host = api.normalize_domain(s)

    host = api.normalize_domain(host)

    # Reject obviously invalid domains/hosts that can appear in noisy inputs.
    # Examples: ".gov", "wv..gov", leading/trailing dot.
    if not host:
        return None
    if host.startswith(".") or host.endswith("."):
        return None
    if ".." in host:
        return None
    return host


@dataclass(frozen=True)
class DomainMeta:
    domain: str
    meta: Dict[str, object]


def build_domains_from_jsonl(
    jsonl_path: Path,
    *,
    domain_field: str,
    host_field: str,
    url_fields: Sequence[str],
) -> Tuple[List[str], Dict[str, Dict[str, object]], Dict[str, List[str]]]:
    meta_by: Dict[str, Dict[str, object]] = {}
    urls_by_domain: Dict[str, set[str]] = {}
    domains: List[str] = []
    seen: set[str] = set()

    def iter_url_values(rec: Dict[str, object], field: str) -> Iterable[str]:
        if not field:
            return
        v = rec.get(field)
        if v is None:
            return
        if isinstance(v, (list, tuple)):
            for x in v:
                s = str(x or "").strip()
                if s:
                    yield s
            return
        s = str(v or "").strip()
        if s:
            yield s

    for rec in _iter_jsonl(jsonl_path):
        dom = _url_to_domain(str(rec.get(domain_field) or "").strip())
        if not dom:
            dom = _url_to_domain(str(rec.get(host_field) or "").strip())
        if not dom:
            for f in url_fields:
                for raw in iter_url_values(rec, f):
                    dom = _url_to_domain(raw)
                    if dom:
                        break
                if dom:
                    break
        if not dom:
            continue

        # Record all input URLs associated with this domain.
        for f in url_fields:
            for raw_u in iter_url_values(rec, f):
                urls_by_domain.setdefault(dom, set()).add(raw_u)

        if dom not in meta_by:
            # Keep the first record's provenance as best-effort enrichment.
            meta_by[dom] = {
                "jurisdiction": rec.get("jurisdiction"),
                "name": rec.get("name"),
                "branch": rec.get("branch"),
                "agency_name": rec.get("agency_name"),
                "agency_url": rec.get("agency_url"),
                "host": rec.get("host"),
                "domain": rec.get("domain"),
                "seed_url": rec.get("seed_url"),
                "seed_source": rec.get("seed_source"),
                "discovered_from": rec.get("discovered_from"),
            }

        if dom in seen:
            continue
        seen.add(dom)
        domains.append(dom)

    urls_list_by_domain: Dict[str, List[str]] = {
        d: sorted(list(v)) for d, v in urls_by_domain.items() if v
    }
    return domains, meta_by, urls_list_by_domain


def _parquet_schema_legacy(pa) -> object:
    # Keep consistent with the municipal pointers schema.
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

        # Legacy schema enrichment.
        cols["gnis"].append(r.get("gnis"))
        cols["place_name"].append(r.get("place_name"))
        cols["state_code"].append(r.get("state_code"))

    arrays = [pa.array(cols[name], type=schema.field(name).type) for name in schema.names]
    return pa.Table.from_arrays(arrays, names=schema.names)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build ccindex pointers parquet from a JSONL (e.g. state agencies)")
    ap.add_argument("--jsonl", type=Path, required=True, help="Input JSONL")
    ap.add_argument("--cache-root", type=Path, default=None, help="Cache root (default: datasets/CCINDEX_WARC_CACHE_DIR)")
    ap.add_argument("--run-id", type=str, default=None, help="Run ID (default: UTC timestamp)")
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=None,
        help="Output parquet path (default: <cache-root>/slice_indexes/<run-id>/pointers.parquet)",
    )
    ap.add_argument(
        "--out-domain-sources-jsonl",
        type=Path,
        default=None,
        help="Output JSONL mapping domain -> input URLs (default: <cache-root>/slice_indexes/<run-id>/domain_sources.jsonl)",
    )

    ap.add_argument("--domain-field", type=str, default="domain")
    ap.add_argument("--host-field", type=str, default="host")
    ap.add_argument("--url-field", type=str, default="agency_url")
    ap.add_argument(
        "--url-fields",
        action="append",
        default=[],
        help="Additional URL fields to use for domain extraction and domain_sources (repeatable; supports list-valued fields)",
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
    ap.add_argument("--parquet-root", type=Path, default=Path("/storage/ccindex_parquet"), help="Parquet root")

    # Caps: <=0 means no cap.
    ap.add_argument("--max-parquet-files", type=int, default=0)
    ap.add_argument("--max-matches", type=int, default=0)
    ap.add_argument("--per-parquet-limit", type=int, default=0)

    ap.add_argument("--domain-workers", type=int, default=12)
    ap.add_argument("--domains-limit", type=int, default=None)
    ap.add_argument("--emit-stats", action="store_true", default=True)
    ap.add_argument("--no-emit-stats", dest="emit_stats", action="store_false")

    args = ap.parse_args(list(argv) if argv is not None else None)

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if args.cache_root is not None
        else _default_cache_root().resolve()
    )

    run_id = str(args.run_id).strip() if args.run_id is not None else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = _run_dir(cache_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = (
        Path(args.out_parquet).expanduser().resolve()
        if args.out_parquet is not None
        else (run_dir / "pointers.parquet").resolve()
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.unlink(missing_ok=True)

    # Back-compat: --url-field is the primary; --url-fields can add more.
    combined_url_fields: List[str] = []
    if args.url_field:
        combined_url_fields.append(str(args.url_field))
    for f in (args.url_fields or []):
        sf = str(f or "").strip()
        if sf:
            combined_url_fields.append(sf)
    # Deduplicate while preserving order.
    _seen_fields: set[str] = set()
    combined_url_fields = [f for f in combined_url_fields if not (f in _seen_fields or _seen_fields.add(f))]

    domains, meta_by, urls_by_domain = build_domains_from_jsonl(
        jsonl_path,
        domain_field=str(args.domain_field),
        host_field=str(args.host_field),
        url_fields=combined_url_fields,
    )

    out_sources = (
        Path(args.out_domain_sources_jsonl).expanduser().resolve()
        if args.out_domain_sources_jsonl is not None
        else (run_dir / "domain_sources.jsonl").resolve()
    )
    out_sources.parent.mkdir(parents=True, exist_ok=True)
    out_sources.write_text("", encoding="utf-8")

    # Persist full input URL lists per domain for later narrowing.
    try:
        with out_sources.open("a", encoding="utf-8") as f:
            for d in sorted(urls_by_domain.keys()):
                f.write(
                    json.dumps({"domain": d, "input_urls": urls_by_domain.get(d) or []}, ensure_ascii=False)
                    + "\n"
                )
    except Exception:
        pass

    if args.domains_limit is not None:
        domains = domains[: max(0, int(args.domains_limit))]

    if not domains:
        sys.stderr.write("No domains found in JSONL\n")
        return 2

    out_parquet_tmp = Path(str(out_parquet) + ".inprogress")

    try:
        (cache_root / "slice_indexes" / "LATEST.txt").write_text(str(run_id) + "\n", encoding="utf-8")
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "stage": "pointers-from-jsonl",
                    "jsonl": str(jsonl_path),
                    "cache_root": str(cache_root),
                    "run_id": str(run_id),
                    "pointers_parquet": str(out_parquet),
                    "pointers_parquet_inprogress": str(out_parquet_tmp),
                    "domain_sources_jsonl": str(out_sources),
                    "domain_field": str(args.domain_field),
                    "host_field": str(args.host_field),
                    "url_field": str(args.url_field),
                    "url_fields": combined_url_fields,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    sys.stderr.write(
        f"domains={len(domains)} out={out_parquet} run_dir={run_dir} ranges_dir={cache_root / 'ranges'}\n"
    )

    try:
        out_parquet_tmp.unlink()
    except FileNotFoundError:
        pass

    pa, pq = _require_pyarrow()
    schema = _parquet_schema_legacy(pa)
    writer = pq.ParquetWriter(
        str(out_parquet_tmp),
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    lock = __import__("threading").Lock()
    writer_lock = __import__("threading").Lock()

    started = time.time()
    completed = 0
    total_emitted = 0
    failed = 0

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
            tbl = _rows_to_table(pa, schema, batch)
            with writer_lock:
                writer.write_table(tbl)
            with lock:
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
                    "gnis": None,
                    "place_name": None,
                    "state_code": None,
                }
            )
            if len(batch) >= 2000:
                flush()

        flush()
        return dom, emitted, str(stats.get("meta_source") or "")

    workers = max(1, int(args.domain_workers or 1))
    try:
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
    finally:
        try:
            writer.close()
        except Exception:
            pass

        # Only publish the final parquet when the run completed without per-domain failures.
        # While running, the .inprogress file is not readable because Parquet footers are written on close.
        if failed == 0:
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
