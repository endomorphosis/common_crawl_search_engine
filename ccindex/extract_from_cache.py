#!/usr/bin/env python3
"""Extract web pages from cached slice blobs (no-origin scraping).

This consumes the artifacts produced by:
  - ccindex plan pointers-from-csv
  - ccindex plan slices-from-pointers
  - ccindex plan fetch-slices

Given:
  - pointers.jsonl (URL + warc_* pointers)
  - slice_members.jsonl (pointer->slice mapping)
  - cached slice blobs in <cache_root>/ranges/

It reconstructs each gzip-member (WARC record) by slicing the cached blob, then
parses the embedded HTTP payload via ccindex.api.extract_http_from_warc_gzip_member().

By default this is cache-only: missing slices are reported and skipped. You can
enable --allow-network to fetch missing slices and populate the cache.
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

from common_crawl_search_engine.ccindex import api


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to read Parquet pointers (install pyarrow)"
        ) from e
    return pq


def _default_cache_root() -> Path:
    return (Path("datasets") / "CCINDEX_WARC_CACHE_DIR").resolve()


def _infer_run_id(cache_root: Path, run_id: Optional[str]) -> str:
    rid = str(run_id).strip() if run_id is not None else ""
    if rid:
        return rid
    try:
        latest = (cache_root / "slice_indexes" / "LATEST.txt").read_text(encoding="utf-8").strip()
        if latest:
            return latest
    except Exception:
        pass
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _iter_parquet(path: Path) -> Iterator[Dict[str, object]]:
    pq = _require_pyarrow()
    pf = pq.ParquetFile(str(path))
    # Pull the columns we most often want. Missing columns are treated as nulls.
    want_cols = [
        "domain",
        "url",
        "collection",
        "timestamp",
        "mime",
        "status",
        "warc_filename",
        "warc_offset",
        "warc_length",
        "gnis",
        "place_name",
        "state_code",
    ]
    # Some files might be missing enrichment columns; ask only for those present.
    schema_names = set(pf.schema.names)
    cols = [c for c in want_cols if c in schema_names]
    for batch in pf.iter_batches(batch_size=65_536, columns=cols, use_threads=True):
        col_py = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        n = len(next(iter(col_py.values()))) if col_py else 0
        for i in range(n):
            yield {k: v[i] for k, v in col_py.items()}


def _iter_pointers(path: Path) -> Iterator[Dict[str, object]]:
    if path.suffix.lower() == ".parquet":
        yield from _iter_parquet(path)
        return
    yield from _iter_jsonl(path)


@dataclass(frozen=True)
class PointerKey:
    warc_filename: str
    warc_offset: int
    warc_length: int


@dataclass(frozen=True)
class SliceKey:
    warc_filename: str
    slice_start: int
    slice_end: int


def _load_pointer_meta(pointers_path: Path) -> Dict[PointerKey, Dict[str, object]]:
    meta: Dict[PointerKey, Dict[str, object]] = {}
    for rec in _iter_pointers(pointers_path):
        wf = str(rec.get("warc_filename") or "").strip()
        if not wf:
            continue
        try:
            off = int(rec.get("warc_offset"))
            ln = int(rec.get("warc_length"))
        except Exception:
            continue
        if off < 0 or ln <= 0:
            continue
        key = PointerKey(warc_filename=wf, warc_offset=int(off), warc_length=int(ln))
        # Keep only one; prefer first occurrence (stable).
        if key not in meta:
            meta[key] = rec
    return meta


def _group_members(members_jsonl: Path) -> Tuple[Dict[SliceKey, List[PointerKey]], int]:
    by_slice: Dict[SliceKey, List[PointerKey]] = {}
    total = 0
    for rec in _iter_jsonl(members_jsonl):
        wf = str(rec.get("warc_filename") or "").strip()
        if not wf:
            continue
        try:
            s0 = int(rec.get("slice_start"))
            s1 = int(rec.get("slice_end"))
            off = int(rec.get("warc_offset"))
            ln = int(rec.get("warc_length"))
        except Exception:
            continue
        if s0 < 0 or s1 < s0 or off < 0 or ln <= 0:
            continue
        sk = SliceKey(warc_filename=wf, slice_start=int(s0), slice_end=int(s1))
        pk = PointerKey(warc_filename=wf, warc_offset=int(off), warc_length=int(ln))
        by_slice.setdefault(sk, []).append(pk)
        total += 1
    # De-dupe pointer keys within each slice.
    for sk in list(by_slice.keys()):
        uniq = sorted(set(by_slice[sk]), key=lambda k: (k.warc_offset, k.warc_length))
        by_slice[sk] = uniq
    return by_slice, total


def _read_cached_slice(
    *,
    range_cache_dir: Path,
    warc_filename: str,
    slice_start: int,
    slice_end: int,
    prefix: str,
) -> Optional[bytes]:
    url = api.warc_download_url(str(warc_filename), prefix=str(prefix))
    p = api._cache_path_for_range(
        Path(range_cache_dir),
        url=str(url),
        start=int(slice_start),
        end_inclusive=int(slice_end),
    )
    want = int(slice_end) - int(slice_start) + 1
    try:
        if p.exists() and p.is_file() and p.stat().st_size == want:
            return p.read_bytes()
    except Exception:
        return None
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract HTTP/page payloads from cached Common Crawl slice blobs")
    ap.add_argument("--cache-root", type=Path, default=None, help="Cache root (default: datasets/CCINDEX_WARC_CACHE_DIR)")
    ap.add_argument("--run-id", type=str, default=None, help="Run ID under <cache-root>/slice_indexes (default: LATEST.txt)")
    ap.add_argument("--prefix", type=str, default="https://data.commoncrawl.org/")
    ap.add_argument("--workers", type=int, default=16, help="Parallelism across slices")

    ap.add_argument(
        "--pointers-jsonl",
        type=Path,
        default=None,
        help="Pointer file (.jsonl or .parquet) (default: pointers.parquet if present else pointers.jsonl)",
    )
    ap.add_argument(
        "--slice-members-jsonl",
        type=Path,
        default=None,
        help="Slice member mapping JSONL (default: <cache-root>/slice_indexes/<run-id>/slice_members.jsonl)",
    )
    ap.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Output JSONL (default: <cache-root>/slice_indexes/<run-id>/http_from_cache.jsonl)",
    )

    ap.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="If a slice is missing in cache, fetch it (and populate cache) instead of skipping.",
    )
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--max-body-bytes", type=int, default=2_000_000)
    ap.add_argument("--max-preview-chars", type=int, default=80_000)
    ap.add_argument("--include-body-base64", action="store_true", default=False)
    ap.add_argument("--progress-every-slices", type=int, default=100)

    args = ap.parse_args(list(argv) if argv is not None else None)

    cache_root = (Path(args.cache_root).expanduser().resolve() if args.cache_root is not None else _default_cache_root())
    run_id = _infer_run_id(cache_root, args.run_id)
    run_dir = (cache_root / "slice_indexes" / run_id).resolve()
    range_dir = (cache_root / "ranges").resolve()

    if args.pointers_jsonl is not None:
        pointers_path = Path(args.pointers_jsonl).expanduser().resolve()
    else:
        p_parq = (run_dir / "pointers.parquet").resolve()
        pointers_path = p_parq if p_parq.exists() else (run_dir / "pointers.jsonl").resolve()
    members_jsonl = (
        Path(args.slice_members_jsonl).expanduser().resolve()
        if args.slice_members_jsonl is not None
        else (run_dir / "slice_members.jsonl").resolve()
    )
    out_jsonl = (
        Path(args.out_jsonl).expanduser().resolve()
        if args.out_jsonl is not None
        else (run_dir / "http_from_cache.jsonl").resolve()
    )

    if not pointers_path.exists():
        raise SystemExit(f"Pointers file not found: {pointers_path}")
    if not members_jsonl.exists():
        raise SystemExit(f"Slice members JSONL not found: {members_jsonl}")
    range_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("", encoding="utf-8")

    sys.stderr.write(f"run_id={run_id} run_dir={run_dir} ranges_dir={range_dir}\n")

    pointer_meta = _load_pointer_meta(pointers_path)
    by_slice, member_rows = _group_members(members_jsonl)

    slice_keys = sorted(by_slice.keys(), key=lambda k: (k.warc_filename, k.slice_start, k.slice_end))
    sys.stderr.write(
        f"slices={len(slice_keys)} members={member_rows} pointers_meta={len(pointer_meta)} out={out_jsonl} allow_network={int(bool(args.allow_network))}\n"
    )

    started = time.time()
    extracted = 0
    missing_slices = 0
    missing_members = 0
    wrote = 0

    out_lock = __import__("threading").Lock()

    def _write_rows(rows: List[Dict[str, object]]) -> None:
        nonlocal wrote
        if not rows:
            return
        with out_lock:
            with out_jsonl.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    wrote += 1

    def _process_slice(sk: SliceKey) -> Tuple[int, int, int]:
        """Return (extracted_members, missing_members, missing_slice_flag)."""

        blob = _read_cached_slice(
            range_cache_dir=range_dir,
            warc_filename=sk.warc_filename,
            slice_start=sk.slice_start,
            slice_end=sk.slice_end,
            prefix=str(args.prefix),
        )
        if blob is None and bool(args.allow_network):
            url = api.warc_download_url(str(sk.warc_filename), prefix=str(args.prefix))
            status, blob2, err = api._http_range_get_cached(
                url=str(url),
                start=int(sk.slice_start),
                end_inclusive=int(sk.slice_end),
                timeout_s=float(args.timeout_s),
                cache_dir=Path(range_dir),
                cache_max_bytes=2_000_000_000,
                cache_max_item_bytes=max(1, int(sk.slice_end) - int(sk.slice_start) + 1),
            )
            if blob2 is not None and err is None:
                blob = blob2

        if blob is None:
            return 0, len(by_slice.get(sk) or []), 1

        rows: List[Dict[str, object]] = []
        ok_count = 0
        miss_count = 0
        for pk in by_slice.get(sk) or []:
            rel = int(pk.warc_offset) - int(sk.slice_start)
            if rel < 0 or rel + int(pk.warc_length) > len(blob):
                miss_count += 1
                continue
            member = blob[rel : rel + int(pk.warc_length)]
            meta = pointer_meta.get(pk) or {}

            parsed = api.extract_http_from_warc_gzip_member(
                member,
                max_body_bytes=int(args.max_body_bytes),
                max_preview_chars=int(args.max_preview_chars),
                include_body_base64=bool(args.include_body_base64),
            )

            rows.append(
                {
                    "ok": bool(parsed.ok),
                    "domain": meta.get("domain"),
                    "url": meta.get("url"),
                    "collection": meta.get("collection"),
                    "timestamp": meta.get("timestamp"),
                    "mime": meta.get("mime"),
                    "status": meta.get("status"),
                    "warc_filename": pk.warc_filename,
                    "warc_offset": int(pk.warc_offset),
                    "warc_length": int(pk.warc_length),
                    "slice_start": int(sk.slice_start),
                    "slice_end": int(sk.slice_end),
                    "http": {
                        "ok": bool(parsed.ok),
                        "status": parsed.http_status,
                        "status_line": parsed.http_status_line,
                        "headers": parsed.http_headers,
                        "warc_headers": parsed.warc_headers,
                        "body_text_preview": parsed.body_text_preview,
                        "body_is_html": parsed.body_is_html,
                        "body_mime": parsed.body_mime,
                        "body_charset": parsed.body_charset,
                        "body_base64": parsed.body_base64,
                        "error": parsed.error,
                    },
                }
            )
            ok_count += 1

        _write_rows(rows)
        return ok_count, miss_count, 0

    workers = max(1, int(args.workers or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_slice, sk): sk for sk in slice_keys}
        for i, fut in enumerate(as_completed(futs), start=1):
            okc, missc, miss_slice = fut.result()
            extracted += int(okc)
            missing_members += int(missc)
            missing_slices += int(miss_slice)

            if int(args.progress_every_slices) > 0 and i % int(args.progress_every_slices) == 0:
                dt = max(0.001, time.time() - started)
                sys.stderr.write(
                    f"progress slices={i}/{len(slice_keys)} extracted={extracted} missing_slices={missing_slices} "
                    f"missing_members={missing_members} wrote={wrote} elapsed_s={dt:.1f}\n"
                )

    dt = max(0.001, time.time() - started)
    sys.stderr.write(
        f"ok=1 extracted={extracted} wrote={wrote} slices={len(slice_keys)} missing_slices={missing_slices} "
        f"missing_members={missing_members} elapsed_s={dt:.1f}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
