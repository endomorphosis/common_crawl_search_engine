#!/usr/bin/env python3
"""Prefetch a slice plan in parallel (warm the range cache).

Reads slice_plan.jsonl produced by plan_slices_from_pointers.py and issues HTTP
Range GETs for each slice. This is primarily intended to warm the on-disk range
cache so subsequent processing runs are local.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

from common_crawl_search_engine.ccindex import api


def _iter_slice_plan(path: Path) -> Iterator[Tuple[str, int, int]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            wf = str(obj.get("warc_filename") or "").strip()
            if not wf:
                continue
            try:
                s0 = int(obj.get("slice_start"))
                s1 = int(obj.get("slice_end"))
            except Exception:
                continue
            if s0 < 0 or s1 < s0:
                continue
            yield wf, s0, s1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch all slices in a slice_plan.jsonl in parallel")
    ap.add_argument("--cache-root", type=Path, default=None, help="Cache root (default: datasets/CCINDEX_WARC_CACHE_DIR)")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID to locate slice plan under <cache-root>/slice_indexes/<run-id>/ (default: LATEST.txt)",
    )
    ap.add_argument(
        "--slice-plan-jsonl",
        type=Path,
        default=None,
        help="Slice plan JSONL (default: <cache-root>/slice_indexes/<run-id>/slice_plan.jsonl)",
    )
    ap.add_argument("--prefix", type=str, default="https://data.commoncrawl.org/")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Range cache dir (default: <cache-root>/ranges; disable via env CCINDEX_WARC_CACHE_DIR='')",
    )
    ap.add_argument("--cache-max-bytes", type=int, default=2_000_000_000)
    ap.add_argument("--cache-max-item-bytes", type=int, default=128_000_000)
    ap.add_argument("--progress-every", type=int, default=200)

    args = ap.parse_args(list(argv) if argv is not None else None)

    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if args.cache_root is not None
        else (Path("datasets") / "CCINDEX_WARC_CACHE_DIR").resolve()
    )

    run_id = str(args.run_id).strip() if args.run_id is not None else ""
    if not run_id:
        try:
            latest = (cache_root / "slice_indexes" / "LATEST.txt").read_text(encoding="utf-8").strip()
            if latest:
                run_id = latest
        except Exception:
            run_id = ""
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    plan_path = (
        Path(args.slice_plan_jsonl).expanduser().resolve()
        if args.slice_plan_jsonl is not None
        else (cache_root / "slice_indexes" / run_id / "slice_plan.jsonl").resolve()
    )
    if not plan_path.exists():
        raise SystemExit(f"Slice plan not found: {plan_path}")

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir is not None
        else (cache_root / "ranges").resolve()
    )
    slices = list(_iter_slice_plan(plan_path))
    if not slices:
        sys.stderr.write("No slices found in plan\n")
        return 2

    started = time.time()
    total_bytes = 0
    ok = 0
    fail = 0

    def _fetch_one(wf: str, s0: int, s1: int) -> Tuple[bool, int, str]:
        url = api.warc_download_url(str(wf), prefix=str(args.prefix))
        status, blob, err = api._http_range_get_cached(
            url=url,
            start=int(s0),
            end_inclusive=int(s1),
            timeout_s=float(args.timeout_s),
            cache_dir=cache_dir,
            cache_max_bytes=int(args.cache_max_bytes),
            cache_max_item_bytes=int(args.cache_max_item_bytes),
        )
        if blob is not None and err is None:
            return True, len(blob), ""
        return False, 0, (err or f"http_failed status={status}")

    workers = max(1, int(args.workers or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_one, wf, s0, s1) for (wf, s0, s1) in slices]
        for i, fut in enumerate(as_completed(futs), start=1):
            ok_i, nbytes, err = fut.result()
            if ok_i:
                ok += 1
                total_bytes += int(nbytes)
            else:
                fail += 1
                if fail <= 10:
                    sys.stderr.write(f"slice_fetch_error: {err}\n")

            if int(args.progress_every) > 0 and i % int(args.progress_every) == 0:
                dt = max(0.001, time.time() - started)
                sys.stderr.write(
                    f"progress slices={i}/{len(slices)} ok={ok} fail={fail} "
                    f"mb={total_bytes/1e6:.1f} bps={total_bytes/dt:.0f}\n"
                )

    dt = max(0.001, time.time() - started)
    sys.stderr.write(
        f"ok={(1 if fail==0 else 0)} slices={len(slices)} ok_slices={ok} fail_slices={fail} "
        f"mb={total_bytes/1e6:.1f} elapsed_s={dt:.1f} bps={total_bytes/dt:.0f}\n"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
