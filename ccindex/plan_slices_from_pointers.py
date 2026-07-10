#!/usr/bin/env python3
"""Compile a canonical slice plan from ccindex pointer JSONL.

Input: JSONL where each line is a dict containing at least:
  - warc_filename
  - warc_offset
  - warc_length

Output:
  - slice_plan.jsonl: one line per slice with start/end/len + member_count
  - slice_members.jsonl: one line per pointer member mapping to its slice

This is intentionally "canonical": exact duplicate (warc_offset, warc_length)
within the same warc_filename are de-duplicated before slicing.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    want_cols = ["warc_filename", "warc_offset", "warc_length"]
    for batch in pf.iter_batches(batch_size=65_536, columns=want_cols, use_threads=True):
        # Convert in a columnar way to reduce per-row overhead.
        cols = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        wf = cols.get("warc_filename") or []
        off = cols.get("warc_offset") or []
        ln = cols.get("warc_length") or []
        n = min(len(wf), len(off), len(ln))
        for i in range(n):
            yield {"warc_filename": wf[i], "warc_offset": off[i], "warc_length": ln[i]}


def _iter_pointers(path: Path) -> Iterator[Dict[str, object]]:
    if path.suffix.lower() == ".parquet":
        yield from _iter_parquet(path)
        return
    yield from _iter_jsonl(path)


def _expand_slices_to_min_size(
    slices: List[Tuple[int, int, List[Tuple[int, int]]]],
    *,
    min_slice_bytes: int,
    max_slice_bytes: int,
) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    min_sb = int(min_slice_bytes or 0)
    if min_sb <= 0:
        return slices
    out: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for s0, s1, members in slices:
        cur_len = int(s1) - int(s0) + 1
        if cur_len >= min_sb:
            out.append((int(s0), int(s1), members))
            continue
        new_s1 = int(s0) + int(min_sb) - 1
        if int(max_slice_bytes) > 0:
            new_s1 = min(int(new_s1), int(s0) + int(max_slice_bytes) - 1)
        out.append((int(s0), int(new_s1), members))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a canonical WARC slice plan from pointer JSONL/Parquet")
    ap.add_argument("--cache-root", type=Path, default=None, help="Cache root (default: datasets/CCINDEX_WARC_CACHE_DIR)")
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for slice index outputs (default: infer from pointers path; else LATEST.txt; else timestamp)",
    )
    ap.add_argument(
        "--pointers-jsonl",
        type=Path,
        required=True,
        help="Pointer file (.jsonl or .parquet). Must contain warc_filename, warc_offset, warc_length.",
    )
    ap.add_argument("--out-slice-plan-jsonl", type=Path, default=None, help="Output slice plan JSONL")
    ap.add_argument("--out-slice-members-jsonl", type=Path, default=None, help="Output slice members JSONL")
    ap.add_argument("--max-slice-bytes", type=int, default=64_000_000)
    ap.add_argument("--max-gap-bytes", type=int, default=1_000_000)
    ap.add_argument("--min-slice-bytes", type=int, default=1_000_000)
    ap.add_argument("--progress-every-warcs", type=int, default=250)

    args = ap.parse_args(list(argv) if argv is not None else None)

    cache_root = (
        Path(args.cache_root).expanduser().resolve()
        if args.cache_root is not None
        else (Path("datasets") / "CCINDEX_WARC_CACHE_DIR").resolve()
    )

    pointers_path = Path(args.pointers_jsonl).expanduser().resolve()
    if not pointers_path.exists():
        raise SystemExit(f"Pointers file not found: {pointers_path}")

    # Determine run_id.
    run_id = str(args.run_id).strip() if args.run_id is not None else ""
    if not run_id:
        try:
            # If pointers are under <cache_root>/slice_indexes/<run-id>/pointers.*, infer <run-id>.
            rel = pointers_path.relative_to(cache_root)
            parts = list(rel.parts)
            if len(parts) >= 3 and parts[0] == "slice_indexes":
                run_id = str(parts[1])
        except Exception:
            run_id = ""
    if not run_id:
        try:
            latest = (cache_root / "slice_indexes" / "LATEST.txt").read_text(encoding="utf-8").strip()
            if latest:
                run_id = latest
        except Exception:
            run_id = ""
    if not run_id:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    run_dir = (cache_root / "slice_indexes" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    out_plan = (
        Path(args.out_slice_plan_jsonl).expanduser().resolve()
        if args.out_slice_plan_jsonl is not None
        else (run_dir / "slice_plan.jsonl").resolve()
    )
    out_members = (
        Path(args.out_slice_members_jsonl).expanduser().resolve()
        if args.out_slice_members_jsonl is not None
        else (run_dir / "slice_members.jsonl").resolve()
    )
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    out_members.parent.mkdir(parents=True, exist_ok=True)
    out_plan.write_text("", encoding="utf-8")
    out_members.write_text("", encoding="utf-8")

    try:
        (cache_root / "slice_indexes" / "LATEST.txt").write_text(str(run_id) + "\n", encoding="utf-8")
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "stage": "slices-from-pointers",
                    "cache_root": str(cache_root),
                    "run_id": str(run_id),
                    "pointers_path": str(pointers_path),
                    "slice_plan_jsonl": str(out_plan),
                    "slice_members_jsonl": str(out_members),
                    "max_slice_bytes": int(args.max_slice_bytes),
                    "max_gap_bytes": int(args.max_gap_bytes),
                    "min_slice_bytes": int(args.min_slice_bytes),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    # Group into memory by warc_filename. This can be large; if this becomes a bottleneck,
    # we can move to a sort-based external grouping.
    by_warc: Dict[str, set[Tuple[int, int]]] = {}
    total_ptr = 0
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
        s = by_warc.get(wf)
        if s is None:
            s = set()
            by_warc[wf] = s
        s.add((int(off), int(ln)))
        total_ptr += 1

    warc_files = sorted(by_warc.keys())
    sys.stderr.write(
        f"warcs={len(warc_files)} pointers_in={total_ptr} out_plan={out_plan} out_members={out_members} "
        f"run_dir={run_dir} ranges_dir={cache_root / 'ranges'}\n"
    )

    plan_f = out_plan.open("a", encoding="utf-8")
    mem_f = out_members.open("a", encoding="utf-8")
    try:
        slice_count = 0
        member_count = 0
        for i, wf in enumerate(warc_files, start=1):
            ranges = sorted(by_warc.get(wf) or set(), key=lambda t: t[0])
            if not ranges:
                continue

            slices = api._merge_ranges_into_slices(
                list(ranges),
                max_slice_bytes=int(args.max_slice_bytes),
                max_gap_bytes=int(args.max_gap_bytes),
            )
            slices = _expand_slices_to_min_size(
                slices,
                min_slice_bytes=int(args.min_slice_bytes),
                max_slice_bytes=int(args.max_slice_bytes),
            )

            for s0, s1, members in slices:
                slen = int(s1) - int(s0) + 1
                plan_f.write(
                    json.dumps(
                        {
                            "warc_filename": wf,
                            "slice_start": int(s0),
                            "slice_end": int(s1),
                            "slice_len": int(slen),
                            "member_count": int(len(members)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                slice_count += 1

                for off, ln in members:
                    mem_f.write(
                        json.dumps(
                            {
                                "warc_filename": wf,
                                "slice_start": int(s0),
                                "slice_end": int(s1),
                                "warc_offset": int(off),
                                "warc_length": int(ln),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    member_count += 1

            if int(args.progress_every_warcs) > 0 and i % int(args.progress_every_warcs) == 0:
                sys.stderr.write(f"progress warcs={i}/{len(warc_files)} slices={slice_count} members={member_count}\n")

        sys.stderr.write(f"ok=1 warcs={len(warc_files)} slices={slice_count} members={member_count}\n")
    finally:
        plan_f.close()
        mem_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
