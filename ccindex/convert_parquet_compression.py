#!/usr/bin/env python3
"""Convert Parquet compression in-place (e.g., snappy -> zstd).

Defaults to zstd and skips files already compressed with the target codec.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq


def _detect_codecs(pq_path: Path) -> set[str]:
    pf = pq.ParquetFile(str(pq_path))
    if pf.metadata is None or pf.metadata.num_row_groups == 0:
        return set()
    rg = pf.metadata.row_group(0)
    codecs = set()
    for i in range(rg.num_columns):
        codecs.add(str(rg.column(i).compression).upper())
    return codecs


def _rewrite_parquet(pq_path: Path, compression: str, temp_dir_name: str) -> None:
    pf = pq.ParquetFile(str(pq_path))
    temp_dir = pq_path.parent / temp_dir_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = temp_dir / f"{pq_path.name}.recompress"
    writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression=str(compression))
    try:
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            writer.write_table(table)
    finally:
        writer.close()
    try:
        tmp.replace(pq_path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _iter_files(root: Path, glob_pat: str) -> List[Path]:
    return sorted(root.glob(glob_pat))


def _convert_one(
    pq_path_str: str,
    compression: str,
    temp_dir_name: str,
    dry_run: bool,
) -> Tuple[str, str, List[str], Optional[str], int, int]:
    pq_path = Path(pq_path_str)
    temp_dir_name = str(temp_dir_name).strip() or ".tmp_recompress"
    try:
        codecs = _detect_codecs(pq_path)
    except Exception as exc:
        return ("error", pq_path_str, [], str(exc), 0, 0)

    target = str(compression).upper()
    if codecs == {target}:
        return ("skipped", pq_path_str, sorted(codecs), None, 0, 0)

    if dry_run:
        return ("would_convert", pq_path_str, sorted(codecs), None, 0, 0)

    before = pq_path.stat().st_size if pq_path.exists() else 0
    _rewrite_parquet(pq_path, compression, temp_dir_name)
    after = pq_path.stat().st_size if pq_path.exists() else 0
    return ("converted", pq_path_str, sorted(codecs), None, int(before), int(after))


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompress parquet files")
    ap.add_argument(
        "--root",
        default="/storage/ccindex_parquet/cc_pointers_by_collection",
        help="Root directory to scan",
    )
    ap.add_argument(
        "--glob",
        default="**/cdx-*.parquet",
        help="Glob pattern under root",
    )
    ap.add_argument(
        "--compression",
        default="zstd",
        help="Target compression codec",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only report changes")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of files")
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers")
    ap.add_argument(
        "--temp-dir-name",
        default=".tmp_recompress",
        help="Temp directory name created under each target folder",
    )
    ap.add_argument(
        "--progress-secs",
        type=int,
        default=60,
        help="Heartbeat interval in seconds",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root).expanduser().resolve()
    files = _iter_files(root, str(args.glob))
    if args.max_files:
        files = files[: max(1, int(args.max_files))]

    if not files:
        print("No parquet files found.")
        return 0

    target = str(args.compression).upper()
    converted = 0
    skipped = 0
    errors = 0
    last_log = time.time()
    start_time = last_log

    def _emit_progress(done: int) -> None:
        nonlocal last_log
        now = time.time()
        if (now - last_log) >= int(args.progress_secs):
            elapsed = max(1.0, now - start_time)
            rate = done / elapsed
            remaining = max(0, len(files) - done)
            eta_s = int(remaining / rate) if rate > 0 else -1
            eta_str = f"{eta_s}s" if eta_s >= 0 else "unknown"
            print(
                f"progress done={done}/{len(files)} converted={converted} skipped={skipped} "
                f"errors={errors} rate={rate:.2f}/s eta={eta_str}",
                flush=True,
            )
            last_log = now

    workers = max(1, int(args.workers))
    temp_dir_name = str(args.temp_dir_name).strip() or ".tmp_recompress"

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                _convert_one,
                str(pq_path),
                str(args.compression),
                temp_dir_name,
                bool(args.dry_run),
            ): str(pq_path)
            for pq_path in files
        }
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                status, path, codecs, err, before, after = fut.result()
            except Exception as exc:
                errors += 1
                print(f"error converting {futs.get(fut, '')}: {exc}")
                _emit_progress(done)
                continue

            if status == "skipped":
                skipped += 1
            elif status == "converted":
                converted += 1
                print(
                    f"convert {path} codecs={codecs} -> {target} size_mb={before/1e6:.1f}->{after/1e6:.1f}",
                    flush=True,
                )
            elif status == "would_convert":
                converted += 1
                print(f"would_convert {path} codecs={codecs} -> {target}", flush=True)
            else:
                errors += 1
                print(f"error converting {path}: {err}")

            _emit_progress(done)

    print(f"done converted={converted} skipped={skipped} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
