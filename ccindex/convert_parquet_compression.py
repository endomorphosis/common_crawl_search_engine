#!/usr/bin/env python3
"""Convert Parquet compression in-place (e.g., snappy -> zstd).

Defaults to zstd and skips files already compressed with the target codec.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

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


def _rewrite_parquet(pq_path: Path, compression: str) -> None:
    pf = pq.ParquetFile(str(pq_path))
    tmp = pq_path.with_suffix(pq_path.suffix + ".recompress")
    writer = pq.ParquetWriter(tmp, pf.schema, compression=str(compression))
    try:
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg)
            writer.write_table(table)
    finally:
        writer.close()
    tmp.replace(pq_path)


def _iter_files(root: Path, glob_pat: str) -> List[Path]:
    return sorted(root.glob(glob_pat))


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

    for pq_path in files:
        try:
            codecs = _detect_codecs(pq_path)
        except Exception as exc:
            print(f"error reading {pq_path}: {exc}")
            continue

        if codecs == {target}:
            skipped += 1
            continue

        if args.dry_run:
            print(f"would_convert {pq_path} codecs={sorted(codecs)} -> {target}")
            converted += 1
            continue

        print(f"convert {pq_path} codecs={sorted(codecs)} -> {target}")
        try:
            _rewrite_parquet(pq_path, args.compression)
            converted += 1
        except Exception as exc:
            print(f"error converting {pq_path}: {exc}")

    print(f"done converted={converted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
