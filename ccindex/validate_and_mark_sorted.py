#!/usr/bin/env python3
"""Validate parquet files are sorted and mark them with .sorted extension.

Compatibility tool for older pipeline runs.

This script:
1) Skips files already marked as *.sorted.parquet
2) Validates unmarked parquet files are sorted by host_rev
3) Marks sorted files by renaming to *.sorted.parquet
4) Optionally sorts unsorted files (DuckDB external sort) and marks them

Notes:
- Sorting uses `ORDER BY host_rev, url, ts`.
- This script intentionally ignores parquet files under hidden/temp directories
  (e.g. `.duckdb_sort_tmp`, `.cc_sort_work_*`) so partial/resume runs don't
  accidentally treat scratch artifacts as inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import multiprocessing
import os
import statistics
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def _parquet_has_columns(parquet_file: Path, required: set[str]) -> bool:
    if not required:
        return True
    try:
        pf = pq.ParquetFile(parquet_file)
        names = set(pf.schema_arrow.names)
        return required.issubset(names)
    except Exception:
        return False


def _parquet_all_zstd(parquet_file: Path) -> bool:
    """Return True if the parquet appears to use ZSTD compression.

    We check row group 0 metadata (fast) and require all columns to report ZSTD.
    """

    try:
        pf = pq.ParquetFile(parquet_file)
        md = pf.metadata
        if md is None or md.num_row_groups == 0:
            return True
        rg0 = md.row_group(0)
        for i in range(rg0.num_columns):
            if str(rg0.column(i).compression).upper() != "ZSTD":
                return False
        return True
    except Exception:
        return False


def _row_group_compressed_bytes(md: pq.FileMetaData, rg_idx: int) -> int:
    rg = md.row_group(rg_idx)
    total = 0
    for i in range(rg.num_columns):
        col = rg.column(i)
        # Prefer compressed size; fall back to total byte size if missing.
        try:
            v = int(getattr(col, "total_compressed_size"))
        except Exception:
            v = 0
        if v <= 0:
            try:
                v = int(getattr(col, "total_uncompressed_size"))
            except Exception:
                v = 0
        if v <= 0:
            try:
                v = int(getattr(rg, "total_byte_size"))
            except Exception:
                v = 0
        total += max(0, int(v))
    return int(total)


def _parquet_rowgroups_approx_target_mb(
    parquet_file: Path,
    *,
    target_mb: int,
    tolerance_ratio: float,
    sample_rowgroups: int = 10,
) -> bool:
    """Return True if the shard's row groups are already near the target size.

    Uses median compressed bytes over a small sample of row groups.
    """

    try:
        pf = pq.ParquetFile(parquet_file)
        md = pf.metadata
        if md is None or md.num_row_groups == 0:
            return True

        n = int(md.num_row_groups)
        k = max(1, min(int(sample_rowgroups), n))
        # Evenly sample row groups.
        if k == 1:
            idxs = [n // 2]
        else:
            idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]

        sizes = [_row_group_compressed_bytes(md, i) for i in idxs]
        sizes = [s for s in sizes if s > 0]
        if not sizes:
            return False

        med = float(statistics.median(sizes))
        target = float(max(1, int(target_mb))) * 1024.0 * 1024.0
        tol = max(0.0, float(tolerance_ratio))
        lo = target * (1.0 - tol)
        hi = target * (1.0 + tol)
        return lo <= med <= hi
    except Exception:
        return False


def _parquet_rowgroups_median_rows(parquet_file: Path, *, sample_rowgroups: int = 10) -> Optional[float]:
    """Return median row-group row count for a shard (metadata-only, sampled)."""

    try:
        pf = pq.ParquetFile(parquet_file)
        md = pf.metadata
        if md is None or md.num_row_groups == 0:
            return None

        n = int(md.num_row_groups)
        k = max(1, min(int(sample_rowgroups), n))
        if k == 1:
            idxs = [n // 2]
        else:
            idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]

        rows = [int(md.row_group(i).num_rows) for i in idxs]
        rows = [r for r in rows if r > 0]
        if not rows:
            return None
        return float(statistics.median(rows))
    except Exception:
        return None


def _parquet_rowgroups_approx_target_rows(
    parquet_file: Path,
    *,
    target_rows: int,
    tolerance_ratio: float,
    sample_rowgroups: int = 10,
) -> bool:
    """Return True if the shard's row groups are already near the target row count."""

    med = _parquet_rowgroups_median_rows(parquet_file, sample_rowgroups=sample_rowgroups)
    if med is None:
        return False

    target = float(max(1, int(target_rows)))
    tol = max(0.0, float(tolerance_ratio))
    lo = target * (1.0 - tol)
    hi = target * (1.0 + tol)
    return lo <= med <= hi


def _parquet_needs_rewrite(
    parquet_file: Path,
    *,
    target_mb: Optional[int],
    tolerance_ratio: float,
    required_columns: set[str],
) -> bool:
    """Return True if we should rewrite the already-sorted shard."""

    # NOTE: rewriting cannot add missing columns (it only copies the existing
    # schema). If required columns are missing, the correct action is to repair
    # the parquet schema upstream (e.g., repair_legacy_parquet_columns.py) and
    # then retry. Treat as "do not rewrite" here.
    if required_columns and not _parquet_has_columns(parquet_file, required_columns):
        return False
    # Non-ZSTD => rewrite.
    if not _parquet_all_zstd(parquet_file):
        return True
    # Row groups not near target => rewrite.
    if target_mb is not None:
        if not _parquet_rowgroups_approx_target_mb(
            parquet_file,
            target_mb=int(target_mb),
            tolerance_ratio=float(tolerance_ratio),
        ):
            return True
    return False


def _parquet_rewrite_reasons(
    parquet_file: Path,
    *,
    target_rows: Optional[int],
    target_mb: Optional[int],
    tolerance_ratio: float,
    required_columns: set[str],
) -> Tuple[bool, List[str]]:
    """Return (needs_rewrite, reasons).

    Reasons are stable strings intended for logging.
    """

    # If required columns are missing, we cannot fix that via rewrite (it only copies
    # the existing schema). Report it for logging but treat it as a skip.
    if required_columns and not _parquet_has_columns(parquet_file, required_columns):
        return False, ["missing_required_columns"]

    reasons: List[str] = []
    if not _parquet_all_zstd(parquet_file):
        reasons.append("non_zstd")

    # Prefer a direct check against the target row group row count when we know it.
    if target_rows is not None:
        if not _parquet_rowgroups_approx_target_rows(
            parquet_file,
            target_rows=int(target_rows),
            tolerance_ratio=float(tolerance_ratio),
        ):
            reasons.append("rowgroup_rows_off")
    elif target_mb is not None:
        if not _parquet_rowgroups_approx_target_mb(
            parquet_file,
            target_mb=int(target_mb),
            tolerance_ratio=float(tolerance_ratio),
        ):
            reasons.append("rowgroup_mb_off")

    return bool(reasons), reasons


def _is_hidden_path(parquet_root: Path, p: Path) -> bool:
    """Return True if the file is under a hidden directory relative to parquet_root."""

    try:
        rel = p.relative_to(parquet_root)
    except Exception:
        return False
    # Ignore hidden directories (not the file name itself).
    return any(part.startswith(".") for part in rel.parts[:-1])


def _iter_candidate_parquet_files(parquet_root: Path) -> List[Path]:
    """Find parquet files, skipping hidden/temp artifacts."""

    files: List[Path] = []
    for p in parquet_root.rglob("*.parquet"):
        try:
            if not p.is_file():
                continue
            if _is_hidden_path(parquet_root, p):
                continue
            # Skip obvious temp outputs.
            if p.name.endswith(".tmp.parquet") or p.name.endswith(".sorted.tmp"):
                continue
            files.append(p)
        except Exception:
            continue
    return sorted(files)


def is_sorted_by_content(parquet_file: Path, sample_size: int = 1000) -> Tuple[bool, str]:
    """Check if a parquet file is sorted by host_rev.

    Returns: (is_sorted, reason)
    """

    try:
        pf = pq.ParquetFile(parquet_file)

        if pf.metadata is None:
            return False, "Missing parquet metadata"

        # An empty parquet file (valid schema but 0 row groups) is trivially sorted.
        if pf.metadata.num_row_groups == 0:
            return True, "Empty parquet (no row groups)"

        # Check within first row group
        table = pf.read_row_group(0, columns=["host_rev"])
        vals = table["host_rev"].to_pylist()

        if len(vals) < 2:
            return True, "Too few rows to check"

        # Sample check within row group
        step = max(1, len(vals) // sample_size)
        sample = vals[::step]

        for i in range(len(sample) - 1):
            if sample[i] > sample[i + 1]:
                return False, f"Unsorted within row group: {sample[i]} > {sample[i+1]}"

        # Check across row groups if multiple exist
        if pf.metadata.num_row_groups > 1:
            last_val = vals[-1]

            for rg_idx in range(1, min(pf.metadata.num_row_groups, 10)):
                table = pf.read_row_group(rg_idx, columns=["host_rev"])
                vals = table["host_rev"].to_pylist()

                if len(vals) > 0:
                    first_val = vals[0]
                    if last_val > first_val:
                        return False, f"Unsorted between row groups: {last_val} > {first_val}"

                    last_val = vals[-1]

        return True, "Verified sorted"

    except Exception as e:
        return False, f"Error: {e}"


def sort_parquet_file(
    input_file: Path,
    output_file: Path,
    memory_limit_gb: float = 4.0,
    temp_directory: Optional[Path] = None,
    row_group_size: Optional[int] = None,
) -> Tuple[bool, str]:
    """Sort a parquet file by host_rev, url, ts using DuckDB."""

    try:
        con = duckdb.connect(":memory:")
        con.execute(f"SET memory_limit='{memory_limit_gb}GB'")
        # Reduce memory pressure for large sorts.
        con.execute("SET preserve_insertion_order=false")
        # Isolate DuckDB temp usage per-sort to avoid contention.
        td = temp_directory if temp_directory else output_file.parent
        con.execute(f"SET temp_directory='{td}'")
        con.execute("PRAGMA threads=1")

        # DuckDB parameter binding inside COPY/TO can be surprising; use escaped literals.
        in_path = str(input_file).replace("'", "''")
        out_path = str(output_file).replace("'", "''")
        copy_opts = ["FORMAT 'parquet'", "COMPRESSION 'zstd'"]
        if row_group_size is not None:
            try:
                rgs = int(row_group_size)
                if rgs > 0:
                    copy_opts.append(f"ROW_GROUP_SIZE {rgs}")
            except Exception:
                pass
        opt_sql = ", ".join(copy_opts)

        con.execute(
            """
            COPY (
                SELECT * FROM read_parquet('{in_path}')
                ORDER BY host_rev, url, ts
            )
            TO '{out_path}' ({opt_sql})
            """.format(in_path=in_path, out_path=out_path, opt_sql=opt_sql)
        )
        con.close()
        return True, ""
    except Exception as e:
        msg = str(e)
        print(f"❌ Error sorting {input_file.name}: {msg}", file=sys.stderr)
        return False, msg


def _row_group_sizes_to_try(row_group_size: Optional[int]) -> List[Optional[int]]:
    """Generate a small set of fallback row_group_size values.

    We only vary row_group_size to reduce memory pressure in DuckDB's parquet
    writer. This can help when rewriting very wide shards.
    """

    if row_group_size is None:
        return [None]
    try:
        rgs = int(row_group_size)
    except Exception:
        return [None]

    if rgs <= 0:
        return [None]

    sizes: List[Optional[int]] = []
    cur = rgs
    min_rgs = 5_000
    while True:
        if cur not in sizes:
            sizes.append(cur)
        if cur <= min_rgs:
            break
        cur = max(min_rgs, cur // 2)
    return sizes


def rewrite_sorted_parquet_file(
    input_file: Path,
    output_file: Path,
    memory_limit_gb: float = 4.0,
    temp_directory: Optional[Path] = None,
    row_group_size: Optional[int] = None,
) -> Tuple[bool, str]:
    """Rewrite an already-sorted parquet file without re-sorting.

    This is used to apply row-group sizing / compression tweaks while keeping the
    existing physical ordering. It intentionally avoids ORDER BY, which can be
    expensive and may OOM on large shards.
    """

    def _streaming_rewrite() -> Tuple[bool, str]:
        """Fallback rewrite using PyArrow streaming to minimize peak memory.

        Preserves record order (no sort), applies ZSTD, and attempts to honor
        row_group_size by using it as the batch size.
        """

        try:
            pf = pq.ParquetFile(input_file)
            schema = pf.schema_arrow
            batch_size = int(row_group_size) if row_group_size is not None and int(row_group_size) > 0 else 65_536
            writer = pq.ParquetWriter(str(output_file), schema=schema, compression="zstd")
            try:
                for batch in pf.iter_batches(batch_size=batch_size):
                    # Write each batch as its own row group.
                    tbl = pa.Table.from_batches([batch])
                    writer.write_table(tbl)
            finally:
                writer.close()
            return True, ""
        except Exception as e:
            return False, str(e)

    try:
        con = duckdb.connect(":memory:")
        con.execute(f"SET memory_limit='{memory_limit_gb}GB'")
        # Preserve scan order; we rely on the file already being sorted.
        con.execute("SET preserve_insertion_order=true")
        td = temp_directory if temp_directory else output_file.parent
        con.execute(f"SET temp_directory='{td}'")
        # Keep deterministic + low-memory.
        con.execute("PRAGMA threads=1")

        in_path = str(input_file).replace("'", "''")
        out_path = str(output_file).replace("'", "''")
        copy_opts = ["FORMAT 'parquet'", "COMPRESSION 'zstd'"]
        if row_group_size is not None:
            try:
                rgs = int(row_group_size)
                if rgs > 0:
                    copy_opts.append(f"ROW_GROUP_SIZE {rgs}")
            except Exception:
                pass
        opt_sql = ", ".join(copy_opts)

        con.execute(
            """
            COPY (
                SELECT * FROM read_parquet('{in_path}')
            )
            TO '{out_path}' ({opt_sql})
            """.format(in_path=in_path, out_path=out_path, opt_sql=opt_sql)
        )
        con.close()
        return True, ""
    except Exception as e:
        msg = str(e)
        # If DuckDB can't rewrite within the memory limit, fall back to a
        # streaming rewrite that does not require loading the whole shard.
        if "Out of Memory" in msg or "out of memory" in msg:
            ok2, msg2 = _streaming_rewrite()
            if ok2:
                return True, ""
            msg = f"duckdb OOM: {msg}; pyarrow streaming rewrite failed: {msg2}"

        print(f"❌ Error rewriting {input_file.name}: {msg}", file=sys.stderr)
        return False, msg


def check_single_file(pq_file: Path, parquet_root: Path, verify_only: bool) -> Tuple[str, Path, bool, str]:
    """Check a single parquet file and optionally mark it as sorted.

    Returns: (status, file_path, is_sorted, reason)
    status: 'already_marked', 'sorted_unmarked', 'unsorted', 'error'
    """

    # Skip already marked files
    if ".sorted." in pq_file.name or pq_file.name.endswith(".sorted.parquet"):
        return ("already_marked", pq_file, True, "Already marked")

    try:
        is_sorted, reason = is_sorted_by_content(pq_file)

        if is_sorted:
            if verify_only:
                return ("sorted_unmarked", pq_file, True, "Sorted but not marked (verify-only)")

            if pq_file.name.endswith(".gz.parquet"):
                new_name = pq_file.name.replace(".gz.parquet", ".gz.sorted.parquet")
            else:
                new_name = pq_file.name.replace(".parquet", ".sorted.parquet")

            new_path = pq_file.parent / new_name

            # If a marked file already exists, treat the unmarked one as a duplicate.
            if new_path.exists():
                try:
                    pq_file.unlink()
                except Exception:
                    pass
                return ("sorted_unmarked", new_path, True, f"Marked already existed; removed duplicate")

            pq_file.rename(new_path)
            return ("sorted_unmarked", new_path, True, f"Marked as {new_name}")

        return ("unsorted", pq_file, False, reason)

    except Exception as e:
        return ("error", pq_file, False, str(e))


def sort_and_mark_one(args: Tuple[str, float, str, Optional[int], str]) -> Tuple[str, bool, str, str]:
    """Sort or rewrite one parquet file.

    Args tuple: (src_path, memory_per_sort_gb, temp_root, row_group_size, mode)
        - mode='sort': sort unsorted file and write a new *.sorted.parquet next to it
        - mode='rewrite': rewrite an already-sorted *.sorted.parquet in-place (keeps name)

    Returns: (source_path, success, message, output_path)
    """

    src_path, memory_per_sort_gb, temp_root, row_group_size, mode = args
    src = Path(src_path)
    tmp_root = Path(temp_root)
    work_dir: Optional[Path] = None
    duckdb_temp_dir: Optional[Path] = None

    try:
        # Write tmp output in destination directory so the final rename is atomic.
        safe = src.name.replace(os.sep, "_")
        work_dir = src.parent / f".cc_sort_work_{safe}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # DuckDB spill temp directory MUST be unique per sort when running in parallel.
        # Include pid + random suffix so duplicate submissions (or stale workers)
        # can't contend on the same directory.
        unique_tag = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        duckdb_temp_dir = tmp_root / f"duckdb_sort_{safe}_{unique_tag}"
        duckdb_temp_dir.mkdir(parents=True, exist_ok=True)

        # Output temp parquet must also be unique per attempt; DuckDB will take an OS
        # lock on the output path. If the same shard is submitted twice (rare but has
        # happened in production runs), a fixed tmp name causes hard failures.
        sorted_tmp = work_dir / f"{src.name}.{unique_tag}.tmp.parquet"

        if str(mode) == "rewrite":
            if not src.name.endswith(".sorted.parquet"):
                return str(src), False, "rewrite requested but source is not *.sorted.parquet", ""

            # Some older runs have incorrectly-marked *.sorted.parquet shards.
            # If the source isn't actually sorted, do a full sort-in-place.
            src_sorted, src_reason = is_sorted_by_content(src)
            if not src_sorted:
                sort_ok = False
                last_err = ""
                for rgs_try in _row_group_sizes_to_try(row_group_size):
                    ok, err = sort_parquet_file(
                        src,
                        sorted_tmp,
                        memory_per_sort_gb,
                        temp_directory=duckdb_temp_dir,
                        row_group_size=rgs_try,
                    )
                    if ok:
                        sort_ok = True
                        break
                    last_err = err
                if not sort_ok:
                    return str(src), False, f"source not sorted ({src_reason}); sort failed: {last_err}", ""

                ok, reason = is_sorted_by_content(sorted_tmp)
                if not ok:
                    try:
                        sorted_tmp.unlink()
                    except Exception:
                        pass
                    return str(src), False, f"sort verification failed: {reason}", ""

                sorted_tmp.replace(src)
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass
                return str(src), True, f"sorted (was mis-marked; {src_reason})", str(src)

            rewrite_ok = False
            last_err = ""
            for rgs_try in _row_group_sizes_to_try(row_group_size):
                ok, err = rewrite_sorted_parquet_file(
                    src,
                    sorted_tmp,
                    memory_per_sort_gb,
                    temp_directory=duckdb_temp_dir,
                    row_group_size=rgs_try,
                )
                if ok:
                    rewrite_ok = True
                    break
                last_err = err
                if "Out of Memory" not in (err or ""):
                    break

            if not rewrite_ok:
                return str(src), False, f"rewrite failed: {last_err}", ""

            ok, reason = is_sorted_by_content(sorted_tmp)
            if not ok:
                # If the output isn't sorted, fall back to a full sort. This can
                # happen when the input was incorrectly marked as sorted.
                try:
                    sorted_tmp.unlink()
                except Exception:
                    pass

                sort_ok = False
                last_err = ""
                for rgs_try in _row_group_sizes_to_try(row_group_size):
                    ok2, err2 = sort_parquet_file(
                        src,
                        sorted_tmp,
                        memory_per_sort_gb,
                        temp_directory=duckdb_temp_dir,
                        row_group_size=rgs_try,
                    )
                    if ok2:
                        sort_ok = True
                        break
                    last_err = err2
                    if "Out of Memory" not in (err2 or ""):
                        break

                if not sort_ok:
                    return str(src), False, f"rewrite verification failed: {reason}; sort fallback failed: {last_err}", ""

                ok3, reason3 = is_sorted_by_content(sorted_tmp)
                if not ok3:
                    try:
                        sorted_tmp.unlink()
                    except Exception:
                        pass
                    return str(src), False, f"sort fallback verification failed: {reason3}", ""

            sorted_tmp.replace(src)
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
            return str(src), True, "rewritten", str(src)

        # Default: sort an unsorted file and produce a sorted sibling.
        sort_ok = False
        last_err = ""
        for rgs_try in _row_group_sizes_to_try(row_group_size):
            ok, err = sort_parquet_file(
                src,
                sorted_tmp,
                memory_per_sort_gb,
                temp_directory=duckdb_temp_dir,
                row_group_size=rgs_try,
            )
            if ok:
                sort_ok = True
                break
            last_err = err
            if "Out of Memory" not in (err or ""):
                break

        if not sort_ok:
            return str(src), False, f"sort failed: {last_err}", ""

        ok, reason = is_sorted_by_content(sorted_tmp)
        if not ok:
            try:
                sorted_tmp.unlink()
            except Exception:
                pass
            return str(src), False, f"verification failed: {reason}", ""

        if src.name.endswith(".gz.parquet"):
            new_name = src.name.replace(".gz.parquet", ".gz.sorted.parquet")
        else:
            new_name = src.name.replace(".parquet", ".sorted.parquet")
        out = src.parent / new_name

        # If a sorted output already exists, treat as success and remove duplicate unsorted.
        if out.exists():
            try:
                src.unlink()
            except Exception:
                pass
            return str(src), True, "already sorted (output existed)", str(out)

        sorted_tmp.replace(out)
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        return str(src), True, "sorted", str(out)

    except Exception as e:
        try:
            if work_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)
            if duckdb_temp_dir is not None:
                shutil.rmtree(duckdb_temp_dir, ignore_errors=True)
        except Exception:
            pass
        return str(src), False, f"exception: {e}", ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and mark sorted parquet files")
    ap.add_argument("--parquet-root", required=True, type=str, help="Root directory of parquet files")
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
        "--row-group-size",
        type=int,
        default=int(os.environ.get("CC_SORT_ROW_GROUP_SIZE", "71680")),
        help=(
            "Parquet row group size in rows when writing/re-writing sorted shards. "
            "Default: env CC_SORT_ROW_GROUP_SIZE else 71680 (≈4MB compressed in 2024 samples). "
            "Use 0 to let DuckDB choose."
        ),
    )

    ap.add_argument(
        "--rewrite-sorted",
        action="store_true",
        default=False,
        help="Also rewrite already-sorted *.sorted.parquet files (keeps name) to apply row-group-size / normalization",
    )
    ap.add_argument(
        "--rewrite-sorted-if-needed",
        action="store_true",
        default=False,
        help=(
            "When used with --rewrite-sorted, only rewrite shards that are not already optimized "
            "(wrong row-group sizing ~target MB, non-ZSTD, or missing required columns)."
        ),
    )
    ap.add_argument(
        "--rewrite-target-mb",
        type=int,
        default=None,
        help=(
            "Target row group size in MB for deciding whether a shard needs rewrite (default: env CC_SORT_ROW_GROUP_TARGET_MB, else 4). "
            "Used only with --rewrite-sorted-if-needed."
        ),
    )
    ap.add_argument(
        "--rewrite-tolerance",
        type=float,
        default=0.35,
        help=(
            "Tolerance ratio for row group size check when deciding whether a shard needs rewrite. "
            "Example: 0.35 means accept within ±35%% of target MB (default: 0.35)."
        ),
    )
    ap.add_argument(
        "--rewrite-require-column",
        action="append",
        default=None,
        help=(
            "Column name that must exist to treat a shard as already optimized. Can be repeated. "
            "When --rewrite-sorted-if-needed is set and this is omitted, defaults to requiring 'collection' and 'shard_file'."
        ),
    )
    ap.add_argument(
        "--rewrite-workers",
        type=int,
        default=None,
        help=(
            "Parallel workers for rewriting already-sorted files when --rewrite-sorted is enabled. "
            "Default: --workers (if set), else --sort-workers."
        ),
    )
    ap.add_argument(
        "--rewrite-reason-log-limit",
        type=int,
        default=20,
        help=(
            "When --rewrite-sorted-if-needed is enabled, log up to N example shards with their rewrite reason(s). "
            "Default: 20; use 0 to disable."
        ),
    )
    ap.add_argument("--sort-unsorted", action="store_true", help="Sort any unsorted files found")
    ap.add_argument("--verify-only", action="store_true", help="Only verify, don't mark or sort")
    ap.add_argument("--memory-per-sort", type=float, default=4.0, help="GB memory per sort operation")
    ap.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    ap.add_argument(
        "--sort-workers",
        type=int,
        default=1,
        help="Parallel workers for sorting unsorted files (default: 1; keep low for memory safety)",
    )
    ap.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Temp directory for DuckDB external sort spill (default: system temp)",
    )
    ap.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="Print a periodic heartbeat every N seconds during long phases (default: 30)",
    )

    args = ap.parse_args()

    # Normalize: allow users to disable explicit row-group sizing.
    if args.row_group_size is not None and int(args.row_group_size) <= 0:
        args.row_group_size = None

    # Default rewrite target MB to match our standard row group sizing.
    if args.rewrite_target_mb is None:
        try:
            args.rewrite_target_mb = int(os.environ.get("CC_SORT_ROW_GROUP_TARGET_MB", "4"))
        except Exception:
            args.rewrite_target_mb = 4

    parquet_root = Path(args.parquet_root).expanduser().resolve()

    if not parquet_root.exists():
        print(f"❌ ERROR: Parquet root not found: {parquet_root}")
        return 1

    print("=" * 80)
    print("PARQUET FILE VALIDATION AND MARKING")
    print("=" * 80)
    print(f"Root: {parquet_root}")
    print()

    all_files = _iter_candidate_parquet_files(parquet_root)
    if args.only:
        only_raw = {str(x).strip() for x in args.only if str(x).strip()}

        def _matches_only(p: Path) -> bool:
            name = p.name
            stem = name
            for suf in (".gz.sorted.parquet", ".gz.parquet", ".sorted.parquet", ".parquet", ".gz"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
            candidates = {
                name,
                name.replace(".sorted.", "."),
                stem,
                f"{stem}.gz",
                f"{stem}.gz.parquet",
                f"{stem}.gz.sorted.parquet",
                f"{stem}.parquet",
                f"{stem}.sorted.parquet",
            }
            return bool(candidates & only_raw)

        all_files = [p for p in all_files if _matches_only(p)]

    print(f"Found {len(all_files)} parquet files")
    print()

    already_marked: List[Path] = []
    sorted_unmarked: List[Path] = []
    unsorted_files: List[Path] = []
    error_files: List[Tuple[Path, str]] = []

    print("Checking files...")
    print("-" * 80)

    num_workers = args.workers or multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers")
    print()

    completed = 0
    heartbeat_seconds = max(1, int(args.heartbeat_seconds))
    start_check = time.monotonic()
    last_hb = start_check

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(check_single_file, pq_file, parquet_root, args.verify_only): pq_file
            for pq_file in all_files
        }

        for future in as_completed(futures):
            completed += 1
            pq_file = futures[future]

            try:
                status, result_path, _is_sorted, reason = future.result()
                rel_path = result_path.relative_to(parquet_root)

                if status == "already_marked":
                    already_marked.append(result_path)
                    if completed % 100 == 0:
                        print(f"[{completed}/{len(all_files)}] ⏭️  {rel_path} (already marked)")

                elif status == "sorted_unmarked":
                    sorted_unmarked.append(result_path)
                    print(f"[{completed}/{len(all_files)}] ✅ {rel_path} - {reason}")

                elif status == "unsorted":
                    unsorted_files.append(result_path)
                    print(f"[{completed}/{len(all_files)}] ❌ {rel_path} - UNSORTED: {reason}")

                elif status == "error":
                    error_files.append((result_path, reason))
                    print(f"[{completed}/{len(all_files)}] ⚠️  {rel_path} - ERROR: {reason}")

                if completed % 50 == 0:
                    print(
                        f"Progress: {completed}/{len(all_files)} - Marked: {len(already_marked)}, "
                        f"Sorted: {len(sorted_unmarked)}, Unsorted: {len(unsorted_files)}",
                        flush=True,
                    )

                now = time.monotonic()
                if now - last_hb >= heartbeat_seconds:
                    elapsed = now - start_check
                    print(
                        f"Heartbeat(check): {completed}/{len(all_files)} done in {elapsed/60:.1f} min "
                        f"(marked={len(already_marked)}, sorted={len(sorted_unmarked)}, "
                        f"unsorted={len(unsorted_files)}, errors={len(error_files)})",
                        flush=True,
                    )
                    last_hb = now

            except Exception as e:
                print(f"[{completed}/{len(all_files)}] ⚠️  {pq_file.name} - Exception: {e}")

    print("-" * 80)
    print()
    print("Summary:")
    print(f"  Total files:           {len(all_files)}")
    print(f"  ✅ Already marked:     {len(already_marked)}")
    print(f"  ✅ Sorted (unmarked):  {len(sorted_unmarked)}")
    print(f"  ❌ Unsorted:           {len(unsorted_files)}")
    print(f"  ⚠️  Errors:            {len(error_files)}")
    total_sorted = len(already_marked) + len(sorted_unmarked)
    print(f"  Total sorted:          {total_sorted}")
    if all_files:
        print(f"  Percentage sorted:     {total_sorted / len(all_files) * 100:.1f}%")
    if args.rewrite_sorted:
        print(
            "  Note: 'Already marked' means the filename already ends with *.sorted.parquet. "
            "Rewrite mode may still rewrite marked shards to normalize row-groups/compression."
        )
    print()

    failed_count = 0
    if (unsorted_files or args.rewrite_sorted) and args.sort_unsorted and not args.verify_only:
        print("=" * 80)
        print("SORTING UNSORTED FILES")
        print("=" * 80)
        print()

        sorted_count = 0

        sort_workers = max(1, int(args.sort_workers))
        rewrite_workers = max(1, int(args.rewrite_workers or args.workers or sort_workers))
        temp_root = Path(args.temp_dir).expanduser().resolve() if args.temp_dir else Path(tempfile.gettempdir())
        temp_root.mkdir(parents=True, exist_ok=True)

        rewrite_files: List[Path] = []
        if args.rewrite_sorted:
            # Only rewrite already-marked sorted files.
            rewrite_files = list(already_marked)
            if args.rewrite_sorted_if_needed and rewrite_files:
                # Decide target MB.
                target_mb = args.rewrite_target_mb
                if target_mb is None:
                    try:
                        target_mb = int((os.environ.get("CC_SORT_ROW_GROUP_TARGET_MB") or "128").strip() or 128)
                    except Exception:
                        target_mb = 128

                # Decide required columns.
                required_cols: set[str]
                if args.rewrite_require_column:
                    required_cols = {str(c).strip() for c in args.rewrite_require_column if str(c).strip()}
                else:
                    required_cols = {"collection", "shard_file"}

                keep: List[Path] = []
                skipped = 0
                missing_cols = 0
                reason_counts: Counter[str] = Counter()
                example_lines: List[str] = []

                target_rows = args.row_group_size if args.row_group_size is not None else None
                for p in rewrite_files:
                    needs, reasons = _parquet_rewrite_reasons(
                        p,
                        target_rows=int(target_rows) if target_rows is not None else None,
                        target_mb=int(target_mb) if target_mb is not None else None,
                        tolerance_ratio=float(args.rewrite_tolerance),
                        required_columns=required_cols,
                    )
                    if reasons == ["missing_required_columns"]:
                        missing_cols += 1
                        continue

                    if needs:
                        keep.append(p)
                        for r in reasons:
                            reason_counts[r] += 1
                        limit = max(0, int(args.rewrite_reason_log_limit or 0))
                        if limit and len(example_lines) < limit:
                            med_rows = _parquet_rowgroups_median_rows(p)
                            if med_rows is not None and target_rows is not None:
                                detail = f"median_rg_rows≈{int(med_rows):,} target={int(target_rows):,} tol=±{float(args.rewrite_tolerance):.2f}"
                            else:
                                detail = f"target≈{target_mb}MB tol=±{float(args.rewrite_tolerance):.2f}"
                            example_lines.append(f"  - {p.name}: {', '.join(reasons)} ({detail})")
                    else:
                        skipped += 1
                rewrite_files = keep
                print(
                    f"Rewrite-if-needed enabled: will rewrite {len(rewrite_files)} shard(s), skip {skipped} already-optimized shard(s) "
                    f"(target_rows={int(target_rows) if target_rows is not None else None}, target≈{target_mb}MB, tol=±{float(args.rewrite_tolerance):.2f}, require_cols={sorted(required_cols)})"
                )
                if reason_counts:
                    # Note: counts are per-reason and a shard may contribute to multiple reasons.
                    summary = ", ".join(
                        f"{k}={v}" for k, v in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    )
                    print(f"Rewrite reasons (files may have multiple): {summary}")
                if example_lines:
                    print("Example rewrite decisions:")
                    for line in example_lines:
                        print(line)
                if missing_cols:
                    print(
                        f"⚠️  Rewrite-if-needed skipped {missing_cols} shard(s) missing required columns {sorted(required_cols)}. "
                        "Rewrite cannot add columns; run repair_legacy_parquet_columns.py to fix schema."
                    )
            elif rewrite_files:
                print(f"Rewrite enabled: will rewrite {len(rewrite_files)} already-sorted file(s)")

        def _looks_like_pool_crash(exc: BaseException) -> bool:
            msg = str(exc)
            return (
                isinstance(exc, BrokenProcessPool)
                or "BrokenProcessPool" in msg
                or "terminated abruptly" in msg
                or "process pool" in msg and "terminated" in msg
            )

        def _make_executor(max_workers: int) -> ProcessPoolExecutor:
            """Prefer spawn for DuckDB/Arrow stability; fall back if unsupported."""

            try:
                ctx = multiprocessing.get_context("spawn")
                return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
            except TypeError:
                # Older Python: no mp_context support.
                return ProcessPoolExecutor(max_workers=max_workers)

        def _run_sort_pass(
            files: List[Path],
            pass_workers: int,
            *,
            initial_inflight: int,
            ramp_step_seconds: float,
        ) -> Tuple[int, List[Path], bool]:
            """Run one sort pass.

            Returns: (num_succeeded, failed_files, pool_crashed)
            """

            if not files:
                return 0, [], False

            print(f"Processing {len(files)} file(s) with {pass_workers} worker(s)")
            # Encode per-file mode.
            work_items = []
            for p in files:
                mode = "rewrite" if (args.rewrite_sorted and p.name.endswith(".sorted.parquet")) else "sort"
                work_items.append((str(p), float(args.memory_per_sort), str(temp_root), args.row_group_size, mode))

            ok_local = 0
            failed_local: List[Path] = []
            pool_crashed = False

            with _make_executor(pass_workers) as executor:
                done = 0
                start_sort = time.monotonic()
                last_sort_hb = start_sort

                # Slow-start ramp: begin with a small number of in-flight tasks, then
                # gradually increase up to pass_workers.
                inflight_limit = max(1, min(int(initial_inflight), pass_workers))
                next_ramp = start_sort + max(0.0, float(ramp_step_seconds or 0.0))
                item_idx = 0
                futures: dict = {}
                pending: set = set()

                def _submit_one() -> None:
                    nonlocal item_idx, pool_crashed
                    if item_idx >= len(work_items):
                        return
                    item = work_items[item_idx]
                    item_idx += 1
                    try:
                        fut = executor.submit(sort_and_mark_one, item)
                    except Exception as e:
                        # If the underlying pool crashed, don't raise here; mark the pool as
                        # unusable and let the caller retry with backoff / reduced parallelism.
                        if _looks_like_pool_crash(e):
                            pool_crashed = True
                            failed_local.append(Path(item[0]))
                            print(f"❌ submit failed for {Path(item[0]).name}: {e}")
                            return
                        raise
                    futures[fut] = item[0]
                    pending.add(fut)

                while item_idx < len(work_items) or pending:
                    if pool_crashed:
                        # Stop early; retry logic in the caller will handle remaining shards.
                        try:
                            for pfut in list(pending):
                                pfut.cancel()
                        except Exception:
                            pass
                        break
                    # Ramp up concurrency if requested.
                    if ramp_step_seconds and ramp_step_seconds > 0:
                        now = time.monotonic()
                        while inflight_limit < pass_workers and now >= next_ramp:
                            inflight_limit += 1
                            next_ramp += float(ramp_step_seconds)

                    while len(pending) < inflight_limit and item_idx < len(work_items):
                        _submit_one()
                        if pool_crashed:
                            break

                    if pool_crashed:
                        continue

                    if not pending:
                        continue

                    # Don't sleep past a ramp boundary.
                    timeout = float(heartbeat_seconds)
                    if ramp_step_seconds and ramp_step_seconds > 0 and inflight_limit < pass_workers:
                        timeout = max(0.5, min(timeout, max(0.5, next_ramp - time.monotonic())))

                    finished, _still_pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)

                    if not finished:
                        now = time.monotonic()
                        if now - last_sort_hb >= heartbeat_seconds:
                            elapsed = now - start_sort
                            print(
                                f"Heartbeat(sort): {done}/{len(files)} done in {elapsed/60:.1f} min "
                                f"(ok={ok_local}, fail={len(failed_local)}, inflight={len(pending)}/{inflight_limit}, remaining={len(work_items)-item_idx})",
                                flush=True,
                            )
                            last_sort_hb = now
                        continue

                    for fut in finished:
                        pending.discard(fut)
                        done += 1
                        src = futures.get(fut, "")
                        try:
                            _src_path, ok, msg, out_path = fut.result()
                            if ok and out_path:
                                ok_local += 1
                                print(f"✅ [{done}/{len(files)}] {Path(src).name} -> {Path(out_path).name} ({msg})")
                            else:
                                failed_local.append(Path(src))
                                print(f"❌ [{done}/{len(files)}] {Path(src).name}: {msg}")
                        except Exception as e:
                            failed_local.append(Path(src))
                            print(f"❌ [{done}/{len(files)}] {Path(src).name}: exception {e}")

                            if _looks_like_pool_crash(e):
                                pool_crashed = True
                                # Cancel remaining work; we'll retry after a backoff.
                                for pfut in list(pending):
                                    psrc = futures.get(pfut)
                                    if psrc:
                                        failed_local.append(Path(psrc))
                                    try:
                                        pfut.cancel()
                                    except Exception:
                                        pass
                                pending.clear()
                                break

            # De-dup (pool crash path can add duplicates).
            uniq_failed = sorted({p.resolve() for p in failed_local if p})
            return ok_local, uniq_failed, pool_crashed

        def _run_with_pool_crash_retries(files: List[Path], pass_workers: int, label: str) -> Tuple[int, List[Path]]:
            """Run a pass with pool-crash detection/backoff.

            Returns: (ok_count, final_failed_files)
            """

            if not files:
                return 0, []

            ok_total = 0
            failed_files: List[Path] = []

            ok1, failed1, crashed1 = _run_sort_pass(
                files,
                pass_workers,
                initial_inflight=pass_workers,
                ramp_step_seconds=0.0,
            )
            ok_total += ok1
            failed_files = failed1

            retries = 0
            retry_files = failed_files
            while retry_files and crashed1 and pass_workers > 1 and retries < 3:
                retries += 1
                backoff = float(30 * (2 ** (retries - 1)))
                ramp_step = float(10 * retries)

                print()
                print(
                    f"⚠️  Detected process-pool crash during {label}. "
                    f"Backing off for {backoff:.0f}s, then retrying with slow-start ramp (to {pass_workers} workers, step={ramp_step:.0f}s)...",
                    flush=True,
                )
                time.sleep(backoff)

                ok_r, failed_r, crashed_r = _run_sort_pass(
                    retry_files,
                    pass_workers,
                    initial_inflight=1,
                    ramp_step_seconds=ramp_step,
                )
                ok_total += ok_r
                retry_files = failed_r
                crashed1 = crashed_r

            if retry_files and pass_workers > 1:
                print()
                print(
                    f"⚠️  Retrying remaining failed shards with 1 worker as a last resort for {label}...",
                    flush=True,
                )
                ok2, failed2, _crashed2 = _run_sort_pass(
                    retry_files,
                    1,
                    initial_inflight=1,
                    ramp_step_seconds=0.0,
                )
                ok_total += ok2
                retry_files = failed2

            return ok_total, retry_files

        # Run unsorted sorts (if any) with the (typically low) sort_workers.
        ok_sort, failed_sort = _run_with_pool_crash_retries(list(unsorted_files), sort_workers, label="sorting")
        sorted_count += ok_sort
        failed_count += len(failed_sort)

        # Run rewrites separately so they can use higher parallelism without affecting
        # unsorted sorting memory-safety defaults.
        ok_rewrite, failed_rewrite = _run_with_pool_crash_retries(list(rewrite_files), rewrite_workers, label="rewriting")
        sorted_count += ok_rewrite
        failed_count += len(failed_rewrite)

        print()
        print("Sorting complete:")
        print(f"  Succeeded: {sorted_count}")
        print(f"  Failed:    {failed_count}")
        # When rewrite is enabled, `sorted_count` includes rewrites of already-sorted files,
        # so it is not meaningful to report it as "total sorted files".
        total_sorted = len(already_marked) + len(sorted_unmarked)
        if args.rewrite_sorted:
            print(f"  Total files processed in sort/rewrite: {sorted_count + failed_count}")
            print(f"  Total files sorted (marked + newly marked): {total_sorted}/{len(all_files)}")
        else:
            print(f"  Total sorted files: {total_sorted + sorted_count}/{len(all_files)}")

    if unsorted_files and not args.sort_unsorted:
        print()
        print("⚠️  WARNING: Some files are not sorted!")
        print("   Run with --sort-unsorted to fix")
        return 1

    if args.sort_unsorted and failed_count:
        print()
        print(f"❌ Sorting failed for {failed_count} file(s)")
        return 2

    print()
    print("✅ All files verified and marked as sorted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
