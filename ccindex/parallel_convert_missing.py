#!/usr/bin/env python3
"""Parallel conversion of missing .gz files to canonical CC Parquet shards.

This script exists as an operational helper: find CC index shard .gz files that
do not yet have a corresponding .parquet output and convert them in parallel.

Critical invariant: produced Parquet shards must include provenance columns
`collection` and `shard_file`.

Implementation note:
- We delegate the actual conversion logic to `bulk_convert_gz_to_parquet.py` so
    the schema stays in-sync with the main pipeline.
"""

from __future__ import annotations

from pathlib import Path
from multiprocessing import Pool

import psutil
import pyarrow.parquet as pq

try:
        # When run as part of the installed package.
        from common_crawl_search_engine.ccindex.bulk_convert_gz_to_parquet import convert_gz_to_parquet
except Exception:
        # When run directly from this folder.
        from bulk_convert_gz_to_parquet import convert_gz_to_parquet

CCINDEX_ROOT = Path("/storage/ccindex")
# Canonical parquet layout is cc_pointers_by_collection/<year>/<collection>/.
PARQUET_ROOT = Path("/storage/ccindex_parquet/cc_pointers_by_collection")
CHUNK_SIZE = 50000
MAX_WORKERS = 8  # Conservative for memory

def get_available_memory_gb():
    """Get available memory in GB"""
    mem = psutil.virtual_memory()
    return mem.available / (1024**3)

def get_year_from_crawl(crawl_name):
    """Extract year from crawl name"""
    return crawl_name.split("-")[2]

def find_missing_conversions():
    """Find all .gz files that don't have corresponding .parquet files"""
    missing = []
    
    for crawl_dir in sorted(CCINDEX_ROOT.glob("CC-MAIN-202[45]-*")):
        crawl_name = crawl_dir.name
        year = get_year_from_crawl(crawl_name)
        
        for gz_file in sorted(crawl_dir.glob("*.gz")):
            parquet_path = PARQUET_ROOT / year / crawl_name / f"{gz_file.name}.parquet"
            if not parquet_path.exists():
                missing.append((str(gz_file), str(parquet_path)))
    
    return missing

def convert_one_file(args):
    """Convert a single .gz to .parquet (for multiprocessing)"""
    gz_path_str, parquet_path_str = args
    gz_path = Path(gz_path_str)
    parquet_path = Path(parquet_path_str)
    
    try:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        ok = bool(convert_gz_to_parquet(gz_path, parquet_path, chunk_size=int(CHUNK_SIZE)))
        if not ok:
            if parquet_path.exists():
                parquet_path.unlink(missing_ok=True)
            return (False, gz_path.name, 0, 0, "converter returned failure")

        total_records = 0
        try:
            pf = pq.ParquetFile(str(parquet_path))
            if pf.metadata is not None:
                total_records = int(pf.metadata.num_rows or 0)
        except Exception:
            total_records = 0

        size_mb = parquet_path.stat().st_size / 1024 / 1024
        return (True, gz_path.name, total_records, size_mb, None)

    except Exception as e:
        if parquet_path.exists():
            parquet_path.unlink(missing_ok=True)
        return (False, gz_path.name, 0, 0, str(e))

def main() -> int:
    print("Finding missing .gz to .parquet conversions...")
    missing = find_missing_conversions()
    
    if not missing:
        print("✓ All .gz files have been converted to .parquet")
        return 0
    
    print(f"Found {len(missing)} files to convert")
    
    # Adjust workers based on available memory
    avail_mem = get_available_memory_gb()
    workers = min(MAX_WORKERS, max(1, int(avail_mem / 2)))  # 2GB per worker
    print(f"Using {workers} parallel workers (available memory: {avail_mem:.1f} GB)")
    
    success_count = 0
    fail_count = 0
    
    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_one_file, missing), 1):
            success, filename, records, size_mb, error = result
            
            if success:
                print(f"[{i}/{len(missing)}] ✓ {filename}: {records:,} records, {size_mb:.1f} MB")
                success_count += 1
            else:
                print(f"[{i}/{len(missing)}] ✗ {filename}: {error}")
                fail_count += 1
            
            if i % 100 == 0:
                mem = psutil.virtual_memory()
                print(f"  Progress: {i}/{len(missing)} ({100*i/len(missing):.1f}%), Memory: {mem.percent}% used")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"{'='*60}")
    
    return 0 if fail_count == 0 else 1

if __name__ == '__main__':
    raise SystemExit(main())
