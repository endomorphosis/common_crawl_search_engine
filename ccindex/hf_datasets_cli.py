#!/usr/bin/env python3
"""CLI for HuggingFace datasets integration.

This script provides commands to interact with the HuggingFace datasets
integration for the Common Crawl Search Engine.

Examples:
    # List available parquet files for a collection
    python -m common_crawl_search_engine.ccindex.hf_datasets_cli list CC-MAIN-2024-10

    # Read a specific rowgroup
    python -m common_crawl_search_engine.ccindex.hf_datasets_cli read \
        --collection CC-MAIN-2024-10 \
        --file cdx-00000.gz.parquet \
        --rowgroup 0 \
        --columns url,timestamp,status

    # Search for URLs using HuggingFace fallback
    python -m common_crawl_search_engine.ccindex.hf_datasets_cli search \
        --urls https://example.com/page1,https://example.com/page2 \
        --collection CC-MAIN-2024-10

    # Check if HuggingFace integration is available
    python -m common_crawl_search_engine.ccindex.hf_datasets_cli check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


def cmd_check(args: argparse.Namespace) -> int:
    """Check if HuggingFace datasets integration is available."""
    try:
        from common_crawl_search_engine.ccindex.hf_datasets_adapter import hf_datasets_available

        if hf_datasets_available():
            print("✓ HuggingFace datasets integration is available")
            print(f"  Dataset: {os.environ.get('HF_DATASET_NAME', 'Publicus/common_crawl_pointers_by_collection')}")
            print(f"  Revision: {os.environ.get('HF_DATASET_REVISION', 'main')}")
            return 0
        else:
            print("✗ HuggingFace datasets integration is NOT available")
            print("  Install required packages:")
            print("    pip install datasets pyarrow")
            return 1
    except Exception as e:
        print(f"✗ Error checking HuggingFace availability: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List available parquet files for a collection."""
    try:
        from common_crawl_search_engine.ccindex.hf_datasets_adapter import HFRowGroupReader

        reader = HFRowGroupReader(
            dataset_name=args.dataset,
            revision=args.revision,
        )

        files = reader.list_available_parquet_files(args.collection)

        if not files:
            print(f"No parquet files found for collection: {args.collection}")
            return 1

        print(f"Available parquet files for {args.collection}:")
        for f in sorted(files):
            print(f"  - {f}")
        print(f"\nTotal: {len(files)} files")
        return 0

    except Exception as e:
        print(f"Error listing files: {e}")
        return 1


def cmd_read(args: argparse.Namespace) -> int:
    """Read a specific rowgroup from a parquet file."""
    try:
        import pyarrow as pa
        from common_crawl_search_engine.ccindex.hf_datasets_adapter import HFRowGroupReader

        reader = HFRowGroupReader(
            dataset_name=args.dataset,
            revision=args.revision,
        )

        columns = args.columns.split(",") if args.columns else None

        table = reader.read_rowgroup(
            collection=args.collection,
            parquet_filename=args.file,
            row_group=args.rowgroup,
            columns=columns,
        )

        if table is None:
            print(f"Failed to read rowgroup {args.rowgroup} from {args.file}")
            return 1

        print(f"Read {table.num_rows} rows from rowgroup {args.rowgroup}")
        print(f"Schema: {[f.name for f in table.schema]}")

        if args.output:
            # Write to file
            output_path = Path(args.output)
            if args.format == "parquet":
                pa.parquet.write_table(table, str(output_path))
            elif args.format == "json":
                df = table.to_pandas()
                df.to_json(output_path, orient="records", lines=True)
            elif args.format == "csv":
                df = table.to_pandas()
                df.to_csv(output_path, index=False)
            print(f"Wrote output to: {output_path}")
        else:
            # Print to stdout
            df = table.to_pandas()
            print("\nData:")
            print(df.to_string())

        return 0

    except Exception as e:
        print(f"Error reading rowgroup: {e}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search for URLs using HuggingFace fallback."""
    try:
        from common_crawl_search_engine.ccindex.api import resolve_urls_to_ccindex

        # Enable HuggingFace fallback
        os.environ["HF_ENABLE_FALLBACK"] = "true"
        if args.dataset:
            os.environ["HF_DATASET_NAME"] = args.dataset
        if args.revision:
            os.environ["HF_DATASET_REVISION"] = args.revision

        urls = args.urls.split(",")

        # Determine parquet_root - use provided or default
        parquet_root = Path(args.parquet_root) if args.parquet_root else Path("/storage/ccindex_parquet")

        print(f"Searching for {len(urls)} URLs...")
        print(f"Collection: {args.collection or 'auto-detect'}")
        print(f"Parquet root: {parquet_root}")
        print(f"HuggingFace dataset: {os.environ.get('HF_DATASET_NAME', 'Publicus/common_crawl_pointers_by_collection')}")

        results = resolve_urls_to_ccindex(
            urls=urls,
            parquet_root=parquet_root,
            year=args.year,
            max_matches_per_domain=args.max_matches,
            per_url_limit=args.per_url_limit,
        )

        # Print results
        total_records = 0
        for url, records in results.items():
            print(f"\n{url}: {len(records)} records")
            total_records += len(records)

            for rec in records[: args.limit]:
                source = rec.get("source", "unknown")
                print(f"  - [{source}] {rec.get('timestamp', 'N/A')} | {rec.get('url', 'N/A')[:80]}")
                if args.verbose:
                    print(f"    Collection: {rec.get('collection', 'N/A')}")
                    print(f"    WARC: {rec.get('warc_filename', 'N/A')}:{rec.get('warc_offset', 'N/A')}")
                    print(f"    Status: {rec.get('status', 'N/A')}, MIME: {rec.get('mime', 'N/A')}")

            if len(records) > args.limit:
                print(f"  ... and {len(records) - args.limit} more records")

        print(f"\nTotal records found: {total_records}")

        # Write output if requested
        if args.output:
            output_data = {
                "urls": urls,
                "results": results,
                "total_records": total_records,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nWrote results to: {args.output}")

        return 0

    except Exception as e:
        print(f"Error searching: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_schema(args: argparse.Namespace) -> int:
    """Get the schema of a parquet file."""
    try:
        from common_crawl_search_engine.ccindex.hf_datasets_adapter import HFRowGroupReader

        reader = HFRowGroupReader(
            dataset_name=args.dataset,
            revision=args.revision,
        )

        schema = reader.get_parquet_schema(args.collection, args.file)

        if schema is None:
            print(f"Failed to get schema for {args.file}")
            return 1

        print(f"Schema for {args.file}:")
        for field in schema:
            print(f"  - {field.name}: {field.type}")

        return 0

    except Exception as e:
        print(f"Error getting schema: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="hf_datasets_cli",
        description="CLI for HuggingFace datasets integration",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Global options
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("HF_DATASET_NAME", "Publicus/common_crawl_pointers_by_collection"),
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=os.environ.get("HF_DATASET_REVISION", "main"),
        help="Dataset revision/tag",
    )

    # check command
    check_parser = subparsers.add_parser("check", help="Check if HuggingFace integration is available")
    check_parser.set_defaults(func=cmd_check)

    # list command
    list_parser = subparsers.add_parser("list", help="List available parquet files for a collection")
    list_parser.add_argument("collection", type=str, help="Collection name (e.g., CC-MAIN-2024-10)")
    list_parser.set_defaults(func=cmd_list)

    # read command
    read_parser = subparsers.add_parser("read", help="Read a specific rowgroup from a parquet file")
    read_parser.add_argument("--collection", type=str, required=True, help="Collection name")
    read_parser.add_argument("--file", type=str, required=True, help="Parquet filename")
    read_parser.add_argument("--rowgroup", type=int, required=True, help="Row group index")
    read_parser.add_argument("--columns", type=str, help="Comma-separated list of columns to read")
    read_parser.add_argument("--output", type=str, help="Output file path")
    read_parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "json", "csv"],
        default="json",
        help="Output format",
    )
    read_parser.set_defaults(func=cmd_read)

    # schema command
    schema_parser = subparsers.add_parser("schema", help="Get the schema of a parquet file")
    schema_parser.add_argument("--collection", type=str, required=True, help="Collection name")
    schema_parser.add_argument("--file", type=str, required=True, help="Parquet filename")
    schema_parser.set_defaults(func=cmd_schema)

    # search command
    search_parser = subparsers.add_parser("search", help="Search for URLs using HuggingFace fallback")
    search_parser.add_argument("--urls", type=str, required=True, help="Comma-separated list of URLs to search")
    search_parser.add_argument("--collection", type=str, help="Specific collection to search")
    search_parser.add_argument("--year", type=str, help="Year to search (e.g., 2024)")
    search_parser.add_argument("--parquet-root", type=str, help="Local parquet root directory")
    search_parser.add_argument("--max-matches", type=int, default=400, help="Max matches per domain")
    search_parser.add_argument("--per-url-limit", type=int, default=5, help="Max records per URL")
    search_parser.add_argument("--limit", type=int, default=10, help="Max records to display per URL")
    search_parser.add_argument("--output", type=str, help="Output JSON file path")
    search_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    search_parser.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
