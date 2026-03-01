"""HuggingFace datasets adapter for Common Crawl rowgroup access.

This module provides an adapter to fetch rowgroups from HuggingFace datasets
at Publicus/common_crawl_pointers_by_collection as a fallback when local
parquet files are not available.

Usage:
    from common_crawl_search_engine.ccindex.hf_datasets_adapter import (
        get_hf_parquet_rowgroup,
        HFRowGroupReader,
    )

    # Read a specific rowgroup from a parquet file via HuggingFace
    table = get_hf_parquet_rowgroup(
        collection="CC-MAIN-2024-10",
        parquet_filename="cdx-00000.gz.parquet",
        row_group=0,
        columns=["url", "timestamp", "status"],
    )

Environment variables:
    - HF_DATASET_NAME: HuggingFace dataset name (default: Publicus/common_crawl_pointers_by_collection)
    - HF_DATASET_REVISION: Dataset revision/tag to use (default: main)
    - HF_TOKEN: HuggingFace API token for private datasets
    - HF_CACHE_DIR: Cache directory for downloaded data (default: ~/.cache/huggingface/datasets)
    - HF_ENABLE_REMOTE: Enable remote HuggingFace access (default: true)
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

# Lazy imports for optional dependencies
_have_datasets = False
_datasets_import_error: Optional[str] = None

try:
    import datasets
    from datasets import load_dataset
    _have_datasets = True
except ImportError as e:
    _datasets_import_error = str(e)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _have_pyarrow = True
except ImportError:
    _have_pyarrow = False


def hf_dataset_resolve_url(dataset_name: str, relpath: str, revision: str = "main") -> str:
    """Build a HuggingFace resolve URL for a dataset file path."""

    ds = (dataset_name or "").strip("/")
    rp = str(relpath or "").lstrip("/")
    rev = (revision or "main").strip() or "main"
    return f"https://huggingface.co/datasets/{ds}/resolve/{rev}/{rp}"


def _is_transient_remote_error(msg: str) -> bool:
    s = (msg or "").lower()
    return any(
        tok in s
        for tok in [
            " 429",
            "too many requests",
            " 502",
            " 503",
            " 504",
            "timeout",
            "timed out",
            "connection reset",
            "temporarily unavailable",
        ]
    )


def _collection_year(collection: str) -> Optional[str]:
    parts = (collection or "").split("-")
    if len(parts) >= 3 and parts[2].isdigit():
        return parts[2]
    return None


class HFMetaIndexSQLReader:
    """DuckDB-backed reader for querying HF-hosted meta indexes via SQL.

    This reader relies on DuckDB's HTTP parquet support, so callers can query
    metadata and pointer shards directly from HuggingFace without downloading
    full datasets.
    """

    def __init__(
        self,
        *,
        index_dataset_name: Optional[str] = None,
        pointers_dataset_name: Optional[str] = None,
        revision: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_base_sleep_s: Optional[float] = None,
    ):
        self.index_dataset_name = (
            index_dataset_name
            or os.environ.get("HF_META_INDEX_DATASET_NAME")
            or "Publicus/common_crawl_pointer_indices"
        )
        self.pointers_dataset_name = (
            pointers_dataset_name
            or os.environ.get("HF_POINTER_DATASET_NAME")
            or os.environ.get("HF_DATASET_NAME")
            or "Publicus/common_crawl_pointers_by_collection"
        )
        self.revision = (
            revision
            or os.environ.get("HF_META_DATASET_REVISION")
            or os.environ.get("HF_DATASET_REVISION")
            or "main"
        )

        try:
            self.max_retries = int(
                max_retries
                if max_retries is not None
                else (os.environ.get("HF_REMOTE_SQL_RETRIES") or 3)
            )
        except Exception:
            self.max_retries = 3
        if self.max_retries < 0:
            self.max_retries = 0

        try:
            self.retry_base_sleep_s = float(
                retry_base_sleep_s
                if retry_base_sleep_s is not None
                else (os.environ.get("HF_REMOTE_SQL_RETRY_BASE_S") or 0.5)
            )
        except Exception:
            self.retry_base_sleep_s = 0.5
        if self.retry_base_sleep_s < 0:
            self.retry_base_sleep_s = 0.0

    def _require_duckdb(self):
        try:
            import duckdb  # type: ignore

            return duckdb
        except Exception as e:
            raise RuntimeError(
                "duckdb is required for HuggingFace remote SQL queries. "
                "Install with: pip install -e '.[ccindex]'"
            ) from e

    def _query_with_retry(self, sql: str, params: List[object]) -> List[Tuple[object, ...]]:
        duckdb = self._require_duckdb()
        attempt = 0
        while True:
            try:
                con = duckdb.connect(database=":memory:")
                try:
                    con.execute("PRAGMA threads=4")
                    return con.execute(sql, params).fetchall()
                finally:
                    con.close()
            except Exception as e:
                msg = str(e)
                if attempt >= self.max_retries or not _is_transient_remote_error(msg):
                    raise
                sleep_s = self.retry_base_sleep_s * (2 ** attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                attempt += 1

    def list_collections(self, year: Optional[str] = None) -> List[Tuple[Optional[str], str]]:
        """Return [(year, collection), ...] from the HF master collection summary."""

        url = hf_dataset_resolve_url(
            self.index_dataset_name,
            "cc_master_index.collection_summary.parquet",
            self.revision,
        )
        if year:
            rows = self._query_with_retry(
                """
                SELECT CAST(year AS VARCHAR), CAST(collection AS VARCHAR)
                FROM read_parquet(?)
                WHERE CAST(year AS VARCHAR) = ?
                ORDER BY collection
                """,
                [url, str(year)],
            )
        else:
            rows = self._query_with_retry(
                """
                SELECT CAST(year AS VARCHAR), CAST(collection AS VARCHAR)
                FROM read_parquet(?)
                ORDER BY year, collection
                """,
                [url],
            )
        out: List[Tuple[Optional[str], str]] = []
        for y, coll in rows:
            out.append((str(y) if y is not None else None, str(coll)))
        return out

    def _domain_shards_urls_for_collection(self, collection: str) -> List[str]:
        y = _collection_year(collection)
        if not y:
            return []

        relpaths = [
            f"{y}/{collection}/{collection}.cc_domain_shards.parquet",
            f"{y}/{collection}/{collection}__cc_domain_shards.parquet",
            f"{y}/cc_pointers_{y}.cc_domain_shards.parquet",
        ]
        return [hf_dataset_resolve_url(self.index_dataset_name, rp, self.revision) for rp in relpaths]

    def parquet_relpaths_for_domain(
        self,
        collection: str,
        host_rev_prefix: str,
        *,
        include_subdomains: bool = True,
    ) -> List[str]:
        """Look up parquet relative paths for a domain in HF-hosted domain-shard indexes."""

        like_pat = host_rev_prefix + ",%" if include_subdomains else None
        urls = self._domain_shards_urls_for_collection(collection)
        if not urls:
            return []

        seen: Set[str] = set()
        out: List[str] = []

        for idx, url in enumerate(urls):
            if include_subdomains:
                if idx <= 1:
                    sql = """
                        SELECT parquet_relpath
                        FROM read_parquet(?)
                        WHERE host_rev = ? OR host_rev LIKE ?
                    """
                    params: List[object] = [url, host_rev_prefix, like_pat or ""]
                else:
                    sql = """
                        SELECT parquet_relpath
                        FROM read_parquet(?)
                        WHERE collection = ? AND (host_rev = ? OR host_rev LIKE ?)
                    """
                    params = [url, collection, host_rev_prefix, like_pat or ""]
            else:
                if idx <= 1:
                    sql = """
                        SELECT parquet_relpath
                        FROM read_parquet(?)
                        WHERE host_rev = ?
                    """
                    params = [url, host_rev_prefix]
                else:
                    sql = """
                        SELECT parquet_relpath
                        FROM read_parquet(?)
                        WHERE collection = ? AND host_rev = ?
                    """
                    params = [url, collection, host_rev_prefix]

            try:
                rows = self._query_with_retry(sql, params)
            except Exception:
                continue

            for row in rows:
                if not row or not row[0]:
                    continue
                rel = str(row[0])
                if rel in seen:
                    continue
                seen.add(rel)
                out.append(rel)

            if out:
                # Prefer collection-level files; stop early if we already have hits.
                break

        return out

    def iter_warc_candidates(
        self,
        collection: str,
        parquet_relpath: str,
        host_rev_prefix: str,
        *,
        limit: int,
    ) -> Iterator[Dict[str, object]]:
        """Stream candidate WARC pointer rows from HF pointer parquet shards."""

        y = _collection_year(collection)
        if not y:
            return

        rel = str(parquet_relpath).lstrip("/")
        url = hf_dataset_resolve_url(
            self.pointers_dataset_name,
            f"{y}/{collection}/{rel}",
            self.revision,
        )
        like_pat = host_rev_prefix + ",%"

        sql = """
            SELECT
                collection,
                shard_file,
                url,
                ts,
                status,
                mime,
                digest,
                warc_filename,
                warc_offset,
                warc_length
            FROM read_parquet(?)
            WHERE host_rev = ? OR host_rev LIKE ?
        """
        params: List[object] = [url, host_rev_prefix, like_pat]
        if int(limit) > 0:
            sql += "\nLIMIT ?"
            params.append(int(limit))

        rows = self._query_with_retry(sql, params)
        for row in rows:
            if not row:
                continue
            (
                row_collection,
                shard_file,
                page_url,
                ts,
                status,
                mime,
                digest,
                warc_filename,
                warc_offset,
                warc_length,
            ) = row
            yield {
                "collection": row_collection or collection,
                "shard_file": shard_file,
                "url": page_url,
                "timestamp": ts,
                "status": int(status) if status is not None else None,
                "mime": mime,
                "digest": digest,
                "warc_filename": warc_filename,
                "warc_offset": int(warc_offset) if warc_offset is not None else None,
                "warc_length": int(warc_length) if warc_length is not None else None,
                "parquet_path": url,
            }


@dataclass(frozen=True)
class HFParquetLocation:
    """Location info for a parquet file in the HuggingFace dataset."""

    collection: str
    parquet_filename: str
    dataset_name: str
    revision: str
    subset: Optional[str] = None


class HFRowGroupReader:
    """Reader for fetching rowgroups from HuggingFace datasets.

    This class manages the connection to HuggingFace datasets and provides
    methods to read specific rowgroups from parquet files stored in the dataset.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_remote: bool = True,
    ):
        """Initialize the HuggingFace rowgroup reader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "Publicus/common_crawl_pointers_by_collection")
            revision: Dataset revision/tag to use
            token: HuggingFace API token for private datasets
            cache_dir: Cache directory for downloaded data
            enable_remote: Whether to enable remote HuggingFace access
        """
        if not _have_datasets:
            raise ImportError(
                f"datasets library is required for HuggingFace access. "
                f"Install with: pip install datasets. Error: {_datasets_import_error}"
            )
        if not _have_pyarrow:
            raise ImportError("pyarrow is required for parquet reading")

        self.dataset_name = dataset_name or os.environ.get(
            "HF_DATASET_NAME", "Publicus/common_crawl_pointers_by_collection"
        )
        self.revision = revision or os.environ.get("HF_DATASET_REVISION", "main")
        self.token = token or os.environ.get("HF_TOKEN")
        self.cache_dir = cache_dir or os.environ.get("HF_CACHE_DIR")
        self.enable_remote = enable_remote
        if os.environ.get("HF_ENABLE_REMOTE", "").lower() in ("false", "0", "no"):
            self.enable_remote = False

        # Cache for loaded datasets by collection
        self._dataset_cache: Dict[str, datasets.Dataset] = {}
        self._parquet_file_cache: Dict[str, pq.ParquetFile] = {}

    def _get_collection_year(self, collection: str) -> Optional[str]:
        """Extract year from collection name (e.g., CC-MAIN-2024-10 -> 2024)."""
        parts = collection.split("-")
        if len(parts) >= 3 and parts[0] == "CC" and parts[1] == "MAIN":
            return parts[2]
        return None

    def _load_dataset_for_collection(self, collection: str) -> Optional[datasets.Dataset]:
        """Load the HuggingFace dataset for a specific collection.

        The dataset is expected to have the following structure:
        - Each collection is a separate subset or organized by year/collection
        - Parquet files are stored as data files within the dataset
        """
        if collection in self._dataset_cache:
            return self._dataset_cache[collection]

        if not self.enable_remote:
            return None

        try:
            year = self._get_collection_year(collection)
            if not year:
                warnings.warn(f"Could not extract year from collection: {collection}")
                return None

            # Try loading with year/collection as subset
            subset_name = f"{year}_{collection}"

            ds = load_dataset(
                self.dataset_name,
                subset_name,
                revision=self.revision,
                token=self.token,
                cache_dir=self.cache_dir,
                streaming=False,  # We need random access to rowgroups
            )

            # Handle DatasetDict (train/test/validation splits)
            if isinstance(ds, datasets.DatasetDict):
                # Prefer 'train' split, otherwise use first available
                if "train" in ds:
                    ds = ds["train"]
                else:
                    ds = ds[list(ds.keys())[0]]

            self._dataset_cache[collection] = ds
            return ds

        except Exception as e:
            warnings.warn(f"Failed to load dataset for collection {collection}: {e}")
            return None

    def _get_parquet_file_from_dataset(
        self,
        collection: str,
        parquet_filename: str,
    ) -> Optional[pq.ParquetFile]:
        """Get a PyArrow ParquetFile for a specific parquet in the dataset.

        This method attempts to locate the parquet file within the loaded dataset
        and return a PyArrow ParquetFile for reading rowgroups.
        """
        cache_key = f"{collection}/{parquet_filename}"
        if cache_key in self._parquet_file_cache:
            return self._parquet_file_cache[cache_key]

        ds = self._load_dataset_for_collection(collection)
        if ds is None:
            return None

        try:
            # Get the underlying data files
            # HuggingFace datasets can provide access to underlying files
            if hasattr(ds, '_data') and hasattr(ds._data, 'files'):
                data_files = ds._data.files
            elif hasattr(ds, 'cache_files'):
                data_files = [f['filename'] for f in ds.cache_files]
            else:
                # Try to find parquet files in the dataset cache directory
                year = self._get_collection_year(collection)
                if year and self.cache_dir:
                    cache_path = Path(self.cache_dir) / self.dataset_name.replace("/", "--") / year / collection
                    if cache_path.exists():
                        parquet_path = cache_path / parquet_filename
                        if parquet_path.exists():
                            pf = pq.ParquetFile(str(parquet_path))
                            self._parquet_file_cache[cache_key] = pf
                            return pf
                return None

            # Find the matching parquet file
            for file_path in data_files:
                if parquet_filename in str(file_path):
                    pf = pq.ParquetFile(str(file_path))
                    self._parquet_file_cache[cache_key] = pf
                    return pf

            return None

        except Exception as e:
            warnings.warn(f"Failed to get parquet file {parquet_filename}: {e}")
            return None

    def read_rowgroup(
        self,
        collection: str,
        parquet_filename: str,
        row_group: int,
        columns: Optional[List[str]] = None,
    ) -> Optional[pa.Table]:
        """Read a specific rowgroup from a parquet file in the HuggingFace dataset.

        Args:
            collection: Collection name (e.g., "CC-MAIN-2024-10")
            parquet_filename: Name of the parquet file (e.g., "cdx-00000.gz.parquet")
            row_group: Row group index to read
            columns: Optional list of columns to read (reads all if None)

        Returns:
            PyArrow Table with the rowgroup data, or None if not available
        """
        pf = self._get_parquet_file_from_dataset(collection, parquet_filename)
        if pf is None:
            return None

        try:
            if row_group < 0 or row_group >= pf.num_row_groups:
                warnings.warn(f"Invalid row group {row_group} for {parquet_filename}")
                return None

            table = pf.read_row_group(row_group, columns=columns)
            return table
        except Exception as e:
            warnings.warn(f"Failed to read rowgroup {row_group} from {parquet_filename}: {e}")
            return None

    def get_parquet_schema(self, collection: str, parquet_filename: str) -> Optional[pa.Schema]:
        """Get the schema of a parquet file in the HuggingFace dataset.

        Args:
            collection: Collection name
            parquet_filename: Name of the parquet file

        Returns:
            PyArrow Schema, or None if not available
        """
        pf = self._get_parquet_file_from_dataset(collection, parquet_filename)
        if pf is None:
            return None
        return pf.schema

    def list_available_parquet_files(self, collection: str) -> List[str]:
        """List available parquet files for a collection in the HuggingFace dataset.

        Args:
            collection: Collection name

        Returns:
            List of parquet filenames available
        """
        ds = self._load_dataset_for_collection(collection)
        if ds is None:
            return []

        parquet_files: List[str] = []

        try:
            if hasattr(ds, '_data') and hasattr(ds._data, 'files'):
                data_files = ds._data.files
            elif hasattr(ds, 'cache_files'):
                data_files = [f['filename'] for f in ds.cache_files]
            else:
                return []

            for file_path in data_files:
                path_str = str(file_path)
                if path_str.endswith('.parquet'):
                    parquet_files.append(Path(path_str).name)

        except Exception as e:
            warnings.warn(f"Failed to list parquet files for {collection}: {e}")

        return parquet_files

    def clear_cache(self) -> None:
        """Clear the internal parquet file cache."""
        self._parquet_file_cache.clear()
        self._dataset_cache.clear()


# Module-level singleton for convenience
_hf_reader: Optional[HFRowGroupReader] = None


def _get_hf_reader() -> Optional[HFRowGroupReader]:
    """Get or create the module-level HuggingFace reader singleton."""
    global _hf_reader
    if _hf_reader is None:
        try:
            _hf_reader = HFRowGroupReader()
        except ImportError:
            return None
    return _hf_reader


def get_hf_parquet_rowgroup(
    collection: str,
    parquet_filename: str,
    row_group: int,
    columns: Optional[List[str]] = None,
) -> Optional["pa.Table"]:
    """Fetch a rowgroup from a parquet file via HuggingFace datasets.

    This is a convenience function that uses the module-level HuggingFace reader.

    Args:
        collection: Collection name (e.g., "CC-MAIN-2024-10")
        parquet_filename: Name of the parquet file (e.g., "cdx-00000.gz.parquet")
        row_group: Row group index to read
        columns: Optional list of columns to read

    Returns:
        PyArrow Table with the rowgroup data, or None if not available
    """
    reader = _get_hf_reader()
    if reader is None:
        return None
    return reader.read_rowgroup(collection, parquet_filename, row_group, columns)


def get_hf_parquet_schema(
    collection: str,
    parquet_filename: str,
) -> Optional["pa.Schema"]:
    """Get the schema of a parquet file via HuggingFace datasets.

    Args:
        collection: Collection name
        parquet_filename: Name of the parquet file

    Returns:
        PyArrow Schema, or None if not available
    """
    reader = _get_hf_reader()
    if reader is None:
        return None
    return reader.get_parquet_schema(collection, parquet_filename)


def list_hf_parquet_files(collection: str) -> List[str]:
    """List available parquet files for a collection via HuggingFace datasets.

    Args:
        collection: Collection name

    Returns:
        List of parquet filenames available
    """
    reader = _get_hf_reader()
    if reader is None:
        return []
    return reader.list_available_parquet_files(collection)


def hf_datasets_available() -> bool:
    """Check if HuggingFace datasets integration is available.

    Returns:
        True if the datasets library is installed and configured
    """
    if not _have_datasets or not _have_pyarrow:
        return False
    reader = _get_hf_reader()
    return reader is not None


def resolve_parquet_path_with_hf_fallback(
    parquet_path: Path,
    collection: str,
    parquet_relpath: str,
) -> Optional[Union[Path, Tuple[str, str, str]]]:
    """Resolve a parquet path, with HuggingFace fallback.

    This function attempts to resolve a parquet file path locally first.
    If the local file doesn't exist, it returns a tuple indicating that
    the file should be fetched from HuggingFace.

    Args:
        parquet_path: Local parquet file path to check
        collection: Collection name for HuggingFace lookup
        parquet_relpath: Relative path/filename of the parquet

    Returns:
        - Path object if local file exists
        - Tuple ("hf", collection, parquet_relpath) if local file doesn't exist
          and HuggingFace fallback should be used
        - None if neither local nor HuggingFace is available
    """
    # Check local file first
    if parquet_path.exists():
        return parquet_path

    # Check if HuggingFace is available
    if not hf_datasets_available():
        return None

    # Check if the file exists in HuggingFace dataset
    reader = _get_hf_reader()
    if reader is None:
        return None

    # Verify the parquet file is available in HuggingFace
    available_files = reader.list_available_parquet_files(collection)
    parquet_filename = Path(parquet_relpath).name

    if parquet_filename in available_files:
        return ("hf", collection, parquet_filename)

    return None
