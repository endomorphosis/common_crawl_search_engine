# HuggingFace Datasets Integration

This document describes the HuggingFace datasets integration for the Common Crawl Search Engine, which allows fetching rowgroups from HuggingFace datasets as a fallback when local parquet files are not available.

## Overview

The integration provides:

1. **Automatic Fallback**: When local parquet files don't exist, the system automatically attempts to fetch data from HuggingFace datasets
2. **Rowgroup-level Access**: Efficiently reads specific rowgroups from parquet files stored in HuggingFace datasets
3. **Transparent Integration**: Works seamlessly with existing search APIs without requiring code changes

## Dataset

The default dataset is:
- **Name**: `Publicus/common_crawl_pointers_by_collection`
- **Structure**: Organized by year and collection (e.g., `2024/CC-MAIN-2024-10/`)
- **Files**: Parquet files with names like `cdx-00000.gz.parquet`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_DATASET_NAME` | HuggingFace dataset name | `Publicus/common_crawl_pointers_by_collection` |
| `HF_DATASET_REVISION` | Dataset revision/tag | `main` |
| `HF_TOKEN` | HuggingFace API token (for private datasets) | `None` |
| `HF_CACHE_DIR` | Cache directory for downloaded data | `~/.cache/huggingface/datasets` |
| `HF_ENABLE_FALLBACK` | Enable HuggingFace fallback when local files missing | `true` |
| `HF_ENABLE_REMOTE` | Enable remote HuggingFace access | `true` |

### Installation

To use the HuggingFace integration, install the required dependencies:

```bash
pip install datasets pyarrow
```

## Usage

### Automatic Fallback

The integration works automatically. When you search for URLs using the existing API:

```python
from common_crawl_search_engine.ccindex.api import resolve_urls_to_ccindex

# This will automatically use HuggingFace if local files don't exist
results = resolve_urls_to_ccindex(
    urls=["https://example.com/page"],
    parquet_root="/storage/ccindex_parquet",  # Local path (may be empty)
)
```

### Direct HuggingFace Access

For direct access to HuggingFace datasets:

```python
from common_crawl_search_engine.ccindex.hf_datasets_adapter import (
    HFRowGroupReader,
    get_hf_parquet_rowgroup,
)

# Method 1: Using the reader class
reader = HFRowGroupReader()
table = reader.read_rowgroup(
    collection="CC-MAIN-2024-10",
    parquet_filename="cdx-00000.gz.parquet",
    row_group=0,
    columns=["url", "timestamp", "status"],
)

# Method 2: Using the convenience function
table = get_hf_parquet_rowgroup(
    collection="CC-MAIN-2024-10",
    parquet_filename="cdx-00000.gz.parquet",
    row_group=0,
    columns=["url", "timestamp", "status"],
)
```

### Checking Availability

```python
from common_crawl_search_engine.ccindex.hf_datasets_adapter import hf_datasets_available

if hf_datasets_available():
    print("HuggingFace datasets integration is available")
else:
    print("HuggingFace datasets integration is not available")
```

## Architecture

### How It Works

1. **Local First**: The system first attempts to read from local parquet files
2. **HuggingFace Fallback**: If a local file doesn't exist, the system checks if it's available in the HuggingFace dataset
3. **Rowgroup Reading**: Uses PyArrow to read specific rowgroups from the HuggingFace-hosted parquet files
4. **Caching**: HuggingFace datasets library handles caching of downloaded files

### File Resolution Flow

```
Search Request
    ↓
Check Local Parquet File
    ↓
If exists → Read locally
If missing → Check HuggingFace
    ↓
If available in HF → Fetch from HF
If not available → Skip
```

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Search API    │────▶│  Local Parquet   │────▶│  Return Results │
│                 │     │  (if exists)     │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         │ (fallback)
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HFRowGroupReader│────▶│ HuggingFace      │────▶│  Return Results │
│                 │     │  Dataset         │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Performance Considerations

1. **Caching**: The HuggingFace datasets library caches downloaded files locally
2. **Selective Downloads**: Only the required rowgroups are read, not entire files
3. **Lazy Loading**: Parquet files are loaded on-demand when rowgroups are requested

## Troubleshooting

### Common Issues

1. **Import Error**: Install `datasets` and `pyarrow`:
   ```bash
   pip install datasets pyarrow
   ```

2. **Dataset Not Found**: Verify the dataset name and that you have access (for private datasets)

3. **Slow Performance**: Check your HuggingFace cache directory and ensure files are cached locally

### Debug Logging

Enable debug logging to see HuggingFace fallback activity:

```python
import os
os.environ["HF_ENABLE_FALLBACK"] = "true"
# The system will emit events for rowgroup reads including source (local/huggingface)
```

## API Reference

### HFRowGroupReader

Main class for reading rowgroups from HuggingFace datasets.

```python
class HFRowGroupReader:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_remote: bool = True,
    )

    def read_rowgroup(
        self,
        collection: str,
        parquet_filename: str,
        row_group: int,
        columns: Optional[List[str]] = None,
    ) -> Optional[pa.Table]

    def get_parquet_schema(
        self,
        collection: str,
        parquet_filename: str,
    ) -> Optional[pa.Schema]

    def list_available_parquet_files(self, collection: str) -> List[str]

    def clear_cache(self) -> None
```

### Module Functions

```python
# Check if HuggingFace integration is available
def hf_datasets_available() -> bool

# Convenience function to read a rowgroup
def get_hf_parquet_rowgroup(
    collection: str,
    parquet_filename: str,
    row_group: int,
    columns: Optional[List[str]] = None,
) -> Optional[pa.Table]

# Get parquet schema from HuggingFace
def get_hf_parquet_schema(
    collection: str,
    parquet_filename: str,
) -> Optional[pa.Schema]

# List available parquet files
def list_hf_parquet_files(collection: str) -> List[str]
```

## Examples

### Example 1: Basic Search with Fallback

```python
import os
from common_crawl_search_engine.ccindex.api import resolve_urls_to_ccindex

# Set HuggingFace dataset
os.environ["HF_DATASET_NAME"] = "Publicus/common_crawl_pointers_by_collection"

# Search will automatically use HuggingFace if local files don't exist
results = resolve_urls_to_ccindex(
    urls=["https://example.com/page1", "https://example.com/page2"],
    parquet_root="/storage/ccindex_parquet",
)

for url, records in results.items():
    print(f"{url}: {len(records)} records")
    for rec in records:
        print(f"  - Source: {rec.get('source', 'unknown')}")
```

### Example 2: Direct HuggingFace Access

```python
from common_crawl_search_engine.ccindex.hf_datasets_adapter import HFRowGroupReader

reader = HFRowGroupReader(
    dataset_name="Publicus/common_crawl_pointers_by_collection",
    revision="main",
)

# List available files
files = reader.list_available_parquet_files("CC-MAIN-2024-10")
print(f"Available files: {files}")

# Read a specific rowgroup
table = reader.read_rowgroup(
    collection="CC-MAIN-2024-10",
    parquet_filename="cdx-00000.gz.parquet",
    row_group=0,
    columns=["url", "timestamp", "warc_filename", "warc_offset"],
)

if table:
    print(f"Read {table.num_rows} rows")
    print(table.to_pandas().head())
```

### Example 3: Mixed Local and Remote

```python
from common_crawl_search_engine.ccindex.api import resolve_urls_to_ccindex
import os

# Enable HuggingFace fallback
os.environ["HF_ENABLE_FALLBACK"] = "true"

# Search - will use local files when available, HuggingFace when not
results = resolve_urls_to_ccindex(
    urls=["https://example.com/page"],
    parquet_root="/storage/ccindex_parquet",
)

# Results indicate source in the 'source' field
for url, records in results.items():
    for rec in records:
        source = rec.get('source', 'unknown')
        print(f"Record from {source}: {rec['url']}")
```

## Future Enhancements

Potential future improvements:

1. **Streaming Support**: Implement streaming reads for large datasets
2. **Prefetching**: Prefetch likely-to-be-needed parquet files
3. **Batch Downloads**: Batch multiple rowgroup requests
4. **Alternative Backends**: Support for other cloud storage providers
