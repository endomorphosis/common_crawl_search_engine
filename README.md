# Common Crawl Search Engine

Utilities for building and querying Common Crawl indexes, with support for
rowgroup slicing, per-collection and per-year rowgroup indexes, and MCP
integrations.

## Features

- Build and query Common Crawl pointer indexes.
- Rowgroup slicing for fast domain/URL lookups.
- Per-collection and per-year rowgroup index support.
- CLI tooling and MCP server integration.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
# Example: run the MCP server
python -m common_crawl_search_engine.mcp_server
```

## Configuration

Common environment variables:

- BRAVE_RESOLVE_ROWGROUP_SLICE_MODE: auto|on|off
- BRAVE_RESOLVE_ROWGROUP_INDEX_DIR: per-collection rowgroup index dir
- BRAVE_RESOLVE_ROWGROUP_YEAR_DIR: per-year rowgroup index dir
- BRAVE_RESOLVE_ROWGROUP_WORKERS: rowgroup scan worker count
- BRAVE_RESOLVE_ROWGROUP_SEGMENT_SOURCE: fastest|auto|year|collection

## Tests

```bash
pytest
```

## Benchmarks

See benchmarks under benchmarks/ccindex in the parent repo.

## License

MIT
