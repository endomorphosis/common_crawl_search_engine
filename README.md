# Common Crawl Search Engine

Utilities for building and querying Common Crawl indexes, with support for
rowgroup slicing, per-collection and per-year rowgroup indexes, and MCP
integrations.

## Features

- Build and query Common Crawl pointer indexes.
- Rowgroup slicing for fast domain/URL lookups.
- Per-collection and per-year rowgroup index support.
- CLI tooling and MCP server integration.
- TCP/IP JSON-RPC MCP server support

## Install

```bash
pip install -e .
```

## CLI Tools

After installation, the following CLI tools are available:

### `ccindex` - Main CLI Interface

Unified command-line interface for all Common Crawl operations:

```bash
# Show all available commands
ccindex --help

# Search operations
ccindex search meta --domain example.com --max-matches 50

# Index management
ccindex index settings-get
ccindex index status --collection CC-MAIN-2024-10

# MCP server operations
ccindex mcp start --host 127.0.0.1 --port 8787
ccindex mcp serve  # stdio mode for MCP clients

# WARC operations
ccindex warc fetch-record --warc-filename <file> --warc-offset <offset> --warc-length <length>
```

### `ccindex-mcp-server` - MCP Server

Run the Model Context Protocol server in stdio or TCP/IP mode:

```bash
# Default stdio mode (for pipe-based communication)
ccindex-mcp-server

# TCP/IP mode with JSON-RPC over HTTP
ccindex-mcp-server --mode tcp --host 127.0.0.1 --port 8787
```

**MCP Server Modes:**
- **stdio** (default): Pipe-based communication for MCP clients
- **tcp**: HTTP server with JSON-RPC endpoint at `POST /mcp`

### `ccindex-dashboard` - Web Dashboard

Run the web dashboard with integrated MCP JSON-RPC endpoint:

```bash
ccindex-dashboard --host 127.0.0.1 --port 8787
```

The dashboard provides:
- Web UI for search and indexing operations
- MCP JSON-RPC endpoint at `POST /mcp`
- RESTful API endpoints

## MCP Server Integration

The package supports the Model Context Protocol (MCP) with JSON-RPC over TCP/IP:

### Starting the MCP Server

```bash
# Option 1: Using ccindex-mcp-server
ccindex-mcp-server --mode tcp --host 127.0.0.1 --port 8787

# Option 2: Using ccindex subcommand
ccindex mcp start --host 127.0.0.1 --port 8787

# Option 3: Using dashboard (includes web UI)
ccindex-dashboard --host 127.0.0.1 --port 8787
```

### Connecting with MCP Client

```python
from common_crawl_search_engine.mcp_client import CcindexMcpClient

# Connect to MCP server
client = CcindexMcpClient(endpoint="http://localhost:8787")

# List available tools
tools = client.list_tools()

# Call a tool
result = client.call_tool("list_collections", {
    "master_db": "/path/to/master.duckdb"
})
```

### Using from Command Line

```bash
# List available MCP tools from a running server
ccindex mcp tools --endpoint http://localhost:8787

# Call an MCP tool
ccindex mcp call --endpoint http://localhost:8787 --tool list_collections --args-json '{}'
```

## Quickstart

```bash
# Example 1: Run the MCP server in TCP/IP mode
ccindex-mcp-server --mode tcp --host 127.0.0.1 --port 8787

# Example 2: Run the dashboard with MCP endpoint
ccindex mcp start --host 127.0.0.1 --port 8787

# Example 3: Search for a domain
ccindex search meta --domain 18f.gov --max-matches 50
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
