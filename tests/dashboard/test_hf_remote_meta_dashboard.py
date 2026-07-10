from __future__ import annotations

from pathlib import Path

import pytest


def test_search_domain_meta_tool_schema_includes_hf_remote_fields() -> None:
    """MCP tool schema should expose HF remote meta fields."""

    from common_crawl_search_engine.dashboard import create_app

    app = create_app(master_db=Path("/storage/ccindex_duckdb/cc_pointers_master/cc_master_index.duckdb"))

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"fastapi.testclient missing: {e}")

    c = TestClient(app)
    r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert r.status_code == 200

    tools = r.json()["result"]["tools"]
    by_name = {t.get("name"): t for t in tools}
    assert "search_domain_meta" in by_name

    schema_props = by_name["search_domain_meta"]["inputSchema"]["properties"]
    assert "hf_remote_meta" in schema_props
    assert "hf_meta_index_dataset" in schema_props
    assert "hf_pointer_dataset" in schema_props
    assert "hf_revision" in schema_props


def test_search_domain_meta_tool_call_forwards_hf_remote_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP search call should forward HF remote args into API search."""

    from common_crawl_search_engine import dashboard

    captured: dict[str, object] = {}

    class FakeResult:
        def __init__(self) -> None:
            self.meta_source = "hf-remote:Publicus/common_crawl_pointer_indices@main"
            self.collections_considered = 3
            self.emitted = 1
            self.elapsed_s = 0.12
            self.records = [{"url": "https://example.com"}]

    def fake_search_domain_via_meta_indexes(domain: str, **kwargs: object) -> FakeResult:
        captured["domain"] = domain
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr(dashboard.api, "search_domain_via_meta_indexes", fake_search_domain_via_meta_indexes)

    app = dashboard.create_app(master_db=Path("/storage/ccindex_duckdb/cc_pointers_master/cc_master_index.duckdb"))

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"fastapi.testclient missing: {e}")

    c = TestClient(app)
    r = c.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_domain_meta",
                "arguments": {
                    "domain": "example.com",
                    "year": "2024",
                    "max_matches": 17,
                    "parquet_root": "/tmp/ignored",
                    "hf_remote_meta": True,
                    "hf_meta_index_dataset": "Publicus/common_crawl_pointer_indices",
                    "hf_pointer_dataset": "Publicus/common_crawl_pointers_by_collection",
                    "hf_revision": "main",
                },
            },
        },
    )
    assert r.status_code == 200

    out = r.json()["result"]
    assert out["meta_source"].startswith("hf-remote:")
    assert captured["domain"] == "example.com"
    kwargs = captured["kwargs"]
    assert kwargs["year"] == "2024"
    assert kwargs["max_matches"] == 17
    assert kwargs["hf_remote_meta"] is True
    assert kwargs["hf_meta_index_dataset"] == "Publicus/common_crawl_pointer_indices"
    assert kwargs["hf_pointer_dataset"] == "Publicus/common_crawl_pointers_by_collection"
    assert kwargs["hf_revision"] == "main"


def test_home_page_contains_hf_remote_controls_and_mode_label() -> None:
    """Home page should render HF remote controls and mode label template."""

    from common_crawl_search_engine.dashboard import create_app

    app = create_app(master_db=Path("/storage/ccindex_duckdb/cc_pointers_master/cc_master_index.duckdb"))

    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"fastapi.testclient missing: {e}")

    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    html = r.text

    assert "id='hf_remote_meta'" in html
    assert "id='hf_meta_index_dataset'" in html
    assert "id='hf_pointer_dataset'" in html
    assert "id='hf_revision'" in html
    assert "mode=<span class='code'>" in html
