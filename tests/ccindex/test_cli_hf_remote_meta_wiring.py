from __future__ import annotations

import pytest


def test_cli_search_meta_forwards_hf_remote_args(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`search meta` should pass HF remote flags directly into API search."""

    from common_crawl_search_engine import cli

    captured: dict[str, object] = {}

    class FakeResult:
        def __init__(self) -> None:
            self.meta_source = "hf-remote:Publicus/common_crawl_pointer_indices@main"
            self.collections_considered = 2
            self.emitted = 1
            self.elapsed_s = 0.25
            self.records = [{"url": "https://example.com", "status": 200}]

    def fake_search(domain: str, **kwargs: object) -> FakeResult:
        captured["domain"] = domain
        captured["kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr(cli.api, "search_domain_via_meta_indexes", fake_search)

    rc = cli.main(
        [
            "search",
            "meta",
            "--domain",
            "example.com",
            "--hf-remote-meta",
            "--hf-meta-index-dataset",
            "Publicus/common_crawl_pointer_indices",
            "--hf-pointer-dataset",
            "Publicus/common_crawl_pointers_by_collection",
            "--hf-revision",
            "main",
            "--max-matches",
            "10",
            "--stats",
        ]
    )

    assert rc == 0

    assert captured["domain"] == "example.com"
    kwargs = captured["kwargs"]
    assert kwargs["hf_remote_meta"] is True
    assert kwargs["hf_meta_index_dataset"] == "Publicus/common_crawl_pointer_indices"
    assert kwargs["hf_pointer_dataset"] == "Publicus/common_crawl_pointers_by_collection"
    assert kwargs["hf_revision"] == "main"
    assert kwargs["max_matches"] == 10

    out = capsys.readouterr()
    assert "https://example.com" in out.out
    assert "meta_source=hf-remote:" in out.err
