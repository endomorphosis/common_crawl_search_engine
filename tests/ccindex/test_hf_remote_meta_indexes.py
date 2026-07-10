from pathlib import Path

from common_crawl_search_engine.ccindex import api
from common_crawl_search_engine.ccindex import hf_datasets_adapter
from common_crawl_search_engine.ccindex.hf_datasets_adapter import HFMetaIndexSQLReader, hf_dataset_resolve_url


def test_hf_dataset_resolve_url_builds_expected_path() -> None:
    url = hf_dataset_resolve_url(
        "Publicus/common_crawl_pointer_indices",
        "2024/CC-MAIN-2024-10/CC-MAIN-2024-10.cc_domain_shards.parquet",
        "main",
    )
    assert url == (
        "https://huggingface.co/datasets/Publicus/common_crawl_pointer_indices/"
        "resolve/main/2024/CC-MAIN-2024-10/CC-MAIN-2024-10.cc_domain_shards.parquet"
    )


def test_iter_domain_records_remote_meta_mode(monkeypatch) -> None:
    class FakeReader:
        def __init__(self, **kwargs):
            self.index_dataset_name = kwargs.get("index_dataset_name") or "idx"
            self.pointers_dataset_name = kwargs.get("pointers_dataset_name") or "ptr"
            self.revision = kwargs.get("revision") or "main"

        def list_collections(self, year=None):
            assert year == "2024"
            return [("2024", "CC-MAIN-2024-10")]

        def parquet_relpaths_for_domain(self, collection, host_rev_prefix, include_subdomains=True):
            assert collection == "CC-MAIN-2024-10"
            assert host_rev_prefix == "com,example"
            assert include_subdomains is True
            return ["cdx-00000.gz.sorted.parquet"]

        def iter_warc_candidates(self, collection, parquet_relpath, host_rev_prefix, *, limit):
            assert collection == "CC-MAIN-2024-10"
            assert parquet_relpath == "cdx-00000.gz.sorted.parquet"
            assert host_rev_prefix == "com,example"
            assert limit == 5
            yield {
                "collection": collection,
                "shard_file": "cdx-00000.gz",
                "url": "https://example.com",
                "timestamp": "20240101000000",
                "status": 200,
                "mime": "text/html",
                "digest": "sha1:abc",
                "warc_filename": "crawl-data/CC-MAIN-2024-10/segments/x/warc/x.warc.gz",
                "warc_offset": 1,
                "warc_length": 2,
                "parquet_path": "https://huggingface.co/datasets/Publicus/common_crawl_pointers_by_collection/resolve/main/2024/CC-MAIN-2024-10/cdx-00000.gz.sorted.parquet",
            }

    monkeypatch.setattr(api, "_HF_AVAILABLE", True)
    monkeypatch.setattr(api, "HFMetaIndexSQLReader", FakeReader)

    stats = {}
    rows = list(
        api.iter_domain_records_via_meta_indexes(
            "example.com",
            parquet_root=Path("/path/that/does/not/exist"),
            master_db=None,
            year="2024",
            max_parquet_files=2,
            max_matches=5,
            per_parquet_limit=5,
            hf_remote_meta=True,
            hf_meta_index_dataset="Publicus/common_crawl_pointer_indices",
            hf_pointer_dataset="Publicus/common_crawl_pointers_by_collection",
            hf_revision="main",
            stats_out=stats,
        )
    )

    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com"
    assert stats["collections_considered"] == 1
    assert str(stats["meta_source"]).startswith("hf-remote:")


def test_search_domain_remote_meta_mode(monkeypatch) -> None:
    def fake_iter(*args, **kwargs):
        assert kwargs["hf_remote_meta"] is True
        assert kwargs["hf_meta_index_dataset"] == "Publicus/common_crawl_pointer_indices"
        assert kwargs["hf_pointer_dataset"] == "Publicus/common_crawl_pointers_by_collection"
        assert kwargs["hf_revision"] == "main"
        stats_out = kwargs["stats_out"]
        stats_out["meta_source"] = "hf-remote:test"
        stats_out["collections_considered"] = 1
        yield {
            "collection": "CC-MAIN-2024-10",
            "url": "https://example.com",
            "timestamp": "20240101000000",
            "status": 200,
            "mime": "text/html",
            "warc_filename": "crawl-data/CC-MAIN-2024-10/segments/x/warc/x.warc.gz",
        }

    monkeypatch.setattr(api, "iter_domain_records_via_meta_indexes", fake_iter)

    res = api.search_domain_via_meta_indexes(
        "example.com",
        hf_remote_meta=True,
        hf_meta_index_dataset="Publicus/common_crawl_pointer_indices",
        hf_pointer_dataset="Publicus/common_crawl_pointers_by_collection",
        hf_revision="main",
    )

    assert res.meta_source == "hf-remote:test"
    assert res.collections_considered == 1
    assert res.emitted == 1
    assert len(res.records) == 1


def test_hf_remote_list_collections_cache_reused_across_readers(monkeypatch) -> None:
    hf_datasets_adapter._HF_COLLECTIONS_CACHE.clear()
    calls = {"count": 0}

    def fake_query(self, sql, params):
        calls["count"] += 1
        return [("2024", "CC-MAIN-2024-10")]

    monkeypatch.setattr(HFMetaIndexSQLReader, "_query_with_retry", fake_query)

    reader_a = HFMetaIndexSQLReader(index_dataset_name="idx", pointers_dataset_name="ptr", revision="main")
    reader_b = HFMetaIndexSQLReader(index_dataset_name="idx", pointers_dataset_name="ptr", revision="main")

    assert reader_a.list_collections(year="2024") == [("2024", "CC-MAIN-2024-10")]
    assert reader_b.list_collections(year="2024") == [("2024", "CC-MAIN-2024-10")]
    assert calls["count"] == 1


def test_hf_remote_parquet_relpaths_cache_reused_across_readers(monkeypatch) -> None:
    hf_datasets_adapter._HF_PARQUET_RELPATHS_CACHE.clear()
    calls = {"count": 0}

    def fake_query(self, sql, params):
        calls["count"] += 1
        return [("cdx-00000.gz.sorted.parquet",)]

    monkeypatch.setattr(HFMetaIndexSQLReader, "_query_with_retry", fake_query)

    reader_a = HFMetaIndexSQLReader(index_dataset_name="idx", pointers_dataset_name="ptr", revision="main")
    reader_b = HFMetaIndexSQLReader(index_dataset_name="idx", pointers_dataset_name="ptr", revision="main")

    assert reader_a.parquet_relpaths_for_domain("CC-MAIN-2024-10", "com,example") == ["cdx-00000.gz.sorted.parquet"]
    assert reader_b.parquet_relpaths_for_domain("CC-MAIN-2024-10", "com,example") == ["cdx-00000.gz.sorted.parquet"]
    assert calls["count"] == 1
