from __future__ import annotations

from typing import Optional


def test_fetch_warc_record_ranges_sliced_splits_and_merges(monkeypatch):
    """Ensure the sliced range fetch merges close-by ranges and splits correctly.

    This is a pure unit test (no network). We monkeypatch the internal range getter
    to return bytes from a synthetic "remote" blob.
    """

    from common_crawl_search_engine.ccindex import api

    # Synthetic remote WARC bytes (not a real WARC; slicing doesn't care).
    remote = bytearray(b"." * 2000)
    remote[100:110] = b"A" * 10
    remote[300:320] = b"B" * 20
    remote[800:805] = b"C" * 5

    calls = []

    def fake_range_get_cached(
        *,
        url: str,
        start: int,
        end_inclusive: int,
        timeout_s: float,
        cache_dir,
        cache_max_bytes: int,
        cache_max_item_bytes: int,
    ):
        calls.append((start, end_inclusive))
        if start < 0 or end_inclusive >= len(remote):
            return 416, None, "out_of_range"
        return 206, bytes(remote[start : end_inclusive + 1]), None

    monkeypatch.setattr(api, "_http_range_get_cached", fake_range_get_cached)

    data_by, err_by = api.fetch_warc_record_ranges_sliced(
        warc_filename="CC-MAIN-TEST.warc.gz",
        ranges=[(100, 10), (300, 20), (800, 5)],
        prefix="https://example.invalid/",
        max_slice_bytes=500,
        max_gap_bytes=1000,  # big enough to merge first two into one slice
        cache_dir=None,
    )

    assert not err_by
    assert data_by[(100, 10)] == b"A" * 10
    assert data_by[(300, 20)] == b"B" * 20
    assert data_by[(800, 5)] == b"C" * 5

    # Should be 2 slice calls: [100..319] and [800..804].
    assert len(calls) == 2
    assert calls[0] == (100, 319)
    assert calls[1] == (800, 804)
