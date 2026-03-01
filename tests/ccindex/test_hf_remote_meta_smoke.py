"""Optional smoke tests for HuggingFace remote meta-index SQL mode.

These tests are intentionally skipped unless explicitly enabled because they
require network access and external HuggingFace availability.
"""

from __future__ import annotations

import os

import pytest

from common_crawl_search_engine.ccindex.hf_datasets_adapter import HFMetaIndexSQLReader


@pytest.mark.skipif(
    os.environ.get("RUN_HF_REMOTE_SMOKE", "").strip().lower() not in {"1", "true", "yes", "on"},
    reason="Set RUN_HF_REMOTE_SMOKE=1 to run networked HuggingFace smoke tests.",
)
def test_hf_remote_meta_list_collections_smoke() -> None:
    reader = HFMetaIndexSQLReader(
        index_dataset_name=os.environ.get("HF_META_INDEX_DATASET_NAME", "Publicus/common_crawl_pointer_indices"),
        pointers_dataset_name=os.environ.get("HF_POINTER_DATASET_NAME", "Publicus/common_crawl_pointers_by_collection"),
        revision=os.environ.get("HF_META_DATASET_REVISION", "main"),
    )

    rows = reader.list_collections(year="2024")
    assert rows
    assert any(coll.startswith("CC-MAIN-2024-") for _, coll in rows)
