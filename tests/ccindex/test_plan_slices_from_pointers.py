from __future__ import annotations

import json
from pathlib import Path

from common_crawl_search_engine.ccindex.plan_slices_from_pointers import main as slices_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_plan_slices_from_pointers_dedup_and_merge(tmp_path: Path) -> None:
    pointers = tmp_path / "pointers.jsonl"
    out_plan = tmp_path / "slice_plan.jsonl"
    out_members = tmp_path / "slice_members.jsonl"

    # Two nearby ranges in same WARC should merge; a duplicate should dedupe.
    _write_jsonl(
        pointers,
        [
            {"warc_filename": "crawl/a.warc.gz", "warc_offset": 100, "warc_length": 10},
            {"warc_filename": "crawl/a.warc.gz", "warc_offset": 120, "warc_length": 5},
            {"warc_filename": "crawl/a.warc.gz", "warc_offset": 120, "warc_length": 5},
            {"warc_filename": "crawl/b.warc.gz", "warc_offset": 50, "warc_length": 7},
        ],
    )

    rc = slices_main(
        [
            "--pointers-jsonl",
            str(pointers),
            "--out-slice-plan-jsonl",
            str(out_plan),
            "--out-slice-members-jsonl",
            str(out_members),
            "--max-slice-bytes",
            "1000",
            "--max-gap-bytes",
            "1000",
            "--min-slice-bytes",
            "0",
            "--progress-every-warcs",
            "0",
        ]
    )
    assert rc == 0

    plan_lines = [json.loads(l) for l in out_plan.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Expect one slice for a.warc.gz (merged) + one for b.warc.gz.
    assert len(plan_lines) == 2

    members_lines = [json.loads(l) for l in out_members.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Duplicate pointer should be deduped; expect 3 members total.
    assert len(members_lines) == 3
