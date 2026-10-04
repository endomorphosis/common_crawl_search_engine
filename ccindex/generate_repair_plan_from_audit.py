#!/usr/bin/env python3
"""Generate a resumable repair/rebuild plan from parquet audit JSONL.

Reads JSONL emitted by audit_parquet_shards.py and produces:
- A machine-readable summary (JSON)
- Per-collection file lists (relative paths)
- Shell scripts with recommended commands to:
  (1) repair missing provenance columns
  (2) normalize row groups / rewrite-if-needed via the orchestrator
  (3) sort any remaining unsorted shards

This intentionally does NOT execute repairs; it is a planner.

Example:
  python src/common_crawl_search_engine/ccindex/generate_repair_plan_from_audit.py \
    --audit-jsonl state/audit_parquet_shards_v2.jsonl \
    --parquet-root /storage/ccindex_parquet/cc_pointers_by_collection \
    --out-dir state/repair_plan_from_audit
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _relpath_under(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _collection_from_path(p: Path) -> Optional[str]:
    # Expected: .../<year>/<collection>/<file>
    try:
        col = p.parent.name
        if col.startswith("CC-MAIN-"):
            return col
        return None
    except Exception:
        return None


@dataclass
class CollectionPlan:
    collection: str
    audited_files: int = 0
    needs_action_files: int = 0
    reasons: Counter = None  # type: ignore[assignment]

    # categorized file lists (store relpaths)
    missing_provenance: List[str] = None  # type: ignore[assignment]
    rowgroup_mismatch: List[str] = None  # type: ignore[assignment]
    codec_mismatch: List[str] = None  # type: ignore[assignment]
    unknown_codec: List[str] = None  # type: ignore[assignment]
    read_error: List[str] = None  # type: ignore[assignment]
    not_sorted_name: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = Counter()
        self.missing_provenance = self.missing_provenance or []
        self.rowgroup_mismatch = self.rowgroup_mismatch or []
        self.codec_mismatch = self.codec_mismatch or []
        self.unknown_codec = self.unknown_codec or []
        self.read_error = self.read_error or []
        self.not_sorted_name = self.not_sorted_name or []


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(x + "\n" for x in items), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate a repair plan from parquet audit JSONL")
    ap.add_argument("--audit-jsonl", required=True, type=Path, help="audit_parquet_shards.py output JSONL")
    ap.add_argument(
        "--parquet-root",
        required=True,
        type=Path,
        help="Root directory that contains parquet shards (used to write relative file lists)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for plan files",
    )
    ap.add_argument(
        "--orchestrator",
        default="src/common_crawl_search_engine/ccindex/cc_pipeline_orchestrator.py",
        help="Path to orchestrator script (default: src/common_crawl_search_engine/ccindex/cc_pipeline_orchestrator.py)",
    )
    ap.add_argument(
        "--repair-script",
        default="src/common_crawl_search_engine/ccindex/repair_legacy_parquet_columns.py",
        help="Path to provenance repair script",
    )
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=int(os.environ.get("CC_SORT_ROW_GROUP_SIZE", "71680")),
        help="Row group size used by recommended commands (default: env CC_SORT_ROW_GROUP_SIZE else 71680)",
    )
    ap.add_argument(
        "--sort-mem-gb",
        type=float,
        default=8.0,
        help="Suggested DuckDB memory per sort worker (GB) for orchestrator commands (default: 8)",
    )
    ap.add_argument(
        "--sort-workers",
        type=int,
        default=1,
        help="Suggested sort-workers for orchestrator commands (default: 1; safer for OOM-prone shards)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Suggested max pipeline workers (default: 8)",
    )
    ap.add_argument(
        "--repair-workers",
        type=int,
        default=4,
        help="Suggested parallel workers for provenance repair (default: 4)",
    )
    ap.add_argument(
        "--repair-temp-dir",
        type=str,
        default=None,
        help="Optional DuckDB temp dir to include in provenance repair commands",
    )
    ap.add_argument(
        "--sort-temp-dir",
        type=str,
        default=None,
        help="Optional sort temp dir to include in orchestrator commands",
    )

    args = ap.parse_args(argv)

    audit_jsonl = args.audit_jsonl.expanduser().resolve()
    parquet_root = args.parquet_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plans: Dict[str, CollectionPlan] = {}
    totals = Counter()

    for rec in _iter_jsonl(audit_jsonl):
        p = Path(str(rec.get("path") or ""))
        col = rec.get("collection_guess") or _collection_from_path(p) or "unknown"
        if col == "unknown":
            # Skip any non-standard paths; the audit already filtered temp dirs.
            continue

        plan = plans.get(col)
        if plan is None:
            plan = CollectionPlan(collection=col)
            plans[col] = plan

        plan.audited_files += 1
        totals["audited_files"] += 1

        needs_action = bool(rec.get("needs_action"))
        if needs_action:
            plan.needs_action_files += 1
            totals["needs_action_files"] += 1

        reasons = rec.get("needs_action_reasons") or []
        for r in reasons:
            plan.reasons[str(r)] += 1
            totals[f"reason:{r}"] += 1

        rel = _relpath_under(p, parquet_root)

        missing_cols = rec.get("missing_required_columns") or []
        if missing_cols:
            plan.missing_provenance.append(rel)

        if "row_group_rows_mismatch" in reasons or "row_group_mb_mismatch" in reasons:
            plan.rowgroup_mismatch.append(rel)

        if "codec_mismatch" in reasons:
            plan.codec_mismatch.append(rel)

        if "unknown_codec" in reasons:
            plan.unknown_codec.append(rel)

        if "read_error" in reasons or rec.get("error"):
            plan.read_error.append(rel)

        name = p.name
        if name and not name.endswith(".sorted.parquet"):
            plan.not_sorted_name.append(rel)

    # Sort collections deterministically.
    collections = sorted(plans.values(), key=lambda x: (-x.needs_action_files, x.collection))

    # Write per-collection lists.
    per_dir = out_dir / "by_collection"
    per_dir.mkdir(parents=True, exist_ok=True)

    for plan in collections:
        cdir = per_dir / plan.collection
        cdir.mkdir(parents=True, exist_ok=True)
        _write_list(cdir / "missing_provenance.txt", sorted(set(plan.missing_provenance)))
        _write_list(cdir / "rowgroup_mismatch.txt", sorted(set(plan.rowgroup_mismatch)))
        _write_list(cdir / "codec_mismatch.txt", sorted(set(plan.codec_mismatch)))
        _write_list(cdir / "unknown_codec.txt", sorted(set(plan.unknown_codec)))
        _write_list(cdir / "read_error.txt", sorted(set(plan.read_error)))
        _write_list(cdir / "not_sorted_name.txt", sorted(set(plan.not_sorted_name)))

        (cdir / "counts.json").write_text(
            json.dumps(
                {
                    "collection": plan.collection,
                    "audited_files": plan.audited_files,
                    "needs_action_files": plan.needs_action_files,
                    "reasons": dict(plan.reasons),
                    "n_missing_provenance": len(set(plan.missing_provenance)),
                    "n_rowgroup_mismatch": len(set(plan.rowgroup_mismatch)),
                    "n_codec_mismatch": len(set(plan.codec_mismatch)),
                    "n_unknown_codec": len(set(plan.unknown_codec)),
                    "n_read_error": len(set(plan.read_error)),
                    "n_not_sorted_name": len(set(plan.not_sorted_name)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # Build command scripts.
    orch = str(args.orchestrator)
    rep = str(args.repair_script)
    repair_temp_arg = f" --temp-dir {args.repair_temp_dir}" if args.repair_temp_dir else ""
    sort_temp_arg = f" --sort-temp-dir {args.sort_temp_dir}" if args.sort_temp_dir else ""

    # Stage 1: provenance repair for collections with missing columns.
    stage1_cols = [p.collection for p in collections if p.missing_provenance]

    # Stage 2: rowgroup normalization via orchestrator rewrite-if-needed.
    # Only include collections with rowgroup mismatch AND no missing provenance (fix those first).
    stage2_cols = [
        p.collection
        for p in collections
        if p.rowgroup_mismatch and not p.missing_provenance
    ]

    # Stage 2b: rowgroup normalization for collections that also had missing provenance
    # (should be run after Stage 1 completes).
    stage2b_cols = [
        p.collection
        for p in collections
        if p.rowgroup_mismatch and p.missing_provenance
    ]

    # Stage 3: sort any non-sorted-named shard files.
    stage3_cols = [p.collection for p in collections if p.not_sorted_name]

    stage1 = out_dir / "commands_stage1_repair_provenance.sh"
    stage2 = out_dir / "commands_stage2_rewrite_rowgroups.sh"
    stage3 = out_dir / "commands_stage3_sort_unsorted.sh"

    stage1_lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    stage1_lines.append("# Stage 1: Repair missing provenance columns (collection, shard_file)")
    stage1_lines.append("# This rewrites only shards missing those columns.")
    stage1_lines.append("")
    for col in stage1_cols:
        stage1_lines.append(
            "python "
            + rep
            + f" --parquet-root /storage/ccindex_parquet --collections {col}"
            + " --overwrite --compression zstd"
            + f" --row-group-size {int(args.row_group_size)}"
            + f" --workers {int(args.repair_workers)}"
            + repair_temp_arg
        )
    stage1_lines.append("")
    stage1.write_text("\n".join(stage1_lines) + "\n", encoding="utf-8")

    stage2_lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    stage2_lines.append("# Stage 2: Normalize row groups (rewrite-if-needed) and rebuild indexes")
    stage2_lines.append("# Uses orchestrator resume; should not download sources when parquet exists.")
    stage2_lines.append("")

    def _orch_cmd(col: str) -> str:
        # Note: rewrite-sorted implies index rebuild for the collection.
        return (
            "python "
            + orch
            + f" --filter {col} --existing-parquet-only --workers {int(args.workers)}"
            + f" --sort-workers {int(args.sort_workers)} --sort-memory-per-worker-gb {float(args.sort_mem_gb)}"
            + f" --sort-row-group-size {int(args.row_group_size)}"
            + sort_temp_arg
            + " --rewrite-sorted-parquet"
        )

    if stage2_cols:
        stage2_lines.append("# Collections with rowgroup mismatch (no missing provenance):")
        for col in stage2_cols:
            stage2_lines.append(_orch_cmd(col))
        stage2_lines.append("")

    if stage2b_cols:
        stage2_lines.append("# Collections with BOTH missing provenance + rowgroup mismatch:")
        stage2_lines.append("# Run Stage 1 first, then run these:")
        for col in stage2b_cols:
            stage2_lines.append(_orch_cmd(col))
        stage2_lines.append("")

    stage2.write_text("\n".join(stage2_lines) + "\n", encoding="utf-8")

    stage3_lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    stage3_lines.append("# Stage 3: Sort any remaining unsorted shards")
    stage3_lines.append("# (Usually there should be none; if present, re-run orchestrator without rewrite.)")
    stage3_lines.append("")
    for col in stage3_cols:
        stage3_lines.append(
            "python "
            + orch
            + f" --filter {col} --existing-parquet-only --workers {int(args.workers)}"
            + f" --sort-workers {int(args.sort_workers)} --sort-memory-per-worker-gb {float(args.sort_mem_gb)}"
            + f" --sort-row-group-size {int(args.row_group_size)}"
            + sort_temp_arg
        )
    stage3_lines.append("")
    stage3.write_text("\n".join(stage3_lines) + "\n", encoding="utf-8")

    # Master summary.
    summary = {
        "ts": _utc_ts(),
        "audit_jsonl": str(audit_jsonl),
        "parquet_root": str(parquet_root),
        "totals": dict(totals),
        "collections": [
            {
                "collection": p.collection,
                "audited_files": p.audited_files,
                "needs_action_files": p.needs_action_files,
                "reasons": dict(p.reasons),
                "n_missing_provenance": len(set(p.missing_provenance)),
                "n_rowgroup_mismatch": len(set(p.rowgroup_mismatch)),
                "n_codec_mismatch": len(set(p.codec_mismatch)),
                "n_unknown_codec": len(set(p.unknown_codec)),
                "n_read_error": len(set(p.read_error)),
                "n_not_sorted_name": len(set(p.not_sorted_name)),
            }
            for p in collections
        ],
        "stages": {
            "stage1_provenance_collections": stage1_cols,
            "stage2_rowgroup_only_collections": stage2_cols,
            "stage2b_rowgroup_after_provenance_collections": stage2b_cols,
            "stage3_sort_collections": stage3_cols,
        },
        "outputs": {
            "commands_stage1": str(stage1),
            "commands_stage2": str(stage2),
            "commands_stage3": str(stage3),
            "by_collection_dir": str(per_dir),
        },
    }

    (out_dir / "plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Convenience TSV for quick review.
    tsv_lines = [
        "collection\taudited\tneeds_action\tmissing_provenance\trowgroup_mismatch\tcodec_mismatch\tunknown_codec\tread_error\tnot_sorted_name",
    ]
    for p in collections:
        tsv_lines.append(
            "\t".join(
                [
                    p.collection,
                    str(p.audited_files),
                    str(p.needs_action_files),
                    str(len(set(p.missing_provenance))),
                    str(len(set(p.rowgroup_mismatch))),
                    str(len(set(p.codec_mismatch))),
                    str(len(set(p.unknown_codec))),
                    str(len(set(p.read_error))),
                    str(len(set(p.not_sorted_name))),
                ]
            )
        )
    (out_dir / "collections.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    # Mark scripts executable (best-effort).
    for sh in (stage1, stage2, stage3):
        try:
            mode = sh.stat().st_mode
            sh.chmod(mode | 0o111)
        except Exception:
            pass

    print(f"Wrote plan to: {out_dir}")
    print(f"  - {stage1}")
    print(f"  - {stage2}")
    print(f"  - {stage3}")
    print(f"  - {out_dir/'plan_summary.json'}")
    print(f"  - {out_dir/'collections.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
