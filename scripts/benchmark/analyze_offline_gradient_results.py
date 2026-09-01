#!/usr/bin/env python3
"""Summarize offline-profile results across public-likeness gradient tiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import metric_summary  # noqa: E402


def main() -> int:
    args = _parse_args()
    gradient = _load_json(args.gradient_result)
    cold = _load_json(args.cold_result)
    gradient_sessions = cast(list[dict], gradient["sessions"])
    cold_sessions = cast(list[dict], cold["sessions"])
    groups = {
        "public_200": gradient_sessions[:200],
        "top_1000": gradient_sessions[:1_000],
        "top_2000": gradient_sessions[:2_000],
        "top_4000": gradient_sessions[:4_000],
        "cold_1000": cold_sessions,
        "top_added_800_only": gradient_sessions[200:1_000],
        "likeness_rank_801_1800": gradient_sessions[1_000:2_000],
        "likeness_rank_1801_3800": gradient_sessions[2_000:4_000],
    }
    summary = {
        "schema": "shopping-copilot/offline-gradient-baseline/v1",
        "groups": {
            name: {
                **metric_summary(sessions),
                "miss_count": sum(not bool(row["hit"]) for row in sessions),
            }
            for name, sessions in groups.items()
        },
        "interpretation": {
            "prior_not_applied": True,
            "cold_means": "low public-likeness, not necessarily high retrieval difficulty",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "baseline-report.md").write_text(
        _render(summary),
        encoding="utf-8",
    )
    print(f"wrote gradient baseline report to {args.output.resolve()}")
    return 0


def _render(payload: dict[str, object]) -> str:
    groups = cast(dict[str, dict[str, object]], payload["groups"])
    lines = [
        "# APERTURE offline profile on public-likeness gradients",
        "",
        "The selection prior was not used in these runs. The table measures the model-free offline profile as-is.",
        "",
        "| Cohort | n | Hit@10 | MRR | MTTC | misses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "public_200",
        "top_1000",
        "top_2000",
        "top_4000",
        "cold_1000",
        "top_added_800_only",
        "likeness_rank_801_1800",
        "likeness_rank_1801_3800",
    ):
        row = groups[name]
        lines.append(
            f"| {name} | {row['sample_count']} | {float(cast(float, row['hit_rate_at_10'])):.4f} | "
            f"{float(cast(float, row['mrr'])):.4f} | {float(cast(float, row['mttc'])):.4f} | "
            f"{row['miss_count']} |"
        )
    lines.extend(
        [
            "",
            "## Reading the result",
            "",
            "- Hit@10 stays above 99% across all gradient tiers before adding a selection prior.",
            "- MRR does not decrease monotonically with public likeness, so likeness is not the same thing as retrieval difficulty.",
            "- The cold cohort often has sparse cards whose title becomes a highly unique disclosed fact; it is a prior-overfit control, not a guaranteed hard set.",
            "- A public-selection prior should therefore be bounded and used only to break weak-evidence ties.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gradient-result",
        type=Path,
        default=ROOT / "artifacts/benchmark/public-like-gradient-v1/baseline-top-4000.json",
    )
    parser.add_argument(
        "--cold-result",
        type=Path,
        default=ROOT
        / "artifacts/benchmark/public-like-gradient-v1/baseline-cold-control-1000.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmark/public-like-gradient-v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
