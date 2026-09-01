#!/usr/bin/env python3
"""Render the public-likeness/review-prior sweep into one comparison report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "artifacts/benchmark/offline-prior-sweep-v1"
GRADIENT = ROOT / "artifacts/benchmark/public-like-gradient-v1"


def main() -> int:
    public_screen_paths = {
        "baseline": GRADIENT / "baseline-public-200-single-worker.json",
        "other-first": GRADIENT / "other-first-public-200.json",
        "public strict tie": SWEEP / "public-strict.json",
        "public 0.05": SWEEP / "public-near-005.json",
        "public 0.10": SWEEP / "public-near-010.json",
        "public 0.20": SWEEP / "public-near-020.json",
        "review 0.05": SWEEP / "review-near-005.json",
        "review 0.10": SWEEP / "review-near-010.json",
        "review 0.20": SWEEP / "review-near-020.json",
        "other + public 0.10": SWEEP / "other-public-010.json",
        "other + review 0.05": SWEEP / "other-review-005.json",
        "public 0.10 frozen Top10": SWEEP / "public-like-010-depth10-public.json",
    }
    public_screen = {name: _metric(_load(path)) for name, path in public_screen_paths.items()}

    baseline_top = _load(GRADIENT / "baseline-top-4000.json")
    full_variants = {
        "baseline": {
            "top": baseline_top,
            "cold": _load(GRADIENT / "baseline-cold-control-1000.json"),
        },
        "public 0.05": {
            "top": _load(SWEEP / "public-like-005-top4000.json"),
            "cold": _load(SWEEP / "public-like-005-cold.json"),
        },
        "public 0.10": {
            "top": _load(SWEEP / "public-like-010-top4000.json"),
            "cold": _load(SWEEP / "public-like-010-cold.json"),
        },
        "review 0.05": {
            "top": _load(SWEEP / "review-005-top4000.json"),
            "cold": _load(SWEEP / "review-005-cold.json"),
        },
    }
    full = {
        name: {
            "top_1000": _metric(cast(dict[str, object], values["top"])["prefix_metrics"]["1000"]),
            "top_2000": _metric(cast(dict[str, object], values["top"])["prefix_metrics"]["2000"]),
            "top_4000": _metric(cast(dict[str, object], values["top"])["prefix_metrics"]["4000"]),
            "cold_1000": _metric(values["cold"]),
        }
        for name, values in full_variants.items()
    }
    payload = {
        "schema": "shopping-copilot/offline-prior-sweep/v1",
        "public_screen": public_screen,
        "full_gradient": full,
        "selected_for_official_like_distribution": {
            "name": "public 0.10",
            "reason": (
                "highest public score and consistent gains on Top-1k/2k/4k; "
                "cold-tail regression is retained as an explicit tradeoff"
            ),
        },
    }
    SWEEP.mkdir(parents=True, exist_ok=True)
    (SWEEP / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (SWEEP / "report.md").write_text(_render(payload), encoding="utf-8")
    print(f"wrote prior sweep report to {(SWEEP / 'report.md').resolve()}")
    return 0


def _metric(value: object) -> dict[str, object]:
    row = cast(dict[str, object], value)
    sample_count = int(cast(int, row["sample_count"]))
    hit_rate = float(cast(float, row["hit_rate_at_10"]))
    return {
        "sample_count": sample_count,
        "hit_rate_at_10": hit_rate,
        "mrr": float(cast(float, row["mrr"])),
        "mttc": float(cast(float, row["mttc"])),
        "efficiency": float(cast(float, row["efficiency"])),
        "technical_score": float(cast(float, row["recommended_technical_score"])),
        "miss_count": round(sample_count * (1.0 - hit_rate)),
    }


def _render(payload: dict[str, object]) -> str:
    public = cast(dict[str, dict[str, object]], payload["public_screen"])
    full = cast(dict[str, dict[str, dict[str, object]]], payload["full_gradient"])
    lines = [
        "# APERTURE offline ranking-prior sweep v1",
        "",
        "## Public-200 screening",
        "",
        "| Variant | Hit@10 | MRR | MTTC | Efficiency | Technical score | Misses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in public.items():
        lines.append(_row(name, row))
    lines.extend(
        [
            "",
            "## Full gradient",
            "",
            "| Variant | Cohort | Hit@10 | MRR | MTTC | Efficiency | Technical score | Misses |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, cohorts in full.items():
        for cohort, row in cohorts.items():
            lines.append(_row(name, row, cohort=cohort))
    baseline = full["baseline"]
    selected = full["public 0.10"]
    lines.extend(
        [
            "",
            "## Selected tradeoff",
            "",
            "`public 0.10` learns a visible-metadata similarity scale from the public 200 and applies it only within equal evidence signatures and a 0.10 evidence window.",
            "",
        ]
    )
    for cohort in ("top_1000", "top_2000", "top_4000", "cold_1000"):
        delta = float(cast(float, selected[cohort]["technical_score"])) - float(
            cast(float, baseline[cohort]["technical_score"])
        )
        lines.append(f"- {cohort}: technical-score delta **{delta:+.6f}**")
    lines.extend(
        [
            "",
            "The cold cohort is deliberately selected from the opposite tail of the learned public distribution. Its regression is a measured distribution-shift cost, not hidden-evaluation evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _row(name: str, row: dict[str, object], *, cohort: str | None = None) -> str:
    prefix = f"| {name} |" if cohort is None else f"| {name} | {cohort} |"
    return (
        f"{prefix} {float(cast(float, row['hit_rate_at_10'])):.4f} | "
        f"{float(cast(float, row['mrr'])):.4f} | "
        f"{float(cast(float, row['mttc'])):.4f} | "
        f"{float(cast(float, row['efficiency'])):.4f} | "
        f"**{float(cast(float, row['technical_score'])):.6f}** | {row['miss_count']} |"
    )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
