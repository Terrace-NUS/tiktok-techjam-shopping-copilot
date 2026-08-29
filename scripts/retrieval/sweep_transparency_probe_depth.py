"""Measure disclosure convergence across several semantic Probe depths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.retrieval.diagnose_transparency_transitions import (  # noqa: E402
    _restore_query,
)
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.retrieval import (  # noqa: E402
    HardMaskResolver,
    SemanticModeProbe,
    build_retrieval_evidence_index,
    create_dense_retriever,
    load_bound_transparency_calibration,
)

DEFAULT_INPUT = Path("artifacts/retrieval/qu-to-probe-simulator-other-16x4-v1.json")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_CALIBRATION = Path("config/retrieval/transparency-calibration-v1.json")
DEFAULT_DEPTHS = (20, 40, 80, 160, 320)


def main() -> int:
    args = _parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    release = load_catalog_semantic_release(args.release)
    calibration = load_bound_transparency_calibration(args.calibration)
    retriever = create_dense_retriever(
        index_path=args.dense_index,
        release_dir=args.release,
        catalog_path=args.catalog,
        device=args.device,
        local_files_only=True,
    )
    catalog_source = args.catalog or args.release / "catalog.jsonl"
    evidence = build_retrieval_evidence_index(
        catalog_source,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(retriever.index.parent_asins),
    )
    resolver = HardMaskResolver(
        release=release,
        evidence_index=evidence,
        dense_index=retriever.index,
    )
    mode_probe = SemanticModeProbe(retriever.index)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source["turns"]:
        if item["status"] == "success":
            grouped[item["conversation_id"]].append(item)

    conversations: list[dict[str, Any]] = []
    for conversation_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item["turn"])
        turns: list[dict[str, Any]] = []
        for item in ordered:
            query = _restore_query(
                item["compiled"],
                runtime=source["runtime"],
                category_graph_id=release.category_registry.category_graph_id,
            )
            mask = resolver.resolve(query)
            scores = retriever.score(query.q_sem)
            depths: dict[str, Any] = {}
            for depth in args.depths:
                ranking = retriever.index.rank_scores(
                    scores,
                    top_k=depth,
                    eligible_mask=mask.eligible_mask,
                )
                observation = mode_probe.observe(
                    ranking,
                    probe_k=depth,
                    threshold=calibration.mode_similarity_threshold,
                )
                depths[str(depth)] = {
                    "hit_count": len(observation.hits),
                    "mode_count": len(observation.modes),
                    "mode_coherence": (
                        observation.equal_mode_coherence.debiased_pairwise_cosine
                    ),
                    "listing_coherence": (
                        observation.raw_listing_coherence.debiased_pairwise_cosine
                    ),
                    "hit_ids": [hit.parent_asin for hit in observation.hits],
                }
            turns.append(
                {
                    "turn": item["turn"],
                    "response_shape": item["response_shape"],
                    "eligible_count": len(mask.eligible_parent_asins),
                    "depths": depths,
                }
            )
        conversations.append(
            {
                "conversation_id": conversation_id,
                "scenario": ordered[0]["scenario_type"],
                "turns": turns,
            }
        )

    summary = {
        str(depth): _summarize_depth(conversations, depth=depth) for depth in args.depths
    }
    report = {
        "schema": "shopping-copilot/transparency-probe-depth-sweep/v1",
        "source": str(args.input),
        "depths": list(args.depths),
        "mode_similarity_threshold": calibration.mode_similarity_threshold,
        "summary": summary,
        "conversations": conversations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")
    return 0


def _summarize_depth(conversations: list[dict[str, Any]], *, depth: int) -> dict[str, Any]:
    transitions: list[dict[str, Any]] = []
    key = str(depth)
    for conversation in conversations:
        turns = conversation["turns"]
        for previous, current in zip(turns, turns[1:], strict=False):
            if current["response_shape"] != "attribute_disclosure":
                continue
            previous_metrics = previous["depths"][key]
            current_metrics = current["depths"][key]
            previous_g = previous_metrics["mode_coherence"]
            current_g = current_metrics["mode_coherence"]
            if previous_g is None or current_g is None:
                continue
            delta = current_g - previous_g
            previous_ids = set(previous_metrics["hit_ids"])
            current_ids = set(current_metrics["hit_ids"])
            union = previous_ids | current_ids
            transitions.append(
                {
                    "to_turn": current["turn"],
                    "delta": delta,
                    "jaccard": len(previous_ids & current_ids) / len(union) if union else 1.0,
                    "previous_hit_count": previous_metrics["hit_count"],
                    "current_hit_count": current_metrics["hit_count"],
                }
            )
    values = [item["delta"] for item in transitions]
    return {
        "comparable_disclosures": len(transitions),
        "direction": _directions(values),
        "mean_delta": statistics.fmean(values) if values else None,
        "median_delta": statistics.median(values) if values else None,
        "mean_top_k_jaccard": (
            statistics.fmean(item["jaccard"] for item in transitions) if transitions else None
        ),
        "by_disclosure_turn": {
            str(turn): _turn_summary(
                [item for item in transitions if item["to_turn"] == turn]
            )
            for turn in (2, 3)
        },
    }


def _turn_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    values = [item["delta"] for item in items]
    return {
        "count": len(items),
        "direction": _directions(values),
        "mean_delta": statistics.fmean(values) if values else None,
        "median_delta": statistics.median(values) if values else None,
        "mean_top_k_jaccard": (
            statistics.fmean(item["jaccard"] for item in items) if items else None
        ),
    }


def _directions(values: list[float]) -> dict[str, int]:
    return dict(
        Counter(
            "up" if value > 1e-12 else "down" if value < -1e-12 else "flat"
            for value in values
        )
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transparency Probe depth sweep",
        "",
        "All depths reuse the same saved QU states, hard masks, dense scores, and mode threshold.",
        "Only Top-K changes. Values are raw G_mode before calibration.",
        "",
        "| K | comparable | up | flat | down | mean ΔG | median ΔG | mean Top-K Jaccard |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in report["depths"]:
        item = report["summary"][str(depth)]
        direction = item["direction"]
        lines.append(
            f"| {depth} | {item['comparable_disclosures']} | {direction.get('up', 0)} | "
            f"{direction.get('flat', 0)} | {direction.get('down', 0)} | "
            f"{_fmt(item['mean_delta'])} | {_fmt(item['median_delta'])} | "
            f"{_fmt(item['mean_top_k_jaccard'])} |"
        )
    lines.extend(["", "## By disclosure turn", ""])
    for depth in report["depths"]:
        lines.extend(
            [
                f"### K={depth}",
                "",
                "| to turn | count | up | flat | down | mean ΔG | median ΔG | Jaccard |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for turn, item in report["summary"][str(depth)]["by_disclosure_turn"].items():
            direction = item["direction"]
            lines.append(
                f"| {turn} | {item['count']} | {direction.get('up', 0)} | "
                f"{direction.get('flat', 0)} | {direction.get('down', 0)} | "
                f"{_fmt(item['mean_delta'])} | {_fmt(item['median_delta'])} | "
                f"{_fmt(item['mean_top_k_jaccard'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:+.6f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval/transparency-probe-depth-sweep-v1.json"),
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--depths", type=int, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if any(depth < 2 for depth in args.depths):
        parser.error("all --depths values must be at least 2")
    if len(set(args.depths)) != len(args.depths):
        parser.error("--depths values must be unique")
    args.depths = tuple(args.depths)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
