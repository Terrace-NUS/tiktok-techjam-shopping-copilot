"""Compute dense-vector semantic coherence over Lexical Probe Top-K results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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
    LexicalProbe,
    build_retrieval_evidence_index,
    compute_catalog_mean,
    compute_probe_coherence,
    load_bound_transparency_calibration,
    load_dense_index,
    load_product_documents,
)

DEFAULT_INPUT = Path("artifacts/retrieval/qu-to-probe-simulator-other-16x4-v1.json")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_CALIBRATION = Path("config/retrieval/transparency-calibration-v1.json")


def main() -> int:
    args = _parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    release = load_catalog_semantic_release(args.release)
    calibration = load_bound_transparency_calibration(args.calibration)
    index = load_dense_index(
        args.dense_index,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )
    catalog_source = args.catalog or args.release / "catalog.jsonl"
    documents = load_product_documents(
        catalog_source,
        expected_parent_asins=set(index.parent_asins),
    )
    lexical = LexicalProbe(documents, probe_k=args.probe_k)
    evidence = build_retrieval_evidence_index(
        catalog_source,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(index.parent_asins),
    )
    resolver = HardMaskResolver(release=release, evidence_index=evidence, dense_index=index)
    catalog_mean = compute_catalog_mean(index.vectors)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source["turns"]:
        if item["status"] == "success":
            grouped[item["conversation_id"]].append(item)

    conversations: list[dict[str, Any]] = []
    for conversation_id, items in sorted(grouped.items()):
        turns: list[dict[str, Any]] = []
        for item in sorted(items, key=lambda value: value["turn"]):
            query = _restore_query(
                item["compiled"],
                runtime=source["runtime"],
                category_graph_id=release.category_registry.category_graph_id,
            )
            mask = resolver.resolve(query)
            observation = lexical.observe(
                query.q_lex,
                eligible_parent_asins=mask.eligible_parent_asins,
            )
            geometry = _semantic_geometry(
                [hit.parent_asin for hit in observation.hits],
                index=index,
                catalog_mean=catalog_mean,
                threshold=args.mode_threshold,
            )
            turns.append(
                {
                    "turn": item["turn"],
                    "response_shape": item["response_shape"],
                    "q_lex": query.q_lex,
                    "eligible_count": len(mask.eligible_parent_asins),
                    "lexical_available": observation.available,
                    "lexical_reason": observation.reason,
                    "lexical_hit_count": len(observation.hits),
                    "lexical_hit_ids": [hit.parent_asin for hit in observation.hits],
                    "mode_count": geometry["mode_count"],
                    "listing_coherence": geometry["listing_coherence"],
                    "mode_coherence": geometry["mode_coherence"],
                    "experimental_mapped_ct": _map_certainty(
                        geometry["mode_coherence"],
                        low=calibration.low_anchor,
                        high=calibration.high_anchor,
                    ),
                    "dense_mode_coherence": item["probe"]["mode_coherence"],
                    "dense_ct": item["probe"]["certainty"],
                }
            )
        conversations.append(
            {
                "conversation_id": conversation_id,
                "scenario": items[0]["scenario_type"],
                "turns": turns,
            }
        )

    transitions = _transitions(conversations)
    report = {
        "schema": "shopping-copilot/lexical-result-semantic-coherence/v1",
        "source": str(args.input),
        "probe_k": args.probe_k,
        "mode_threshold": args.mode_threshold,
        "warning": (
            "experimental_mapped_ct reuses Dense-route calibration only for scale inspection; "
            "directional raw mode coherence is the valid comparison"
        ),
        "summary": _summary(conversations, transitions),
        "transitions": transitions,
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
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {markdown_path}")
    return 0


def _semantic_geometry(
    parent_asins: list[str],
    *,
    index: Any,
    catalog_mean: Any,
    threshold: float,
) -> dict[str, Any]:
    vectors = [
        np.asarray(index.vectors[index.row_index(parent_asin)], dtype=np.float64)
        for parent_asin in parent_asins
    ]
    listing = compute_probe_coherence(
        np.asarray(vectors, dtype=np.float64),
        catalog_mean,
    )
    leaders: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    members: list[list[np.ndarray[Any, np.dtype[np.float64]]]] = []
    for vector in vectors:
        selected: int | None = None
        selected_similarity = -1.0
        for index_value, leader in enumerate(leaders):
            similarity = float(round(min(1.0, max(-1.0, float(np.dot(vector, leader)))), 6))
            if similarity >= threshold and similarity > selected_similarity:
                selected = index_value
                selected_similarity = similarity
        if selected is None:
            leaders.append(vector)
            members.append([vector])
        else:
            members[selected].append(vector)

    centroids = []
    for mode_members in members:
        mean = np.mean(np.stack(mode_members), axis=0, dtype=np.float64)
        centroids.append(mean / np.linalg.norm(mean))
    mode = compute_probe_coherence(
        np.asarray(centroids, dtype=np.float64),
        catalog_mean,
    )
    return {
        "mode_count": len(centroids),
        "listing_coherence": listing.debiased_pairwise_cosine,
        "mode_coherence": mode.debiased_pairwise_cosine,
    }


def _transitions(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for conversation in conversations:
        turns = conversation["turns"]
        for previous, current in zip(turns, turns[1:], strict=False):
            if current["response_shape"] != "attribute_disclosure":
                continue
            previous_g = previous["mode_coherence"]
            current_g = current["mode_coherence"]
            dense_previous = previous["dense_mode_coherence"]
            dense_current = current["dense_mode_coherence"]
            previous_ids = set(previous["lexical_hit_ids"])
            current_ids = set(current["lexical_hit_ids"])
            union = previous_ids | current_ids
            result.append(
                {
                    "conversation_id": conversation["conversation_id"],
                    "scenario": conversation["scenario"],
                    "from_turn": previous["turn"],
                    "to_turn": current["turn"],
                    "q_lex_changed": previous["q_lex"] != current["q_lex"],
                    "lexical_delta": _delta(previous_g, current_g),
                    "dense_delta": _delta(dense_previous, dense_current),
                    "lexical_top_k_jaccard": (
                        len(previous_ids & current_ids) / len(union) if union else 1.0
                    ),
                    "previous_lexical_hit_count": previous["lexical_hit_count"],
                    "current_lexical_hit_count": current["lexical_hit_count"],
                }
            )
    return result


def _summary(
    conversations: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> dict[str, Any]:
    turns = [turn for conversation in conversations for turn in conversation["turns"]]
    comparable = [item for item in transitions if item["lexical_delta"] is not None]
    qlex_changed = [item for item in comparable if item["q_lex_changed"]]
    qlex_unchanged = [item for item in comparable if not item["q_lex_changed"]]
    return {
        "turn_count": len(turns),
        "lexical_available_count": sum(item["lexical_available"] for item in turns),
        "mode_coherence_available_count": sum(
            item["mode_coherence"] is not None for item in turns
        ),
        "comparable_attribute_disclosures": len(comparable),
        "all_disclosure_direction": _directions(comparable, "lexical_delta"),
        "all_disclosure_mean_delta": _mean(comparable, "lexical_delta"),
        "qlex_changed_count": len(qlex_changed),
        "qlex_changed_direction": _directions(qlex_changed, "lexical_delta"),
        "qlex_changed_mean_delta": _mean(qlex_changed, "lexical_delta"),
        "qlex_unchanged_count": len(qlex_unchanged),
        "qlex_unchanged_direction": _directions(qlex_unchanged, "lexical_delta"),
        "dense_direction_on_same_comparable_transitions": _directions(
            comparable,
            "dense_delta",
        ),
        "mean_lexical_top_k_jaccard": statistics.fmean(
            item["lexical_top_k_jaccard"] for item in comparable
        )
        if comparable
        else None,
        "by_disclosure_turn": {
            str(turn): {
                "count": len(selected),
                "direction": _directions(selected, "lexical_delta"),
                "mean_delta": _mean(selected, "lexical_delta"),
            }
            for turn in (2, 3)
            if (selected := [item for item in comparable if item["to_turn"] == turn])
        },
    }


def _directions(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [item[field] for item in items if item[field] is not None]
    return dict(
        Counter(
            "up" if value > 1e-12 else "down" if value < -1e-12 else "flat"
            for value in values
        )
    )


def _mean(items: list[dict[str, Any]], field: str) -> float | None:
    values = [item[field] for item in items if item[field] is not None]
    return statistics.fmean(values) if values else None


def _delta(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return current - previous


def _map_certainty(value: float | None, *, low: float, high: float) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dense-vector coherence over Lexical Probe results",
        "",
        "> Absolute mapped C_t values are experimental because the frozen anchors were calibrated on Dense-route selection. Raw G_mode direction is the valid comparison.",
        "",
        "## Summary",
        "",
        f"- Turns: {summary['turn_count']}",
        f"- Lexical available: {summary['lexical_available_count']}",
        f"- Semantic mode coherence available: {summary['mode_coherence_available_count']}",
        f"- Comparable disclosures: {summary['comparable_attribute_disclosures']}",
        f"- Lexical-result direction: `{summary['all_disclosure_direction']}`",
        f"- Mean lexical-result ΔG: `{summary['all_disclosure_mean_delta']}`",
        f"- Dense-result direction on same transitions: `{summary['dense_direction_on_same_comparable_transitions']}`",
        f"- q_lex changed: {summary['qlex_changed_count']} `{summary['qlex_changed_direction']}`",
        f"- q_lex unchanged: {summary['qlex_unchanged_count']} `{summary['qlex_unchanged_direction']}`",
        f"- Mean Lexical Top-K Jaccard: `{summary['mean_lexical_top_k_jaccard']}`",
        "",
        "## Every disclosure",
        "",
        "| Task / transition | scenario | q_lex changed | Lexical ΔG | Dense ΔG | Lexical Jaccard | hits before→after |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in report["transitions"]:
        lines.append(
            f"| `{item['conversation_id']} {item['from_turn']}→{item['to_turn']}` | "
            f"{item['scenario']} | {'yes' if item['q_lex_changed'] else 'no'} | "
            f"{_fmt(item['lexical_delta'])} | {_fmt(item['dense_delta'])} | "
            f"{_fmt(item['lexical_top_k_jaccard'])} | "
            f"{item['previous_lexical_hit_count']}→{item['current_lexical_hit_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4f}"


def _markdown(report: dict[str, Any]) -> str:
    """Render an ASCII-safe review report for cross-platform ZIP consumers."""

    summary = report["summary"]
    lines = [
        "# Dense-vector coherence over Lexical Probe results",
        "",
        "> Absolute mapped C_t values are experimental because the frozen anchors were calibrated on Dense-route selection. Raw G_mode direction is the valid comparison.",
        "",
        "## Summary",
        "",
        f"- Turns: {summary['turn_count']}",
        f"- Lexical available: {summary['lexical_available_count']}",
        f"- Semantic mode coherence available: {summary['mode_coherence_available_count']}",
        f"- Comparable disclosures: {summary['comparable_attribute_disclosures']}",
        f"- Lexical-result direction: `{summary['all_disclosure_direction']}`",
        f"- Mean lexical-result delta G: `{summary['all_disclosure_mean_delta']}`",
        f"- Dense-result direction on same transitions: `{summary['dense_direction_on_same_comparable_transitions']}`",
        f"- q_lex changed: {summary['qlex_changed_count']} `{summary['qlex_changed_direction']}`",
        f"- q_lex unchanged: {summary['qlex_unchanged_count']} `{summary['qlex_unchanged_direction']}`",
        f"- Mean Lexical Top-K Jaccard: `{summary['mean_lexical_top_k_jaccard']}`",
        "",
        "## Every disclosure",
        "",
        "| Task / transition | scenario | q_lex changed | Lexical delta G | Dense delta G | Lexical Jaccard | hits before->after |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in report["transitions"]:
        lines.append(
            f"| `{item['conversation_id']} {item['from_turn']}->{item['to_turn']}` | "
            f"{item['scenario']} | {'yes' if item['q_lex_changed'] else 'no'} | "
            f"{_fmt_review(item['lexical_delta'])} | {_fmt_review(item['dense_delta'])} | "
            f"{_fmt_review(item['lexical_top_k_jaccard'])} | "
            f"{item['previous_lexical_hit_count']}->{item['current_lexical_hit_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_review(value: float | None) -> str:
    return "--" if value is None else f"{value:+.4f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/retrieval/lexical-semantic-coherence-v1.json"),
    )
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--probe-k", type=int, default=80)
    parser.add_argument("--mode-threshold", type=float, default=0.94)
    args = parser.parse_args()
    if args.probe_k < 2:
        parser.error("--probe-k must be at least 2")
    if not 0.0 <= args.mode_threshold <= 1.0:
        parser.error("--mode-threshold must be between zero and one")
    if not math.isfinite(args.mode_threshold):
        parser.error("--mode-threshold must be finite")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
