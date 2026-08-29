"""Run the frozen Intent Volume v1 runtime on saved real DeepSeek sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.query_compiler import (  # noqa: E402
    COMPILED_QUERY_SCHEMA,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
    DiversityDirective,
)
from shopping_copilot.retrieval import (  # noqa: E402
    IntentTransparencyEstimate,
    IntentVolumeEstimator,
    IntentVolumePolicy,
    SentenceTransformerTextEmbedder,
    build_retrieval_evidence_index,
    load_catalog_density,
    load_dense_index,
)
from shopping_copilot.retrieval.hard_mask import HardMaskResolver  # noqa: E402
from shopping_copilot.session_context import (  # noqa: E402
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    SemanticPolarity,
)

DEFAULT_EVALUATION = Path("artifacts/retrieval/qu-to-probe-intent-space-natural-v3.json")
DEFAULT_EXPERIMENT = Path("artifacts/retrieval/fuzzy-intent-volume-natural-v3.json")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_DENSITY = Path("artifacts/retrieval/intent-volume-density-v0.npz")
DEFAULT_OUTPUT = Path("artifacts/retrieval/intent-transparency-runtime-v1-demo.json")
DEFAULT_MARKDOWN = Path("artifacts/retrieval/intent-transparency-runtime-v1-demo.md")
DEFAULT_CONVERSATIONS = (
    "n17_progressive_trail_shoes",
    "b01_release_heel_constraints",
    "s03_sort_only",
    "o01_shoes_to_earrings",
)
EXPERIMENT_CONFIGURATION = "soft_hybrid_d0.025_q0.850_m0.060_h0.010"


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    evaluation = _load_json(args.evaluation)
    experiment = _load_json(args.experiment)
    release = load_catalog_semantic_release(args.release)
    dense = load_dense_index(
        args.dense_index,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )
    evidence = build_retrieval_evidence_index(
        args.release / "catalog.jsonl",
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(dense.parent_asins),
    )
    resolver = HardMaskResolver(
        release=release,
        evidence_index=evidence,
        dense_index=dense,
    )
    policy = IntentVolumePolicy()
    density = load_catalog_density(
        args.density_cache,
        dense_index=dense,
        temperature=policy.density_temperature,
    )
    initialization_started = time.perf_counter()
    estimator = IntentVolumeEstimator(
        dense_index=dense,
        embedder=SentenceTransformerTextEmbedder(
            dense.manifest.embedding,
            device=args.device,
            local_files_only=True,
        ),
        hard_mask_resolver=resolver,
        density=density,
        policy=policy,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    experiment_rows = {
        (str(row["conversation_id"]), int(row["turn"])): row
        for row in cast(list[dict[str, object]], experiment["turns"])
    }
    available_conversation_ids = {
        str(turn["conversation_id"]) for turn in cast(list[dict[str, object]], evaluation["turns"])
    }
    selected = (
        available_conversation_ids
        if args.all_conversations
        else set(args.conversation_id or DEFAULT_CONVERSATIONS)
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for turn in cast(list[dict[str, object]], evaluation["turns"]):
        conversation_id = str(turn["conversation_id"])
        if conversation_id in selected:
            grouped[conversation_id].append(turn)

    conversations = []
    volume_absolute_errors: list[float] = []
    volume_relative_errors: list[float] = []
    transparency_errors: list[float] = []
    estimate_latencies_ms: list[float] = []
    parity_failure_count = 0
    for conversation_id in sorted(selected):
        previous: IntentTransparencyEstimate | None = None
        rows = []
        for turn in sorted(grouped.get(conversation_id, []), key=lambda item: int(item["turn"])):
            if turn.get("status") != "success":
                rows.append(
                    {
                        "turn": turn["turn"],
                        "user_message": turn["user_message"],
                        "status": turn["status"],
                    }
                )
                continue
            intent = _decode_intent(_mapping(turn, "final_intent"))
            compiled = _decode_compiled(_mapping(turn, "compiled"))
            goal_switched = _explicit_goal_switch(turn, previous=previous, current=intent)
            estimate_started = time.perf_counter()
            estimate = estimator.estimate(
                session_id=f"demo/{conversation_id}",
                intent=intent,
                compiled=compiled,
                previous=previous,
                goal_switched=goal_switched,
            )
            estimate_latency_ms = (time.perf_counter() - estimate_started) * 1_000
            estimate_latencies_ms.append(estimate_latency_ms)
            previous = estimate
            experiment_row = experiment_rows[(conversation_id, int(turn["turn"]))]
            observed = _mapping(experiment_row, "configurations")[EXPERIMENT_CONFIGURATION]
            if type(observed) is not dict:
                raise ValueError("experiment configuration must be an object")
            expected_volume = float(observed["remaining_volume"])
            expected_transparency = float(observed["transparency"])
            volume_error = abs((estimate.remaining_intent_volume or 0.0) - expected_volume)
            volume_relative_error = volume_error / max(abs(expected_volume), 1e-12)
            transparency_error = abs((estimate.transparency or 0.0) - expected_transparency)
            parity_ok = (
                volume_error <= 1e-3 or volume_relative_error <= 1e-4
            ) and transparency_error <= 1e-6
            if not parity_ok:
                parity_failure_count += 1
            volume_absolute_errors.append(volume_error)
            volume_relative_errors.append(volume_relative_error)
            transparency_errors.append(transparency_error)
            rows.append(
                {
                    "turn": turn["turn"],
                    "user_message": turn["user_message"],
                    "status": "success",
                    "estimate_latency_ms": estimate_latency_ms,
                    "estimate": estimate.as_payload(),
                    "experiment_comparison": {
                        "configuration": EXPERIMENT_CONFIGURATION,
                        "remaining_volume_absolute_error": volume_error,
                        "remaining_volume_relative_error": volume_relative_error,
                        "transparency_absolute_error": transparency_error,
                        "parity_ok": parity_ok,
                    },
                }
            )
        conversations.append(
            {
                "conversation_id": conversation_id,
                "turns": rows,
            }
        )

    report = {
        "schema": "shopping-copilot/intent-transparency-runtime-demo/v1",
        "inputs": {
            "evaluation": str(args.evaluation.resolve()),
            "experiment": str(args.experiment.resolve()),
            "dense_index": str(args.dense_index.resolve()),
            "semantic_release": str(args.release.resolve()),
            "density_cache": str(args.density_cache.resolve()),
            "conversation_ids": sorted(selected),
        },
        "runtime": {
            "device": args.device,
            "elapsed_seconds": time.perf_counter() - started,
            "model_initialization_seconds": initialization_seconds,
            "mean_estimate_latency_ms": (
                sum(estimate_latencies_ms) / len(estimate_latencies_ms)
                if estimate_latencies_ms
                else 0.0
            ),
            "max_estimate_latency_ms": max(estimate_latencies_ms, default=0.0),
            "max_volume_absolute_error": max(volume_absolute_errors, default=0.0),
            "max_volume_relative_error": max(volume_relative_errors, default=0.0),
            "max_transparency_absolute_error": max(transparency_errors, default=0.0),
            "parity_failure_count": parity_failure_count,
        },
        "policy": {
            "policy_id": policy.policy_id,
            "mapping_id": policy.mapping_id,
            "density_temperature": policy.density_temperature,
            "membership_quantile": policy.membership_quantile,
            "membership_temperature": policy.membership_temperature,
            "hard_mismatch_floor": policy.hard_mismatch_floor,
            "soft_preference_exponent": policy.soft_preference_exponent,
        },
        "conversations": conversations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["runtime"], indent=2))
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--density-cache", type=Path, default=DEFAULT_DENSITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conversation-id", action="append")
    parser.add_argument("--all-conversations", action="store_true")
    return parser.parse_args()


def _decode_intent(raw: dict[str, object]) -> IntentState:
    preferences = []
    for value in cast(list[object], raw.get("preferences", [])):
        if type(value) is not dict:
            raise ValueError("preference must be an object")
        item = cast(dict[str, object], value)
        raw_value = item.get("value")
        if type(raw_value) is list:
            raw_value = tuple(cast(list[object], raw_value))
        polarity = item.get("semantic_polarity")
        preferences.append(
            Preference(
                id=_string(item, "id"),
                facet=(
                    _nullable_string(item, "canonical_facet")
                    if "canonical_facet" in item
                    else _nullable_string(item, "facet")
                ),
                operator=(
                    None if item.get("operator") is None else Operator(_string(item, "operator"))
                ),
                value=cast(Any, raw_value),
                semantic_text=_nullable_string(item, "semantic_text"),
                semantic_polarity=(None if polarity is None else SemanticPolarity(str(polarity))),
                commitment=Commitment(_string(item, "strength")),
                source=PreferenceSource(_string(item, "source")),
                source_turn=int(item["source_turn"]),
                evidence_text=_string(item, "evidence_text"),
                interpretation_confidence=float(item["interpretation_confidence"]),
            )
        )
    goal = raw.get("goal")
    return IntentState(
        goal=None if goal is None else str(goal),
        preferences=tuple(preferences),
        dont_care_facets=frozenset(str(item) for item in raw.get("dont_care_facets", [])),
        version=int(raw["version"]),
    )


def _decode_compiled(raw: dict[str, object]) -> CompiledQuery:
    constraints = []
    for value in cast(list[object], raw.get("hard_constraints", [])):
        if type(value) is not dict:
            raise ValueError("hard constraint must be an object")
        item = cast(dict[str, object], value)
        raw_value = item.get("value")
        if type(raw_value) is list:
            raw_value = tuple(cast(list[object], raw_value))
        constraints.append(
            CompiledHardConstraint(
                preference_id=_string(item, "preference_id"),
                facet=_string(item, "facet"),
                operator=Operator(_string(item, "operator")),
                value=cast(Any, raw_value),
                policy=ConstraintPolicy(_string(item, "policy")),
            )
        )
    directives = _mapping(raw, "directives")
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=_string(raw, "compiler_version"),
        catalog_id=_string(raw, "catalog_id"),
        catalog_semantic_release_id=_string(raw, "catalog_semantic_release_id"),
        category_graph_id=_string(raw, "category_graph_id"),
        intent_version=int(raw["intent_version"]),
        q_lex=_string_allow_empty(raw, "q_lex"),
        q_sem=_string_allow_empty(raw, "q_sem"),
        search_ready=bool(raw["search_ready"]),
        hard_constraints=tuple(constraints),
        ranking_preferences=(),
        dont_care_facets=tuple(str(item) for item in raw.get("dont_care_facets", [])),
        directives=CompiledDirectives(
            diversity=DiversityDirective(_string(directives, "diversity")),
            comparison_requested=bool(directives["comparison_requested"]),
            explanation_requested=bool(directives["explanation_requested"]),
        ),
        requires_clarification=bool(raw["requires_clarification"]),
        clarification_reason=_nullable_string(raw, "clarification_reason"),
        trace=(),
    )


def _explicit_goal_switch(
    turn: dict[str, object],
    *,
    previous: IntentTransparencyEstimate | None,
    current: IntentState,
) -> bool:
    """Recover the accepted QU switch/refine distinction from update carry-over.

    A true cross-product switch drops the previous preference set, while a goal
    refinement carries active preferences into the revised label.  The runtime
    estimator accepts the resulting explicit hint instead of guessing from two
    unequal goal strings.
    """

    if previous is None or previous.goal is None or current.goal is None:
        return False
    if previous.goal.casefold() == current.goal.casefold():
        return False
    resolved = turn.get("resolved_turn")
    if type(resolved) is not dict:
        return False
    update = cast(dict[str, object], resolved).get("update")
    if type(update) is not dict:
        return False
    for operation in cast(dict[str, object], update).get("operations", []):
        if type(operation) is not dict:
            continue
        item = cast(dict[str, object], operation)
        if item.get("op") == "switch_goal":
            carried = item.get("carry_preference_ids")
            return type(carried) is list and not carried
    return False


def _render_markdown(report: dict[str, object]) -> str:
    runtime = _mapping(report, "runtime")
    lines = [
        "# Intent Transparency runtime v1 replay",
        "",
        f"- Maximum runtime / experiment transparency error: "
        f"`{float(runtime['max_transparency_absolute_error']):.8g}`",
        f"- Maximum remaining-volume relative error: "
        f"`{float(runtime['max_volume_relative_error']):.8g}`",
        f"- Cold model initialization: `{float(runtime['model_initialization_seconds']):.2f}s`",
        f"- Mean warm estimate latency: `{float(runtime['mean_estimate_latency_ms']):.2f}ms`",
        f"- Maximum warm estimate latency: `{float(runtime['max_estimate_latency_ms']):.2f}ms`",
        "",
    ]
    for conversation in cast(list[dict[str, object]], report["conversations"]):
        lines.extend(
            [
                f"## `{conversation['conversation_id']}`",
                "",
                "| turn | user | T_t | delta | direction | D_t | N_t |",
                "| ---: | --- | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for turn in cast(list[dict[str, object]], conversation["turns"]):
            if turn["status"] != "success":
                lines.append(
                    f"| {turn['turn']} | {_cell(str(turn['user_message']))} | — | — | "
                    f"unavailable | {turn['status']} | — |"
                )
                continue
            estimate = _mapping(turn, "estimate")
            diagnostics = _mapping(estimate, "diagnostics")
            lines.append(
                f"| {turn['turn']} | {_cell(str(turn['user_message']))} | "
                f"{_number(estimate['transparency'])} | {_signed(estimate['change'])} | "
                f"{estimate['direction']} | {diagnostics['status']} | "
                f"{_number(estimate['remaining_intent_volume'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "These are real saved DeepSeek Session Context states rerun through the runtime v1 component. "
            "The experiment comparison checks implementation parity; it is not another relevance label.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("input must be a JSON object")
    return cast(dict[str, object], value)


def _mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if type(value) is not dict:
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_allow_empty(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str:
        raise ValueError(f"{key} must be a string")
    return value


def _nullable_string(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{key} must be a string or null")
    return value


def _number(value: object) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _signed(value: object) -> str:
    return "—" if value is None else f"{float(value):+.4f}"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
