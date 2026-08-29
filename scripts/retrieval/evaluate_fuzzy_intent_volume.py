"""Evaluate density-corrected fuzzy intent volume on saved QU sessions.

This is an offline experiment. It does not mutate the catalog, semantic release,
or runtime dense index. Atomic Session Context preferences become independent
Product-of-Experts factors; inverse catalog-kernel density prevents duplicate
listings from contributing one full vote each.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    VerifiedCatalogSemanticRelease,
    load_catalog_semantic_release,
)
from shopping_copilot.query_compiler import (  # noqa: E402
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
    DiversityDirective,
)
from shopping_copilot.retrieval import (  # noqa: E402
    HardMaskResolver,
    SentenceTransformerTextEmbedder,
    build_retrieval_evidence_index,
    load_dense_index,
    project_intent_transparency,
)
from shopping_copilot.retrieval.dense import DenseIndex  # noqa: E402
from shopping_copilot.session_context import Operator  # noqa: E402

REPORT_SCHEMA = "shopping-copilot/fuzzy-intent-volume-experiment/v0"
DEFAULT_EVALUATION = Path("artifacts/retrieval/qu-to-probe-intent-space-natural-v1.json")
DEFAULT_DENSE_INDEX = Path("artifacts/retrieval/dense-v0")
DEFAULT_RELEASE = Path("artifacts/catalog-semantic/release-v0")
DEFAULT_DENSITY_CACHE = Path("artifacts/retrieval/intent-volume-density-v0.npz")
DEFAULT_OUTPUT = Path("artifacts/retrieval/fuzzy-intent-volume-natural-v0.json")
DEFAULT_MARKDOWN = Path("artifacts/retrieval/fuzzy-intent-volume-natural-v0.md")

DENSITY_TEMPERATURES = (0.025, 0.05, 0.1)
MEMBERSHIP_QUANTILES = (0.85, 0.9, 0.95)
MEMBERSHIP_TEMPERATURES = (0.02, 0.04, 0.06)
SOFT_EXPONENT = 0.5
QSEM_ANCHOR_EXPONENT = 0.5
HARD_MISMATCH_FLOORS = (0.01, 0.05, 0.2)
PRIMARY_HARD_MISMATCH_FLOOR = 0.05
STABLE_RELATIVE_TOLERANCE = 0.10
EXPECTED_TAGS = (
    "expected_narrower",
    "expected_broader",
    "expected_stable",
    "expected_override",
)


@dataclass(frozen=True, slots=True)
class SemanticFactor:
    text: str
    polarity: int
    exponent: float
    source: str


@dataclass(frozen=True, slots=True)
class ExperimentTurn:
    suite_id: str
    cohort: str
    conversation_id: str
    turn: int
    tags: tuple[str, ...]
    scenario_type: str
    user_message: str
    goal: str | None
    preferences: tuple[dict[str, object], ...]
    compiled: dict[str, object]

    @property
    def identity(self) -> str:
        return f"{self.suite_id}/{self.conversation_id}/turn-{self.turn}"


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    source: ExperimentTurn
    hybrid_allowed: np.ndarray
    hard_factor_count: int
    hard_oracle_masks: tuple[np.ndarray, ...]
    hybrid_factors: tuple[SemanticFactor, ...]
    fuzzy_factors: tuple[SemanticFactor, ...]
    qsem_factors: tuple[SemanticFactor, ...]
    anchored_factors: tuple[SemanticFactor, ...]
    relaxed_hard_preference_ids: tuple[str, ...]


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    payload = _load_json(args.evaluation)
    turns = _load_turns(payload, cohort=args.cohort, max_turn=args.max_turn)
    if not turns:
        raise ValueError("the selected evaluation contains no successful searchable turns")

    release = load_catalog_semantic_release(args.release)
    dense = load_dense_index(
        args.dense_index,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )
    device = _resolve_device(args.device)
    densities, density_seconds = _load_or_compute_densities(
        dense,
        cache_path=args.density_cache,
        device=device,
        block_size=args.density_block_size,
    )
    prepared, prepare_seconds = _prepare_turns(
        turns,
        release=release,
        dense=dense,
        release_dir=args.release,
    )
    factors = _factor_registry(prepared)
    factor_scores, encode_seconds, score_seconds = _score_factors(
        factors,
        dense=dense,
        device=device,
    )
    result_rows, evaluation_seconds = _evaluate(
        prepared,
        factors=factors,
        factor_scores=factor_scores,
        densities=densities,
        parent_asins=dense.parent_asins,
        device=device,
    )
    summary = _summarize(result_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "inputs": {
            "evaluation": str(args.evaluation.resolve()),
            "source_evaluation_schema": payload.get("schema"),
            "dense_index": str(args.dense_index.resolve()),
            "semantic_release": str(args.release.resolve()),
            "density_cache": str(args.density_cache.resolve()),
            "cohort": args.cohort,
            "max_turn": args.max_turn,
            "catalog_id": dense.manifest.catalog_id,
            "catalog_semantic_release_id": dense.manifest.catalog_semantic_release_id,
            "dense_index_id": dense.index_id,
        },
        "algorithm": {
            "compatibility": "Product-of-Experts over independent Session Context factors",
            "hybrid_mode": (
                "one fail-soft evidence mask per compiled hard constraint; relaxed constraints "
                "fall back to semantic factors"
            ),
            "soft_hybrid_mode": (
                "structured hard evidence uses membership 1 when matched and a non-zero floor "
                "when missed; remaining factors use semantic membership"
            ),
            "fuzzy_mode": "all goal and preference factors use semantic membership",
            "qsem_mode": "the compiler q_sem string is embedded as one semantic factor",
            "anchored_mode": (
                "atomic fuzzy factors plus the compiler q_sem as a lower-weight semantic anchor"
            ),
            "fuzzy_shuffled_control": (
                "same fuzzy factors after one fixed product-column permutation; preserves score "
                "marginals while destroying product semantics"
            ),
            "qsem_shuffled_control": (
                "the q_sem score vector after the same fixed product-column permutation"
            ),
            "semantic_membership": (
                "sigmoid((cosine - per-factor catalog quantile) / temperature)"
            ),
            "negative_membership": "one minus positive semantic membership",
            "soft_exponent": SOFT_EXPONENT,
            "qsem_anchor_exponent": QSEM_ANCHOR_EXPONENT,
            "hard_mismatch_floors": list(HARD_MISMATCH_FLOORS),
            "density_kernel": "exp((cosine - 1) / density_temperature)",
            "density_weight": "one divided by full-catalog kernel density",
            "remaining_volume": "sum_i compatibility_i / density_i",
            "transparency": ("1 - log(1 + remaining_volume) / log(1 + catalog_effective_volume)"),
            "density_temperatures": list(DENSITY_TEMPERATURES),
            "membership_quantiles": list(MEMBERSHIP_QUANTILES),
            "membership_temperatures": list(MEMBERSHIP_TEMPERATURES),
            "stable_relative_tolerance": STABLE_RELATIVE_TOLERANCE,
        },
        "runtime": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "product_count": dense.manifest.product_count,
            "embedding_dimension": dense.manifest.embedding.dimension,
            "turn_count": len(turns),
            "unique_semantic_factor_count": len(factors),
            "density_seconds": density_seconds,
            "turn_prepare_seconds": prepare_seconds,
            "factor_encode_seconds": encode_seconds,
            "factor_score_seconds": score_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "summary": summary,
        "turns": result_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--density-cache", type=Path, default=DEFAULT_DENSITY_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cohort", choices=("all", "natural", "simulator"), default="all")
    parser.add_argument("--max-turn", type=int)
    parser.add_argument("--density-block-size", type=int, default=1_024)
    args = parser.parse_args()
    if args.max_turn is not None and args.max_turn <= 0:
        parser.error("--max-turn must be positive")
    if args.density_block_size <= 0:
        parser.error("--density-block-size must be positive")
    return args


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError("evaluation must be a JSON object")
    return cast(dict[str, object], value)


def _load_turns(
    payload: dict[str, object],
    *,
    cohort: str,
    max_turn: int | None,
) -> list[ExperimentTurn]:
    raw_turns = payload.get("turns")
    if type(raw_turns) is not list:
        raise ValueError("evaluation.turns must be an array")
    result: list[ExperimentTurn] = []
    for raw in cast(list[object], raw_turns):
        if type(raw) is not dict:
            raise ValueError("evaluation turn must be an object")
        item = cast(dict[str, object], raw)
        if item.get("status") != "success":
            continue
        observed_cohort = _string(item, "cohort")
        if cohort != "all" and cohort != observed_cohort:
            continue
        turn_number = _integer(item, "turn")
        if max_turn is not None and turn_number > max_turn:
            continue
        compiled = _mapping(item, "compiled")
        if not bool(compiled.get("search_ready")):
            continue
        intent = _mapping(item, "final_intent")
        raw_preferences = intent.get("preferences")
        if type(raw_preferences) is not list:
            raise ValueError("final_intent.preferences must be an array")
        preferences: list[dict[str, object]] = []
        for preference in cast(list[object], raw_preferences):
            if type(preference) is not dict:
                raise ValueError("final_intent preference must be an object")
            preferences.append(cast(dict[str, object], preference))
        result.append(
            ExperimentTurn(
                suite_id=_string(item, "suite_id"),
                cohort=observed_cohort,
                conversation_id=_string(item, "conversation_id"),
                turn=turn_number,
                tags=_string_tuple(item, "tags"),
                scenario_type=_optional_string(item, "scenario_type", default="unspecified"),
                user_message=_string(item, "user_message"),
                goal=_nullable_string(intent, "goal"),
                preferences=tuple(preferences),
                compiled=compiled,
            )
        )
    return result


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_or_compute_densities(
    dense: DenseIndex,
    *,
    cache_path: Path,
    device: torch.device,
    block_size: int,
) -> tuple[dict[float, np.ndarray], float]:
    started = time.perf_counter()
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            index_id = str(cached["index_id"].item())
            cached_temperatures = tuple(float(item) for item in cached["temperatures"])
            if index_id != dense.index_id or cached_temperatures != DENSITY_TEMPERATURES:
                raise ValueError("density cache does not match the dense index or temperatures")
            result = {
                temperature: np.asarray(cached[_density_key(temperature)], dtype=np.float32)
                for temperature in DENSITY_TEMPERATURES
            }
        _validate_densities(result, product_count=dense.manifest.product_count)
        return result, time.perf_counter() - started

    catalog = torch.from_numpy(np.array(dense.vectors, copy=True)).to(device)
    product_count = dense.manifest.product_count
    densities = {
        temperature: np.empty(product_count, dtype=np.float32)
        for temperature in DENSITY_TEMPERATURES
    }
    with torch.inference_mode():
        for start in range(0, product_count, block_size):
            stop = min(start + block_size, product_count)
            similarities = catalog[start:stop] @ catalog.T
            for temperature in DENSITY_TEMPERATURES:
                values = torch.exp((similarities - 1.0) / temperature).sum(dim=1)
                densities[temperature][start:stop] = values.cpu().numpy()
            if stop % (block_size * 10) == 0 or stop == product_count:
                print(f"density {stop}/{product_count}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _validate_densities(densities, product_count=product_count)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        index_id=np.asarray(dense.index_id),
        temperatures=np.asarray(DENSITY_TEMPERATURES, dtype=np.float64),
        **{_density_key(key): value for key, value in densities.items()},
    )
    return densities, time.perf_counter() - started


def _validate_densities(
    densities: Mapping[float, np.ndarray],
    *,
    product_count: int,
) -> None:
    for temperature in DENSITY_TEMPERATURES:
        values = np.asarray(densities[temperature])
        if values.shape != (product_count,):
            raise ValueError("density vector has the wrong shape")
        if not np.isfinite(values).all() or np.any(values < 1.0 - 1e-4):
            raise ValueError("density vector contains invalid values")


def _prepare_turns(
    turns: Sequence[ExperimentTurn],
    *,
    release: VerifiedCatalogSemanticRelease,
    dense: DenseIndex,
    release_dir: Path,
) -> tuple[list[PreparedTurn], float]:
    started = time.perf_counter()
    evidence = build_retrieval_evidence_index(
        release_dir / "catalog.jsonl",
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(dense.parent_asins),
    )
    resolver = HardMaskResolver(release=release, evidence_index=evidence, dense_index=dense)
    prepared: list[PreparedTurn] = []
    for turn in turns:
        constraints = _hard_constraints(turn)
        masks: list[np.ndarray] = []
        oracle_masks: list[np.ndarray] = []
        relaxed: set[str] = set()
        for constraint in constraints:
            resolution = resolver.resolve(
                _single_constraint_query(turn, constraint, release=release, dense=dense)
            )
            masks.append(np.asarray(resolution.eligible_mask.values, dtype=np.bool_))
            if resolution.hard_filter_relaxed:
                relaxed.add(constraint.preference_id)
            else:
                oracle_masks.append(np.asarray(resolution.eligible_mask.values, dtype=np.bool_))
        hybrid_allowed = (
            np.logical_and.reduce(masks)
            if masks
            else np.ones(dense.manifest.product_count, dtype=np.bool_)
        )
        hard_ids = {item.preference_id for item in constraints}
        prepared.append(
            PreparedTurn(
                source=turn,
                hybrid_allowed=np.asarray(hybrid_allowed, dtype=np.bool_),
                hard_factor_count=len(constraints),
                hard_oracle_masks=tuple(oracle_masks),
                hybrid_factors=_semantic_factors(
                    turn,
                    excluded_preference_ids=hard_ids - relaxed,
                ),
                fuzzy_factors=_semantic_factors(turn, excluded_preference_ids=set()),
                qsem_factors=_qsem_factors(turn),
                anchored_factors=(
                    *_semantic_factors(turn, excluded_preference_ids=set()),
                    *_qsem_factors(turn, exponent=QSEM_ANCHOR_EXPONENT),
                ),
                relaxed_hard_preference_ids=tuple(sorted(relaxed)),
            )
        )
    return prepared, time.perf_counter() - started


def _hard_constraints(turn: ExperimentTurn) -> tuple[CompiledHardConstraint, ...]:
    raw_constraints = turn.compiled.get("hard_constraints")
    if type(raw_constraints) is not list:
        raise ValueError(f"{turn.identity}: hard_constraints must be an array")
    result = []
    for raw in cast(list[object], raw_constraints):
        if type(raw) is not dict:
            raise ValueError(f"{turn.identity}: hard constraint must be an object")
        item = cast(dict[str, object], raw)
        value = item.get("value")
        if type(value) is list:
            value = tuple(cast(list[object], value))
        result.append(
            CompiledHardConstraint(
                preference_id=_string(item, "preference_id"),
                facet=_string(item, "facet"),
                operator=Operator(_string(item, "operator")),
                value=cast(Any, value),
                policy=ConstraintPolicy(_string(item, "policy")),
            )
        )
    return tuple(result)


def _single_constraint_query(
    turn: ExperimentTurn,
    constraint: CompiledHardConstraint,
    *,
    release: VerifiedCatalogSemanticRelease,
    dense: DenseIndex,
) -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=dense.manifest.catalog_id,
        catalog_semantic_release_id=dense.manifest.catalog_semantic_release_id,
        category_graph_id=release.category_registry.category_graph_id,
        intent_version=_integer(turn.compiled, "intent_version"),
        q_lex=str(turn.compiled.get("q_lex", "")),
        q_sem=str(turn.compiled.get("q_sem", "")),
        search_ready=True,
        hard_constraints=(constraint,),
        ranking_preferences=(),
        dont_care_facets=tuple(cast(list[str], turn.compiled.get("dont_care_facets", []))),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def _semantic_factors(
    turn: ExperimentTurn,
    *,
    excluded_preference_ids: set[str],
) -> tuple[SemanticFactor, ...]:
    factors: list[SemanticFactor] = []
    if turn.goal:
        factors.append(
            SemanticFactor(
                text=f"Product goal: {turn.goal}",
                polarity=1,
                exponent=1.0,
                source="goal",
            )
        )
    for preference in turn.preferences:
        preference_id = _string(preference, "id")
        if preference_id in excluded_preference_ids:
            continue
        text = _preference_text(preference)
        if not text:
            continue
        relation = str(preference.get("relation") or preference.get("operator") or "")
        polarity = -1 if relation in {"neq", "not_in", "semantic_negative"} else 1
        exponent = SOFT_EXPONENT if preference.get("strength") == "soft" else 1.0
        factors.append(
            SemanticFactor(
                text=text,
                polarity=polarity,
                exponent=exponent,
                source=f"preference:{preference_id}",
            )
        )
    unique: dict[tuple[str, int], SemanticFactor] = {}
    for factor in factors:
        key = (factor.text.casefold(), factor.polarity)
        observed = unique.get(key)
        if observed is None or factor.exponent > observed.exponent:
            unique[key] = factor
    return tuple(unique.values())


def _qsem_factors(
    turn: ExperimentTurn,
    *,
    exponent: float = 1.0,
) -> tuple[SemanticFactor, ...]:
    text = str(turn.compiled.get("q_sem", "")).strip()
    if not text:
        return ()
    return (
        SemanticFactor(
            text=text,
            polarity=1,
            exponent=exponent,
            source="q_sem",
        ),
    )


def _preference_text(preference: dict[str, object]) -> str | None:
    semantic = preference.get("semantic_text")
    if type(semantic) is str and semantic.strip():
        return semantic.strip()
    evidence = preference.get("evidence_text")
    if type(evidence) is str and evidence.strip():
        facet = preference.get("canonical_facet") or preference.get("facet")
        prefix = "Preference" if facet is None else str(facet).replace("_", " ")
        return f"{prefix}: {evidence.strip()}"
    value = preference.get("value")
    if value is None:
        return None
    facet = preference.get("canonical_facet") or preference.get("facet") or "preference"
    return f"{str(facet).replace('_', ' ')}: {_render_value(value)}"


def _render_value(value: object) -> str:
    if type(value) is list:
        return " or ".join(str(item) for item in cast(list[object], value))
    return str(value)


def _factor_registry(prepared: Sequence[PreparedTurn]) -> tuple[SemanticFactor, ...]:
    registry: dict[tuple[str, int, float], SemanticFactor] = {}
    for turn in prepared:
        for factor in (
            *turn.hybrid_factors,
            *turn.fuzzy_factors,
            *turn.qsem_factors,
            *turn.anchored_factors,
        ):
            registry[(factor.text, factor.polarity, factor.exponent)] = factor
    return tuple(registry[key] for key in sorted(registry))


def _score_factors(
    factors: Sequence[SemanticFactor],
    *,
    dense: DenseIndex,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    encode_started = time.perf_counter()
    embedder = SentenceTransformerTextEmbedder(
        dense.manifest.embedding,
        device=str(device),
        local_files_only=True,
    )
    vectors = np.stack([embedder.encode_query(factor.text) for factor in factors])
    encode_seconds = time.perf_counter() - encode_started
    score_started = time.perf_counter()
    factor_tensor = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(device)
    catalog = torch.from_numpy(np.array(dense.vectors, copy=True)).to(device)
    with torch.inference_mode():
        scores = factor_tensor @ catalog.T
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = scores.cpu().numpy().astype(np.float32, copy=False)
    return result, encode_seconds, time.perf_counter() - score_started


def _evaluate(
    prepared: Sequence[PreparedTurn],
    *,
    factors: Sequence[SemanticFactor],
    factor_scores: np.ndarray,
    densities: Mapping[float, np.ndarray],
    parent_asins: Sequence[str],
    device: torch.device,
) -> tuple[list[dict[str, object]], float]:
    started = time.perf_counter()
    factor_index = {
        (factor.text, factor.polarity, factor.exponent): index
        for index, factor in enumerate(factors)
    }
    scores = torch.from_numpy(factor_scores).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20_260_829)
    shuffled_scores = scores[
        :, torch.randperm(factor_scores.shape[1], generator=generator, device=device)
    ]
    threshold_rows = {
        quantile: torch.from_numpy(
            np.quantile(factor_scores, quantile, axis=1).astype(np.float32)
        ).to(device)
        for quantile in MEMBERSHIP_QUANTILES
    }
    log_density_weights = {
        temperature: torch.from_numpy(
            -np.log(np.asarray(density, dtype=np.float64)).astype(np.float32)
        ).to(device)
        for temperature, density in densities.items()
    }
    catalog_volumes = {
        temperature: float(torch.exp(torch.logsumexp(weights, dim=0)).item())
        for temperature, weights in log_density_weights.items()
    }
    all_allowed = torch.ones(factor_scores.shape[1], dtype=torch.bool, device=device)
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for turn_index, prepared_turn in enumerate(prepared):
            configurations: dict[str, object] = {}
            top_products: dict[str, list[str]] = {}
            hard_reference = torch.from_numpy(prepared_turn.hybrid_allowed).to(device)
            hard_oracle_masks = (
                torch.from_numpy(np.stack(prepared_turn.hard_oracle_masks)).to(device)
                if prepared_turn.hard_oracle_masks
                else None
            )
            has_hard_oracle = hard_oracle_masks is not None
            for mode in (
                "hybrid",
                "soft_hybrid",
                "fuzzy",
                "fuzzy_shuffled",
                "qsem",
                "qsem_shuffled",
                "anchored",
                "anchored_shuffled",
            ):
                if mode in {"hybrid", "soft_hybrid"}:
                    selected = prepared_turn.hybrid_factors
                elif mode.startswith("qsem"):
                    selected = prepared_turn.qsem_factors
                elif mode.startswith("anchored"):
                    selected = prepared_turn.anchored_factors
                else:
                    selected = prepared_turn.fuzzy_factors
                allowed = (
                    torch.from_numpy(prepared_turn.hybrid_allowed).to(device)
                    if mode == "hybrid"
                    else all_allowed
                )
                active_scores = shuffled_scores if mode.endswith("_shuffled") else scores
                indices = torch.tensor(
                    [factor_index[(item.text, item.polarity, item.exponent)] for item in selected],
                    dtype=torch.long,
                    device=device,
                )
                polarities = torch.tensor(
                    [item.polarity for item in selected],
                    dtype=torch.float32,
                    device=device,
                )
                exponents = torch.tensor(
                    [item.exponent for item in selected],
                    dtype=torch.float32,
                    device=device,
                )
                for quantile in MEMBERSHIP_QUANTILES:
                    for membership_temperature in MEMBERSHIP_TEMPERATURES:
                        if selected:
                            z = (
                                active_scores[indices] - threshold_rows[quantile][indices, None]
                            ) / membership_temperature
                            log_compatibility = torch.sum(
                                -functional.softplus(-polarities[:, None] * z) * exponents[:, None],
                                dim=0,
                            )
                        else:
                            log_compatibility = torch.zeros(
                                factor_scores.shape[1],
                                dtype=torch.float32,
                                device=device,
                            )
                        hard_floors: tuple[float | None, ...] = (
                            HARD_MISMATCH_FLOORS if mode == "soft_hybrid" else (None,)
                        )
                        for hard_floor in hard_floors:
                            effective_log_compatibility = log_compatibility
                            if hard_floor is not None and hard_oracle_masks is not None:
                                effective_log_compatibility = (
                                    effective_log_compatibility
                                    + torch.sum(
                                        torch.where(
                                            hard_oracle_masks,
                                            torch.tensor(0.0, device=device),
                                            torch.tensor(
                                                math.log(hard_floor),
                                                device=device,
                                            ),
                                        ),
                                        dim=0,
                                    )
                                )
                            effective_log_compatibility = torch.where(
                                allowed,
                                effective_log_compatibility,
                                torch.tensor(-torch.inf, device=device),
                            )
                            top_count = min(
                                20,
                                int(torch.count_nonzero(allowed).item()),
                            )
                            top_indices = torch.topk(
                                effective_log_compatibility,
                                k=top_count,
                            ).indices
                            all_hard_compliance = (
                                float(torch.mean(hard_reference[top_indices].float()).item())
                                if has_hard_oracle and top_count > 0
                                else None
                            )
                            mean_hard_factor_compliance = (
                                float(torch.mean(hard_oracle_masks[:, top_indices].float()).item())
                                if hard_oracle_masks is not None and top_count > 0
                                else None
                            )
                            primary_floor = hard_floor is None or math.isclose(
                                hard_floor,
                                PRIMARY_HARD_MISMATCH_FLOOR,
                            )
                            if (
                                primary_floor
                                and math.isclose(quantile, 0.9)
                                and math.isclose(membership_temperature, 0.04)
                            ):
                                top_products[mode] = [
                                    parent_asins[index]
                                    for index in top_indices[: min(5, top_count)].cpu().tolist()
                                ]
                            for (
                                density_temperature,
                                log_weights,
                            ) in log_density_weights.items():
                                log_volume = torch.logsumexp(
                                    log_weights + effective_log_compatibility,
                                    dim=0,
                                )
                                volume = float(torch.exp(log_volume).item())
                                catalog_volume = catalog_volumes[density_temperature]
                                transparency = project_intent_transparency(
                                    volume,
                                    reference_volume=catalog_volume,
                                )
                                key = _config_key(
                                    mode,
                                    density_temperature,
                                    quantile,
                                    membership_temperature,
                                    hard_mismatch_floor=hard_floor,
                                )
                                configurations[key] = {
                                    "mode": mode,
                                    "density_temperature": density_temperature,
                                    "membership_quantile": quantile,
                                    "membership_temperature": membership_temperature,
                                    "hard_mismatch_floor": hard_floor,
                                    "remaining_volume": volume,
                                    "catalog_effective_volume": catalog_volume,
                                    "transparency": transparency,
                                    "top20_all_hard_compliance": all_hard_compliance,
                                    "top20_mean_hard_factor_compliance": (
                                        mean_hard_factor_compliance
                                    ),
                                }
            rows.append(
                {
                    "identity": prepared_turn.source.identity,
                    "suite_id": prepared_turn.source.suite_id,
                    "cohort": prepared_turn.source.cohort,
                    "conversation_id": prepared_turn.source.conversation_id,
                    "turn": prepared_turn.source.turn,
                    "tags": list(prepared_turn.source.tags),
                    "scenario_type": prepared_turn.source.scenario_type,
                    "user_message": prepared_turn.source.user_message,
                    "goal": prepared_turn.source.goal,
                    "hybrid_hard_factor_count": prepared_turn.hard_factor_count,
                    "hard_oracle_factor_count": len(prepared_turn.hard_oracle_masks),
                    "hybrid_excluded_product_count": int(
                        np.count_nonzero(~prepared_turn.hybrid_allowed)
                    ),
                    "hybrid_eligible_count": int(np.count_nonzero(prepared_turn.hybrid_allowed)),
                    "hybrid_semantic_factors": [
                        _factor_payload(item) for item in prepared_turn.hybrid_factors
                    ],
                    "fuzzy_semantic_factors": [
                        _factor_payload(item) for item in prepared_turn.fuzzy_factors
                    ],
                    "qsem_factors": [_factor_payload(item) for item in prepared_turn.qsem_factors],
                    "anchored_factors": [
                        _factor_payload(item) for item in prepared_turn.anchored_factors
                    ],
                    "relaxed_hard_preference_ids": list(prepared_turn.relaxed_hard_preference_ids),
                    "primary_top_products": top_products,
                    "configurations": configurations,
                }
            )
            if (turn_index + 1) % 10 == 0 or turn_index + 1 == len(prepared):
                print(f"volume {turn_index + 1}/{len(prepared)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return rows, time.perf_counter() - started


def _factor_payload(factor: SemanticFactor) -> dict[str, object]:
    return {
        "text": factor.text,
        "polarity": factor.polarity,
        "exponent": factor.exponent,
        "source": factor.source,
    }


def _summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    first_configurations = cast(dict[str, object], rows[0]["configurations"])
    config_keys = list(first_configurations)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["conversation_id"])].append(row)
    configurations: dict[str, object] = {}
    for config_key in config_keys:
        expected = {tag: [] for tag in EXPECTED_TAGS}
        simulator_pairs: list[dict[str, object]] = []
        progressive_pairs: list[dict[str, object]] = []
        for conversation_id, items in sorted(grouped.items()):
            ordered = sorted(items, key=lambda item: int(item["turn"]))
            if len(ordered) < 2:
                continue
            first = ordered[0]
            last = ordered[-1]
            first_value = _volume(first, config_key)
            last_value = _volume(last, config_key)
            pair = _pair_payload(
                conversation_id,
                first_value=first_value,
                last_value=last_value,
            )
            if first["cohort"] == "simulator":
                pair["matched"] = last_value < first_value
                simulator_pairs.append(pair)
            tags = _row_tags(first)
            if first["cohort"] == "natural" and "progressive_narrowing" in tags:
                for previous, current in zip(ordered, ordered[1:], strict=False):
                    adjacent = _pair_payload(
                        conversation_id,
                        first_value=_volume(previous, config_key),
                        last_value=_volume(current, config_key),
                    )
                    adjacent["from_turn"] = int(previous["turn"])
                    adjacent["to_turn"] = int(current["turn"])
                    adjacent["matched"] = adjacent["last"] < adjacent["first"]
                    progressive_pairs.append(adjacent)
            matched_tags = [tag for tag in EXPECTED_TAGS if tag in tags]
            if len(matched_tags) > 1:
                raise ValueError(f"{conversation_id} has conflicting expected tags")
            if matched_tags:
                tag = matched_tags[0]
                pair["matched"] = _matches_expectation(
                    tag,
                    first=first_value,
                    last=last_value,
                )
                expected[tag].append(pair)
        configurations[config_key] = {
            "natural_expected": {tag: _pair_summary(pairs) for tag, pairs in expected.items()},
            "simulator_first_to_last": _pair_summary(simulator_pairs),
            "natural_progressive_adjacent": _pair_summary(progressive_pairs),
            "top20_all_hard_compliance": _compliance_summary(
                rows,
                config_key,
                field="top20_all_hard_compliance",
            ),
            "top20_mean_hard_factor_compliance": _compliance_summary(
                rows,
                config_key,
                field="top20_mean_hard_factor_compliance",
            ),
        }
    return {
        "configuration_count": len(configurations),
        "configurations": configurations,
        "leaderboard": _leaderboard(configurations),
    }


def _volume(row: dict[str, object], config_key: str) -> float:
    configurations = cast(dict[str, object], row["configurations"])
    config = cast(dict[str, object], configurations[config_key])
    return float(config["remaining_volume"])


def _pair_payload(
    conversation_id: str,
    *,
    first_value: float,
    last_value: float,
) -> dict[str, object]:
    delta = last_value - first_value
    return {
        "conversation_id": conversation_id,
        "first": first_value,
        "last": last_value,
        "delta": delta,
        "relative_change": abs(delta) / max(abs(first_value), abs(last_value), 1e-12),
    }


def _matches_expectation(tag: str, *, first: float, last: float) -> bool | None:
    if tag == "expected_narrower":
        return last < first
    if tag == "expected_broader":
        return last > first
    if tag == "expected_stable":
        return abs(last - first) / max(abs(first), abs(last), 1e-12) <= STABLE_RELATIVE_TOLERANCE
    if tag == "expected_override":
        return None
    raise ValueError(f"unsupported expected tag: {tag}")


def _pair_summary(pairs: Sequence[dict[str, object]]) -> dict[str, object]:
    scored = [item for item in pairs if item.get("matched") is not None]
    return {
        "pair_count": len(pairs),
        "scored_pair_count": len(scored),
        "matched_count": sum(bool(item["matched"]) for item in scored),
        "not_matched_count": sum(not bool(item["matched"]) for item in scored),
        "pairs": list(pairs),
    }


def _leaderboard(configurations: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for config_key, raw in configurations.items():
        config = cast(dict[str, object], raw)
        expected = cast(dict[str, object], config["natural_expected"])
        natural_matched = 0
        natural_scored = 0
        for tag in ("expected_narrower", "expected_broader", "expected_stable"):
            item = cast(dict[str, object], expected[tag])
            natural_matched += int(item["matched_count"])
            natural_scored += int(item["scored_pair_count"])
        simulator = cast(dict[str, object], config["simulator_first_to_last"])
        progressive = cast(dict[str, object], config["natural_progressive_adjacent"])
        all_compliance = cast(dict[str, object], config["top20_all_hard_compliance"])
        factor_compliance = cast(
            dict[str, object],
            config["top20_mean_hard_factor_compliance"],
        )
        rows.append(
            {
                "configuration": config_key,
                "natural_matched": natural_matched,
                "natural_scored": natural_scored,
                "simulator_matched": int(simulator["matched_count"]),
                "simulator_scored": int(simulator["scored_pair_count"]),
                "progressive_matched": int(progressive["matched_count"]),
                "progressive_scored": int(progressive["scored_pair_count"]),
                "top20_all_hard_compliance_mean": all_compliance["mean"],
                "top20_mean_hard_factor_compliance": factor_compliance["mean"],
                "top20_hard_compliance_turn_count": factor_compliance["turn_count"],
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -int(item["natural_matched"]),
            -int(item["progressive_matched"]),
            -int(item["simulator_matched"]),
            -float(item["top20_mean_hard_factor_compliance"] or 0.0),
            str(item["configuration"]),
        ),
    )


def _compliance_summary(
    rows: Sequence[dict[str, object]],
    config_key: str,
    *,
    field: str,
) -> dict[str, object]:
    values = []
    for row in rows:
        configurations = cast(dict[str, object], row["configurations"])
        config = cast(dict[str, object], configurations[config_key])
        value = config[field]
        if value is not None:
            values.append(float(value))
    return {
        "turn_count": len(values),
        "mean": None if not values else float(np.mean(values, dtype=np.float64)),
        "median": None if not values else float(np.median(values)),
        "minimum": None if not values else float(np.min(values)),
    }


def _render_markdown(report: dict[str, object]) -> str:
    runtime = cast(dict[str, object], report["runtime"])
    summary = cast(dict[str, object], report["summary"])
    leaderboard = cast(list[object], summary["leaderboard"])
    lines = [
        "# Density-corrected fuzzy intent volume",
        "",
        f"- Turns: **{runtime['turn_count']}**",
        f"- Products: **{runtime['product_count']}**",
        f"- Unique semantic factors: **{runtime['unique_semantic_factor_count']}**",
        f"- Device: **{runtime['device']}**",
        "",
        "## Top configurations",
        "",
        "| configuration | natural | progressive | simulator | all hard | mean hard facet |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for raw in leaderboard[:20]:
        item = cast(dict[str, object], raw)
        lines.append(
            f"| `{item['configuration']}` | "
            f"{item['natural_matched']}/{item['natural_scored']} | "
            f"{item['progressive_matched']}/{item['progressive_scored']} | "
            f"{item['simulator_matched']}/{item['simulator_scored']} | "
            f"{_optional_decimal(item['top20_all_hard_compliance_mean'])} | "
            f"{_optional_decimal(item['top20_mean_hard_factor_compliance'])} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is an offline parameter sweep. A leading configuration is not a frozen runtime contract.",
            "Direction scores test intent-space behavior; they do not by themselves prove product relevance.",
            "",
        ]
    )
    return "\n".join(lines)


def _config_key(
    mode: str,
    density_temperature: float,
    membership_quantile: float,
    membership_temperature: float,
    *,
    hard_mismatch_floor: float | None = None,
) -> str:
    key = (
        f"{mode}_d{density_temperature:.3f}_q{membership_quantile:.3f}"
        f"_m{membership_temperature:.3f}"
    )
    if hard_mismatch_floor is not None:
        key += f"_h{hard_mismatch_floor:.3f}"
    return key


def _density_key(temperature: float) -> str:
    return f"density_{temperature:.3f}"


def _optional_decimal(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _mapping(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if type(value) is not dict:
        raise ValueError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _string(mapping: dict[str, object], key: str) -> str:
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


def _optional_string(mapping: dict[str, object], key: str, *, default: str) -> str:
    value = mapping.get(key)
    if value is None:
        return default
    if type(value) is not str:
        raise ValueError(f"{key} must be a string or null")
    return value


def _integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def _string_tuple(mapping: dict[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(cast(list[str], value))


def _row_tags(row: dict[str, object]) -> frozenset[str]:
    value = row.get("tags")
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("row.tags must be an array of strings")
    return frozenset(cast(list[str], value))


if __name__ == "__main__":
    raise SystemExit(main())
