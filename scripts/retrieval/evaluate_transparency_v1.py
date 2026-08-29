"""Evaluate the V1 lexical + semantic-mode transparency story."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from shopping_copilot.catalog.semantic import canonical_json_bytes  # noqa: E402
from shopping_copilot.catalog.semantic.release import (  # noqa: E402
    load_catalog_semantic_release,
)
from shopping_copilot.retrieval.dense import DenseRetriever  # noqa: E402
from shopping_copilot.retrieval.documents import load_product_documents  # noqa: E402
from shopping_copilot.retrieval.factory import create_dense_retriever  # noqa: E402
from shopping_copilot.retrieval.lexical import LexicalProbe  # noqa: E402
from shopping_copilot.retrieval.modes import SemanticModeProbe  # noqa: E402

SUITE_SCHEMA = "shopping-copilot/transparency-prompt-suite/v1"
REPORT_SCHEMA = "shopping-copilot/transparency-evaluation/v1"
EXPECTED_FAMILY_COUNT = 24
PROBE_K = 80
MODE_THRESHOLD = 0.94
MIN_DIRECTION_RATE = 0.70
LOW_ANCHOR_QUANTILE = 0.10
HIGH_ANCHOR_QUANTILE = 0.90


@dataclass(frozen=True, slots=True)
class PromptVariant:
    q_lex: str
    q_sem: str


@dataclass(frozen=True, slots=True)
class PromptFamily:
    identifier: str
    domain: str
    split: Literal["calibration", "audit"]
    vague: PromptVariant
    specific: PromptVariant


@dataclass(frozen=True, slots=True)
class VariantEvidence:
    mode_coherence: float | None
    listing_coherence: float | None
    mode_count: int
    effective_mode_count: float
    lexical_token_coverage: float | None


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    identifier: str
    domain: str
    split: Literal["calibration", "audit"]
    vague: VariantEvidence
    specific: VariantEvidence


class EvidenceScorer(Protocol):
    def score(self, prompt: PromptVariant) -> VariantEvidence: ...


def evaluate_families(
    families: tuple[PromptFamily, ...], scorer: EvidenceScorer
) -> dict[str, object]:
    """Score families and apply the pre-registered V1 calibration gate."""

    observations = tuple(
        FamilyEvidence(
            identifier=family.identifier,
            domain=family.domain,
            split=family.split,
            vague=scorer.score(family.vague),
            specific=scorer.score(family.specific),
        )
        for family in families
    )
    return build_report(observations)


def build_report(observations: tuple[FamilyEvidence, ...]) -> dict[str, object]:
    """Pure deterministic statistics used by both CLI and unit tests."""

    calibration = tuple(item for item in observations if item.split == "calibration")
    audit = tuple(item for item in observations if item.split == "audit")
    vague_anchor_values = [
        cast(float, item.vague.mode_coherence)
        for item in calibration
        if item.vague.mode_coherence is not None
    ]
    specific_anchor_values = [
        cast(float, item.specific.mode_coherence)
        for item in calibration
        if item.specific.mode_coherence is not None
    ]
    pooled_anchor_values = vague_anchor_values + specific_anchor_values
    low_anchor = _linear_quantile(pooled_anchor_values, LOW_ANCHOR_QUANTILE)
    high_anchor = _linear_quantile(pooled_anchor_values, HIGH_ANCHOR_QUANTILE)

    available_pairs = tuple(
        item
        for item in audit
        if item.vague.mode_coherence is not None and item.specific.mode_coherence is not None
    )
    deltas = tuple(
        cast(float, item.specific.mode_coherence) - cast(float, item.vague.mode_coherence)
        for item in available_pairs
    )
    availability = len(available_pairs) / len(audit) if audit else 0.0
    strict_direction_rate = sum(delta > 0.0 for delta in deltas) / len(deltas) if deltas else 0.0
    median_delta = _median_or_none(list(deltas))

    anchor_gate = low_anchor is not None and high_anchor is not None and high_anchor > low_anchor
    availability_gate = availability == 1.0
    direction_gate = strict_direction_rate >= MIN_DIRECTION_RATE
    delta_gate = median_delta is not None and median_delta > 0.0
    gate_passed = anchor_gate and availability_gate and direction_gate and delta_gate

    return {
        "schema": REPORT_SCHEMA,
        "policy": {
            "probe_k": PROBE_K,
            "mode_similarity_threshold": MODE_THRESHOLD,
            "minimum_strict_direction_rate": MIN_DIRECTION_RATE,
            "low_anchor_quantile": LOW_ANCHOR_QUANTILE,
            "high_anchor_quantile": HIGH_ANCHOR_QUANTILE,
        },
        "family_count": len(observations),
        "split_counts": {"calibration": len(calibration), "audit": len(audit)},
        "calibration": {
            "available_vague_count": len(vague_anchor_values),
            "available_specific_count": len(specific_anchor_values),
            "pooled_available_count": len(pooled_anchor_values),
            "low_anchor": low_anchor,
            "high_anchor": high_anchor,
        },
        "audit": {
            "pair_count": len(audit),
            "available_pair_count": len(available_pairs),
            "availability": availability,
            "strict_direction_rate": strict_direction_rate,
            "median_delta": median_delta,
            "listing_coherence": _paired_diagnostics(audit, "listing_coherence"),
            "mode_count": _paired_diagnostics(audit, "mode_count"),
            "effective_mode_count": _paired_diagnostics(audit, "effective_mode_count"),
            "lexical_token_coverage": _paired_diagnostics(audit, "lexical_token_coverage"),
        },
        "gate": {
            "anchors_high_above_low": anchor_gate,
            "audit_availability_is_one": availability_gate,
            "audit_direction_rate_at_least_0_70": direction_gate,
            "audit_median_delta_positive": delta_gate,
            "passed": gate_passed,
        },
        "recommended_calibration": {
            "policy_id": "semantic_mode_linear_v1",
            "low_anchor": low_anchor,
            "high_anchor": high_anchor,
            "approved": gate_passed,
        },
        "families": [_family_document(item) for item in observations],
    }


class _ProductionScorer:
    def __init__(self, *, dense_retriever: DenseRetriever, lexical_probe: LexicalProbe) -> None:
        self._dense = dense_retriever
        self._lexical = lexical_probe
        self._mode = SemanticModeProbe(dense_retriever.index)

    def score(self, prompt: PromptVariant) -> VariantEvidence:
        ranking = self._dense.search_with_scores(prompt.q_sem, top_k=PROBE_K)
        semantic = self._mode.observe(
            ranking,
            probe_k=PROBE_K,
            threshold=MODE_THRESHOLD,
        )
        lexical = self._lexical.observe(prompt.q_lex)
        return VariantEvidence(
            mode_coherence=semantic.equal_mode_coherence.debiased_pairwise_cosine,
            listing_coherence=semantic.raw_listing_coherence.debiased_pairwise_cosine,
            mode_count=len(semantic.modes),
            effective_mode_count=semantic.effective_mode_count,
            lexical_token_coverage=(
                lexical.matched_token_count / len(lexical.tokens)
                if lexical.available and lexical.tokens
                else None
            ),
        )


def load_suite(path: Path) -> tuple[PromptFamily, ...]:
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    root = _object(parsed, "suite")
    _exact_keys(root, {"schema", "language", "authorship", "families"}, "suite")
    if root["schema"] != SUITE_SCHEMA:
        raise ValueError("suite schema is invalid")
    if root["language"] != "en":
        raise ValueError("suite language must be en")
    _text(root["authorship"], "suite.authorship")
    raw_families = _array(root["families"], "families")
    if len(raw_families) != EXPECTED_FAMILY_COUNT:
        raise ValueError(f"suite must contain exactly {EXPECTED_FAMILY_COUNT} families")
    families: list[PromptFamily] = []
    identifiers: set[str] = set()
    for index, raw_family in enumerate(raw_families):
        item = _object(raw_family, f"families[{index}]")
        _exact_keys(
            item,
            {"id", "domain", "split", "vague", "specific"},
            f"families[{index}]",
        )
        identifier = _text(item["id"], f"families[{index}].id")
        if identifier in identifiers:
            raise ValueError("family IDs must be unique")
        identifiers.add(identifier)
        split = item["split"]
        if split not in ("calibration", "audit"):
            raise ValueError(f"families[{index}].split is invalid")
        families.append(
            PromptFamily(
                identifier=identifier,
                domain=_text(item["domain"], f"families[{index}].domain"),
                split=cast(Literal["calibration", "audit"], split),
                vague=_variant(item["vague"], f"families[{index}].vague"),
                specific=_variant(item["specific"], f"families[{index}].specific"),
            )
        )
    if sum(item.split == "calibration" for item in families) != 12:
        raise ValueError("suite must contain exactly 12 calibration families")
    if sum(item.split == "audit" for item in families) != 12:
        raise ValueError("suite must contain exactly 12 audit families")
    return tuple(families)


def _variant(value: object, path: str) -> PromptVariant:
    item = _object(value, path)
    _exact_keys(item, {"q_lex", "q_sem"}, path)
    return PromptVariant(
        q_lex=_text(item["q_lex"], f"{path}.q_lex"), q_sem=_text(item["q_sem"], f"{path}.q_sem")
    )


def _family_document(item: FamilyEvidence) -> dict[str, object]:
    vague = _evidence_document(item.vague)
    specific = _evidence_document(item.specific)
    delta = (
        item.specific.mode_coherence - item.vague.mode_coherence
        if item.specific.mode_coherence is not None and item.vague.mode_coherence is not None
        else None
    )
    return {
        "id": item.identifier,
        "domain": item.domain,
        "split": item.split,
        "vague": vague,
        "specific": specific,
        "mode_coherence_delta": delta,
    }


def _evidence_document(value: VariantEvidence) -> dict[str, object]:
    return {
        "mode_coherence": value.mode_coherence,
        "listing_coherence": value.listing_coherence,
        "mode_count": value.mode_count,
        "effective_mode_count": value.effective_mode_count,
        "lexical_token_coverage": value.lexical_token_coverage,
    }


def _paired_diagnostics(items: tuple[FamilyEvidence, ...], field: str) -> dict[str, object]:
    vague = [getattr(item.vague, field) for item in items]
    specific = [getattr(item.specific, field) for item in items]
    vague_values = [float(value) for value in vague if value is not None]
    specific_values = [float(value) for value in specific if value is not None]
    return {
        "available_vague_count": len(vague_values),
        "available_specific_count": len(specific_values),
        "vague_median": _median_or_none(vague_values),
        "specific_median": _median_or_none(specific_values),
    }


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _linear_quantile(values: list[float], quantile: float) -> float | None:
    """Return the deterministic NumPy-style linear quantile without NumPy."""

    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return float(ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index]))


def _object(value: object, path: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{path} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, path: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{path} must be an array")
    return cast(list[object], value)


def _text(value: object, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _exact_keys(value: dict[str, object], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} has missing or unexpected fields")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-index", type=Path, default=Path("artifacts/retrieval/dense-v0"))
    parser.add_argument(
        "--release", type=Path, default=Path("artifacts/catalog-semantic/release-v0")
    )
    parser.add_argument(
        "--suite", type=Path, default=Path("config/retrieval/transparency-prompts-v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/retrieval/transparency-v1/report.json")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    release = load_catalog_semantic_release(args.release, expected_product_count=50_000)
    documents = load_product_documents(
        args.release / "catalog.jsonl",
        expected_parent_asins=frozenset(
            item.parent_asin for item in release.product_category_assignments.assignments
        ),
    )
    dense = create_dense_retriever(index_path=args.dense_index, release_dir=args.release)
    report = evaluate_families(
        load_suite(args.suite),
        _ProductionScorer(
            dense_retriever=dense, lexical_probe=LexicalProbe(documents, probe_k=PROBE_K)
        ),
    )
    payload = canonical_json_bytes(report) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    return 0 if cast(dict[str, object], report["gate"])["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
