"""Run reproducible Retrieval Evidence and hard-mask smoke checks on the 50k catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

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
from shopping_copilot.retrieval.bundle import load_dense_index  # noqa: E402
from shopping_copilot.retrieval.evidence import (  # noqa: E402
    build_retrieval_evidence_index,
)
from shopping_copilot.retrieval.hard_mask import HardMaskResolver  # noqa: E402
from shopping_copilot.session_context import Operator  # noqa: E402


def main() -> int:
    args = _parse_args()
    release = load_catalog_semantic_release(args.release_dir)
    dense_index = load_dense_index(
        args.dense_index,
        expected_catalog_id=release.manifest.catalog_id,
        expected_release_id=release.release_id,
    )
    before = _content_id(args.catalog)
    started = time.perf_counter()
    evidence = build_retrieval_evidence_index(
        args.catalog,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        expected_parent_asins=set(dense_index.parent_asins),
    )
    build_seconds = time.perf_counter() - started
    after = _content_id(args.catalog)
    resolver = HardMaskResolver(
        release=release,
        evidence_index=evidence,
        dense_index=dense_index,
    )

    cases = _cases()
    resolved = []
    for case_id, constraints in cases:
        result = resolver.resolve(_query(release, case_id, constraints))
        resolved.append(
            {
                "case_id": case_id,
                "eligible_count": len(result.eligible_parent_asins),
                "hard_filter_relaxed": result.hard_filter_relaxed,
                "relaxed_preference_ids": [
                    item.preference_id for item in result.relaxed_constraints
                ],
                "trace": [
                    {
                        "preference_id": item.preference_id,
                        "facet": item.facet,
                        "operator": item.operator.value,
                        "before_count": item.before_count,
                        "matched_count": item.matched_count,
                        "after_count": item.after_count,
                        "disposition": item.disposition.value,
                        "reason": item.reason,
                    }
                    for item in result.trace
                ],
            }
        )

    report = {
        "schema": "shopping-copilot/hard-mask-smoke/v0",
        "catalog_id": release.manifest.catalog_id,
        "catalog_semantic_release_id": release.release_id,
        "dense_index_id": dense_index.index_id,
        "evidence_index_id": evidence.index_id,
        "product_count": len(evidence.parent_asins),
        "catalog_unchanged": before == after == release.manifest.catalog_id,
        "evidence_build_seconds": build_seconds,
        "direct_match_counts": {
            "brand=columbia": len(evidence.match("brand", "columbia")),
            "color=black": len(evidence.match("color", "black")),
            "feature=rfid blocking": len(evidence.match("feature", "rfid blocking")),
            "feature=waterproof": len(evidence.match("feature", "waterproof")),
            "material=leather": len(evidence.match("material", "leather")),
            "material=stainless steel": len(evidence.match("material", "stainless steel")),
            "size=8": len(evidence.match("size", "8")),
            "size=medium": len(evidence.match("size", "medium")),
            "style=vintage": len(evidence.match("style", "vintage")),
            "use_case=hiking": len(evidence.match("use_case", "hiking")),
        },
        "resolved_cases": resolved,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["catalog_unchanged"] else 1


def _cases() -> tuple[tuple[str, tuple[CompiledHardConstraint, ...]], ...]:
    text = ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE
    return (
        ("no_constraints", ()),
        (
            "category_root",
            (
                _hard(
                    "category",
                    "system_product_category",
                    Operator.EQ,
                    "ROOT",
                    ConstraintPolicy.VERIFIED_CATEGORY,
                ),
            ),
        ),
        ("color_black", (_hard("color", "color", Operator.EQ, "black", text),)),
        ("material_leather", (_hard("material", "material", Operator.EQ, "leather", text),)),
        ("exclude_leather", (_hard("not_material", "material", Operator.NEQ, "leather", text),)),
        (
            "budget_100",
            (_hard("budget", "price", Operator.LE, 10_000, ConstraintPolicy.CONSERVATIVE_PRICE),),
        ),
        ("size_8", (_hard("size", "size", Operator.EQ, "8", text),)),
        ("feature_rfid", (_hard("rfid", "feature", Operator.EQ, "rfid blocking", text),)),
        ("use_case_hiking", (_hard("hiking", "use_case", Operator.EQ, "hiking", text),)),
        (
            "missing_include_relaxes",
            (_hard("missing", "feature", Operator.EQ, "zzzxqv hard mask sentinel", text),),
        ),
        (
            "exclude_before_conflicting_include",
            (
                _hard("want_black", "color", Operator.EQ, "black", text),
                _hard("not_black", "color", Operator.NEQ, "black", text),
            ),
        ),
    )


def _hard(
    preference_id: str,
    facet: str,
    operator: Operator,
    value: str | int,
    policy: ConstraintPolicy,
) -> CompiledHardConstraint:
    return CompiledHardConstraint(
        preference_id=preference_id,
        facet=facet,
        operator=operator,
        value=value,
        policy=policy,
    )


def _query(
    release: VerifiedCatalogSemanticRelease,
    case_id: str,
    constraints: tuple[CompiledHardConstraint, ...],
) -> CompiledQuery:
    resolved_constraints = tuple(
        CompiledHardConstraint(
            preference_id=item.preference_id,
            facet=item.facet,
            operator=item.operator,
            value=(
                release.category_registry.root_scope_id
                if item.facet == "system_product_category" and item.value == "ROOT"
                else item.value
            ),
            policy=item.policy,
        )
        for item in constraints
    )
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=release.manifest.catalog_id,
        catalog_semantic_release_id=release.release_id,
        category_graph_id=release.category_registry.category_graph_id,
        intent_version=1,
        q_lex=case_id.replace("_", " "),
        q_sem=f"Hard-mask audit: {case_id}.",
        search_ready=True,
        hard_constraints=resolved_constraints,
        ranking_preferences=(),
        dont_care_facets=(),
        directives=CompiledDirectives(
            diversity=DiversityDirective.AUTO,
            comparison_requested=False,
            explanation_requested=False,
        ),
        requires_clarification=False,
        clarification_reason=None,
        trace=(),
    )


def _content_id(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/catalog-semantic/release-v0/catalog.jsonl",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/catalog-semantic/release-v0",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/retrieval/dense-v0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
