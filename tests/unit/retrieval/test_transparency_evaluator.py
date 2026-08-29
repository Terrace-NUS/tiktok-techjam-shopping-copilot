from __future__ import annotations

import pytest

from scripts.retrieval.evaluate_transparency_v1 import (
    FamilyEvidence,
    PromptFamily,
    PromptVariant,
    VariantEvidence,
    build_report,
    evaluate_families,
)


def _evidence(mode: float | None, *, offset: float = 0.0) -> VariantEvidence:
    return VariantEvidence(
        mode_coherence=mode,
        listing_coherence=None if mode is None else mode + 0.1,
        mode_count=4,
        effective_mode_count=3.5 + offset,
        lexical_token_coverage=1.0,
    )


def _family(
    identifier: str, split: str, vague: float | None, specific: float | None
) -> FamilyEvidence:
    return FamilyEvidence(
        identifier=identifier,
        domain="test_domain",
        split=split,  # type: ignore[arg-type]
        vague=_evidence(vague),
        specific=_evidence(specific),
    )


def test_pure_statistics_build_anchors_and_pass_gate() -> None:
    observations = (
        _family("c1", "calibration", 0.1, 0.7),
        _family("c2", "calibration", 0.3, 0.9),
        *(_family(f"a{index}", "audit", 0.2, 0.4) for index in range(10)),
    )

    report = build_report(observations)

    calibration = report["recommended_calibration"]
    assert calibration["policy_id"] == "semantic_mode_linear_v1"  # type: ignore[index]
    assert calibration["low_anchor"] == pytest.approx(0.16)  # type: ignore[index]
    assert calibration["high_anchor"] == pytest.approx(0.84)  # type: ignore[index]
    assert calibration["approved"] is True  # type: ignore[index]
    assert report["audit"]["availability"] == 1.0  # type: ignore[index]
    assert report["audit"]["strict_direction_rate"] == 1.0  # type: ignore[index]
    assert report["audit"]["median_delta"] == 0.2  # type: ignore[index]


def test_gate_fails_on_unavailable_audit_pair_even_when_available_pairs_win() -> None:
    report = build_report(
        (
            _family("c", "calibration", 0.1, 0.8),
            _family("a1", "audit", 0.2, 0.5),
            _family("a2", "audit", None, 0.6),
        )
    )

    assert report["audit"]["availability"] == 0.5  # type: ignore[index]
    assert report["gate"]["audit_availability_is_one"] is False  # type: ignore[index]
    assert report["recommended_calibration"]["approved"] is False  # type: ignore[index]


class _FakeScorer:
    def score(self, prompt: PromptVariant) -> VariantEvidence:
        return _evidence(0.8 if "specific" in prompt.q_sem else 0.2)


def test_evaluate_families_uses_fake_scorer_without_loading_models() -> None:
    families = (
        PromptFamily(
            identifier="calibration",
            domain="test_domain",
            split="calibration",
            vague=PromptVariant(q_lex="vague", q_sem="vague"),
            specific=PromptVariant(q_lex="specific", q_sem="specific"),
        ),
        PromptFamily(
            identifier="audit",
            domain="test_domain",
            split="audit",
            vague=PromptVariant(q_lex="vague", q_sem="vague"),
            specific=PromptVariant(q_lex="specific", q_sem="specific"),
        ),
    )

    report = evaluate_families(families, _FakeScorer())

    assert report["family_count"] == 2
    assert report["gate"]["passed"] is True  # type: ignore[index]
