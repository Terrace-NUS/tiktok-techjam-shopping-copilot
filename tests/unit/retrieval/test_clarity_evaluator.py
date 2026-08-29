from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.retrieval import evaluate_clarity_prompts as evaluator

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROMPT_SUITE = REPOSITORY_ROOT / "config/retrieval/clarity-prompts-v0.json"


def _pair(identifier: str, domain: str, delta: float | None) -> evaluator.PairDatum:
    return evaluator.PairDatum(
        identifier=identifier,
        cluster_id=identifier,
        domain=domain,
        lower_label="lower",
        higher_label="higher",
        delta=delta,
    )


def _metrics(concordance: float) -> evaluator.PairedMetrics:
    return evaluator.PairedMetrics(
        pair_count=10,
        available_count=10,
        unavailable_count=0,
        wins=10,
        ties=0,
        losses=0,
        concordance=concordance,
        median_delta=0.1,
    )


def _passing_gate_inputs() -> dict[str, object]:
    vague_specific = {
        20: _metrics(0.70),
        40: _metrics(0.75),
        80: _metrics(0.70),
    }
    family_points = {
        "vague_to_focused": {40: _metrics(0.65)},
        "focused_to_specific": {40: _metrics(0.65)},
        "vague_to_specific": vague_specific,
    }
    family_bootstraps = {
        "vague_to_specific": {
            40: evaluator.PairBootstrap(
                concordance=evaluator.ConfidenceInterval(0.61, 0.90),
                median_delta=evaluator.ConfidenceInterval(0.01, 0.20),
                by_domain={},
            )
        }
    }
    return {
        "family_points": family_points,
        "family_bootstraps": family_bootstraps,
        "chain_metrics": evaluator.ChainMetrics(10, 10, 0, 7, 0.70),
        "length_points": {40: _metrics(0.75)},
        "audit_material_change_rate": 0.20,
        "availability_rate": 1.0,
    }


def test_frozen_suite_strictly_loads_and_expands_to_168_unique_requests(
    tmp_path: Path,
) -> None:
    suite = evaluator._load_prompt_suite(PROMPT_SUITE)
    requests = evaluator._query_requests(suite)

    assert (len(suite.families), len(suite.length_controls)) == (40, 10)
    assert (len(suite.invariance_controls), len(suite.diagnostics)) == (10, 8)
    assert len(requests) == 168
    assert len({request.key for request in requests}) == 168
    assert len({request.q_sem for request in requests}) == 168

    document = json.loads(PROMPT_SUITE.read_text(encoding="utf-8"))
    document["unexpected"] = True
    invalid_suite = tmp_path / "invalid-suite.json"
    invalid_suite.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$ has invalid keys"):
        evaluator._load_prompt_suite(invalid_suite)


def test_paired_metrics_apply_symmetric_epsilon_to_wins_ties_and_losses() -> None:
    records = [
        _pair("win", "shoes", 0.11),
        _pair("positive_boundary", "shoes", 0.10),
        _pair("zero", "shoes", 0.0),
        _pair("negative_boundary", "shoes", -0.10),
        _pair("loss", "shoes", -0.11),
        _pair("unavailable", "shoes", None),
    ]

    metrics = evaluator._paired_metrics(records, epsilon=0.10)

    assert (metrics.wins, metrics.ties, metrics.losses) == (1, 3, 1)
    assert (metrics.available_count, metrics.unavailable_count) == (5, 1)
    assert metrics.concordance == pytest.approx(0.5)
    assert metrics.median_delta == pytest.approx(0.0)


def test_bootstrap_is_reproducible_and_preserves_domain_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "BOOTSTRAP_REPLICATES", 32)
    records = [
        _pair("only_positive_cluster", "positive_domain", 1.0),
        _pair("only_negative_cluster", "negative_domain", -1.0),
    ]

    first = evaluator._bootstrap_pairs(records, epsilon=0.1, label="degenerate-strata")
    second = evaluator._bootstrap_pairs(records, epsilon=0.1, label="degenerate-strata")

    assert first == second
    assert first.concordance == evaluator.ConfidenceInterval(0.5, 0.5)
    assert first.median_delta == evaluator.ConfidenceInterval(0.0, 0.0)
    assert first.by_domain["positive_domain"][0] == evaluator.ConfidenceInterval(1.0, 1.0)
    assert first.by_domain["negative_domain"][0] == evaluator.ConfidenceInterval(0.0, 0.0)


def test_pre_registered_gate_passes_all_checks_and_reports_one_failure() -> None:
    passing_inputs = _passing_gate_inputs()
    passing = evaluator._gate_payload(**passing_inputs)

    assert passing["status"] == "pass"
    assert all(check["pass"] for check in passing["checks"])

    failing_inputs = {
        **passing_inputs,
        "chain_metrics": replace(
            passing_inputs["chain_metrics"],
            no_material_reversal_count=6,
            no_material_reversal_rate=0.60,
        ),
    }
    failing = evaluator._gate_payload(**failing_inputs)
    failed_checks = [check["id"] for check in failing["checks"] if not check["pass"]]

    assert failing["status"] == "fail"
    assert failed_checks == ["full_chain_k40_no_material_reversal_rate"]
