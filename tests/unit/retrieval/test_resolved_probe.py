from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledQuery,
    DiversityDirective,
)
from shopping_copilot.retrieval.dense import DenseEligibilityMask
from shopping_copilot.retrieval.hard_mask import HardMaskResolver, ResolvedHardMask
from shopping_copilot.retrieval.multi_probe import CompiledProbeRun, CompiledProbeRunner
from shopping_copilot.retrieval.resolved_probe import ResolvedCompiledProbeRunner

CATALOG_ID = "sha256:" + "1" * 64
RELEASE_ID = "sha256:" + "2" * 64


def _query() -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        category_graph_id="sha256:" + "3" * 64,
        intent_version=4,
        q_lex="black walking shoe",
        q_sem="Looking for a black walking shoe.",
        search_ready=True,
        hard_constraints=(),
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


def _resolution() -> ResolvedHardMask:
    return ResolvedHardMask(
        eligible_mask=DenseEligibilityMask(
            index_id="sha256:" + "4" * 64,
            catalog_semantic_release_id=RELEASE_ID,
            values=np.array([False, True, True], dtype=np.bool_),
            _binding=object(),
        ),
        eligible_parent_asins=("B", "C"),
        hard_filter_relaxed=True,
        relaxed_constraints=(),
        trace=(),
    )


def test_runner_resolves_once_and_passes_the_exact_mask_and_flag_to_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _query()
    resolution = _resolution()
    expected_probe_run = cast(CompiledProbeRun, object())
    resolver = object.__new__(HardMaskResolver)
    probe_runner = object.__new__(CompiledProbeRunner)
    resolver_calls: list[CompiledQuery] = []
    probe_calls: list[tuple[CompiledQuery, DenseEligibilityMask | None, bool]] = []

    def resolve(
        _self: HardMaskResolver,
        observed_query: CompiledQuery,
    ) -> ResolvedHardMask:
        resolver_calls.append(observed_query)
        return resolution

    def run_probe(
        _self: CompiledProbeRunner,
        observed_query: CompiledQuery,
        *,
        eligible_mask: DenseEligibilityMask | None = None,
        hard_filter_relaxed: bool = False,
    ) -> CompiledProbeRun:
        probe_calls.append((observed_query, eligible_mask, hard_filter_relaxed))
        return expected_probe_run

    monkeypatch.setattr(HardMaskResolver, "resolve", resolve)
    monkeypatch.setattr(CompiledProbeRunner, "run", run_probe)

    result = ResolvedCompiledProbeRunner(
        resolver=resolver,
        probe_runner=probe_runner,
    ).run(query)

    assert resolver_calls == [query]
    assert len(probe_calls) == 1
    assert probe_calls[0][0] is query
    assert probe_calls[0][1] is resolution.eligible_mask
    assert probe_calls[0][2] is True
    assert result.mask_resolution is resolution
    assert result.probe_run is expected_probe_run


def test_runner_stops_before_probe_when_mask_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = object.__new__(HardMaskResolver)
    probe_runner = object.__new__(CompiledProbeRunner)
    probe_called = False

    def fail_resolution(
        _self: HardMaskResolver,
        _query: CompiledQuery,
    ) -> ResolvedHardMask:
        raise RuntimeError("mask failed")

    def unexpected_probe(
        _self: CompiledProbeRunner,
        _query: CompiledQuery,
        *,
        eligible_mask: DenseEligibilityMask | None = None,
        hard_filter_relaxed: bool = False,
    ) -> CompiledProbeRun:
        nonlocal probe_called
        probe_called = True
        return cast(CompiledProbeRun, object())

    monkeypatch.setattr(HardMaskResolver, "resolve", fail_resolution)
    monkeypatch.setattr(CompiledProbeRunner, "run", unexpected_probe)

    runner = ResolvedCompiledProbeRunner(
        resolver=resolver,
        probe_runner=probe_runner,
    )
    with pytest.raises(RuntimeError, match="mask failed"):
        runner.run(_query())

    assert probe_called is False


def test_runner_requires_the_two_concrete_bound_components() -> None:
    resolver = object.__new__(HardMaskResolver)
    probe_runner = object.__new__(CompiledProbeRunner)

    with pytest.raises(TypeError, match="exact HardMaskResolver"):
        ResolvedCompiledProbeRunner(
            resolver=cast(HardMaskResolver, object()),
            probe_runner=probe_runner,
        )
    with pytest.raises(TypeError, match="exact CompiledProbeRunner"):
        ResolvedCompiledProbeRunner(
            resolver=resolver,
            probe_runner=cast(CompiledProbeRunner, object()),
        )
