from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shopping_copilot.query_compiler import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    ConstraintPolicy,
    DiversityDirective,
)
from shopping_copilot.retrieval import (
    CatalogDensitySnapshot,
    DenseIndex,
    DenseIndexManifest,
    EmbeddingSpec,
    IntentVolumeDirection,
    IntentVolumeEstimator,
    IntentVolumePolicy,
    IntentVolumeStatus,
    load_catalog_density,
    project_intent_transparency,
)
from shopping_copilot.retrieval.hard_mask import ResolvedHardMask
from shopping_copilot.retrieval.models import DENSE_INDEX_SCHEMA, DenseArtifactRef
from shopping_copilot.session_context import (
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    SemanticPolarity,
)

CATALOG_ID = "sha256:" + "1" * 64
RELEASE_ID = "sha256:" + "2" * 64
INDEX_ID = "sha256:" + "3" * 64
GRAPH_ID = "sha256:" + "4" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class _FakeEmbedder:
    def __init__(self, spec: EmbeddingSpec) -> None:
        self._spec = spec
        self.calls: list[str] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def encode_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        raise AssertionError("runtime intent volume must not encode catalog documents")

    def encode_query(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if "earrings" in text.casefold():
            return np.array([0.0, 1.0], dtype=np.float32)
        if "waterproof" in text.casefold():
            return np.array([0.8, 0.6], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeResolver:
    def __init__(self, index: DenseIndex, *, relaxed_ids: set[str] | None = None) -> None:
        self._index = index
        self._relaxed_ids = relaxed_ids or set()

    def resolve(self, query: CompiledQuery) -> ResolvedHardMask:
        constraint = query.hard_constraints[0]
        relaxed = constraint.preference_id in self._relaxed_ids
        products = self._index.parent_asins if relaxed else ("A", "B")
        return ResolvedHardMask(
            eligible_mask=self._index.make_eligibility_mask(products),
            eligible_parent_asins=tuple(products),
            hard_filter_relaxed=relaxed,
            relaxed_constraints=(constraint,) if relaxed else (),
            trace=(),
        )


def _spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        backend="fake",
        backend_version="1.0",
        model_id="example/model",
        model_revision="revision",
        dimension=2,
        max_sequence_length=32,
        query_instruction="",
        document_instruction="",
        pooling="cls",
    )


def _index() -> DenseIndex:
    spec = _spec()
    manifest = DenseIndexManifest(
        schema=DENSE_INDEX_SCHEMA,
        builder_version="dense_index_v0",
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        document_template_id="product_document_v1",
        document_corpus_id="sha256:" + "5" * 64,
        product_count=4,
        embedding=spec,
        vector_dtype="float32",
        artifacts=(
            DenseArtifactRef(
                kind="parent_asins",
                filename="parent-asins.json",
                content_id="sha256:" + "6" * 64,
                byte_size=1,
            ),
            DenseArtifactRef(
                kind="vectors",
                filename="vectors.npy",
                content_id="sha256:" + "7" * 64,
                byte_size=1,
            ),
        ),
    )
    return DenseIndex(
        index_id=INDEX_ID,
        manifest=manifest,
        parent_asins=("A", "B", "C", "D"),
        vectors=np.array(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        ),
    )


def _density(index: DenseIndex) -> CatalogDensitySnapshot:
    return CatalogDensitySnapshot(
        index_id=index.index_id,
        catalog_semantic_release_id=index.manifest.catalog_semantic_release_id,
        temperature=0.025,
        values=np.ones(index.manifest.product_count, dtype=np.float32),
    )


def _semantic_preference(
    preference_id: str,
    text: str,
    *,
    commitment: Commitment = Commitment.HARD,
) -> Preference:
    return Preference(
        id=preference_id,
        facet=None,
        operator=None,
        value=None,
        semantic_text=text,
        semantic_polarity=SemanticPolarity.POSITIVE,
        commitment=commitment,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text=text,
        interpretation_confidence=1.0,
    )


def _hard_preference(preference_id: str) -> Preference:
    return Preference(
        id=preference_id,
        facet="color",
        operator=Operator.EQ,
        value="red",
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=1,
        evidence_text="red",
        interpretation_confidence=1.0,
    )


def _intent(
    *,
    goal: str | None = "running shoes",
    preferences: tuple[Preference, ...] = (),
    version: int = 1,
) -> IntentState:
    return IntentState(
        goal=goal,
        preferences=preferences,
        dont_care_facets=frozenset(),
        version=version,
    )


def _compiled(
    intent: IntentState,
    *,
    hard_constraints: tuple[CompiledHardConstraint, ...] = (),
) -> CompiledQuery:
    return CompiledQuery(
        schema=COMPILED_QUERY_SCHEMA,
        compiler_version=QUERY_COMPILER_VERSION,
        catalog_id=CATALOG_ID,
        catalog_semantic_release_id=RELEASE_ID,
        category_graph_id=GRAPH_ID,
        intent_version=intent.version,
        q_lex=intent.goal or "",
        q_sem=f"Looking for {intent.goal}." if intent.goal else "",
        search_ready=intent.goal is not None,
        hard_constraints=hard_constraints,
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


def _estimator(
    index: DenseIndex,
    *,
    relaxed_ids: set[str] | None = None,
) -> tuple[IntentVolumeEstimator, _FakeEmbedder]:
    embedder = _FakeEmbedder(index.manifest.embedding)
    estimator = IntentVolumeEstimator(
        dense_index=index,
        embedder=embedder,
        hard_mask_resolver=_FakeResolver(index, relaxed_ids=relaxed_ids),
        density=_density(index),
        policy=IntentVolumePolicy(
            membership_quantile=0.5,
            membership_temperature=0.2,
        ),
    )
    return estimator, embedder


def test_projection_has_clear_endpoints_and_clips_out_of_reference_mass() -> None:
    assert project_intent_transparency(100.0, reference_volume=100.0) == 0.0
    assert project_intent_transparency(0.0, reference_volume=100.0) == 1.0
    assert project_intent_transparency(200.0, reference_volume=100.0) == 0.0


def test_checked_in_policy_matches_runtime_defaults() -> None:
    payload = json.loads(
        (REPOSITORY_ROOT / "config/retrieval/intent-volume-v1.json").read_text(encoding="utf-8")
    )
    policy = IntentVolumePolicy()

    assert payload["schema"] == "shopping-copilot/intent-volume-policy/v1"
    assert payload["policy_id"] == policy.policy_id
    assert payload["mapping_id"] == policy.mapping_id
    assert payload["approved"] is policy.approved
    assert payload["parameters"] == {
        "density_temperature": policy.density_temperature,
        "membership_quantile": policy.membership_quantile,
        "membership_temperature": policy.membership_temperature,
        "hard_mismatch_floor": policy.hard_mismatch_floor,
        "soft_preference_exponent": policy.soft_preference_exponent,
        "stable_relative_tolerance": policy.stable_relative_tolerance,
        "diagnostic_top_k": policy.diagnostic_top_k,
    }


def test_runtime_classifies_narrower_stable_broader_and_goal_move() -> None:
    index = _index()
    estimator, _ = _estimator(index)
    broad_intent = _intent(version=1)
    broad = estimator.estimate(
        session_id="session-1",
        intent=broad_intent,
        compiled=_compiled(broad_intent),
    )
    narrower_intent = _intent(
        version=2,
        preferences=(_semantic_preference("p1", "waterproof"),),
    )
    narrower = estimator.estimate(
        session_id="session-1",
        intent=narrower_intent,
        compiled=_compiled(narrower_intent),
        previous=broad,
    )
    stable = estimator.estimate(
        session_id="session-1",
        intent=narrower_intent,
        compiled=_compiled(narrower_intent),
        previous=narrower,
    )
    broader_intent = _intent(version=3)
    broader = estimator.estimate(
        session_id="session-1",
        intent=broader_intent,
        compiled=_compiled(broader_intent),
        previous=stable,
    )
    moved_intent = _intent(goal="pearl earrings", version=4)
    moved = estimator.estimate(
        session_id="session-1",
        intent=moved_intent,
        compiled=_compiled(moved_intent),
        previous=broader,
        goal_switched=True,
    )

    assert broad.direction is IntentVolumeDirection.INITIAL
    assert narrower.direction is IntentVolumeDirection.NARROWER
    assert stable.direction is IntentVolumeDirection.STABLE
    assert broader.direction is IntentVolumeDirection.BROADER
    assert moved.direction is IntentVolumeDirection.MOVED
    assert narrower.transparency is not None and broad.transparency is not None
    assert narrower.transparency > broad.transparency
    assert broader.transparency == pytest.approx(broad.transparency)
    assert moved.change is not None


def test_hard_evidence_affects_volume_and_diagnostics_without_zeroing_mass() -> None:
    index = _index()
    estimator, _ = _estimator(index)
    preference = _hard_preference("hard-red")
    intent = _intent(preferences=(preference,))
    constraint = CompiledHardConstraint(
        preference_id=preference.id,
        facet="color",
        operator=Operator.EQ,
        value="red",
        policy=ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
    )

    estimate = estimator.estimate(
        session_id="session-hard",
        intent=intent,
        compiled=_compiled(intent, hard_constraints=(constraint,)),
    )

    assert estimate.remaining_intent_volume is not None
    assert estimate.remaining_intent_volume > 0.0
    assert estimate.diagnostics.hard_factor_count == 1
    assert estimate.diagnostics.top_all_hard_compliance == pytest.approx(0.5)
    assert estimate.diagnostics.top_mean_hard_factor_compliance == pytest.approx(0.5)


def test_relaxed_hard_constraint_falls_back_to_semantics_and_marks_degraded() -> None:
    index = _index()
    estimator, embedder = _estimator(index, relaxed_ids={"hard-red"})
    preference = _hard_preference("hard-red")
    intent = _intent(preferences=(preference,))
    constraint = CompiledHardConstraint(
        preference_id=preference.id,
        facet="color",
        operator=Operator.EQ,
        value="red",
        policy=ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE,
    )

    estimate = estimator.estimate(
        session_id="session-relaxed",
        intent=intent,
        compiled=_compiled(intent, hard_constraints=(constraint,)),
    )

    assert estimate.diagnostics.status is IntentVolumeStatus.DEGRADED
    assert estimate.diagnostics.reason_codes == ("hard_constraint_relaxed",)
    assert estimate.diagnostics.relaxed_hard_preference_ids == ("hard-red",)
    assert any("color: red" in text for text in embedder.calls)


def test_unsearchable_intent_is_explicitly_unavailable_and_json_safe() -> None:
    index = _index()
    estimator, _ = _estimator(index)
    intent = _intent(goal=None)

    estimate = estimator.estimate(
        session_id="session-vague",
        intent=intent,
        compiled=_compiled(intent),
        open_facets=("style", "category", "style"),
    )
    payload = estimate.as_payload()

    assert estimate.transparency is None
    assert estimate.direction is IntentVolumeDirection.UNAVAILABLE
    assert estimate.diagnostics.status is IntentVolumeStatus.UNAVAILABLE
    assert estimate.diagnostics.reason_codes == ("intent_not_searchable",)
    assert payload["transparency"] is None
    assert payload["direction"] == "unavailable"
    assert payload["diagnostics"]["open_facets"] == ["category", "style"]  # type: ignore[index]


def test_semantic_only_state_without_goal_remains_measurable() -> None:
    index = _index()
    estimator, _ = _estimator(index)
    gift = _semantic_preference("gift", "small anniversary gift")
    broad_intent = _intent(goal=None, preferences=(gift,), version=1)
    broad_compiled = replace(
        _compiled(broad_intent),
        q_sem="Preference: small anniversary gift.",
        search_ready=True,
    )
    broad = estimator.estimate(
        session_id="session-semantic",
        intent=broad_intent,
        compiled=broad_compiled,
    )
    necklace_intent = _intent(goal="necklace", preferences=(gift,), version=2)
    necklace = estimator.estimate(
        session_id="session-semantic",
        intent=necklace_intent,
        compiled=_compiled(necklace_intent),
        previous=broad,
    )

    assert broad.transparency is not None
    assert broad.goal_reference_volume is None
    assert broad.direction is IntentVolumeDirection.INITIAL
    assert necklace.direction is IntentVolumeDirection.NARROWER


def test_goal_wording_change_is_not_a_move_without_explicit_switch_evidence() -> None:
    index = _index()
    estimator, _ = _estimator(index)
    broad_intent = _intent(goal="footwear", version=1)
    broad = estimator.estimate(
        session_id="session-refine",
        intent=broad_intent,
        compiled=_compiled(broad_intent),
    )
    refined_intent = _intent(
        goal="boots",
        preferences=(_semantic_preference("winter", "waterproof"),),
        version=2,
    )
    refined = estimator.estimate(
        session_id="session-refine",
        intent=refined_intent,
        compiled=_compiled(refined_intent),
        previous=broad,
    )

    assert refined.direction is IntentVolumeDirection.NARROWER


def test_factor_scores_are_cached_across_unchanged_turns() -> None:
    index = _index()
    estimator, embedder = _estimator(index)
    intent = _intent()

    first = estimator.estimate(
        session_id="session-cache",
        intent=intent,
        compiled=_compiled(intent),
    )
    estimator.estimate(
        session_id="session-cache",
        intent=replace(intent, version=2),
        compiled=replace(_compiled(intent), intent_version=2),
        previous=first,
    )

    assert embedder.calls == ["Product goal: running shoes"]


def test_density_cache_loader_enforces_index_and_temperature_binding(tmp_path: Path) -> None:
    index = _index()
    path = tmp_path / "density.npz"
    np.savez_compressed(
        path,
        **{
            "index_id": np.asarray(index.index_id),
            "temperatures": np.asarray((0.025,), dtype=np.float64),
            "density_0.025": np.ones(4, dtype=np.float32),
        },
    )

    density = load_catalog_density(path, dense_index=index, temperature=0.025)

    assert density.index_id == index.index_id
    assert density.values.tolist() == [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="requested temperature"):
        load_catalog_density(path, dense_index=index, temperature=0.05)


def test_previous_estimate_cannot_cross_sessions() -> None:
    index = _index()
    estimator, _ = _estimator(index)
    intent = _intent()
    previous = estimator.estimate(
        session_id="session-a",
        intent=intent,
        compiled=_compiled(intent),
    )

    with pytest.raises(ValueError, match="another session"):
        estimator.estimate(
            session_id="session-b",
            intent=replace(intent, version=2),
            compiled=replace(_compiled(intent), intent_version=2),
            previous=previous,
        )
