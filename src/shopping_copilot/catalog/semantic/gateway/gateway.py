"""Catalog-semantic invariant boundary around the unchanged intent reducer."""

from __future__ import annotations

from typing import NoReturn, cast

from shopping_copilot.session_context import (
    AddPreference,
    ClearFacet,
    FacetAuthority,
    FacetKind,
    FacetRegistry,
    FacetStats,
    IntentState,
    Operator,
    Preference,
    RemovePreference,
    ReplaceFacet,
    SearchBelief,
    SetDontCare,
    StateUpdateBatch,
    SwitchGoal,
    reduce_intent,
    validate_intent_state,
    validate_search_belief,
    validate_state_update_batch,
    with_retrieval_derived_facets,
)

from ..release import VerifiedCatalogSemanticRelease
from ..runtime import (
    SYSTEM_PRODUCT_CATEGORY_FACET_ID,
    ExactCapabilityIndex,
    project_session_facet_registry,
)
from .equality import exact_domain_equal
from .errors import CatalogGatewayError, CatalogGatewayErrorCode


class CatalogSemanticGateway:
    """Validate catalog-sensitive intent and belief state against one release."""

    __slots__ = (
        "_capabilities",
        "_category_scope_ids",
        "_registry",
        "_release",
    )

    def __init__(self, release: VerifiedCatalogSemanticRelease) -> None:
        if type(release) is not VerifiedCatalogSemanticRelease:
            raise TypeError("gateway requires an exact VerifiedCatalogSemanticRelease")
        self._release = release
        self._registry = with_retrieval_derived_facets(
            project_session_facet_registry(
                runtime_registry=release.runtime_registry,
                runtime_lexicon=release.runtime_value_lexicon,
                category_registry=release.category_registry,
                capabilities=release.effective_capabilities,
            )
        )
        self._capabilities = ExactCapabilityIndex(release.effective_capabilities)
        self._category_scope_ids = frozenset(scope.id for scope in release.category_registry.scopes)

    @property
    def release_id(self) -> str:
        """Return the single semantic release identity accepted by this gateway."""

        return self._release.release_id

    @property
    def registry(self) -> FacetRegistry:
        """Return the immutable projected registry for read-only composition."""

        return self._registry

    def preview(
        self,
        current: IntentState,
        batch: StateUpdateBatch,
        *,
        catalog_semantic_release_id: str,
    ) -> IntentState:
        """Reduce and validate a batch without granting authority to commit it."""

        self.require_release(catalog_semantic_release_id)
        validate_intent_state(current, self._registry)
        previous_scope = self._validate_complete_intent(current)
        self._validate_reserved_operation_matrix(current, batch)
        validate_state_update_batch(batch, self._registry)
        candidate = reduce_intent(current, batch, self._registry)
        final_scope = self._effective_category_scope(candidate)
        self._validate_complete_intent(
            candidate,
            prior=current,
            category_changed=final_scope != previous_scope,
        )
        return candidate

    def validate_intent(
        self,
        intent: IntentState,
        *,
        catalog_semantic_release_id: str,
    ) -> None:
        """Validate one already materialized active intent under the bound release."""

        self.require_release(catalog_semantic_release_id)
        validate_intent_state(intent, self._registry)
        self._validate_complete_intent(intent)

    def validate_search_belief(
        self,
        belief: SearchBelief,
        *,
        intent: IntentState,
        catalog_semantic_release_id: str,
    ) -> None:
        """Validate current Probe values and permissions in the active exact scope."""

        self.require_release(catalog_semantic_release_id)
        validate_intent_state(intent, self._registry)
        scope_id = self._validate_complete_intent(intent)
        if type(belief) is SearchBelief and type(belief.facet_stats) is tuple:
            for index, stats in enumerate(belief.facet_stats):
                if type(stats) is FacetStats and stats.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                    _fail(
                        CatalogGatewayErrorCode.PROBE_FACET_NOT_ELIGIBLE,
                        path=("facet_stats", index, "facet"),
                    )
        validate_search_belief(belief, self._registry)
        if belief.based_on_intent_version != intent.version:
            _fail(
                CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF,
                path=("based_on_intent_version",),
                details=(
                    ("actual", belief.based_on_intent_version),
                    ("expected", intent.version),
                ),
            )
        for index, stats in enumerate(belief.facet_stats):
            path = ("facet_stats", index)
            spec = self._registry.require(stats.facet)
            if spec.authority is FacetAuthority.CATALOG_VERIFIED:
                permission = self._capabilities.lookup(stats.facet, scope_id)
                if not permission.probe_eligible:
                    _fail(
                        CatalogGatewayErrorCode.PROBE_FACET_NOT_ELIGIBLE,
                        path=path + ("facet",),
                        details=(("facet", stats.facet), ("scope", scope_id)),
                    )
            for value_index, value_mass in enumerate(stats.top_values):
                try:
                    normalized = spec.normalizer(value_mass.value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise CatalogGatewayError(
                        code=CatalogGatewayErrorCode.VALUE_NOT_GROUNDED,
                        path=path + ("top_values", value_index, "value"),
                        details=(("facet", stats.facet),),
                    ) from error
                if not exact_domain_equal(value_mass.value, normalized):
                    _fail(
                        CatalogGatewayErrorCode.VALUE_NOT_GROUNDED,
                        path=path + ("top_values", value_index, "value"),
                        details=(("facet", stats.facet),),
                    )

    def require_release(self, release_id: str) -> None:
        """Reject a caller attempting to cross the store's release boundary."""

        if type(release_id) is not str or release_id != self._release.release_id:
            _fail(
                CatalogGatewayErrorCode.RELEASE_MISMATCH,
                path=("catalog_semantic_release_id",),
            )

    def _validate_reserved_operation_matrix(
        self,
        current: IntentState,
        batch: StateUpdateBatch,
    ) -> None:
        if type(batch) is not StateUpdateBatch or type(batch.operations) is not tuple:
            return
        current_category = _reserved_preferences(current)
        current_category_ids = {item.id for item in current_category}
        batch_category_ids = set(current_category_ids)
        for operation in batch.operations:
            if (
                type(operation) is ReplaceFacet
                and operation.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID
                and type(operation.preferences) is tuple
            ):
                batch_category_ids.update(
                    item.id
                    for item in operation.preferences
                    if type(item) is Preference and item.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID
                )
        replacement_indexes: list[int] = []
        for index, operation in enumerate(batch.operations):
            if type(operation) is AddPreference:
                if operation.preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                    _invalid_reserved(index)
            elif type(operation) is ReplaceFacet:
                if operation.facet != SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                    continue
                replacement_indexes.append(index)
                if type(operation.preferences) is not tuple or len(operation.preferences) != 1:
                    _invalid_reserved(index)
                self._validate_reserved_preference(operation.preferences[0], index=index)
            elif type(operation) is RemovePreference:
                if batch_category_ids.intersection(operation.preference_ids):
                    _invalid_reserved(index)
            elif type(operation) is ClearFacet:
                if operation.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                    _invalid_reserved(index)
            elif type(operation) is SetDontCare:
                if operation.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                    _invalid_reserved(index)
            elif type(operation) is SwitchGoal:
                carried_category = current_category_ids.intersection(operation.carry_preference_ids)
                if len(carried_category) > 1:
                    _invalid_reserved(index)
        if len(replacement_indexes) > 1:
            _invalid_reserved(replacement_indexes[1])
        if replacement_indexes:
            expected = 1 if batch.operations and type(batch.operations[0]) is SwitchGoal else 0
            if replacement_indexes[0] != expected:
                _invalid_reserved(replacement_indexes[0])

    def _validate_complete_intent(
        self,
        intent: IntentState,
        *,
        prior: IntentState | None = None,
        category_changed: bool = False,
    ) -> str:
        scope_id = self._effective_category_scope(intent)
        prior_preference_ids = (
            frozenset(item.id for item in prior.preferences) if prior is not None else frozenset()
        )
        prior_dont_care = prior.dont_care_facets if prior is not None else frozenset()
        for index, preference in enumerate(intent.preferences):
            if preference.facet is None or preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                continue
            spec = self._registry.require(preference.facet)
            if spec.authority is FacetAuthority.CATALOG_VERIFIED:
                permission = self._capabilities.lookup(preference.facet, scope_id)
                if not permission.intent_committable:
                    retained = preference.id in prior_preference_ids
                    code = (
                        CatalogGatewayErrorCode.INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE
                        if category_changed and retained
                        else CatalogGatewayErrorCode.FACET_NOT_COMMITTABLE
                    )
                    _fail(
                        code,
                        path=("preferences", index, "facet"),
                        details=(("facet", preference.facet), ("scope", scope_id)),
                    )
            self._validate_grounded_fixed_point(preference, index=index)
        for facet in sorted(intent.dont_care_facets):
            if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
                _fail(
                    CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION,
                    path=("dont_care_facets", facet),
                )
            spec = self._registry.require(facet)
            if spec.authority is FacetAuthority.CATALOG_VERIFIED:
                permission = self._capabilities.lookup(facet, scope_id)
                if not permission.intent_committable:
                    code = (
                        CatalogGatewayErrorCode.INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE
                        if category_changed and facet in prior_dont_care
                        else CatalogGatewayErrorCode.FACET_NOT_COMMITTABLE
                    )
                    _fail(
                        code,
                        path=("dont_care_facets", facet),
                        details=(("facet", facet), ("scope", scope_id)),
                    )
        return scope_id

    def _effective_category_scope(self, intent: IntentState) -> str:
        preferences = _reserved_preferences(intent)
        if not preferences:
            return self._release.category_registry.root_scope_id
        if len(preferences) != 1:
            _fail(
                CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION,
                path=("preferences",),
            )
        preference = preferences[0]
        self._validate_reserved_preference(preference)
        return cast(str, preference.value)

    def _validate_reserved_preference(
        self,
        preference: Preference,
        *,
        index: int | None = None,
    ) -> None:
        path = ("preferences",) if index is None else ("operations", index, "preferences")
        if (
            type(preference) is not Preference
            or preference.facet != SYSTEM_PRODUCT_CATEGORY_FACET_ID
            or preference.operator is not Operator.EQ
            or type(preference.value) is not str
        ):
            _fail(
                CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION,
                path=path,
                operation_index=index,
            )
        if preference.value not in self._category_scope_ids:
            _fail(
                CatalogGatewayErrorCode.UNKNOWN_CATEGORY_SCOPE,
                path=path + ("value",),
                operation_index=index,
                details=(("scope", preference.value),),
            )

    def _validate_grounded_fixed_point(self, preference: Preference, *, index: int) -> None:
        assert preference.facet is not None
        assert preference.operator is not None
        assert preference.value is not None
        spec = self._registry.require(preference.facet)
        try:
            normalized = self._registry.normalize_value(
                preference.facet,
                preference.operator,
                preference.value,
            )
        except (TypeError, ValueError) as error:
            raise CatalogGatewayError(
                code=CatalogGatewayErrorCode.VALUE_NOT_GROUNDED,
                path=("preferences", index, "value"),
                details=(("facet", preference.facet),),
            ) from error
        if not exact_domain_equal(preference.value, normalized):
            _fail(
                CatalogGatewayErrorCode.VALUE_NOT_GROUNDED,
                path=("preferences", index, "value"),
                details=(("facet", preference.facet),),
            )
        if spec.kind is FacetKind.NUMERIC and type(preference.value) not in (int, float):
            _fail(
                CatalogGatewayErrorCode.VALUE_NOT_GROUNDED,
                path=("preferences", index, "value"),
                details=(("facet", preference.facet),),
            )


def _reserved_preferences(intent: IntentState) -> tuple[Preference, ...]:
    return tuple(
        item for item in intent.preferences if item.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID
    )


def _invalid_reserved(operation_index: int) -> NoReturn:
    _fail(
        CatalogGatewayErrorCode.INVALID_RESERVED_CATEGORY_OPERATION,
        path=("operations", operation_index),
        operation_index=operation_index,
    )


def _fail(
    code: CatalogGatewayErrorCode,
    *,
    path: tuple[str | int, ...] = (),
    operation_index: int | None = None,
    details: tuple[tuple[str, str | int | float | bool], ...] = (),
) -> NoReturn:
    raise CatalogGatewayError(
        code=code,
        path=path,
        operation_index=operation_index,
        details=details,
    )
