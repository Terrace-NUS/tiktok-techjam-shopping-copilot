"""Build deterministic, model-safe views of trusted local session state."""

from __future__ import annotations

from shopping_copilot.catalog.semantic.category import CategoryRegistry
from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
from shopping_copilot.session_context import IntentState
from shopping_copilot.session_context.wide_facets import RETRIEVAL_DERIVED_FACET_IDS

from .models import (
    ActivePreferenceView,
    CategoryOption,
    ReconcileRequest,
    ShownProductView,
)

DEFAULT_DONT_CARE_FACETS = tuple(sorted((*RETRIEVAL_DERIVED_FACET_IDS, "price")))


def category_options_from_registry(registry: CategoryRegistry) -> tuple[CategoryOption, ...]:
    """Project category scopes to stable turn-local references."""

    if type(registry) is not CategoryRegistry:
        raise TypeError("category options require an exact CategoryRegistry")
    ordered = tuple(
        sorted(
            registry.scopes,
            key=lambda scope: (scope.id != registry.root_scope_id, scope.id),
        )
    )
    return tuple(
        CategoryOption(
            ref=f"category_{index}",
            scope_id=scope.id,
            label=scope.label,
            is_root=scope.id == registry.root_scope_id,
        )
        for index, scope in enumerate(ordered)
    )


def build_reconcile_request(
    *,
    turn: int,
    latest_utterance: str,
    current_intent: IntentState,
    category_options: tuple[CategoryOption, ...],
    shown_products: tuple[ShownProductView, ...] = (),
    last_assistant_message: str | None = None,
    last_question: str | None = None,
    allowed_dont_care_facets: tuple[str, ...] | None = None,
) -> ReconcileRequest:
    """Create the complete QU input without SearchBelief, C_t, or internal IDs."""

    if type(turn) is not int or turn < 1:
        raise ValueError("turn must be a positive integer")
    if type(latest_utterance) is not str or not latest_utterance.strip():
        raise ValueError("latest utterance must be non-empty")
    if type(current_intent) is not IntentState:
        raise TypeError("current_intent must be an exact IntentState")
    category_refs = {option.scope_id: option.ref for option in category_options}
    active = tuple(
        _active_preference_view(
            index=index,
            preference=preference,
            category_refs=category_refs,
        )
        for index, preference in enumerate(current_intent.preferences)
    )
    allowed = (
        DEFAULT_DONT_CARE_FACETS
        if allowed_dont_care_facets is None
        else tuple(sorted(set(allowed_dont_care_facets)))
    )
    if any(type(facet) is not str or not facet for facet in allowed):
        raise ValueError("allowed don't-care facets must be non-empty strings")
    return ReconcileRequest(
        turn=turn,
        base_intent_version=current_intent.version,
        latest_utterance=latest_utterance,
        current_goal=current_intent.goal,
        active_preferences=active,
        dont_care_facets=tuple(sorted(current_intent.dont_care_facets)),
        last_assistant_message=last_assistant_message,
        last_question=last_question,
        category_options=category_options,
        shown_products=shown_products,
        allowed_dont_care_facets=allowed,
    )


def request_payload(request: ReconcileRequest) -> dict[str, object]:
    """Return the exact JSON-compatible payload embedded in the user message."""

    return {
        "turn": request.turn,
        "base_intent_version": request.base_intent_version,
        "latest_utterance": request.latest_utterance,
        "current_intent": {
            "goal": request.current_goal,
            "active_preferences": [
                {
                    "ref": item.ref,
                    "facet": item.facet,
                    "relation": item.relation,
                    "value": list(item.value) if type(item.value) is tuple else item.value,
                    "meaning": item.meaning,
                    "strength": item.strength,
                    "source": item.source,
                }
                for item in request.active_preferences
            ],
            "dont_care_facets": list(request.dont_care_facets),
        },
        "interaction": {
            "last_assistant_message": request.last_assistant_message,
            "last_question": request.last_question,
            "shown_products": [
                {"ref": item.ref, "label": item.label} for item in request.shown_products
            ],
        },
        "category_options": [
            {"ref": item.ref, "label": item.label, "is_root": item.is_root}
            for item in request.category_options
        ],
        "allowed_dont_care_facets": list(request.allowed_dont_care_facets),
    }


def _active_preference_view(
    *,
    index: int,
    preference: object,
    category_refs: dict[str, str],
) -> ActivePreferenceView:
    from shopping_copilot.session_context import Preference

    assert type(preference) is Preference
    value = preference.value
    facet = preference.facet
    if facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID and type(value) is str:
        value = category_refs.get(value, "category_out_of_view")
        facet = "category"
    if preference.operator is not None:
        relation = preference.operator.value
    else:
        assert preference.semantic_polarity is not None
        relation = f"semantic_{preference.semantic_polarity.value}"
    meaning = preference.semantic_text or preference.evidence_text
    return ActivePreferenceView(
        ref=f"active_{index}",
        facet=facet,
        relation=relation,
        value=value,
        meaning=meaning,
        strength=preference.commitment.value,
        source=preference.source.value,
    )
