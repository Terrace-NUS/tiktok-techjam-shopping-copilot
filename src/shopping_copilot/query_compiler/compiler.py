"""Deterministically compile accepted Session Context intent for retrieval."""

from __future__ import annotations

from collections.abc import Iterable

from shopping_copilot.catalog.semantic.category import CategoryRegistry
from shopping_copilot.catalog.semantic.runtime import SYSTEM_PRODUCT_CATEGORY_FACET_ID
from shopping_copilot.query_understanding import ResolvedTurnIntent
from shopping_copilot.session_context import (
    Commitment,
    Operator,
    Preference,
    PreferenceSource,
    SemanticPolarity,
)
from shopping_copilot.session_context.models import PreferenceValue, ScalarValue
from shopping_copilot.session_context.wide_facets import RETRIEVAL_DERIVED_FACET_IDS

from .errors import QueryCompilerError
from .models import (
    COMPILED_QUERY_SCHEMA,
    QUERY_COMPILER_VERSION,
    CompilationTarget,
    CompiledDirectives,
    CompiledHardConstraint,
    CompiledQuery,
    CompiledRankingPreference,
    ConstraintPolicy,
    DiversityDirective,
    PreferenceCompilationTrace,
    RankingReason,
)

_POSITIVE_OPERATORS = frozenset({Operator.EQ, Operator.IN})
_NEGATIVE_OPERATORS = frozenset({Operator.NEQ, Operator.NOT_IN})
_NUMERIC_OPERATORS = frozenset({Operator.LT, Operator.LE, Operator.GT, Operator.GE})
_RETRIEVAL_HARD_FACETS = frozenset(RETRIEVAL_DERIVED_FACET_IDS)


class QueryCompiler:
    """Pure compiler bound to one catalog release and category vocabulary."""

    __slots__ = (
        "_catalog_semantic_release_id",
        "_category_labels",
        "_category_registry",
    )

    def __init__(
        self,
        *,
        catalog_semantic_release_id: str,
        category_registry: CategoryRegistry,
    ) -> None:
        if type(catalog_semantic_release_id) is not str or not catalog_semantic_release_id.strip():
            raise ValueError("catalog_semantic_release_id must be non-empty")
        if type(category_registry) is not CategoryRegistry:
            raise TypeError("category_registry must be an exact CategoryRegistry")
        self._catalog_semantic_release_id = catalog_semantic_release_id
        self._category_registry = category_registry
        self._category_labels = {scope.id: scope.label for scope in category_registry.scopes}

    def compile(self, resolved: ResolvedTurnIntent) -> CompiledQuery:
        """Compile one accepted, uncommitted QU result without model calls or writes."""

        if type(resolved) is not ResolvedTurnIntent:
            raise TypeError("resolved must be an exact ResolvedTurnIntent")

        intent = resolved.final_intent
        lexical_parts: list[str] = []
        semantic_parts: list[str] = []
        hard_constraints: list[CompiledHardConstraint] = []
        ranking_preferences: list[CompiledRankingPreference] = []
        trace: list[PreferenceCompilationTrace] = []

        if intent.goal is not None:
            goal = _clean_phrase(intent.goal)
            lexical_parts.append(goal)
            semantic_parts.append(f"Looking for {goal}.")

        for preference in intent.preferences:
            if self._is_root_category(preference):
                trace.append(
                    PreferenceCompilationTrace(
                        preference_id=preference.id,
                        targets=(CompilationTarget.NOOP,),
                        reason="root_category_removes_category_restriction",
                    )
                )
                continue

            targets: list[CompilationTarget] = []
            phrase = self._semantic_phrase(preference)
            if phrase is not None:
                semantic_parts.append(phrase)
                targets.append(CompilationTarget.Q_SEM)

            lexical_values = self._lexical_values(preference)
            if lexical_values:
                lexical_parts.extend(lexical_values)
                targets.append(CompilationTarget.Q_LEX)

            policy = _hard_constraint_policy(preference)
            if policy is not None:
                assert preference.facet is not None
                assert preference.operator is not None
                assert preference.value is not None
                hard_constraints.append(
                    CompiledHardConstraint(
                        preference_id=preference.id,
                        facet=preference.facet,
                        operator=preference.operator,
                        value=preference.value,
                        policy=policy,
                    )
                )
                targets.append(CompilationTarget.HARD_CONSTRAINT)
                reason = "explicit_structured_hard_requirement"
            else:
                ranking_reason = _ranking_reason(preference)
                ranking_preferences.append(
                    CompiledRankingPreference(
                        preference_id=preference.id,
                        facet=preference.facet,
                        operator=preference.operator,
                        value=preference.value,
                        semantic_text=preference.semantic_text,
                        semantic_polarity=preference.semantic_polarity,
                        commitment=preference.commitment,
                        source=preference.source,
                        reason=ranking_reason,
                    )
                )
                targets.append(CompilationTarget.RANKING_PREFERENCE)
                reason = ranking_reason.value

            trace.append(
                PreferenceCompilationTrace(
                    preference_id=preference.id,
                    targets=tuple(targets),
                    reason=reason,
                )
            )

        q_lex = " ".join(_deduplicate_phrases(lexical_parts))
        q_sem = " ".join(semantic_parts)
        directives = resolved.directives
        clarification = resolved.clarification
        return CompiledQuery(
            schema=COMPILED_QUERY_SCHEMA,
            compiler_version=QUERY_COMPILER_VERSION,
            catalog_id=self._category_registry.catalog_id,
            catalog_semantic_release_id=self._catalog_semantic_release_id,
            category_graph_id=self._category_registry.category_graph_id,
            intent_version=intent.version,
            q_lex=q_lex,
            q_sem=q_sem,
            search_ready=bool(q_sem),
            hard_constraints=tuple(hard_constraints),
            ranking_preferences=tuple(ranking_preferences),
            dont_care_facets=tuple(sorted(intent.dont_care_facets)),
            directives=CompiledDirectives(
                diversity=DiversityDirective(directives.diversity.value),
                comparison_requested=directives.comparison_requested,
                explanation_requested=directives.explanation_requested,
            ),
            requires_clarification=clarification.needed,
            clarification_reason=clarification.reason,
            trace=tuple(trace),
        )

    def _is_root_category(self, preference: Preference) -> bool:
        return (
            preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID
            and preference.value == self._category_registry.root_scope_id
        )

    def _category_label(self, scope_id: object) -> str:
        if type(scope_id) is not str:
            raise QueryCompilerError("category preference must contain scope IDs")
        try:
            return self._category_labels[scope_id]
        except KeyError as error:
            raise QueryCompilerError(
                "category preference is not bound to the active category registry"
            ) from error

    def _lexical_values(self, preference: Preference) -> tuple[str, ...]:
        operator = preference.operator
        if operator not in _POSITIVE_OPERATORS:
            return ()
        if preference.facet == "price" or preference.value is None:
            return ()
        if preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
            return tuple(self._category_label(value) for value in _values(preference.value))
        return tuple(_scalar_text(value) for value in _values(preference.value))

    def _semantic_phrase(self, preference: Preference) -> str | None:
        if preference.facet is None:
            assert preference.semantic_text is not None
            if preference.semantic_polarity is SemanticPolarity.NEGATIVE:
                label = "Avoid" if preference.commitment is Commitment.HARD else "Prefer avoiding"
                return f"{label}: {_sentence_fragment(preference.semantic_text)}."
            label = "Requirement" if preference.commitment is Commitment.HARD else "Preference"
            return f"{label}: {_sentence_fragment(preference.semantic_text)}."

        assert preference.operator is not None
        assert preference.value is not None
        strength = "Required" if preference.commitment is Commitment.HARD else "Preferred"
        if preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
            labels = ", ".join(self._category_label(value) for value in _values(preference.value))
            if preference.operator in _NEGATIVE_OPERATORS:
                label = "Exclude" if preference.commitment is Commitment.HARD else "Prefer avoiding"
                return f"{label} category: {labels}."
            return f"{strength} category: {labels}."
        if preference.facet == "price" and preference.operator in _NUMERIC_OPERATORS:
            return _price_phrase(preference, strength=strength)

        if preference.semantic_text is not None:
            label = "Requirement" if preference.commitment is Commitment.HARD else "Preference"
            return f"{label}: {_sentence_fragment(preference.semantic_text)}."

        values = ", ".join(_scalar_text(value) for value in _values(preference.value))
        if preference.operator in _NEGATIVE_OPERATORS:
            label = "Exclude" if preference.commitment is Commitment.HARD else "Prefer avoiding"
            return f"{label} {preference.facet}: {values}."
        return f"{strength} {preference.facet}: {values}."


def _hard_constraint_policy(preference: Preference) -> ConstraintPolicy | None:
    if (
        preference.commitment is not Commitment.HARD
        or preference.source is not PreferenceSource.USER_EXPLICIT
        or preference.facet is None
    ):
        return None
    if preference.facet == SYSTEM_PRODUCT_CATEGORY_FACET_ID:
        return ConstraintPolicy.VERIFIED_CATEGORY
    if preference.facet == "price":
        return ConstraintPolicy.CONSERVATIVE_PRICE
    if preference.facet in _RETRIEVAL_HARD_FACETS:
        return ConstraintPolicy.CLOSED_WORLD_RETRIEVAL_EVIDENCE
    return None


def _ranking_reason(preference: Preference) -> RankingReason:
    if preference.facet is None:
        return RankingReason.SEMANTIC_ONLY
    if preference.commitment is Commitment.SOFT:
        return RankingReason.SOFT_COMMITMENT
    if preference.source is not PreferenceSource.USER_EXPLICIT:
        return RankingReason.NON_EXPLICIT_SOURCE
    return RankingReason.UNSUPPORTED_HARD_FACET


def _price_phrase(preference: Preference, *, strength: str) -> str:
    assert preference.operator is not None
    value = preference.value
    if type(value) not in (int, float):
        raise QueryCompilerError("price preference must contain a numeric cent value")
    assert isinstance(value, (int, float))
    amount = float(value) / 100
    relation = {
        Operator.LT: "below",
        Operator.LE: "at most",
        Operator.GT: "above",
        Operator.GE: "at least",
    }[preference.operator]
    return f"{strength} price: {relation} USD {amount:.2f}."


def _values(value: PreferenceValue) -> tuple[ScalarValue, ...]:
    return value if isinstance(value, tuple) else (value,)


def _scalar_text(value: ScalarValue) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def _clean_phrase(value: str) -> str:
    return " ".join(value.split()).rstrip(". ")


def _sentence_fragment(value: str) -> str:
    return _clean_phrase(value)


def _deduplicate_phrases(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_phrase(value)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)
