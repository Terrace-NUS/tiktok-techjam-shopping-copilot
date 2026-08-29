"""One-call-plus-one-repair orchestration for Query Understanding."""

from __future__ import annotations

from typing import Protocol

from shopping_copilot.session_context import IntentState

from .errors import QueryUnderstandingError, QueryUnderstandingErrorCode
from .models import (
    ProviderResult,
    ReconcileRequest,
    ResolvedTurnIntent,
    UnderstandingTrace,
)
from .planner import IntentMaterializer


class UnderstandingProvider(Protocol):
    """Provider seam implemented by DeepSeek and deterministic test fakes."""

    def reconcile(
        self,
        request: ReconcileRequest,
        *,
        repair_instruction: str | None = None,
    ) -> ProviderResult: ...


class QueryUnderstandingService:
    """Resolve an intent atomically, with at most one fresh repair attempt."""

    __slots__ = ("_materializer", "_provider")

    def __init__(
        self,
        *,
        provider: UnderstandingProvider,
        materializer: IntentMaterializer,
    ) -> None:
        self._provider = provider
        self._materializer = materializer

    def resolve(
        self,
        *,
        current: IntentState,
        request: ReconcileRequest,
    ) -> ResolvedTurnIntent:
        attempts = []
        repair_instruction: str | None = None
        for attempt_index in range(2):
            try:
                provider_result = self._provider.reconcile(
                    request,
                    repair_instruction=repair_instruction,
                )
                attempts.append(provider_result.trace)
                materialized = self._materializer.materialize(
                    current=current,
                    request=request,
                    frame=provider_result.frame,
                )
            except QueryUnderstandingError as error:
                if error.code in _NON_REPAIRABLE_CODES:
                    raise
                if attempt_index == 1:
                    raise QueryUnderstandingError(
                        code=QueryUnderstandingErrorCode.REPAIR_EXHAUSTED,
                        details=(
                            ("attempt_count", attempt_index + 1),
                            ("last_error", error.code.value),
                            (
                                "last_path",
                                ".".join(str(item) for item in error.path) or "root",
                            ),
                            *tuple((f"last_detail_{key}", value) for key, value in error.details),
                        ),
                    ) from error
                repair_instruction = _repair_instruction(error, request=request)
                continue
            return ResolvedTurnIntent(
                update=materialized.update,
                final_intent=materialized.final_intent,
                feedback=materialized.feedback,
                directives=materialized.directives,
                clarification=materialized.clarification,
                trace=UnderstandingTrace(
                    attempts=tuple(attempts),
                    interpretation_summary=provider_result.frame.summary,
                    semantic_fallback_facets=materialized.semantic_fallback_facets,
                    ignored_dont_care_facets=materialized.ignored_dont_care_facets,
                ),
            )
        raise AssertionError("two-attempt loop must return or raise")


_NON_REPAIRABLE_CODES = frozenset(
    {
        QueryUnderstandingErrorCode.MISSING_API_KEY,
        QueryUnderstandingErrorCode.PROVIDER_AUTH,
        QueryUnderstandingErrorCode.PROVIDER_RATE_LIMIT,
        QueryUnderstandingErrorCode.PROVIDER_TIMEOUT,
        QueryUnderstandingErrorCode.PROVIDER_UNAVAILABLE,
        QueryUnderstandingErrorCode.STALE_INTENT_VERSION,
    }
)


def _repair_instruction(
    error: QueryUnderstandingError,
    *,
    request: ReconcileRequest,
) -> str:
    location = ".".join(str(item) for item in error.path) or "root"
    details = "; ".join(f"{key}={value}" for key, value in error.details)
    suffix = "" if not details else f"; details={details}"
    allowed = ",".join(request.allowed_dont_care_facets)
    return (
        f"local_error={error.code.value}; path={location}{suffix}; "
        f"allowed_dont_care_facets=[{allowed}]; "
        "omit an old active ref to remove only that preference; use dont_care_facets "
        "only when the whole listed facet is explicitly irrelevant; use goal.revise "
        "to remove stale constraints while keeping the same product task"
    )
