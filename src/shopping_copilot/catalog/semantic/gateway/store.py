"""Release-bound atomic authority over the unchanged in-memory session store."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from shopping_copilot.session_context import (
    InMemorySessionStore,
    IntentState,
    InteractionContext,
    ProfilePrior,
    SearchBelief,
    SessionContext,
    SessionState,
    SessionTransaction,
    StateUpdateBatch,
    TurnRecord,
)

from ..release import VerifiedCatalogSemanticRelease
from .envelope import decode_catalog_bound_session, encode_catalog_bound_session
from .equality import exact_domain_equal
from .errors import CatalogGatewayError, CatalogGatewayErrorCode
from .gateway import CatalogSemanticGateway


@dataclass(frozen=True, slots=True)
class CatalogProbeToken:
    """Opaque process-local, one-use proof of a release-bound Probe result."""

    _authority: object
    _release_id: str
    _session_id: str
    _transaction_token: object
    _captured_context: SessionContext
    _expected_final_intent: IntentState
    _belief: SearchBelief
    _used: bool = False


class _CatalogProbeProducer:
    """Private CS8 handoff; application code never receives this producer."""

    __slots__ = ("_authority", "_gateway")

    def __init__(self, gateway: CatalogSemanticGateway, authority: object) -> None:
        self._gateway = gateway
        self._authority = authority

    def issue_token(
        self,
        transaction: CatalogBoundSessionTransaction,
        *,
        expected_final_intent: IntentState,
        belief: SearchBelief,
    ) -> CatalogProbeToken:
        """Attest a belief produced by the future private CS8 Probe implementation."""

        captured = transaction.context
        self._gateway.validate_search_belief(
            belief,
            intent=expected_final_intent,
            catalog_semantic_release_id=self._gateway.release_id,
        )
        return CatalogProbeToken(
            _authority=self._authority,
            _release_id=self._gateway.release_id,
            _session_id=captured.session_id,
            _transaction_token=transaction._probe_transaction_token,
            _captured_context=captured,
            _expected_final_intent=expected_final_intent,
            _belief=belief,
        )


class CatalogBoundSessionStore:
    """Own one verified release, one projected registry, and one private raw store."""

    __slots__ = ("_gateway", "_probe_authority", "_probe_producer", "_store")

    def __init__(self, release: VerifiedCatalogSemanticRelease) -> None:
        self._gateway = CatalogSemanticGateway(release)
        self._store = InMemorySessionStore(self._gateway.registry)
        self._probe_authority = object()
        self._probe_producer = _CatalogProbeProducer(
            self._gateway,
            self._probe_authority,
        )

    @property
    def catalog_semantic_release_id(self) -> str:
        """Return the immutable release identity owned by this store."""

        return self._gateway.release_id

    def reset(
        self,
        *,
        session_id: str,
        profile: ProfilePrior | None = None,
        expected_release_id: str | None = None,
    ) -> SessionContext:
        """Create one catalog-bound empty session."""

        self._require_expected_release(expected_release_id)
        context = self._store.reset(session_id=session_id, profile=profile)
        self._gateway.validate_intent(
            context.state.intent,
            catalog_semantic_release_id=self._gateway.release_id,
        )
        return context

    def get(
        self,
        session_id: str,
        *,
        expected_release_id: str | None = None,
    ) -> SessionContext:
        """Read the current immutable snapshot from the bound store."""

        self._require_expected_release(expected_release_id)
        return self._store.get(session_id)

    def turn(
        self,
        *,
        session_id: str,
        turn: int,
        expected_release_id: str | None = None,
    ) -> CatalogBoundSessionTransaction:
        """Return the only transaction type allowed to commit application state."""

        self._require_expected_release(expected_release_id)
        return CatalogBoundSessionTransaction(
            raw_transaction=self._store.turn(session_id=session_id, turn=turn),
            gateway=self._gateway,
            probe_authority=self._probe_authority,
            turn=turn,
        )

    def encode(
        self,
        context: SessionContext,
        *,
        expected_release_id: str | None = None,
    ) -> bytes:
        """Encode one replay-verified catalog-bound session envelope."""

        self._require_expected_release(expected_release_id)
        return encode_catalog_bound_session(context, self._gateway)

    def decode(
        self,
        data: bytes,
        *,
        expected_release_id: str | None = None,
    ) -> SessionContext:
        """Decode and replay a catalog-bound session without inserting it into the store."""

        self._require_expected_release(expected_release_id)
        return decode_catalog_bound_session(data, self._gateway)

    def _require_expected_release(self, release_id: str | None) -> None:
        if release_id is not None:
            self._gateway.require_release(release_id)


class CatalogBoundSessionTransaction:
    """Single-use wrapper that reruns gateway checks inside the held raw-store lock."""

    __slots__ = (
        "_commit_attempted",
        "_entered",
        "_gateway",
        "_probe_authority",
        "_probe_transaction_token",
        "_raw_transaction",
        "_turn",
    )

    def __init__(
        self,
        *,
        raw_transaction: SessionTransaction,
        gateway: CatalogSemanticGateway,
        probe_authority: object,
        turn: int,
    ) -> None:
        self._raw_transaction = raw_transaction
        self._gateway = gateway
        self._probe_authority = probe_authority
        self._probe_transaction_token = object()
        self._turn = turn
        self._entered = False
        self._commit_attempted = False

    @property
    def context(self) -> SessionContext:
        """Return the immutable snapshot captured under the per-session lock."""

        if not self._entered:
            raise CatalogGatewayError(code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH)
        return self._raw_transaction.context

    def __enter__(self) -> CatalogBoundSessionTransaction:
        self._raw_transaction.__enter__()
        self._entered = True
        return self

    def preview_update(self, batch: StateUpdateBatch) -> IntentState:
        """Preview a QU-planned update against the snapshot held by this turn."""

        captured = self.context
        if type(batch) is not StateUpdateBatch or batch.turn != self._turn:
            raise CatalogGatewayError(
                code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH,
                path=("accepted_update", "turn"),
            )
        return self._gateway.preview(
            captured.state.intent,
            batch,
            catalog_semantic_release_id=self._gateway.release_id,
        )

    def commit(
        self,
        next_context: SessionContext,
        *,
        probe_token: CatalogProbeToken | None = None,
    ) -> SessionContext:
        """Validate catalog authority and perform the private raw commit atomically."""

        captured = self.context
        if self._commit_attempted:
            raise CatalogGatewayError(code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH)
        self._commit_attempted = True
        if probe_token is not None:
            _consume_token(probe_token)
        appended = _require_appended_turn(captured, next_context, expected_turn=self._turn)
        batch = appended.accepted_update
        if batch is None:
            expected_intent = captured.state.intent
        else:
            expected_intent = self._gateway.preview(
                captured.state.intent,
                batch,
                catalog_semantic_release_id=self._gateway.release_id,
            )
        if not exact_domain_equal(expected_intent, next_context.state.intent):
            raise CatalogGatewayError(
                code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH,
                path=("state", "intent"),
            )
        self._gateway.validate_intent(
            next_context.state.intent,
            catalog_semantic_release_id=self._gateway.release_id,
        )
        belief = next_context.state.search_belief
        if belief is not None:
            self._gateway.validate_search_belief(
                belief,
                intent=next_context.state.intent,
                catalog_semantic_release_id=self._gateway.release_id,
            )
        self._validate_probe_authority(
            captured=captured,
            next_context=next_context,
            probe_token=probe_token,
        )
        return self._raw_transaction.commit(next_context)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self._raw_transaction.__exit__(exc_type, exc_value, traceback)
        finally:
            self._entered = False

    def _validate_probe_authority(
        self,
        *,
        captured: SessionContext,
        next_context: SessionContext,
        probe_token: CatalogProbeToken | None,
    ) -> None:
        previous = captured.state.search_belief
        candidate = next_context.state.search_belief
        changed = candidate is not None and not exact_domain_equal(candidate, previous)
        if not changed:
            if probe_token is not None:
                raise CatalogGatewayError(
                    code=CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF,
                    path=("probe_token",),
                )
            return
        if probe_token is None:
            raise CatalogGatewayError(
                code=CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF,
                path=("state", "search_belief"),
            )
        trusted = (
            type(probe_token) is CatalogProbeToken
            and probe_token._authority is self._probe_authority
            and probe_token._release_id == self._gateway.release_id
            and probe_token._session_id == captured.session_id
            and probe_token._transaction_token is self._probe_transaction_token
            and probe_token._captured_context is captured
            and exact_domain_equal(
                probe_token._expected_final_intent,
                next_context.state.intent,
            )
            and candidate is not None
            and exact_domain_equal(probe_token._belief, candidate)
        )
        if not trusted:
            raise CatalogGatewayError(
                code=CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF,
                path=("probe_token",),
            )


def _consume_token(token: CatalogProbeToken) -> None:
    if type(token) is not CatalogProbeToken or token._used:
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.UNTRUSTED_SEARCH_BELIEF,
            path=("probe_token",),
        )
    object.__setattr__(token, "_used", True)


def _require_appended_turn(
    captured: SessionContext,
    next_context: SessionContext,
    *,
    expected_turn: int,
) -> TurnRecord:
    if (
        type(next_context) is not SessionContext
        or next_context.session_id != captured.session_id
        or type(next_context.state) is not SessionState
        or type(next_context.state.interaction) is not InteractionContext
        or type(next_context.state.interaction.turns) is not tuple
    ):
        raise CatalogGatewayError(code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH)
    previous_turns = captured.state.interaction.turns
    next_turns = next_context.state.interaction.turns
    if (
        len(next_turns) != len(previous_turns) + 1
        or not exact_domain_equal(next_turns[:-1], previous_turns)
        or type(next_turns[-1]) is not TurnRecord
        or next_turns[-1].turn != expected_turn
    ):
        raise CatalogGatewayError(
            code=CatalogGatewayErrorCode.CATALOG_COMMIT_MISMATCH,
            path=("state", "interaction", "turns"),
        )
    return next_turns[-1]
