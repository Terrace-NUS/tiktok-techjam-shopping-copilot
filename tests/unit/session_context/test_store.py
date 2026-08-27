"""Lifecycle, atomicity, and concurrency tests for the in-memory session store."""

from __future__ import annotations

from _thread import LockType
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event, Lock

import pytest

from shopping_copilot.session_context.aggregates import (
    InteractionContext,
    SessionContext,
    SessionState,
    TurnRecord,
)
from shopping_copilot.session_context.errors import ErrorCode, SessionContextError
from shopping_copilot.session_context.models import (
    CandidateMode,
    CertaintyEvidence,
    Commitment,
    IntentState,
    Operator,
    Preference,
    PreferenceSource,
    ProbeQuality,
    ProfilePrior,
    SearchBelief,
)
from shopping_copilot.session_context.operations import AddPreference, StateUpdateBatch
from shopping_copilot.session_context.reducer import reduce_intent
from shopping_copilot.session_context.registry import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    canonical_number,
    canonical_text,
)
from shopping_copilot.session_context.store import InMemorySessionStore, SessionTransaction

_PRESERVE_BELIEF = object()


class _FirstAcquireGate:
    """Pause the first lock acquisition and fail if the transaction tries a second."""

    def __init__(self, delegate: LockType) -> None:
        self._delegate = delegate
        self._calls_guard = Lock()
        self._calls = 0
        self.first_acquire_waiting = Event()
        self.allow_first_acquire = Event()
        self.unexpected_second_acquire = Event()

    def acquire(self) -> bool:
        with self._calls_guard:
            self._calls += 1
            call_number = self._calls
        if call_number != 1:
            self.unexpected_second_acquire.set()
            raise AssertionError("a second enter reached the shared session lock")
        self.first_acquire_waiting.set()
        if not self.allow_first_acquire.wait(timeout=5):
            raise AssertionError("timed out releasing the first transaction enter")
        return self._delegate.acquire()

    def release(self) -> None:
        self._delegate.release()


@pytest.fixture
def registry() -> FacetRegistry:
    return FacetRegistry(
        specs=(
            FacetSpec(
                id="color",
                kind=FacetKind.CATEGORICAL,
                operators=CATEGORICAL_OPERATORS,
                normalizer=canonical_text,
            ),
            FacetSpec(
                id="budget",
                kind=FacetKind.NUMERIC,
                operators=NUMERIC_OPERATORS,
                normalizer=canonical_number,
            ),
        )
    )


@pytest.fixture
def store(registry: FacetRegistry) -> InMemorySessionStore:
    return InMemorySessionStore(registry)


def profile(*, rating: float | None = 4.5, summary: str = "Durable products") -> ProfilePrior:
    return ProfilePrior(
        purchase_frequency="monthly",
        average_prior_rating=rating,
        rating_style="balanced",
        preference_tags=("durable",),
        summary=summary,
    )


def color_preference(*, turn: int = 1) -> Preference:
    return Preference(
        id=f"p_{turn}_0_0",
        facet="color",
        operator=Operator.EQ,
        value="blue",
        semantic_text=None,
        semantic_polarity=None,
        commitment=Commitment.HARD,
        source=PreferenceSource.USER_EXPLICIT,
        source_turn=turn,
        evidence_text="I want blue.",
        interpretation_confidence=1.0,
    )


def belief(*, intent_version: int, probe_id: str = "probe-1") -> SearchBelief:
    return SearchBelief(
        based_on_intent_version=intent_version,
        certainty=1.0,
        certainty_method="bods_v1",
        certainty_evidence=CertaintyEvidence(
            probe_id=probe_id,
            probe_size=1,
            raw_concentration=1.0,
            quality_status=ProbeQuality.VALID,
            quality_reasons=(),
        ),
        candidate_modes=(
            CandidateMode(
                id="primary",
                label="primary mode",
                mass=1.0,
                representative_ids=("sku-1",),
            ),
        ),
        facet_stats=(),
    )


def unchanged_next_context(
    previous: SessionContext,
    *,
    turn: int,
    assistant_message: str = "Response",
) -> SessionContext:
    return next_context(
        previous,
        turn=turn,
        assistant_message=assistant_message,
    )


def next_context(
    previous: SessionContext,
    *,
    turn: int,
    update: StateUpdateBatch | None = None,
    next_belief: SearchBelief | None | object = _PRESERVE_BELIEF,
    assistant_message: str = "Response",
) -> SessionContext:
    next_intent = (
        previous.state.intent
        if update is None
        else reduce_intent(previous.state.intent, update, _registry_for_context_tests())
    )
    if next_belief is _PRESERVE_BELIEF:
        resolved_belief = previous.state.search_belief
    else:
        assert next_belief is None or isinstance(next_belief, SearchBelief)
        resolved_belief = next_belief
    probe_id = (
        resolved_belief.certainty_evidence.probe_id
        if resolved_belief is not None and resolved_belief != previous.state.search_belief
        else None
    )
    record = TurnRecord(
        turn=turn,
        user_message=f"User turn {turn}",
        intent_version_before=previous.state.intent.version,
        accepted_update=update,
        intent_version_after=next_intent.version,
        assistant_message=assistant_message,
        question=None,
        question_key=None,
        ask_attribute=None,
        shown_product_ids=(),
        feedback=(),
        search_belief_probe_id=probe_id,
    )
    return SessionContext(
        session_id=previous.session_id,
        profile=previous.profile,
        state=SessionState(
            intent=next_intent,
            interaction=InteractionContext(turns=previous.state.interaction.turns + (record,)),
            search_belief=resolved_belief,
        ),
    )


def _registry_for_context_tests() -> FacetRegistry:
    return FacetRegistry(
        specs=(
            FacetSpec(
                id="color",
                kind=FacetKind.CATEGORICAL,
                operators=CATEGORICAL_OPERATORS,
                normalizer=canonical_text,
            ),
            FacetSpec(
                id="budget",
                kind=FacetKind.NUMERIC,
                operators=NUMERIC_OPERATORS,
                normalizer=canonical_number,
            ),
        )
    )


def assert_error(
    expected: ErrorCode,
    action: object,
) -> SessionContextError:
    assert callable(action)
    with pytest.raises(SessionContextError) as caught:
        action()
    assert caught.value.code is expected
    return caught.value


def commit_unchanged_turn(
    store: InMemorySessionStore,
    *,
    session_id: str,
    turn: int,
    assistant_message: str = "Response",
) -> SessionContext:
    with store.turn(session_id=session_id, turn=turn) as transaction:
        candidate = unchanged_next_context(
            transaction.context,
            turn=turn,
            assistant_message=assistant_message,
        )
        return transaction.commit(candidate)


def test_reset_creates_and_returns_the_exact_initial_snapshot(
    store: InMemorySessionStore,
) -> None:
    initial_profile = profile(rating=None)

    context = store.reset(session_id="session-1", profile=initial_profile)

    assert context == SessionContext(
        session_id="session-1",
        profile=initial_profile,
        state=SessionState(
            intent=IntentState(
                goal=None,
                preferences=(),
                dont_care_facets=frozenset(),
                version=0,
            ),
            interaction=InteractionContext(turns=()),
            search_belief=None,
        ),
    )
    assert store.get("session-1") is context


def test_reset_rejects_duplicate_session_without_replacing_original(
    store: InMemorySessionStore,
) -> None:
    original = store.reset(session_id="session-1", profile=profile(summary="original"))

    error = assert_error(
        ErrorCode.SESSION_ALREADY_EXISTS,
        lambda: store.reset(session_id="session-1", profile=profile(summary="replacement")),
    )

    assert error.path == ("session_id",)
    assert store.get("session-1") is original


@pytest.mark.parametrize("session_id", ["", " ", " session", "session ", None, 1, True])
def test_reset_rejects_invalid_session_ids(
    store: InMemorySessionStore,
    session_id: object,
) -> None:
    assert_error(
        ErrorCode.INVALID_SESSION_ID,
        lambda: store.reset(session_id=session_id),  # type: ignore[arg-type]
    )


def test_get_and_turn_apply_the_same_session_id_validation(
    store: InMemorySessionStore,
) -> None:
    assert_error(ErrorCode.INVALID_SESSION_ID, lambda: store.get(" session"))
    assert_error(
        ErrorCode.INVALID_SESSION_ID,
        lambda: store.turn(session_id="session ", turn=1),
    )


def test_reset_rejects_invalid_profile_before_creating_session(
    store: InMemorySessionStore,
) -> None:
    invalid = profile()
    invalid = replace(invalid, average_prior_rating=True)

    assert_error(
        ErrorCode.INVALID_PROFILE,
        lambda: store.reset(session_id="session-1", profile=invalid),
    )
    assert_error(ErrorCode.SESSION_NOT_FOUND, lambda: store.get("session-1"))


def test_get_and_turn_reject_missing_sessions(store: InMemorySessionStore) -> None:
    assert_error(ErrorCode.SESSION_NOT_FOUND, lambda: store.get("missing"))
    assert_error(
        ErrorCode.SESSION_NOT_FOUND,
        lambda: store.turn(session_id="missing", turn=1),
    )


@pytest.mark.parametrize("turn", [0, -1, True, 1.5])
def test_turn_rejects_invalid_turn_values(
    store: InMemorySessionStore,
    turn: object,
) -> None:
    store.reset(session_id="session-1")
    assert_error(
        ErrorCode.TURN_OUT_OF_ORDER,
        lambda: store.turn(session_id="session-1", turn=turn),  # type: ignore[arg-type]
    )


def test_transaction_exposes_the_exact_captured_snapshot(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        assert transaction.context is initial

    assert store.get("session-1") is initial


def test_leaving_transaction_without_commit_changes_nothing(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        unchanged_next_context(transaction.context, turn=1)

    assert store.get("session-1") is initial


def test_exception_before_commit_changes_nothing_and_releases_lock(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        with store.turn(session_id="session-1", turn=1):
            raise RuntimeError("pipeline failed")

    assert store.get("session-1") is initial
    committed = commit_unchanged_turn(store, session_id="session-1", turn=1)
    assert store.get("session-1") is committed


def test_successful_commit_swaps_intent_belief_and_interaction_together(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")
    added = color_preference(turn=1)
    update = StateUpdateBatch(
        turn=1,
        base_intent_version=0,
        operations=(AddPreference(preference=added),),
    )
    next_belief = belief(intent_version=1)

    with store.turn(session_id="session-1", turn=1) as transaction:
        candidate = next_context(
            transaction.context,
            turn=1,
            update=update,
            next_belief=next_belief,
        )
        committed = transaction.commit(candidate)

    assert committed is candidate
    assert store.get("session-1") is candidate
    assert candidate is not initial
    assert candidate.state.intent.preferences == (added,)
    assert candidate.state.intent.version == 1
    assert candidate.state.search_belief is next_belief
    assert candidate.state.interaction.turns[-1].accepted_update is update
    assert candidate.state.interaction.turns[-1].search_belief_probe_id == "probe-1"


def test_failed_aggregate_commit_keeps_old_snapshot_and_consumes_attempt(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        valid = unchanged_next_context(transaction.context, turn=1)
        record = valid.state.interaction.turns[-1]
        invalid_record = replace(record, question="Which color?")
        invalid = replace(
            valid,
            state=replace(
                valid.state,
                interaction=InteractionContext(turns=(invalid_record,)),
            ),
        )
        assert_error(ErrorCode.INVALID_QUESTION_FIELDS, lambda: transaction.commit(invalid))
        assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.commit(valid))

    assert store.get("session-1") is initial
    commit_unchanged_turn(store, session_id="session-1", turn=1)


def test_failed_transition_commit_keeps_old_snapshot(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        valid = unchanged_next_context(transaction.context, turn=1)
        forged = replace(valid, session_id="another-session")
        assert_error(
            ErrorCode.INVALID_SESSION_TRANSITION,
            lambda: transaction.commit(forged),
        )

    assert store.get("session-1") is initial


def test_commit_remains_visible_if_later_code_in_with_block_raises(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")
    committed: SessionContext | None = None

    with pytest.raises(RuntimeError, match="after commit"):
        with store.turn(session_id="session-1", turn=1) as transaction:
            candidate = unchanged_next_context(transaction.context, turn=1)
            committed = transaction.commit(candidate)
            raise RuntimeError("after commit")

    assert committed is not None
    assert store.get("session-1") is committed


def test_context_and_commit_before_enter_are_commit_conflicts(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")
    transaction = store.turn(session_id="session-1", turn=1)
    candidate = unchanged_next_context(initial, turn=1)

    assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.context)
    assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.commit(candidate))
    assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, transaction.__enter__)
    assert store.get("session-1") is initial


def test_double_commit_is_rejected_without_appending_twice(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        candidate = unchanged_next_context(transaction.context, turn=1)
        assert transaction.commit(candidate) is candidate
        assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.commit(candidate))

    assert store.get("session-1") is candidate
    assert len(candidate.state.interaction.turns) == 1


def test_commit_and_context_after_exit_are_commit_conflicts(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")
    transaction = store.turn(session_id="session-1", turn=1)

    with transaction:
        candidate = unchanged_next_context(transaction.context, turn=1)

    assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.commit(candidate))
    assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.context)
    assert store.get("session-1") is initial


def test_transaction_cannot_be_reentered(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")
    transaction = store.turn(session_id="session-1", turn=1)

    with transaction:
        assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, transaction.__enter__)
        candidate = unchanged_next_context(transaction.context, turn=1)
        transaction.commit(candidate)

    assert store.get("session-1") is candidate
    assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, transaction.__enter__)


def test_same_transaction_concurrent_enter_is_claimed_before_waiting_for_session_lock(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")
    transaction = store.turn(session_id="session-1", turn=1)
    entry = transaction._entry
    original_lock = entry.lock
    gate = _FirstAcquireGate(original_lock)
    entry.lock = gate  # type: ignore[assignment]

    def enter_once() -> str | BaseException:
        try:
            with transaction:
                return "entered"
        except BaseException as error:
            return error

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        first = executor.submit(enter_once)
        assert gate.first_acquire_waiting.wait(timeout=5)
        second = executor.submit(enter_once)
        second_result = second.result(timeout=5)
        gate.allow_first_acquire.set()
        first_result = first.result(timeout=5)
    finally:
        gate.allow_first_acquire.set()
        executor.shutdown(wait=True, cancel_futures=True)
        entry.lock = original_lock

    assert first_result == "entered"
    assert isinstance(second_result, SessionContextError)
    assert second_result.code is ErrorCode.SESSION_COMMIT_CONFLICT
    assert not gate.unexpected_second_acquire.is_set()
    assert not original_lock.locked()
    assert entry.active_token is None
    assert store.get("session-1") is initial


def test_foreign_thread_cannot_use_or_close_an_active_transaction(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        candidate = unchanged_next_context(transaction.context, turn=1)

        def capture(call: Callable[[], object]) -> object:
            try:
                return call()
            except BaseException as error:
                return error

        with ThreadPoolExecutor(max_workers=1) as executor:
            foreign_context = executor.submit(capture, lambda: transaction.context).result(
                timeout=5
            )
            foreign_commit = executor.submit(
                capture,
                lambda: transaction.commit(candidate),
            ).result(timeout=5)
            foreign_exit = executor.submit(
                capture,
                lambda: transaction.__exit__(None, None, None),
            ).result(timeout=5)

        for result in (foreign_context, foreign_commit, foreign_exit):
            assert isinstance(result, SessionContextError)
            assert result.code is ErrorCode.SESSION_COMMIT_CONFLICT

        assert transaction.context is not None
        assert transaction.commit(candidate) is candidate

    assert store.get("session-1") is candidate


def test_nested_same_session_transaction_is_rejected_without_deadlock(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as outer:
        nested = store.turn(session_id="session-1", turn=1)
        assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, nested.__enter__)
        candidate = unchanged_next_context(outer.context, turn=1)
        outer.commit(candidate)

    assert store.get("session-1") is candidate


def test_nested_same_session_get_is_rejected_without_deadlock(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: store.get("session-1"))
        candidate = unchanged_next_context(transaction.context, turn=1)
        transaction.commit(candidate)

    assert store.get("session-1") is candidate


def test_forged_token_fails_closed_and_does_not_poison_session(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")

    with store.turn(session_id="session-1", turn=1) as transaction:
        candidate = unchanged_next_context(transaction.context, turn=1)
        transaction._token = object()
        assert_error(ErrorCode.SESSION_COMMIT_CONFLICT, lambda: transaction.commit(candidate))

    assert store.get("session-1") is initial
    committed = commit_unchanged_turn(store, session_id="session-1", turn=1)
    assert store.get("session-1") is committed


def test_skipped_turn_is_rejected_when_transaction_enters(
    store: InMemorySessionStore,
) -> None:
    initial = store.reset(session_id="session-1")
    transaction = store.turn(session_id="session-1", turn=2)

    def enter() -> None:
        with transaction:
            raise AssertionError("out-of-order transaction entered")

    error = assert_error(ErrorCode.TURN_OUT_OF_ORDER, enter)
    assert error.details == (("actual", 2), ("expected", 1))
    assert store.get("session-1") is initial


@pytest.mark.parametrize("turn", [1, 3], ids=("duplicate", "skipped"))
def test_duplicate_and_skipped_turns_do_not_advance_session(
    store: InMemorySessionStore,
    turn: int,
) -> None:
    store.reset(session_id="session-1")
    first = commit_unchanged_turn(store, session_id="session-1", turn=1)

    def enter() -> None:
        with store.turn(session_id="session-1", turn=turn):
            raise AssertionError("out-of-order transaction entered")

    assert_error(ErrorCode.TURN_OUT_OF_ORDER, enter)
    assert store.get("session-1") is first
    second = commit_unchanged_turn(store, session_id="session-1", turn=2)
    assert len(second.state.interaction.turns) == 2


def test_turn_can_be_retried_after_earlier_out_of_order_rejection(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")

    def enter_turn_two() -> None:
        with store.turn(session_id="session-1", turn=2):
            raise AssertionError("turn two entered before turn one")

    assert_error(ErrorCode.TURN_OUT_OF_ORDER, enter_turn_two)
    commit_unchanged_turn(store, session_id="session-1", turn=1)
    second = commit_unchanged_turn(store, session_id="session-1", turn=2)

    assert tuple(record.turn for record in second.state.interaction.turns) == (1, 2)


def test_two_sessions_can_hold_turn_transactions_concurrently(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-a")
    store.reset(session_id="session-b")
    a_entered = Event()
    b_entered = Event()
    release_a = Event()

    def process_a() -> SessionContext:
        with store.turn(session_id="session-a", turn=1) as transaction:
            a_entered.set()
            if not release_a.wait(timeout=5):
                raise TimeoutError("session A was not released")
            candidate = unchanged_next_context(transaction.context, turn=1)
            return transaction.commit(candidate)

    def process_b() -> SessionContext:
        if not a_entered.wait(timeout=5):
            raise TimeoutError("session A never entered")
        with store.turn(session_id="session-b", turn=1) as transaction:
            b_entered.set()
            candidate = unchanged_next_context(transaction.context, turn=1)
            return transaction.commit(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(process_a)
        future_b = executor.submit(process_b)
        try:
            assert a_entered.wait(timeout=5)
            assert b_entered.wait(timeout=2)
        finally:
            release_a.set()
        committed_a = future_a.result(timeout=5)
        committed_b = future_b.result(timeout=5)

    assert store.get("session-a") is committed_a
    assert store.get("session-b") is committed_b


def test_reader_observes_only_complete_snapshot_after_active_commit(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")
    writer_entered = Event()
    reader_started = Event()
    allow_commit = Event()

    def writer() -> SessionContext:
        with store.turn(session_id="session-1", turn=1) as transaction:
            candidate = unchanged_next_context(transaction.context, turn=1)
            writer_entered.set()
            if not allow_commit.wait(timeout=5):
                raise TimeoutError("writer was not released")
            return transaction.commit(candidate)

    def reader() -> SessionContext:
        if not writer_entered.wait(timeout=5):
            raise TimeoutError("writer never entered")
        reader_started.set()
        return store.get("session-1")

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer_future = executor.submit(writer)
        reader_future = executor.submit(reader)
        assert reader_started.wait(timeout=5)
        allow_commit.set()
        committed = writer_future.result(timeout=5)
        observed = reader_future.result(timeout=5)

    assert observed is committed
    assert len(observed.state.interaction.turns) == 1


def test_two_concurrent_requests_for_same_turn_commit_exactly_once(
    store: InMemorySessionStore,
) -> None:
    store.reset(session_id="session-1")
    start = Barrier(3)

    def worker(label: str) -> tuple[str, ErrorCode | None, SessionContext | None]:
        start.wait(timeout=5)
        try:
            with store.turn(session_id="session-1", turn=1) as transaction:
                candidate = unchanged_next_context(
                    transaction.context,
                    turn=1,
                    assistant_message=label,
                )
                committed = transaction.commit(candidate)
            return "committed", None, committed
        except SessionContextError as error:
            return "error", error.code, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(worker, "worker-a"), executor.submit(worker, "worker-b"))
        start.wait(timeout=5)
        results = tuple(future.result(timeout=5) for future in futures)

    assert sorted((status, code) for status, code, _ in results) == [
        ("committed", None),
        ("error", ErrorCode.TURN_OUT_OF_ORDER),
    ]
    committed = next(context for status, _, context in results if status == "committed")
    assert committed is not None
    assert store.get("session-1") is committed
    assert len(committed.state.interaction.turns) == 1


def test_concurrent_reset_creates_one_session_and_one_lock_entry(
    store: InMemorySessionStore,
) -> None:
    start = Barrier(3)

    def worker() -> tuple[str, ErrorCode | None, SessionContext | None]:
        start.wait(timeout=5)
        try:
            return "created", None, store.reset(session_id="session-1")
        except SessionContextError as error:
            return "error", error.code, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(worker), executor.submit(worker))
        start.wait(timeout=5)
        results = tuple(future.result(timeout=5) for future in futures)

    assert sorted((status, code) for status, code, _ in results) == [
        ("created", None),
        ("error", ErrorCode.SESSION_ALREADY_EXISTS),
    ]
    created = next(context for status, _, context in results if status == "created")
    assert created is not None
    assert store.get("session-1") is created
    committed = commit_unchanged_turn(store, session_id="session-1", turn=1)
    assert len(committed.state.interaction.turns) == 1


def test_store_constructor_rejects_registry_substitutes() -> None:
    with pytest.raises(TypeError):
        InMemorySessionStore(object())  # type: ignore[arg-type]


def test_turn_returns_single_use_transaction_object(store: InMemorySessionStore) -> None:
    store.reset(session_id="session-1")

    transaction = store.turn(session_id="session-1", turn=1)

    assert type(transaction) is SessionTransaction
