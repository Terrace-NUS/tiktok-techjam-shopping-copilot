"""Synchronous copy-on-write storage for complete session snapshots."""

from __future__ import annotations

import threading
from _thread import LockType
from dataclasses import dataclass
from enum import Enum, auto
from types import TracebackType
from typing import NoReturn

from .aggregate_validation import validate_session_context, validate_session_transition
from .aggregates import InteractionContext, SessionContext, SessionState
from .errors import ErrorCode, SessionContextError
from .models import IntentState, ProfilePrior
from .registry import FacetRegistry
from .validation import validate_profile_prior


@dataclass(slots=True)
class _SessionEntry:
    context: SessionContext
    lock: LockType
    active_token: object | None = None


class _TransactionState(Enum):
    CREATED = auto()
    ENTERING = auto()
    ACTIVE = auto()
    COMMITTED = auto()
    FAILED = auto()
    CLOSED = auto()


class _ThreadState(threading.local):
    active_sessions: set[str]

    def __init__(self) -> None:
        self.active_sessions = set()


class InMemorySessionStore:
    """Serialize turns per session while allowing independent sessions to progress."""

    def __init__(self, registry: FacetRegistry) -> None:
        if type(registry) is not FacetRegistry:
            raise TypeError("registry must be an exact FacetRegistry")
        self._registry = registry
        self._entries: dict[str, _SessionEntry] = {}
        self._entries_guard = threading.Lock()
        self._thread_state = _ThreadState()

    def reset(
        self,
        *,
        session_id: str,
        profile: ProfilePrior | None = None,
    ) -> SessionContext:
        """Create an initial session without silently replacing an existing one."""

        _validate_session_id(session_id)
        if profile is not None:
            validate_profile_prior(profile)
        context = SessionContext(
            session_id=session_id,
            profile=profile,
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
        validate_session_context(context, self._registry)

        with self._entries_guard:
            if session_id in self._entries:
                raise SessionContextError(
                    code=ErrorCode.SESSION_ALREADY_EXISTS,
                    path=("session_id",),
                )
            self._entries[session_id] = _SessionEntry(
                context=context,
                lock=threading.Lock(),
            )
        return context

    def get(self, session_id: str) -> SessionContext:
        """Return one immutable snapshot after any active same-session turn."""

        _validate_session_id(session_id)
        entry = self._entry(session_id)
        if session_id in self._thread_state.active_sessions:
            _raise_commit_conflict()
        with entry.lock:
            return entry.context

    def turn(self, *, session_id: str, turn: int) -> SessionTransaction:
        """Create a transaction whose turn order is checked after lock acquisition."""

        _validate_session_id(session_id)
        if type(turn) is not int or turn < 1:
            raise SessionContextError(code=ErrorCode.TURN_OUT_OF_ORDER, path=("turn",))
        entry = self._entry(session_id)
        return SessionTransaction(
            store=self,
            entry=entry,
            session_id=session_id,
            turn=turn,
        )

    def _entry(self, session_id: str) -> _SessionEntry:
        with self._entries_guard:
            entry = self._entries.get(session_id)
        if entry is None:
            raise SessionContextError(
                code=ErrorCode.SESSION_NOT_FOUND,
                path=("session_id",),
            )
        return entry


class SessionTransaction:
    """One single-use, opaque-token transaction over a captured session snapshot."""

    def __init__(
        self,
        *,
        store: InMemorySessionStore,
        entry: _SessionEntry,
        session_id: str,
        turn: int,
    ) -> None:
        self._store = store
        self._entry = entry
        self._session_id = session_id
        self._turn = turn
        self._state = _TransactionState.CREATED
        self._lifecycle_guard = threading.Lock()
        self._owner_thread_id: int | None = None
        self._captured: SessionContext | None = None
        self._token: object | None = None
        self._entry_token: object | None = None
        self._commit_attempted = False
        self._lock_acquired = False

    @property
    def context(self) -> SessionContext:
        """Return the immutable snapshot captured when the transaction entered."""

        with self._lifecycle_guard:
            if self._owner_thread_id != threading.get_ident() or self._state not in (
                _TransactionState.ACTIVE,
                _TransactionState.COMMITTED,
            ):
                _raise_commit_conflict()
            assert self._captured is not None
            return self._captured

    def __enter__(self) -> SessionTransaction:
        owner_thread_id = threading.get_ident()
        with self._lifecycle_guard:
            if self._state is not _TransactionState.CREATED:
                _raise_commit_conflict()
            self._state = _TransactionState.ENTERING
            self._owner_thread_id = owner_thread_id

        active_sessions = self._store._thread_state.active_sessions
        if self._session_id in active_sessions:
            with self._lifecycle_guard:
                self._state = _TransactionState.CLOSED
            _raise_commit_conflict()

        active_sessions.add(self._session_id)
        try:
            self._entry.lock.acquire()
            self._lock_acquired = True
            expected_turn = len(self._entry.context.state.interaction.turns) + 1
            if self._turn != expected_turn:
                with self._lifecycle_guard:
                    self._state = _TransactionState.FAILED
                raise SessionContextError(
                    code=ErrorCode.TURN_OUT_OF_ORDER,
                    path=("turn",),
                    details=(
                        ("actual", self._turn),
                        ("expected", expected_turn),
                    ),
                )
            if self._entry.active_token is not None:
                with self._lifecycle_guard:
                    self._state = _TransactionState.FAILED
                _raise_commit_conflict()

            token = object()
            self._token = token
            self._entry_token = token
            self._entry.active_token = token
            self._captured = self._entry.context
            with self._lifecycle_guard:
                if (
                    self._state is not _TransactionState.ENTERING
                    or self._owner_thread_id != owner_thread_id
                ):
                    _raise_commit_conflict()
                self._state = _TransactionState.ACTIVE
            return self
        except BaseException:
            with self._lifecycle_guard:
                if self._state is not _TransactionState.ACTIVE:
                    self._close_resources()
                    self._state = _TransactionState.CLOSED
            raise

    def commit(self, next_context: SessionContext) -> SessionContext:
        """Validate and atomically swap one complete replacement snapshot."""

        owner_thread_id = threading.get_ident()
        with self._lifecycle_guard:
            if self._state is _TransactionState.CREATED:
                self._state = _TransactionState.CLOSED
            if (
                self._owner_thread_id != owner_thread_id
                or self._state is not _TransactionState.ACTIVE
                or self._commit_attempted
            ):
                _raise_commit_conflict()
            self._commit_attempted = True

        try:
            captured = self._captured
            if (
                self._token is None
                or self._entry_token is None
                or self._token is not self._entry_token
                or self._entry.active_token is not self._entry_token
                or captured is None
                or self._entry.context is not captured
            ):
                _raise_commit_conflict()

            validate_session_context(next_context, self._store._registry)
            validate_session_transition(
                captured,
                next_context,
                self._turn,
                self._store._registry,
            )
            if (
                self._entry.active_token is not self._entry_token
                or self._entry.context is not captured
            ):
                _raise_commit_conflict()

            with self._lifecycle_guard:
                if (
                    self._owner_thread_id != owner_thread_id
                    or self._state is not _TransactionState.ACTIVE
                ):
                    _raise_commit_conflict()
                self._entry.context = next_context
                self._state = _TransactionState.COMMITTED
            return next_context
        except BaseException:
            with self._lifecycle_guard:
                if self._state is not _TransactionState.COMMITTED:
                    self._state = _TransactionState.FAILED
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        with self._lifecycle_guard:
            if self._state is _TransactionState.CREATED:
                self._state = _TransactionState.CLOSED
                _raise_commit_conflict()
            if self._owner_thread_id != threading.get_ident() or self._state not in (
                _TransactionState.ACTIVE,
                _TransactionState.COMMITTED,
                _TransactionState.FAILED,
            ):
                _raise_commit_conflict()
            self._close_resources()
            self._state = _TransactionState.CLOSED

    def _close_resources(self) -> None:
        if self._entry_token is not None and self._entry.active_token is self._entry_token:
            self._entry.active_token = None
        if self._lock_acquired:
            self._entry.lock.release()
            self._lock_acquired = False
        self._store._thread_state.active_sessions.discard(self._session_id)


def _validate_session_id(session_id: object) -> None:
    if type(session_id) is not str or not session_id.strip() or session_id != session_id.strip():
        raise SessionContextError(
            code=ErrorCode.INVALID_SESSION_ID,
            path=("session_id",),
        )


def _raise_commit_conflict() -> NoReturn:
    raise SessionContextError(code=ErrorCode.SESSION_COMMIT_CONFLICT)
