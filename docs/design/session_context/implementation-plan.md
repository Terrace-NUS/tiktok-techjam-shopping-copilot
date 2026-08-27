# Session Context v1 Implementation Plan

- Status: **ready for implementation**
- Contract: [contract-v1.md](contract-v1.md)
- Target runtime: **Python 3.10+**

This document turns the frozen semantic contract into an implementation and
test sequence. It does not choose the query-understanding model, retriever,
ranker, certainty formula, or question policy.

## 1. Delivery boundary

The first delivery is a small domain package with:

- immutable session-context values;
- a trusted facet registry and validation boundary;
- one deterministic intent reducer;
- append-only interaction views;
- validated serialization;
- an in-memory, per-session transaction store;
- unit and replay tests.

It will not yet replace the BM25 baseline. The official adapter is connected
only after the package passes its own contract tests.

## 2. Target repository layout

Create source and test directories only when their first real file is added:

    pyproject.toml
    src/
      shopping_copilot/
        __init__.py
        session_context/
          __init__.py
          models.py
          operations.py
          aggregates.py
          registry.py
          errors.py
          validation.py
          reducer.py
          aggregate_validation.py
          views.py
          serialization.py
          store.py
    tests/
      unit/
        session_context/
          test_models.py
          test_registry.py
          test_validation.py
          test_reducer.py
          test_aggregates.py
          test_aggregate_validation.py
          test_views.py
          test_serialization.py
          test_store.py
      integration/
        test_official_adapter.py

Responsibilities are deliberately narrow:

| Module | Owns | Must not own |
| --- | --- | --- |
| models.py | Frozen enums, leaf values, intent, and belief values | Parsing, retrieval, policy |
| operations.py | Closed operation union and update batch | Mutation logic |
| aggregates.py | Turn records, interaction history, and session aggregates | Validation or transition logic |
| registry.py | Immutable facet specifications and canonical scalar normalization | Catalog values or counts |
| errors.py | Stable domain error codes and structured exception | User-facing prose |
| validation.py | Local shape, cross-field, and canonical validation | Replay or state transitions |
| reducer.py | Pure, ordered, atomic intent reduction | Session storage or I/O |
| aggregate_validation.py | History, replay, and previous-to-next transition validation | Business pipeline execution |
| views.py | Derived interaction-history queries | Cached mutable mirrors |
| serialization.py | Versioned JSON-compatible snapshot codec | Pickle or arbitrary Python objects |
| store.py | Per-session locking, turn transaction, copy-on-write commit | Business pipeline decisions |

The package root exports only the values and functions needed by application
code. Helper validators and concrete lock bookkeeping remain private.

## 3. Dependency direction

Imports follow this topological order and do not form cycles:

    errors
    models
      ├── operations
      └── registry
    models + operations
      └── aggregates
    models + operations + registry + errors
      └── validation
    validation + operations
      └── reducer
    aggregates + validation + reducer
      └── aggregate_validation
    aggregates
      └── views
    all domain types + aggregate_validation
      ├── serialization
      └── store

Hard rules:

- session_context MUST NOT import starter, evaluator, catalog loaders,
  retrievers, rankers, model clients, or question policy.
- models.py MUST NOT import any sibling module.
- aggregates.py owns the only domain reference from TurnRecord to
  StateUpdateBatch, avoiding a models/operations cycle.
- reducer.py may depend on models, operations, registry, validation, and
  errors only.
- aggregate_validation.py may call the pure reducer to prove replay and
  transition correctness.
- store.py MUST call aggregate transition validation. It does not directly run
  the reducer or any parser, Probe, retriever, ranker, or adapter.
- the official adapter may depend on the application package; the dependency
  never points back toward starter.

These boundaries keep the reducer usable in isolated tests and make the
official simulator one consumer rather than the architecture's center.

## 4. Core APIs

### 4.1 Reducer

The stable functional entry point is:

    def reduce_intent(
        current: IntentState,
        batch: StateUpdateBatch,
        registry: FacetRegistry,
    ) -> IntentState:
        ...

It returns a new frozen state or raises SessionContextError. It never mutates
current, never partially returns a failed batch, and never reads external
state.

The implementation applies operations to a temporary canonical working state,
validates every intermediate result, and constructs IntentState only after the
batch succeeds. Version calculation occurs once at the end.

### 4.2 Validation

Validation is explicit at every trust boundary:

    validate_preference(preference, registry)
    validate_intent_state(intent, registry)
    validate_search_belief(belief, registry)

Aggregate validation is in the higher-level, reducer-aware module:

    validate_turn_record(record, prior_interaction)
    validate_session_context(context, registry)
    validate_session_transition(previous, next, expected_turn, registry)

Constructing a dataclass is not proof that it is valid. Public reducers,
deserializers, and store commits all invoke the relevant validator.

Domain errors contain:

    code: ErrorCode
    path: tuple[str | int, ...]
    operation_index: int | None
    details: tuple[tuple[str, ScalarValue], ...]

Details are canonically sorted and MUST NOT contain raw model output, secrets,
or large catalog payloads. The serializer converts them to a JSON object only
at the wire boundary.

### 4.3 Facet registry

FacetRegistry is built once from reviewed local configuration and injected:

    FacetSpec(
        id="budget",
        kind=FacetKind.NUMERIC,
        operators=frozenset({...}),
        normalizer=...,
    )

The P0 registry may be defined in Python to keep validation deterministic and
dependency-free. Moving it to a config file later requires schema validation
and does not change reducer semantics.

Value canonicalization happens before committed Preference IDs are allocated.
The reducer verifies canonical form; it does not guess aliases or repair
untrusted values.

### 4.4 Interaction views

views.py computes, without side effects:

- last non-empty recommendation batch;
- all shown product IDs in first-seen order;
- question keys since the last goal switch;
- most recent question and assistant message.

Each view is tested against histories containing empty recommendations,
rejected batches, logical no-ops, goal switches, and fallback responses.

### 4.5 Session transactions

The P0 store is synchronous because the official Agent API is synchronous. It
uses one non-reentrant lock per session and copy-on-write snapshots. A short
global guard protects the session-entry and lock map; it is never held while a
turn is processed. A thread-local active-session guard rejects a nested
same-session transaction immediately rather than deadlocking.

Conceptual usage:

    with store.turn(session_id, turn) as transaction:
        previous = transaction.context

        # Application services compute intent, belief, normalized reply,
        # and the matching TurnRecord while the session is serialized.

        transaction.commit(next_context)

The transaction:

1. acquires the session lock;
2. verifies that the session exists and turn is exactly previous turn + 1;
3. exposes one immutable snapshot;
4. validates the complete replacement aggregate;
5. runs validate_session_transition against the captured snapshot;
6. swaps the snapshot exactly once using the active transaction token;
7. releases the lock on commit or exception.

Leaving the context manager without commit changes nothing. A failed commit
keeps the prior snapshot. The application returns an external response only
after the matching context has committed.

Transition validation rejects an otherwise well-formed next snapshot if it
changes session_id or profile, rewrites the history prefix, appends other than
one turn, bypasses the reducer result, violates belief invalidation, or makes a
TurnRecord disagree with the previous and next intent versions. Double commit,
commit after exit, nested entry, or a wrong token uses
SESSION_COMMIT_CONFLICT.

reset creates the initial aggregate and rejects duplicate session IDs. A
separate explicit replace/reset policy can be added later; silent overwrite is
not allowed in v1.

## 5. End-to-end turn flow

Application orchestration sits outside session_context:

    official Agent.respond
            ↓
    acquire per-session turn transaction
            ↓
    parse into PreferenceDraft / draft operations
            ↓
    trusted grounding + normalization + committed ID assignment
            ↓
    reduce_intent, or preserve intent on a domain error
            ↓
    Probe and derive SearchBelief when available
            ↓
    route, retrieve, rank, and decide whether to ask
            ↓
    normalize the external reply to the official API
            ↓
    construct TurnRecord from the actual normalized reply
            ↓
    validate and atomically commit the new SessionContext
            ↓
    return that same reply

Important failure behavior:

- Parser or grounding failure produces no committed batch.
- Reducer failure preserves intent; both of these pre-acceptance failures append
  one unchanged-version TurnRecord with accepted_update=None.
- Once the reducer accepts a batch, later Probe, retrieval, ranking, or
  normalization failure MUST NOT discard it. The reduced intent and
  accepted_update=batch are committed with a safe fallback reply.
- An accepted logical no-op records the batch while retaining equal
  before/after versions.
- A real intent change invalidates the prior SearchBelief before Probe.
- After a real change, a newly validated belief may survive a later downstream
  failure; the belief from the previous intent never does.
- A failed final commit returns no unrecorded successful response.
- Trace diagnostics are written outside active session state and are not
  required for the P0 transaction to succeed.

## 6. Serialization

Snapshots use an explicit JSON-compatible envelope:

    {
      "schema": "shopping-copilot/session-context/v1",
      "payload": { ... }
    }

The codec:

- emits enum values as strings;
- preserves semantic sequence order and emits set-like values in the canonical
  orders defined by the contract;
- sorts JSON object keys lexicographically;
- emits compact UTF-8 JSON with unescaped Unicode and no non-finite numbers;
- rejects unknown schema versions and fields with their stable error codes;
- maps malformed JSON, enum values, and operation discriminators to the
  documented boundary error;
- reconstructs domain objects, reruns mechanical validation, and replays
  accepted updates;
- produces byte-identical output for repeated encoding of the same snapshot.

Pickle is prohibited for persisted or untrusted state. Pydantic may later be
used only for boundary schemas; the domain remains frozen dataclasses. The P0
core should have no mandatory runtime dependency beyond the standard library.

## 7. Test strategy

### 7.1 Model and registry tests

- every enum round-trips through its wire value;
- bool is rejected as numeric;
- NaN and infinities are rejected;
- empty or mutable collection payloads are rejected;
- structured and semantic representation pairing is enforced;
- operator/value/facet-kind compatibility is enforced;
- confidence, source, commitment, and source-turn rules are enforced;
- canonical IN and NOT_IN tuples are homogeneous, unique, and ordered;
- numeric exact/range language is normalized into the allowed bound operators;
- a committed unknown structured facet returns UNKNOWN_FACET, while a separate
  grounding-boundary test verifies fallback to a semantic-only draft;
- preference-ID sets and state tuples follow their canonical orders.

### 7.2 Reducer table tests

Every operation receives success, no-op, and failure cases:

- AddPreference removes don't-care and rejects ambiguous selectors;
- ReplaceFacet is complete, non-empty, and facet-consistent;
- RemovePreference rejects unknown IDs;
- ClearFacet returns the facet to unset;
- SetDontCare removes active facet preferences;
- SwitchGoal runs first, carries only pre-batch IDs, and preserves their
  identity and evidence;
- categorical positive/negative intersections remain non-empty;
- categorical hard/soft positive and negative combinations cover the complete
  cross-commitment conflict matrix;
- numeric bounds retain the strongest equivalent constraint;
- hard and soft ranges cannot become mutually unusable;
- strict endpoint cases distinguish `>= 50 + < 50` from
  `>= 50 + <= 50`;
- semantic-only preferences are removed by ID;
- stale versions, duplicate IDs, semantic duplicates, and ID conflicts use
  their stable error codes.

For every rejected batch, tests assert:

    returned error code is stable
    current state is byte-for-byte unchanged after canonical serialization
    current version is unchanged

For every accepted batch, tests assert that version changes exactly once if
and only if canonical intent changed.

### 7.3 Ordered atomicity tests

Dedicated cases cover:

- a valid first operation followed by an invalid second operation;
- a later operation that would otherwise repair an invalid intermediate
  state;
- duplicate and late SwitchGoal;
- logical reassertion using the active ID;
- replaying an accepted batch against its old base version;
- two batches racing from the same base version.

### 7.4 Interaction and belief tests

- turns begin at one and remain contiguous;
- batch turn, new source turns, and record turn agree;
- rejected updates still create one unchanged-version record;
- question, question_key, and ask_attribute share presence;
- feedback refers only to products shown on earlier turns;
- comparative feedback sets are non-empty and disjoint;
- shown products match normalized external order;
- derived views respect goal-switch boundaries;
- certainty follows the quality availability table;
- mode mass, conditional value mass, coverage, and entropy use the contract's
  denominators;
- the 1e-9 tolerance, positive emitted masses, uniqueness, canonical ordering,
  probe-size requirements, and quality-reason rules are enforced;
- stale beliefs are rejected or cleared after intent changes.

### 7.5 Failure-stage matrix

Separate tests inject failure at parsing, grounding, reduction, Probe,
retrieval/ranking, response normalization, and final commit. Each test asserts
the resulting intent, belief, accepted_update, TurnRecord, returned response or
raised error, and trace behavior. In particular, every post-reducer failure
retains an accepted intent update.

### 7.6 Codec, store, replay, and property tests

- different sessions do not share state or locks;
- for two concurrent requests with the same turn, exactly one commits and the
  other receives TURN_OUT_OF_ORDER;
- if turn 2 acquires the lock before turn 1 commits, turn 2 is rejected and the
  caller may retry; the store does not reorder requests;
- nested transactions, double commit, commit after exit, and invalid tokens
  receive SESSION_COMMIT_CONFLICT;
- concurrent reset/turn cannot create two locks for one session;
- an exception before commit leaves the old snapshot active;
- one successful turn swaps intent, belief, and interaction together;
- transition attacks that change profile/session ID, rewrite history, append
  multiple turns, substitute intent directly, or mismatch reducer output are
  rejected;
- serialize-deserialize is an identity operation;
- golden codec tests cover unknown schema/field, malformed operation and enum,
  canonical tuple/frozenset bytes, and repeated byte-identical encoding;
- replaying accepted batches reconstructs current IntentState;
- generated operation sequences preserve all aggregate invariants.

Property-based testing is valuable for reducer sequences but is a development
dependency, not a runtime dependency. Deterministic regression seeds are kept
for every discovered failure.

Every stable ErrorCode has at least one reachability case, maintained as a
single parameterized error-code table.

### 7.7 Official adapter regression

The integration test verifies only the organizer-facing contract:

- reset precedes respond;
- message and usage shapes are valid;
- ask_attribute is an allowed protocol value or None;
- recommendation IDs are valid, unique, ordered, and limited to top_k and 10;
- TurnRecord stores the post-normalization recommendation order;
- the unchanged official evaluator still runs end to end.

Passing this test demonstrates compatibility, not product quality.

## 8. Implementation sequence

### M0 — Package foundation

- add pyproject.toml with a src layout and Python 3.10 floor;
- configure one test runner, formatting, and static checks;
- add package exports without placeholder business modules;
- preserve the current official baseline test command.

Exit criterion: editable install and both old and empty new test suites run
from a clean checkout.

### M1 — Values and validation

- implement models, operations, aggregates, registry, errors, and local
  validators;
- freeze collection ownership and error-code behavior;
- add model, registry, and aggregate invariant tests.

Exit criterion: every contract invariant that does not require transition
logic has a positive and negative test.

### M2 — Pure reducer

- implement canonical working state and all six operations;
- add ordered atomicity, no-op, version, replay, and generated-sequence tests;
- benchmark reducer overhead separately from model or retrieval latency.

Exit criterion: reducer has no I/O imports and all failure paths preserve the
input snapshot.

### M3 — Interaction, belief, codec, and store

- implement derived views and search-belief validators;
- implement reducer-aware aggregate and transition validation;
- implement the versioned snapshot codec;
- implement per-session transactions and concurrency tests;
- test rejected-update and fallback turn records.

Exit criterion: aggregate commits are atomic and recorded history can
reconstruct the current intent.

### M4 — Official adapter integration

- move product logic into shopping_copilot application modules;
- reduce starter/agent.py to API adaptation and composition;
- normalize replies before constructing TurnRecord;
- run official evaluator regression and record the result.

Exit criterion: official API compatibility is unchanged and no project-domain
logic lives in evaluator.

### M5 — Strategy consumers

- connect query understanding through the draft boundary;
- connect Probe and SearchBelief creation;
- add retrieval, ranking, asking, and profile-prior consumers one at a time;
- evaluate each feature with ablation tests against the frozen baseline.

Exit criterion: strategy modules consume session context without adding
policy fields to it or weakening its invariants.

## 9. Repository hygiene

- Do not edit evaluator or frozen official data to make a test pass.
- Keep starter/agent.py thin once integration begins.
- Do not commit catalog files, result files, caches, virtual environments,
  credentials, model downloads, or generated traces.
- Do not create empty directories or placeholder modules for future features.
- Add a short package docstring only where it establishes a real boundary;
  avoid decorative boilerplate.
- Keep one authoritative contract. Research drafts link forward and do not
  duplicate normative rules.
- Use stable lower_snake_case wire values and explicit schema versions.
- Store regression fixtures under tests, not under production packages.
- Preserve deterministic ordering in serialized state, error details, and
  test output.
- Any contract change requires an explicit versioned design review before its
  implementation and tests change.

## 10. Definition of done for session-context v1

The feature is complete when:

1. every frozen contract invariant is implemented and directly tested;
2. the public package API is small, documented, and cycle-free;
3. invalid batches and failed commits cannot partially mutate state;
4. accepted history replays to the exact active intent;
5. same-session turn processing is serialized and cross-session processing is
   independent;
6. snapshot serialization is deterministic and validates on load;
7. the official adapter regression passes without evaluator modification;
8. the repository contains no generated artifacts or duplicate design truth;
9. the design status is updated from implementation pending to implemented
   only after all preceding conditions hold.
