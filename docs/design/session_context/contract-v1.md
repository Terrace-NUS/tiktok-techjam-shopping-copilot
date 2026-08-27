# Session Context Contract v1

- Status: **frozen for implementation**
- Date: **2026-08-27**

This is the normative contract for session-context v1. The words **MUST**,
**MUST NOT**, **SHOULD**, and **MAY** describe implementation requirements.
Research motivation remains in
[`design-rationale.md`](design-rationale.md); implementation sequencing remains
in [`implementation-plan.md`](implementation-plan.md).

## 1. Scope

Session context provides one shared, replayable state boundary for:

- query understanding;
- query compilation;
- probe retrieval;
- retrieval and ranking;
- proactive asking;
- the official `reset/respond` adapter.

It MUST distinguish four semantic categories:

```text
SessionContext
├── ProfilePrior           immutable external prior
└── SessionState
    ├── IntentState        current user-need facts
    ├── InteractionContext append-only interaction history
    └── SearchBelief       current catalog-derived observation
```

Session context does not choose a retriever, ranker, question, or certainty
algorithm. Those components consume this contract.

## 2. Trust boundaries

The runtime has three input stages:

```text
untrusted parser / model output
            │
            ▼
PreferenceDraft and draft operations
            │
            ▼
trusted grounding, normalization, ID assignment
            │
            ▼
StateUpdateBatch containing committed operations
            │
            ▼
deterministic reducer
```

`StateUpdateBatch` is not an LLM wire schema. A model MUST NOT allocate
preference IDs or call the reducer directly.

The reducer MAY receive an immutable `FacetRegistry` as domain configuration.
It MUST NOT access a product catalog, retriever, LLM, ranker, profile compiler,
official evaluator, network, or clock.

## 3. Primitive types and enums

The persisted scalar domain is deliberately JSON-compatible:

```python
ScalarValue = str | int | float | bool
PreferenceValue = ScalarValue | tuple[ScalarValue, ...]
```

Boolean values MUST NOT be accepted as numeric values. Numeric values MUST be
finite; `NaN` and infinities are invalid.

```python
class Operator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"

class SemanticPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class Commitment(str, Enum):
    HARD = "hard"
    SOFT = "soft"

class PreferenceSource(str, Enum):
    USER_EXPLICIT = "user_explicit"
    BEHAVIORAL_FEEDBACK = "behavioral_feedback"
    SYSTEM_INFERRED = "system_inferred"

class ProbeQuality(str, Enum):
    VALID = "valid"
    LOW_QUALITY = "low_quality"
    INSUFFICIENT = "insufficient"

class FeedbackSignal(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    SELECTED = "selected"
    REJECTED = "rejected"
    COMPARATIVE = "comparative"
```

Range language such as “between 50 and 100” MUST be normalized by the trusted
boundary into lower and upper predicates before committed IDs are assigned.
`BETWEEN` is therefore not a persisted v1 operator.

## 4. Facet registry

`FacetRegistry` is an immutable schema, not a product-catalog dependency. For
each canonical facet ID it provides at least:

- the facet kind (`categorical` or `numeric`);
- canonical value normalization;
- the allowed operator family.

The v1 operator families are closed:

- categorical facets accept `EQ`, `NEQ`, `IN`, and `NOT_IN`;
- numeric facets accept `LT`, `LE`, `GT`, and `GE`.

An exact numeric value is normalized into inclusive lower and upper predicates.
Numeric `EQ` and `NEQ` are not committed forms in v1.

Structured preferences MUST use a registered canonical facet. A requirement
that cannot be grounded safely MUST remain semantic-only with `facet=None`.
The committed validator rejects an unknown non-null facet with
`UNKNOWN_FACET`; only the trusted grounding layer may fall back to a new
semantic-only draft.

The registry is internal. Official `ask_attribute` values are adapter protocol
keys and MUST NOT be treated as the facet registry. In particular, `other` is
not a facet.

## 5. Profile prior

The profile is outside current intent:

```python
@dataclass(frozen=True)
class ProfilePrior:
    purchase_frequency: str
    average_prior_rating: float | None
    rating_style: str
    preference_tags: tuple[str, ...]
    summary: str
```

`SessionContext.profile` MAY be `None` for generic callers. If present, it MUST
match the official profile shape; only `average_prior_rating` is nullable.

Profile data MUST NOT create a hard filter and MUST NOT override a current
explicit session preference. Conversion into weak ranking priors belongs to a
future profile compiler, not this contract.

## 6. Preferences

### 6.1 Draft and committed forms

```python
@dataclass(frozen=True)
class PreferenceDraft:
    facet: str | None
    operator: Operator | None
    value: PreferenceValue | None

    semantic_text: str | None
    semantic_polarity: SemanticPolarity | None

    commitment: Commitment
    source: PreferenceSource

    source_turn: int
    evidence_text: str
    interpretation_confidence: float

@dataclass(frozen=True)
class Preference(PreferenceDraft):
    id: str
```

A committed preference MUST satisfy all of the following:

1. `facet`, `operator`, and `value` are either all present or all absent.
2. `semantic_text` and `semantic_polarity` are either both present or both
   absent.
3. At least one representation is present.
4. `id`, present text values, and evidence are non-empty after trimming.
5. `source_turn >= 1`.
6. `interpretation_confidence` is finite and lies in `[0, 1]`.
7. `BEHAVIORAL_FEEDBACK` and `SYSTEM_INFERRED` MUST be soft.
8. `USER_EXPLICIT` MAY be hard or soft.
9. `IN` and `NOT_IN` require a non-empty, homogeneous, deduplicated tuple in
   canonical order.
10. Numeric operators require a finite `int` or `float`, excluding `bool`.

Structured and semantic representations MAY coexist only when they describe
the same atomic preference. The validator can mechanically enforce that
`EQ/IN` do not carry negative semantic polarity and `NEQ/NOT_IN` do not carry
positive semantic polarity. Deeper semantic agreement is the trusted
grounding layer's responsibility.

### 6.2 Identity and logical equality

New IDs are assigned by trusted code using the session-scoped form:

```text
p_{turn}_{operation_index}_{preference_index}
```

An ID MUST be unique for the entire session lifetime, including removed
preferences visible in interaction history. IDs MUST NOT be recycled.

Logical equality ignores `id`, `source_turn`, and evidence and compares the
canonical meaning of a preference. Before constructing a batch, the trusted
adapter MUST reuse an active preference ID for a logical reassertion.

- Same ID and same canonical meaning is a valid no-op and retains the existing
  Preference object, including its original evidence, confidence, source, and
  source turn.
- Same ID and different canonical payload is `PREFERENCE_ID_CONFLICT`.
- Same canonical payload with a new ID is rejected as
  `DUPLICATE_PREFERENCE_SEMANTICS` unless normalized before batching.

## 7. Intent state

```python
@dataclass(frozen=True)
class IntentState:
    goal: str | None
    preferences: tuple[Preference, ...]
    dont_care_facets: frozenset[str]
    version: int
```

The initial state has `goal=None`, no preferences, no don't-care facets, and
`version=0`. `SwitchGoal` sets the first goal as well as replacing a later one.
A non-null goal MUST be non-empty after trimming.

Preference tuple position has no semantic meaning. Active preferences use the
numeric ID order `(turn, operation_index, preference_index)` parsed from their
required ID form. The reducer emits that order after every successful batch.
`dont_care_facets` is a set in memory and is sorted by facet ID on the wire.

Don't-care has three-state semantics:

```text
no preference, no marker   → unset / unknown
facet in dont_care_facets  → explicitly does not matter
active facet preference    → expressed preference
```

The invariant is one-way:

```text
facet in dont_care_facets ⇒ no active preference for that facet
```

Absence of a preference does not imply don't-care.

### 7.1 Canonical facet state

For each categorical facet and each commitment level, active state allows at
most:

- one positive selector (`EQ` or `IN`);
- one negative selector (`NEQ` or `NOT_IN`).

Multiple `EQ` values MUST NOT imply OR; the trusted understanding layer uses a
single `IN` preference. Every active positive selector MUST retain at least
one value after:

1. intersection with the other commitment level's positive selector, when it
   exists; and
2. removal of values named by either active negative selector.

This rejects hard/soft positive selectors with no overlap, a positive selector
fully excluded by a hard negative, and a soft negative that rules out the only
hard-positive value. Negative-only state is valid because an unknown catalog
universe is not materialized by the reducer.

For each numeric facet and commitment level, active state allows at most one
effective lower bound and one effective upper bound. Adding a numeric bound is
conjunctive and keeps only the strongest bound. For an equal lower endpoint,
`GT` is stronger than `GE`; for an equal upper endpoint, `LT` is stronger
than `LE`. The interval within each commitment and the common hard/soft
intersection MUST be non-empty. Equal lower and upper endpoints form a valid
singleton only when both effective bounds are inclusive. Conflicts are
rejected for clarification rather than silently retained.

Semantic-only preferences have no facet-level canonicalization and are
addressed by ID.

## 8. Committed state operations

The reducer accepts only this closed v1 union:

```python
@dataclass(frozen=True)
class AddPreference:
    op: Literal["add_preference"]
    preference: Preference

@dataclass(frozen=True)
class ReplaceFacet:
    op: Literal["replace_facet"]
    facet: str
    preferences: tuple[Preference, ...]

@dataclass(frozen=True)
class RemovePreference:
    op: Literal["remove_preference"]
    preference_ids: tuple[str, ...]

@dataclass(frozen=True)
class ClearFacet:
    op: Literal["clear_facet"]
    facet: str

@dataclass(frozen=True)
class SetDontCare:
    op: Literal["set_dont_care"]
    facet: str

@dataclass(frozen=True)
class SwitchGoal:
    op: Literal["switch_goal"]
    new_goal: str
    carry_preference_ids: tuple[str, ...] = ()
```

Operation semantics are deterministic:

- `AddPreference` adds a non-conflicting preference. It removes that facet
  from don't-care. Ambiguous categorical addition is rejected; understanding
  must issue `ReplaceFacet`.
- `ReplaceFacet` removes all current preferences for the facet, installs a
  non-empty canonical replacement, and removes the facet from don't-care.
- `RemovePreference` removes existing IDs. Unknown IDs reject the batch.
- `ClearFacet` removes all preferences and any don't-care marker for the facet,
  returning it to unset.
- `SetDontCare` removes all preferences for the facet and adds its explicit
  don't-care marker.
- `SwitchGoal` clears all preferences and don't-care markers, then restores
  only the listed pre-batch preferences with their original IDs and evidence.

`ReplaceFacet.preferences` MUST be non-empty and every preference MUST have the
same facet as the operation. Semantic-only preferences cannot be replaced or
cleared by facet and must be removed by ID.

A batch MAY contain at most one `SwitchGoal`, and it MUST be the first
operation. Additional operations can express the new goal's constraints or
don't-care markers. Carried IDs must be unique and exist in the pre-batch
active state.

Operation order is semantic and MUST be preserved. Within fields whose order is
not semantic, `ReplaceFacet.preferences`, `RemovePreference.preference_ids`,
and `SwitchGoal.carry_preference_ids` MUST be unique and use canonical numeric
preference-ID order before a batch reaches the reducer.

## 9. Atomic update batch

```python
@dataclass(frozen=True)
class StateUpdateBatch:
    turn: int
    base_intent_version: int
    operations: tuple[StateOperation, ...]
```

Requirements:

- `turn >= 1` and operations are non-empty.
- `base_intent_version` MUST equal the current intent version.
- Operations execute in tuple order against a temporary immutable state.
- Each operation's shape and preconditions are validated before application.
- Every intermediate state and the final state MUST satisfy canonical
  invariants. A later operation cannot repair an earlier invalid state.
- Any failure rolls back the entire batch.
- A valid batch that changes canonical intent increments version exactly once.
- A valid logical no-op leaves version unchanged.
- An invalid batch leaves version and intent unchanged.
- The same state, registry, and batch MUST produce the same result or the same
  error code.
- A non-canonical set-like tuple is rejected with `NON_CANONICAL_VALUE`;
  the committed reducer does not silently reorder untrusted input.

## 10. Interaction context

Interaction history is append-only. Convenience state is derived rather than
mutated separately.

```python
@dataclass(frozen=True)
class ProductFeedback:
    product_ids: tuple[str, ...]
    signal: FeedbackSignal
    compared_to_ids: tuple[str, ...]
    evidence_text: str

@dataclass(frozen=True)
class TurnRecord:
    turn: int
    user_message: str

    intent_version_before: int
    accepted_update: StateUpdateBatch | None
    intent_version_after: int

    assistant_message: str
    question: str | None
    question_key: str | None
    ask_attribute: str | None
    shown_product_ids: tuple[str, ...]
    feedback: tuple[ProductFeedback, ...]

    search_belief_probe_id: str | None

@dataclass(frozen=True)
class InteractionContext:
    turns: tuple[TurnRecord, ...]
```

`question_key` is the internal, canonical key used to prevent repeated
questions. `ask_attribute` is the exact official adapter value; it may be
`other` and is not a facet ID. In v1, `question`, `question_key`, and
`ask_attribute` are either all present or all absent.

Feedback product IDs are ordered, unique, non-empty, and MUST refer to products
shown strictly before the current turn. `COMPARATIVE` feedback requires
non-empty, disjoint `compared_to_ids`; other signals leave it empty.
`POSITIVE` and `NEGATIVE` record verbal sentiment; `SELECTED` and
`REJECTED` record a behavioral choice. For `COMPARATIVE`, `product_ids`
are preferred over `compared_to_ids`.

`shown_product_ids` records the actual externally returned order after adapter
normalization, deduplication, validation, and Top-K truncation.

Turn records MUST start at 1 and be contiguous. There is exactly one record per
processed user turn. Batch turn, new preference source turns, and TurnRecord
turn MUST agree; carried and logically reasserted preferences retain their
original source turn.

`state_changed` is a derived property:

```python
intent_version_before != intent_version_after
```

Required derived views include:

- last non-empty shown product batch;
- all previously shown products in order;
- question keys asked since the last goal switch;
- the most recent question and assistant message.

Rejected parsing or validation details go to trace storage, not active intent.
The user turn still receives a TurnRecord with `accepted_update=None` and equal
before/after versions.

## 11. Search belief

Search belief is an immutable observation created by Probe, not an intent fact
or policy decision.

```python
@dataclass(frozen=True)
class CertaintyEvidence:
    probe_id: str
    probe_size: int
    raw_concentration: float | None
    quality_status: ProbeQuality
    quality_reasons: tuple[str, ...]

@dataclass(frozen=True)
class ValueMass:
    value: ScalarValue
    mass: float

@dataclass(frozen=True)
class FacetStats:
    facet: str
    entropy: float
    coverage: float
    top_values: tuple[ValueMass, ...]

@dataclass(frozen=True)
class CandidateMode:
    id: str
    label: str
    mass: float
    representative_ids: tuple[str, ...]

@dataclass(frozen=True)
class SearchBelief:
    based_on_intent_version: int
    certainty: float | None
    certainty_method: str
    certainty_evidence: CertaintyEvidence
    candidate_modes: tuple[CandidateMode, ...]
    facet_stats: tuple[FacetStats, ...]
```

Belief validation uses the fixed constant
`MASS_TOLERANCE = 1e-9`. Definitions use the unique valid probe candidate set
as denominator:

- `probe_size` is its size.
- `CandidateMode.mass` is the fraction of all probe candidates assigned to the
  mode. Modes may omit a tail; total mode mass lies in `[0, 1]` within numeric
  tolerance.
- `FacetStats.coverage` is the fraction of all probe candidates with one usable
  canonical value for the facet.
- `ValueMass.mass` is conditional on covered candidates. `top_values` may be
  truncated, so its mass sum lies in `(0, 1]`.
- `FacetStats.entropy` is normalized to `[0, 1]` over the complete covered
  value distribution, not only `top_values`. A facet with zero coverage is
  omitted; a single observed value has entropy zero.

All probability-like values are finite and lie in `[0, 1]`. A sum may exceed
one only by `MASS_TOLERANCE`. IDs and method names are non-empty;
`certainty_method` is a versioned identifier such as `bods_v1`, not display
text.

Additional canonical invariants are:

- `based_on_intent_version >= 0` and `probe_size >= 0`;
- a `VALID` probe has `probe_size > 0` and no quality reasons;
- a non-valid probe has at least one unique, non-empty, lower-snake-case reason
  code; reason codes are sorted;
- candidate-mode IDs are unique, each mode has positive mass and at least one
  unique representative ID, and modes are sorted by descending mass then ID;
- representative IDs preserve probe ranking and are not otherwise sorted;
- facet-stat facet IDs are unique, registered canonical facets;
- each emitted facet has positive coverage and at least one top value;
- top values are unique, have positive mass, and are sorted by descending mass
  then the canonical scalar wire key;
- facet stats are sorted by facet ID.

The Probe producer validates canonical facet values against the same injected
`FacetRegistry`. Structural loading can verify the registry value and the
statistics above, but cannot recompute entropy without the producer's complete
covered-value distribution.

Availability truth table:

```text
quality_status == VALID
    ⇒ certainty and raw_concentration are present and in [0, 1]

quality_status != VALID
    ⇒ certainty is None; raw_concentration may be None
```

Whenever `raw_concentration` is present, it is finite and lies in `[0, 1]`.

Low certainty means a valid probe observed a broad decision space. Unavailable
certainty means the system lacks reliable evidence. These states MUST NOT share
the same policy fallback.

When attached to active state, `based_on_intent_version` MUST equal
`IntentState.version`. A real intent change clears the previous belief. A
logical no-op MAY preserve it. Question utility, route weights, and policy
decisions MUST NOT be stored in `SearchBelief`.

`TurnRecord.search_belief_probe_id` is a trusted audit reference. When the
turn is created it MUST equal the probe ID used for that turn's belief, if one
was produced. Because active state retains only the current belief, a later
snapshot loader can shape-check historical probe IDs but cannot reconstruct
their complete referential provenance.

## 12. Session context and commit semantics

```python
@dataclass(frozen=True)
class SessionState:
    intent: IntentState
    interaction: InteractionContext
    search_belief: SearchBelief | None

@dataclass(frozen=True)
class SessionContext:
    session_id: str
    profile: ProfilePrior | None
    state: SessionState
```

The session store MUST serialize processing per session with a non-reentrant
exclusive lock. A separate guard protects creation and lookup of session-lock
entries. Nested transactions for the same session, double commit, commit after
transaction exit, and an invalid transaction token fail with
`SESSION_COMMIT_CONFLICT` rather than blocking or overwriting a newer
snapshot. The store MUST reject duplicate, skipped, or out-of-order turns.
Different sessions remain independent and MAY run concurrently.

`base_intent_version` protects intent updates and Probe invalidation; contiguous
turn checks protect interaction history. P0 does not require a second session
revision counter while the per-session lock is enforced.

A processed turn has two distinct guarantees:

1. Intent-batch atomicity: an invalid batch does not mutate IntentState.
2. Session-commit atomicity: the final IntentState, active SearchBelief,
   normalized external response, and one TurnRecord replace the previous
   SessionContext together.

Before swapping a snapshot, the store MUST run a pure
`validate_session_transition(previous, next, expected_turn, registry)` check.
It verifies at least:

- session ID and profile are unchanged;
- the old turn tuple is an exact prefix of the next tuple and exactly one
  record is appended;
- the appended turn is the expected contiguous turn;
- its before/after versions match the previous and next intents;
- `accepted_update=None` implies byte-equivalent previous and next intent;
- an accepted batch, including a logical no-op, reproduces the next intent
  exactly when passed to the reducer with the previous intent;
- belief preservation, invalidation, and version attachment follow this
  contract.

An aggregate that is valid in isolation but fails this transition check MUST
NOT commit.

Failure behavior is frozen by processing stage:

| Outcome before commit | Intent in next snapshot | accepted_update | SearchBelief | External behavior |
| --- | --- | --- | --- | --- |
| Parse or grounding fails | Previous intent | None | Previous valid belief may remain | Fallback plus one record |
| Reducer rejects batch | Previous intent | None | Previous valid belief may remain | Fallback plus one record |
| Accepted logical no-op | Previous intent and version | Batch | Previous valid belief may remain | Normal or fallback response plus one record |
| Accepted real change; downstream succeeds | Reduced intent | Batch | Newly validated belief or None | Normal response plus one record |
| Accepted real change; downstream later fails | Reduced intent | Batch | Previous belief is cleared; a newly validated belief may remain | Fallback plus one record |
| Final store commit fails | Previous complete snapshot | No new record | Previous belief | Raise commit failure; do not return an unrecorded successful response |

Thus a Probe, retrieval, ranking, or normalization failure MUST NOT erase an
intent update already accepted by the reducer. A returned normalized response
is released to the caller only after its matching TurnRecord and context
transition commit successfully.

## 13. Stable error model

Domain failures expose a stable machine-readable code, field path, optional
operation index, and safe details. Details use an immutable, canonically sorted
tuple of key/scalar pairs in the domain and become an object only on the wire.
Tests assert codes, not error prose. Validation order is deterministic, and
every listed code MUST have at least one direct reachability test.

### Lifecycle

- `SESSION_NOT_FOUND`
- `SESSION_ALREADY_EXISTS`
- `INVALID_SESSION_ID`
- `INVALID_PROFILE`
- `TURN_OUT_OF_ORDER`
- `SESSION_COMMIT_CONFLICT`
- `INVALID_SESSION_TRANSITION`

### Batch and operations

- `EMPTY_BATCH`
- `STALE_BASE_VERSION`
- `INVALID_OPERATION_ORDER`
- `MULTIPLE_GOAL_SWITCH`
- `UNKNOWN_PREFERENCE_ID`
- `INVALID_CARRY_ID`
- `EMPTY_REPLACEMENT`
- `FACET_MISMATCH`
- `NON_CANONICAL_VALUE`

### Preference and canonical state

- `UNKNOWN_FACET`
- `INVALID_GOAL`
- `INVALID_REPRESENTATION`
- `INVALID_OPERATOR_VALUE`
- `INVALID_OPERATOR_FOR_FACET`
- `INVALID_COMMITMENT_FOR_SOURCE`
- `INVALID_SOURCE_TURN`
- `INVALID_CONFIDENCE`
- `DUPLICATE_PREFERENCE_ID`
- `PREFERENCE_ID_CONFLICT`
- `DUPLICATE_PREFERENCE_SEMANTICS`
- `MULTIPLE_POSITIVE_SELECTOR`
- `MULTIPLE_NEGATIVE_SELECTOR`
- `EMPTY_NUMERIC_INTERSECTION`
- `EMPTY_CATEGORICAL_DOMAIN`
- `DONT_CARE_CONFLICT`

### Interaction and belief

- `INVALID_TURN_RECORD`
- `INVALID_TURN_SEQUENCE`
- `INVALID_QUESTION_FIELDS`
- `INVALID_FEEDBACK`
- `INVALID_FEEDBACK_REFERENCE`
- `TURN_RECORD_VERSION_MISMATCH`
- `STALE_SEARCH_BELIEF`
- `CERTAINTY_QUALITY_MISMATCH`
- `INVALID_PROBE_EVIDENCE`
- `INVALID_MASS_DISTRIBUTION`
- `DUPLICATE_MODE_ID`
- `DUPLICATE_FACET_STATS`
- `DUPLICATE_FACET_VALUE`

### Serialization boundary

- `UNKNOWN_SCHEMA_VERSION`
- `INVALID_SNAPSHOT`
- `UNKNOWN_FIELD`

`INVALID_SNAPSHOT` covers malformed JSON, an unknown operation
discriminator, and invalid enum wire values when no more specific domain code
can be reached safely. `SESSION_COMMIT_CONFLICT` is reserved for invalid
transaction lifecycle or token use; ordinary duplicate or out-of-order user
turns use `TURN_OUT_OF_ORDER`.

## 14. Serialization and public API

Frozen dataclasses are the domain representation. Tuples and frozensets MUST be
used instead of mutable collections; a frozen object MUST NOT contain a mutable
dict. Pydantic or another schema library MAY validate untrusted JSON at the
boundary and then construct domain objects.

The session-context package will expose a small public API from its
`__init__.py` and keep storage and parser models private. Serialized snapshots
MUST include a schema identifier:

```text
shopping-copilot/session-context/v1
```

Canonical snapshot JSON uses UTF-8, lexicographically sorted object keys,
compact separators, unescaped Unicode, and rejects non-finite numbers.
Set-like arrays use the canonical orders defined above; sequence-like arrays,
including turns, operations, shown products, and representative IDs, preserve
their semantic order. Encoding the same domain snapshot twice MUST produce
identical bytes.

Deserialization rejects unknown fields and schema versions, reconstructs the
frozen domain values, and validates every mechanically verifiable invariant
from live construction. Producer-only facts that require unavailable catalog
rows, a complete Probe distribution, or semantic interpretation are validated
when produced and cannot be recomputed by the loader. Replay of all accepted
batches from the initial intent MUST reproduce the current intent exactly.

## 15. Explicit non-goals for v1

- Long-term memory writes
- Relative `NUDGE` operations
- Catalog access inside the reducer
- Supersession graph in active state
- Arbitrary dynamic Python values
- LLM-generated committed IDs
- Retrieval or ranking policy inside state
- Certainty algorithm selection
- Question-utility persistence
- Distributed session coordination
