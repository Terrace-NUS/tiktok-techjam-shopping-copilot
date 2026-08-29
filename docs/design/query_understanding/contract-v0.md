# Query Understanding Working Contract v0

- Status: **review draft; QU core proposed, QU4 compiler wire deferred**
- Date: **2026-08-28**
- Compatibility targets: **Session Context Contract v1**, **Catalog Semantic
  Layer Contract v0**, and **Retrieval and Ranking Working Contract v0**
- P0 provider: **DeepSeek API**

This document defines how one user turn becomes a safe update proposal, how
that proposal reaches the existing session state, and how the accepted state
is compiled for retrieval. It is the project-owned implementation contract for
Query Understanding (QU).

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe proposed
requirements. They become implementation requirements only after this review
draft is accepted and marked frozen.

## 1. Core decision

### PROPOSED: DeepSeek is an interpreter, not the state owner

DeepSeek interprets the latest user turn and emits an **untrusted,
evidence-bearing proposal**. It MUST NOT rewrite the complete `IntentState`,
construct committed session operations, or calculate intent transparency.

The trusted application layer remains responsible for:

- checking that quoted evidence exists;
- resolving local references;
- assigning provenance and commitment;
- parsing currency, units, and numbers;
- invoking the release-bound catalog grounder;
- assigning committed preference IDs;
- planning the closed session operation vocabulary;
- previewing the result through `CatalogSemanticGateway`; and
- committing through the release-bound session transaction.

The existing deterministic reducer remains the sole accumulator of canonical
intent state.

This follows the deployed ShopTalk pattern: contextual language understanding
emits a small set of intent operators, while a mostly rules-based state tracker
owns state transitions.

## 2. Scope

Query Understanding v0 owns:

1. a bounded, provider-independent input view;
2. the DeepSeek prompt and structured-output adapter;
3. an untrusted `UnderstandingProposal` wire contract;
4. evidence and local-reference validation;
5. deterministic provenance and authority mapping;
6. release-bound grounding orchestration;
7. planning into the existing `StateUpdateBatch` vocabulary;
8. a compiled retrieval view built from accepted state; and
9. trace data and QU-specific failure semantics.

Query Understanding v0 does not own:

- raw catalog semantics or facet discovery;
- canonical value or category creation;
- session storage or reducer semantics;
- ProductFacetIndex mutation;
- retrieval route implementation;
- Probe or `C_t` calculation;
- ranking or question-selection policy;
- official evaluator behavior; or
- long-term profile learning.

## 3. End-to-end boundary

The normal turn flow is:

```text
CatalogBoundSessionTransaction captures SessionContext
    -> build bounded UnderstandingRequest
    -> DeepSeek returns untrusted UnderstandingProposal
    -> strict wire validation
    -> exact evidence and local-reference validation
    -> trusted normalization and provenance mapping
    -> release-bound RuntimeValueGrounder
    -> deterministic operation planner
    -> StateUpdateBatch, or no committed batch
    -> CatalogSemanticGateway.preview
    -> accepted IntentState
    -> Query Compiler
    -> q_lex + q_sem + constraints + soft preferences + direct instructions
    -> fixed Probe
    -> C_t and ephemeral D_t
    -> retrieval / ranking / response
    -> one atomic CatalogBoundSessionTransaction.commit
```

The causal order is therefore:

```text
natural language
    -> proposed meaning
    -> accepted canonical intent
    -> compiled candidate field
    -> fixed Probe observation
    -> C_t
```

Neither the previous `C_t` nor the previous `SearchBelief` is an input to
DeepSeek. This prevents a circular definition in which an old transparency
score changes the interpretation used to calculate the next score.

## 4. Trust and authority matrix

| Capability | DeepSeek | Trusted QU coordinator | Catalog grounder | Gateway / reducer |
| --- | --- | --- | --- | --- |
| Interpret user language | Propose | Validate | No | No |
| Quote evidence | Propose | Verify exactly | No | No |
| Resolve request-local handles | Propose | Verify and map | No | No |
| Assign provenance | No | Yes | No | Validate |
| Assign HARD/SOFT commitment | Hint only | Yes | No | Validate |
| Normalize currency and units | No authority | Yes | Validate operand | Validate |
| Declare canonical facet/value | No | No | Yes | Validate |
| Declare grounding success | No | No | Yes | Validate |
| Allocate Preference IDs | No | Yes | No | Validate |
| Construct committed operations | No | Yes | No | Execute |
| Modify IntentState | No | No | No | Yes |
| Create SearchBelief or `C_t` | No | No | No | Separate Probe authority |
| Control retrieval directly | No | Compile accepted facts only | No | No |

Prompt vocabulary is guidance, not permission. Seeing a facet name, value
example, active preference, or product reference in the request does not grant
DeepSeek authority to write it.

## 5. Understanding input

### 5.1 Request envelope

Each provider call consumes one immutable request:

```python
@dataclass(frozen=True)
class UnderstandingRequest:
    schema: Literal["shopping-copilot/query-understanding-request/v0"]
    request_id: str
    session_ref: str
    prompt_version: str
    turn: int
    base_intent_version: int
    catalog_semantic_release_id: str
    latest_utterance: str
    current_intent: IntentPromptView
    interaction: InteractionPromptView
    facets: tuple[FacetPromptSpec, ...]
```

Requirements:

- `request_id` matches `^qu_[a-f0-9]{32}$` and is unique per logical turn
  extraction; retries reuse it.
- `session_ref` matches `^s_[a-f0-9]{32}$`, is stable within one session, and
  is derived without embedding the raw official session ID or other PII.
- `prompt_version` is a non-empty lower-snake-case identifier no longer than
  64 characters.
- `turn` is the transaction's expected contiguous turn.
- `session_ref` is an opaque per-session binding produced locally. It is not
  required to expose the raw official session ID.
- `prompt_version` identifies the exact system instructions, facet
  descriptions, and few-shot bundle.
- `base_intent_version` equals the captured `IntentState.version`.
- `catalog_semantic_release_id` is supplied by the bound store, never chosen by
  the model.
- `latest_utterance` is the exact user message before a `TurnRecord` exists.
- All collections use deterministic order.
- The request is a view, not a serializable `SessionContext` replacement.
- Unknown request fields are rejected.

Canonical request JSON uses UTF-8, lexicographically sorted object keys,
compact separators, and unescaped Unicode. Semantic-order arrays preserve
order; set-like arrays use their domain's canonical order. Canonical encoding
exists for hashing and replay, not because model output is deterministic.

### 5.2 Current intent view

The model receives the current canonical meaning but not committed preference
IDs:

```python
@dataclass(frozen=True)
class ActivePreferencePromptView:
    ref: str
    meaning: str
    facet_hint: str | None
    operator_hint: str | None
    value_texts: tuple[str, ...]
    polarity: Literal["positive", "negative"]
    commitment: Literal["hard", "soft"]

@dataclass(frozen=True)
class IntentPromptView:
    goal: str | None
    active_preferences: tuple[ActivePreferencePromptView, ...]
    dont_care_facets: tuple[str, ...]
```

`ref` is a request-local opaque handle such as `active_0`. The coordinator
keeps the private mapping from local handle to real preference ID. A model
MUST NOT see, copy, alter, or mint the canonical `p_{turn}_{op}_{preference}`
identifier.

`meaning` is a deterministic rendering of the committed preference. It is not
free text generated anew for every request.

### 5.3 Interaction view

The default request contains only information required to resolve the latest
turn:

```python
@dataclass(frozen=True)
class ProductPromptView:
    ref: str
    title: str
    short_description: str
    shown_position: int

@dataclass(frozen=True)
class PromptMessageView:
    ref: str
    turn: int
    role: Literal["assistant"]
    text: str

@dataclass(frozen=True)
class InteractionPromptView:
    previous_assistant_message: PromptMessageView | None
    previous_question: PromptMessageView | None
    last_shown_products: tuple[ProductPromptView, ...]
```

The coordinator builds this view from existing pure interaction views and
bounded catalog presentation data.

Rules:

- The most recent assistant question is included with a local message handle
  because “yes”, “no”, and “either is fine” depend on it.
- The last relevant shown products are included when the user can refer to
  “the second one”, “that bag”, or a comparison.
- P0 does not include an arbitrary older-message collection. Canonical active
  intent carries persistent meaning; the previous assistant turn and last
  shown batch cover the supported reference window.
- The request MUST NOT contain the entire transcript.
- Raw catalog rows, embeddings, SearchBelief, route scores, and Probe
  candidates MUST NOT appear.
- `ProfilePrior` is excluded in P0. A future profile compiler may create
  separately identified weak priors; it cannot silently enter this request as
  an explicit user need.

Product handles such as `shown_1` are also request-local. The coordinator maps
them back to previously shown `parent_asin` values.

### 5.4 Facet prompt schema

DeepSeek receives compact natural-language descriptions, not only opaque
facet IDs:

```python
@dataclass(frozen=True)
class FacetPromptSpec:
    facet_hint: str
    description: str
    kind: Literal["categorical", "numeric", "semantic"]
    allowed_language_operators: tuple[str, ...]
    cardinality: Literal["single", "multi", "range", "open"]
    authority_mode: Literal[
        "catalog_grounded",
        "competition_evidence",
        "semantic_only",
    ]
    normalization_examples: tuple[str, ...]
```

`authority_mode` tells the model how the system normally treats the facet, but
the trusted grounder and coordinator remain authoritative.

The P0 request may describe the wider competition vocabulary:

```text
category
price / budget
brand
material
color
size
style
department / gender
feature
use_case
```

At the current verified Catalog Semantic release, `price` is the only ordinary
runtime facet. `system_product_category` is a separate reserved storage
adapter with special Gateway rules. Other needs are preserved as semantic-only
session preferences unless and until the release promotes them. Explicit
competition facets may also become ephemeral retrieval evidence under the
Retrieval Working Contract; that does not promote them to catalog truth.

Facet descriptions and few-shot examples form one versioned static prompt
prefix. Changing their semantics requires a prompt-version change and golden
test review.

## 6. Understanding output

### 6.1 Proposal envelope

DeepSeek returns exactly one structured object:

```python
class ProposalDisposition(str, Enum):
    APPLY = "apply"
    NO_CHANGE = "no_change"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNPARSEABLE = "unparseable"

@dataclass(frozen=True)
class UnderstandingProposal:
    schema: Literal["shopping-copilot/query-understanding-proposal/v0"]
    request_id: str
    session_ref: str
    prompt_version: str
    turn: int
    base_intent_version: int
    disposition: ProposalDisposition
    goal_change: GoalChangeDraft | None
    preference_changes: tuple[PreferenceChangeDraft, ...]
    feedback: tuple[FeedbackDraft, ...]
    behavioral_directives: tuple[BehavioralDirectiveDraft, ...]
    ambiguities: tuple[AmbiguityDraft, ...]
```

The response MUST echo `request_id`, `session_ref`, `prompt_version`, `turn`,
and `base_intent_version`. Mismatches are stale or cross-request responses and
are rejected.

All fields are present in the JSON Schema. Empty tuples are encoded as empty
arrays; absent `goal_change` is JSON `null`. Unknown fields and unknown enum
values are rejected.

### 6.2 Evidence

Every semantic claim that could change state or interaction history carries:

```python
@dataclass(frozen=True)
class DraftEvidence:
    user_quote: str
    context_refs: tuple[str, ...]
```

`user_quote` MUST be a non-empty contiguous substring of
`latest_utterance` after converting CRLF and lone CR line endings to LF on
both sides. No case-folding, whitespace collapse, ellipsis substitution, or
Unicode compatibility normalization is allowed for evidence matching. The
quote is never a paraphrase. Committed evidence is copied from the
corresponding original user span.

`context_refs` may point only to local handles present in the request, for
example:

- the actual `PromptMessageView.ref` for the previous question when the user
  says “yes”;
- `shown_1` when the user says “the second one”; or
- `active_2` when the user says “remove that condition”.

Context alone cannot create a user-explicit preference. There must always be a
current user quote that confirms, rejects, changes, or refers to it.

The coordinator checks every quote and handle. Model-generated evidence that
does not exist is a hard proposal failure.

Exact quotation is necessary but not sufficient. Before any proposed
facet/operator/value can become `USER_EXPLICIT + HARD`, a trusted
`EvidenceClaimValidator` MUST prove that the complete atomic claim is
supported by the evidence chain:

- a categorical or text value is present in the current quote, modulo a
  reviewed deterministic alias, or is exactly supplied by a referenced
  assistant option that the current quote unambiguously confirms;
- operator and polarity are supported by closed lexical/confirmation rules;
  the model cannot turn an unmarked positive value into an exclusion or invent
  an OR/AND relation;
- a numeric value, unit, and comparison direction are deterministically
  parsed from the quote or confirmed referenced option;
- a removal or carry targets only the referenced active preference;
- a shown-product reference proves which product received feedback, but does
  not prove that any of its unspoken attributes were explicitly requested;
  and
- every inferred attribute that is not textually supported is
  `SYSTEM_INFERRED + SOFT` or is rejected. It cannot be silently relabeled as
  explicit.

For example, the quote “I need a dress” does not support `color=black` even
though the quote itself exists and black is a valid catalog value. A proposal
making that claim fails with `QU_UNSUPPORTED_EVIDENCE_CLAIM`. The coordinator
does not silently downgrade a model-declared required claim; a separate,
properly labeled inference is required.

The evidence validator is closed and versioned. It may use exact matching,
reviewed aliases, numeric/unit parsers, yes/no confirmation rules, and the
request's verified local references. It MUST NOT use another generative model
to grant hard-filter authority.

### 6.3 Goal change

```python
@dataclass(frozen=True)
class GoalChangeDraft:
    new_goal: str
    carry_refs: tuple[str, ...]
    evidence: DraftEvidence
```

Rules:

- A true product-task change is separate from changing one facet.
- `new_goal` is a self-contained product-task phrase, not a category scope ID
  and not a copy of the whole utterance.
- A trusted `GoalNormalizer` verifies the product-task meaning against the
  evidence chain and removes discourse or control language such as “just
  browsing”, “ready to buy”, “show diverse results”, and politeness. Genuine
  product use cases such as “for a wedding” remain product meaning.
- If a self-contained product goal cannot be verified, no `SwitchGoal` is
  planned.
- If the normalized product goal equals the active goal, the planner does not
  emit `SwitchGoal`; doing so would accidentally invoke reset/carry semantics.
- Carry defaults to empty.
- An active preference is carried only when the latest user turn explicitly
  reasserts it or explicitly says to retain it, such as “same budget”.
- The model proposes local `carry_refs`; the coordinator maps and validates
  them.
- A category proposal, if any, remains a preference change and is not hidden
  inside `new_goal`.

### 6.4 Preference changes

The wire shape is a discriminated union. Each action contains only fields that
have meaning for that action:

```python
class DraftModality(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    INFERRED = "inferred"

class DraftOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    BETWEEN = "between"
    UNSPECIFIED = "unspecified"

@dataclass(frozen=True)
class AssertPreferenceDraft:
    action: Literal["assert"]
    facet_surface: str
    facet_hint: str | None
    operator_hint: DraftOperator
    value_surfaces: tuple[str, ...]
    semantic_text: str
    semantic_polarity: Literal["positive", "negative"]
    modality: DraftModality
    evidence: DraftEvidence

@dataclass(frozen=True)
class ReplacePreferenceDraft:
    action: Literal["replace"]
    target_refs: tuple[str, ...]
    facet_surface: str
    facet_hint: str | None
    operator_hint: DraftOperator
    value_surfaces: tuple[str, ...]
    semantic_text: str
    semantic_polarity: Literal["positive", "negative"]
    modality: DraftModality
    evidence: DraftEvidence

@dataclass(frozen=True)
class RemovePreferenceDraft:
    action: Literal["remove"]
    target_refs: tuple[str, ...]
    evidence: DraftEvidence

@dataclass(frozen=True)
class ClearFacetDraft:
    action: Literal["clear_facet"]
    facet_surface: str
    facet_hint: str
    evidence: DraftEvidence

@dataclass(frozen=True)
class SetDontCareDraft:
    action: Literal["set_dont_care"]
    facet_surface: str
    facet_hint: str
    evidence: DraftEvidence

PreferenceChangeDraft = (
    AssertPreferenceDraft
    | ReplacePreferenceDraft
    | RemovePreferenceDraft
    | ClearFacetDraft
    | SetDontCareDraft
)
```

Each assert or replacement is one atomic preference:

- conjunctions over different facets are separate draft objects;
- same-facet alternatives use one `IN`/`NOT_IN` object with multiple
  `value_surfaces`;
- one range may use `BETWEEN` with exactly two ordered endpoint surfaces; and
- a semantic quality with no safe facet uses `facet_hint=None`,
  `operator_hint=UNSPECIFIED`, and non-empty self-contained `semantic_text`.

Reference and target cardinality is exact:

- `RemovePreferenceDraft.target_refs` is non-empty, ordered, unique, and
  contains only active local preference refs.
- For a complete registered-facet replacement,
  `ReplacePreferenceDraft.target_refs` is empty; the facet identifies the
  complete state being replaced.
- For a semantic-only replacement, `target_refs` is non-empty, ordered,
  unique, and contains only active semantic-only refs. The trusted planner
  verifies that every target belongs to the meaning being superseded.
- `ClearFacetDraft` and `SetDontCareDraft` are rejected during QU business
  validation when the facet is unknown, non-committable in the final scope, or
  `system_product_category`. The planner does not wait for Gateway rejection
  to discover this shape error.

Action semantics are:

- `AssertPreferenceDraft` means “add this meaning to the current need”. The
  trusted planner may still compile it to `ReplaceFacet` when canonical state
  requires one combined `IN` selector.
- `ReplacePreferenceDraft` means that prior state on the affected dimension is
  superseded, as in “actually blue”.
- Exclusion is expressed by a negative operator and polarity, not a free-form
  delete action.
- `RemovePreferenceDraft` retracts one or more specific active meanings, as in
  “black is not necessary anymore”.
- `ClearFacetDraft` returns one registered structured facet to unknown.
- `SetDontCareDraft` records that one registered, committable facet explicitly
  does not matter. It is not equivalent to absence.
- For an unsupported facet such as current-release color, “color does not
  matter” MUST be represented as `RemovePreferenceDraft` over every matching
  active semantic-only local ref. If none exists, it is `NO_CHANGE`. P0 has no
  persistent unsupported-facet don't-care marker and MUST NOT smuggle one into
  `SearchBelief` or a directive.
- “between 50 and 100” may use draft-only `BETWEEN`. Trusted normalization
  expands it into canonical lower and upper predicates before ID assignment.
- Relative language such as “cheaper” does not create a reducer `NUDGE`
  operation. It remains semantic-only unless a trusted deterministic
  reference resolver can produce an ordinary predicate.

Required modality is used for unhedged explicit constraints, prohibitions,
and words such as “must”, “only”, and “under”. Preferred modality is used for
phrases such as “prefer”, “ideally”, and “if possible”. Inferred modality
represents a model hypothesis rather than a user assertion.

DeepSeek's modality is only a hint. Section 8 defines the trusted mapping.

### 6.5 Feedback

Feedback about products already shown is kept separate from inferred product
attributes:

```python
@dataclass(frozen=True)
class FeedbackDraft:
    signal: Literal[
        "positive",
        "negative",
        "selected",
        "rejected",
        "comparative",
    ]
    product_refs: tuple[str, ...]
    compared_to_refs: tuple[str, ...]
    evidence: DraftEvidence
```

The coordinator maps valid local product handles into the existing
`ProductFeedback` shape. “I like the second one” records feedback; it does not
automatically turn every property of that product into a user-explicit
preference. Any later attribute inference is separately marked
`SYSTEM_INFERRED` and remains soft.

Feedback shape validation mirrors Session Context v1:

- `product_refs` is non-empty, ordered, and unique;
- every ref maps to a product shown strictly before the current turn;
- `COMPARATIVE` requires non-empty `compared_to_refs` disjoint from
  `product_refs`; and
- all other signals require empty `compared_to_refs`.

### 6.6 Behavioral directives

Direct instructions about how to present or search are one-turn policy
overrides:

```python
@dataclass(frozen=True)
class BehavioralDirectiveDraft:
    kind: Literal[
        "increase_diversity",
        "decrease_diversity",
    ]
    evidence: DraftEvidence
```

Behavioral directives:

- do not become `Preference` objects;
- do not enter `q_sem` as product meaning;
- do not change `C_t`;
- do not override hard constraints; and
- expire after the current turn unless the user expresses an actual
  persistent product preference.

Thus “show me very different styles” changes the diversity policy for this
turn but does not make the underlying product intent artificially vague.
Comparison, explanation, and repetition policy remain response-orchestration
concerns in P0 rather than expanding the QU directive schema.

### 6.7 Ambiguity

```python
@dataclass(frozen=True)
class AmbiguityDraft:
    kind: Literal[
        "goal",
        "action",
        "facet",
        "value",
        "reference",
    ]
    subject: str
    alternatives: tuple[str, ...]
    evidence: DraftEvidence
```

Rules:

- The model MUST NOT silently choose among materially different alternatives.
- `subject` is a non-empty self-contained description of the unresolved item.
- `alternatives` contains two or three ordered, unique values or validated
  local refs.
- `NEEDS_CLARIFICATION` requires at least one ambiguity and contains no
  goal or preference mutation.
- A release-bound `AMBIGUOUS` grounding result also produces no committed
  batch for that turn.
- P0 does not commit a “safe subset” of a turn that otherwise requires
  clarification. The whole intent update remains unchanged so the next answer
  can be interpreted against one unambiguous base state.
- Because the proposal schema permits only ambiguities under
  `NEEDS_CLARIFICATION`, feedback and behavioral directives from that same
  proposal are also discarded. No partial semantic payload survives.
- Unknown or unsupported meaning is not automatically ambiguity. If its
  semantics are clear, it is preserved as semantic-only.

DeepSeek does not write the user-facing clarification question. A separate
trusted clarification adapter consumes the validated ambiguity and produces
the complete existing interaction triple:

```text
question       template-rendered user-facing text
question_key   stable versioned key for repeat suppression
ask_attribute  mapped official attribute, or "other"
```

The three fields are all present or all absent, matching Session Context v1.
The general question-selection policy still belongs outside QU; this adapter
only turns a blocking QU ambiguity into a legal candidate question. Model text
is never copied directly into `TurnRecord.question`.

### 6.8 Disposition invariants

| Disposition | Allowed proposal content | State result |
| --- | --- | --- |
| `APPLY` | at least one goal change, preference change, feedback, or directive; no unresolved ambiguity | coordinator may produce a batch, feedback/directives only, or both |
| `NO_CHANGE` | no mutations, feedback, directives, or ambiguity | no batch |
| `NEEDS_CLARIFICATION` | one or more ambiguities only | no batch |
| `UNPARSEABLE` | no semantic payload | no batch |

The coordinator may reject an `APPLY` proposal and treat it as a pre-acceptance
failure. DeepSeek cannot force a state update through its disposition.

## 7. DeepSeek adapter

### PROPOSED: one non-thinking structured call

The P0 default is:

```python
response = client.responses.create(
    model="deepseek-v4-flash",
    instructions=VERSIONED_SYSTEM_PROMPT,
    input=serialized_request,
    reasoning={"effort": "none"},
    temperature=0.0,
    text={
        "format": {
            "type": "json_schema",
            "name": "query_understanding_proposal_v0",
            "schema": PROPOSAL_JSON_SCHEMA,
        }
    },
    max_output_tokens=1500,
)
```

Requirements:

- Use the current Responses API JSON Schema output, not plain JSON mode.
- Thinking is explicitly disabled to remove unnecessary reasoning overhead and
  improve the latency profile. This does not make model output or latency
  deterministic.
- Each turn creates one logical extraction request. The same request identity
  may have at most two provider attempts under Section 11.
- `deepseek-v4-pro` MAY be evaluated as an offline comparison or an explicitly
  configured fallback; P0 does not create a hidden multi-model agent loop.
- The adapter is stateless. The repository's `SessionContext` is the source of
  multi-turn state.
- The exact final JSON Schema, including nested objects, nullable goal change,
  enums, arrays, required fields, and `additionalProperties=false`, MUST pass
  an opt-in live compatibility test. Official support for `json_schema` does
  not substitute for testing the exact schema or for local revalidation.
- Request-body `session_ref` is only a local response-binding value. It is not
  DeepSeek's provider-level cache or scheduling isolation parameter. P0 does
  not rely on provider isolation. A future deployment that sets the top-level
  `user` parameter uses a separate non-PII local hash matching the provider's
  character and length rules.
- The API key is read from `DEEPSEEK_API_KEY` and never written to the
  repository, prompt trace, or error details.

The static prompt order is:

```text
system role and authority boundary
    -> output schema explanation
    -> facet descriptions
    -> 6-10 reviewed edge examples
    -> changing UnderstandingRequest
```

Keeping static textual instructions and examples before dynamic input may
improve prefix-cache hit rate. Whether a request actually hits is determined
only from returned usage fields. The Schema parameter is not assumed to be
cached, first use is not assumed to hit, and provider caching remains
best-effort, short-lived, and never state storage.

The few-shot set MUST include at least:

- exclusion versus retraction;
- replace versus additive alternative;
- clear versus don't-care;
- goal switch with explicit carry;
- confirmation of the previous assistant question;
- reference to a shown product;
- behavioral diversity override;
- unsupported but clear semantic-only need; and
- ambiguity that requires clarification.

## 8. Trusted interpretation policy

### 8.1 Validation order

The coordinator processes a response in this fixed order:

1. transport completion and provider status;
2. JSON parsing and exact schema validation;
3. echoed request, session, prompt, turn, and base-version validation;
4. disposition and action-shape validation;
5. exact user-evidence validation;
6. local preference, message, and product-reference validation;
7. atomic evidence-claim support validation;
8. provenance and modality mapping;
9. deterministic language, currency, unit, and numeric normalization;
10. final category-scope resolution;
11. release-bound runtime grounding;
12. operation planning and committed ID assignment;
13. `CatalogSemanticGateway.preview`; and
14. query compilation from the accepted final intent.

Failure at steps 1-13 cannot create a partial `StateUpdateBatch` or mutate the
captured `SessionContext`.

### 8.2 Provenance

DeepSeek does not output `PreferenceSource`. The coordinator assigns it:

| Verified origin | Committed source |
| --- | --- |
| Direct current-turn requirement, correction, rejection, or explicit confirmation | `USER_EXPLICIT` |
| Feedback referring to products shown before this turn | `BEHAVIORAL_FEEDBACK` |
| Attribute inferred by the system rather than asserted by the user | `SYSTEM_INFERRED` |

The user quote and resolved context handles are stored in trace. Committed
`evidence_text` uses the exact current user quote. `source_turn` is always the
current batch turn for new preferences.

### 8.3 Commitment

The coordinator maps verified language to commitment:

- verified `REQUIRED` + `USER_EXPLICIT` becomes `HARD`;
- verified `PREFERRED` becomes `SOFT`;
- `INFERRED`, `BEHAVIORAL_FEEDBACK`, and `SYSTEM_INFERRED` are always `SOFT`;
- ambiguous modality cannot create a hard filter; and
- profile data cannot create a preference in P0.

The model's confidence or modality alone never grants hard-filter authority.

`interpretation_confidence` uses the frozen `qu_confidence_v0` mapping:

| Accepted interpretation path | Value |
| --- | ---: |
| Self-contained direct current-turn user assertion | `1.0` |
| Explicit yes/no or reference-based assertion with a fully verified context chain | `0.9` |
| Preference derived from verified behavioral feedback | `0.75` |
| System inference | `0.5` |

These values are audit labels, not calibrated probabilities, and never
override evidence, source, grounding, or Gateway validation.

Before constructing any replacement candidate, the planner checks whether the
canonical meaning is already active. A logical reassertion reuses the complete
existing `Preference` object and ID, including its original source, evidence,
source turn, commitment, and confidence. It does not rebuild that object using
the current policy. `qu_confidence_v0` is fixed for the lifetime of QU v0.

### 8.4 Semantic text

For structured `GROUNDED` output, the grounder controls canonical predicates.
A semantic representation is attached only when it describes the same atomic
preference.

For `SEMANTIC_ONLY` output:

- the need is retained rather than dropped;
- the grounding result may retain a recognized `facet_id` for diagnostics,
  but the committed semantic-only `Preference` has
  `facet=None, operator=None, value=None`;
- a recognized but non-committable facet retains the grounder's reason code
  in trace rather than pretending the committed Preference is structured;
- semantic polarity is preserved;
- committed `semantic_text` is one self-contained, atomic product meaning
  that can be understood without replaying the conversation; and
- the coordinator, not DeepSeek, produces that text with a versioned renderer
  from the verified atomic claim.

An exact user quote is used as `semantic_text` only when it is already
self-contained, for example “do not show black dresses”. A contextual answer
such as “yes”, “no”, or “the second one” is never persisted as the semantic
meaning by itself. If a verified previous question supplies an exact atomic
claim, the coordinator may render a self-contained meaning from that validated
chain. If it cannot do so safely, it does not commit the semantic-only
preference and requests clarification or records a pre-acceptance failure.

One composite range phrase is not copied onto two unrelated atomic
preferences.

### 8.5 Grounding

Every structured candidate calls the existing release-bound
`RuntimeValueGrounder`. Query Understanding does not reproduce its lexicon,
category registry, capability, or normalizer logic.

Grounding outcomes are handled exactly:

- `GROUNDED`: eligible for trusted structured Preference construction;
- `SEMANTIC_ONLY`: preserved as semantic meaning, never promoted by guessing;
- `AMBIGUOUS`: no batch; produce clarification.

Price language is converted into safe integer `USD_CENT` operands before
grounding. Currency conversion with a live exchange rate is outside P0; an
unsupported currency remains semantic-only or requires clarification.

Category language passes through a trusted `CategoryCandidateResolver` before
the reserved-category grounder:

```text
model category surface
    -> reviewed aliases and published scope labels
    -> zero, one, or multiple published CategoryScope IDs
    -> semantic-only, grounded, or clarification
```

The resolver is release-bound and deterministic. It may produce a bounded
shortlist from published scope labels and reviewed aliases, but it cannot
create a scope or hash an arbitrary label. The reserved grounder still makes
the final canonical decision.

The model proposal contains human category surfaces, never an opaque
`CategoryScope.id`.

Category is resolved first because its final proposed scope determines which
ordinary facets are committable. The coordinator then grounds all remaining
ordinary candidates against that same scope.

P0 treats the reserved category as an effective grounding and retrieval scope,
not an ordinary soft preference:

- only an explicit, unhedged task category supported as `REQUIRED` may create
  the reserved `system_product_category` Preference;
- that reserved Preference is `USER_EXPLICIT + HARD`;
- hedged language such as “maybe jewelry” remains a soft semantic-only need
  and does not change the effective scope; and
- price is currently the only ordinary runtime facet. Reserved category does
  not pass through ordinary capability or facet-entropy logic.

Within the same active goal, “all categories”, “broaden the category”, or
“category does not matter” maps to one reserved `ReplaceFacet` whose value is
the published `root_scope_id`. It never maps to `ClearFacet` or
`SetDontCare`. During a true `SwitchGoal`, omitting the existing category from
carry refs naturally returns the effective scope to root without a separate
category operation.

The final scope is computed before grounding other facets:

```text
scope = current reserved category, or root when absent

if SwitchGoal is proposed:
    scope = current category only when its active local ref is explicitly
            present in validated carry_refs
    otherwise scope = root

if a new reserved-category replacement is proposed:
    scope = that one newly grounded published CategoryScope

ground every ordinary candidate against scope
```

Category ambiguity blocks the full turn update because a dependent ordinary
candidate cannot be grounded against an unknown final scope.

## 9. Operation planning

DeepSeek draft actions are not one-to-one aliases for committed operations.
The trusted planner emits only:

```text
AddPreference
ReplaceFacet
RemovePreference
ClearFacet
SetDontCare
SwitchGoal
```

The mapping rules are:

| Draft meaning | Committed plan |
| --- | --- |
| New non-conflicting atomic condition | `AddPreference` |
| Correction or complete registered-facet replacement | `ReplaceFacet` |
| Replace one or more semantic-only meanings | ordered `RemovePreference` followed by `AddPreference` in the same batch |
| Additive categorical alternative that must become one canonical selector | canonicalized `ReplaceFacet` with one `IN` preference |
| Retract specific active meaning | `RemovePreference` using mapped real IDs |
| Return structured facet to unknown | `ClearFacet` |
| Explicitly say a registered facet does not matter | `SetDontCare` |
| Explicitly drop an unsupported semantic facet | `RemovePreference` for all matching active local refs; otherwise no-op |
| Change product task | first `SwitchGoal` with validated carry IDs |

Planning is relative to a deterministic **post-switch baseline**:

```text
no SwitchGoal:
    baseline = current IntentState

SwitchGoal:
    baseline goal = new goal
    baseline preferences = exactly the validated carried Preference objects
    baseline dont-care = empty
```

Consequences:

- Removing an old target that was not carried is already satisfied by the
  reset and emits no `RemovePreference`.
- Replacing a semantic-only target that was not carried emits only the new
  semantic `AddPreference`.
- A replacement on an ordinary facet absent from the post-switch baseline
  emits `AddPreference`; if carried state exists, it emits complete
  `ReplaceFacet`.
- A new or broadened reserved category always uses its special
  `ReplaceFacet` even when the post-switch baseline is root.
- Remove or semantic replacement after `SwitchGoal` may target only a carried
  preference.
- All ordinary grounding, category validation, and later operation choices
  use this same post-switch baseline and the final-scope formula in Section
  8.5.

Planner invariants:

1. `SwitchGoal` is first and appears at most once.
   It is also the only operation that initializes a previously null goal.
2. Operation order remains semantic and is not repaired by the reducer.
3. Local refs are resolved against the captured request only.
4. Existing IDs are reused for logical reassertions.
5. New IDs use the frozen `p_{turn}_{operation}_{preference}` form and are
   assigned only after grounding.
6. `ReplaceFacet` is complete, never a partial patch.
7. Semantic-only preferences are removed by ID, not cleared by facet.
   A semantic-only replacement removes old IDs before adding the new
   Preference, and each intermediate state remains valid.
8. Category writes use exactly one reserved `ReplaceFacet` immediately after
   an optional `SwitchGoal`.
9. A category change must remove, replace, or semantically preserve every old
   preference that is invalid in the final category, within the same batch.
10. The batch's `base_intent_version` equals the captured request version.
11. Every intermediate reducer state is valid.
12. Preview or reduction failure rejects the whole proposal.
13. `CatalogSemanticGateway.preview` has planning authority only. The final
    catalog-bound transaction reruns Gateway checks under the session lock
    before one atomic commit.

The planner never asks the Gateway to auto-repair a batch.

## 10. Query compilation boundary — QU4 deferred

The semantic boundary below is proposed, but the exact wire types for
`CompiledConstraint`, `CompiledPreference`, `CompiledDirective`, and
`retrieval_evidence_index_id` are intentionally not frozen in this document.
They require a joint QU4 review with the Retrieval Working Contract.

Sections 1-9 define the QU request, proposal, trust, grounding, and planning
core independently of that later wire decision. The overall Query
Understanding v0 definition of done still requires QU4.

### PROPOSED: compile from accepted state, not from a model-written full query

After an accepted intent result, a pure Query Compiler will produce at least:

```python
@dataclass(frozen=True)
class CompiledQuery:
    schema: Literal["shopping-copilot/compiled-query/v0"]
    request_id: str
    turn: int
    intent_version: int
    catalog_semantic_release_id: str
    retrieval_evidence_index_id: str
    compiler_version: str
    competition_binder_version: str
    q_lex: tuple[str, ...]
    q_sem: str
    hard_constraints: tuple[CompiledConstraint, ...]
    soft_preferences: tuple[CompiledPreference, ...]
    behavioral_directives: tuple[CompiledDirective, ...]
```

This section proposes the field meanings and mandatory version pins; it does
not yet freeze the three nested retrieval DTOs or the evidence-index identity
format.

Compilation rules:

- `q_sem` describes the product need represented by the accepted goal and
  active preferences.
- `q_lex` contains deterministic exact product/category/value language.
- Structured hard preferences compile through the verified semantic release.
- Explicit hard competition facets may compile through the separate
  Retrieval Evidence Index, following its contract.
- Unsupported long-lived needs remain semantic-only in Session Context.
- Soft and inferred meaning affects ranking but not eligibility.
- Negative semantic meaning remains explicit; it is not converted into a
  positive embedding claim.
- Direct behavioral instructions are current-turn fields, not product
  semantics.
- Words such as “buying”, “browsing”, “just looking”, and “for inspiration”
  do not create a hidden binary intent class. Product constraints expressed in
  the same utterance still compile normally.
- Explicit diversity instructions override the default implied by `C_t` but
  do not change `C_t`.

The compiler MUST NOT read the raw transcript, call DeepSeek again, invent
missing state, or use the previous `C_t`. Given the same accepted intent,
release, and current-turn directives, it produces the same compiled query.

`request_id` and `turn` prove which one-turn directives belong to this
compilation. The release, evidence-index, compiler, and binder IDs make
competition constraints reproducible and prevent a mask built for one index
from being reused with another. Every downstream mask additionally follows
the Retrieval Contract's `parent_asin` and row-order binding.

### 10.1 Rebinding semantic-only competition preferences

Current Session Context v1 deliberately cannot persist a structured color,
material, style, or other facet that the verified Catalog Semantic release
does not support. Nevertheless, an explicit long-lived preference such as
“do not show black products” must not disappear from retrieval on the next
turn.

P0 therefore uses a pure, versioned `CompetitionEvidenceBinder`:

```text
all active semantic-only Preferences
    + their exact evidence and coordinator-rendered semantic text
    + pinned Retrieval Evidence Index vocabulary
    -> zero or one unambiguous ephemeral competition binding per atomic need
```

Rules:

- The binder runs over **all active preferences on every compilation**, not
  only the latest DeepSeek proposal.
- It does not mutate Session Context or promote a facet to catalog truth.
- It does not call an LLM.
- The coordinator renders a recognized explicit competition need into the
  versioned, deterministic, self-contained semantic form required by Section
  8.4 while preserving the exact user quote as evidence.
- Only `USER_EXPLICIT + HARD` meaning with an unambiguous deterministic
  facet/value binding can create an ephemeral competition hard constraint.
- `SOFT` and inferred meaning remain ranking evidence.
- A binding that cannot be reproduced becomes semantic ranking evidence, not
  a guessed hard mask.
- An implementation MAY cache the pure result by preference ID and binder
  version, but the cache is disposable and rebuildable.
- Removing the semantic-only preference by its stable ID removes its
  cross-turn competition binding.

This is how “不要黑色” remains effective across turns without adding color as
fake structured catalog truth. The exact `CompiledConstraint` representation
and evidence-index matcher remain jointly owned with the Retrieval Contract.

### 10.2 Hard-mask authority

The compiler follows the already-decided retrieval policy:

- `system_product_category` and price use the verified Catalog Semantic
  release;
- competition text facets use only explicit user meaning plus deterministic
  Retrieval Evidence Index matches;
- profile priors, model inference, dense similarity, and speculative
  attributes cannot create a hard mask;
- price `UNKNOWN` remains eligible;
- competition text `INCLUDE/EXCLUDE` uses the frozen closed-world policy;
- `EXCLUDE` is never silently relaxed; and
- an emptying `INCLUDE` is downgraded only through the retrieval contract's
  explicit relaxation rule.

## 11. Failure semantics

Stable QU error families are proposed:

### Provider and transport

- `QU_PROVIDER_TIMEOUT`
- `QU_PROVIDER_RATE_LIMITED`
- `QU_PROVIDER_UNAVAILABLE`
- `QU_PROVIDER_AUTH_FAILED`
- `QU_PROVIDER_QUOTA_EXHAUSTED`
- `QU_PROVIDER_INCOMPLETE`
- `QU_PROVIDER_CONTENT_FILTERED`
- `QU_PROVIDER_SCHEMA_REJECTED`

### Wire and evidence

- `QU_INVALID_JSON`
- `QU_SCHEMA_MISMATCH`
- `QU_UNKNOWN_FIELD`
- `QU_STALE_RESPONSE`
- `QU_INVALID_DISPOSITION`
- `QU_INVALID_ACTION_SHAPE`
- `QU_INVALID_EVIDENCE`
- `QU_UNSUPPORTED_EVIDENCE_CLAIM`
- `QU_UNKNOWN_LOCAL_REFERENCE`

### Interpretation and planning

- `QU_NEEDS_CLARIFICATION`
- `QU_GROUNDING_AMBIGUOUS`
- `QU_NORMALIZATION_FAILED`
- `QU_UNSAFE_SEMANTIC_RENDERING`
- `QU_PLANNING_FAILED`
- `QU_GATEWAY_REJECTED`

Error prose is not stable. Trace records a stable code, safe field path,
attempt number, and non-sensitive provider metadata.

Retry policy:

- timeout, 429, 500, and 503 receive bounded exponential backoff with jitter;
- empty, malformed, or locally schema-invalid generated output receives at
  most one repair retry;
- an incomplete response caused by `max_output_tokens` may retry once with a
  bounded larger output limit;
- an incomplete response caused by content filtering fails safely and is not
  retried with the same content;
- provider `failed` responses are mapped by their concrete error;
- provider 400/422 request or Schema rejection is an implementation error and
  is not blindly retried;
- authentication and quota failures fail fast; and
- a retry never changes request identity or commits twice.

P0 permits at most two provider attempts for one turn. If both fail, the
intent remains unchanged, `accepted_update=None`, and the application returns
a safe fallback after appending the normal unchanged-version `TurnRecord`.

A pre-acceptance QU failure does not clear a still-valid previous
`SearchBelief`. Once a batch is accepted, the existing session transaction
failure matrix applies.

## 12. Trace and demonstration contract

Trace lives outside active `SessionContext` and may record:

- request ID, turn, and base intent version;
- prompt, schema, model, adapter, and confidence-policy versions;
- returned provider response ID and model string;
- latency and token/cache usage;
- canonical request and validated-response hashes;
- proposal disposition;
- evidence-validation results;
- grounding dispositions and reason codes;
- planned operation types without secrets;
- Gateway preview result;
- final accepted/rejected status; and
- stable failure code.

Raw prompts or outputs containing user text require the same handling as other
session traces and are disabled by default outside local development.
`DEEPSEEK_API_KEY`, authorization headers, and provider secrets are never
logged.

The demo should render one intelligible chain:

```text
user utterance
    -> DeepSeek proposal
    -> accepted / rejected evidence
    -> canonical SessionContext change
    -> q_sem / q_lex / constraints
    -> fixed-Probe candidate map
    -> C_t
    -> resolved retrieval behavior
```

This trace is part of the project's main story: transparency comes from the
observed concentration of the accepted product need, not from a model claiming
that the user is “buying”.

## 13. Test strategy

### 13.1 Pure contract tests

Tests MUST cover:

- strict request and response field sets;
- every enum and discriminated action shape;
- exact evidence matching;
- invalid and cross-request local refs;
- stale turn and intent-version echoes;
- canonical request ordering and encoding; and
- failure codes for all validation stages.

### 13.2 Golden turn cases

The initial reviewed set SHOULD contain at least 60 cases, including:

1. category-only and category-plus-price requests;
2. positive and negative preferences;
3. hard versus soft language;
4. “not black” versus “black is no longer required”;
5. “actually blue” versus “blue is also fine”;
6. clear versus explicit don't-care;
7. lower, upper, equality, and range price language;
8. currency and unit edge cases;
9. goal switch with no carry;
10. goal switch with explicit “same budget” carry;
11. confirmation and rejection of the prior question;
12. confirmation that must become self-contained semantic meaning;
13. references to shown products;
14. product feedback without attribute over-inference;
15. an exact quote paired with a hallucinated unsupported HARD value;
16. explicit diversity instructions;
17. semantic-only color/material/style needs across multiple later turns;
18. unsupported-facet don't-care with and without active refs;
19. explicit versus hedged reserved category language;
20. unknown facet and unknown value;
21. ambiguous category, value, action, and reference;
22. prompt-injection text inside the user utterance;
23. multilingual and code-switched shopping language;
24. repeated request, stale response, and provider failure;
25. compiler/retrieval-index version mismatch; and
26. accepted intent change followed by downstream retrieval failure.

Rare correction, retraction, clear, don't-care, and goal-switch cases are
deliberately over-sampled. Average extraction accuracy must not hide unsafe
state mutations.

### 13.3 Primary metrics

The primary correctness measures are:

- JSON/schema validity rate;
- exact evidence-valid rate;
- atomic evidence-claim support rate;
- draft operation macro-F1;
- normalized argument accuracy;
- hallucinated facet/value rate;
- hard-preference false-positive rate;
- self-contained semantic rendering validity;
- state exact match after reducer;
- multi-turn replay exact match;
- ambiguity recall and unsafe-commit rate;
- provider availability, latency, token use, and cost; and
- semantic-query invariance under politeness and buying/browsing wording.

For safety-critical golden cases, an incorrect HARD condition or unsafe commit
is a test failure even if aggregate F1 remains high.

### 13.4 Test layers

```text
unit
    wire codec, evidence, reference mapping, normalizers, planner, compiler

integration
    mocked DeepSeek -> grounder -> Gateway -> reducer -> transaction

golden replay
    multi-turn conversations with exact final SessionContext

live smoke
    small optional DeepSeek suite; never required for ordinary offline tests
```

Live model tests are marked, rate-limited, and skipped without
`DEEPSEEK_API_KEY`. CI correctness does not depend on provider availability.

## 14. Proposed repository layout

```text
docs/design/query_understanding/
    README.md
    contract-v0.md

src/shopping_copilot/query_understanding/
    __init__.py
    models.py          provider-independent request/proposal DTOs
    codec.py           strict wire parsing and canonical encoding
    evidence.py        quote, local-reference, and atomic-claim validation
    prompt.py          versioned prompt construction
    deepseek.py        provider adapter only
    category.py        trusted category surface -> published scope resolver
    goal.py            trusted self-contained product-goal normalization
    clarification.py   ambiguity -> legal question field triple
    coordinator.py     trusted provenance and grounding orchestration
    planner.py         proposal -> StateUpdateBatch
    compiler.py        accepted IntentState -> CompiledQuery
    competition.py     pure semantic preference -> retrieval evidence binding
    errors.py          stable QU error codes

tests/unit/query_understanding/
tests/integration/test_query_understanding_turn.py
tests/golden/query_understanding/
```

`query_understanding` is an application-layer package. The DeepSeek client
MUST NOT be added to `session_context` or `catalog.semantic.runtime`.

## 15. Implementation sequence

### QU0 — Contract review

- accept or revise this contract;
- freeze request/proposal schemas;
- approve the initial modality and ambiguity policy; and
- approve the first golden conversations.

### QU1 — Pure boundary

- implement immutable DTOs and codecs;
- implement evidence, local-reference, and atomic-claim checks;
- implement the published-scope category resolver;
- implement goal normalization and clarification adaptation;
- implement prompt views and fixtures; and
- keep the provider fully mocked.

### QU2 — DeepSeek adapter

- implement the Responses API JSON Schema call;
- verify the exact final proposal Schema against the live Responses endpoint;
- add timeout, retry, redaction, and trace metadata;
- add opt-in live smoke tests; and
- record actual latency/cost on the golden subset.

### QU3 — Trusted coordinator and planner

- call `RuntimeValueGrounder`;
- map provenance and commitment;
- allocate/reuse IDs;
- construct canonical batches;
- preview through `CatalogSemanticGateway`; and
- add multi-turn reducer replay tests.

### QU4 — Query Compiler

- freeze the exact `CompiledQuery` wire DTO with retrieval;
- implement repeatable binding for all active semantic-only competition needs;
- produce `q_lex`, `q_sem`, constraints, preferences, and directives;
- verify buying/browsing wording invariance; and
- hand the accepted compiled view to the fixed Probe.

### QU5 — Demonstration trace

- render the understanding-to-transparency chain;
- expose accepted versus rejected interpretations;
- show state version and compiled-query changes; and
- keep provider diagnostics out of active intent.

## 16. Definition of done

Query Understanding v0 is complete when:

1. DeepSeek has no direct state, grounding, ID, Gateway, Probe, or retrieval
   authority.
2. Every accepted meaning has verified current-turn evidence.
3. Every structured value passes the pinned release-bound grounder.
4. Every state change uses the existing closed operation vocabulary and
   Gateway.
5. Ambiguity, provider failure, and invalid output cannot partially mutate
   intent.
6. Unsupported but clear user needs survive as semantic-only preferences.
7. The compiler consumes accepted state rather than reinterpreting the full
   transcript.
8. `C_t` is calculated only after compilation and the fixed Probe.
9. Golden multi-turn replay reproduces exact final state.
10. The demonstration can explain every step from user words to `C_t`.

## 17. Research basis

- [ShopTalk: A System for Conversational Faceted Search](https://research.google/pubs/shoptalk-a-system-for-conversational-faceted-search/):
  deployed shopping architecture separating contextual understanding, minimal
  intent operators, and a rules-based state tracker.
- [Description-Driven Task-Oriented Dialog Modeling](https://arxiv.org/abs/2201.08904):
  natural-language schema descriptions improve state tracking and transfer.
- [TripPy](https://aclanthology.org/2020.sigdial-1.4/):
  values and references may come from the current user span, system memory, or
  existing state.
- [Diable](https://aclanthology.org/2023.findings-acl.615/):
  state deltas can be interpreted and accumulated separately from language
  generation.
- [DeepSeek Responses API](https://api-docs.deepseek.com/api/create-response/):
  current stateless Responses interface and JSON Schema output.
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/):
  explicit non-thinking configuration.
- [DeepSeek Context Caching](https://api-docs.deepseek.com/guides/kv_cache/):
  prefix-based, best-effort caching that must not be treated as memory.

## 18. Open review decisions

Before changing the status to frozen, review must explicitly decide:

1. whether to accept the proposed no-safe-subset policy for a materially
   ambiguous turn;
2. whether to accept `qu_confidence_v0` and the P0 context window limited to
   canonical intent, the previous assistant turn, and the last shown batch;
3. whether to accept the policy that only explicit required category language
   changes the reserved effective scope;
4. whether to accept the proposed versioned `CompetitionEvidenceBinder` as
   the cross-turn bridge for semantic-only explicit needs; and
5. the exact retrieval-side `CompiledConstraint` DTO and evidence-index
   identity format; and
6. whether `deepseek-v4-pro` is needed as an explicit fallback after the Flash
   golden evaluation.

None of these decisions may be made implicitly inside the DeepSeek prompt.
