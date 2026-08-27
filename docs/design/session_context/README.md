# Session Context

- Contract: **v1 frozen for implementation**
- Implementation: **M1 values and local validation complete; M2 pending**
- Last design review: **2026-08-27**

Session context is the shared state boundary between query understanding,
probe retrieval, ranking, asking, and the official Agent adapter. It separates:

```text
SessionContext
├── ProfilePrior          immutable input prior
└── SessionState
    ├── IntentState       current user-need facts
    ├── InteractionContext append-only interaction history
    └── SearchBelief      catalog-derived observation
```

## Documents

- [`contract-v1.md`](contract-v1.md): normative types, operations, invariants,
  transaction semantics, and error codes. Implementation must conform to this
  document.
- [`implementation-plan.md`](implementation-plan.md): package layout,
  dependency direction, test plan, integration sequence, and repository
  hygiene.
- [`design-rationale.md`](design-rationale.md): research draft and
  architectural reasoning. Retained for context; superseded as an
  implementation contract.

## Frozen boundaries

- Intent state records what the current user need means; it does not store
  retrieval weights or policy decisions.
- Interaction context records what happened; mutable convenience views are
  derived from its turn records.
- Search belief records what a valid probe observed in the catalog; it does
  not store question utility or orchestration decisions.
- Intent changes only through a typed, version-checked, atomic update batch.
- The reducer is deterministic and cannot access a catalog, retriever, LLM,
  ranker, or official evaluator.
- The official `starter.Agent` will remain a thin compatibility adapter.

## Not frozen here

- Query-understanding model or prompting
- Certainty algorithm or calibration
- Candidate-mode discovery algorithm
- Retrieval routes and weights
- Ranking model
- Question-utility policy
- Profile-prior weighting

Those components consume this contract but cannot redefine it implicitly.

## Implemented boundary

M1 provides frozen leaf and aggregate values, the closed operation vocabulary,
stable domain errors, an injected immutable facet registry, and explicit local
validators. The registry deliberately contains no default production facet
list and does not derive facets from the official `ask_attribute` protocol.

Validation that needs a current state, prior history, catalog observation, or
transaction lifecycle remains outside this milestone. In particular, reducer
semantics and atomic versioned updates begin in M2; belief, history, codec, and
store validation begin in M3.
