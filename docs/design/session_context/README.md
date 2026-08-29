# Session Context

- Contract: **v1 frozen for implementation**
- Implementation: **M3 core, CS7 catalog-bound integration, and QU P0 integration complete**
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

M1 provides frozen values, the closed operation vocabulary, stable domain
errors, an injected immutable facet registry, and explicit local validators.
M2 adds the deterministic, ordered, atomic intent reducer. M3 completes the
session-context core with belief and replay-aware aggregate validation,
derived interaction views, a canonical versioned snapshot codec, and a
copy-on-write in-memory store with per-session transactions.

Catalog Semantic CS7 now supplies the application-layer authority deliberately
kept out of this catalog-independent contract. It binds one verified semantic
release to a private raw store, reruns catalog checks and the raw commit under
one session lock, restricts reserved category writes, gates live SearchBelief
provenance, and wraps snapshots in a release-pinned outer envelope. The v1
types, reducer, codec, store, and error meanings remain unchanged.

The catalog-independent reducer registry still contains no implicit production
facet list and does not derive facets from the official `ask_attribute`
protocol. At the application boundary, CS7 now composes the verified
price/category registry with the explicit retrieval-derived competition
vocabulary; the two paths remain distinguishable through `FacetAuthority`.
Query Understanding consumes that combined boundary under its own
[`contract-v1`](../query_understanding/contract-v1.md). Query Compiler, Probe
production, final orchestration, response normalization, and the thin official
adapter remain downstream work.
