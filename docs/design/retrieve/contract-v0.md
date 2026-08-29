# Retrieval and Ranking Working Contract v0

> 历史设计记录：本文保留早期讨论，不再代表当前 runtime 执行协议。当前指标名为 `T_t`，正式三路
> 召回、RRF 和向量 MMR 以
> [`formal-multi-route-v0.md`](formal-multi-route-v0.md) 为准。

- Status: **working draft; only decisions marked `DECIDED` are agreed**
- Date: **2026-08-28**
- Compatibility targets: **Session Context Contract v1** and
  **Catalog Semantic Layer Contract v0**
- Primary objective: **a distinctive, effective three-day hackathon system**

This file is the project-owned design record for Retrieval and Ranking. It is
written from decisions made in repository review discussions.

The following files are research and discussion inputs, not implementation
contracts:

- [`retrive_design.md`](retrive_design.md)
- [`retrieve.md`](retrieve.md)

An idea appearing in either input document does not become part of this
contract until it is recorded here as `DECIDED`. Open questions are listed at
the end of this file so that implementation does not silently make product or
architecture decisions.

## 1. Design objective

### DECIDED: optimize for the competition

This is not a long-lived production commerce platform. Its first objective is
to express a strong system philosophy and perform well in the official
environment under hackathon time constraints.

The design therefore MAY use deterministic, competition-oriented catalog
evidence that would require broader governance in a production system. It MUST
still preserve the official catalog as read-only data and MUST keep derived
indexes separate from the source catalog.

Engineering safeguards are retained where they prevent obvious target loss,
nondeterminism, or an unrecoverable retrieval failure. Production-grade facet
governance is not a prerequisite for every competition facet.

## 2. Core system claim

### DECIDED: intent transparency replaces binary routing

The system does not try to predict whether the user will eventually purchase.
It estimates:

> **How safely can the current expressed need compress the catalog search
> space?**

The turn-level score is:

$$
C_t \in [0,1]
$$

and is called **Catalog-Grounded Intent Transparency**.

- Low $C_t$ means that the system should preserve a wider search space and
  more default diversity.
- High $C_t$ means that the system can focus on a narrower search space with
  stronger precision emphasis.
- $C_t$ is not purchase probability.
- $C_t$ is not a hidden `BUYING / BROWSING` classifier.

For example, “I am browsing black dresses for inspiration” can still have high
$C_t$: whether the user purchases is not a retrieval decision, while the
expressed product space is already narrow.

The official Buying/Browsing routes are treated as common behaviors near the
two ends of a continuous control policy, not as the system's ontology of user
intent.

### DECIDED: explicit instructions override defaults

$C_t$ controls default system behavior. A direct instruction such as “show me
very different styles” overrides the default diversity implied by $C_t$. It
does not require changing $C_t$ or introducing a second binary intent class.

## 3. One control signal across the retrieval stack

### DECIDED: $C_t$ controls four layers

The single transparency score MAY continuously control:

1. per-route fetch depth;
2. route-fusion emphasis;
3. final-ranking emphasis; and
4. default diversity strength.

This shared control is part of the project's core story, not an accidental
coupling to remove before the hackathon.

The exact functions and endpoint values are not yet frozen, but every enabled
mapping MUST satisfy these invariants:

- the fixed Probe itself is independent of $C_t$;
- a mapping is bounded;
- its intended direction is monotonic in $C_t$;
- $C_t$ alone MUST NOT abruptly disable an otherwise available core route;
- every turn records the resolved route depths, fusion weights, rank weights,
  and diversity strength for debugging and demonstration; and
- if $C_t$ is unavailable, the controller uses $C_t=0.5$ and records the
  fallback.

### DECIDED: diagnostics are a safety channel, not a second philosophy

The fixed Probe also emits ephemeral retrieval diagnostics $D_t$. $D_t$
distinguishes an unhealthy retrieval environment from a genuinely broad user
need. It MAY trigger a bounded fallback for conditions such as an unavailable
route, an empty eligible catalog, or a failed score computation.

$D_t$ is not a second exploration-to-precision axis. Route weights and policy
decisions derived from $D_t$ MUST NOT be stored as user intent in
`SessionContext` or `SearchBelief`.

The exact diagnostic fields and fallback table remain open.

## 4. Query compilation boundary

### DECIDED: Retrieval consumes compiled views

Retrieval does not reinterpret the complete conversation. Query Understanding
compiles the current catalog-bound session state into at least:

- `q_lex`: exact and lexical product language;
- `q_sem`: natural-language semantic intent;
- explicit hard constraints;
- soft preferences; and
- direct behavioral instructions such as an explicit diversity request.

The exact API schema is frozen in
[`../query_compiler/contract-v0.md`](../query_compiler/contract-v0.md). Semantically,
an explicit textual or categorical hard constraint is one of:

```text
INCLUDE(facet, value)
EXCLUDE(facet, value)
```

Numeric constraints such as price retain their typed comparison operator.

Only an explicit current-turn/session requirement can create a competition
hard mask. Profile priors, LLM-inferred attributes, dense similarity, and
speculative IntentCard fields remain ranking evidence.

## 5. Competition facet policy

### DECIDED: use a wide hard-facet vocabulary

The competition retrieval layer MAY compile explicit hard constraints for:

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

This is deliberately wider than the current ordinary Catalog Semantic runtime
facet set. It aligns with the official question attributes and with the toy
simulator, which derives user constraints from target-product metadata.

### DECIDED: canonical and competition evidence remain distinct

- `system_product_category` and `price` continue to use the verified Catalog
  Semantic release.
- Other competition facets use a separate, derived **Retrieval Evidence
  Index**.
- A competition evidence entry is retrieval-time evidence, not a promoted
  catalog truth and not a mutation of the source dataset.
- Unsupported long-lived preferences remain semantic-only in Session Context;
  Query Understanding may compile them into ephemeral retrieval constraints
  for the current turn.

The initial Retrieval Evidence Index may use normalized evidence from:

```text
title
features
details key/value pairs
description
categories
store
```

It MAY include a small versioned alias table for deterministic normalizations
such as `grey -> gray`. Dense similarity or unconstrained LLM inference MUST
NOT create hard evidence.

## 6. Hard-constraint policy

### DECIDED: competition text facets use a closed-world mask

For the provisional competition facets, the searchable catalog evidence is the
system's operational world. Retrieval does not maintain a separate confirmed
pool and fallback pool.

For one normalized facet value (v), let (M(f,v)) be the set of products
whose Retrieval Evidence Index matches it. Hard masks behave as follows:

```text
INCLUDE(f, v): eligible <- eligible INTERSECT M(f, v)
EXCLUDE(f, v): eligible <- eligible MINUS     M(f, v)
```

Consequently, for “do not show black products”:

- a product with matching black evidence is excluded;
- a product without matching black evidence remains eligible; and
- no separate UNKNOWN ranking penalty is introduced.

This is an intentional competition-oriented closed-world assumption.

### DECIDED: relaxation is asymmetric

- An `EXCLUDE` constraint is never silently relaxed, even when it leaves a
  very small or empty catalog.
- An `INCLUDE` constraint that would make the eligible catalog empty is
  downgraded to a strong soft preference and records
  `hard_filter_relaxed=true`.
- The system has no confirmed/fallback candidate pools.

Negative constraints run first in stable compiler order. Positive constraints
then run in stable compiler order; the first and every later `INCLUDE` that
would empty a non-empty pool is relaxed independently. If negatives already
emptied the pool, later positives are marked as skipped, not relaxed. This
execution rule is frozen in
[`evidence-hard-mask-v0.md`](evidence-hard-mask-v0.md).

### DECIDED: price is the conservative exception

Price continues to use the verified four-state Catalog Semantic matcher:

- a price proven to violate the numeric constraint is excluded;
- a price proven to satisfy it remains eligible; and
- missing, conflicting, or otherwise unknown price evidence remains eligible.

Price UNKNOWN receives no universal penalty. This exception is required
because price is absent or unusable for most catalog products.

### DECIDED: semantic qualities are not hard masks

Requirements such as “does not look cheap”, “comfortable for long walks”, or
an LLM-inferred `sporty=true` do not produce catalog exclusions unless the
compiler can bind them to explicit deterministic catalog evidence. Otherwise
they are semantic ranking features or penalties.

## 7. Mandatory execution order

### DECIDED: filter before route truncation

The resolved hard mask MUST be applied before any adaptive route takes its
top-$K$ results. A later route, fusion stage, dense score, or ranker cannot
resurrect an excluded product.

The agreed high-level flow is:

```text
SessionContext
    -> Query Understanding / Query Compiler
    -> q_lex + q_sem + explicit constraints + soft preferences
    -> build competition hard eligible_mask
       -> downgrade only an emptying INCLUDE according to the relaxation rule
       -> never silently relax EXCLUDE
    -> Fixed Probe over the eligible catalog
    -> C_t + ephemeral D_t
    -> C_t-controlled multi-route retrieval over the same eligible catalog
    -> candidate fusion and truncation
    -> ranking
    -> C_t-controlled default diversity
    -> valid, unique parent_asin Top-10
```

Required invariants:

- every active retrieval route sees the same resolved `eligible_mask`;
- the runtime mask is constructed from `parent_asin` values and bound to the
  active Catalog Semantic release and retrieval-index row order; cross-module
  callers never exchange an unbound 50k boolean array;
- hard filtering occurs before route top-$K$ and before union truncation;
- route fusion cannot override a hard exclusion;
- explicit diversity instructions override only the default diversity policy,
  not hard constraints;
- all derived indexes remain separate from the read-only catalog; and
- the final external identifier is `parent_asin`.

## 8. Current decision ledger

| ID | Decision | Status |
| --- | --- | --- |
| R-D01 | Replace binary Buying/Browsing routing with continuous intent transparency | DECIDED |
| R-D02 | Define $C_t$ as safe catalog-space compression, not purchase probability | DECIDED |
| R-D03 | Let $C_t$ coordinate route depth, fusion, rank emphasis, and default diversity | DECIDED |
| R-D04 | Keep Probe fixed and use $C_t=0.5$ when transparency is unavailable | DECIDED |
| R-D05 | Treat $D_t$ as ephemeral safety diagnostics, not a second intent axis | DECIDED |
| R-D06 | Support wide competition hard facets through a derived evidence index | DECIDED |
| R-D07 | Compile explicit hard text constraints into closed-world INCLUDE/EXCLUDE masks | DECIDED |
| R-D08 | Never auto-relax EXCLUDE; relax an emptying INCLUDE to a strong soft preference | DECIDED |
| R-D09 | Keep price's conservative matcher as an exception | DECIDED |
| R-D10 | Apply the hard mask before every route's top-$K$ and before union truncation | DECIDED |
| R-D11 | Do not introduce confirmed/fallback candidate pools | DECIDED |
| R-D12 | Let direct diversity instructions override $C_t$'s default diversity | DECIDED |
| R-D13 | Bind runtime masks to `parent_asin`, active release, and index row order | DECIDED |
| R-D14 | Freeze deterministic CompiledQuery v0 and the fixed Dense Probe entry | DECIDED |
| R-D15 | Fix the first multi-view Probe at Top-80 with lexical and dense views | DECIDED |
| R-D16 | Collapse dense results into fixed-leader semantic modes at cosine 0.94 | DECIDED |
| R-D17 | Derive C_t only from calibrated equal-mode coherence; keep counts out | DECIDED |
| R-D18 | Keep lexical evidence, route overlap, counts and duplicate warnings in D_t | DECIDED |
| R-D19 | Store measured certainty as None when evidence/calibration is unavailable | DECIDED |
| R-D20 | Build a deterministic, read-only Retrieval Evidence Index for wide competition facets | DECIDED |
| R-D21 | Resolve exclusions first, then relax emptying includes in stable compiler order | DECIDED |

## 9. Open design decisions

The following items are intentionally not frozen:

1. **Transparency calibration**
   - final target-free calibration anchors for the pinned dense index;
   - held-out audit results and future recalibration triggers;
   - optional catalog-support veto thresholds.

2. **Production route implementations**
   - true BM25F versus a weighted field-BM25 ensemble;
   - raw dense document construction and model;
   - whether IntentCard and Facet routes enter P0;
   - reuse of route scoring between Probe and adaptive retrieval beyond the
     already shared dense score snapshot.

3. **Control functions**
   - exact per-route depth endpoints;
   - fusion weights;
   - rank-weight interpolation;
   - diversity function and bounds;
   - exact $D_t$ safety fallback table.

4. **Multi-turn behavior and asking**
   - treatment of already shown products and feedback;
   - reset behavior after Intent Override;
   - SearchBelief fields needed by proactive clarification;
   - behavior when non-relaxable exclusions leave no products.

5. **Interfaces and reproducibility**
   - candidate/evidence DTO;
   - release, intent, probe, model, index, and policy version pins;
   - deterministic tie-breaking and unavailable-route representation.

6. **Ranking scope**
   - pre-rank formula;
   - cross-encoder inclusion;
   - analytic final ranker;
   - MMR behavior;
   - whether LambdaMART is attempted at all.

Until an open item is moved into the decision ledger, its treatment in either
research input document remains non-normative.
