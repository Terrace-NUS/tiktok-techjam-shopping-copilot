# Retrieval Evidence and Hard Mask v0

- Status: **implemented competition contract**
- Date: **2026-08-29**
- Inputs: `CompiledQuery v0`, Catalog Semantic release, read-only catalog
- Output: one release- and dense-index-bound eligibility mask

## 1. Purpose

The Query Compiler can say:

```text
color != black
material = leather
price <= 10000 USD cents
```

but a retrieval route cannot safely execute those statements by itself. It
needs one common answer to a simpler question: **which `parent_asin` values are
still eligible?**

This stage has two deliberately separate parts:

```text
read-only catalog
    -> RetrievalEvidenceIndex: facet/value -> matching parent_asin set

CompiledQuery.hard_constraints
    + verified category and price facts
    + RetrievalEvidenceIndex
    -> HardMaskResolver
    -> DenseEligibilityMask
```

The evidence index does not read the conversation. The resolver does not run
an LLM or reinterpret natural language. Query Understanding and the Query
Compiler have already done that work.

## 2. Retrieval Evidence Index

`RetrievalEvidenceIndex` is a deterministic, derived lookup over the unchanged
catalog. It supports the competition-wide text facets:

```text
brand, color, department, feature, gender,
material, size, style, use_case
```

The index is explicitly **retrieval evidence**, not promoted catalog truth.
For example, a store name can be useful brand evidence in the competition
catalog without claiming that every marketplace store is the legal brand.

### 2.1 Source policy

| Facet | Catalog evidence used by v0 |
| --- | --- |
| `brand` | `store`, plus explicit `Brand` / `Brand Name` detail values |
| `color` | color detail values, title, and feature bullets |
| `department` | `Department` details and category paths |
| `gender` | controlled audience tokens from department and category paths |
| `material` | material/fabric detail values, title, and feature bullets |
| `size` | explicit size detail values and size-marked title phrases only |
| `style` | style/pattern/theme/closure details, title, features, and categories |
| `feature` | feature detail values, feature bullets, and description |
| `use_case` | occasion/use details, categories, feature bullets, and description |

The policy intentionally favors hackathon compatibility and coverage. Its
boundaries prevent two high-impact catalog errors discovered in the 50k audit:

- package/product dimensions never become `size` evidence; and
- a locally negated occurrence such as `not waterproof` does not become
  positive `feature=waterproof` evidence.

### 2.2 Matching

Both catalog evidence and lookup values use the same deterministic Unicode,
case, whitespace, token, and small-alias normalization. Examples include
`grey -> gray`, `colour -> color`, and normalized audience spellings.

Matching uses whole normalized tokens and contiguous multi-token phrases.
It is not arbitrary substring matching, so `men` does not match `women` and a
bare `8` in `10 x 8 x 2 inches` does not become product-size evidence.

The public operation is:

```python
matches = evidence_index.match("material", "stainless steel")
```

It returns a frozen set of matching `parent_asin` values. A supported value
with no evidence returns an empty set. An unknown facet or malformed value is
an explicit error rather than a silent empty match.

### 2.3 Identity and read-only guarantee

The index records:

- the source `catalog_id`;
- the Catalog Semantic release ID;
- its evidence policy ID;
- the complete, sorted product ID set; and
- a content-derived index ID.

It is built in memory from the catalog and never writes into the official
dataset. Runtime construction verifies the expected product IDs when supplied.

## 3. Hard Mask Resolver

`HardMaskResolver` consumes only `CompiledQuery.hard_constraints`. It begins
with every product in the active dense index, then executes constraints in a
fixed order.

### 3.1 Execution order

1. Apply every negative constraint (`NEQ`, `NOT_IN`) in original compiled
   order. These constraints are never relaxed.
2. Apply every positive constraint in original compiled order.
3. If a positive constraint would turn a non-empty pool into an empty pool,
   keep the current pool and mark that constraint `RELAXED_TO_RANKING`.
4. If a negative constraint has already emptied the pool, later positive
   constraints are recorded as `SKIPPED_EMPTY_UPSTREAM`; they did not cause
   the empty result and are not falsely called relaxed.

For a multi-value `IN` or `NOT_IN`, values are ORed within that one constraint
before the constraint is applied.

This freezes the previously open relaxation rule: stable compiler order is
the tie-breaker; there is no score-based or random choice.

### 3.2 Evidence semantics by facet

- `system_product_category` uses the verified category assignment and scope
  matcher from the Catalog Semantic release.
- `price` uses the verified sparse price index and conservative interval
  matcher. Only a product proven to violate the bound is removed; missing or
  conflicting price stays eligible.
- competition text facets use the Retrieval Evidence Index's closed-world
  match set. Positive conditions intersect it; negative conditions subtract
  it.

There is one resulting pool. v0 does not create a confirmed pool and a second
fallback pool.

### 3.3 Trace

Every execution produces an ordered `ConstraintResolutionTrace` containing:

```text
preference_id, facet, operator,
before_count, matched_count, after_count,
disposition, reason
```

The final result also exposes the eligible product IDs, bound dense mask,
whether any positive hard filter was relaxed, and the exact relaxed compiled
constraints. This is runtime evidence for debugging and the demo; it is not
written back as a user preference.

## 4. Binding and Probe integration

Construction rejects catalog, release, category graph, product-set, or dense
row binding mismatches. The resolver creates the boolean vector only through
the active `DenseIndex`, so callers never exchange an unbound 50k array.

The integrated call path is:

```text
CompiledQuery
    -> HardMaskResolver.resolve(...)
    -> one DenseEligibilityMask
    -> CompiledProbeRunner.run(
           eligible_mask=that_same_mask,
           hard_filter_relaxed=resolution.hard_filter_relaxed,
       )
```

The fixed lexical and semantic Probe views therefore observe the same eligible
catalog, and filtering happens before either Top-K truncation. The relaxation
flag enters `D_t`; it never directly changes `C_t`.

## 5. Deliberate v0 limits

- Open-text evidence can still contain catalog marketing noise. The index is
  a competition retrieval device, not a general product-ontology authority.
- There is no fuzzy spelling model or LLM-generated product evidence.
- There is no source-confidence pool or second fallback candidate set.
- Positive hard conditions can be downgraded only by the deterministic empty
  result rule; negative conditions remain authoritative.
- The same resolved mask is intended to be reused by later production recall
  routes, which are outside this module.

## 6. Real 50k verification

The checked-in audit command is:

```powershell
.\.venv-3.10\Scripts\python.exe scripts\retrieval\audit_hard_mask_v0.py
```

On the frozen official catalog and current dense index it verified:

| Check | Observed result |
| --- | ---: |
| Catalog products / bound dense rows | `50,000 / 50,000` |
| Source catalog bytes unchanged | `true` |
| `color = black` | `7,128` eligible |
| `material != leather` | `43,042` eligible |
| `size = 8` | `299` eligible |
| `feature = rfid blocking` | `120` eligible |
| `use_case = hiking` | `1,627` eligible |
| `price <= USD 100` | `49,238` eligible, including unknown prices |
| Deliberately missing positive evidence | relaxed with `50,000` retained |
| `color != black` plus `color = black` | exclusion applied first; include relaxed |

The in-memory evidence build took approximately 35 seconds on the development
machine. A traced run peaked near 489 MiB of Python allocations and settled
near 290 MiB. This is acceptable as a one-time hackathon/demo startup index,
but it should be cached or prebuilt before treating the factory as a low-latency
production cold-start path.
