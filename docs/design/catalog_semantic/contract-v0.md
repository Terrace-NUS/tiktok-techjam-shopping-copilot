# Catalog Semantic Layer Contract v0

- Status: **normative for P0 implementation**
- Compatibility target: **Session Context Contract v1**
- Evidence policy: **`structured_resolution_v1` only**

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe
implementation requirements. This contract defines the versioned semantic
adapter between the frozen 50,000-product catalog and session-context v1. It
does not change any session-context v1 dataclass, operation, reducer rule,
snapshot schema, or error code.

## 1. Scope and architecture

The layer owns the following facts:

- which catalog category concepts exist;
- which reviewed facets exist and where they are applicable;
- how raw catalog fields produce typed product values;
- whether a product value is known, unknown, conflicting, or not applicable;
- which facets can be committed, retrieved, probed, or clarified in one
  category scope;
- which runtime values can be grounded into session-context preferences; and
- which mutually compatible artifacts comprise one semantic release.

The normative build and runtime flow is:

```text
50k catalog
    -> CategoryRegistry
    -> ProductCategoryAssignmentSet
    -> raw facet profiles and FacetCandidateRecords
    -> human Gate A: extraction approval
    -> CatalogFacetSchema + FacetApplicability + FacetSourceBinding
    -> structured_resolution_v1 resolver
    -> ProductFacetIndex + resolved statistics
    -> human Gate B: runtime promotion
    -> EffectiveFacetCapability rows
    -> RuntimeValueLexicon + session FacetSpec projection
    -> CatalogSemanticRelease
    -> CatalogSemanticGateway
    -> unchanged session-context v1
```

Query Understanding, retrieval weighting, question utility, model-based value
inference, and official `ask_attribute` mapping are outside this contract.

## 2. Primitive and identifier rules

All domain objects MUST be immutable after publication. Set-like collections
MUST be serialized as duplicate-free tuples in the canonical order specified
by this contract.

`canonical_json(value)` means the RFC 8785 JSON Canonicalization Scheme (JCS)
serialization of `value`, encoded as UTF-8 without a BOM. Before JCS is
applied, an Enum is replaced by its wire `.value`, a dataclass is replaced by
an object containing exactly its declared fields, a tuple is replaced by an
array, and `None` is replaced by JSON `null`. Unknown or duplicate object
members, non-string object keys, lone Unicode surrogates, and non-finite
numbers are invalid. Integers in catalog-semantic canonical data MUST lie in
the inclusive I-JSON/JCS interoperable range
`[-9007199254740991, 9007199254740991]`; larger Python integers are rejected,
not rounded or coerced. Every contract field declared as set-like MUST already be
in its contract-defined order; canonical JSON sorts object keys but never
silently reorders an array. All hashes written as `sha256(canonical_json(...))`
are over these UTF-8 bytes.

Canonical scalar order is deterministic, not semantic: values are ordered by
type rank `bool < int < float < str`, then by their `canonical_json` UTF-8
bytes. Boolean is tested before integer. This order is used only where this
contract explicitly calls a scalar tuple canonical.

```python
ScalarValue = str | int | float | bool
```

Boolean values are not numeric. Floats MUST be finite and MUST NOT be negative
zero. Empty strings, lone Unicode surrogates, and strings containing control
characters are invalid in canonical semantic fields. The frozen raw catalog
artifact may contain other valid JSON source values; they remain opaque raw
data until a reviewed extractor either rejects or converts them, and
`raw_value_json` stores their canonical JSON text rather than treating them as
semantic ScalarValues.

Reviewed facet IDs, binding IDs, extractor IDs, normalizer IDs, resolver IDs,
and policy IDs MUST match:

```text
^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$
```

`other` is reserved by the official adapter and MUST NOT be a facet ID.
`system_product_category` is reserved by this contract and MUST NOT be created,
renamed, rejected, or reused by ordinary facet discovery.

Published IDs MUST NOT be reused for a different meaning. Display labels MAY
change without changing identity. A facet migration alias is valid only in an
explicit offline migration tool; it MUST NOT silently reinterpret a pinned
session snapshot under a different semantic release.

Content identifiers use:

```text
sha256:<64 lowercase hexadecimal digits>
```

unless an object-specific rule below defines a prefixed content ID.

## 3. Category nodes, scopes, and product assignments

### 3.1 Canonical category graph

A category path segment is canonicalized with Unicode NFKC, trimming,
whitespace collapse, and case-folding. A canonical segment MUST be non-empty
and contain no control character. This normalization is lexical only; it MUST
NOT merge two different paths merely because their labels appear semantically
similar.

```python
@dataclass(frozen=True)
class CategoryNode:
    id: str
    parent_id: str | None
    canonical_path: tuple[str, ...]
```

`CategoryNode.id` is:

```text
"cn_" + lowercase_hex(sha256(canonical_json({"canonical_path": [...] })))
```

using the full 64-character digest. A node's parent is the node for the
immediate path prefix. Root nodes have `parent_id=None`. Nodes are sorted by
ID.

The **category graph core** consists of the catalog content ID and all nodes,
but not reviewed scopes or product assignments. Its ID is exactly:

```text
category_graph_id = "sha256:" + lowercase_hex(sha256(canonical_json({
    "schema": "shopping-copilot/category-graph-core/v0",
    "catalog_id": catalog_id,
    "nodes": [
        {
            "id": node.id,
            "parent_id": node.parent_id,
            "canonical_path": list(node.canonical_path)
        }
        for node in nodes_sorted_by_id
    ]
})))
```

Category lexical normalization and its Unicode data version belong to the
closed implementation identified by `builder_version`; changing either
requires a new builder version and therefore a different release.

### 3.2 CategoryScope exact schema

A user-facing catalog category is a scope, not necessarily one raw tree node.
A scope is exactly a reviewed union of one or more complete raw subtrees.

```python
@dataclass(frozen=True)
class CategoryScope:
    id: str
    label: str
    root_node_ids: tuple[str, ...]
    member_node_ids: tuple[str, ...]

@dataclass(frozen=True)
class CategoryRegistry:
    schema: Literal["shopping-copilot/category-registry/v0"]
    catalog_id: str
    category_graph_id: str
    root_scope_id: str
    nodes: tuple[CategoryNode, ...]
    scopes: tuple[CategoryScope, ...]
```

For scope `S`, its node membership is defined exactly as:

```text
N(S) = union(descendants_or_self(root) for root in S.root_node_ids)
```

`member_node_ids` MUST equal `N(S)` exactly and be sorted and unique. A scope's
roots MUST be sorted, unique, and non-redundant: no root may be a descendant of
another root in the same scope.
CategoryRegistry nodes and scopes are each unique and sorted by ID.

The scope ID is:

```text
"cs_" + lowercase_hex(sha256(canonical_json({
    "category_graph_id": category_graph_id,
    "root_node_ids": [...]
})))
```

using the full digest. Consequently, one scope can safely represent a concept
such as `shoes` even when the concept is a union of several disjoint raw
subtrees. Query grounding MUST store the `CategoryScope.id`, never a raw label
or a `CategoryNode.id`.

Scope `B` is a taxonomy refinement of scope `A` when `B.member_node_ids` is a
strict subset of `A.member_node_ids`. Two published scopes with equal member
node sets are operationally equivalent and MUST be represented by one scope
identity, not two. Alternative surface phrases are Query Understanding
resources that reference that identity; they are not fields of CategoryScope.

`root_scope_id` MUST name the union of every root-node subtree and therefore
contain every CategoryNode. It is the effective category context when
IntentState contains no grounded category preference.

Only published scopes may enter runtime state. Query Understanding MUST NOT
create a dynamic union, intersection, difference, or arbitrary boolean
category expression. If an utterance requires an unregistered scope, it
remains semantic-only or is clarified.

`IntentState.goal` and the reserved category preference remain independent.
The goal is the user's open-ended shopping task; `system_product_category` is
only a reliably grounded catalog anchor. Neither field is derived from,
overwrites, or semantically substitutes for the other. A session may have a
goal with no category, a category with no goal, both, or neither.

### 3.3 ProductCategoryAssignment artifact

CategoryRegistry says which taxonomy concepts exist. A separate assignment
artifact says what is known about each product's raw taxonomy membership.

```python
class ProductCategoryAssignmentStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"

@dataclass(frozen=True)
class ProductCategoryAssignment:
    parent_asin: str
    status: ProductCategoryAssignmentStatus
    leaf_node_ids: tuple[str, ...]

@dataclass(frozen=True)
class ProductCategoryAssignmentSet:
    schema: Literal["shopping-copilot/product-category-assignment/v0"]
    catalog_id: str
    category_graph_id: str
    assignments: tuple[ProductCategoryAssignment, ...]
```

There is exactly one assignment per catalog `parent_asin`, sorted by that ID.
For `KNOWN`, `leaf_node_ids` is non-empty, sorted, unique, and contains only
valid CategoryNode IDs. Here “leaf” means the terminal node of a product's raw
category path; it MAY have children in the global category graph because other
products can terminate deeper in the same subtree. For `UNKNOWN`, the tuple is
empty. For `CONFLICT`, it contains at least two sorted, unique candidate
terminal-node IDs; candidate membership MUST NOT be treated as resolved
membership. `category_graph_id` MUST equal the graph ID embedded in the pinned
CategoryRegistry, so assignment identity is not coupled to later reviewed
scope-label changes.

For assignment `A` and scope `S`, category matching is exactly:

```text
if A.status == KNOWN and A.leaf_node_ids intersects S.member_node_ids:
    SATISFIED
else if A.status == KNOWN:
    VIOLATED
else:
    UNKNOWN
```

The generic schema and matcher MUST preserve all three statuses. The official
P0 50k release gate MAY be stricter and, for this known frozen catalog, MUST
require all 50,000 assignments to be `KNOWN` before runtime publication. Raw
profiling and failed candidate builds still report `UNKNOWN` and `CONFLICT`
rather than pretending they are known.

## 4. Gate A: extraction approval

Gate A answers only:

> Is this concept stable enough to run deterministic catalog extraction?

An `EXTRACTION_APPROVED` decision permits publication of the following
objects. It does not permit a structured IntentState preference.

```python
class FacetDataType(str, Enum):
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    TEXT = "text"

class ItemCardinality(str, Enum):
    SINGLE = "single"
    MULTI = "multi"

@dataclass(frozen=True)
class CatalogFacetDefinition:
    id: str
    name: str
    data_type: FacetDataType
    item_cardinality: ItemCardinality

@dataclass(frozen=True)
class CatalogFacetSchema:
    schema: Literal["shopping-copilot/catalog-facet-schema/v0"]
    facets: tuple[CatalogFacetDefinition, ...]

@dataclass(frozen=True)
class FacetApplicability:
    facet_id: str
    category_scope_ids: tuple[str, ...]

@dataclass(frozen=True)
class FacetApplicabilitySet:
    schema: Literal["shopping-copilot/facet-applicability/v0"]
    category_registry_id: str
    facet_schema_id: str
    entries: tuple[FacetApplicability, ...]
```

Facet definitions are unique and sorted by ID. BOOLEAN, NUMERIC, and TEXT
facets MUST use `SINGLE`; CATEGORICAL facets may use `SINGLE` or `MULTI`.
Runtime-v0 `price` is NUMERIC and SINGLE.

FacetApplicability is an independent reviewed artifact. For a product
assignment `A` and facet applicability entry `F`:

```text
if A.status in {UNKNOWN, CONFLICT}:
    applicability is UNKNOWN
else if any A.leaf_node_id belongs to any F.category_scope_id:
    APPLICABLE
else:
    NOT_APPLICABLE
```

`belongs` in this algorithm means that the leaf ID occurs in the referenced
scope's materialized `member_node_ids`; it is not a label comparison.

An unknown category assignment therefore yields product-facet `UNKNOWN`, never
`NOT_APPLICABLE`. FacetApplicability is the sole P0 basis for deriving
`NOT_APPLICABLE`; the absence of a SourceBinding is never such proof. Scope
tuples are sorted, unique, non-empty, and every scope ID exists in the pinned
CategoryRegistry. There is exactly one applicability entry per approved facet,
and entries are sorted by `facet_id`.

The three category-conditioned concepts have disjoint responsibilities:

| Contract | Question answered | Build gate |
| --- | --- | --- |
| `FacetApplicability` | Is this facet semantically meaningful for this product category? | Gate A |
| `FacetSourceBinding` | Which reviewed raw source can produce evidence here? | Gate A |
| `EffectiveFacetCapability` | May runtime commit, retrieve, probe, or clarify it in this exact active scope? | Gate B |

None of these three artifacts may be inferred at runtime from another one.

### 4.1 Source locator and binding schema

```python
class SourceKind(str, Enum):
    TOP_LEVEL = "top_level"
    DETAILS = "details"

class ValueCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"

@dataclass(frozen=True)
class SourceLocator:
    kind: SourceKind
    key: str

@dataclass(frozen=True)
class FacetSourceBinding:
    id: str
    facet_id: str
    source: SourceLocator
    applicable_category_scope_ids: tuple[str, ...]
    extractor_id: str
    catalog_value_normalizer_id: str
    priority: int
    completeness: ValueCompleteness
    resolver_id: str

@dataclass(frozen=True)
class FacetSourceBindingSet:
    schema: Literal["shopping-copilot/facet-source-bindings/v0"]
    category_registry_id: str
    facet_schema_id: str
    facet_applicability_id: str
    bindings: tuple[FacetSourceBinding, ...]
```

`SourceLocator.key` is the exact, case-sensitive top-level key or `details`
object key observed in the raw catalog. Discovery MAY group lexical variants
for human review, but publication represents each distinct raw key with a
separate binding; it does not invent an unversioned source-key normalization
layer. Bindings do not contain query-language aliases.

Binding scopes answer where this source interpretation is valid; they do not
replace FacetApplicability. A binding is applicable only when the product has a
`KNOWN` category assignment and at least one assigned leaf belongs to one of
the binding's scopes. An `UNKNOWN` or `CONFLICT` category assignment cannot
activate a binding. Overlapping bindings for the same `(facet_id, source)` are
permitted only when their priority makes the intended authority explicit. If
two overlapping bindings have equal priority but different extractor,
normalizer, resolver, or completeness declarations, the build MUST fail.

`priority` is a non-negative integer; smaller values are more authoritative
(`0` is higher priority than `100`).
Extractor, normalizer, and resolver IDs MUST resolve through the closed
implementation registries owned by the manifest's immutable
`builder_version`. Dynamic import strings, `eval`, and silent fallback to
identity normalization are prohibited. A logical implementation ID MUST NOT
be reused for different behavior within one builder version.

Binding IDs are unique and bindings are sorted by ID. Each binding scope tuple
is non-empty, sorted, unique, and references only scopes in the pinned
CategoryRegistry. Every binding facet exists in the pinned CatalogFacetSchema
and has a FacetApplicability entry. The union of a binding's materialized
scope nodes MUST be a subset of the union of that facet's applicability scope
nodes; a binding cannot claim validity where the facet is semantically
inapplicable. Every approved facet has at least one binding, and every
SourceLocator key is observed at least once in the raw catalog under its
declared SourceKind.

All published P0 bindings are human-reviewed structured bindings; P0 has no
weak-text, embedding, or model evidence tier. A binding's completeness is the
maximum claim permitted for its evidence. A categorical row-level extractor
MAY downgrade `COMPLETE` to `PARTIAL` but MUST NOT upgrade a binding declared
`PARTIAL`.
Completeness is operational for categorical values. Boolean, numeric, and TEXT
bindings MUST declare `COMPLETE`; numeric uncertainty is represented by its
interval rather than categorical completeness.

### 4.2 The single P0 resolution policy

P0 has exactly one product-fact policy:

```text
resolution_policy_id = structured_resolution_v1
```

It accepts only evidence produced by published, human-reviewed
FacetSourceBindings. Title, feature, description, embedding, weak-text, and
model-inferred evidence MUST NOT enter `structured_resolution_v1`. Such
enrichment requires a later policy and release.

```python
class EvidenceStatus(str, Enum):
    VALID = "valid"
    EMPTY = "empty"
    INVALID = "invalid"

@dataclass(frozen=True)
class FacetValueEvidence:
    id: str
    parent_asin: str
    facet_id: str
    binding_id: str
    status: EvidenceStatus
    raw_value_json: str
    canonical_value: ResolvedFacetValue | None

@dataclass(frozen=True)
class FacetEvidenceStore:
    schema: Literal["shopping-copilot/facet-evidence-store/v0"]
    catalog_id: str
    product_category_assignment_id: str
    facet_applicability_id: str
    facet_source_bindings_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    evidence: tuple[FacetValueEvidence, ...]
```

`raw_value_json` is the canonical JSON serialization of the copied source
value, never a mutable catalog object. Only `VALID` evidence has a canonical
value. A valid categorical evidence value carries its row-level completeness
inside `CategoricalValue`; the binding limits the maximum completeness it may
claim. Boolean, numeric, and text values are implicitly complete as specified
above. Empty and invalid evidence remains available for audit but never wins
resolution. Every evidence row names an existing catalog product and binding;
its `facet_id` MUST equal that binding's `facet_id`, and the binding MUST be
applicable to that product's `KNOWN` category assignment. Emitted evidence
rows are unique by `(parent_asin, binding_id)`
and sorted by that pair. A missing row, `EMPTY`, and `INVALID` all contribute
no resolved fact; unlike an emitted status, absence makes no audit claim about
why the binding produced no accepted value.

`FacetValueEvidence.id` is deterministic:

```text
"ev_" + lowercase_hex(sha256(canonical_json({
    "parent_asin": parent_asin,
    "facet_id": facet_id,
    "binding_id": binding_id,
    "status": status,
    "raw_value_json": raw_value_json,
    "canonical_value": canonical_value
})))
```

using the full 64-character digest of the complete evidence payload except its
own ID. The tagged canonical serialization of `canonical_value` is fixed by
section 5. Because the store permits at most one evidence row per
`(parent_asin, binding_id)`, any status or payload change replaces that row with
a different evidence identity in the new release.

All bindings for one facet MUST declare the same `resolver_id`; otherwise the
Gate A build fails. Every used resolver ID MUST exist in the release-pinned,
closed resolver registry belonging to `builder_version`; dynamic import or
fallback to an unpinned generic resolver is prohibited. For one
`(parent_asin, facet_id)`, the P0 resolver MUST:

1. derive facet applicability from ProductCategoryAssignment and the separate
   FacetApplicability artifact;
2. return `NOT_APPLICABLE` immediately when applicability is known false;
3. return `UNKNOWN` when category assignment or applicability is unknown;
4. filter to bindings applicable to the product's `KNOWN` category assignment;
5. retain every emitted evidence row from applicable bindings for audit;
6. filter to policy-allowed `VALID` evidence, ignoring missing, `EMPTY`, and
   `INVALID` evidence for truth resolution;
7. return `UNKNOWN` when no accepted valid evidence remains;
8. select the **minimum** binding `priority` represented among the remaining
   valid evidence;
9. pass only valid evidence from that priority layer to the facet's pinned
   resolver; and
10. accept only the exact `KNOWN` or `CONFLICT` result defined by that
    release-pinned resolver.

Once a priority layer contains valid evidence, larger-number, lower-priority
evidence never overrides or conflicts with it; it remains audit evidence. A
lower-priority valid layer is used only when every higher-priority applicable
binding is missing, `EMPTY`, `INVALID`, or otherwise rejected by the pinned
policy. Binding ID ordering is only a serialization tie-breaker and MUST NOT
decide truth. The generic `priority_exact_v1` resolver returns `KNOWN` when
every same-layer value payload is canonically identical and `CONFLICT` when
they are not, comparing categorical values before completeness metadata. It
does not union differing MULTI values. For identical categorical values with
mixed completeness, it returns `COMPLETE` when at least one evidence item is
complete and `PARTIAL` otherwise. A MULTI facet may union same-layer evidence
only when its reviewed `resolver_id` names a different pinned resolver whose
exact union, completeness, support, and conflict rules are immutable parts of
that builder version. Runtime code MUST NOT guess union behavior from
`item_cardinality`.

For a `KNOWN` or `CONFLICT` row, `evidence_ids` contains exactly the valid
selected-priority evidence that the pinned resolver declares supportive of
that result. Under `priority_exact_v1`, this is respectively all agreeing or
all disagreeing evidence in the selected layer. Lower-ranked, empty, and
invalid evidence remains in the evidence store but is not listed as support
for the resolved row.

Any facet that requires different canonical domains or query normalizers in
different category scopes MUST be split into different facet IDs before Gate
A approval.

## 5. Product facet index and matching

### 5.1 Tagged values and status

```python
@dataclass(frozen=True)
class CategoricalValue:
    kind: Literal["categorical"]
    values: tuple[str, ...]
    completeness: ValueCompleteness

@dataclass(frozen=True)
class BooleanValue:
    kind: Literal["boolean"]
    value: bool

@dataclass(frozen=True)
class TextValue:
    kind: Literal["text"]
    value: str

@dataclass(frozen=True)
class NumericValue:
    kind: Literal["numeric"]
    lower: int | float | None
    lower_inclusive: bool
    upper: int | float | None
    upper_inclusive: bool
    unit: str

ResolvedFacetValue = (
    CategoricalValue | BooleanValue | TextValue | NumericValue
)

# Canonical JSON tagged representations:
# {"kind":"categorical","values":[...],"completeness":"complete|partial"}
# {"kind":"boolean","value":true|false}
# {"kind":"text","value":"..."}
# {"kind":"numeric","lower":...,"lower_inclusive":...,
#  "upper":...,"upper_inclusive":...,"unit":"..."}

class ProductFacetStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    NOT_APPLICABLE = "not_applicable"

@dataclass(frozen=True)
class ResolvedProductFacetValue:
    parent_asin: str
    facet_id: str
    status: ProductFacetStatus
    value: ResolvedFacetValue | None
    evidence_ids: tuple[str, ...]
    resolution_policy_id: Literal["structured_resolution_v1"]

@dataclass(frozen=True)
class ProductFacetIndex:
    schema: Literal["shopping-copilot/product-facet-index/v0"]
    catalog_id: str
    product_category_assignment_id: str
    facet_applicability_id: str
    facet_source_bindings_id: str
    facet_evidence_store_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    entries: tuple[ResolvedProductFacetValue, ...]
```

The value variant is closed by CatalogFacetDefinition:

| `data_type` | Only valid value variant |
| --- | --- |
| `BOOLEAN` | `BooleanValue(kind="boolean")` |
| `CATEGORICAL` | `CategoricalValue(kind="categorical")` |
| `NUMERIC` | `NumericValue(kind="numeric")` |
| `TEXT` | `TextValue(kind="text")` |

This mapping applies identically to `VALID` evidence and `KNOWN` index rows;
any cross-kind payload is a build error. A CATEGORICAL `SINGLE` facet has
exactly one atomic value and a CATEGORICAL `MULTI` facet has one or more.
BOOLEAN, NUMERIC, and TEXT definitions are `SINGLE` as required in section 4.
Every index row names an existing catalog product and approved facet. Every
listed evidence ID resolves exactly once and has the same `(parent_asin,
facet_id)` as the index row; it must also belong to the selected priority layer
and support the result under the pinned resolver. Cross-product or cross-facet
support is invalid.

Categorical tuples are non-empty, unique, and sorted after normalization. A
SINGLE facet has exactly one value. Text values are non-empty canonical strings
and are available only to the explicitly SEARCH_ONLY path described below;
they have no session structured-matching operator in runtime v0. Numeric
endpoints are finite and not boolean; at least one endpoint is present. An
absent endpoint has a `None` value and its corresponding inclusivity flag MUST
be `False`; a present
endpoint has the declared boolean inclusivity. Equal endpoints are valid only
when both are inclusive. Units are non-empty canonical IDs defined by the
facet's normalizer.

The NumericValue protocol is generic, but runtime v0 may promote exactly one
numeric facet: `price`. A resolved `price` value has:

```text
unit = USD_CENT
lower/upper = int | None
```

Float endpoints are invalid for `price`. Catalog dollar text is converted to
integer minor units only when the structured parser can do so exactly; it MUST
parse JSON numeric tokens without an intermediate binary-float round trip and
MUST NOT round an ambiguous or over-precision value into cents. For example,
`$12.99` becomes the inclusive singleton `[1299, 1299]`, and `from $12.99`
becomes `[1299, +infinity)`. An unparseable price yields no valid evidence.
Committed session values for `price` are also integer USD cents: user
constraint `price <= 100 USD` becomes `price <= 10000`.

`KNOWN` has a non-null value. Other statuses have `value=None`. Evidence IDs
are sorted, unique, and non-empty for `KNOWN` and `CONFLICT`; they are empty
for `UNKNOWN` and `NOT_APPLICABLE`.
ProductFacetIndex entries are unique and sorted by `(parent_asin, facet_id)`.

The physical P0 index MAY store only `KNOWN` and `CONFLICT`. Lookup derives the
other states exactly as follows:

```text
assignment = product_category_assignment(product)
if assignment.status in {UNKNOWN, CONFLICT}:
    UNKNOWN
else if facet_applicability says NOT_APPLICABLE:
    NOT_APPLICABLE
else if a stored resolved row exists:
    that row
else:
    UNKNOWN
```

### 5.2 Categorical completeness and four-state matching

Product matching returns exactly:

```text
SATISFIED | VIOLATED | UNKNOWN | NOT_APPLICABLE
```

For `NOT_APPLICABLE`, return `NOT_APPLICABLE`. For product status `UNKNOWN` or
`CONFLICT`, return `UNKNOWN`.

For a known categorical product value set `V` and query value set `Q`:

| Operator family | Condition | COMPLETE result | PARTIAL result |
| --- | --- | --- | --- |
| positive (`EQ`, `IN`) | `V intersects Q` | `SATISFIED` | `SATISFIED` |
| positive (`EQ`, `IN`) | no intersection | `VIOLATED` | `UNKNOWN` |
| negative (`NEQ`, `NOT_IN`) | `V intersects Q` | `VIOLATED` | `VIOLATED` |
| negative (`NEQ`, `NOT_IN`) | no intersection | `SATISFIED` | `UNKNOWN` |

Thus positive evidence for `cotton` never proves the absence of `leather`
unless the resolved value is complete. Boolean values are exhaustive and use
the same table with a singleton complete set.

The reserved product-category facet does not use this table. It uses the exact
ProductCategoryAssignment matcher from section 3.3: `KNOWN` plus intersection
is `SATISFIED`, `KNOWN` plus disjoint membership is `VIOLATED`, and `UNKNOWN`
or `CONFLICT` assignment is `UNKNOWN`. The official P0 release gate requires
all assignments known, but the schema and matcher MUST still implement every
status.

Numeric matching treats the resolved interval as the possible product-value
set `D`. It constructs allowed set `C` from either one atomic predicate or all
numeric predicates for the same facet **and the same Commitment**. Repeated
lower and upper predicates intersect within that one group:

```text
D subset_of C        -> SATISFIED
D disjoint_from C    -> VIOLATED
otherwise            -> UNKNOWN
```

Endpoint inclusivity participates in subset and disjointness. Matching MUST
NOT coerce `UNKNOWN` or `NOT_APPLICABLE` into violation; a later retrieval
policy decides how each result affects recall and ranking. HARD and SOFT
predicates are never intersected into one `C`: retrieval evaluates their match
results separately, using HARD results for filtering policy and SOFT results
for ranking policy.

### 5.3 Resolved catalog statistics

Gate B statistics are a deterministic view of the same resolved index, not a
second evidence policy:

```python
@dataclass(frozen=True)
class ResolvedValueCount:
    canonical_value_json: str
    product_count: int

@dataclass(frozen=True)
class FacetScopeCatalogStats:
    facet_id: str
    category_scope_id: str
    scope_product_count: int
    known_count: int
    unknown_count: int
    conflict_count: int
    not_applicable_count: int
    known_value_counts: tuple[ResolvedValueCount, ...]

@dataclass(frozen=True)
class CatalogFacetStatsArtifact:
    schema: Literal["shopping-copilot/catalog-facet-stats/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    facet_schema_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    rows: tuple[FacetScopeCatalogStats, ...]
```

There is exactly one row for every approved ordinary facet and every published
CategoryScope, sorted by `(facet_id, category_scope_id)`. A product contributes
once to a row when its `KNOWN` assignment intersects that scope, even if
several assigned leaves intersect. The four status counts are obtained by the
section 5.1 lookup and MUST sum to `scope_product_count`.

`canonical_value_json` is the exact tagged section 5.1 value serialized with
`canonical_json`. `known_value_counts` contains every distinct `KNOWN` payload
in the row, not a truncated top-k; each count is positive, their sum equals
`known_count`, and rows are sorted by descending `product_count` then ascending
UTF-8 bytes of `canonical_value_json`. Ratios and presentation top-k values are
derived report data and are not part of this artifact.

## 6. Gate B, capabilities, and runtime projection

Gate B reviews resolved coverage, conflict rate, value distribution, examples,
and category-conditioned statistics produced under
`structured_resolution_v1`. Thresholds may recommend a decision but MUST NOT
automatically promote a facet.

Gate B decisions are:

```text
RUNTIME_ACCEPT | SEARCH_ONLY | SEMANTIC_ONLY | REJECT
```

```python
class RuntimePromotionDecision(str, Enum):
    RUNTIME_ACCEPT = "runtime_accept"
    SEARCH_ONLY = "search_only"
    SEMANTIC_ONLY = "semantic_only"
    REJECT = "reject"

@dataclass(frozen=True)
class EffectiveFacetCapability:
    facet_id: str
    category_scope_id: str
    decision: RuntimePromotionDecision
    resolution_policy_id: Literal["structured_resolution_v1"]
    intent_committable: bool
    retrieval_eligible: bool
    probe_eligible: bool
    clarification_eligible: bool

@dataclass(frozen=True)
class EffectiveFacetCapabilitySet:
    schema: Literal["shopping-copilot/effective-facet-capabilities/v0"]
    category_registry_id: str
    facet_schema_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    entries: tuple[EffectiveFacetCapability, ...]
```

Capabilities are **effective exact-scope rows**. P0 performs no runtime parent,
child, or union inheritance. Every CategoryScope that can be stored in active
state MUST have an explicit effective row for a usable facet. A missing row is
equivalent to all capabilities being false. In particular, a capability on a
raw parent scope does not implicitly apply to a union scope.

Entries are unique and sorted by `(facet_id, category_scope_id)`. Every facet,
scope, applicability, index, and policy reference resolves inside the same
release.

Let `A(facet)` be the union of the materialized node sets of that facet's
FacetApplicability scopes. If
`N(capability.category_scope_id) intersects A(facet)` is empty, all four
capability booleans MUST be false and the decision MUST be `SEMANTIC_ONLY` or
`REJECT`. Publishing `RUNTIME_ACCEPT`, `SEARCH_ONLY`, or any true permission
for a wholly inapplicable scope is a build error. A non-empty intersection
does not itself grant any permission; Gate B still publishes the exact row.

The following implications are mandatory:

```text
clarification_eligible -> probe_eligible
probe_eligible         -> retrieval_eligible
intent_committable     -> retrieval_eligible
intent_committable     -> decision is RUNTIME_ACCEPT
```

`SEMANTIC_ONLY` and `REJECT` rows have every capability false. A
`SEARCH_ONLY` row has `intent_committable=False` and
`clarification_eligible=False`; it MAY be retrieval-eligible. A
`RUNTIME_ACCEPT` row is the only row that may be intent-committable.

For runtime v0, any numeric facet other than `price` MUST have
`intent_committable=False`, `retrieval_eligible=False`,
`probe_eligible=False`, and `clarification_eligible=False`, and MUST NOT
project into the runtime FacetRegistry. It may remain a profiled Gate A
candidate for a later release.

A TEXT facet may be profiled and resolved from a reviewed structured binding,
but runtime v0 never makes it intent-committable, probe-eligible, or
clarification-eligible and never projects it into session-context. A reviewed
TEXT facet MAY be `SEARCH_ONLY` and retrieval-eligible.

`intent_committable` means that a user expression may be grounded as a
structured session preference in that exact category context. It does not mean
that the user is forbidden from expressing the requirement. If it is false,
the expression remains semantic-only.

P0 uses the same `structured_resolution_v1` resolved fact for hard filters,
soft ranking, Probe, and clarification. It MUST NOT claim different evidence
trust policies without introducing a new versioned resolution policy,
policy-keyed index, policy-keyed statistics, and a new semantic release.

An ordinary facet projects into the session-context `FacetRegistry` when at
least one exact-scope row has `intent_committable=True` **or**
`probe_eligible=True`. This includes a Probe-visible SEARCH_ONLY facet even
when it can never be committed as a Preference; the Gateway still rejects
writes wherever `intent_committable` is false. A facet that is retrieval-only
and never Probe-visible need not project. Projection is:

```text
Catalog boolean      -> FacetKind.CATEGORICAL with bool values
Catalog categorical  -> FacetKind.CATEGORICAL
Catalog numeric      -> FacetKind.NUMERIC (runtime v0: price only)
Catalog text         -> no session FacetSpec
```

The projected operator families MUST equal the closed session-context v1
families. Their content-addressed declarative artifact is:

```python
@dataclass(frozen=True)
class RuntimeFacetSpecRecord:
    facet_id: str
    kind: Literal["categorical", "numeric"]
    operator_values: tuple[str, ...]
    intent_value_normalizer_id: str

@dataclass(frozen=True)
class RuntimeFacetRegistryArtifact:
    schema: Literal["shopping-copilot/runtime-facet-registry/v0"]
    category_registry_id: str
    facet_schema_id: str
    effective_capabilities_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    entries: tuple[RuntimeFacetSpecRecord, ...]
```

Entries are unique and sorted by `facet_id`. `operator_values` contains the
lexicographically sorted enum values of exactly `CATEGORICAL_OPERATORS` or
`NUMERIC_OPERATORS` according to `kind`. The loader resolves each
`intent_value_normalizer_id` through the closed normalizer registry belonging
to `builder_version`, to the callable required by
`session_context.FacetSpec`. The callable MUST be
deterministic, MUST return a published canonical value for either a canonical
input or a reviewed alias, MUST reject every other input, and MUST be a fixed
point for every published canonical value.

The projected `price` normalizer accepts non-boolean `int` values in
`USD_CENT` within the section 2 safe-integer range only and rejects every
float or out-of-range integer. Unit/currency parsing happens before Preference
construction at the trusted grounding boundary.

The projection MUST additionally inject:

```text
FacetSpec(
    id="system_product_category",
    kind=CATEGORICAL,
    operators=CATEGORICAL_OPERATORS,
    normalizer=<release-bound CategoryScope ID validator>,
)
```

The reserved entry is serialized in RuntimeFacetRegistryArtifact with
`intent_value_normalizer_id="category_scope_id_v1"`; that pinned normalizer
accepts only CategoryScope IDs from the artifact's CategoryRegistry. It has no
ordinary facet-schema or capability row. Every other entry maps to exactly one
projected Gate-B intent-committable or Probe-visible facet and must have a
RuntimeValueLexicon domain.

This full operator family exists only because session-context v1 requires it.
The CatalogSemanticGateway below imposes the stricter category operation
semantics. `system_product_category` is outside ordinary Gate A/B discovery and
has no ordinary EffectiveFacetCapability row.

## 7. Runtime value lexicon and grounding

### 7.1 Lexicon schema

```python
class RuntimeValueMode(str, Enum):
    CLOSED = "closed"

@dataclass(frozen=True)
class RuntimeValueAlias:
    surface_form: str
    canonical_value: str | bool

@dataclass(frozen=True)
class CategoricalRuntimeDomain:
    kind: Literal["categorical"]
    facet_id: str
    value_mode: Literal["closed"]
    intent_value_normalizer_id: str
    canonical_values: tuple[str, ...]
    catalog_verified_values: tuple[str, ...]
    aliases: tuple[RuntimeValueAlias, ...]

@dataclass(frozen=True)
class BooleanRuntimeDomain:
    kind: Literal["boolean"]
    facet_id: str
    value_mode: Literal["closed"]
    intent_value_normalizer_id: str
    canonical_values: tuple[bool, ...]
    catalog_verified_values: tuple[bool, ...]
    aliases: tuple[RuntimeValueAlias, ...]

@dataclass(frozen=True)
class NumericRuntimeDomain:
    kind: Literal["numeric"]
    facet_id: Literal["price"]
    intent_value_normalizer_id: str
    canonical_unit: Literal["USD_CENT"]
    integer_only: Literal[True]

RuntimeFacetDomain = (
    CategoricalRuntimeDomain | BooleanRuntimeDomain | NumericRuntimeDomain
)

@dataclass(frozen=True)
class RuntimeValueLexicon:
    schema: Literal["shopping-copilot/runtime-value-lexicon/v0"]
    runtime_registry_id: str
    category_registry_id: str
    facet_applicability_id: str
    product_facet_index_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    domains: tuple[RuntimeFacetDomain, ...]
```

There is exactly one domain per projected **ordinary** runtime facet, excluding
`system_product_category`, sorted by `facet_id`. Runtime v0 supports only
`CLOSED` categorical and boolean domains; it does not claim OPEN or HYBRID
grounding. For each categorical or boolean P0 domain, `canonical_values` and
`catalog_verified_values` are equal typed sets: every groundable value has
support from at least one `KNOWN` product fact under
`structured_resolution_v1` somewhere in the facet's published applicability.
For a categorical facet this set is the union of the atomic strings inside
those known `CategoricalValue.values` tuples, not the set of whole tuple
payloads; for a boolean facet it is the set of known BooleanValue scalars.
Both tuples are non-empty, unique, and canonically sorted. Each domain's
kind mapping is exact:

| Runtime domain `kind` | Catalog data type | RuntimeFacetSpecRecord `kind` |
| --- | --- | --- |
| `categorical` | `CATEGORICAL` | `categorical` |
| `boolean` | `BOOLEAN` | `categorical` |
| `numeric` | `NUMERIC` facet `price` | `numeric` |

Its `intent_value_normalizer_id` MUST exactly equal the matching
RuntimeFacetSpecRecord ID; two different normalizers that happen to accept the
current values are still invalid.

The lexicon defines a facet's stable semantic value domain; it does not encode
current inventory availability per CategoryScope. Exact-scope applicability
and runtime permission come from FacetApplicability and
EffectiveFacetCapability. Category-conditioned counts remain statistics for
retrieval and asking. If the meaning or normalization of a value really
differs by category, Gate A MUST split it into different facet IDs rather than
publish scope-specific lexicon semantics.

Aliases are reviewed **value aliases**, such as a `"NIKE"` surface form mapping
to canonical `"nike"`. Each alias target MUST belong to both canonical and
catalog-verified values. `surface_form` stores the canonical lookup key
produced by the pinned normalizer's lexical pre-pass. Alias rows are sorted by
that key and unique within the facet domain; one lookup key cannot target two
values. The complete intent normalizer performs the lexical pre-pass, exact
canonical/alias lookup, and canonical return. Facet-language aliases such as
`made of` -> `material` remain Query Understanding resources and are not
RuntimeValueLexicon entries.

Grounding uses exact normalization and reviewed alias lookup only. P0 MUST NOT
use edit distance, fuzzy spelling repair, embeddings, or an LLM to turn an
unknown/typo value such as `nkie` into `nike`. An input that is neither a
canonical/catalog-verified value nor an unambiguous reviewed alias is not
structurally grounded.

`system_product_category` is grounded against CategoryRegistry scopes, not an
ordinary lexicon domain.

### 7.2 Grounding result

Runtime grounding for an ordinary facet consumes an extracted candidate, the
final proposed category scope for the turn, the release, and the effective
capability row. Reserved category grounding instead consumes the extracted
category candidate and CategoryRegistry; it has no ordinary capability or
lexicon row. Neither path allocates a Preference ID.

```python
class GroundingDisposition(str, Enum):
    GROUNDED = "grounded"
    SEMANTIC_ONLY = "semantic_only"
    AMBIGUOUS = "ambiguous"

@dataclass(frozen=True)
class GroundedPredicate:
    facet_id: str
    operator: Operator
    value: PreferenceValue

@dataclass(frozen=True)
class RuntimeValueGroundingResult:
    facet_id: str | None
    disposition: GroundingDisposition
    predicates: tuple[GroundedPredicate, ...]
    reason_code: str | None
    candidate_values: tuple[ScalarValue, ...]
    semantic_text: str | None
    semantic_polarity: SemanticPolarity | None
```

`predicates` is duplicate-free and sorted by `(facet_id, operator.value,
canonical_json(value))`, comparing the last component as UTF-8 bytes.
Consequently, numeric equality has the exact order `GE` then `LE`.
`candidate_values` is duplicate-free in the canonical scalar order from
section 2. These orders apply in every disposition, so grounding output never
depends on extractor discovery order.

`facet_id` is the one recognized ordinary or reserved facet being grounded.
It is non-null for `GROUNDED` and `AMBIGUOUS`, and every grounded predicate
MUST carry that same ID. It may be null for `SEMANTIC_ONLY` only with
`reason_code="unknown_facet"`; for that reason it MUST be null, while all
other semantic-only reasons retain the recognized facet ID. `AMBIGUOUS`
represents value ambiguity inside that one
facet, not unresolved ambiguity between several facet meanings.

For `GROUNDED`:

- predicates are non-empty, canonical, and accepted by the projected
  `FacetSpec`;
- for an ordinary facet, the exact effective capability has
  `intent_committable=True`;
- for `system_product_category`, the tuple contains exactly one predicate,
  using `EQ` with one published CategoryScope ID;
- categorical/boolean values belong to the facet's release-pinned lexicon
  domain;
- numeric values are integer `USD_CENT` values for `price`; and
- `reason_code` is absent and `candidate_values` is empty.

`semantic_text` and `semantic_polarity` are always both present or both absent.
A `GROUNDED` result MAY retain them when trusted grounding can prove that the
semantic representation describes the same atomic preference. If no reliable
dual representation exists, both are `None`; the original utterance remains
available separately as Preference `evidence_text`.

Numeric equality is emitted as inclusive `GE` and `LE` predicates because
numeric `EQ` is not a committed session-context v1 form.

For `SEMANTIC_ONLY`, no predicates or candidates are present and non-empty
semantic text, semantic polarity, and a lower-snake-case reason code are
required. Reasons include
at least `unknown_facet`, `facet_not_committable`, `unknown_value`,
`unsupported_operator`, and `unregistered_category_scope`.

For `AMBIGUOUS`, no predicates are present, at least two unique canonical
candidates are present in canonical scalar order, and semantic text is
retained with semantic polarity. For an ordinary categorical/boolean facet,
each candidate belongs to its pinned lexicon domain; for numeric `price`, each
candidate is a fixed point of the pinned safe-integer normalizer; for the
reserved facet, each is a published CategoryScope ID. Its reason code is
`ambiguous_value`.
Ambiguous output MUST NOT be committed as structured state without a
subsequent disambiguation.

The trusted coordinator assigns Preference IDs only after a `GROUNDED` result.
For a single predicate, it MAY copy the verified semantic pair onto the same
Preference. For a multi-predicate numeric expansion, it MUST NOT copy one
composite semantic phrase onto every atomic bound; it either supplies separately
verified per-bound representations, retains only structured predicates plus
their evidence text, or creates a separately justified semantic-only
Preference. This preserves session-context v1's same-atomic-preference rule.
Failure to ground is a normal semantic-only outcome, not permission to invent a
facet or value.

## 8. CatalogSemanticGateway

The gateway is an application-layer invariant boundary around the unchanged
session-context reducer. Calling `reduce_intent` directly proves only
session-context validity, not catalog-semantic validity.

For one update, the gateway MUST:

1. verify that the active semantic release is loaded and matches the
   release-bound store;
2. validate the batch with the unchanged session-context registry;
3. enforce the operation matrix below;
4. call the unchanged reducer against a temporary state;
5. validate the complete resulting IntentState under the final category
   context, capabilities, and lexicon; and
6. return the reduced state only if all checks succeed.

Any gateway failure discards the temporary result. The gateway MUST NOT repair,
reorder, or partially apply a batch.

### 8.1 Reserved category representation

The active category is stored as exactly one structured Preference on
`system_product_category` with:

```text
operator = EQ
value    = one published CategoryScope.id string
```

Commitment, source, source turn, evidence, confidence, and optional dual
semantic representation follow the unchanged session-context v1 rules. The
gateway adds no P0 restriction beyond one preference, `EQ`, and one valid scope
ID.

It is a storage adapter, not an ordinary catalog facet. It does not participate
in facet discovery, ordinary value distributions, facet entropy,
clarification selection, or official attribute mapping.

No category preference means the category is unset and the effective context
is `CategoryRegistry.root_scope_id`. Category absence does not mean don't-care.
Within an existing goal, deliberate broadening to the whole catalog is
expressed by replacing the reserved facet with `root_scope_id`, not by
`ClearFacet` or `SetDontCare`.

### 8.2 Operation validation matrix

| Operation | Ordinary structured facet | Semantic-only preference | `system_product_category` |
| --- | --- | --- | --- |
| `AddPreference` | Allowed if the final exact-scope capability is intent-committable and value is grounded | Allowed under session-context rules | **Rejected**; use `ReplaceFacet` |
| `ReplaceFacet` | Allowed with the same capability and grounding checks | Not applicable | Allowed only with exactly one canonical category Preference of the reserved shape |
| `RemovePreference` | Allowed | Allowed | **Rejected by ID** |
| `ClearFacet` | Allowed | Not applicable | **Rejected** |
| `SetDontCare` | Allowed only when the exact-scope facet is intent-committable | Not applicable | **Rejected** |
| `SwitchGoal` | Session-context rules apply; carried facts are revalidated in final context | Session-context rules apply | It may carry an already-active exact-shape category Preference or drop it as part of the goal reset; it cannot introduce or alter one |

Among operations that explicitly target `system_product_category`, only
`ReplaceFacet` is admitted, and a batch may contain at most one such operation.
If present, it MUST be the first operation after an optional first
`SwitchGoal`. The gateway evaluates all other operations against the batch's
final proposed category scope.

`SwitchGoal` is not a facet-targeting operation, but its reset/carry semantics
still affect category state. Before reduction, the gateway MUST validate the
pre-batch reserved preference. A SwitchGoal may carry exactly that existing ID
or omit it; it MUST NOT carry any other category-shaped ID, manufacture a new
category preference, or alter the carried object. Omission makes category
unset, with `root_scope_id` as the effective context, unless the immediately
following operation is a valid reserved `ReplaceFacet`. Both carried and final
category state are revalidated. Thus SwitchGoal cannot bypass the reserved
shape or use its carry list as an alternative category write operation.

### 8.3 Final-state validation

After temporary reduction:

- the reserved facet has zero or one active Preference of the exact shape;
- it is absent from `dont_care_facets`;
- every other structured active preference has an explicit exact-scope
  `intent_committable=True` capability;
- every active categorical/boolean value belongs to its facet's runtime
  lexicon;
- every active numeric value is a fixed point of the release-pinned intent
  normalizer;
- every ordinary don't-care facet has an explicit exact-scope
  `intent_committable=True` capability; and
- semantic-only preferences remain valid regardless of facet capability.

Changing the effective category can make prior structured preferences invalid.
The trusted planner MUST include the necessary removals, replacements, or
semantic fallbacks in the same batch. If the final effective category differs
from the pre-batch context, whether through reserved `ReplaceFacet` or a
SwitchGoal reset, and a previously active structured preference or don't-care
facet remains but is no longer applicable or intent-committable, the gateway
rejects the batch with
`INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE`. It never deletes, renames, or
reinterprets that preference automatically. Other final-state capability
violations use `FACET_NOT_COMMITTABLE`.

The same outer gateway also validates newly produced SearchBelief data:

- `system_product_category` MUST NOT appear in `facet_stats`;
- every emitted facet has `probe_eligible=True` for the active exact scope; and
- every categorical/boolean top value belongs to the release-pinned lexicon,
  and every numeric top value satisfies the pinned runtime normalizer.

The frozen SearchBelief DTO has no semantic-release or resolution-policy
field, so its payload alone cannot prove which index produced its statistics.
`CatalogBoundSessionStore` therefore accepts a SearchBelief only from its
private Probe producer, which is constructed with the same verified release
and `structured_resolution_v1` ProductFacetIndex; arbitrary application callers
receive no direct belief-commit handle. Snapshot decode can revalidate the
current belief's structure and values under the pinned release, but MUST NOT
claim that the DTO cryptographically proves historical producer provenance.

Gateway failures use a separate catalog-semantic error namespace and MUST NOT
change frozen session-context v1 error meanings. Stable codes include at least:

```text
RELEASE_MISMATCH
INVALID_RESERVED_CATEGORY_OPERATION
UNKNOWN_CATEGORY_SCOPE
FACET_NOT_COMMITTABLE
VALUE_NOT_GROUNDED
INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE
PROBE_FACET_NOT_ELIGIBLE
UNTRUSTED_SEARCH_BELIEF
CATALOG_COMMIT_MISMATCH
```

### 8.4 Atomic store authority

Gateway approval and session commit are one operation, not two independently
callable steps. `CatalogBoundSessionStore.turn(...)` returns only a
catalog-bound transaction. It privately enters the underlying
`SessionTransaction`, retains that transaction and its per-session lock, and
exposes the captured immutable `SessionContext`; it MUST NOT expose the raw
store, raw transaction, raw `commit`, or projected mutable registry.

The catalog-bound transaction's sole write method is conceptually:

```python
commit(
    next_context: SessionContext,
    *,
    probe_token: CatalogProbeToken | None = None,
) -> SessionContext
```

While the underlying session lock is still held, this method MUST perform the
following sequence:

1. require the wrapper's verified release ID and captured context identity;
2. require `next_context` to append exactly one TurnRecord and take the exact
   `accepted_update` from that appended record;
3. run the CatalogSemanticGateway against the captured IntentState and that
   batch (or require an unchanged IntentState when it is `None`), then require
   exact domain equality with `next_context.state.intent`;
4. validate the complete final category, preferences, don't-care set,
   capabilities, lexicon, and current SearchBelief under that same release;
5. enforce the Probe provenance rule below; and
6. call the private raw transaction's `commit(next_context)` before releasing
   the lock.

Any failure performs no state swap. The raw commit still runs all unchanged
session-context aggregate and transition validation; catalog validation does
not replace it. A pure gateway preview MAY return a proposed IntentState for
planning, but that value carries no commit authority and the entire sequence
above MUST run again inside the locked write.

`CatalogProbeToken` is a process-local, opaque, non-serializable, one-use
capability. Only the catalog-bound transaction's private Probe producer may
mint one, and it binds exact identities for the release, session, captured
context, expected final IntentState, and produced SearchBelief. A new or
changed `next_context.state.search_belief` requires a token whose hidden belief
is exactly domain-equal to it; an unchanged or cleared belief requires no
token. A token from another transaction, session, release, intended final
state, or belief is rejected and a token is consumed on its first commit
attempt. Thus a structurally valid caller-constructed SearchBelief cannot enter
live state. Decode remains limited to the historical-provenance claim in
section 9.3.

## 9. CatalogSemanticRelease and session envelope

### 9.1 Artifact references

```python
ArtifactKind = Literal[
    "catalog",
    "category_registry",
    "product_category_assignment",
    "facet_schema",
    "facet_applicability",
    "facet_source_bindings",
    "facet_evidence_store",
    "product_facet_index",
    "facet_stats",
    "effective_capabilities",
    "runtime_value_lexicon",
    "runtime_registry",
    "reviewed_config",
]

@dataclass(frozen=True)
class ArtifactRef:
    kind: ArtifactKind
    schema: str
    content_id: str
    byte_size: int

@dataclass(frozen=True)
class ReviewedRuntimeFacetConfig:
    facet_id: str
    intent_value_normalizer_id: str
    aliases: tuple[RuntimeValueAlias, ...]

@dataclass(frozen=True)
class ReviewedSemanticConfig:
    schema: Literal["shopping-copilot/reviewed-semantic-config/v0"]
    catalog_id: str
    category_graph_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    builder_version: str
    category_scopes: tuple[CategoryScope, ...]
    facets: tuple[CatalogFacetDefinition, ...]
    facet_applicability: tuple[FacetApplicability, ...]
    source_bindings: tuple[FacetSourceBinding, ...]
    capabilities: tuple[EffectiveFacetCapability, ...]
    runtime_facets: tuple[ReviewedRuntimeFacetConfig, ...]

@dataclass(frozen=True)
class CatalogSemanticReleaseManifest:
    schema: Literal["shopping-copilot/catalog-semantic-release/v0"]
    catalog_id: str
    category_registry_id: str
    product_category_assignment_id: str
    facet_schema_id: str
    facet_applicability_id: str
    facet_source_bindings_id: str
    facet_evidence_store_id: str
    product_facet_index_id: str
    facet_stats_id: str
    effective_capabilities_id: str
    runtime_value_lexicon_id: str
    runtime_registry_id: str
    reviewed_config_id: str
    resolution_policy_id: Literal["structured_resolution_v1"]
    builder_version: str
    artifacts: tuple[ArtifactRef, ...]
```

`resolution_policy_id` is the single logical policy fixed by this contract.
`builder_version` matches the lower-snake-case ID grammar in section 2 and
identifies one immutable builder implementation, including its closed
extractor, catalog normalizer, intent normalizer, resolver, category
normalizer, matcher, and projection registries. A loader MUST reject an
unsupported builder version or an implementation ID absent from its registries.
Changing behavior under an existing builder version or logical implementation
ID is prohibited; the version must change, which changes the release ID.

`ReviewedSemanticConfig` is the exact human-reviewed build input, not an opaque
config directory. Its scope, facet, applicability, binding, and capability
tuples use the same uniqueness and ordering rules as their published artifact
counterparts. `runtime_facets` has exactly one entry per projected ordinary
facet, sorted by `facet_id`; its normalizer ID MUST equal the ID in both the
RuntimeFacetSpecRecord and RuntimeFacetDomain. Its aliases use the ordering and
validation rules in section 7.1 and must equal that domain's aliases exactly;
a numeric runtime config has no aliases. A successful build requires exact
domain equality between each reviewed tuple and its corresponding published
projection. The config's catalog, graph, policy, and builder references MUST
equal the manifest and generated artifacts. This makes review provenance
content-addressed without introducing executable config plugins.

Every manifest ID other than `resolution_policy_id` and `builder_version`
names artifact content. Each `ArtifactRef.content_id` is `sha256:` plus the
SHA-256 digest of the exact artifact bytes, and `byte_size` is their exact
length. Artifact refs are sorted by `kind`, unique by kind, and the tuple has
exactly the 13 kinds in `ArtifactKind`.

The P0 field-to-ref and schema mapping is exact:

| Manifest field | `ArtifactRef.kind` | `ArtifactRef.schema` |
| --- | --- | --- |
| `catalog_id` | `catalog` | `shopping-copilot/raw-catalog-jsonl/v1` |
| `category_registry_id` | `category_registry` | `shopping-copilot/category-registry/v0` |
| `product_category_assignment_id` | `product_category_assignment` | `shopping-copilot/product-category-assignment/v0` |
| `facet_schema_id` | `facet_schema` | `shopping-copilot/catalog-facet-schema/v0` |
| `facet_applicability_id` | `facet_applicability` | `shopping-copilot/facet-applicability/v0` |
| `facet_source_bindings_id` | `facet_source_bindings` | `shopping-copilot/facet-source-bindings/v0` |
| `facet_evidence_store_id` | `facet_evidence_store` | `shopping-copilot/facet-evidence-store/v0` |
| `product_facet_index_id` | `product_facet_index` | `shopping-copilot/product-facet-index/v0` |
| `facet_stats_id` | `facet_stats` | `shopping-copilot/catalog-facet-stats/v0` |
| `effective_capabilities_id` | `effective_capabilities` | `shopping-copilot/effective-facet-capabilities/v0` |
| `runtime_value_lexicon_id` | `runtime_value_lexicon` | `shopping-copilot/runtime-value-lexicon/v0` |
| `runtime_registry_id` | `runtime_registry` | `shopping-copilot/runtime-facet-registry/v0` |
| `reviewed_config_id` | `reviewed_config` | `shopping-copilot/reviewed-semantic-config/v0` |

For every row, the manifest MUST contain exactly one ref of the named kind and
its `content_id` MUST equal the corresponding field. No ref may point to bytes
whose embedded schema or cross-references disagree with the manifest. Every
semantic artifact after the raw catalog is exactly one `canonical_json`
document with the declared dataclass shape; missing, unknown, or duplicate
fields fail loading.

`shopping-copilot/raw-catalog-jsonl/v1` is UTF-8 without BOM and contains
exactly 50,000 non-blank JSON object records, one per physical line. JSON
duplicate keys, lone surrogates, non-finite numbers, and invalid trailing data
are rejected; integer tokens outside the section 2 safe range are also
rejected. Every record has a unique, non-empty, trimmed string
`parent_asin`, a non-empty array `categories` whose elements are non-empty
strings, and an object `details`; other top-level fields and JSON values are
preserved as raw source data. `catalog_id` hashes the exact source bytes, not a
parsed or reserialized form. The loader validates every record against this
format before accepting the release.

### 9.2 Canonical release hash

The manifest is serialized with the single `canonical_json` definition in
section 2. It intentionally contains no release ID field. Its external ID is
exactly:

```text
catalog_semantic_release_id
    = "sha256:" + lowercase_hex(sha256(canonical_json(manifest)))
```

The manifest contains only strings, integers, booleans, objects, and arrays;
floats are prohibited in the manifest itself.

Before use, a loader MUST verify the release ID, every artifact content hash
and size, every artifact schema, all cross-references, the single
`structured_resolution_v1` policy, runtime projection invariants, and
availability of `builder_version` and all referenced implementation IDs in its
closed registries. Any mismatch fails closed
before a session is created or decoded.

### 9.3 Outer session envelope

SessionContext and its existing codec remain unchanged. Release pinning uses a
new outer envelope:

```python
@dataclass(frozen=True)
class CatalogBoundSessionEnvelope:
    schema: Literal["shopping-copilot/catalog-bound-session/v0"]
    session_id: str
    catalog_semantic_release_id: str
    session_snapshot_sha256: str
    session_snapshot_base64url: str
```

`session_snapshot_base64url` is the exact byte output of the existing
session-context v1 `encode_snapshot`, encoded as unpadded RFC 4648 base64url.
`session_snapshot_sha256` is the lowercase 64-character SHA-256 digest of
those inner bytes. The outer envelope uses the same canonical JSON rules as
the release manifest. `session_id` follows session-context v1's canonical
session-ID rules and MUST equal the decoded inner `SessionContext.session_id`.

The outer JSON object has exactly the four declared fields, all as strings;
missing, unknown, or duplicate fields are rejected. Its input bytes MUST equal
`canonical_json` of the parsed object. `catalog_semantic_release_id` matches
`^sha256:[0-9a-f]{64}$` and `session_snapshot_sha256` matches
`^[0-9a-f]{64}$`. The base64url field is non-empty, contains only
`[A-Za-z0-9_-]`, contains no `=`, and has length modulo four other than one.
Strict decoding must reject non-zero unused trailing bits, and re-encoding the
decoded bytes without padding MUST reproduce the field byte-for-byte.

Decode order is mandatory:

1. parse JSON with duplicate detection; enforce exact keys, types, schema,
   canonical bytes, identifier patterns, and base64url lexical shape;
2. load and fully verify the named semantic release;
3. strictly decode base64url, require exact re-encoding, and verify the inner
   snapshot hash;
4. construct the release's projected session-context `FacetRegistry`;
5. call the unchanged session-context v1 `decode_snapshot` with that registry;
6. call session-context v1 `encode_snapshot` on the decoded context and require
   byte-for-byte equality with the decoded inner bytes;
7. require outer and inner session IDs to be equal;
8. replay accepted batches through CatalogSemanticGateway and require that
   they reproduce the decoded final IntentState; and
9. validate only the active `SessionState.search_belief`, when present, against
   the pinned release and active exact category scope.

The first failing stage determines the error class; implementations MUST NOT
continue into later stages and replace an earlier envelope, release, encoding,
hash, inner-snapshot, session-ID, replay, or active-belief failure with a later
one. This gives malformed or adversarial envelopes deterministic precedence.

Historical TurnRecords retain only `search_belief_probe_id`, not complete
historical belief payloads. Decode and replay MUST NOT claim to reconstruct or
revalidate historical beliefs that are no longer present.

P0 permits exactly one semantic release per `CatalogBoundSessionStore`. The
wrapper owns one verified release and one existing `InMemorySessionStore`
constructed with that release's projected registry. Reset, get, turn, commit,
encode, and decode MUST reject a different release ID. Serving old and new
releases concurrently requires separate wrapper stores or processes; a single
existing `InMemorySessionStore` MUST NOT mix them.

The wrapped `InMemorySessionStore` is private implementation state. Every
application commit MUST first pass CatalogSemanticGateway; callers MUST NOT be
given a direct commit handle that can bypass the gateway.

## 10. Required invariants and tests

At minimum, implementation tests MUST prove:

- canonical JSON, category node IDs, category graph ID, and scope IDs match
  their exact hash preimages and are deterministic;
- catalog-semantic integers outside the JCS safe range are rejected without
  coercion, including committed price values;
- a CategoryScope's materialized nodes equal the exact union-of-subtrees
  closure;
- redundant roots, unknown nodes, and equal-node-membership duplicate scopes
  fail publication;
- ProductCategoryAssignment preserves `KNOWN`, `UNKNOWN`, and `CONFLICT`, and
  its matcher returns the exact results in section 3.3;
- the generic assignment schema accepts valid unknown/conflict records while
  the official 50k P0 runtime gate requires all 50,000 assignments known;
- `root_scope_id` contains the whole category graph and matches all products in
  the all-known P0 assignment artifact;
- goal and reserved category can each be present or absent without deriving or
  overwriting the other;
- CategoryRegistry and ProductCategoryAssignmentSet are separate artifacts,
  and the assignment's `category_graph_id` must match the registry graph;
- semantic facet applicability, source-binding applicability, and runtime
  capability are distinct and independently validated;
- source locators preserve exact raw catalog keys; two lexical key variants do
  not collapse without two explicit reviewed bindings;
- every approved facet has a binding and no binding scope extends outside the
  facet's semantic applicability region;
- ProductCategoryAssignment uncertainty yields product-facet `UNKNOWN`, while
  known category disjointness may yield `NOT_APPLICABLE`;
- overlapping equal-priority bindings with incompatible extractor,
  normalizer, resolver, or completeness declarations fail the build;
- lower numeric `priority` wins, and lower-priority evidence never overrides a
  valid higher-priority layer but does provide fallback when every higher layer
  lacks policy-allowed valid evidence;
- `priority_exact_v1` merges same-priority identical evidence, conflicts on
  incompatible evidence, and never unions differing MULTI values, while any
  reviewed MULTI union resolver follows its release-pinned rules;
- only reviewed structured bindings enter `structured_resolution_v1`;
- evidence and index value variants exactly match facet data type/cardinality,
  and every evidence/index reference preserves the same product and facet;
- evidence IDs change with any status or canonical evidence-payload change;
- sparse ProductFacetIndex lookup derives `UNKNOWN` and `NOT_APPLICABLE`
  exactly as specified;
- the full positive/negative COMPLETE/PARTIAL categorical matching table is
  covered and PARTIAL values never prove absence;
- numeric predicates combine only within one `(facet, Commitment)` group and
  HARD and SOFT bounds are never intersected together;
- CatalogFacetStatsArtifact contains all facet/scope rows, exact four-status
  count sums, and deterministically ordered complete known-value counts;
- effective capability lookup performs no implicit inheritance;
- a capability row wholly disjoint from FacetApplicability cannot publish a
  runtime/search decision or any true permission;
- exactly the Gate-B facets that are intent-committable or Probe-visible in at
  least one exact scope, plus `system_product_category`, project to the session
  registry; Probe-only facets can therefore appear in a valid SearchBelief;
- RuntimeFacetRegistryArtifact is deterministic, contains the reserved entry
  exactly once, and resolves every normalizer through `builder_version`'s
  closed registry;
- each RuntimeFacetDomain and RuntimeFacetSpecRecord agree exactly on facet and
  intent normalizer ID, and their kinds satisfy the section 7.1 mapping table;
- TEXT facets never project and may only use the reviewed SEARCH_ONLY path;
- no numeric facet other than `price` projects in runtime v0;
- price uses integer `USD_CENT` end to end, rejects float committed values, and
  never rounds over-precision catalog input;
- the reserved facet uses a valid lower-snake-case ID and a CategoryScope
  membership normalizer;
- the unchanged raw session reducer may accept operations that the gateway
  rejects, and every application path uses the gateway;
- catalog validation, SearchBelief provenance validation, and the private raw
  session commit execute under one session lock with no bypass handle;
- a caller-built SearchBelief and a Probe token from a different release,
  session, transaction, final intent, or belief are rejected;
- category `AddPreference`, `RemovePreference`, `ClearFacet`, `SetDontCare`,
  non-EQ values, multiple replacement preferences, and category facet stats
  are rejected by the gateway;
- SwitchGoal can carry only the already-valid reserved preference or omit it;
  omission produces the root effective context, and carry cannot introduce or
  alter category state;
- a category change with an incompatible retained structured preference or
  don't-care facet is rejected specifically with
  `INAPPLICABLE_PREFERENCE_AFTER_CATEGORY_CHANGE` and no automatic repair
  occurs;
- runtime v0 categorical/boolean domains are global per facet and CLOSED,
  every canonical value is catalog-verified somewhere in its applicability,
  category-specific inventory absence does not invalidate that value, aliases
  cannot introduce values, and unknown/typo strings do not ground;
- a GROUNDED result may preserve a paired semantic representation, and the
  coordinator does not copy a composite phrase onto unrelated atomic bounds;
- a grounding result cannot mix facet IDs, and ambiguous candidates must be
  release-valid values for its one named facet;
- unobserved categorical/boolean, ambiguous, or non-committable runtime values
  never become structured preferences;
- grounding candidates and predicates have canonical order, and numeric
  equality grounds to inclusive `GE` then `LE` predicates;
- the release manifest contains and verifies separate category-registry,
  product-category-assignment, facet-applicability, source-binding, evidence,
  index, stats, capability, lexicon, runtime-registry, and reviewed-config
  artifacts, with every content field equal to its exact required ArtifactRef;
- reviewed config projections exactly equal their corresponding published
  semantic decisions;
- two builds from identical catalog, builder version, and reviewed config produce
  byte-identical artifacts and the same release ID;
- changing any artifact byte, builder version, or pinned implementation ID
  changes or invalidates the release;
- a catalog-bound snapshot cannot be decoded with a different release or a
  mismatched outer `session_id`;
- outer-envelope unknown/duplicate fields, noncanonical JSON, malformed or
  noncanonical base64url, and snapshot-hash tampering fail in the defined
  decode precedence;
- an inner snapshot whose decoded object re-encodes to different bytes is
  rejected as noncanonical;
- decode replays accepted batches through the gateway, validates only the
  current SearchBelief, reproduces final IntentState, and makes no claim to
  reconstruct historical beliefs; and
- existing session-context v1 serialization and reducer tests remain unchanged
  and passing.

## 11. Explicit P0 deferrals

The following require a later, separately versioned contract:

- title/features/description value enrichment;
- model-inferred product facts;
- different hard, soft, Probe, or clarification evidence policies;
- runtime capability inheritance;
- OPEN or HYBRID runtime value domains;
- runtime numeric facets other than integer-USD-cent `price`;
- dynamic category-scope construction;
- automatic query-language synonym generation;
- silent session migration across semantic releases; and
- multiple semantic releases inside one session store.

These deferrals do not prevent a user from expressing an unsupported need.
Such a need is retained as a semantic-only preference until a future release
can ground it reliably.
