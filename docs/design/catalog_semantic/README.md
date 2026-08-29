# Catalog Semantic Layer

- Contract: **v0 normative for P0 implementation**
- Implemented: **CS0 profiler, frozen CS1 category candidate, CS2 source profiling,
  the reviewed Gate-A `price` candidate, CS3 price evidence/index/statistics,
  CS4 approved capabilities, CS5 runtime grounding, the CS6 immutable release,
  and the CS7 catalog-bound session gateway**
- Current checkpoint: **CS7 verified; Query Understanding P0 integrated;
  Query Compiler and production retrieval integration pending**
- Last contract review: **2026-08-27**

This layer turns the frozen catalog into category, facet, evidence, and runtime
artifacts without changing the session-context v1 contract. The raw profiler
is intentionally earlier in the pipeline than any reviewed semantic decision:

```text
raw catalog
    -> CS0 deterministic raw profile                  implemented
    -> CS1 Category Foundation
         Pass A graph proposal                       implemented
         Pass B reviewed scopes / candidates         accepted and frozen
    -> CS2 Gate A
         exhaustive source profile                   implemented
         price facet/applicability/binding decision  accepted and frozen
         other 288 source locators                    awaiting review
    -> CS3 resolver / ProductFacetIndex               implemented and verified
    -> CS4 Gate B review / capability candidate       approved and verified
    -> CS5A runtime registry / numeric lexicon        implemented and verified
    -> CS5B deterministic grounding                   implemented and verified
    -> CS6 verified release                           implemented and verified
    -> CS7 CatalogSemanticGateway                     implemented and verified
    -> CS8 Query Understanding handoff                 P0 implemented
         Query Compiler / production Probe             pending
```

## Documents

- [`methodology-v0.md`](methodology-v0.md): end-to-end engineering workflow,
  human review gates, milestone inputs/outputs, acceptance criteria, and
  repository hygiene. It is non-normative; the contract remains authoritative.
- [`contract-v0.md`](contract-v0.md): normative category, facet, resolution,
  grounding, gateway, and release rules.
- [Facet Registry research report](<../facet/TechJam Facet Registry v0：从 50k Catalog 抽取和构建 Facet 的实施规范.md>):
  discovery and implementation research retained as design input; the contract
  is authoritative where the documents differ.

## Raw profiler

Run the read-only profiler against the downloaded 50,000-product catalog:

```powershell
python -m shopping_copilot.catalog.profiling `
  data/catalog.jsonl `
  artifacts/catalog-profile
```

The installed `catalog-profile` command accepts the same arguments. It writes
the following deterministic UTF-8 files only after a successful profiling
pass:

| File | Contents |
| --- | --- |
| `bundle-manifest.json` | Hash and byte-size integrity record, published after the data files |
| `profile.json` | Complete machine-readable profile bound to the raw file SHA-256 |
| `report.md` | Human review summary |
| `category-nodes.jsonl` | Exact raw full-path prefix tree and support |
| `product-category-assignments.jsonl` | Per-row raw category assignment audit |
| `detail-keys.jsonl` | Raw-key support, shape, value mass, and stable samples |
| `category-detail-coverage.jsonl` | Sparse category-subtree × raw-key coverage |

Generated reports live under the ignored `artifacts/` directory. The profiler
preserves raw category paths and keys exactly; it does not publish canonical
`CategoryScope`, facet IDs, source bindings, or runtime values.

Bundle data files are staged before publication and the integrity manifest is
replaced last. A single-writer lock prevents concurrent publication. A reader
must call `validate_profile_bundle(...)` before using a bundle; an interrupted
multi-file publication then fails closed instead of being mistaken for a
coherent generation.

An abrupt process or machine termination can leave the sibling
`.catalog-profile.write.lock` file behind. Remove that file only after
confirming that no profiler process is still writing the same bundle.

## Frozen implementation boundary

The semantic build implements the contract in order. It must not infer
facets directly inside session context, retrieval, the official adapter, or the
toy evaluator. Query Understanding remains responsible for interpretation and
candidate extraction, calls the CS5 deterministic grounding service, and must
enter session context through the implemented `CatalogSemanticGateway`.

## CS1 category build

CS1 is deliberately split around a human review boundary. Pass A rebuilds the
canonical prefix graph from the exact raw catalog, records every raw path
mapping and lexical collision, and emits an empty selection template. It does
not infer user-facing scopes:

```powershell
catalog-category propose `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-proposal

catalog-category validate `
  artifacts/catalog-semantic/category-proposal
```

The proposal bundle contains:

| File | Purpose |
| --- | --- |
| `category-graph-proposal.json` | Canonical nodes and graph identity |
| `raw-path-mapping.jsonl` | Exact raw-prefix provenance and support |
| `collision-report.json` | Lexical normalization collision audit |
| `category-scope-selection.template.json` | Graph-pinned review template; intentionally not a valid selection |
| `report.md` | Human-readable Pass A summary and checkpoint |
| `bundle-manifest.json` | Exact byte-size/hash integrity record, published last |

For the frozen 50,000-product catalog, the verified Pass A baseline is:

```text
catalog_id        sha256:da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67
category_graph_id sha256:04dd760c60a5035cb5a461a52af381d2eabe66477da3f734a4c47ce02323a8e9
raw prefixes      1,832
canonical nodes   1,832
canonical roots   2
collision groups  0
```

After the graph is accepted, copy the pinned IDs into a source-controlled
`shopping-copilot/category-scope-selection/v0` document under
`config/catalog_semantic/v0/`. Every scope supplies only a reviewed label and
sorted root node IDs. Pass B then deterministically materializes full subtree
closures, scope IDs, the `CategoryRegistry`, and all product assignments:

```powershell
catalog-category build `
  data/catalog.jsonl `
  config/catalog_semantic/v0/category-scope-selection.json `
  artifacts/catalog-semantic/category-candidate

catalog-category validate `
  artifacts/catalog-semantic/category-candidate `
  --catalog data/catalog.jsonl
```

`propose` applies the official 50,000-record raw catalog gate. `build`
additionally applies the 50,000-assignment/all-`KNOWN` publication gate by
default. Generated proposal and candidate bundles are review artifacts, not a
`CatalogSemanticRelease`. Nothing under `artifacts/` enters Git; only an
explicitly reviewed selection and later reviewed config fragment do.

The current closed normalizer is `catalog_semantic_v0_ucd17_0`. Its NFKC
backend, generated default-full-casefold table, and 29-code-point whitespace
domain are all version-pinned and independent of the host Python Unicode
tables. This lets Python 3.10+ build and validate the same category IDs while
still failing closed if the pinned components disagree. See
[`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) for provenance.
The offline scripts in `scripts/catalog_semantic/` regenerate the case-fold
module and verify it plus every explicitly listed NFKC test row against
separately downloaded, hash-pinned Unicode 17 source files; the scripts never
fetch Unicode data at runtime or claim coverage beyond those listed rows.

## CS2 Gate-A source profile

CS2 starts with an observation checkpoint, not an automatically generated
FacetRegistry. The source-controlled
[`gate-a-profile-selection.json`](../../../config/catalog_semantic/v0/gate-a-profile-selection.json)
pins the accepted CS1 CategoryRegistry and ProductCategoryAssignmentSet by
content ID. The official selection profiles exact top-level `price` and
`store` lanes plus every observed `details` key; low-support keys are retained.

```powershell
catalog-facet-profile build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  config/catalog_semantic/v0/gate-a-profile-selection.json `
  artifacts/catalog-semantic/gate-a-source-profile

catalog-facet-profile validate `
  artifacts/catalog-semantic/gate-a-source-profile `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  config/catalog_semantic/v0/gate-a-profile-selection.json
```

The generated, ignored bundle contains:

| File | Purpose |
| --- | --- |
| `profile.json` | Exact upstream pins, scope denominators, source inventory, and profiling config |
| `scope-source-profiles.jsonl` | Complete reviewed-scope × exact-source count/value matrix |
| `source-samples.jsonl` | Stable bounded nonempty examples with category provenance |
| `price-audit.json` | Exact numeric-cent and unmodified string-lane observations |
| `report.md` | Gate-A review queue and category-conditioned evidence summary |
| `bundle-manifest.json` | Exact artifact size/hash integrity record, published last |

For the frozen catalog the verified source-profile baseline is 289 exact
locators (two selected top-level fields and all 287 details keys), 15 reviewed
scopes, and 4,335 scope-source rows. The build preserves the observed empty
details key instead of silently dropping it. That raw locator is reviewable but
is not thereby eligible for a published binding.

The first `price` lane observes 39,473 nulls, 10,410 JSON numbers that are all
exactly representable as non-negative integer cents, and 117 strings (112 em
dashes and five `from ...` values). None of the string values is interpreted by
the profiler. The source-profile stage itself publishes no
CatalogFacetDefinition, FacetApplicability, FacetSourceBinding, extractor,
normalizer, resolver, or runtime capability; its review queue remains immutable
observation evidence even after a later decision is accepted.

## CS2 reviewed Gate-A `price` candidate

The repository owner accepted the first extraction proposal on 2026-08-28.
[`gate-a-selection.json`](../../../config/catalog_semantic/v0/gate-a-selection.json)
records the exact decision: one NUMERIC/SINGLE `price` facet, root-scope
applicability, and one exact top-level `price` binding. The closed implementation
IDs are `top_level_price_usd_v1`, `usd_cent_interval_v1`, and
`priority_exact_v1`.

Build and independently revalidate the ignored candidate bundle with:

```powershell
catalog-facet-gate-a build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  config/catalog_semantic/v0/gate-a-profile-selection.json `
  artifacts/catalog-semantic/gate-a-source-profile `
  config/catalog_semantic/v0/gate-a-selection.json `
  artifacts/catalog-semantic/gate-a-candidate

catalog-facet-gate-a validate `
  artifacts/catalog-semantic/gate-a-candidate `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  config/catalog_semantic/v0/gate-a-profile-selection.json `
  artifacts/catalog-semantic/gate-a-source-profile `
  config/catalog_semantic/v0/gate-a-selection.json
```

The candidate bundle contains:

| File | Purpose |
| --- | --- |
| `catalog-facet-schema.json` | Approved `price` identity and numeric/single shape |
| `facet-applicability.json` | Root-scope semantic applicability |
| `facet-source-bindings.json` | Exact source plus closed implementation IDs |
| `extraction-audit.json` | Rebuilt frozen-catalog result counts |
| `reviewed-gate-a-selection.json` | Canonical copy and identity of the owner-approved input |
| `candidate.json` | Upstream pins and content IDs |
| `report.md` | Human-readable approval boundary and audit summary |
| `bundle-manifest.json` | Exact artifact size/hash integrity record |

The accepted candidate identities are:

```text
CatalogFacetSchema      sha256:058fe6f118c9b94f0bf1fb45d14fa1c9baa4b6068f00650608eecd62014d07d6
FacetApplicabilitySet   sha256:7ab8513f05fa31f033359826e20d3ea155c2f1724f67ee1a7d498168a575c0b9
FacetSourceBindingSet   sha256:6676e33714c8cb1fd1204161536759fc4baa1bfb994f22529855b695a9008ea3
GateASelection          sha256:a922e6d34005d69cee95be0bc29c0694555de81c81aa838edd9daae0e6e3ddaf
bundle manifest SHA-256 73bc03150032e96106587c0ec931dcffb457a8ff9c138d162839d39215d20f58
```

The real build produces 10,415 VALID prices, 39,473 EMPTY values, and 112
INVALID placeholders. VALID contains 10,410 exact intervals and five inclusive
lower-bound intervals; the sole exact zero is retained and audited. Numeric JSON
tokens are parsed through `Decimal`; floats, rounding, negative values, signed
zero, and over-precision values are rejected.

This is extraction approval only. It does not approve hard budget filtering,
session commits, clarification, Probe behavior, or official adapter mapping.
Those decisions remain behind human Gate B.

## CS3 `price` evidence, index, and statistics

CS3 applies only the reviewed structured price binding. It reads the exact
catalog bytes and writes derived files to a separate ignored directory; it
never rewrites `data/catalog.jsonl`, fills a missing source value, or injects a
product. The command also refuses an output target that contains any input and
records the catalog hash and byte size before and after staging.

```powershell
catalog-facet-resolution build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate

catalog-facet-resolution validate `
  artifacts/catalog-semantic/resolution-candidate `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate
```

The ignored candidate bundle contains:

| File | Purpose |
| --- | --- |
| `facet-evidence-store.json` | 50,000 immutable per-product price outcomes with copied raw JSON, status, typed value, and stable evidence ID |
| `product-facet-index.json` | Sparse query index containing only resolved `KNOWN` or `CONFLICT` rows |
| `catalog-facet-stats.json` | Complete four-state counts and full known-value distributions for every reviewed category scope |
| `catalog-read-only-audit.json` | Matching catalog SHA-256 and byte size before and after output staging |
| `candidate.json` | Compact input pins, artifact IDs, and explicit `gate_b_runtime_approved=false` |
| `report.md` | Human-readable evidence examples, category coverage, and safety boundary |
| `bundle-manifest.json` | Exact artifact byte-size/hash integrity record, published last |

The verified 50,000-product baseline is:

```text
FacetEvidenceStore       sha256:649b47d00fbf7ef170acb06a2075b9df15deee267d1e9085f67e1c9a990a243e
ProductFacetIndex        sha256:8f9dc45f8a7e953a04b912b49f52ac6d409b112e18eda5682bfecdcd0ddaf0da
CatalogFacetStats        sha256:232dd59497d1f2f8641f971059b2fe8d18c200c4d8b031b2232fd8993480146c
bundle manifest SHA-256  8f8c9c48ad010cccec07d5f081a87bbbfa4fc0b7e53d403c2dc1af7ed4abab9a
```

All 50,000 products have a present top-level price source: 10,415 evidence
rows are `VALID`, 39,473 are `EMPTY`, and 112 are `INVALID`. Resolution yields
10,415 `KNOWN`, 39,585 `UNKNOWN`, zero `CONFLICT`, and zero `NOT_APPLICABLE`
price states. The physical index therefore stores only 10,415 rows.

The matching primitive keeps `UNKNOWN` products and drops only prices proven
to be `VIOLATED`. This prevents missing prices from becoming false negatives.
Gate B now authorizes conservative price use, but the retrieval, session,
Probe, grounding, and official-adapter consumers remain disabled until CS5
implements and validates them against the approved artifact.

## CS4 Gate-B `price` review packet

CS4 now generates the evidence needed for an owner decision, but it does not
make that decision. Build and independently revalidate the ignored packet with:

```powershell
catalog-facet-gate-b-review build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl `
  artifacts/catalog-semantic/gate-b-review

catalog-facet-gate-b-review validate `
  artifacts/catalog-semantic/gate-b-review `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl
```

The packet contains:

| File | Purpose |
| --- | --- |
| `price-review-proposal.json` | Exact-scope decision and four-boolean capability proposal, still marked `awaiting_owner_approval` |
| `public-target-audit.json` | All 200 public target price states and safe-retention results |
| `candidate.json` | Compact input pins and explicit false runtime/approval flags |
| `report.md` | Plain-language owner review and approval question |
| `bundle-manifest.json` | Canonical artifact sizes, hashes, input identities, and false publication flags |

The verified proposal covers all 15 exact category scopes. Each row proposes
`RUNTIME_ACCEPT` with intent, retrieval, and Probe enabled, while proactive
price clarification remains disabled. The proposed intent boundary is integer
`USD_CENT` through `usd_cent_int_v1`; numeric price has no reviewed value
aliases. These are recommendations only.

On the official public set, 178 target products have a known price and 22 have
an unknown price. The conservative rule retains all 200 under a compatible
synthetic budget, whereas a known-and-satisfied-only rule retains 178. This
proves the missing-price false-negative hazard, but the toy set contains no
real user budget request and therefore does not prove that proactively asking
for a budget is useful.

```text
GateBPriceReviewProposal sha256:ec064460c7bc0c64862978ced376d79a699f20a9891f1cc2ce3ce0bce91995f6
PublicTargetPriceAudit   sha256:884331876d343b4d317f93527825e1fd2156fa7a3aff9af27b39d6e0e0959a56
```

The review packet remains a proposal artifact and never grants authority by
itself. Approval is recorded separately in source control and materialized as
described below.

## CS4 approved Gate-B capability candidate

The repository owner approved the exact 15-row proposal on 2026-08-28.
[`gate-b-selection.json`](../../../config/catalog_semantic/v0/gate-b-selection.json)
pins the proposal, public audit, CS3 index/statistics, integer-cent intent
normalizer, empty numeric alias list, and every exact-scope permission row.
It may be built and revalidated with:

```powershell
catalog-facet-gate-b build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl `
  artifacts/catalog-semantic/gate-b-review `
  config/catalog_semantic/v0/gate-b-selection.json `
  artifacts/catalog-semantic/gate-b-candidate

catalog-facet-gate-b validate `
  artifacts/catalog-semantic/gate-b-candidate `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl `
  artifacts/catalog-semantic/gate-b-review `
  config/catalog_semantic/v0/gate-b-selection.json
```

The candidate contains:

| File | Purpose |
| --- | --- |
| `effective-facet-capabilities.json` | Normative `EffectiveFacetCapabilitySet` with all 15 independent exact-scope rows |
| `reviewed-gate-b-selection.json` | Canonical copy of the source-controlled owner decision |
| `candidate.json` | Approval and artifact IDs plus `runtime_integration_complete=false` |
| `report.md` | Human-readable approved permissions and remaining CS5 boundary |
| `bundle-manifest.json` | Exact artifact hashes and all upstream identities |

```text
GateBSelection                sha256:2463fce33b253371fb50845008002cbd092096befb80ba103f6be0f6ba2e7dce
EffectiveFacetCapabilitySet   sha256:27785224230795094693de55ff820f63aae2c206f8374fc6b2e43d6b23ffbaae
```

Approval means an explicit budget may be grounded, conservative retrieval may
consume price, and Probe may inspect price in every published exact scope.
Proactive budget clarification remains false. The candidate still does not
install a session-context `FacetRegistry`, parse user language, change the
retriever, generate beliefs, or call the official adapter; those are CS5.

## CS5A runtime registry and numeric lexicon

CS5A consumes the approved Gate-B artifact and produces the first runtime
projection. It can be built and exactly revalidated with:

```powershell
catalog-runtime-projection build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl `
  artifacts/catalog-semantic/gate-b-review `
  config/catalog_semantic/v0/gate-b-selection.json `
  artifacts/catalog-semantic/gate-b-candidate `
  artifacts/catalog-semantic/runtime-projection-candidate

catalog-runtime-projection validate `
  artifacts/catalog-semantic/runtime-projection-candidate `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl `
  artifacts/catalog-semantic/gate-b-review `
  config/catalog_semantic/v0/gate-b-selection.json `
  artifacts/catalog-semantic/gate-b-candidate
```

The ignored candidate contains:

| File | Purpose |
| --- | --- |
| `runtime-facet-registry.json` | Declarative `price` plus reserved `system_product_category` FacetSpec records |
| `runtime-value-lexicon.json` | P0 numeric `price` domain: integer `USD_CENT` only |
| `candidate.json` | Runtime artifact IDs plus true grounding and false retrieval/gateway flags |
| `report.md` | Implemented boundaries, normalizers, and deliberate non-goals |
| `bundle-manifest.json` | Exact hashes and complete upstream approval identities |

```text
RuntimeFacetRegistryArtifact sha256:3bf926c022b99c43f6205c8dabc791bdb6f3cc30bab0d1368d1bf50160504362
RuntimeValueLexicon          sha256:17b849fa9ae5eab5e03cc1d8efa06f02f25d8512fb81cc1e4e852679cdeb0868
```

`usd_cent_int_v1` accepts only non-boolean signed I-JSON-safe integers and is a
fixed point for all accepted values. It rejects strings, floats, booleans, and
out-of-range integers; natural-language currency parsing remains upstream.
`category_scope_id_v1` is release-bound and accepts only IDs in the exact
CategoryRegistry used to build the artifact.

The runtime loader resolves both declarative records into a real session-context
`FacetRegistry`. Exact capability lookup returns all-false permissions for a
missing `(facet_id, category_scope_id)` row and never performs scope inheritance.
## CS5B deterministic grounding

CS5B adds the narrow handoff that Query Understanding will call later. It
accepts an `ExtractedRuntimeValueCandidate` containing an already selected
facet, operator, integer-cent value (or alternative values), and semantic
fallback text. `RuntimeValueGrounder` then:

1. recognizes only a facet in the projected registry;
2. verifies the final CategoryScope and its exact Gate-B row for ordinary facets;
3. checks the operator and applies the pinned closed value normalizer;
4. returns one canonical `RuntimeValueGroundingResult`.

The result is `GROUNDED`, `SEMANTIC_ONLY`, or `AMBIGUOUS`. Price equality is
expanded deterministically to inclusive `GE` followed by `LE`. Alternative
values are normalized, deduplicated, and sorted, and one invalid alternative
fails closed instead of being silently discarded. Reserved category grounding
accepts only `EQ` with one published CategoryScope ID.

The bundle loader `load_runtime_value_grounder` constructs this service only
after loading mutually pinned CS1, CS4, and CS5 artifacts. Until CS6 exists,
this remains a verified candidate-artifact assembly rather than a named final
release.

This stage still does not parse user language, allocate a Preference ID, call
the reducer, rank products, create a SearchBelief, ask a question, or modify
Session Context.

## CS6 immutable release

CS6 converts the previously separate candidate directories into one
self-contained runtime release. Build it with:

```powershell
catalog-semantic-release build `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-candidate `
  artifacts/catalog-semantic/gate-a-candidate `
  artifacts/catalog-semantic/resolution-candidate `
  data/public_set.jsonl `
  artifacts/catalog-semantic/gate-b-review `
  config/catalog_semantic/v0/gate-b-selection.json `
  artifacts/catalog-semantic/gate-b-candidate `
  artifacts/catalog-semantic/runtime-projection-candidate `
  artifacts/catalog-semantic/release-v0

catalog-semantic-release validate artifacts/catalog-semantic/release-v0
```

The release contains exactly the contract's 13 artifacts plus
`catalog-semantic-release.json`. The raw `catalog.jsonl` member is an exact byte
copy: it is never parsed and rewritten, and the original file remains
unchanged. The other members are the exact canonical semantic artifacts plus
`reviewed-semantic-config.json`, which is deterministically projected from all
human-reviewed scopes, facets, bindings, capabilities, runtime normalizers,
and aliases.

Every member records its exact SHA-256 content ID, schema, and byte size. The
external release ID is the SHA-256 content ID of the canonical manifest. The
loader independently verifies the release ID, all 13 members, the 50k raw
catalog format, category graph and assignments, evidence/index/statistics,
all artifact cross-references, closed implementation IDs, runtime projection,
and the CS5B grounder. Missing, additional, reordered, stale, or modified
content fails before a runtime object is returned.

The current verified P0 generation is:

```text
CatalogSemanticRelease sha256:325af3fea978f34116ffec7ea13a1cbf1edd654ee3457c5877b51a3ec92a8907
ReviewedSemanticConfig sha256:b459163b695c642114a2ff10152d837161df4be1308e59362600c0d502154d3d
artifacts                13
reviewed scopes          15
ordinary facets          1 (price)
exact capability rows    15
```

Publication happens in a temporary generation directory. The completed
directory becomes visible only after the self-contained loader accepts it.
An existing release directory is immutable: rebuilding identical inputs reuses
it, while different or damaged content is rejected rather than overwritten.

CS6 does not create sessions or provide migration. CS7 binds one verified
release to one gateway/store boundary.

## CS7 catalog-bound session gateway

CS7 is the only application write authority for catalog-sensitive session state.
It leaves the session-context v1 reducer and codec unchanged, then adds the
catalog rules they intentionally do not know about:

- `CatalogSemanticGateway` previews a batch through the unchanged reducer and
  accepts the result only when the final exact category, capabilities, values,
  don't-care set, and active SearchBelief all agree with the bound release;
- reserved `system_product_category` state is zero or one `EQ` Preference over
  a published scope ID, and explicit category writes use only one correctly
  ordered `ReplaceFacet`;
- a category change that leaves an inapplicable structured preference or
  don't-care facet rejects the complete batch without automatic repair;
- the application session registry additionally contains the explicit
  `retrieval_derived` competition facets. They receive shape and canonical-form
  validation here but do not become Catalog Semantic release artifacts or
  claim catalog capability;
- `CatalogBoundSessionStore` privately owns the projected registry and raw
  `InMemorySessionStore`; gateway validation and raw commit run while the same
  per-session transaction lock remains held;
- a new or changed live SearchBelief needs a process-local, one-use token bound
  to the exact release, session, transaction, captured context, final intent,
  and belief. CS8's private Probe producer will be the only production issuer;
- `shopping-copilot/catalog-bound-session/v0` wraps the unchanged v1 snapshot
  with the release ID, exact inner SHA-256, and strict unpadded base64url.
  Decode verifies the envelope in order and replays every accepted update
  through the Gateway.

The public entry point is:

```python
from shopping_copilot.catalog.semantic import CatalogBoundSessionStore

store = CatalogBoundSessionStore(verified_release)
context = store.reset(session_id="session-1")

with store.turn(session_id=context.session_id, turn=1) as transaction:
    captured = transaction.context
    previewed_intent = transaction.preview_update(query_understanding_batch)
    # Query Compiler, Probe, and response generation construct next_context.
    transaction.commit(next_context)
```

The wrapper exposes no public raw commit or projected-registry handle. A pure
Gateway preview helps a trusted coordinator plan the next context, but it grants
no commit authority; the same checks always rerun inside the bound transaction.
Sessions are not silently migrated between releases, and separate releases
require separate wrapper stores.
