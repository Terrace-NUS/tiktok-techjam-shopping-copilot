# Transparency-aware Multi-center Recall v1

- Status: implemented and evaluated on the bound 50k catalog.
- Runtime: `src/shopping_copilot/retrieval/transparency_recall.py`.
- Controller integration: `src/shopping_copilot/retrieval/controller.py`.
- Reproducible evaluation: `scripts/retrieval/evaluate_transparency_recall_v1.py`.
- Latest report: `artifacts/retrieval/transparency-recall-evaluation-v1.md`.

## 1. Contract

`T_t` controls the geometric breadth of candidate recall. It does not classify a
user as Buying or Browsing, change confirmed hard constraints, or merely change
the number of returned listings.

The candidate pool target stays at 300. A low `T_t` spreads its dense candidates
across more catalog-grounded semantic directions. A high `T_t` spends its dense
budget more deeply around fewer directions.

```text
CompiledQuery + T_t
  -> resolve hard mask
  -> score q_sem against the complete catalog once
  -> retain a wide semantic frontier
  -> select T-dependent direction centers
  -> recall round-robin around those centers
  -> add T-dependent Lexical and Facet budgets
  -> refill overlap from the dense direction reserve
  -> RRF candidate pool
```

Ranking is deliberately outside this contract. The existing MMR remains attached
after recall only so the current application continues to return Top-10 results.

## 2. Invariants

1. Confirmed hard exclusions are applied before the frontier and before every
   route truncation.
2. Unknown product evidence is not treated as a confirmed violation.
3. The algorithm never reads a target product or simulator hidden state.
4. Direction centers are real eligible catalog products, not LLM-invented labels.
5. Product category is not used to select centers or enforce diversity.
6. The same normalized product vectors used by Dense retrieval define directions.
7. Every result records centers, route budgets, refill counts, and wall-clock time.

## 3. Policy

The first experimental policy is:

```text
candidate_pool_k = 300
frontier_k = 2,000
maximum_directions = 6
minimum_normalized_center_relevance = 0.35
maximum_center_similarity = 0.90
```

The requested direction count is:

```text
M(T) = 1 + round_half_up((1 - T) * 5)
```

This maps `T=0` to six directions and `T=1` to one direction. The observed
direction count can be smaller when the eligible frontier does not contain enough
separated, relevant centers. The implementation never fabricates directions to
meet the requested count.

## 4. Direction selection

The most query-relevant eligible product is the first center. Every later center
must remain relevant to the query and separated from all selected centers.

For each frontier product:

```text
center_score = relevance_weight * normalized_query_relevance
             + diversity_weight * normalized_distance_to_centers

relevance_weight = 0.40 + 0.50 * T
diversity_weight = 1 - relevance_weight
```

Candidates below the relevance guard or above the maximum center-to-center
similarity are not allowed to become a new center.

## 5. Dense expansion

For direction center `c_j`, each eligible product receives:

```text
direction_score(i, j)
  = alpha(T) * cosine(query, product_i)
  + (1 - alpha(T)) * cosine(center_j, product_i)

alpha(T) = 0.75 - 0.35 * T
```

This mapping was changed after the first real-catalog run. Increasing the query
weight at high `T` re-broadened a deliberately narrow direction whenever the raw
query text was broad. The tested mapping keeps broad exploration query-relevant at
low `T`, then deepens around the chosen center at high `T`.

Each direction has its own ordered list. Dense recall consumes these lists
round-robin. A global Top-K cut is forbidden here because it would allow one
popular direction to erase all smaller directions.

## 6. Route budgets

The 300 planned slots are distributed continuously:

```text
dense   = 150 + round(60 * (1 - T))
lexical =  45 + round(30 * T)
facet   = 300 - dense - lexical
```

| T | Dense | Lexical | Facet | Total |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 210 | 45 | 45 | 300 |
| 0.5 | 180 | 60 | 60 | 300 |
| 1.0 | 150 | 75 | 75 | 300 |

Routes can be unavailable or overlap. The controller preserves the planned route
hits, deduplicates their union, and refills missing unique slots round-robin from
the dense reserve. The final pool can remain below 300 only when fewer than 300
products survive the hard mask.

## 7. Audit output

`FormalRetrievalResult` now includes:

- `recall_trace`: requested and actual direction counts, real center products,
  route budgets, actual route counts, dense refill count, and direction provenance
  for every selected dense candidate;
- `timings`: hard mask, dense score, recall planning, Lexical, Facet, fusion,
  retained ranking, and total wall-clock milliseconds.

The policy is switchable. `RecallStrategy.LEGACY_SINGLE_CENTER` reproduces the old
single-query Top-K route, while `RecallStrategy.TRANSPARENCY_MULTI_CENTER` selects
the new behavior.

## 8. Real 50k observations

The evaluation compares the old Top-80 pool with the new pool on six queries and
then fixes the Hokkaido query while sweeping only `T`.

Representative results:

| Query | T | Old directions/groups | New directions/groups | Old pair cosine | New pair cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hokkaido winter trip | 0.10 | 1 / 4 | 6 / 7 | 0.754 | 0.728 |
| Summer wedding | 0.20 | 1 / 7 | 5 / 11 | 0.690 | 0.721 |
| New office job | 0.25 | 1 / 10 | 5 / 16 | 0.678 | 0.676 |
| Black waterproof snow boots | 0.90 | 1 / 1 | 1 / 1 | 0.726 | 0.726 |

The direction-count causal sweep is `6, 5, 4, 2, 1` for
`T = 0.00, 0.25, 0.50, 0.75, 1.00`. It proves that `T` now changes candidate
generation itself rather than only changing the final selector.

Pairwise cosine is not strictly monotone in the complete fused pool. Lexical and
Facet routes remain active, route overlap changes, and an artificially high `T`
does not rewrite a broad compiled query. Strict cross-turn monotonicity is not an
invariant: a user can remove constraints or change goals.

## 9. Time

On the recorded Windows/CUDA development machine:

- one-time controller initialization: about 95 seconds;
- warm full retrieval in the evaluated cases: about 0.10--0.42 seconds;
- multi-center planning itself: about 13 ms for one direction and 39--53 ms for
  three to six directions;
- the first dense matrix multiplication can be slower due to backend warm-up.

Initialization builds in-memory text and Facet indexes and loads the embedding
model. A serving process must build the controller once and reuse it. Per-turn
factory construction is outside the supported runtime design.

## 10. Known limits

- A real catalog center is easy to audit but can have a noisy title. Ranking must
  still decide whether an individual product is good enough for the user.
- The 2,000 frontier and center guards are hackathon parameters, not production
  calibration.
- Vector directions describe semantic regions; they do not guarantee one product
  from every human category.
- `D_t` health is not yet passed into the recall controller. The next integration
  should define the neutral policy for degraded or unavailable transparency.
