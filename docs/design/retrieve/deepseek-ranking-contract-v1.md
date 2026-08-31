# DeepSeek Candidate Ranking Contract v1

Status: implemented experimental contract for the real-system demo path.

## 1. Scope

This contract begins after hard eligibility and multi-route recall. It ends with a
quality-ranked candidate list and an optional T-aware DPP Top-10. It does not modify
Session Context, discover facets, or implement the toy-simulator scoring branch.

## 2. Pipeline

```text
eligible multi-route candidate pool (normally 300)
  -> pinned BGE-reranker-v2-m3
  -> direction-protected shortlist (normally 48)
  -> one DeepSeek V4 Flash native tool call
  -> exact local validation
  -> 0.8 DeepSeek fit + 0.2 BGE relevance
  -> T-aware greedy DPP Top-10
```

Hard eligibility is never reopened by ranking.

## 3. BGE shortlist invariant

The shortlist is generated in two passes:

1. assign every BGE-ranked product to its nearest multi-center recall direction;
2. protect the top `protected_per_direction` products from every direction, then fill
   remaining slots from global BGE order.

The retained cards remain in BGE order. Default values are `shortlist_k=48` and
`protected_per_direction=6`. If the hard mask leaves fewer products, all survivors
may be passed onward.

## 4. Model request

`DeepSeekRankingRequest` contains:

- stable `request_id`;
- exact resolved `IntentState`;
- matching-version `CompiledQuery`;
- `RankingShortlist`;
- optional `RankingUserProfile`.

`RankingUserProfile` is an opaque, recursively immutable, JSON-compatible envelope:

```json
{
  "schema": "shopping-copilot/user-profile/...",
  "version": 1,
  "payload": {}
}
```

The ranking boundary does not own or interpret the profile schema. Prompt precedence
is normative: explicit current Session Context overrides long-term profile evidence.

## 5. Model-visible product evidence

Product documents retain bounded title, category path, store, features, details, and
description fields. The prompt exposes only candidate ID and product evidence.

The following ranking anchors are deliberately hidden from DeepSeek:

- BGE score;
- BGE rank or shortlist rank;
- recall route;
- semantic direction ID;
- `T_t`;
- final diversity policy.

Candidate presentation order is a deterministic SHA-256 permutation of request ID and
candidate ID.

## 6. Native tool output

DeepSeek is forced to call `submit_candidate_judgements` exactly once. Tool arguments
contain one item per supplied candidate with:

- `candidate_id`;
- integer `fit_score` in `[0,100]`;
- band-consistent `verdict`;
- matched, unsupported, and conflict current-preference ID arrays;
- short concerns and reason.

The three preference arrays are mutually exclusive. Unknown evidence is unsupported;
explicitly incompatible evidence is conflict. No array may refer to profile fields or
invented preference IDs.

## 7. Local acceptance and recovery

The decoder requires unique-key JSON, the exact field set, the exact candidate set,
known preference IDs, unique arrays, disjoint judgement groups, and consistent score
bands. Decoder failures do not mutate any state.

Repairable response failures receive one complete-call retry. Provider availability,
authentication, timeout, or second-attempt failure produces a BGE fallback result.
Fallbacks are explicit in `QualityRankingResult.mode` and `fallback_reason`.

## 8. Individual quality

For an accepted judgement:

```text
deepseek_fit = fit_score / 100
quality = 0.8 * deepseek_fit + 0.2 * bge_relevance
```

Sorting is descending quality with candidate ID as a deterministic tie-break. The
DeepSeek weight is configurable, but `0.8` is the v1 default.

## 9. Final slate

The quality list is passed to greedy DPP. `T_t` enters only here:

```text
w_rel(T_t) = 0.30 + 0.60 * T_t
```

A request-level diversity directive shifts the relevance weight by `0.10`, clamped to
`[0,1]`. Low `T_t` therefore emphasizes set diversity; high `T_t` emphasizes individual
quality. DeepSeek is not asked to simulate this set objective.

## 10. Observability

The evaluation log contains full Session Context intent, compiled query, optional
profile envelope, complete shortlisted product cards, all accepted judgements, BGE and
DeepSeek scores, final DPP slate, model token usage, attempts, fallback reason, and
stage timings. Accepted calls include provider-reported token usage; a rejected tool
response may lack usage after the provider raises its typed error. Credentials are
never included.

## 11. Implementation map

- contracts: `src/shopping_copilot/retrieval/deepseek_ranking/models.py`
- shortlist: `shortlist.py`
- prompt: `prompt.py`
- tool schema and decoder: `wire.py`
- provider: `deepseek.py`
- repair/fallback and fusion: `service.py`
- BGE-to-DeepSeek pipeline: `pipeline.py`
- final DPP: `slate.py`
- real-world orchestration and fallbacks:
  `src/shopping_copilot/application/quality_ranking.py`
- real-catalog runner: `scripts/retrieval/evaluate_deepseek_ranking_v1.py`
- unit tests: `tests/unit/retrieval/test_deepseek_ranking.py`
