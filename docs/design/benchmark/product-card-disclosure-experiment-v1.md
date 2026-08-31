# Grounded Product-Card Disclosure Experiment v1

Status: completed diagnostic experiment on 20 public sessions.

## Question

Can the benchmark disclose source-grounded product-card facts instead of four mechanically selected strings, while the real pipeline consumes those facts through QU, Session Context, intent transparency, multi-route retrieval, BGE reranking, and DPP selection?

The 20-session slice contains five sessions from each scenario and all five previously inspected hard-mask failures. Every assistant question remains `ask_attribute=other`.

## What changed

- The customer side replayed the reviewed, grounded conversations from `artifacts/benchmark/product-card-disclosure-review-v1/conversations.jsonl`.
- Covered target products used replacement product cards in lexical documents, the Dense index, BGE input, and DPP vectors.
- The new Dense index used its own catalog-density cache, so `T_t` was never computed with statistics from the old vector space.
- The target ASIN remained evaluator-only and was not passed into QU or retrieval.
- Sessions continued after their first hit so the experiment could detect memory corruption and target disappearance.

This is a diagnostic A/B, not an isolated causal comparison. The legacy run used both legacy four-value disclosure and the old retrieval corpus; the new run changed disclosure and target-card representation together by design.

## Results

| Metric | Legacy same-20 baseline | Grounded product-card run |
| --- | ---: | ---: |
| Sessions that ever hit Top 10 | 11/20 | 20/20 |
| Hit rate | 55.0% | 100.0% |
| MRR | 0.394 | 1.000 |
| Mean first-hit turn among hits | 2.45 | 2.05 |

Scenario results for the grounded run:

| Scenario | Sessions | Hit rate | Mean first-hit turn |
| --- | ---: | ---: | ---: |
| Buying | 5 | 100% | 1.0 |
| Browsing | 5 | 100% | 1.6 |
| Intent override | 5 | 100% | 3.0 |
| Boundary | 5 | 100% | 2.6 |

All nine samples missed by the legacy same-20 baseline were recovered: `public_0002`, `public_0016`, `public_0041`, `public_0045`, `public_0098`, `public_0137`, `public_0154`, `public_0169`, and `public_0199`.

## Integrity checks

- QU accepted 130/132 turns (98.5%).
- A reproducible lexical lower bound found 178/182 disclosed facts in the resolved QU payload (97.8%). The four misses are exactly the two rejected QU turns, not silent omissions on successful turns.
- Once first found, the target remained in Top 10 on 109/111 later visible turns (98.2%).
- The target was present on the final scripted turn in 19/20 sessions (95.0%).
- No target loss was caused by a hard mask in this slice.

## Intent-transparency behavior

Mean applied `T_t` increased as facts accumulated:

| Scenario | Turn 1 | Turn 3 | Turn 7 |
| --- | ---: | ---: | ---: |
| Buying | 0.167 | 0.367 | 0.519 |
| Browsing | 0.079 | 0.390 | 0.623 |
| Intent override | 0.154 | 0.262 | 0.468 |
| Boundary | 0.099 | 0.197 | 0.508 |

This is descriptive evidence, not a monotonicity requirement. A real user can change direction, and `T_t` is allowed to move sharply in either direction.

## Observed failures

### `public_0016` turn 1: product-type projection

The generated initial message used `Mid-Calf` as if it were a product category. QU correctly declined to turn that attribute-like phrase into a searchable goal, leaving the compiled query not search-ready. The product-card projector needs a category sanity check and a title/category fallback.

### `public_0038` turn 5 and `public_0098` turn 2: strict material execution

The user messages contained `Synthetic`, `100% Rubber`, and `50% ... recycled content`, but end-to-end QU materialization rejected the entire turn with `material value contains no executable keyword`. This is the strict-validation failure mode already suspected in design review: one difficult fact discards another plainly executable fact in the same utterance.

The next implementation should preserve the executable keyword (`synthetic` or `rubber`) and retain recycled-content wording as semantic evidence instead of failing the whole turn.

### `public_0016` turns 6-7: ranking, not recall

The target was rank 1 in Dense, lexical, facet, and fused retrieval. BGE moved it to rank 17, after which DPP did not select it into the final Top 10. Therefore:

- recall and hard-mask logic were healthy;
- the final-turn miss was introduced by reranking;
- route protection or a bounded fusion-prior term should be considered before scaling the benchmark.

## Runtime

- New catalog-density cache on RTX 4070 Ti: approximately 1.1 seconds.
- One-time model and corpus initialization: 115.8 seconds.
- Twenty sessions / 132 turns with eight concurrent sessions: 595.0 seconds.
- Single-session smoke test: 15.5 seconds for two turns, about 7.8 seconds per turn.
- Reported QU usage for the full run: 679,393 tokens.

The 32-second mean recorded inside the eight-worker run includes GPU-lock contention and is not a valid estimate of single-user latency. The single-session smoke run is the relevant current measurement.

## Reproduce

```powershell
.\.venv-3.10\Scripts\python.exe scripts/simulator/evaluate_full_pipeline_other.py `
  --dense-index artifacts/retrieval/dense-public-200-replaced-v1 `
  --density-cache artifacts/retrieval/intent-volume-density-public-200-replaced-v1.npz `
  --scripted-conversations artifacts/benchmark/product-card-disclosure-review-v1/conversations.jsonl `
  --max-turns 7 `
  --continue-after-hit `
  --workers 8 `
  --output-dir artifacts/simulator/product-card-disclosure-20-v1
```

The machine-readable comparison is `artifacts/simulator/product-card-disclosure-20-v1/comparison.json`; complete QU, Session Context, `T_t`, retrieval, ranking, and target-rank evidence is in `turns.jsonl` beside it.
