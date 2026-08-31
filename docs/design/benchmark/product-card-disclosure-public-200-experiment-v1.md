# Grounded Product-Card Disclosure: Public 200 Experiment v1

Status: completed on all 200 public simulator sessions.

## Purpose

This experiment replaces the benchmark's legacy four mechanically selected hidden strings with bounded, source-grounded disclosures derived from the same product-card representation used by the real retrieval pipeline.

All 200 target products use replacement product cards in:

- customer fact disclosure;
- lexical retrieval documents;
- Dense embeddings;
- BGE reranking documents;
- DPP diversity vectors.

The target ASIN remains evaluator-only. The agent receives only the user message, prior Session Context, and shown-product history. Every clarification uses `ask_attribute=other`.

This is an intentional representation-alignment test, not an isolated causal A/B. The old run used both legacy disclosure and the old retrieval corpus; the new run changes disclosure and target-card representation together.

## Dataset

- Sessions: 200
- Buying: 80
- Browsing: 80
- Intent override: 30
- Boundary: 10
- Grounded disclosure facts available: 1,805
- Mean facts per card: 9.025
- Minimum / maximum facts: 4 / 10
- Scripted user turns available: 1,313
- DeepSeek calls used to build the disclosure packet: 0

The original catalog and public dataset are not modified. Product cards, conversations, Dense vectors, and density statistics are derived sidecars or experiment artifacts.

## Main result

| Metric | Legacy public-200 run | Grounded product-card run |
| --- | ---: | ---: |
| Hit@10 | 78.0% | 100.0% |
| Sessions hit | 156/200 | 200/200 |
| MRR | 0.419 | 0.755 |
| Official MTTC | 4.045 | 1.805 |
| Suggested technical score | 0.655 | 0.910 |

The legacy run's MTTC includes the evaluator's penalty for missed sessions. Among legacy sessions that did hit, the mean first-hit turn was 2.08.

All 44 targets missed by the legacy run were recovered.

## Scenario result

| Scenario | Sessions | Hit@10 | MRR | Mean first-hit turn |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 100% | 0.781 | 1.213 |
| Browsing | 80 | 100% | 0.644 | 1.838 |
| Intent override | 30 | 100% | 0.939 | 3.000 |
| Boundary | 10 | 100% | 0.883 | 2.700 |

Intent-override sessions are not scored before the scripted change of intent on turn 3.

## Hit distribution

| First-hit turn | Sessions |
| ---: | ---: |
| 1 | 89 |
| 2 | 62 |
| 3 | 48 |
| 4 | 1 |

No session required turns 5-7. The only turn-4 first hit was `public_0187`, a Boundary session.

Final hit ranks:

| Rank | Sessions |
| ---: | ---: |
| 1 | 128 |
| 2 | 23 |
| 3 | 14 |
| 4 | 14 |
| 5 | 7 |
| 6 | 5 |
| 7 | 4 |
| 8 | 2 |
| 9 | 1 |
| 10 | 2 |

## QU and fact integrity

- QU/materialization accepted 355/361 executed turns: 98.3%.
- A reproducible lexical audit found 376/386 disclosed facts in the resolved QU payload: 97.4%.
- Nine of the ten lexical misses occurred on rejected turns.
- One successful turn silently omitted a fact: `public_0168` turn 3 retained the brand but omitted the explicit style value `4-Heart`.
- Every session still hit because another fact or later turn supplied enough evidence.

### Material execution failures

Six turns were rejected with `material value contains no executable keyword`:

| Sample | Turn | Relevant text |
| --- | ---: | --- |
| `public_0026` | 1 | `100% Synthetic` |
| `public_0048` | 2 | `rubber`; `100% Synthetic` |
| `public_0094` | 1 | `Synthetic` |
| `public_0128` | 2 | `100% recycled materials` |
| `public_0151` | 2 | `synthetic` |
| `public_0186` | 3 | `100% Synthetic` |

This confirms the earlier 20-session diagnosis. A difficult material normalization can reject the entire turn, including a second plainly useful fact. The correct fix is atomic degradation: retain executable keywords such as `rubber` or `synthetic`, keep unsupported composition wording as semantic evidence, and never discard unrelated clauses from the same utterance.

### Non-searchable broad category

Two first turns compiled to no executable query:

- `public_0062`: `I'm looking for Casual, but I'm still exploring.`
- `public_0073`: `I'm looking for Casual, but I'm still exploring.`

Both hit on turn 2 after grounded facts arrived. Product-type projection should prefer `casual dress`, `casual skirt`, or another category-bearing phrase instead of the attribute-only leaf `Casual`.

## Intent transparency

Mean applied `T_t` among the turns that were actually executed:

| Scenario | Turn 1 | Turn 2 | Turn 3 | Turn 4 |
| --- | ---: | ---: | ---: | ---: |
| Buying | 0.157 | 0.239 | 0.335 | - |
| Browsing | 0.098 | 0.209 | 0.257 | - |
| Intent override | 0.150 | 0.244 | 0.255 | - |
| Boundary | 0.106 | 0.108 | 0.207 | 0.312 |

The values move in the intended direction as more facts arrive. Because evaluation stops at the first hit, later-turn means contain only the harder surviving sessions and must not be treated as a controlled convergence curve. The 20-session continue-after-hit experiment remains the cleaner measurement for that question.

## Runtime and cost

- One-time model and corpus initialization: 116.8 seconds.
- Evaluation wall time: 1,646.2 seconds (27 minutes 26 seconds).
- End-to-end wall time including initialization: approximately 29 minutes 23 seconds.
- Mean throughput cost: 8.23 wall-clock seconds per completed session with eight concurrent sessions.
- Executed turns: 361.
- Reported QU tokens: 1,744,757 total.

Per-turn timing recorded inside the eight-worker run includes time waiting for the serialized local GPU ranking section, so it is not a single-user latency measurement. The earlier one-session smoke test remains approximately 7.8 seconds per user turn.

## Artifacts

- Disclosure packet: `artifacts/benchmark/product-card-disclosure-public-200-v1/`
- Full run: `artifacts/simulator/product-card-disclosure-public-200-v1/`
- Machine-readable comparison: `comparison.json`
- Complete turn evidence: `turns.jsonl`
- Session outcomes: `sessions.jsonl`
- Frozen run configuration: `run.json`

## Conclusion

The aligned product-card design works across the complete public target set and removes the large representation mismatch seen in the legacy simulator. The 100% result is meaningful as an engineering consistency test, but it must be described honestly: the simulator disclosures and covered retrieval cards share the same grounded fact source.

Before scaling the policy beyond this public diagnostic set, the concrete fixes are:

1. make material execution degrade atomically instead of rejecting a complete turn;
2. normalize attribute-only product types such as `Casual` into category-bearing phrases;
3. require QU to account for every explicit clause, including semantic fallback facts such as `4-Heart`;
4. retain the legacy and grounded modes as an explicit benchmark A/B rather than replacing provenance silently.
