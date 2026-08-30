# Formal Retrieval: how the system builds a product candidate pool

This document explains recall only. Ranking is a later stage.

## 1. The whole path

```text
User language
  -> Query Understanding updates Session Context
  -> Query Compiler produces q_sem, q_lex, and structured constraints
  -> Intent Volume produces T_t
  -> hard mask removes confirmed violations
  -> T-aware multi-center Dense recall
  -> Lexical and Facet recall
  -> merge into a candidate pool of up to 300 products
  -> ranking later chooses the displayed Top-10
```

There is no separate Buying search engine and Browsing search engine. The same
retrieval system changes continuously with `T_t`.

## 2. What recall is responsible for

Recall does not decide which ten products are best. Its job is to make sure the
later ranker receives the right search space:

- vague intent: several different but still relevant shopping directions;
- clear intent: fewer directions and deeper candidates inside them;
- every case: confirmed hard violations stay out.

The pool target stays at 300. `T_t` changes how spread out those products are,
not merely how many listings the system returns.

## 3. Hard conditions happen first

Examples are explicit exclusions, verified category scope, price, color, material,
or another condition compiled as hard evidence.

- A product known to violate a condition is removed before any Top-K operation.
- Missing evidence is not treated as a confirmed violation.
- A positive hard condition that would empty the entire catalog can be relaxed to
  ranking evidence, and that relaxation is recorded.
- Explicit exclusions are never relaxed.

`T_t` cannot override these rules.

## 4. The three recall routes

### Dense

Dense recall reads the overall meaning of `q_sem`. It is useful for scenario-like
requests such as "I am going to Hokkaido; show me things I might need."

The system scores the query against all 50k product vectors once. It keeps a wide
Top-2,000 frontier only for finding different semantic regions. That frontier is
not the final candidate pool.

### Lexical

Lexical recall reads explicit words in `q_lex`, such as a brand, model, color,
material, or `waterproof`. It uses the catalog text through SQLite FTS5.

### Facet

Facet recall consumes facts Query Understanding has already structured, for example:

```json
{"facet": "color", "operator": "eq", "value": "red"}
```

It does not reread the conversation or invent another interpretation. If there is
no positive structured evidence, this route is honestly marked unavailable.

## 5. How T_t changes Dense recall

The maximum number of semantic directions is:

```text
directions = 1 + round((1 - T_t) * 5)
```

| T_t | Maximum directions | Meaning |
| ---: | ---: | --- |
| 0.0 | 6 | explore several plausible shopping regions |
| 0.5 | 3--4 | balance exploration and focus |
| 1.0 | 1 | search deeply inside one region |

The first direction center is the product most similar to the query. Later centers
must satisfy both conditions:

1. still sufficiently related to the query;
2. sufficiently different from all centers already selected.

Centers are real products selected only from eligible catalog vectors. The system
does not ask an LLM to invent labels such as "hat" or "boots", and it does not use
category quotas.

For a vague Hokkaido request, the real experiment found regions represented by
cold-weather accessories, snow clothing, winter footwear, travel gear, and outdoor
gloves. For a focused waterproof snow-boot request, the hard mask left 29 products
and recall used one footwear direction.

## 6. How products are taken from each direction

Every direction produces its own list. The product score combines:

- similarity to the original query;
- similarity to that direction center.

At low `T_t`, the original query protects relevance while several directions are
explored. At high `T_t`, the selected center receives more weight so retrieval
deepens inside the narrow region.

Products are taken round-robin from the direction lists. A single global Top-K is
not allowed at this point because one popular direction could erase every smaller
direction.

## 7. Route budgets

The planned 300 slots change continuously:

| T_t | Multi-center Dense | Lexical | Facet |
| ---: | ---: | ---: | ---: |
| 0.0 | 210 | 45 | 45 |
| 0.5 | 180 | 60 | 60 |
| 1.0 | 150 | 75 | 75 |

Low transparency spends more budget on semantic exploration. High transparency
gives more budget to exact words and structured evidence.

Routes can overlap or be unavailable. The system deduplicates their union and
refills missing slots from the Dense direction lists. If fewer than 300 products
survive the hard mask, the result is allowed to remain smaller.

## 8. What is logged

Every search result records:

- requested and actual direction counts;
- the real product chosen as each direction center;
- planned and actual route counts;
- how many Dense products were used to refill route overlap;
- the direction that admitted every Dense candidate;
- hard-mask decisions;
- time spent in hard mask, Dense scoring, center planning, Lexical, Facet, fusion,
  retained ranking, and the complete retrieval call.

These fields are enough for a demo to show where the system is exploring and for
an engineer to reproduce a surprising result.

## 9. Real 50k result and time

The old single-center Top-80 and new policy were run on the same six queries.

- Hokkaido, `T=0.10`: directions `1 -> 6`, broad product groups `4 -> 7`, candidate
  pair cosine `0.754 -> 0.728`.
- Summer wedding, `T=0.20`: directions `1 -> 5`, product groups `7 -> 11`.
- New office job, `T=0.25`: directions `1 -> 5`, product groups `10 -> 16`.
- Focused snow boots, `T=0.90`: both policies stay in one footwear group.

With the Hokkaido query held fixed and only `T_t` changed from 0 to 1, the actual
direction count was `6, 5, 4, 2, 1`. This is the important causal result: `T_t`
now changes recall itself.

On the development machine:

- one-time initialization was about 95 seconds;
- warm complete retrieval was about 0.10--0.42 seconds per query;
- multi-center planning added about 13 ms for one direction and 39--53 ms for
  three to six directions.

The controller must be initialized once and reused. Rebuilding the model, text
index, and Facet evidence every turn is not a supported runtime path.

## 10. Current boundary

- The retrieval controller still carries its legacy MMR Top-10 for backward-compatible
  callers. The new demo path consumes the 300-product fused pool through BGE,
  DeepSeek quality judgement, and a separate T-aware DPP finalizer.
- Direction centers can have noisy catalog titles. Recall only guarantees a useful
  candidate space; ranking still judges individual products.
- The Top-2,000 frontier and center thresholds are hackathon parameters.
- The next recall integration issue is how degraded or unavailable `D_t` should
  choose a neutral fallback policy.

Ranking details are explained in [06-ranking.md](06-ranking.md).

Detailed contract and complete numbers:

- [Transparency-aware recall design](../design/retrieve/transparency-aware-recall-v1.md)
- [50k evaluation report](../../artifacts/retrieval/transparency-recall-evaluation-v1.md)
