# 06: How ranking works

## One-sentence version

Recall finds a safe and broad candidate space. BGE removes obvious noise. DeepSeek
judges how well each remaining product fits the current user intent. Finally,
`T_t` decides whether the displayed set should explore or focus.

```text
up to 300 recalled products
  -> BGE relevance ranking
  -> 48 products, with every recall direction protected
  -> DeepSeek judges individual product fit
  -> quality = 0.8 * DeepSeek + 0.2 * BGE
  -> T-aware DPP selects the displayed Top-10
```

The four stages answer different questions. They should not be collapsed into one
opaque LLM prompt.

## 1. What BGE does

BGE is a local cross-encoder. Unlike embedding search, it reads the shopping query
and one complete product document together. It is good at saying whether that one
product is basically relevant.

Running an LLM over all 300 products would be slow and expensive, so BGE reduces the
pool to 48. The reduction is not a plain global Top-48: up to six products from every
semantic recall direction are protected first. This prevents a popular direction
such as snow boots from deleting gloves, coats, or other valid directions before the
LLM sees them.

## 2. What DeepSeek reads

DeepSeek receives one batch containing:

- the complete resolved `IntentState` from the current Session Context;
- the compiled semantic and lexical query;
- an optional versioned `user_profile` envelope;
- 48 compact product evidence cards.

It does not receive `T_t`, BGE scores, recall routes, direction IDs, or the original
candidate ranks. Hiding these values prevents the model from copying an earlier
ranking instead of reading the product evidence.

The candidate order is deterministically shuffled for each request. The same request
is reproducible, but position is not a quality signal.

## 3. Current intent versus long-term memory

The Session Context is authoritative. A future long-term user profile is supporting
evidence only.

For example, if the profile says that the user usually likes blue, but the current
session asks for a red wedding accessory, DeepSeek must judge red products against
the current request. This precedence is stated in the prompt and is represented in
the request contract. The long-term-memory schema itself is intentionally not frozen
yet; ranking accepts a generic versioned JSON envelope.

## 4. What DeepSeek outputs

DeepSeek is forced to call one native tool named
`submit_candidate_judgements`. It must return exactly one judgement for every input
candidate:

```json
{
  "candidate_id": "B07...",
  "fit_score": 82,
  "verdict": "strong_match",
  "matched_preference_ids": ["p_1_1_0"],
  "unsupported_preference_ids": ["p_1_1_1"],
  "conflict_preference_ids": [],
  "concerns": ["Size is not stated."],
  "reason": "The product matches the requested use case and color."
}
```

The score bands are fixed:

- `75-100`: `strong_match`;
- `40-74`: `possible_match`;
- `0-39`: `weak_match`.

Missing evidence is `unsupported`, not a conflict. An explicit incompatible value is
a conflict: size 13 conflicts with a current size 10 preference. Hard current-session
preferences matter more than soft ones.

DeepSeek judges individual fit only. It does not choose the final set and does not
reward diversity.

## 5. Why BGE still keeps 20 percent

The final individual quality score is:

```text
quality = 0.8 * DeepSeek_fit + 0.2 * BGE_relevance
```

DeepSeek supplies evidence-aware reasoning and understands the complete Session
Context. BGE supplies a stable local relevance anchor. The 80/20 split makes DeepSeek
decisive without making one model response the only numerical signal.

This is a hackathon policy value, not a learned optimum.

## 6. How failures behave

The local decoder checks that every candidate appears exactly once, preference IDs
are real, judgement groups are disjoint, and score bands agree with verdicts.

If the tool arguments fail those checks, the system retries once with a focused repair
instruction. If the second call fails, times out, is rate-limited, or is unavailable,
ranking falls back to BGE. Retrieval and the demo do not stop.

One real focused test produced an `invalid_judgements` response. After the repair
instruction explicitly restated the candidate, score-band, and disjoint-group rules,
the second call returned all 29 judgements successfully. This is why repair exists,
rather than silently trusting malformed JSON.

## 7. Where T_t acts

`T_t` does not change DeepSeek's opinion of one product. A black waterproof boot does
not become individually worse merely because the user is browsing broadly.

After quality ranking, DPP looks at the set of products together. Its relevance weight
is continuous:

```text
relevance_weight = 0.30 + 0.60 * T_t
```

- low `T_t`: stronger product-to-product repulsion, so the Top-10 explores;
- high `T_t`: stronger quality weight, so the Top-10 focuses;
- a user diversity directive may move this weight by 0.10.

This separation is the main story: DeepSeek answers "is this product good for the
current intent?" while `T_t` answers "what should this group of ten look like?"

## 8. Real 50k smoke result

The implementation was run against the real catalog, not a fabricated unit fixture.

- Broad Hokkaido request: 300 candidates, 48 DeepSeek cards, 35 strong / 9 possible /
  4 weak judgements in one tool call. The quality Top-10 included snow boots, gloves,
  ski pants, and a down jacket.
- Red lightweight wedding accessory: one strong, 39 possible, and eight weak. The
  top product was a red wedding fascinator. A deliberately conflicting profile that
  preferred blue did not override the current red request.
- Focused men's black waterproof insulated snow boots: hard mask left 29 products.
  Six were possible and 23 weak, largely because exact size evidence was absent or
  incompatible.

Accepted model calls used roughly 14k-20k reported tokens. First-pass calls took
23-28 seconds; the focused case took 46 seconds including its repair call. BGE took
about 0.7-5.3 seconds after model initialization. These are development-machine smoke
numbers, not a latency SLA.

Replaying the accepted quality scores through final DPP produced the expected graded
response. At `T=0.10`, mean Top-10 pair cosine fell from `0.7842` to `0.7452` and four
products changed. At `T=0.55`, cosine fell only from `0.7298` to `0.7160` and one
product changed. At `T=0.90`, the focused boot Top-10 was unchanged.

## 9. Current boundary

Implemented now:

- direction-protected BGE shortlist;
- optional long-term-profile input with Session Context precedence;
- forced DeepSeek native tool call and exact batch decoder;
- one repair attempt and BGE fallback;
- 80/20 individual quality fusion;
- separate T-aware DPP finalizer;
- complete request, judgement, token, and timing logs.

Still intentionally separate from this module:

- the final long-term-memory schema and update policy;
- answer generation and product explanations shown to the user;
- complete DeepSeek-generated product fact cards for catalog rows not yet processed;
- the dedicated toy-simulator scoring branch.

The normative design is in
[`deepseek-ranking-contract-v1.md`](../design/retrieve/deepseek-ranking-contract-v1.md).
