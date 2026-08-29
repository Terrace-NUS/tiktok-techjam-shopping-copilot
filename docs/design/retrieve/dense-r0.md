# Dense Retrieval R0

- Status: **implemented experimental vertical slice**
- Date: **2026-08-28**
- Contract relationship: informs the open decisions in [`contract-v0.md`](contract-v0.md);
  it does not freeze the final $C_t$ formula or routing policy.

## Outcome

R0 establishes one reproducible dense route over the complete 50,000-product
catalog and a fixed, shadow-only semantic-coherence Probe.

The first-turn public component evaluation shows that dense retrieval is worth
keeping as a complementary route. On the 170 sessions where a target is
officially eligible on turn one:

| Metric | BM25 | Dense | Unordered route union |
| --- | ---: | ---: | ---: |
| Target Recall@10 | 0.123529 | 0.164706 | 0.223529 |
| Target Recall@40 | 0.300000 | 0.305882 | 0.435294 |
| Target Recall@100 | 0.470588 | 0.429412 | 0.617647 |
| MRR@10 | 0.061657 | 0.076501 | not applicable |

At Top-10, Dense uniquely retrieves 17 eligible targets that BM25 misses; BM25
uniquely retrieves 10. The union column consumes up to $2K$ candidates, so it is
candidate coverage, not a fair fused Top-$K$ comparison or an official
multi-turn score.

For continuity, the all-200 diagnostic remains BM25/Dense/union Recall@10
`0.185/0.190/0.275`, Recall@40 `0.350/0.330/0.475`, and Recall@100
`0.525/0.450/0.660`. The 30 Intent Override observations in this view occur
before the override and are not official eligible hits.

## Implemented boundary

```text
verified Catalog Semantic release
    -> deterministic labeled product documents
    -> pinned local embedding model
    -> normalized float32 vectors
    -> one exact full-catalog score vector per q_sem
         -> index/release-bound eligibility mask before stable Top-K
         -> Dense candidate route
         -> fixed Top-20/40/80 shadow Probe
```

Retrieval consumes `q_sem`; it does not parse conversation history. Until Query
Understanding exists, the component evaluator temporarily sets
`q_sem = user_message`. The eligibility interface already exists, but this R0
evaluation uses an all-eligible mask. Callers provide eligible `parent_asin`
values; `DenseIndex` maps them to its private row order and returns an immutable
mask bound to the index and Catalog Semantic release. Scores carry the same
binding. Dense ranking and all three Probe views consume one
`DenseSearchResult`, so a raw 50k array from another index cannot be silently
misinterpreted.

The implementation lives in `src/shopping_copilot/retrieval/`. The official
adapter and evaluator remain unchanged.

## Pinned encoder and index

- Model: `BAAI/bge-small-en-v1.5`
- Model revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- Backend: `sentence-transformers==5.7.0`
- Dimension: 384
- Maximum sequence length: 512
- Query instruction: `Represent this sentence for searching relevant passages: `
- Document instruction: none
- Similarity: normalized dot product, equivalent to cosine similarity
- Search: exact NumPy matrix multiplication; no approximate index or vector database

The product document contains fixed labeled sections in this order:

```text
title
categories
store
features
details
description
```

Whitespace, list joining, details-key ordering, and field limits are
deterministic. A tokenizer audit found 3,974 of 50,000 documents (7.948%) exceed
512 tokens; median length is 201, p90 is 474, and p95 is 578. The most important
fields occur first, but the long-tail truncation remains an explicit R0
limitation.

The generated bundle contains exactly:

```text
bundle-manifest.json
parent-asins.json
vectors.npy
```

The observed vector matrix is 76,800,128 bytes including the NumPy header. The
manifest binds catalog ID, Catalog Semantic release ID, rendered-document hash,
model revision, backend version, prompt contract, dimension, dtype, normalization,
ASIN bytes, and vector bytes. Loading copies a verified snapshot into owned,
read-only memory and rechecks artifact hashes after materialization. Runtime
construction also requires the active Catalog Semantic release ID.

Observed local index ID:

```text
sha256:af5b074e3889eebae00402c22cea717c18cde37ba6e81aba19ef4c7b48f647c2
```

The source catalog and the catalog copied into the verified semantic release
both remain:

```text
sha256:da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67
```

On the current Windows workstation with an RTX 4070 Ti, the complete 50k index
build took about 106 seconds. Measurements used Python 3.10.21, NumPy 2.2.6,
Sentence Transformers 5.7.0, and local CUDA build `torch 2.13.0+cu130`. The
200-query component run observed:

| Boundary | Warm median | Warm p95 | One-time initialization |
| --- | ---: | ---: | ---: |
| BM25 query | 6.516 ms | 19.361 ms | 1,359.107 ms |
| Dense encode + exact Top-100 | 12.502 ms | 16.502 ms | 27,497.941 ms |
| Three shadow Probes (20/40/80) | 0.480 ms | 0.672 ms | 110.018 ms |

These are local wall-clock component measurements, not promises for the final
submission environment. Dense initialization includes deep validation of the
active semantic release, index hash verification, the owned vector copy, and
model loading. Its first query took 177.404 ms; later queries are represented
by the warm figures above. Probe latency fell after all sizes began sharing the
same Top-100 ordering rather than sorting the catalog three more times.

## Shadow coherence result

For the fixed dense Top-$K$, R0 computes catalog-mean-centered unit vectors and:

$$
G = \frac{nR^2 - 1}{n - 1},
\qquad
R = \left\|\frac{1}{n}\sum_i z_i\right\|.
$$

`G` is recorded as uncalibrated `C_raw`. It never changes the Probe, route
depth, score, or ranking.

Across the 200 first-turn messages:

| Probe size | Mean raw G | Median | p10 | p90 |
| --- | ---: | ---: | ---: | ---: |
| 20 | 0.403618 | 0.400225 | 0.325331 | 0.505737 |
| 40 | 0.374170 | 0.367654 | 0.300562 | 0.474598 |
| 80 | 0.340951 | 0.336501 | 0.270422 | 0.434920 |

The rank-order stability is strong:

- Spearman Top-20 versus Top-40: `0.945271`
- Spearman Top-40 versus Top-80: `0.947152`

However, $C_t$ is **not ready to freeze**. On 100 offline target-derived
specificity chains:

- broad category $<$ full category path in 89% of cases;
- full category path $<$ path plus two explicit constraints in only 26%; and
- the complete strict ordering holds in only 21%.

The corresponding mean raw values are `0.299740`, `0.423865`, and `0.412052`.
Cases are selected in stable SHA-256 order with round-robin scenario/category
stratification. The chains are target-derived synthetic evidence used only to
try to falsify the signal; they never enter runtime or calibration.

This is consistent with the current missing Query Understanding and hard-mask
stage: explicit constraints are merely appended to one semantic sentence, not
grounded and applied before the Probe. Raw $G$ also does not predict retrieval
success here: at Top-40, the `neither` group has mean `0.382726`, above `both`
at `0.362424`. R0 therefore provides no basis for treating raw $G$ as intent
transparency or using it to control the four retrieval layers. Vector geometry
can remain a research lead and must be tested again only after grounded Query
Understanding and mask application exist.

A later target-free audit replaced those synthetic chains with 40 hand-written
real-world prompt families. It found strong vague-to-focused and
vague-to-specific separation, but no focused-to-specific separation. The
frozen prompts, pre-registered gate, and complete interpretation are in
[`clarity-evaluation-v0.md`](clarity-evaluation-v0.md). This narrows the
conclusion: raw $G$ is useful concentration evidence, but it is not complete
intent transparency.

## Reproduction

Install the platform-specific retrieval stack in Python 3.10:

```powershell
python -m pip install -e ".[dev,retrieval]"
```

Build and validate the generated index:

```powershell
retrieval-dense build `
  artifacts/catalog-semantic/release-v0 `
  artifacts/retrieval/dense-v0 `
  --batch-size 128

retrieval-dense validate artifacts/retrieval/dense-v0
```

Query it and inspect the shadow Probe:

```powershell
retrieval-dense query `
  artifacts/retrieval/dense-v0 `
  "lightweight red running shoes for women" `
  --release-dir artifacts/catalog-semantic/release-v0 `
  --top-k 10 `
  --probe-k 40
```

Run the first-turn component evaluation:

```powershell
python scripts/retrieval/evaluate_first_turn.py `
  --dense-factory shopping_copilot.retrieval:create_dense_retriever `
  --dense-index artifacts/retrieval/dense-v0 `
  --semantic-release artifacts/catalog-semantic/release-v0 `
  --specificity-chains 100 `
  --output artifacts/retrieval/first-turn-evaluation.json
```

Generated models and index artifacts are machine-local and excluded from Git.
The source-controlled lock is `requirements/retrieval.lock`.

## Next gate

The next retrieval step is a fixed BM25 + Dense fusion baseline. $C_t$ remains
shadow-only until Query Understanding can produce a grounded `q_sem` and apply
the agreed eligible mask before the fixed Probe. Only then should the
specificity-chain test be rerun and calibration considered.
