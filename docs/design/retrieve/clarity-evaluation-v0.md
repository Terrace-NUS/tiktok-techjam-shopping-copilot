# Real-World Clarity Evaluation V0

- Protocol status: **prompt suite and gate frozen before first run**
- Run status: **first fixed run complete; pre-registered gate failed**
- Date: **2026-08-28**
- Prompt identity:
  `sha256:ed72bac1cab8c5048fb93dd132f2a96e6d419bdb03a338a1cbbc27f39ad23087`
- Scope: raw Dense semantic Probe only; this is not yet end-to-end $C_t$

## Question

The test asks one narrow question:

> For the same real-world shopping intent, does the fixed semantic Probe assign
> greater concentration when the user expresses a more specific product region?

It does not inspect a hidden product, measure recall, or use the official
simulator. Absolute values across unrelated categories are not evidence. The
statistical unit is one prompt family and the primary evidence is its paired
change.

## Frozen prompt suite

The source-controlled suite is
[`config/retrieval/clarity-prompts-v0.json`](../../../config/retrieval/clarity-prompts-v0.json).
It was written from ordinary shopping language after checking only broad
catalog-domain support. No product title, target ASIN, public-session label, or
retrieval score was used to compose it.

The suite contains:

| Group | Count | Expected relationship |
| --- | ---: | --- |
| Intent families | 40 | `vague < focused < specific` |
| Length traps | 10 | `long_vague < short_specific` |
| Invariance calibration | 5 pairs | equivalent meaning; used only to set a noise margin |
| Invariance audit | 5 pairs | equivalent meaning; material changes should be rare |
| Safety diagnostics | 8 | logged individually; no forced ordinal label |

The 40 families are balanced across ten domains: women's dresses, women's tops
and outerwear, women's active/swim apparel, women's shoes, men's clothing,
men's shoes, jewelry, bags/travel, watches/accessories, and
children/workwear/costume products.

Real users mention sizes, budgets, colors, materials, exclusions, and product
features even when a field is poorly structured in the catalog. V0 keeps those
requests instead of making an artificially easy vector benchmark. Because
price and exact size are not reliably represented in the current product
documents, their eventual contribution must come from Query Understanding and
the hard-mask layer. Failures here must therefore be separated into semantic
Probe limitations versus missing structured grounding.

## Fixed execution

```text
one frozen natural-language prompt
    -> q_sem = prompt (temporary; Query Understanding does not exist yet)
    -> pinned BGE query embedding
    -> one full 50k score vector
    -> one stable all-eligible Top-80 ranking
    -> prefix views at K=20, 40, and 80
    -> catalog-mean-centered raw G for each K
```

The model revision, document corpus, semantic release, score vector, and row
ordering are those bound by Dense R0. The Probe does not observe the expected
prompt level and never changes its own candidate depth.

The primary $K$ is 40. Top-20 and Top-80 are robustness checks chosen before
the run; no result-dependent $K$ selection is allowed.

## Noise margin and paired score

Five calibration paraphrase pairs express equivalent product constraints. At
Top-40, the practical tie margin is frozen per run as:

$$
\epsilon = P_{95}(|G(q_a)-G(q_b)|).
$$

For an expected increase with paired difference $\Delta$:

- win when $\Delta > \epsilon$;
- tie when $|\Delta| \le \epsilon$;
- loss when $\Delta < -\epsilon$; and
- concordance is $(\text{wins}+0.5\,\text{ties})/N$.

The report also retains raw positive direction, median paired delta, all
family-level values, and a deterministic 5,000-sample bootstrap. Bootstrap
resampling keeps a complete family together and is stratified by domain.

## Pre-registered gate

Raw $G$ passes this small hackathon falsification gate only if every condition
holds:

1. Top-40 vague-to-specific concordance is at least `0.70`.
2. Its stratified-bootstrap 95% lower bound is at least `0.60`.
3. The bootstrap 95% interval for median vague-to-specific delta is entirely
   above zero.
4. Top-40 vague-to-focused and focused-to-specific concordance are each at
   least `0.60`.
5. At least `0.65` of full chains have no material reversal in either adjacent
   step.
6. Long-vague to short-specific concordance is at least `0.70`.
7. Vague-to-specific concordance is at least `0.65` at both Top-20 and Top-80.
8. No more than `20%` of locked invariance-audit pairs exceed the calibration
   margin.
9. Availability is `100%`; missing values cannot be replaced with a neutral
   constant.

These thresholds test ordinal discrimination, not an absolute controller
scale. Passing does not justify interpreting `0.7` as a universal clarity
boundary. Failing means raw $G$ cannot be called $C_t$.

## Diagnostics outside the gate

Contradictory constraints, negative-only requests, multi-item requests,
subjective language, intentionally diverse browsing, and unsupported product
technology are reported separately. The current raw semantic Probe cannot be
expected to resolve all of them. In the complete architecture they also require
Query Understanding, feasibility checks, and ephemeral diagnostics $D_t$.

## Anti-overfitting rule

All failures remain in the report. V0 cannot be edited, pruned, or reworded
after scores are visible. If the formula, centering, weighting, prompt compiler,
or Probe membership is changed using these results, a separately authored V1
suite is required before claiming confirmation.

## First fixed run

The first run used the 50,000-product Dense R0 index and the pinned
`BAAI/bge-small-en-v1.5` model revision. All 168 natural-language prompts and
all 504 Probe observations were available. The generated machine-readable
report is `artifacts/retrieval/clarity-evaluation-v0.json`.

The result is useful but not sufficient: raw $G$ clearly separates a broad
request from a named product region, but it does not reliably recognize the
extra clarity contributed by fine-grained constraints.

| Pre-registered comparison at Top-40 | Wins / ties / losses | Concordance | Result |
| --- | ---: | ---: | --- |
| Vague to focused | 29 / 10 / 1 | `0.850` | pass |
| Focused to specific | 7 / 26 / 7 | `0.500` | **fail** |
| Vague to specific | 26 / 14 / 0 | `0.825` | pass |
| Long-vague to short-specific | 4 / 5 / 1 | `0.650` | **fail** |

The mean Top-40 values make the saturation visible:

| Prompt level | Mean raw G | Median raw G |
| --- | ---: | ---: |
| Vague | `0.346513` | `0.373164` |
| Focused | `0.436435` | `0.441983` |
| Specific | `0.435931` | `0.433654` |

The strongest result is not tied to one Probe size. Vague-to-specific
concordance was `0.825`, `0.825`, and `0.7875` at Top-20, Top-40, and Top-80.
At Top-40 its stratified-bootstrap lower bound was `0.7625`, and the lower
bound for median paired improvement was `0.037924`. Eighty percent of families
had no material reversal in either adjacent step. All five locked audit
paraphrases remained within the independently calibrated tie margin
`epsilon=0.03755645`.

Two of eleven gate checks failed:

1. Focused-to-specific concordance was `0.50`, below `0.60`.
2. The length-control concordance was `0.65`, below `0.70`; eight of ten raw
   differences still pointed in the expected direction, but several were too
   small to exceed the calibrated noise margin.

This rejects the claim that raw $G$ is already $C_t$. It supports a narrower
claim: semantic candidate concentration is one useful input for distinguishing
an open product space from an established product type. Exact color, size,
budget, material, feature, and exclusion constraints need grounded Query
Understanding and mask evidence; they cannot be inferred reliably from this
single geometric value.

Reproduce the fixed run with:

```powershell
.\.venv-3.10\Scripts\python.exe scripts/retrieval/evaluate_clarity_prompts.py `
  --dense-factory shopping_copilot.retrieval:create_dense_retriever `
  --dense-index artifacts/retrieval/dense-v0 `
  --semantic-release artifacts/catalog-semantic/release-v0 `
  --output artifacts/retrieval/clarity-evaluation-v0.json
```
