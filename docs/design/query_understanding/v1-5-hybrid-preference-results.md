# Query Understanding v1.5: hybrid preference experiment

## Decision

One structured `Preference` now carries two retrieval views:

- a short executable `value` for hard eligibility, Facet recall, and `q_lex`;
- complete `semantic_text` for `q_sem`, BGE, and DeepSeek ranking.

The original `evidence_text` remains the user quote used for audit and explanation.
No Session Context schema change was required because the existing contract already
allowed structured and semantic fields to coexist.

For material, the hard view checks only component presence. Percentages, purity,
blend ratios, provenance, and similar qualifiers remain semantic ranking evidence.

```text
pure polyester
  value         = polyester
  semantic_text = material must be pure polyester

95% polyester, 5% spandex
  value         = polyester OR spandex aliases
  semantic_text = preferred composition is 95% polyester and 5% spandex
```

## Implementation

- Query Understanding prompt v1.5 asks DeepSeek to separate executable anchors,
  complete meaning, and quoted evidence.
- The local materializer deterministically removes material percentages and
  qualifiers from the executable view. It does not trust prompt compliance alone.
- Retrieval-derived structured preferences retain their complete `meaning` as
  `semantic_text`.
- Query Compiler uses the keyword for `q_lex` and hard constraints, while preferring
  `semantic_text` when constructing `q_sem`.

## Regression verification

- Focused unit and compiler tests: 65 passed.
- Full repository suite after the change: 1,096 passed.
- Ruff checks passed.

An offline replay transformed only the material constraints in the 44 previous
misses, with all other stored QU and retrieval inputs held fixed:

- 10 target products changed from hard-ineligible to eligible;
- 0 target products changed from eligible to ineligible.

The recovered cases covered `pure polyester`, `pure cotton`, percentage blends,
and `genuine leather` versus catalog text containing a broader leather token.

## Targeted real-chain run

The ten recovered hard-mask cases were then rerun through DeepSeek QU, Session
Context, intent transparency, multi-route recall, BGE, and DPP. DeepSeek customer
surface messages were reused from the prior cache; targets remained evaluator-only.

| Result | Count |
| --- | ---: |
| Previously missed cases tested | 10 |
| Newly hit in Top 10 | 5 |
| Still missed after eligibility recovery | 5 |

New hits were `public_0006`, `public_0009`, `public_0100`, `public_0112`, and
`public_0187`. Their best BGE ranks were 1, 1, 7, 4, and 10 respectively.

The remaining cases show the expected next boundary: restoring eligibility does
not guarantee final Top-10 placement. Two targets reached BGE ranks 18 and 40;
three did not enter the fused candidate pool under the newly reconciled intent.
They are recall/ranking or missing-goal issues rather than material string equality.

This is a targeted diagnostic subset, not a replacement estimate for the complete
200-session benchmark.
