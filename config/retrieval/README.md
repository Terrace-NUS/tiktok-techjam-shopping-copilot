# Retrieval evaluation inputs

`clarity-prompts-v0.json` is a hand-authored, target-free development audit for
semantic clarity. It contains natural English shopping requests from catalog
domains with substantial product support:

- 40 independent intent families, each with `vague`, `focused`, and `specific`
  wording;
- 10 long-but-vague versus short-but-specific controls;
- 10 semantic-invariance pairs, split into five calibration and five audit
  controls; and
- 8 contradiction, multi-intent, unsupported, and open-ended diagnostics.

The suite never reads simulator targets, product labels, or expected result
IDs. Prompts were frozen before the first run of the clarity evaluator. Their
initial source identity is:

```text
sha256:ed72bac1cab8c5048fb93dd132f2a96e6d419bdb03a338a1cbbc27f39ad23087
```

Do not edit v0 after observing scores. Corrections or new prompts require a new
version and a new hash. If a clarity formula is tuned using v0, v0 becomes a
development set and cannot also serve as the locked confirmation set.

## Intent Volume runtime policy

[`intent-volume-v1.json`](intent-volume-v1.json) mirrors the exact parameters
exported by `shopping_copilot.retrieval.IntentVolumePolicy`. It is the frozen
hackathon runtime policy selected after the 60-conversation / 130-turn v1.3
run. A unit test prevents the checked-in document and code defaults from
drifting apart.

This is a presentation and transition policy, not an independent held-out
claim of universal product relevance. Its evidence artifacts and limitations
are named directly in the document.

## Ranking strategy experiment

Ranking model revisions and runtime parameters are frozen in
`scripts/retrieval/evaluate_ranking_strategies_v0.py`. The experiment compares
all methods over the same RRF-bounded Top-80 pool and writes generated results
under ignored `artifacts/retrieval/`. The BGE-only augmentation entry point is
`scripts/retrieval/augment_bge_dpp_ranking_v0.py`; it adds the final BGE+DPP
low/high-T comparison without re-running Qwen. Conclusions and exact metrics are
documented in
[`../../docs/design/retrieve/ranking-strategy-evaluation-v0.md`](../../docs/design/retrieve/ranking-strategy-evaluation-v0.md).
