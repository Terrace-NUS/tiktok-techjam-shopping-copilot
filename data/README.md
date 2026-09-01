# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` from the GitHub Release and decompress it as `catalog.jsonl` in this directory. Expected row count: 50,000.

Never place API keys, private evaluation data, or participant outputs in this directory.

After downloading the catalog, create the deterministic read-only profile with:

```powershell
python -m shopping_copilot.catalog.profiling `
  data/catalog.jsonl `
  artifacts/catalog-profile
```

Generated reports are intentionally written outside `data/` under the ignored
`artifacts/` tree.

## `benchmark_product_cards/public_200_v1/`

Contains the reviewed, source-grounded product-card fixture for the 200 unique
targets in `public_set.jsonl`. It is checked in so the team can run the same
zero-new-token benchmark experiment. It does not modify `catalog.jsonl`, and scores
using this known target-pool fixture must be labeled diagnostic rather than directly
comparable 50k retrieval scores.

## `product_fact_cards/deepseek_7011_v1/`

Contains 7,011 source-grounded product fact cards extracted with DeepSeek V4 Flash.
The cards are a partial-catalogue sidecar: they add model-derived retrieval evidence
without modifying `catalog.jsonl`. The runtime loader reads the compressed JSONL
directly and revalidates every retained fact against the immutable source row.

See the bundle manifest and README in that directory for provenance and scope.

After validating the profile, generate the deterministic category graph review
packet with:

```powershell
catalog-category propose `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-proposal
```

This command does not approve or publish any user-facing category scope.
