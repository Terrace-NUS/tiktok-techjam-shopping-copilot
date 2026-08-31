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

Generated reports are intentionally written outside `data/`; see
[`docs/design/catalog_semantic/README.md`](../docs/design/catalog_semantic/README.md).

## `benchmark_product_cards/public_200_v1/`

Contains the reviewed, source-grounded product-card fixture for the 200 unique
targets in `public_set.jsonl`. It is checked in so the team can run the same
zero-new-token benchmark experiment. It does not modify `catalog.jsonl`, and scores
using this known target-pool fixture must be labeled diagnostic rather than directly
comparable 50k retrieval scores. See
[`public-benchmark-product-cards-v1.md`](../docs/design/catalog_semantic/public-benchmark-product-cards-v1.md).

After validating the profile, generate the deterministic category graph review
packet with:

```powershell
catalog-category propose `
  data/catalog.jsonl `
  artifacts/catalog-semantic/category-proposal
```

This command does not approve or publish any user-facing category scope.
