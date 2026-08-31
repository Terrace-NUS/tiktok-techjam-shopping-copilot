# Public benchmark 200-product card bundle v1

Status: implemented diagnostic fixture, 2026-08-31.

## What this bundle is

The public benchmark contains 200 sessions and 200 unique target products. This
bundle gives those 200 products richer, source-grounded search cards without
spending any new model tokens:

- 161 products reuse the previously generated DeepSeek v4 Flash fact cards;
- 39 products use a deterministic fallback built from the complete raw catalog row;
- all 200 cards also receive deterministic core facts and shopping aliases, so
  model omissions such as `Button closure -> button fastening` do not remain holes;
- every fact is revalidated against the exact raw catalog bytes and carries an
  exact `source_ref` and evidence span.

The generated bundle contains 200 cards and 8,218 grounded facts. Generation made
zero API calls and reports zero token usage.

The checked-in files are:

- [`product-facts.jsonl`](../../../data/benchmark_product_cards/public_200_v1/product-facts.jsonl)
- [`manifest.json`](../../../data/benchmark_product_cards/public_200_v1/manifest.json)

Regenerate them with:

```powershell
python scripts/catalog_semantic/build_benchmark_product_cards.py
```

The raw catalog is never edited.

## Card shape

Each line has:

- `parent_asin` and `source_id`: identity plus the SHA-256-bound source row;
- `summary`: compact natural-language product meaning;
- `facts`: structured attributes such as material, color, care, closure, style,
  use case, size, and feature;
- `aliases`: equivalent search wording;
- `polarity`: whether the source asserts or excludes the fact;
- `evidence` and `source_ref`: the exact catalog text supporting the fact;
- extractor metadata and zero-token trace information.

The runtime loader recalculates each source hash and re-runs fact grounding. A stale
card or an unsupported fact therefore fails before retrieval starts.

## Runtime modes

The runtime supports two explicit views without editing the raw catalog:

- `augment`: append grounded facts to the old raw product document;
- `replace`: for covered products, use only the new fact-card view and discard the
  old search document. Uncovered products remain byte-for-byte equivalent at the
  `ProductDocument` boundary.

The public runner defaults to `replace` for Lexical, hard/facet evidence, and BGE.
Pass `--raw-product-cards` to recover the old 50k raw-card path.

Dense is a separate prebuilt artifact. A true full replacement run must also pass
the matching partial Dense index; pointing the runner at the old Dense index tests
new card text with old vectors and must not be described as full replacement.
The implemented public-200 partial index changes exactly 200 rows and preserves
the other 49,800 rows exactly. See
[`public-200-full-replacement-experiment-v1.md`](public-200-full-replacement-experiment-v1.md).

## Regression evidence

The enhanced hard/facet index was checked against the five previously inspected
hard-mask failure sessions. All eight relevant assertions now behave as intended:

| Session | Target | Required evidence | Result |
| --- | --- | --- | --- |
| `public_0041` | `B09MSY8926` | imported | kept |
| `public_0045` | `B07Z8NTWVV` | polyester | kept |
| `public_0045` | `B07Z8NTWVV` | button fastening | kept |
| `public_0098` | `B08CZ34D75` | rubber | kept |
| `public_0098` | `B08CZ34D75` | excludes polyester | kept |
| `public_0154` | `B00CYNKSTE` | cotton | kept |
| `public_0154` | `B00CYNKSTE` | hand wash only | kept |
| `public_0199` | `B089M57PSQ` | machine washable | kept |

This verifies the data path and the known failure modes. The later no-new-token
A/B also replayed all 200 saved QU results through the full replacement stack;
it is documented separately because it has a different experiment contract.

## Important score limitation

This bundle is selected from the known public target pool. It does not reveal the
target for a particular session, but it does tell retrieval which 200 catalog items
might be targets. That is target-pool leakage.

Therefore any score produced with this bundle is a **diagnostic product-card
score**, not a directly comparable 50k retrieval score. The real demo architecture
still requires cards generated independently for the full catalog. This 200-card
fixture exists to improve and test the card contract while token budget is limited.
