# Catalog semantic review inputs

This directory contains small, source-controlled human decisions and profiling
inputs. Generated catalog-semantic artifacts remain under the Git-ignored
`artifacts/catalog-semantic/` directory.

- `v0/category-scope-selection.json` pins the accepted CS1 graph and reviewed
  CategoryScope roots. Full closures and IDs are deterministic builder output.
- `v0/gate-a-profile-selection.json` pins the accepted CS1 artifacts and the
  exhaustive CS2 observation lanes. It does not approve any facet or binding.
- `v0/gate-a-selection.json` records the repository owner's first Gate-A
  extraction approval. It publishes only the `price` definition,
  applicability, exact top-level binding, closed implementation IDs, and
  frozen-data result counts.
- `v0/gate-b-selection.json` records the repository owner's approval of the
  exact reviewed `price` proposal. It grants intent, conservative retrieval,
  and Probe permission independently in all 15 published scopes, while
  proactive price clarification remains disabled. Runtime modules must consume
  the generated capability artifact; this config is not executable logic.

Every input is content-bound to the frozen catalog. If a pinned catalog,
registry, assignment set, or builder identity changes, the corresponding build
must fail closed and return to review.
