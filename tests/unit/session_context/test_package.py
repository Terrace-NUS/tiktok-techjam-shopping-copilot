from __future__ import annotations

import shopping_copilot


def test_session_context_namespace_is_public() -> None:
    assert "session_context" in shopping_copilot.__all__
    assert shopping_copilot.session_context.__name__ == "shopping_copilot.session_context"


def test_query_understanding_namespace_is_public_and_explicit() -> None:
    assert "query_understanding" in shopping_copilot.__all__
    public_names = shopping_copilot.query_understanding.__all__
    assert public_names
    assert len(public_names) == len(set(public_names))
    assert all(hasattr(shopping_copilot.query_understanding, name) for name in public_names)


def test_query_compiler_namespace_is_public_and_explicit() -> None:
    assert "query_compiler" in shopping_copilot.__all__
    public_names = shopping_copilot.query_compiler.__all__
    assert public_names
    assert len(public_names) == len(set(public_names))
    assert all(hasattr(shopping_copilot.query_compiler, name) for name in public_names)


def test_session_context_public_api_is_explicit_and_resolvable() -> None:
    public_names = shopping_copilot.session_context.__all__

    assert public_names
    assert len(public_names) == len(set(public_names))
    assert all(hasattr(shopping_copilot.session_context, name) for name in public_names)
    assert {
        "FacetAuthority",
        "RETRIEVAL_DERIVED_FACET_IDS",
        "retrieval_derived_facet_specs",
        "with_retrieval_derived_facets",
    }.issubset(public_names)
