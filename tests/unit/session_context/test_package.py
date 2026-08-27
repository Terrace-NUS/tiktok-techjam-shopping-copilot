from __future__ import annotations

import shopping_copilot


def test_session_context_namespace_is_public() -> None:
    assert "session_context" in shopping_copilot.__all__
    assert shopping_copilot.session_context.__name__ == "shopping_copilot.session_context"


def test_session_context_public_api_is_explicit_and_resolvable() -> None:
    public_names = shopping_copilot.session_context.__all__

    assert public_names
    assert len(public_names) == len(set(public_names))
    assert all(hasattr(shopping_copilot.session_context, name) for name in public_names)
