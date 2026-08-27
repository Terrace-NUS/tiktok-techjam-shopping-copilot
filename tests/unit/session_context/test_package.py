from __future__ import annotations

import shopping_copilot


def test_session_context_namespace_is_public() -> None:
    assert "session_context" in shopping_copilot.__all__
    assert shopping_copilot.session_context.__name__ == "shopping_copilot.session_context"
