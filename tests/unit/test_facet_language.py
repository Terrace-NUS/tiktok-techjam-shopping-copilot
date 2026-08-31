from __future__ import annotations

import pytest

from shopping_copilot.facet_language import material_keywords


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("pure polyester", ("polyester",)),
        ("100% Polyester", ("polyester",)),
        (
            "95% polyester, 5% spandex",
            ("polyester", "elastane", "lycra", "spandex"),
        ),
        ("genuine leather", ("leather",)),
        ("100% gossypium", ("cotton", "gossypium")),
    ],
)
def test_material_keywords_keep_only_broad_executable_anchors(
    phrase: str,
    expected: tuple[str, ...],
) -> None:
    assert material_keywords(phrase) == expected


def test_material_keywords_preserve_unknown_material_without_quantifier() -> None:
    assert material_keywords("100% cork textile") == ("cork textile",)
    assert material_keywords("soft fabric") == ("soft fabric",)
