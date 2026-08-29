from __future__ import annotations

import pytest

from shopping_copilot.retrieval.documents import ProductDocument
from shopping_copilot.retrieval.lexical import LEXICAL_QUERY_TOKEN_LIMIT, LexicalProbe


def _document(
    parent_asin: str,
    *,
    title: str = "",
    categories: str = "",
    store: str = "",
    features: str = "",
    details: str = "",
    description: str = "",
) -> ProductDocument:
    return ProductDocument(
        parent_asin=parent_asin,
        text=(
            f"title: {title}\n"
            f"categories: {categories}\n"
            f"store: {store}\n"
            f"features: {features}\n"
            f"details: {details}\n"
            f"description: {description}"
        ),
    )


def test_observe_is_stable_and_title_weight_beats_description() -> None:
    probe = LexicalProbe(
        (
            _document("B", title="red shoes"),
            _document("A", description="red shoes"),
            _document("C", title="blue shoes"),
        ),
        probe_k=3,
    )

    first = probe.observe("RED red shoes")
    second = probe.observe("RED red shoes")

    assert first == second
    assert first.available is True
    assert first.reason is None
    assert first.tokens == ("red", "shoes")
    assert first.matched_count == 3
    assert first.matched_token_count == 2
    assert first.mean_normalized_idf is not None
    assert [hit.parent_asin for hit in first.hits] == ["B", "A", "C"]
    assert first.hits[0].raw_bm25 < first.hits[-1].raw_bm25
    assert probe.parent_asins == frozenset({"A", "B", "C"})


def test_fixed_field_weights_prefer_title_for_the_same_term() -> None:
    probe = LexicalProbe(
        (_document("A", description="red"), _document("C", title="red")), probe_k=2
    )

    observed = probe.observe("red")

    assert [hit.parent_asin for hit in observed.hits] == ["C", "A"]
    assert observed.hits[0].raw_bm25 < observed.hits[1].raw_bm25


def test_eligibility_is_applied_before_top_k() -> None:
    probe = LexicalProbe(
        (
            _document("A", title="red red red"),
            _document("B", title="red red"),
            _document("C", title="red"),
        ),
        probe_k=1,
    )

    observed = probe.observe("red", eligible_parent_asins={"B", "C"})

    assert observed.eligible_count == 2
    assert observed.matched_count == 2
    assert [hit.parent_asin for hit in observed.hits] == ["B"]


def test_equal_scores_use_parent_asin_as_stable_tie_break() -> None:
    probe = LexicalProbe((_document("Z", title="green"), _document("A", title="green")), probe_k=2)

    observed = probe.observe("green")

    assert [hit.parent_asin for hit in observed.hits] == ["A", "Z"]


@pytest.mark.parametrize(
    ("query", "eligibility", "reason"),
    [
        ("...", None, "empty_query"),
        ("unknown", None, "no_matches"),
        ("red", set(), "no_eligible_documents"),
    ],
)
def test_unavailable_reasons(query: str, eligibility: set[str] | None, reason: str) -> None:
    probe = LexicalProbe((_document("A", title="red"),))

    observed = probe.observe(query, eligible_parent_asins=eligibility)

    assert observed.available is False
    assert observed.reason == reason
    assert observed.hits == ()
    assert observed.mean_normalized_idf is None


def test_unicode_diacritics_and_fixed_token_limit() -> None:
    probe = LexicalProbe((_document("A", title="cafe lamp"),))
    query = "Café " + " ".join(f"word{index}" for index in range(50))

    observed = probe.observe(query)

    assert observed.tokens[0] == "cafe"
    assert len(observed.tokens) == LEXICAL_QUERY_TOKEN_LIMIT
    assert observed.available is True


def test_invalid_documents_and_eligibility_fail_closed() -> None:
    malformed = ProductDocument(parent_asin="A", text="title: red")
    with pytest.raises(ValueError, match="malformed ProductDocument"):
        LexicalProbe((malformed,))

    probe = LexicalProbe((_document("A", title="red"),))
    with pytest.raises(KeyError, match="unknown eligible"):
        probe.observe("red", eligible_parent_asins={"B"})
    with pytest.raises(TypeError, match="iterable"):
        probe.observe("red", eligible_parent_asins="A")
