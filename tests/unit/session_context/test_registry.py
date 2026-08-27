from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass

import pytest

from shopping_copilot.session_context.errors import ErrorCode, SessionContextError
from shopping_copilot.session_context.models import Operator
from shopping_copilot.session_context.registry import (
    CATEGORICAL_OPERATORS,
    NUMERIC_OPERATORS,
    FacetKind,
    FacetRegistry,
    FacetSpec,
    canonical_number,
    canonical_text,
)


@pytest.fixture
def color_spec() -> FacetSpec:
    return FacetSpec(
        id="color",
        kind=FacetKind.CATEGORICAL,
        operators=CATEGORICAL_OPERATORS,
        normalizer=canonical_text,
    )


@pytest.fixture
def budget_spec() -> FacetSpec:
    return FacetSpec(
        id="budget",
        kind=FacetKind.NUMERIC,
        operators=NUMERIC_OPERATORS,
        normalizer=canonical_number,
    )


@pytest.fixture
def registry(color_spec: FacetSpec, budget_spec: FacetSpec) -> FacetRegistry:
    return FacetRegistry(specs=(color_spec, budget_spec))


def test_registry_copies_sorts_and_freezes_configuration(
    color_spec: FacetSpec,
    budget_spec: FacetSpec,
) -> None:
    source = [color_spec, budget_spec]
    registry = FacetRegistry(specs=source)
    source.clear()

    assert registry.specs == (budget_spec, color_spec)
    assert tuple(registry) == registry.specs
    assert len(registry) == 2
    assert isinstance(registry.specs, tuple)

    with pytest.raises(FrozenInstanceError):
        color_spec.id = "shade"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registry._specs = ()  # type: ignore[misc]


def test_registry_lookup_is_exact_and_require_preserves_error_context(
    registry: FacetRegistry,
    color_spec: FacetSpec,
) -> None:
    assert registry.get("color") is color_spec
    assert registry.require("color") is color_spec
    assert registry.get("colour") is None

    path = ("operations", 0, "preference", "facet")
    with pytest.raises(SessionContextError) as caught:
        registry.require("colour", path=path)

    assert caught.value.code is ErrorCode.UNKNOWN_FACET
    assert caught.value.path == path
    assert caught.value.details == (("facet", "colour"),)


@pytest.mark.parametrize("facet", ["Color", " color", "color ", "colour"])
def test_registry_does_not_guess_aliases_or_canonicalize_facet_ids(
    registry: FacetRegistry,
    facet: str,
) -> None:
    with pytest.raises(SessionContextError) as caught:
        registry.normalize_value(facet, Operator.EQ, "blue")

    assert caught.value.code is ErrorCode.UNKNOWN_FACET


def test_closed_operator_families_are_exposed_by_kind(
    color_spec: FacetSpec,
    budget_spec: FacetSpec,
) -> None:
    assert color_spec.operators == frozenset(
        {Operator.EQ, Operator.NEQ, Operator.IN, Operator.NOT_IN}
    )
    assert budget_spec.operators == frozenset({Operator.LT, Operator.LE, Operator.GT, Operator.GE})


def test_text_values_are_nfkc_normalized_casefolded_and_whitespace_collapsed() -> None:
    assert canonical_text("  Ｒｅｄ\u00a0  BLUE  ") == "red blue"
    assert canonical_text("Straße") == "strasse"


@pytest.mark.parametrize("value", [42, True, 1.5])
def test_canonical_text_rejects_non_strings(value: object) -> None:
    with pytest.raises(TypeError):
        canonical_text(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", "   ", "blue\nred", "blue\x00red"])
def test_canonical_text_rejects_empty_or_control_text(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_text(value)


def test_canonical_number_preserves_finite_numbers_and_collapses_integral_floats() -> None:
    assert canonical_number(7) == 7
    assert type(canonical_number(7)) is int
    assert canonical_number(7.0) == 7
    assert type(canonical_number(7.0)) is int
    assert canonical_number(7.25) == 7.25
    assert type(canonical_number(7.25)) is float


@pytest.mark.parametrize("value", [True, False, "7"])
def test_canonical_number_rejects_bool_and_non_numbers(value: object) -> None:
    with pytest.raises(TypeError):
        canonical_number(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_number_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_number(value)


def test_registry_normalizes_categorical_scalars_and_canonicalizes_sets(
    registry: FacetRegistry,
) -> None:
    assert registry.normalize_value("color", Operator.EQ, "  Ｂｌｕｅ  ") == "blue"
    assert registry.normalize_value(
        "color",
        Operator.IN,
        ("RED", " blue ", "red", "Green"),
    ) == ("blue", "green", "red")


@pytest.mark.parametrize("operator", [Operator.LT, Operator.LE, Operator.GT, Operator.GE])
def test_registry_normalizes_every_numeric_bound_operator(
    registry: FacetRegistry,
    operator: Operator,
) -> None:
    assert registry.normalize_value("budget", operator, 100.0) == 100


@pytest.mark.parametrize(
    ("facet", "operator", "value"),
    [
        ("color", Operator.LT, "blue"),
        ("budget", Operator.EQ, 100),
        ("color", "eq", "blue"),
    ],
)
def test_registry_rejects_operators_outside_the_facet_family(
    registry: FacetRegistry,
    facet: str,
    operator: object,
    value: object,
) -> None:
    path = ("preference", "operator")
    with pytest.raises(SessionContextError) as caught:
        registry.normalize_value(  # type: ignore[arg-type]
            facet,
            operator,
            value,
            path=path,
        )

    assert caught.value.code is ErrorCode.INVALID_OPERATOR_FOR_FACET
    assert caught.value.path == path
    assert caught.value.details == (("facet", facet),)


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -math.inf, "100"])
def test_registry_converts_invalid_numeric_values_to_a_domain_error(
    registry: FacetRegistry,
    value: object,
) -> None:
    path = ("preference", "value")
    with pytest.raises(SessionContextError) as caught:
        registry.normalize_value(  # type: ignore[arg-type]
            "budget",
            Operator.LE,
            value,
            path=path,
        )

    assert caught.value.code is ErrorCode.INVALID_OPERATOR_VALUE
    assert caught.value.path == path
    assert caught.value.details == ()


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        (Operator.EQ, ("blue",)),
        (Operator.IN, []),
        (Operator.IN, ()),
        (Operator.IN, ("blue", 1)),
        (Operator.LE, (100,)),
    ],
)
def test_registry_rejects_values_with_the_wrong_operator_shape(
    registry: FacetRegistry,
    operator: Operator,
    value: object,
) -> None:
    facet = "budget" if operator is Operator.LE else "color"
    with pytest.raises(SessionContextError) as caught:
        registry.normalize_value(facet, operator, value)  # type: ignore[arg-type]

    assert caught.value.code is ErrorCode.INVALID_OPERATOR_VALUE


@pytest.mark.parametrize(
    "facet_id",
    ["", "Color", "color-name", "color__name", "1color", "color ", "other"],
)
def test_facet_spec_rejects_noncanonical_or_reserved_ids(facet_id: str) -> None:
    with pytest.raises(ValueError):
        FacetSpec(
            id=facet_id,
            kind=FacetKind.CATEGORICAL,
            operators=CATEGORICAL_OPERATORS,
            normalizer=canonical_text,
        )


def test_facet_spec_rejects_invalid_trusted_configuration() -> None:
    with pytest.raises(TypeError):
        FacetSpec(  # type: ignore[arg-type]
            id="color",
            kind="categorical",
            operators=CATEGORICAL_OPERATORS,
            normalizer=canonical_text,
        )
    with pytest.raises(TypeError):
        FacetSpec(  # type: ignore[arg-type]
            id="color",
            kind=FacetKind.CATEGORICAL,
            operators=frozenset({"eq"}),
            normalizer=canonical_text,
        )
    with pytest.raises(ValueError):
        FacetSpec(
            id="color",
            kind=FacetKind.CATEGORICAL,
            operators=frozenset({Operator.EQ}),
            normalizer=canonical_text,
        )
    with pytest.raises(TypeError):
        FacetSpec(  # type: ignore[arg-type]
            id="color",
            kind=FacetKind.CATEGORICAL,
            operators=CATEGORICAL_OPERATORS,
            normalizer=None,
        )


def test_registry_rejects_duplicate_or_non_spec_configuration(
    color_spec: FacetSpec,
) -> None:
    with pytest.raises(ValueError):
        FacetRegistry(specs=(color_spec, color_spec))
    with pytest.raises(TypeError):
        FacetRegistry(specs=(color_spec, object()))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtendedFacetSpec(FacetSpec):
    extras: list[str]


def test_registry_rejects_facet_spec_subclasses_with_extra_mutable_state() -> None:
    extended = ExtendedFacetSpec(
        id="color",
        kind=FacetKind.CATEGORICAL,
        operators=CATEGORICAL_OPERATORS,
        normalizer=canonical_text,
        extras=[],
    )

    with pytest.raises(TypeError):
        FacetRegistry(specs=(extended,))
