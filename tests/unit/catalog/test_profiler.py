from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shopping_copilot.catalog.profiling import (
    ProductCategoryAssignment,
    ProfileConfig,
    canonical_json_dumps,
    catalog_file_sha256,
    category_id_for_path,
    profile_catalog,
)


def _synthetic_catalog_bytes() -> bytes:
    products: list[object] = [
        {
            "parent_asin": "p1",
            "title": "One",
            "categories": ["Root", "Shoes"],
            "details": {
                "Color": "Red",
                "Nested": {"b": 2, "a": 1},
                "Tags": ["waterproof", "red"],
            },
        },
        {
            "parent_asin": "p2",
            "title": "Two",
            "categories": ["Root", "Shoes"],
            "details": {"Color": "Blue", "Nested": None},
        },
        {
            "parent_asin": "p3",
            "title": "Three",
            "categories": ["Root", "Clothing", "Shirts"],
            "details": {"Color": "  ", "Material": ["Cotton", "Linen"]},
            "optional": None,
        },
        {
            "parent_asin": "p4",
            "title": "",
            "categories": ["Root", "Clothing"],
        },
        {
            "parent_asin": "p5",
            "title": "Five",
            "categories": "Root",
            "details": {"Color": "Red"},
        },
        {
            "parent_asin": "p2",
            "title": "Duplicate",
            "categories": ["Root", "Shoes"],
            "details": [],
        },
    ]
    valid_lines = [
        json.dumps(product, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for product in products
    ]
    return b"\n".join([*valid_lines, b"not json", b"[1,2]", b""]) + b"\n"


@pytest.fixture
def synthetic_catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.jsonl"
    path.write_bytes(_synthetic_catalog_bytes())
    return path


def test_profile_is_read_only_deterministic_and_bound_to_raw_sha256(
    synthetic_catalog: Path,
) -> None:
    before = synthetic_catalog.read_bytes()
    assignments_a: list[ProductCategoryAssignment] = []
    assignments_b: list[ProductCategoryAssignment] = []
    config = ProfileConfig(seed="test-seed", sample_limit=2, top_value_limit=3)

    first = profile_catalog(synthetic_catalog, config=config, assignment_sink=assignments_a.append)
    second = profile_catalog(synthetic_catalog, config=config, assignment_sink=assignments_b.append)

    assert first == second
    assert assignments_a == assignments_b
    assert synthetic_catalog.read_bytes() == before
    expected_hash = hashlib.sha256(before).hexdigest()
    assert first.catalog_sha256 == expected_hash
    assert catalog_file_sha256(synthetic_catalog) == expected_hash
    assert first.file_size_bytes == len(before)
    assert first.physical_line_count == 9
    assert first.product_row_count == 6
    assert first.invalid_record_count == 3
    assert first.category_assignment_count == 6
    assert first.valid_category_assignment_count == 5
    assert first.product_row_with_diagnostics_count == 3
    assert first.unique_parent_asin_count == 5
    assert len(assignments_a) == 6


def test_top_level_and_raw_detail_statistics_preserve_nested_json_values(
    synthetic_catalog: Path,
) -> None:
    profile = profile_catalog(
        synthetic_catalog,
        config=ProfileConfig(seed="stats", sample_limit=10, top_value_limit=10),
    )
    fields = {field.field: field for field in profile.top_level_fields}
    assert fields["details"].present_count == 5
    assert fields["details"].missing_count == 1
    assert fields["details"].empty_count == 1
    assert {item.value_type: item.count for item in fields["details"].type_counts} == {
        "array": 1,
        "object": 4,
    }
    assert fields["title"].empty_count == 1
    assert fields["optional"].null_count == 1
    assert fields["optional"].missing_count == 5

    details = {detail.raw_key: detail for detail in profile.detail_keys}
    color = details["Color"]
    assert color.support_count == 4
    assert color.nonempty_count == 3
    assert color.empty_count == 1
    assert color.null_count == 0
    assert color.distinct_value_count == 3
    assert color.distinct_nonempty_value_count == 2
    assert color.top_values[0].canonical_value_json == '"Red"'
    assert color.top_values[0].count == 2

    nested = details["Nested"]
    assert {item.value_type: item.count for item in nested.type_counts} == {
        "null": 1,
        "object": 1,
    }
    assert nested.null_count == 1
    assert nested.empty_count == 0
    assert nested.nonempty_count == 1
    assert nested.distinct_value_count == 2
    assert nested.distinct_nonempty_value_count == 1
    assert {item.canonical_value_json for item in nested.top_values} == {
        "null",
        '{"a":1,"b":2}',
    }
    assert details["Tags"].type_counts[0].value_type == "array"
    assert details["Material"].type_counts[0].value_type == "array"


def test_raw_category_tree_uses_full_path_ids_and_exact_support_semantics(
    synthetic_catalog: Path,
) -> None:
    profile = profile_catalog(synthetic_catalog)
    nodes = {node.path: node for node in profile.category_nodes}

    assert tuple(nodes) == (
        ("Root",),
        ("Root", "Clothing"),
        ("Root", "Clothing", "Shirts"),
        ("Root", "Shoes"),
    )
    assert nodes[("Root",)].direct_support == 0
    assert nodes[("Root",)].subtree_support == 5
    assert nodes[("Root", "Clothing")].direct_support == 1
    assert nodes[("Root", "Clothing")].subtree_support == 2
    assert nodes[("Root", "Clothing", "Shirts")].direct_support == 1
    assert nodes[("Root", "Clothing", "Shirts")].subtree_support == 1
    assert nodes[("Root", "Shoes")].direct_support == 3
    assert nodes[("Root", "Shoes")].subtree_support == 3

    expected_root_id = hashlib.sha256(b'["Root"]').hexdigest()
    expected_shoes_id = hashlib.sha256(b'["Root","Shoes"]').hexdigest()
    assert nodes[("Root",)].category_id == expected_root_id
    assert nodes[("Root", "Shoes")].category_id == expected_shoes_id
    assert nodes[("Root", "Shoes")].parent_id == expected_root_id

    unicode_path = ("Clothing, Shoes & Jewelry", "女装")
    payload = json.dumps(
        list(unicode_path),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert category_id_for_path(unicode_path) == hashlib.sha256(payload).hexdigest()


def test_category_subtree_coverage_uses_valid_assignments_as_denominator(
    synthetic_catalog: Path,
) -> None:
    profile = profile_catalog(synthetic_catalog)
    coverage = {(item.category_id, item.raw_key): item for item in profile.category_detail_coverage}
    root_id = category_id_for_path(("Root",))
    shoes_id = category_id_for_path(("Root", "Shoes"))

    root_color = coverage[(root_id, "Color")]
    assert root_color.product_count == 5
    assert root_color.present_count == 3
    assert root_color.nonempty_count == 2
    assert root_color.presence_coverage == pytest.approx(3 / 5)
    assert root_color.nonempty_coverage == pytest.approx(2 / 5)

    shoes_color = coverage[(shoes_id, "Color")]
    assert shoes_color.product_count == 3
    assert shoes_color.present_count == 2
    assert shoes_color.nonempty_count == 2
    assert shoes_color.presence_coverage == pytest.approx(2 / 3)


def test_assignments_and_diagnostics_report_malformed_source_without_coercion(
    synthetic_catalog: Path,
) -> None:
    assignments: list[ProductCategoryAssignment] = []
    profile = profile_catalog(synthetic_catalog, assignment_sink=assignments.append)
    diagnostics = {item.code: item.count for item in profile.diagnostics}

    assert diagnostics == {
        "blank_line": 1,
        "categories_not_array": 1,
        "details_missing": 1,
        "details_not_object": 1,
        "invalid_json": 1,
        "parent_asin_duplicate": 1,
        "row_not_object": 1,
    }
    invalid_category = assignments[4]
    assert invalid_category.parent_asin == "p5"
    assert invalid_category.raw_categories_json == '"Root"'
    assert invalid_category.raw_path == ()
    assert invalid_category.category_node_ids == ()
    assert invalid_category.leaf_category_id is None
    assert invalid_category.category_valid is False
    assert invalid_category.diagnostics == ("categories_not_array",)

    duplicate = assignments[5]
    assert duplicate.category_valid is True
    assert duplicate.diagnostics == ("details_not_object", "parent_asin_duplicate")


def test_stable_samples_use_the_documented_hash_inputs(synthetic_catalog: Path) -> None:
    config = ProfileConfig(seed="sample-seed", sample_limit=10, top_value_limit=2)
    profile = profile_catalog(synthetic_catalog, config=config)
    color = next(detail for detail in profile.detail_keys if detail.raw_key == "Color")

    assert len(color.samples) == 4
    assert tuple(sample.sample_hash for sample in color.samples) == tuple(
        sorted(sample.sample_hash for sample in color.samples)
    )
    for sample in color.samples:
        assert sample.parent_asin is not None
        payload = canonical_json_dumps(
            [profile.catalog_sha256, config.seed, sample.parent_asin, "Color"]
        ).encode("utf-8")
        assert sample.sample_hash == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize("path", [(), ("Root", 1), (None,)])
def test_category_id_rejects_noncanonical_paths(path: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        category_id_for_path(path)  # type: ignore[arg-type]


def test_invalid_category_component_shapes_are_diagnosed(tmp_path: Path) -> None:
    rows = [
        {"parent_asin": "a", "categories": [], "details": {}},
        {"parent_asin": "b", "categories": ["Root", "  "], "details": {}},
        {"parent_asin": "c", "categories": ["Root", 3], "details": {}},
        {"parent_asin": "d", "details": {}},
    ]
    path = tmp_path / "malformed-categories.jsonl"
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )

    profile = profile_catalog(path)

    assert profile.category_nodes == ()
    assert profile.valid_category_assignment_count == 0
    assert {item.code: item.count for item in profile.diagnostics} == {
        "categories_empty": 1,
        "categories_missing": 1,
        "category_component_empty": 1,
        "category_component_not_string": 1,
    }


def test_nonfinite_numbers_and_unpaired_surrogates_are_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "invalid-values.jsonl"
    path.write_bytes(
        b'{"parent_asin":"finite","categories":["Root"],"details":{"x":1.5}}\n'
        b'{"parent_asin":"overflow","categories":["Root"],"details":{"x":1e999}}\n'
        b'{"parent_asin":"constant","categories":["Root"],"details":{"x":NaN}}\n'
        b'{"parent_asin":"unicode","categories":["Root"],"details":{"x":"\\ud800"}}\n'
    )

    profile = profile_catalog(path)

    assert profile.product_row_count == 1
    assert profile.invalid_record_count == 3
    assert {item.code: item.count for item in profile.diagnostics} == {
        "invalid_json": 2,
        "non_canonical_json_value": 1,
    }


def test_duplicate_json_object_keys_are_rejected_without_last_wins(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-keys.jsonl"
    path.write_bytes(
        b'{"parent_asin":"a","categories":["Root"],"details":{"Color":"first","Color":"last"}}\n'
    )

    profile = profile_catalog(path)

    assert profile.product_row_count == 0
    assert profile.invalid_record_count == 1
    assert profile.detail_keys == ()
    assert {item.code: item.count for item in profile.diagnostics} == {"duplicate_json_key": 1}


def test_excessively_nested_json_is_diagnosed_instead_of_crashing(tmp_path: Path) -> None:
    path = tmp_path / "deep.jsonl"
    nesting = 5_000
    path.write_bytes(b"[" * nesting + b"0" + b"]" * nesting + b"\n")

    profile = profile_catalog(path)

    assert profile.product_row_count == 0
    assert profile.invalid_record_count == 1
    assert {item.code: item.count for item in profile.diagnostics} == {"invalid_json": 1}
