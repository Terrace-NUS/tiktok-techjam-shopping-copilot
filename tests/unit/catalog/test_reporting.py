from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import shopping_copilot.catalog.profiling.bundle as bundle_module
from shopping_copilot.catalog.profiling import (
    CanonicalAssignmentJsonlSink,
    ProductCategoryAssignment,
    ProfileBundleBusyError,
    ProfileBundleIntegrityError,
    ProfileConfig,
    catalog_profile_to_json,
    catalog_profile_to_markdown,
    product_category_assignment_to_json,
    profile_catalog,
    validate_profile_bundle,
    write_catalog_profile_json,
    write_category_detail_coverage_jsonl,
    write_category_nodes_jsonl,
    write_detail_keys_jsonl,
    write_profile_bundle,
)
from shopping_copilot.catalog.profiling.cli import main


def _catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"parent_asin":"a","categories":["Root"],"details":{"Color":"Red"}}',
                (
                    '{"parent_asin":"b","categories":["Root","Shoes"],'
                    '"details":{"Color":"Blue","Nested":{"z":1,"a":2}}}'
                ),
            )
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return path


def test_canonical_json_and_markdown_reports_are_deterministic(tmp_path: Path) -> None:
    profile = profile_catalog(
        _catalog(tmp_path),
        config=ProfileConfig(seed="report", sample_limit=2, top_value_limit=2),
    )

    first_json = catalog_profile_to_json(profile)
    second_json = catalog_profile_to_json(profile)
    assert first_json == second_json
    assert json.loads(first_json)["catalog_sha256"] == profile.catalog_sha256
    assert "\n" not in first_json

    first_markdown = catalog_profile_to_markdown(profile)
    second_markdown = catalog_profile_to_markdown(profile)
    assert first_markdown == second_markdown
    assert f"`{profile.catalog_sha256}`" in first_markdown
    assert "Root > Shoes" in first_markdown
    assert "| Color | 2 | 2 |" in first_markdown


def test_markdown_escapes_table_delimiters_and_control_characters(tmp_path: Path) -> None:
    path = tmp_path / "controls.jsonl"
    path.write_text(
        json.dumps(
            {
                "parent_asin": "a",
                "categories": ["Root\rBad"],
                "details": {"Bad\t|\rKey": "value"},
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )

    markdown = catalog_profile_to_markdown(profile_catalog(path))

    assert "Root\\u000dBad" in markdown
    assert "Bad\\u0009\\|\\u000dKey" in markdown


def test_json_and_jsonl_writers_emit_one_canonical_record_per_line(tmp_path: Path) -> None:
    assignments: list[ProductCategoryAssignment] = []
    profile = profile_catalog(_catalog(tmp_path), assignment_sink=assignments.append)

    profile_stream = io.StringIO()
    write_catalog_profile_json(profile, profile_stream)
    assert profile_stream.getvalue() == catalog_profile_to_json(profile) + "\n"

    node_stream = io.StringIO()
    write_category_nodes_jsonl(profile, node_stream)
    node_lines = node_stream.getvalue().splitlines()
    assert len(node_lines) == len(profile.category_nodes)
    assert [json.loads(line)["category_id"] for line in node_lines] == [
        node.category_id for node in profile.category_nodes
    ]

    details_stream = io.StringIO()
    write_detail_keys_jsonl(profile, details_stream)
    detail_lines = details_stream.getvalue().splitlines()
    assert len(detail_lines) == len(profile.detail_keys)
    assert [json.loads(line)["raw_key"] for line in detail_lines] == [
        detail.raw_key for detail in profile.detail_keys
    ]

    coverage_stream = io.StringIO()
    write_category_detail_coverage_jsonl(profile, coverage_stream)
    assert len(coverage_stream.getvalue().splitlines()) == len(profile.category_detail_coverage)

    assignment_stream = io.StringIO()
    sink = CanonicalAssignmentJsonlSink(assignment_stream)
    for assignment in assignments:
        sink(assignment)
    assert assignment_stream.getvalue().splitlines() == [
        product_category_assignment_to_json(assignment) for assignment in assignments
    ]


def test_profile_bundle_is_complete_and_byte_deterministic(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "artifacts" / "profile"
    config = ProfileConfig(seed="bundle", sample_limit=3, top_value_limit=4)

    first = write_profile_bundle(catalog, output, config=config)
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = write_profile_bundle(catalog, output, config=config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first == second
    assert first_bytes == second_bytes
    validate_profile_bundle(output)
    assert set(first_bytes) == {
        "bundle-manifest.json",
        "profile.json",
        "report.md",
        "category-nodes.jsonl",
        "detail-keys.jsonl",
        "category-detail-coverage.jsonl",
        "product-category-assignments.jsonl",
    }
    assert len(first_bytes["product-category-assignments.jsonl"].splitlines()) == 2

    (output / "detail-keys.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProfileBundleIntegrityError, match="integrity"):
        validate_profile_bundle(output)


def test_cli_writes_bundle_and_prints_source_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "profile"

    assert main((str(catalog), str(output), "--sample-limit", "1")) == 0

    captured = capsys.readouterr()
    profile = json.loads((output / "profile.json").read_text(encoding="utf-8"))
    assert f"sha256={profile['catalog_sha256']}" in captured.out
    assert captured.err == ""


def test_profile_bundle_refuses_to_overwrite_its_source(tmp_path: Path) -> None:
    output = tmp_path / "profile"
    output.mkdir()
    catalog = output / "profile.json"
    original = b'{"parent_asin":"a","categories":["Root"],"details":{}}\n'
    catalog.write_bytes(original)

    with pytest.raises(ValueError, match="collides"):
        write_profile_bundle(catalog, output)

    assert catalog.read_bytes() == original


def test_profile_bundle_refuses_a_concurrent_writer(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "profile"
    lock_path = tmp_path / ".profile.write.lock"
    lock_path.touch()

    with pytest.raises(ProfileBundleBusyError, match="already being written"):
        write_profile_bundle(catalog, output)

    assert not output.exists()


def test_interrupted_bundle_publication_is_not_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _catalog(tmp_path)
    output = tmp_path / "profile"
    real_replace = bundle_module.os.replace
    replacement_count = 0

    def fail_third_replace(source: Path, target: Path) -> None:
        nonlocal replacement_count
        replacement_count += 1
        if replacement_count == 3:
            raise PermissionError("simulated Windows file lock")
        real_replace(source, target)

    monkeypatch.setattr(bundle_module.os, "replace", fail_third_replace)

    with pytest.raises(PermissionError, match="simulated"):
        write_profile_bundle(catalog, output)
    with pytest.raises(ProfileBundleIntegrityError):
        validate_profile_bundle(output)
    assert not (tmp_path / ".profile.write.lock").exists()
