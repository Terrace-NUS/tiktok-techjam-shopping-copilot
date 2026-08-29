from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

from shopping_copilot.catalog.semantic import canonical_json_bytes
from shopping_copilot.catalog.semantic.category import cli as category_cli


def _write_catalog(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog.jsonl"
    rows = (
        {"parent_asin": "p-shoes", "categories": ["Clothing", "Shoes"], "details": {}},
        {"parent_asin": "p-clothing", "categories": ["Clothing"], "details": {}},
        {"parent_asin": "p-belt", "categories": ["Accessories", "Belts"], "details": {}},
    )
    catalog.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    return catalog


def _allow_small_proposal_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    real_writer = category_cli.write_category_graph_proposal_bundle
    monkeypatch.setattr(
        category_cli,
        "write_category_graph_proposal_bundle",
        partial(real_writer, expected_product_count=3),
    )


def test_cli_propose_then_validate_small_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = _write_catalog(tmp_path)
    output = tmp_path / "proposal"
    _allow_small_proposal_fixture(monkeypatch)

    assert category_cli.main(("propose", str(catalog), str(output))) == 0
    proposed = capsys.readouterr()
    assert "proposal 3 products" in proposed.out
    assert "4 canonical nodes" in proposed.out
    assert "graph=sha256:" in proposed.out
    assert proposed.err == ""

    assert category_cli.main(("validate", str(output))) == 0
    validated = capsys.readouterr()
    assert f"valid bundle {output}" in validated.out
    assert validated.err == ""


def test_cli_validate_reports_tampered_bundle_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = _write_catalog(tmp_path)
    output = tmp_path / "proposal"
    _allow_small_proposal_fixture(monkeypatch)
    assert category_cli.main(("propose", str(catalog), str(output))) == 0
    capsys.readouterr()
    manifest_path = output / "bundle-manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(SystemExit) as raised:
        category_cli.main(("validate", str(output)))

    assert raised.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "catalog-category: error:" in captured.err
