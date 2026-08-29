"""Deterministic human review reports for CS1 proposal and candidate bundles."""

from __future__ import annotations

from collections import Counter

from .models import CategoryCandidateBuild, CategoryGraphProposal


def category_graph_proposal_markdown(proposal: CategoryGraphProposal) -> str:
    """Render a stable Pass-A graph and collision review summary."""

    subtree_support_by_node: Counter[str] = Counter()
    for mapping in proposal.raw_path_mappings:
        subtree_support_by_node[mapping.node_id] += mapping.subtree_product_count
    roots = tuple(node for node in proposal.nodes if node.parent_id is None)
    lines = [
        "# Category Graph Proposal",
        "",
        "> Review artifact only. This is not a CategoryRegistry or semantic release.",
        "",
        "## Source identity",
        "",
        f"- Catalog: `{proposal.catalog_id}`",
        f"- Catalog bytes: {proposal.catalog_byte_size:,}",
        f"- Products: {proposal.product_count:,}",
        f"- Builder: `{proposal.builder_version}`",
        f"- Unicode data: `{proposal.unicode_data_version}`",
        f"- Category graph: `{proposal.category_graph_id}`",
        "",
        "## Graph summary",
        "",
        f"- Exact raw prefixes: {proposal.raw_prefix_count:,}",
        f"- Canonical nodes: {len(proposal.nodes):,}",
        f"- Canonical roots: {len(roots):,}",
        f"- Normalization collision groups: {len(proposal.collisions):,}",
        "",
        "## Canonical roots",
        "",
        "| Canonical path | Node ID | Product subtree support |",
        "| --- | --- | ---: |",
    ]
    for root in sorted(roots, key=lambda node: node.canonical_path):
        lines.append(
            "| "
            f"{_markdown_path(root.canonical_path)} | `{root.id}` | "
            f"{subtree_support_by_node[root.id]:,} |"
        )

    lines.extend(("", "## Normalization collision audit", ""))
    if not proposal.collisions:
        lines.append("No distinct raw full-path prefixes collapse to one canonical path.")
    else:
        lines.extend(
            (
                "| Canonical path | Distinct raw paths |",
                "| --- | --- |",
            )
        )
        for collision in proposal.collisions:
            raw_paths = "<br>".join(_markdown_path(path) for path in collision.raw_paths)
            lines.append(f"| {_markdown_path(collision.canonical_path)} | {raw_paths} |")

    lines.extend(
        (
            "",
            "## Human checkpoint",
            "",
            "1. Accept or reject the closed canonical graph and collision audit.",
            "2. If accepted, create a `shopping-copilot/category-scope-selection/v0` file.",
            "3. Include one reviewed root scope using all canonical root IDs.",
            "4. Add only user-facing scopes whose complete subtree unions are intended.",
            "",
        )
    )
    return "\n".join(lines)


def category_candidate_markdown(candidate: CategoryCandidateBuild) -> str:
    """Render stable Pass-B category scope and assignment acceptance stats."""

    status_counts = Counter(item.status.value for item in candidate.assignments.assignments)
    registry = candidate.registry
    root_scope = next(scope for scope in registry.scopes if scope.id == registry.root_scope_id)
    lines = [
        "# Category Candidate Build",
        "",
        "> Candidate artifacts only. They are not a CatalogSemanticRelease.",
        "",
        f"- Catalog: `{registry.catalog_id}`",
        f"- Category graph: `{registry.category_graph_id}`",
        f"- Builder: `{candidate.builder_version}`",
        f"- Canonical nodes: {len(registry.nodes):,}",
        f"- Reviewed scopes: {len(registry.scopes):,}",
        f"- Assignments: {len(candidate.assignments.assignments):,}",
        f"- KNOWN: {status_counts['known']:,}",
        f"- UNKNOWN: {status_counts['unknown']:,}",
        f"- CONFLICT: {status_counts['conflict']:,}",
        "",
        "## Materialized scopes",
        "",
        "| Label | Scope ID | Roots | Members | Root scope |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for scope in registry.scopes:
        lines.append(
            "| "
            f"{_escape_markdown(scope.label)} | `{scope.id}` | "
            f"{len(scope.root_node_ids):,} | {len(scope.member_node_ids):,} | "
            f"{'yes' if scope.id == root_scope.id else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_path(path: tuple[str, ...]) -> str:
    return " &gt; ".join(_escape_markdown(item) for item in path)


def _escape_markdown(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        codepoint = ord(character)
        if codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F:
            escaped.append(f"\\u{codepoint:04x}")
        elif character == "|":
            escaped.append("\\|")
        else:
            escaped.append(character)
    return "".join(escaped)
