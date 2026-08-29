"""Deterministic human-review report for CS2 source-profile proposals."""

from __future__ import annotations

import unicodedata

from .models import (
    GateASourceProfileBuild,
    ScopeSourceProfile,
    SourceKind,
    SourceLocator,
    TypeCount,
)


def gate_a_source_profile_markdown(build: GateASourceProfileBuild) -> str:
    """Render the Gate-A observation checkpoint without approving any facet."""

    if type(build) is not GateASourceProfileBuild:
        raise TypeError("report requires GateASourceProfileBuild")
    root_scope = next(scope for scope in build.scopes if scope.is_root)
    by_source_scope = {
        (item.source, item.category_scope_id): item for item in build.scope_source_profiles
    }
    root_profiles = {
        source: by_source_scope[(source, root_scope.category_scope_id)] for source in build.sources
    }
    detail_sources = tuple(source for source in build.sources if source.kind is SourceKind.DETAILS)
    lines = [
        "# CS2 Gate-A Structured Source Profile",
        "",
        "> Observation proposal only. No facet, applicability, binding, extractor, or runtime "
        "capability is approved by this bundle.",
        "",
        f"- Catalog: `{build.catalog_id}`",
        f"- CategoryRegistry: `{build.category_registry_id}`",
        f"- ProductCategoryAssignmentSet: `{build.product_category_assignment_id}`",
        f"- Builder: `{build.builder_version}`",
        f"- Reviewed category scopes: {len(build.scopes):,}",
        f"- Profiled exact source locators: {len(build.sources):,}",
        f"- Exact details keys: {len(detail_sources):,}",
        f"- Scope × source rows: {len(build.scope_source_profiles):,}",
        f"- Stable nonempty samples: {len(build.samples):,}",
        f"- Sample seed: `{_escape(build.selection.sample_seed)}`",
        "- Gate-A decisions: **all remain `NEEDS_REVIEW`**",
        "",
        "## Top-level catalog inventory",
        "",
        "| Exact field | Present | Missing | Null | Empty | JSON types | Gate-A lane |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    selected_top_level = set(build.selection.top_level_keys)
    for field in build.top_level_fields:
        lane = "profiled" if field.key in selected_top_level else _top_level_boundary(field.key)
        lines.append(
            f"| {_escape(field.key)} | {field.present_count:,} | {field.missing_count:,} | "
            f"{field.null_count:,} | {field.empty_count:,} | "
            f"{_escape(_render_types(field.type_counts))} | {_escape(lane)} |"
        )

    price = build.price_audit
    lines.extend(
        (
            "",
            "## Priority lane 1: exact top-level `price`",
            "",
            "Decision: **NEEDS_REVIEW**. These are observations, not an approved extractor.",
            "",
            f"- Present: {price.present_count:,}",
            f"- Null / missing value: {price.null_count:,}",
            f"- JSON numeric: {price.numeric_count:,}",
            f"- Non-negative numeric values exactly representable as integer cents: "
            f"{price.numeric_exact_cent_count:,}",
            f"- Numeric values requiring rounding or otherwise invalid for exact cents: "
            f"{price.numeric_non_cent_count:,}",
            f"- String values: {price.string_count:,}",
            f"- Other JSON types: {price.other_count:,}",
            (
                "- Exact-cent range: "
                + (
                    "_none_"
                    if price.minimum_exact_cents is None
                    else f"{price.minimum_exact_cents:,}–{price.maximum_exact_cents:,} cents"
                )
            ),
            "",
            "### Exact string lane",
            "",
            "| Canonical raw string JSON | Count |",
            "| --- | ---: |",
        )
    )
    for item in price.string_values:
        lines.append(f"| `{_escape(item.canonical_value_json)}` | {item.count:,} |")
    if not price.string_values:
        lines.append("| _none_ | 0 |")
    lines.extend(
        (
            "",
            "The string lane must be classified explicitly before approval. The profiler does not "
            "strip currency text, interpret `from`, replace invalid glyphs, or round values.",
            "",
            "### Category-conditioned price presence",
            "",
            "| Scope | Products | Nonempty price | Coverage |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    price_locator = SourceLocator(kind=SourceKind.TOP_LEVEL, key="price")
    for scope in build.scopes:
        profile = by_source_scope[(price_locator, scope.category_scope_id)]
        lines.append(
            f"| {_escape(scope.label)} | {scope.product_count:,} | {profile.nonempty_count:,} | "
            f"{_coverage(profile.nonempty_count, scope.product_count)} |"
        )

    lines.extend(
        (
            "",
            "## Priority lane 2: store / brand-like exact sources",
            "",
            '`store`, `details["Brand"]`, and `details["Brand Name"]` are shown as separate '
            "source lanes. Name similarity or value overlap does not merge them.",
            "",
        )
    )
    _append_source_table(
        lines,
        build=build,
        root_profiles=root_profiles,
        sources=(
            SourceLocator(kind=SourceKind.TOP_LEVEL, key="store"),
            SourceLocator(kind=SourceKind.DETAILS, key="Brand"),
            SourceLocator(kind=SourceKind.DETAILS, key="Brand Name"),
        ),
    )

    lines.extend(
        (
            "",
            "## Priority lane 3: exact `Department`",
            "",
            "Decision: **NEEDS_REVIEW**. High support does not decide whether this is an ordinary "
            "facet, taxonomy evidence, or target-audience metadata.",
            "",
        )
    )
    _append_source_table(
        lines,
        build=build,
        root_profiles=root_profiles,
        sources=(SourceLocator(kind=SourceKind.DETAILS, key="Department"),),
    )

    color_sources = tuple(source for source in detail_sources if _is_color_key(source.key))
    material_sources = tuple(source for source in detail_sources if _is_material_key(source.key))
    size_sources = tuple(source for source in detail_sources if _is_size_key(source.key))
    for heading, explanation, sources in (
        (
            "Priority lane 4: color-like lexical group",
            "Lexical grouping is only a review queue. Every exact raw key would require its own "
            "binding, and combined-color strings are not split here.",
            color_sources,
        ),
        (
            "Priority lane 5: material/fabric lexical group",
            "`Material`, `Fabric Type`, outer/inner/frame/band sources are not automatically merged; "
            "their semantics must be reviewed by scope.",
            material_sources,
        ),
        (
            "Priority lane 6: size/dimension lexical group",
            "This queue intentionally exposes leakage: product dimensions, package dimensions, "
            "capacity, apparel size, ring size, and band size cannot become one global facet.",
            size_sources,
        ),
    ):
        lines.extend(("", f"## {heading}", "", f"Decision: **NEEDS_REVIEW**. {explanation}", ""))
        _append_source_table(
            lines,
            build=build,
            root_profiles=root_profiles,
            sources=sources,
        )

    ranked_details = sorted(
        (root_profiles[source] for source in detail_sources),
        key=lambda item: (-item.nonempty_count, item.source.key),
    )
    lines.extend(
        (
            "",
            "## Largest exact details sources",
            "",
            "The JSONL artifact contains all exact keys, including low-support keys. This table is "
            "only a compact navigation view.",
            "",
            "| Exact key | Nonempty | Root coverage | Distinct nonempty | Dominant ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for profile in ranked_details[:40]:
        lines.append(
            f"| {_escape(profile.source.key)} | {profile.nonempty_count:,} | "
            f"{_coverage(profile.nonempty_count, profile.product_count)} | "
            f"{profile.distinct_nonempty_value_count:,} | {_dominant_ratio(profile)} |"
        )

    lines.extend(
        (
            "",
            "## Review boundary",
            "",
            "This bundle may support a human decision, but it cannot promote a facet. Gate A still "
            "requires an explicit definition, applicability scopes, one binding per exact source "
            "key, closed implementation IDs, priority/completeness, real-sample review, and an "
            "`EXTRACTION_APPROVED` or `REJECT` decision. Coverage thresholds are signals only.",
            "",
            "Recommended next review order: `price`, then the three store/brand lanes, then "
            "`Department`, color, material-family sources, and category-specific size domains.",
            "",
        )
    )
    return "\n".join(lines)


def _append_source_table(
    lines: list[str],
    *,
    build: GateASourceProfileBuild,
    root_profiles: dict[SourceLocator, ScopeSourceProfile],
    sources: tuple[SourceLocator, ...],
) -> None:
    lines.extend(
        (
            "| Exact source | Nonempty | Root coverage | Distinct | JSON types | Top reviewed scopes |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        )
    )
    observed = set(build.sources)
    for source in sources:
        if source not in observed:
            lines.append(f"| `{_render_locator(source)}` | 0 | 0.000% | 0 | _absent_ | _none_ |")
            continue
        profile = root_profiles[source]
        lines.append(
            f"| `{_escape(_render_locator(source))}` | {profile.nonempty_count:,} | "
            f"{_coverage(profile.nonempty_count, profile.product_count)} | "
            f"{profile.distinct_nonempty_value_count:,} | "
            f"{_escape(_render_types(profile.type_counts))} | "
            f"{_escape(_top_scopes(build, source))} |"
        )
    if not sources:
        lines.append("| _none_ | 0 | 0.000% | 0 | _none_ | _none_ |")


def _top_scopes(build: GateASourceProfileBuild, source: SourceLocator) -> str:
    scope_by_id = {scope.category_scope_id: scope for scope in build.scopes}
    ranked = sorted(
        (
            item
            for item in build.scope_source_profiles
            if item.source == source
            and not scope_by_id[item.category_scope_id].is_root
            and item.nonempty_count
        ),
        key=lambda item: (
            -(item.nonempty_count / item.product_count),
            -item.nonempty_count,
            scope_by_id[item.category_scope_id].label,
        ),
    )
    return (
        ", ".join(
            f"{scope_by_id[item.category_scope_id].label} "
            f"{_coverage(item.nonempty_count, item.product_count)} ({item.nonempty_count})"
            for item in ranked[:3]
        )
        or "none"
    )


def _render_locator(source: SourceLocator) -> str:
    return source.key if source.kind is SourceKind.TOP_LEVEL else f'details["{source.key}"]'


def _render_types(values: tuple[TypeCount, ...]) -> str:
    return ", ".join(f"{item.value_type}:{item.count}" for item in values)


def _coverage(numerator: int, denominator: int) -> str:
    return "n/a" if denominator == 0 else f"{numerator / denominator:.3%}"


def _dominant_ratio(profile: ScopeSourceProfile) -> str:
    if profile.nonempty_count == 0:
        return "n/a"
    return f"{profile.dominant_nonempty_value_count / profile.nonempty_count:.3%}"


def _top_level_boundary(key: str) -> str:
    if key in {"title", "features", "description"}:
        return "P0 binding prohibited (free text)"
    if key in {"average_rating", "rating_number"}:
        return "ranking signal; not selected"
    if key == "categories":
        return "owned by CS1 category"
    if key == "parent_asin":
        return "identifier"
    if key == "details":
        return "container; exact child keys profiled"
    return "not selected"


def _is_color_key(key: str) -> bool:
    folded = key.casefold()
    return "color" in folded or "colour" in folded


def _is_material_key(key: str) -> bool:
    folded = key.casefold()
    return "material" in folded or "fabric" in folded


def _is_size_key(key: str) -> bool:
    folded = key.casefold()
    return any(token in folded for token in ("size", "width", "dimension", "capacity"))


def _escape(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        if character == "\\":
            escaped.append("\\\\")
        elif character == "|":
            escaped.append("\\|")
        elif character in ("\r", "\n"):
            escaped.append(" ")
        elif unicodedata.category(character) in ("Cc", "Zl", "Zp"):
            escaped.append(f"\\u{ord(character):04x}")
        else:
            escaped.append(character)
    return "".join(escaped)
