"""Human-readable Gate-B price review report that grants no runtime permission."""

from __future__ import annotations

from collections.abc import Callable

from ..canonical import content_id_for_value
from .gate_a_models import EvidenceStatus, NumericValue
from .gate_b_build import GateBPriceReviewBuild
from .resolution_models import FacetValueEvidence, ResolutionCandidateBuild


def gate_b_price_review_markdown(
    build: GateBPriceReviewBuild,
    *,
    resolution: ResolutionCandidateBuild,
) -> str:
    """Render the decision packet in plain language for repository-owner review."""

    proposal = build.proposal
    audit = build.public_target_audit
    scope_count = len(proposal.proposed_capabilities)
    lines = [
        "# Gate B 价格能力审核材料 v0",
        "",
        "## 现在的状态",
        "",
        "- **等待仓库所有者批准**",
        "- **尚未发布任何运行时权限**",
        "- **没有修改原始商品数据，也没有修改 session context 或检索器**",
        f"- 提案 ID：`{content_id_for_value(proposal)}`",
        f"- 公共目标审计 ID：`{content_id_for_value(audit)}`",
        "",
        "## 要批准的内容",
        "",
        f"对 {scope_count} 个已发布商品类别范围，逐行提议同一组价格权限：",
        "",
        "- 用户明确说出预算时，允许把它记录为结构化价格条件；",
        "- 允许检索使用价格条件，但只能删除已经证明超预算的商品；",
        "- 允许内部查看价格分布，为后续决策提供证据；",
        "- 暂不允许系统主动询问用户预算；",
        "- 每个范围都是单独的一行，不从父类别或其他范围继承权限。",
        "",
        "输入价格的拟定边界是整数 `USD_CENT`（美分），对应的未来运行时",
        f"normalizer ID 是 `{proposal.proposed_intent_normalizer_id}`。`$25` 之类的自然",
        "语言解析属于 Query Understanding；交给该 normalizer 之前必须已经变成 `2500`。",
        "价格是数值型，因此 reviewed value aliases 为空。",
        "",
        "## 为什么未知价格不能直接删掉",
        "",
        f"公共集有 {audit.target_count} 个目标商品，其中 {audit.known_count} 个有可用价格，",
        f"{audit.unknown_count} 个价格未知。采用安全规则时 {audit.compatible_budget_safe_retained_count}",
        "个目标都能保留；如果错误地只留下‘已知且满足预算’的商品，则只能保留",
        f"{audit.unsafe_satisfied_only_retained_count} 个，会误删价格未知的目标。",
        "",
        "这里使用的是兼容目标价格的合成预算，只验证‘未知值不会被误当作超预算’。",
        "公共 competition simulator 没有真实用户预算文本，因此它不能证明主动询问预算有收益。",
        "这也是本提案把主动询问保持关闭的原因。",
        "",
        "## 全目录预算安全检查",
        "",
        "| 预算上限 | 明确满足 | 明确超出（可删除） | 无法判断（必须保留） | 安全规则保留 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget_row in proposal.budget_safety_rows:
        lines.append(
            f"| ${budget_row.budget_cents / 100:,.2f} | {budget_row.satisfied_count:,} | "
            f"{budget_row.violated_count:,} | {budget_row.unknown_count:,} | "
            f"{budget_row.safe_retained_count:,} |"
        )
    lines.extend(
        [
            "",
            "`无法判断` 包括商品价格缺失，也包括价格区间与预算只有部分重叠的情况。",
            "安全规则只删除 `明确超出`。",
            "",
            "## 各类别范围",
            "",
            "| 范围 | 商品 | 已知 | 未知 | 冲突 | 不适用 | 公共目标（已知） | 中位价格下界 | P90 下界 | 提议 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for scope_row in proposal.scope_reviews:
        known_public = f"{scope_row.public_target_count} ({scope_row.public_target_known_count})"
        lines.append(
            f"| {_cell(scope_row.scope_label)} | {scope_row.scope_product_count:,} | "
            f"{scope_row.known_count:,} | {scope_row.unknown_count:,} | "
            f"{scope_row.conflict_count:,} | {scope_row.not_applicable_count:,} | "
            f"{known_public} | {_money(scope_row.median_lower_cents)} | "
            f"{_money(scope_row.p90_lower_cents)} | "
            "RUNTIME_ACCEPT；不主动询问 |"
        )
    lines.extend(
        [
            "",
            "机器文件同时保留精确 scope ID。表格用标签是为了方便人工阅读。",
            "",
            "## 可追溯样本",
            "",
            "| 情况 | 商品 | 原始 price | 解析结果 | evidence ID |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for label, evidence in _stable_examples(resolution):
        result = evidence.status.value.upper()
        if type(evidence.canonical_value) is NumericValue:
            result += f" {_interval(evidence.canonical_value)}"
        lines.append(
            f"| {label} | `{evidence.parent_asin}` | `{_code(evidence.raw_value_json)}` | "
            f"{result} | `{evidence.id}` |"
        )
    lines.extend(
        [
            "",
            "## 请确认",
            "",
            f"是否批准这 {scope_count} 行方案：允许记录明确预算、允许保留未知值的安全价格检索、",
            "允许内部价格统计，但暂不允许系统主动询问预算？",
            "",
            "如果批准，下一步才会把决定写成 source-controlled Gate-B selection，随后",
            "实现并测试运行时 capability artifact。直接编辑本目录里的生成文件不算批准。",
            "",
        ]
    )
    return "\n".join(lines)


def _stable_examples(
    resolution: ResolutionCandidateBuild,
) -> tuple[tuple[str, FacetValueEvidence], ...]:
    predicates: tuple[tuple[str, Callable[[FacetValueEvidence], bool]], ...] = (
        (
            "精确价格",
            lambda item: (
                type(item.canonical_value) is NumericValue
                and item.canonical_value.upper is not None
            ),
        ),
        (
            "只有下界",
            lambda item: (
                type(item.canonical_value) is NumericValue and item.canonical_value.upper is None
            ),
        ),
        ("空值", lambda item: item.status is EvidenceStatus.EMPTY),
        ("无效占位符", lambda item: item.status is EvidenceStatus.INVALID),
    )
    result: list[tuple[str, FacetValueEvidence]] = []
    for label, predicate in predicates:
        evidence = next(
            (item for item in resolution.evidence_store.evidence if predicate(item)),
            None,
        )
        if evidence is not None:
            result.append((label, evidence))
    return tuple(result)


def _money(value: int | None) -> str:
    return "—" if value is None else f"${value / 100:,.2f}"


def _interval(value: NumericValue) -> str:
    upper = "+∞" if value.upper is None else str(value.upper)
    return f"[{value.lower}, {upper}] {value.unit}"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _code(value: str) -> str:
    return value.replace("`", "\\`").replace("|", "\\|")
