"""Build the expanded hand-authored intent-space prompt suite."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY_ROOT / "config/query_understanding/intent-space-natural-prompts-v1.json"
OUTPUT = REPOSITORY_ROOT / "config/query_understanding/intent-space-natural-prompts-v2.json"


def _turn(
    user_message: str,
    *,
    assistant: str | None = None,
    question: str | None = None,
) -> dict[str, object]:
    return {
        "user_message": user_message,
        "last_assistant_message": assistant,
        "last_question": question,
        "shown_products": [],
        "critical_assertions": [],
    }


def _conversation(
    identifier: str,
    *,
    language: str,
    domain: str,
    tags: list[str],
    turns: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "id": identifier,
        "tier": "full",
        "language": language,
        "domain": domain,
        "tags": ["intent_space", *tags],
        "initial_goal": None,
        "turns": turns,
    }


ADDITIONS = [
    _conversation(
        "n17_progressive_trail_shoes",
        language="en",
        domain="footwear",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I need shoes for weekend trails, but I haven't decided what kind yet."),
            _turn(
                "Probably men's trail-running shoes rather than hiking boots, size 10.",
                assistant="Let's narrow down the trail footwear.",
                question="Are you after boots or running shoes, and what size?",
            ),
            _turn(
                "Make them waterproof, wide fit, dark grey, with strong grip, under $150.",
                assistant="I'll focus on men's size 10 trail-running shoes.",
                question="Any weather protection, fit, color, sole, or budget requirements?",
            ),
        ],
    ),
    _conversation(
        "n18_progressive_work_tote",
        language="en",
        domain="handbags",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("Could you help me find a nicer bag for work?"),
            _turn(
                "I'd prefer a structured tote that can hold a 14-inch laptop.",
                assistant="What style and capacity should the work bag have?",
                question="What kind of bag and what must fit inside?",
            ),
            _turn(
                "Black pebbled leather, zip closure, silver hardware, no visible logo, up to $170.",
                assistant="I'll look for a structured laptop tote.",
                question="What material, color, closure, branding, and budget do you want?",
            ),
        ],
    ),
    _conversation(
        "n19_progressive_carry_on",
        language="en",
        domain="luggage",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I'm replacing my travel suitcase."),
            _turn(
                "It needs to be a carry-on, not checked luggage, preferably hard-sided.",
                assistant="What type of suitcase do you need?",
                question="Carry-on or checked, and soft or hard shell?",
            ),
            _turn(
                "Twenty inches, matte black, spinner wheels, TSA lock, expandable, below $130.",
                assistant="I'll focus on hard-shell carry-ons.",
                question="What size, color, features, and budget matter?",
            ),
        ],
    ),
    _conversation(
        "n20_progressive_anniversary_necklace",
        language="en",
        domain="jewelry",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I want a small anniversary gift for my wife."),
            _turn(
                "Let's make it a necklace, something understated rather than flashy.",
                assistant="What kind of anniversary gift would she wear?",
                question="Could you choose a jewelry type and overall style?",
            ),
            _turn(
                "A 14k yellow-gold heart pendant on an 18-inch chain, no diamonds, under $250.",
                assistant="I'll look for understated necklaces.",
                question="What metal, motif, chain length, stones, and budget?",
            ),
        ],
    ),
    _conversation(
        "n21_progressive_winter_boots",
        language="en",
        domain="footwear",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I need something warmer for my feet this winter."),
            _turn(
                "Women's winter boots, mostly for walking around the city, size 8.",
                assistant="What kind of warm footwear and what size?",
                question="Where will you use it, and what size do you wear?",
            ),
            _turn(
                "Mid-calf, waterproof, black, faux-fur lining, flat sole, under $120—and no suede.",
                assistant="I'll search for women's city winter boots in size 8.",
                question="What height, weather protection, color, lining, sole, material, and budget?",
            ),
        ],
    ),
    _conversation(
        "n22_progressive_everyday_watch",
        language="en",
        domain="watches",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I'd like a watch that works with everyday clothes."),
            _turn(
                "An analog men's watch, simple enough for the office.",
                assistant="What sort of everyday watch do you picture?",
                question="Analog or digital, and what style?",
            ),
            _turn(
                "Blue dial, stainless-steel bracelet, 38 to 41 mm, water resistant, below $180.",
                assistant="I'll focus on simple men's analog watches.",
                question="What dial, band, case size, features, and budget?",
            ),
        ],
    ),
    _conversation(
        "n23_progressive_yoga_leggings",
        language="en",
        domain="apparel",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I need comfortable bottoms for exercise classes."),
            _turn(
                "High-waisted women's leggings for yoga, size medium.",
                assistant="What activity and garment are these for?",
                question="Can you choose the type, rise, and size?",
            ),
            _turn(
                "Full length, dark navy, squat-proof, side pockets, no front seam, under $55.",
                assistant="I'll look for high-waisted yoga leggings in medium.",
                question="Any length, color, opacity, pocket, seam, or budget preferences?",
            ),
        ],
    ),
    _conversation(
        "n24_progressive_office_blazer",
        language="en",
        domain="apparel",
        tags=["expected_narrower", "progressive_narrowing"],
        turns=[
            _turn("I could use another layer for the office."),
            _turn(
                "A women's blazer, tailored but not too formal, size 10.",
                assistant="What kind of office layer do you need?",
                question="Could you choose the garment, fit, and size?",
            ),
            _turn(
                "Single-breasted camel blazer with full-length sleeves, no shoulder pads, under $140.",
                assistant="I'll search for tailored women's blazers in size 10.",
                question="What color, construction, sleeves, exclusions, and budget?",
            ),
        ],
    ),
    _conversation(
        "n25_progressive_commute_backpack_zh",
        language="zh",
        domain="luggage",
        tags=["expected_narrower", "progressive_narrowing", "multilingual"],
        turns=[
            _turn("想换一个每天上班背的包，但还没想好具体款式。"),
            _turn(
                "还是双肩包吧，至少要能放下 16 寸电脑。",
                assistant="你更喜欢哪种通勤包？",
                question="包的类型和容量有什么要求？",
            ),
            _turn(
                "要深灰色、防泼水、背部透气，带行李箱固定带，预算 90 美元以内。",
                assistant="我会找能装 16 寸电脑的通勤双肩包。",
                question="颜色、面料、背负功能和预算还有要求吗？",
            ),
        ],
    ),
    _conversation(
        "n26_progressive_summer_dress_mixed",
        language="mixed",
        domain="apparel",
        tags=["expected_narrower", "progressive_narrowing", "code_switch"],
        turns=[
            _turn("想找一条 summer dress，平时周末穿。"),
            _turn(
                "Midi length，女款 M 码，整体轻松一点。",
                assistant="你希望是什么长度和尺码？",
                question="可以先确定长度、尺码和风格吗？",
            ),
            _turn(
                "浅绿色 linen blend，短袖，有口袋，不要露背，under $90。",
                assistant="我会找休闲的女款中长连衣裙。",
                question="颜色、材质、袖子、功能、排除项和预算呢？",
            ),
        ],
    ),
    _conversation(
        "n27_formal_oxfords",
        language="en",
        domain="footwear",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I need shoes for a formal work event."),
            _turn(
                "Men's black leather cap-toe Oxford shoes, size 11, no patent finish, below $160.",
                assistant="What kind of formal shoes should I look for?",
                question="Please specify style, material, color, size, exclusions, and budget.",
            ),
        ],
    ),
    _conversation(
        "n28_swim_cover_up",
        language="en",
        domain="apparel",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I need something easy to throw on at the pool."),
            _turn(
                "A women's white cotton beach cover-up, knee length, loose fit, size large, under $45.",
                assistant="What kind of poolside item do you want?",
                question="Could you narrow down type, material, length, fit, size, and budget?",
            ),
        ],
    ),
    _conversation(
        "n29_minimalist_earrings",
        language="en",
        domain="jewelry",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I'm looking for simple earrings I can wear every day."),
            _turn(
                "Small 14k rose-gold huggie hoops, plain with no stones, under $130.",
                assistant="What does simple everyday jewelry mean to you?",
                question="Can you choose the earring type, metal, decoration, and budget?",
            ),
        ],
    ),
    _conversation(
        "n30_polarized_sunglasses",
        language="en",
        domain="accessories",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I need sunglasses for driving."),
            _turn(
                "Men's matte-black rectangular sunglasses with polarized grey lenses, medium fit, under $100.",
                assistant="What sunglasses work best for you?",
                question="Please specify shape, color, lens, fit, and budget.",
            ),
        ],
    ),
    _conversation(
        "n31_leather_belt",
        language="en",
        domain="accessories",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I'd like a belt that works with jeans."),
            _turn(
                "Men's dark-brown full-grain leather belt, 1.5 inches wide, size 36, brass buckle, under $65.",
                assistant="What kind of jeans belt do you prefer?",
                question="Could you specify material, color, width, size, hardware, and budget?",
            ),
        ],
    ),
    _conversation(
        "n32_compact_wallet_zh",
        language="zh",
        domain="accessories",
        tags=["expected_narrower", "vague_to_specific", "multilingual"],
        turns=[
            _turn("想买个小一点的钱包，平时随身带。"),
            _turn(
                "要女士黑色真皮短款钱包，带拉链零钱袋和至少 6 个卡位，不要大 logo，70 美元以内。",
                assistant="你希望小钱包有哪些特点？",
                question="可以补充款式、材质、颜色、容量、排除项和预算吗？",
            ),
        ],
    ),
    _conversation(
        "n33_running_socks",
        language="en",
        domain="apparel",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I need better socks for running."),
            _turn(
                "Men's white ankle running socks, moisture-wicking, cushioned heel, no wool, pack of at least three, under $30.",
                assistant="What running socks would work for you?",
                question="Please specify department, color, height, features, material exclusions, quantity, and budget.",
            ),
        ],
    ),
    _conversation(
        "n34_cotton_pajamas",
        language="en",
        domain="apparel",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I want something more comfortable to sleep in."),
            _turn(
                "Women's navy cotton pajama set, long sleeves and full-length pants, size medium, no satin, below $65.",
                assistant="What sleepwear would feel comfortable?",
                question="Could you specify type, material, color, coverage, size, exclusions, and budget?",
            ),
        ],
    ),
    _conversation(
        "n35_sun_hat_mixed",
        language="mixed",
        domain="accessories",
        tags=["expected_narrower", "vague_to_specific", "code_switch"],
        turns=[
            _turn("想买个帽子去海边，主要是防晒。"),
            _turn(
                "Women's wide-brim straw hat，米白色，可折叠，有 chin strap，under $40。",
                assistant="你想要哪种海边防晒帽？",
                question="款式、材质、颜色、便携功能和预算有什么要求？",
            ),
        ],
    ),
    _conversation(
        "n36_tennis_bracelet",
        language="en",
        domain="jewelry",
        tags=["expected_narrower", "vague_to_specific"],
        turns=[
            _turn("I'm considering a bracelet as a gift."),
            _turn(
                "A sterling-silver tennis bracelet with clear cubic zirconia, 7 inches, lobster clasp, under $90.",
                assistant="What kind of gift bracelet are you considering?",
                question="Please choose style, material, stones, length, clasp, and budget.",
            ),
        ],
    ),
    _conversation(
        "b04_release_boot_color_budget",
        language="en",
        domain="footwear",
        tags=["expected_broader", "constraint_release"],
        turns=[
            _turn("Women's black waterproof ankle boots, size 8, under $90, with a low heel."),
            _turn(
                "Keep waterproof ankle boots in size 8, but any color, price, or heel height is fine.",
                assistant="I'll keep all those boot requirements.",
            ),
        ],
    ),
    _conversation(
        "b05_release_watch_material",
        language="en",
        domain="watches",
        tags=["expected_broader", "constraint_release"],
        turns=[
            _turn(
                "A men's blue-dial automatic watch on a steel bracelet, 40 mm or smaller, below $300."
            ),
            _turn(
                "Actually only the automatic movement matters. Dial, band, size, and budget can all be open.",
                assistant="I'll search for that exact watch configuration.",
            ),
        ],
    ),
    _conversation(
        "b06_release_bag_brand_color",
        language="en",
        domain="handbags",
        tags=["expected_broader", "constraint_release"],
        turns=[
            _turn("Find a black Coach leather crossbody bag under $180 with gold hardware."),
            _turn(
                "Any small crossbody bag is okay now; brand, color, material, hardware, and price don't matter.",
                assistant="I'll stick with the black Coach bag.",
            ),
        ],
    ),
    _conversation(
        "b07_release_dress_exclusions",
        language="en",
        domain="apparel",
        tags=["expected_broader", "constraint_release"],
        turns=[
            _turn("A navy midi dress with sleeves, no sequins or lace, size 8, under $100."),
            _turn(
                "Keep it as a size 8 midi dress, but release the color, sleeves, fabric exclusions, and budget.",
                assistant="I'll keep all of those dress constraints.",
            ),
        ],
    ),
    _conversation(
        "b08_release_sneaker_features_zh",
        language="zh",
        domain="footwear",
        tags=["expected_broader", "constraint_release", "multilingual"],
        turns=[
            _turn("找男款白色低帮运动鞋，42 码，要真皮、鞋带款，预算 100 美元以内。"),
            _turn(
                "只保留男款 42 码运动鞋就行，颜色、鞋帮、材质、闭合方式和价格都不限。",
                assistant="我会保留刚才所有运动鞋条件。",
            ),
        ],
    ),
    _conversation(
        "b09_release_necklace_stone",
        language="mixed",
        domain="jewelry",
        tags=["expected_broader", "constraint_release", "code_switch"],
        turns=[
            _turn("想找 18-inch sterling silver necklace，要蓝宝石吊坠，under $120。"),
            _turn(
                "长度和宝石都不重要了，any silver necklace under $120 is fine。",
                assistant="我会继续找指定长度的蓝宝石项链。",
            ),
        ],
    ),
    _conversation(
        "b10_release_coat_most_constraints",
        language="en",
        domain="apparel",
        tags=["expected_broader", "constraint_release"],
        turns=[
            _turn("Women's camel wool knee-length coat, belted, size medium, no hood, under $220."),
            _turn(
                "That's over-specified. Any women's coat in medium works; everything else is flexible.",
                assistant="I'll keep looking for that exact camel coat.",
            ),
        ],
    ),
    _conversation(
        "s03_sort_only",
        language="en",
        domain="footwear",
        tags=["expected_stable", "presentation_directive"],
        turns=[
            _turn("Men's brown leather loafers, size 10, under $120."),
            _turn(
                "Sort the results from lowest to highest price, but keep the search requirements unchanged.",
                assistant="I have the loafer requirements.",
            ),
        ],
    ),
    _conversation(
        "s04_explanation_only",
        language="en",
        domain="luggage",
        tags=["expected_stable", "presentation_directive"],
        turns=[
            _turn("A black 20-inch hard-shell carry-on with spinner wheels and a TSA lock."),
            _turn(
                "For each option, explain the warranty. Don't alter the kind of suitcase we're finding.",
                assistant="I'll search for that carry-on.",
            ),
        ],
    ),
    _conversation(
        "s05_repeat_requirements_zh",
        language="zh",
        domain="jewelry",
        tags=["expected_stable", "state_confirmation", "multilingual"],
        turns=[
            _turn("要 925 银的小号圆圈耳环，不带宝石，预算 50 美元。"),
            _turn(
                "对，就按刚才这些条件继续，没有要补充或删除的。",
                assistant="我会按这些耳环条件搜索。",
            ),
        ],
    ),
    _conversation(
        "s06_irrelevant_weather",
        language="en",
        domain="apparel",
        tags=["expected_stable", "irrelevant_turn"],
        turns=[
            _turn("A women's black cotton cardigan, size small, under $60."),
            _turn(
                "It started raining outside, but that has nothing to do with this search. Keep everything the same.",
                assistant="I'll look for the cardigan you described.",
            ),
        ],
    ),
    _conversation(
        "s07_result_count_only_mixed",
        language="mixed",
        domain="handbags",
        tags=["expected_stable", "presentation_directive", "code_switch"],
        turns=[
            _turn("要一个 black leather tote，能装 15-inch laptop，under $150。"),
            _turn(
                "Show me 8 results instead of 5，search conditions 不要变。",
                assistant="我已经记住通勤包条件。",
            ),
        ],
    ),
    _conversation(
        "o04_watch_to_boots",
        language="en",
        domain="cross_domain",
        tags=["expected_override", "goal_change"],
        turns=[
            _turn("I'm looking for a simple men's analog watch under $100."),
            _turn(
                "Forget watches. I need women's waterproof hiking boots, size 7, under $140.",
                assistant="I'll keep searching for watches.",
            ),
        ],
    ),
    _conversation(
        "o05_earrings_to_luggage",
        language="en",
        domain="cross_domain",
        tags=["expected_override", "goal_change"],
        turns=[
            _turn("Find small sterling-silver stud earrings under $50."),
            _turn(
                "New priority: a navy carry-on suitcase with spinner wheels. Drop all jewelry preferences.",
                assistant="I'll show you silver earrings.",
            ),
        ],
    ),
    _conversation(
        "o06_coat_to_handbag_zh",
        language="zh",
        domain="cross_domain",
        tags=["expected_override", "goal_change", "multilingual"],
        turns=[
            _turn("先找女款黑色羊毛大衣，M 码，200 美元以内。"),
            _turn(
                "大衣先不买了，改成棕色皮质斜挎包，小号，预算 120 美元。",
                assistant="我会继续找黑色羊毛大衣。",
            ),
        ],
    ),
    _conversation(
        "o07_shoes_to_bracelet_mixed",
        language="mixed",
        domain="cross_domain",
        tags=["expected_override", "goal_change", "code_switch"],
        turns=[
            _turn("想找 men's white sneakers，size 10，under $90。"),
            _turn(
                "Change of plan，不要鞋了，找一条 sterling silver bracelet 当礼物，under $80。",
                assistant="我会继续找白色男鞋。",
            ),
        ],
    ),
]


def _preference_assertion(
    *,
    kind: str = "preference",
    facet: str | None = None,
    relation: str | None = None,
    values: list[str] | None = None,
    text_contains: str | None = None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "facet": facet,
        "relation": relation,
        "values": values or [],
        "strength": None,
        "text_contains": text_contains,
    }


REGRESSION_ASSERTIONS: dict[tuple[str, int], list[dict[str, object]]] = {
    ("n04_everyday_watch", 2): [
        _preference_assertion(
            relation="semantic_positive",
            text_contains="40 mm",
        ),
        _preference_assertion(facet="price", relation="upper", values=["150"]),
    ],
    ("b01_release_heel_constraints", 2): [
        {"kind": "goal_contains", "text": "heels"},
        {"kind": "goal_not_contains", "text": "red"},
        {"kind": "goal_not_contains", "text": "leather"},
        {"kind": "facet_absent", "facet": "color"},
        {"kind": "facet_absent", "facet": "material"},
        {"kind": "facet_absent", "facet": "price"},
        _preference_assertion(kind="preference_absent", text_contains="closed-toe"),
    ],
    ("b02_release_necklace_constraints", 2): [
        {"kind": "goal_contains", "text": "necklace"},
        {"kind": "goal_not_contains", "text": "star"},
        {"kind": "goal_not_contains", "text": "sterling"},
        {"kind": "facet_absent", "facet": "material"},
        {"kind": "facet_absent", "facet": "price"},
        _preference_assertion(kind="preference_absent", text_contains="18-inch"),
        _preference_assertion(kind="preference_absent", text_contains="stone"),
    ],
    ("b03_release_jacket_constraints", 2): [
        {"kind": "goal_contains_any", "texts": ["coat", "outerwear", "外套"]},
        {"kind": "goal_not_contains", "text": "黄色"},
        {"kind": "facet_absent", "facet": "color"},
        {"kind": "facet_absent", "facet": "material"},
        {"kind": "facet_absent", "facet": "size"},
        {"kind": "facet_absent", "facet": "price"},
        {"kind": "facet_absent", "facet": "feature"},
    ],
    ("b04_release_boot_color_budget", 2): [
        {"kind": "goal_contains", "text": "boot"},
        {"kind": "goal_not_contains", "text": "black"},
        {"kind": "facet_absent", "facet": "color"},
        {"kind": "facet_absent", "facet": "price"},
        _preference_assertion(text_contains="waterproof"),
        _preference_assertion(facet="size", relation="include", values=["8"]),
        _preference_assertion(kind="preference_absent", text_contains="low heel"),
    ],
    ("b05_release_watch_material", 1): [
        _preference_assertion(
            relation="semantic_positive",
            text_contains="40 mm",
        ),
    ],
    ("b05_release_watch_material", 2): [
        {"kind": "goal_contains", "text": "watch"},
        {"kind": "goal_not_contains", "text": "blue"},
        {"kind": "facet_absent", "facet": "color"},
        {"kind": "facet_absent", "facet": "material"},
        {"kind": "facet_absent", "facet": "price"},
        _preference_assertion(text_contains="automatic"),
        _preference_assertion(kind="preference_absent", text_contains="40 mm"),
    ],
    ("b07_release_dress_exclusions", 2): [
        {"kind": "goal_contains", "text": "dress"},
        {"kind": "goal_not_contains", "text": "navy"},
        {"kind": "goal_not_contains", "text": "sleeve"},
        {"kind": "facet_absent", "facet": "color"},
        {"kind": "facet_absent", "facet": "price"},
    ],
    ("b09_release_necklace_stone", 2): [
        {"kind": "goal_contains", "text": "necklace"},
        {"kind": "goal_not_contains", "text": "sapphire"},
        _preference_assertion(facet="material", relation="include", text_contains="silver"),
        _preference_assertion(facet="price", relation="upper", values=["120"]),
        {"kind": "facet_absent", "facet": "size"},
        {"kind": "facet_absent", "facet": "feature"},
    ],
}


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload.get("suite_id") != "intent-space-natural-prompts-v1":
        raise ValueError("source suite identity does not match v1")
    existing = {item["id"] for item in payload["conversations"]}
    added = {item["id"] for item in ADDITIONS}
    if len(added) != len(ADDITIONS) or existing & added:
        raise ValueError("conversation identifiers must be unique")
    payload["suite_id"] = "intent-space-natural-prompts-v2"
    payload["language"] = "mixed English, Simplified Chinese, and code-switching"
    payload["authorship"] = (
        "Expanded hand-authored conversations for intent-space evaluation; "
        "no catalog targets or simulator hidden state."
    )
    payload["oracle_policy"] = (
        "Each conversation carries one expected-transition tag. Narrower and "
        "broader conversations score first-to-last scalar direction, stable "
        "conversations allow ten percent relative movement, and override "
        "conversations are observational. Progressive conversations also retain "
        "their intermediate state for adjacent-turn diagnostics. Known QU failure "
        "cases additionally carry retrieval-changing semantic assertions."
    )
    payload["conversations"].extend(ADDITIONS)
    for conversation in payload["conversations"]:
        identifier = conversation["id"]
        for turn_index, turn in enumerate(conversation["turns"], start=1):
            assertions = REGRESSION_ASSERTIONS.get((identifier, turn_index))
            if assertions is not None:
                turn["critical_assertions"] = assertions
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT}")
    print(f"conversations: {len(payload['conversations'])}")
    print(f"turns: {sum(len(item['turns']) for item in payload['conversations'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
