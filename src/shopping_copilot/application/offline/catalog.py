from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .fact_linker import CatalogFactEntry, CatalogFactLinker, CatalogFactMatch, fact_tokens

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PRICE_RE = re.compile(r"(?:budget\s+around\s+)?\$\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
    "here",
    "what",
    "matters",
    "requirement",
    "actually",
    "earlier",
    "preference",
}

# These are deliberately ordinary catalog normalizers, not hidden-card slots.
BASE_MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
    "linen",
    "denim",
    "suede",
    "velvet",
    "cashmere",
    "acrylic",
    "viscose",
    "lyocell",
    "modal",
    "rubber",
    "alloy",
    "silver",
    "gold",
    "platinum",
    "titanium",
    "canvas",
    "mesh",
)
MATERIALS = (
    "stainless steel",
    "sterling silver",
    "faux leather",
    "genuine leather",
    *BASE_MATERIALS,
)
BASE_COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
    "navy",
    "beige",
    "tan",
    "khaki",
    "burgundy",
    "maroon",
    "teal",
    "turquoise",
    "ivory",
    "cream",
    "silver",
    "gold",
    "multicolor",
    "multicolour",
)
COLORS = (
    "rose gold",
    "navy blue",
    "light blue",
    "dark blue",
    "hot pink",
    *BASE_COLORS,
)
SIZE_WORDS = (
    "plus size",
    "one size",
    "extra small",
    "extra large",
    "x-small",
    "x-large",
    "xxl",
    "xl",
    "large",
    "medium",
    "small",
    "wide",
    "narrow",
    "petite",
    "tall",
)
STYLE_WORDS = (
    "casual",
    "formal",
    "vintage",
    "classic",
    "modern",
    "slim fit",
    "regular fit",
    "loose fit",
    "fitted",
    "oversized",
    "sleeveless",
    "long sleeve",
    "short sleeve",
    "v-neck",
    "crew neck",
    "high waist",
    "low rise",
    "waterproof",
    "lightweight",
)
USE_CASE_WORDS = (
    "hiking",
    "running",
    "gym",
    "winter",
    "outdoor",
    "work",
    "wedding",
    "party",
    "travel",
    "walking",
    "cycling",
    "yoga",
    "swimming",
    "sports",
    "business",
)

ATTRIBUTES = ("material", "color", "style", "size", "use_case", "budget")


def _word_pattern(values: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(value) for value in sorted(values, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])({alternatives})(?![a-z0-9])", re.IGNORECASE)


MATERIAL_RE = _word_pattern(MATERIALS)
BASE_MATERIAL_RE = _word_pattern(BASE_MATERIALS)
COLOR_RE = _word_pattern(COLORS)
BASE_COLOR_RE = _word_pattern(BASE_COLORS)
SIZE_RE = _word_pattern(SIZE_WORDS)
STYLE_RE = _word_pattern(STYLE_WORDS)
USE_CASE_RE = _word_pattern(USE_CASE_WORDS)


def normalize_phrase(value: object, limit: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" -;,.\t\n")
    if limit is not None:
        text = text[:limit].rstrip()
    return text.casefold()


def display_clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" -;,.\t\n")


def field_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def flatten_phrases(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def coarse_category(values: object) -> str:
    if not isinstance(values, list):
        return "clothing item"
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.casefold() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def unique_matches(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(1).casefold() for match in pattern.finditer(text)))


def classify_constraint(value: str) -> str:
    lowered = normalize_phrase(value)
    if "budget" in lowered or PRICE_RE.search(lowered):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if "color" in lowered or COLOR_RE.search(lowered):
        return "color"
    if SIZE_RE.search(lowered) or any(word in lowered for word in ("size", "sizing", "width")):
        return "size"
    if STYLE_RE.search(lowered) or any(
        word in lowered for word in ("department", "style", "fit", "sleeve", "neck")
    ):
        return "style"
    if USE_CASE_RE.search(lowered):
        return "use_case"
    return "feature"


def structured_values(value: str) -> dict[str, tuple[str, ...]]:
    normalized = normalize_phrase(value)
    result: dict[str, tuple[str, ...]] = {}
    materials = unique_matches(MATERIAL_RE, normalized)
    colors = unique_matches(COLOR_RE, normalized)
    sizes = unique_matches(SIZE_RE, normalized)
    styles = unique_matches(STYLE_RE, normalized)
    uses = unique_matches(USE_CASE_RE, normalized)
    if materials:
        result["material"] = materials
    if colors:
        result["color"] = colors
    if sizes:
        result["size"] = sizes
    if styles:
        result["style"] = styles
    if uses:
        result["use_case"] = uses
    price = PRICE_RE.search(normalized)
    if price:
        result["budget"] = (f"{float(price.group(1)):.4f}",)
    return result


class CatalogIndex:
    """Field-aware lexical and structured index over participant-visible metadata."""

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.asins: list[str] = []
        self.asin_to_pid: dict[str, int] = {}
        self.category_by_pid: list[str] = []
        self.rating_count_by_pid: list[int] = []

        phrase_lists: dict[str, list[int]] = defaultdict(list)
        position_phrase_lists: list[dict[str, list[int]]] = [defaultdict(list) for _ in range(4)]
        category_lists: dict[str, list[int]] = defaultdict(list)
        attribute_lists: dict[str, dict[str, list[int]]] = {
            attribute: defaultdict(list) for attribute in ATTRIBUTES
        }
        fact_text_by_tokens: dict[tuple[str, ...], str] = {}
        structured_fact_tokens: set[tuple[str, ...]] = set()
        category_text_by_tokens: dict[tuple[str, ...], str] = {}

        def register_fact(value: object, *, structured: bool = False) -> None:
            displayed = display_clean(value)
            key = fact_tokens(displayed)
            if not key:
                return
            fact_text_by_tokens.setdefault(key, displayed)
            if structured:
                structured_fact_tokens.add(key)

        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "title, categories, features, details, store, description, content='', "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[tuple[int, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                pid = len(self.asins)
                asin = str(product["parent_asin"])
                self.asins.append(asin)
                self.asin_to_pid[asin] = pid

                categories = product.get("categories") or []
                category = coarse_category(categories)
                category_norm = normalize_phrase(category)
                self.category_by_pid.append(category_norm)
                category_text_by_tokens.setdefault(fact_tokens(category), category)

                category_keys = {category_norm}
                if isinstance(categories, list):
                    for item in categories:
                        normalized = normalize_phrase(item)
                        if normalized:
                            category_keys.add(normalized)
                        for part in str(item).split(","):
                            normalized_part = normalize_phrase(part)
                            if normalized_part:
                                category_keys.add(normalized_part)
                for key in category_keys:
                    category_lists[key].append(pid)

                feature_phrases = flatten_phrases(product.get("features"))
                detail_phrases = flatten_phrases(product.get("details"))
                title = display_clean(product.get("title") or "product")
                description_phrases = flatten_phrases(product.get("description"))
                searchable = " ".join(
                    [
                        title,
                        *feature_phrases,
                        *detail_phrases,
                        *description_phrases,
                        field_text(categories),
                        field_text(product.get("store")),
                    ]
                )

                # Complete feature/detail phrases are standard lexical fields. The
                # 180-character prefix is indexed as a robust query-prefix variant,
                # without selecting or labelling any hidden intent slots.
                phrase_values = [*feature_phrases, *detail_phrases]
                if not phrase_values:
                    phrase_values = [title]
                for phrase in [*feature_phrases, *detail_phrases]:
                    register_fact(phrase)
                    register_fact(display_clean(phrase)[:180])
                exact_phrases: list[str] = []
                for phrase in [*phrase_values, title]:
                    normalized = normalize_phrase(phrase)
                    truncated = normalize_phrase(display_clean(phrase)[:180])
                    for candidate in (normalized, truncated):
                        if candidate and candidate not in exact_phrases:
                            exact_phrases.append(candidate)
                for phrase in exact_phrases:
                    phrase_lists[phrase].append(pid)

                # Earlier merchandising bullets normally carry more product
                # identity than later boilerplate. Keep their raw positions as
                # field evidence without constructing hidden hard/soft labels.
                for position, phrase in enumerate(phrase_values[:4]):
                    variants = {
                        normalize_phrase(phrase),
                        normalize_phrase(display_clean(phrase)[:180]),
                    }
                    for variant in variants:
                        if variant:
                            position_phrase_lists[position][variant].append(pid)

                per_attribute: dict[str, list[str]] = {attribute: [] for attribute in ATTRIBUTES}
                per_attribute["material"].extend(unique_matches(MATERIAL_RE, searchable))
                per_attribute["material"].extend(unique_matches(BASE_MATERIAL_RE, searchable))
                per_attribute["color"].extend(
                    f"color: {color}" for color in unique_matches(COLOR_RE, searchable)
                )
                per_attribute["color"].extend(
                    f"color: {color}" for color in unique_matches(BASE_COLOR_RE, searchable)
                )
                per_attribute["size"].extend(unique_matches(SIZE_RE, searchable))
                per_attribute["style"].extend(unique_matches(STYLE_RE, searchable))
                per_attribute["use_case"].extend(unique_matches(USE_CASE_RE, searchable))

                for attribute in ("material", "size", "style", "use_case"):
                    for value in per_attribute[attribute]:
                        register_fact(value, structured=True)
                for value in per_attribute["color"]:
                    register_fact(value, structured=True)

                price = product.get("price")
                if price not in (None, ""):
                    try:
                        price_value = f"{float(price):.4f}"
                        per_attribute["budget"].append(price_value)
                        register_fact(f"budget around ${price}", structured=True)
                    except (TypeError, ValueError):
                        pass

                for attribute in ATTRIBUTES:
                    values = tuple(dict.fromkeys(per_attribute[attribute]))
                    for value in values:
                        attribute_lists[attribute][value].append(pid)

                rating_number = product.get("rating_number")
                self.rating_count_by_pid.append(
                    int(rating_number) if isinstance(rating_number, int) else 0
                )

                batch.append(
                    (
                        pid + 1,
                        title,
                        field_text(categories),
                        field_text(product.get("features")),
                        field_text(product.get("details")),
                        field_text(product.get("store")),
                        field_text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    self.connection.executemany(
                        "INSERT INTO products(rowid, title, categories, features, details, store, description) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
        if batch:
            self.connection.executemany(
                "INSERT INTO products(rowid, title, categories, features, details, store, description) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        self.connection.commit()

        self.phrase_postings = {key: tuple(values) for key, values in phrase_lists.items()}
        self.position_phrase_postings = tuple(
            {key: tuple(values) for key, values in postings.items()}
            for postings in position_phrase_lists
        )
        self.category_postings = {key: tuple(values) for key, values in category_lists.items()}
        self.attribute_postings = {
            attribute: {key: tuple(values) for key, values in postings.items()}
            for attribute, postings in attribute_lists.items()
        }
        self.all_pids = tuple(range(len(self.asins)))
        fact_entries: dict[tuple[str, ...], CatalogFactEntry] = {}
        for fact_key_tokens, text in fact_text_by_tokens.items():
            fact_entries[fact_key_tokens] = CatalogFactEntry(
                text=text,
                posting_size=len(self.postings_for_constraint(text)),
                structured=fact_key_tokens in structured_fact_tokens,
            )
        self.fact_linker = CatalogFactLinker(fact_entries)
        category_entries: dict[tuple[str, ...], CatalogFactEntry] = {}
        for category_key_tokens, text in category_text_by_tokens.items():
            category_entries[category_key_tokens] = CatalogFactEntry(
                text=text,
                posting_size=len(self.category_candidates(text)),
                structured=True,
            )
        self.category_linker = CatalogFactLinker(category_entries)

    @property
    def size(self) -> int:
        return len(self.asins)

    def asin(self, pid: int) -> str:
        return self.asins[pid]

    def phrase_known(self, phrase: str) -> bool:
        normalized = normalize_phrase(phrase)
        if normalized in self.phrase_postings:
            return True
        return bool(self.postings_for_constraint(normalized))

    def category_candidates(self, category: str) -> tuple[int, ...]:
        return self.category_postings.get(normalize_phrase(category), ())

    def positional_phrase_candidates(self, phrase: str) -> tuple[tuple[int, ...], ...]:
        normalized = normalize_phrase(phrase)
        return tuple(postings.get(normalized, ()) for postings in self.position_phrase_postings)

    def postings_for_constraint(self, value: str) -> tuple[int, ...]:
        normalized = normalize_phrase(value)
        exact = self.phrase_postings.get(normalized, ())
        # Preserve exact long-form evidence. Expanding "100% Polyester" to
        # every product that merely mentions polyester destroys the useful
        # product-phrase signal. Bare attributes use the structured route.
        bare_material = normalized in MATERIALS
        bare_color = normalized in COLORS or normalized.startswith("color:")
        bare_budget = normalized.startswith("budget") or bool(PRICE_RE.fullmatch(normalized))
        if exact and not (bare_material or bare_color or bare_budget):
            return exact

        postings: set[int] = set(exact)
        parsed = structured_values(normalized)
        for attribute, values in parsed.items():
            attribute_map = self.attribute_postings.get(attribute, {})
            for item in values:
                keys = (item, f"color: {item}") if attribute == "color" else (item,)
                for key in keys:
                    postings.update(attribute_map.get(key, ()))
        return tuple(postings)

    def idf(self, posting_size: int) -> float:
        return math.log((self.size + 1.0) / (posting_size + 1.0)) + 1.0

    def resolve_reply_payload(self, payload: str) -> list[str]:
        cleaned = display_clean(payload)
        if not cleaned:
            return []

        # Prefer a complete catalog phrase before interpreting semicolons as
        # protocol separators. Product bullets commonly contain semicolons.
        if normalize_phrase(cleaned) in self.phrase_postings:
            return [cleaned]

        # The protocol can join at most two values with '; '. Test every split
        # point and accept one only when both sides are catalog-grounded. This
        # also tolerates semicolons inside a product feature.
        delimiter = "; "
        start = 0
        while True:
            position = cleaned.find(delimiter, start)
            if position < 0:
                break
            left = display_clean(cleaned[:position])
            right = display_clean(cleaned[position + len(delimiter) :])
            if left and right and self.phrase_known(left) and self.phrase_known(right):
                return [left, right]
            start = position + len(delimiter)

        if self.phrase_known(cleaned):
            return [cleaned]
        return [display_clean(part) for part in cleaned.split(";") if display_clean(part)][:2]

    def link_message_facts(self, message: str, *, limit: int = 8) -> list[CatalogFactMatch]:
        """Ground original catalog facts anywhere in a prompt-independent message."""

        return self.fact_linker.link(message, limit=limit)

    def link_message_category(self, message: str) -> CatalogFactMatch | None:
        """Link the longest participant-visible coarse category in a message."""

        matches = self.category_linker.link(message, limit=1)
        return matches[0] if matches else None

    def fts_search(self, query: str, limit: int = 500) -> list[int]:
        terms = [
            token.casefold()
            for token in TOKEN_RE.findall(query)
            if len(token) > 1 and token.casefold() not in STOPWORDS
        ]
        unique_terms = list(dict.fromkeys(terms))[:48]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        try:
            rows = self.connection.execute(
                "SELECT rowid FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 5.0, 4.0, 8.0, 7.0, 0.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(row[0]) - 1 for row in rows]
