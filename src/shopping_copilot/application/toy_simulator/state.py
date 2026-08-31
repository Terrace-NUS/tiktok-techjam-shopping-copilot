from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import CatalogIndex, classify_constraint, display_clean, normalize_phrase

BUYING_RE = re.compile(
    r"^I'm looking for (.+?)\.\s*A key requirement is:\s*(.+)$",
    re.IGNORECASE,
)
EXPLORING_RE = re.compile(
    r"^I'm looking for (.+?),\s*but I'm still exploring\.?$",
    re.IGNORECASE,
)
INITIAL_OVERRIDE_RE = re.compile(r"^I'm looking for (.+?)\.\s+(.+)$", re.IGNORECASE)
OVERRIDE_RE = re.compile(
    r"^Actually,\s*ignore my earlier preference\.\s*What I need is:\s*(.+)$",
    re.IGNORECASE,
)
MATTERS_RE = re.compile(r"^For that,\s*what matters is:\s*(.+)$", re.IGNORECASE)
NO_PREFERENCE_RE = re.compile(
    r"^I don't have a preference for ([a-z_]+);\s*please use your judgment\.?$",
    re.IGNORECASE,
)
NO_ADDITIONAL_RE = re.compile(
    r"^I don't have an additional preference for ([a-z_]+)\.?$",
    re.IGNORECASE,
)
REPLACEMENT_CUE_RE = re.compile(
    r"\b(?:"
    r"ignore|forget|instead|rather\s+than|changed?\s+my\s+mind|"
    r"switch(?:ing)?\s+to|different\s+direction|no\s+longer|"
    r"now\s+(?:need|want|prefer)|what\s+i\s+need\s+now"
    r")\b",
    re.IGNORECASE,
)
NO_PREFERENCE_CUE_RE = re.compile(
    r"\b(?:no\s+(?:additional\s+)?preference|do\s+not\s+care|don't\s+care|"
    r"either\s+is\s+fine|use\s+your\s+judg(?:e)?ment)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ConstraintObservation:
    text: str
    normalized: str
    attribute: str
    source: str
    introduced_turn: int
    active: bool = True


@dataclass
class SessionState:
    session_id: str
    profile: dict[str, object]
    turn: int = 0
    scenario: str = "unknown"
    category: str | None = None
    constraints: list[ConstraintObservation] = field(default_factory=list)
    initial_preference_norms: set[str] = field(default_factory=set)
    dont_care_attributes: set[str] = field(default_factory=set)
    exhausted_attributes: set[str] = field(default_factory=set)
    asked_counts: dict[str, int] = field(default_factory=dict)
    answer_counts: dict[str, int] = field(default_factory=dict)
    last_ask: str | None = None
    boundary_seen: bool = False
    override_seen: bool = False
    emitted_pids: set[int] = field(default_factory=set)
    messages: list[str] = field(default_factory=list)
    last_event: str = "reset"

    @property
    def active_constraints(self) -> list[ConstraintObservation]:
        return [constraint for constraint in self.constraints if constraint.active]

    @property
    def active_norms(self) -> set[str]:
        return {constraint.normalized for constraint in self.constraints if constraint.active}

    @property
    def query_text(self) -> str:
        values = [constraint.text for constraint in self.active_constraints]
        if self.category:
            values.insert(0, self.category)
        return " ".join(values)

    @property
    def override_pending(self) -> bool:
        return self.scenario == "intent_override" and not self.override_seen

    def record_action(self, ask_attribute: str | None, recommended_pids: list[int]) -> None:
        self.last_ask = ask_attribute
        if ask_attribute:
            self.asked_counts[ask_attribute] = self.asked_counts.get(ask_attribute, 0) + 1
        self.emitted_pids.update(recommended_pids)

    def _add_constraint(
        self,
        text: str,
        source: str,
        *,
        initial_preference: bool = False,
        replace_same_attribute: bool = False,
    ) -> None:
        cleaned = display_clean(text)
        normalized = normalize_phrase(cleaned)
        if not normalized:
            return
        attribute = classify_constraint(cleaned)
        if replace_same_attribute:
            for constraint in self.constraints:
                if constraint.active and constraint.attribute == attribute:
                    constraint.active = False
        for constraint in self.constraints:
            if constraint.normalized == normalized:
                constraint.active = True
                if initial_preference:
                    self.initial_preference_norms.add(normalized)
                return
        self.constraints.append(
            ConstraintObservation(
                text=cleaned,
                normalized=normalized,
                attribute=attribute,
                source=source,
                introduced_turn=self.turn,
            )
        )
        if initial_preference:
            self.initial_preference_norms.add(normalized)

    def _supersede_initial_preference(self) -> None:
        for constraint in self.constraints:
            if constraint.normalized in self.initial_preference_norms:
                constraint.active = False

    def observe(self, user_message: str, turn: int, catalog: CatalogIndex) -> None:
        self.turn = turn
        self.messages.append(user_message)
        message = user_message.strip()

        buying = BUYING_RE.match(message)
        if buying:
            self.scenario = "buying"
            self.category = display_clean(buying.group(1))
            self._add_constraint(buying.group(2), "initial_hard")
            self.last_event = "constraint"
            return

        exploring = EXPLORING_RE.match(message)
        if exploring:
            self.scenario = "browsing_or_boundary"
            self.category = display_clean(exploring.group(1))
            self.last_event = "exploring"
            return

        override = OVERRIDE_RE.match(message)
        if override:
            self.scenario = "intent_override"
            self.override_seen = True
            self._supersede_initial_preference()
            self.emitted_pids.clear()
            # `feature` is a broad API class, not a single-valued slot. Preserve
            # independently learned constraints and erase only the explicitly
            # superseded initial preference.
            self._add_constraint(override.group(1), "override")
            self.last_event = "override"
            return

        matters = MATTERS_RE.match(message)
        if matters:
            # The known evaluator template stays a strict fast path. Catalog
            # linking happens only when no official wrapper matched at all.
            values = catalog.resolve_reply_payload(matters.group(1))
            if self.last_ask:
                self.answer_counts[self.last_ask] = self.answer_counts.get(self.last_ask, 0) + 1
            for value in values:
                self._add_constraint(value, f"answer:{self.last_ask or 'unknown'}")
            self.last_event = "constraint" if values else "unparsed_answer"
            return

        no_preference = NO_PREFERENCE_RE.match(message)
        if no_preference:
            attribute = no_preference.group(1).casefold()
            self.boundary_seen = True
            self.scenario = "boundary"
            self.dont_care_attributes.add(attribute)
            self.last_event = "boundary_no_preference"
            return

        no_additional = NO_ADDITIONAL_RE.match(message)
        if no_additional:
            attribute = no_additional.group(1).casefold()
            self.exhausted_attributes.add(attribute)
            self.last_event = "no_additional"
            return

        if turn == 1:
            initial_override = INITIAL_OVERRIDE_RE.match(message)
            if initial_override:
                self.scenario = "intent_override"
                self.category = display_clean(initial_override.group(1))
                self._add_constraint(
                    initial_override.group(2),
                    "initial_preference",
                    initial_preference=True,
                )
                self.last_event = "constraint"
                return

        linked_category = None
        if turn == 1:
            linked_category = catalog.link_message_category(message)
            if linked_category is not None:
                self.category = linked_category.text
        linked = catalog.link_message_facts(message, limit=4)
        if linked_category is not None:
            category_span = set(range(linked_category.token_start, linked_category.token_end))
            linked = [
                match
                for match in linked
                if not category_span.intersection(range(match.token_start, match.token_end))
            ]
        if turn >= 2 and REPLACEMENT_CUE_RE.search(message):
            self.scenario = "intent_override"
            self.override_seen = True
            self._supersede_initial_preference()
            self.emitted_pids.clear()
            for match in linked:
                self._add_constraint(match.text, "override:catalog_fact")
            self.last_event = "override"
            return

        if self.last_ask and NO_PREFERENCE_CUE_RE.search(message):
            self.boundary_seen = True
            if "additional" in message.casefold():
                self.exhausted_attributes.add(self.last_ask)
                self.last_event = "no_additional"
            else:
                self.scenario = "boundary"
                self.dont_care_attributes.add(self.last_ask)
                self.last_event = "boundary_no_preference"
            return

        if linked:
            if self.last_ask:
                self.answer_counts[self.last_ask] = self.answer_counts.get(self.last_ask, 0) + 1
            for match in linked:
                self._add_constraint(
                    match.text,
                    "initial_catalog_fact"
                    if turn == 1
                    else f"catalog_fact:{self.last_ask or 'unknown'}",
                    initial_preference=turn == 1,
                )
            self.last_event = "constraint"
            return

        self.last_event = "unparsed"
