"""Template-driven question generation for gold-set candidates."""

from __future__ import annotations

import random
import re
import hashlib
from typing import Dict, Iterable, List

TemplateBank = Dict[str, List[str]]

_TEMPLATE_BANK: TemplateBank = {
    "definition": [
        "According to {heading}, how is {term} defined?",
        "In {doc_name}, what does {term} mean?",
        "Why is {term} emphasized in {section}?",
        "Who applies {term} within {section}?",
    ],
    "numeric": [
        "How many {unit_item} are specified for {scope}?",
        "By how much does {metric} change in {scope}?",
        "When is {metric} measured for {scope}?",
        "Does {scope} meet the target for {metric}?",
    ],
    "table": [
        "Which {entity_type} has the highest {metric}?",
        "Which {entity_type} meets {constraint}?",
        "Where is the {topic} entry summarized in {section}?",
        "How does {entity_type} compare with {baseline} in {section}?",
    ],
    "acronym": [
        "What does the acronym {acronym} stand for?",
        "Which section introduces {acronym}?",
        "How is {acronym} applied within {section}?",
        "Can {acronym} refer to {term} in {scope}?",
    ],
    "caption": [
        "What does the figure about {topic} show?",
        "Where is the {topic} table summarized?",
        "How does the captioned {topic} relate to {section}?",
        "When was the {topic} visual captured according to {heading}?",
    ],
    "whyhow": [
        "Why is {decision} required in {section}?",
        "How does {process} handle {condition}?",
        "Who authorizes {process} in {section}?",
        "Can {process} continue if {condition} changes?",
    ],
    "generic": [
        "Which part of {doc_name} details {topic}?",
        "Where does {doc_name} describe {topic}?",
        "When does {event} occur in {scope}?",
        "Who is responsible for {responsibility}?",
        "How much {metric} is allocated to {scope}?",
        "How many {unit_item} support {scope}?",
        "Does {subject} comply with {constraint}?",
        "Can {subject} complete {action} in {section}?",
    ],
}

_PLACEHOLDER_RE = re.compile(r"{([^{}]+)}")


def _iter_templates(tag: str) -> Iterable[str]:
    primary = _TEMPLATE_BANK.get(tag, [])
    if tag == "generic":
        return list(primary)
    return list(primary) + _TEMPLATE_BANK["generic"]


def _placeholders(template: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(template)


def _slot_ready(template: str, slots: Dict[str, str]) -> bool:
    return all(slots.get(name, "").strip() for name in _placeholders(template))


def _seed_from_slots(tag: str, slots: Dict[str, str]) -> int:
    ordered = sorted((key, value) for key, value in slots.items())
    payload = "|".join(f"{key}:{value}" for key, value in ordered)
    digest = hashlib.sha1(f"{tag}|{payload}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generate_questions(tag: str, slots: Dict[str, str], max_per_item: int = 3) -> List[str]:
    """Return formatted questions for the provided tag and slots."""

    if max_per_item <= 0:
        return []
    templates = list(_iter_templates(tag))
    rng = random.Random(_seed_from_slots(tag, slots))
    rng.shuffle(templates)
    seen: set[str] = set()
    questions: List[str] = []
    for template in templates:
        if len(questions) >= max_per_item:
            break
        if not _slot_ready(template, slots):
            continue
        try:
            rendered = template.format(**slots).strip()
        except KeyError:
            continue
        if not rendered or rendered.lower() in seen:
            continue
        if not rendered.endswith("?"):
            rendered = f"{rendered}?"
        seen.add(rendered.lower())
        questions.append(rendered)
    return questions


__all__ = ["generate_questions", "_TEMPLATE_BANK"]
