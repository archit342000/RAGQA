"""Shared helpers for building the in-domain gold set."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import regex as re
import yaml
from rank_bm25 import BM25Okapi

from gold.models import CandidateItem, Evidence, GoldItem, ParsedDoc, ParsedPage

logger = logging.getLogger("gold.utils")

# Regex primitives -----------------------------------------------------------
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][\w\-/ ]{2,80}|[A-Z0-9][A-Z0-9\-/ ]{2,80})[:.]?$"
)
DEFINITION_RE = re.compile(
    r"\b([\w\-/ ]{2,70}?)(?:\s+is|\s+are|\s+refers to|\s+means|\s+defined as)\b",
    re.IGNORECASE,
)
CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s+\d+[\s:\-.]", re.IGNORECASE)
METRIC_RE = re.compile(
    r"\b\d{1,4}(?:\.\d+)?\s?(%|percent|days?|hours?|hrs?|mins?|ms|s|gb|mb|kb|tb|users?|items?|units?)\b",
    re.IGNORECASE,
)
DELIM_RE = re.compile(r"[,;:\t\|]")
ACRONYM_RE = re.compile(r"\b[A-Z]{3,}\b")
PARA_SPLIT_RE = re.compile(r"\n\s*\n")
_TERM_FROM_QUESTION_RE = re.compile(
    r"(?i)(?:what|which|who)\s+(?:is|are|does|do|stands for|refers to)\s+(?:the\s+)?([^?]+)"
)
_WHICH_ENTITY_RE = re.compile(r"(?i)^which\s+([a-z0-9][\w\- ]{2,50})")
_ABOUT_TOPIC_RE = re.compile(r"(?i)about\s+([^?]+)")
_PROCESS_RE = re.compile(r"(?i)how\s+does\s+([^?]+?)\s+(?:handle|process|manage|address)\s+([^?]+)")
_DECISION_RE = re.compile(r"(?i)why\s+is\s+([^?]+?)\s+(?:required|needed|important)")
_EVENT_RE = re.compile(r"(?i)when\s+is\s+([^?]+?)\s+(?:scheduled|expected|performed)")


# Generic IO ----------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def read_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


# Parsed document helpers ----------------------------------------------------

def load_parsed_docs(parsed_dir: str | Path) -> Iterator[ParsedDoc]:
    parsed_dir = Path(parsed_dir)
    for json_path in sorted(parsed_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        pages = [
            ParsedPage(
                doc_id=payload["doc_id"],
                doc_name=payload.get("doc_name", payload["doc_id"]),
                page_num=page["page_num"],
                text=page["text"],
                offset_start=page.get("offset_start"),
                offset_end=page.get("offset_end"),
            )
            for page in payload.get("pages", [])
            if page.get("text")
        ]
        if pages:
            yield ParsedDoc(
                doc_id=payload["doc_id"],
                doc_name=payload.get("doc_name", payload["doc_id"]),
                pages=pages,
                meta=payload.get("meta", {}),
            )


def paragraphs_with_spans(text: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    cursor = 0
    for block in PARA_SPLIT_RE.split(text):
        block = block.strip()
        if not block:
            cursor += 2
            continue
        start = text.find(block, cursor)
        if start == -1:
            start = text.find(block)
        end = start + len(block)
        spans.append((block, start, end))
        cursor = end
    return spans


def extract_sentence(text: str, start: int) -> Tuple[str, int, int]:
    end = len(text)
    for idx in range(start, len(text)):
        if text[idx] in ".!?" and (idx + 1 == len(text) or text[idx + 1].isspace()):
            end = idx + 1
            break
    snippet = text[start:end].strip()
    return snippet, start, end


# Candidate utilities --------------------------------------------------------

def hash_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:8]}"


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def verify_span(text: str, start: int, end: int, answer: str) -> bool:
    return 0 <= start < end <= len(text) and text[start:end] == answer


def tag_question(question: str, answer: str) -> List[str]:
    tags: set[str] = set()
    q_lower = question.lower()
    a_lower = answer.lower()
    if q_lower.startswith("why") or q_lower.startswith("how"):
        tags.add("whyhow")
    if METRIC_RE.search(a_lower):
        tags.add("numeric")
    if DELIM_RE.search(answer):
        tags.add("table")
    if DEFINITION_RE.search(answer):
        tags.add("definition")
    if CAPTION_RE.match(question):
        tags.add("caption")
    if ACRONYM_RE.search(answer):
        tags.add("acronym")
    return sorted(tags)


def candidate_to_dict(item: CandidateItem) -> dict:
    payload = asdict(item)
    return payload


def from_candidate(obj: dict) -> CandidateItem:
    return CandidateItem(**obj)


def gold_to_dict(item: GoldItem) -> dict:
    payload = asdict(item)
    payload["evidence"] = [asdict(ev) for ev in item.evidence]
    return payload


def make_candidate(
    doc: ParsedDoc,
    page: ParsedPage,
    question: str,
    answer: str,
    char_start: int,
    char_end: int,
    tags: List[str],
    source_type: str,
) -> Optional[CandidateItem]:
    answer = answer.strip()
    if not answer:
        return None
    if len(answer) > 300 or len(answer) < 5:
        return None
    if not verify_span(page.text, char_start, char_end, answer):
        return None
    tag_union = sorted(set(tags) | set(tag_question(question, answer)))
    return CandidateItem(
        id=hash_id("cand", doc.doc_id, str(page.page_num), str(char_start), str(char_end)),
        question=question.strip(),
        answer_text=answer,
        doc_id=doc.doc_id,
        doc_name=doc.doc_name,
        page_start=page.page_num,
        page_end=page.page_num,
        char_start=char_start,
        char_end=char_end,
        tags=tag_union,
        source={"type": source_type, "doc_name": doc.doc_name, "page_num": page.page_num},
        meta={"page_text": page.text},
    )


def dedupe_candidates(items: List[CandidateItem], per_page_limit: int) -> List[CandidateItem]:
    seen = set()
    counts: Dict[Tuple[str, int], int] = defaultdict(int)
    deduped: List[CandidateItem] = []
    for item in items:
        key = (item.doc_id, item.page_start, item.char_start, item.char_end, item.question.lower())
        if key in seen:
            continue
        page_key = (item.doc_id, item.page_start)
        if counts[page_key] >= per_page_limit:
            continue
        seen.add(key)
        counts[page_key] += 1
        deduped.append(item)
    return deduped


# Slot extraction ------------------------------------------------------------

def _nearest_heading(page_text: str, char_start: int) -> str:
    if char_start is None or char_start < 0:
        prefix = page_text
    else:
        prefix = page_text[:char_start]
    for line in reversed(prefix.splitlines()):
        candidate = line.strip()
        if HEADING_RE.match(candidate):
            return candidate.rstrip(" :.")
    return ""


def _clean_fragment(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_term(question: str, answer: str) -> str:
    defn = DEFINITION_RE.search(answer)
    if defn:
        return _clean_fragment(defn.group(1))
    match = _TERM_FROM_QUESTION_RE.search(question)
    if match:
        fragment = re.split(r"\b(?:in|for|of|within|on)\b", match.group(1))[0]
        return _clean_fragment(fragment)
    return ""


def _extract_entity_type(question: str) -> str:
    which = _WHICH_ENTITY_RE.search(question)
    if which:
        return _clean_fragment(which.group(1))
    return "entry"


def _extract_topic(question: str, answer: str, heading: str) -> str:
    if heading:
        return heading
    about = _ABOUT_TOPIC_RE.search(question)
    if about:
        return _clean_fragment(about.group(1))
    answer_sentence = answer.split(".")[0]
    if answer_sentence:
        return _clean_fragment(answer_sentence)
    remaining = re.sub(r"^(?:what|which|how|why|where|when|who|does|can)\s+", "", question, flags=re.IGNORECASE)
    return _clean_fragment(remaining)


def _extract_metric(answer: str, question: str) -> Tuple[str, str]:
    match = METRIC_RE.search(answer) or METRIC_RE.search(question)
    if match:
        metric = match.group(0)
        unit = match.group(1) if match.lastindex else "units"
        return metric, unit
    return "", ""


def _extract_acronym(answer: str, question: str) -> str:
    match = ACRONYM_RE.search(answer) or ACRONYM_RE.search(question)
    return match.group(0) if match else ""


def _extract_process_condition(question: str) -> Tuple[str, str]:
    match = _PROCESS_RE.search(question)
    if match:
        return _clean_fragment(match.group(1)), _clean_fragment(match.group(2))
    return "", ""


def _extract_decision(question: str) -> str:
    match = _DECISION_RE.search(question)
    if match:
        return _clean_fragment(match.group(1))
    return ""


def _extract_event(question: str) -> str:
    match = _EVENT_RE.search(question)
    if match:
        return _clean_fragment(match.group(1))
    return ""


def extract_slots_from_page(page_text: str, candidate: dict) -> Dict[str, str]:
    question = candidate.get("question", "")
    answer = candidate.get("answer_text", "")
    char_start = candidate.get("char_start", 0) or 0
    doc_name = candidate.get("doc_name", "the document")
    heading = candidate.get("source", {}).get("heading") or _nearest_heading(page_text, char_start)
    section = heading or candidate.get("source", {}).get("section", "the section")
    scope = candidate.get("source", {}).get("scope") or section or doc_name

    term = _extract_term(question, answer)
    topic = _extract_topic(question, answer, heading)
    acronym = _extract_acronym(answer, question)
    metric, unit_item = _extract_metric(answer, question)
    entity_type = _extract_entity_type(question)
    process, condition = _extract_process_condition(question)
    decision = _extract_decision(question)
    event = _extract_event(question)

    constraint = candidate.get("source", {}).get("constraint", "")
    if not constraint and "for" in question.lower():
        tail = question.split(" for ", 1)[-1]
        constraint = _clean_fragment(tail.replace("?", ""))

    baseline = candidate.get("source", {}).get("baseline", "baseline")
    responsibility = candidate.get("source", {}).get("responsibility", "responsibility")
    subject = term or topic or doc_name
    action = candidate.get("source", {}).get("action", "comply")

    slots: Dict[str, str] = {
        "doc_name": doc_name,
        "doc_id": candidate.get("doc_id", "doc"),
        "heading": heading or section,
        "section": section,
        "scope": scope,
        "term": term or topic,
        "topic": topic,
        "acronym": acronym,
        "metric": metric or candidate.get("source", {}).get("metric", ""),
        "unit_item": unit_item or "items",
        "entity_type": entity_type,
        "constraint": constraint,
        "baseline": baseline,
        "decision": decision or term or topic,
        "process": process or term or topic,
        "condition": condition or constraint or scope,
        "event": event or topic,
        "responsibility": responsibility,
        "subject": subject,
        "action": action,
    }
    # Remove empty values
    return {key: value for key, value in slots.items() if value and value.strip()}


# Hard-negative mining -------------------------------------------------------

def build_bm25_index(chunks: List[dict]) -> Tuple[BM25Okapi, List[dict]]:
    corpus = [simple_tokenize(chunk.get("text", "")) for chunk in chunks]
    return BM25Okapi(corpus), chunks


def bm25_hard_negatives(
    question: str,
    answer: str,
    index: Optional[Tuple[BM25Okapi, List[dict]]],
    doc_id: str,
    page_start: int,
    limit: int,
) -> List[str]:
    if not index:
        return []
    bm25, chunks = index
    query_tokens = simple_tokenize(question + " " + answer)
    scores = bm25.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1]
    negatives: List[str] = []
    for idx in ranked:
        chunk = chunks[idx]
        if chunk.get("doc_id") == doc_id and chunk.get("page_start") == page_start:
            continue
        chunk_id = chunk.get("id")
        if chunk_id and chunk_id not in negatives:
            negatives.append(chunk_id)
        if len(negatives) >= limit:
            break
    return negatives


# Gold conversion ------------------------------------------------------------

def to_gold_item(candidate: dict, hard_negatives: List[str], idx: int) -> GoldItem:
    evidence = [
        Evidence(
            doc_id=candidate["doc_id"],
            page=candidate["page_start"],
            char_start=candidate["char_start"],
            char_end=candidate["char_end"],
        )
    ]
    return GoldItem(
        id=f"g-{idx:04d}",
        question=candidate["question"],
        answer_text=candidate["answer_text"],
        doc_id=candidate["doc_id"],
        page_start=candidate["page_start"],
        page_end=candidate["page_end"],
        char_start=candidate["char_start"],
        char_end=candidate["char_end"],
        tags=list(candidate.get("tags", [])),
        hard_negative_ids=hard_negatives,
        evidence=evidence,
    )


def summary_stats(items: List[CandidateItem]) -> dict:
    by_tag = Counter(tag for item in items for tag in item.tags)
    by_doc = Counter(item.doc_id for item in items)
    lengths = [len(item.answer_text) for item in items]
    try:
        from gold.quality import detect_wh as _detect_wh
    except Exception:  # pragma: no cover - defensive
        _detect_wh = lambda _: "what"  # type: ignore
    wh_counts = Counter(item.meta.get("wh_type") or _detect_wh(item.question) for item in items)
    return {
        "total": len(items),
        "by_tag": dict(by_tag),
        "by_doc": dict(by_doc),
        "wh": dict(wh_counts),
        "mean_answer_chars": float(np.mean(lengths)) if lengths else 0.0,
    }


def stratified_sample(items: List[dict], quotas: Dict[str, float], total: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = {tag: [] for tag in quotas}
    leftover: List[dict] = []
    for item in items:
        assigned = False
        for tag in item.get("tags", []):
            if tag in buckets:
                buckets[tag].append(item)
                assigned = True
                break
        if not assigned:
            leftover.append(item)
    sample: List[dict] = []
    for tag, proportion in quotas.items():
        bucket = buckets[tag]
        rng.shuffle(bucket)
        take = min(len(bucket), max(0, int(math.ceil(proportion * total))))
        sample.extend(bucket[:take])
    if len(sample) < total:
        pool = leftover + [item for bucket in buckets.values() for item in bucket]
        rng.shuffle(pool)
        for item in pool:
            if len(sample) >= total:
                break
            sample.append(item)
    return sample[:total]


__all__ = [
    "ACRONYM_RE",
    "CAPTION_RE",
    "CandidateItem",
    "DEFINITION_RE",
    "HEADING_RE",
    "METRIC_RE",
    "bm25_hard_negatives",
    "build_bm25_index",
    "candidate_to_dict",
    "dedupe_candidates",
    "extract_sentence",
    "extract_slots_from_page",
    "from_candidate",
    "gold_to_dict",
    "hash_id",
    "load_parsed_docs",
    "load_yaml",
    "make_candidate",
    "paragraphs_with_spans",
    "read_jsonl",
    "simple_tokenize",
    "stratified_sample",
    "summary_stats",
    "tag_question",
    "to_gold_item",
    "verify_span",
    "write_jsonl",
]
