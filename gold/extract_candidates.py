"""Harvest question/answer candidates from parsed documents with quality checks."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import regex as re

from gold.models import CandidateItem, ParsedDoc, ParsedPage
from gold.question_bank import generate_questions
from gold.quality import (
    answerability_check,
    detect_wh,
    enforce_wh_distribution,
    has_banned_opening,
    is_entity_anchored,
    mmr_select,
    no_vague_pronouns,
    readability_bounds,
)
from gold.utils import (
    CAPTION_RE,
    DEFINITION_RE,
    METRIC_RE,
    candidate_to_dict,
    dedupe_candidates,
    extract_sentence,
    extract_slots_from_page,
    hash_id,
    load_parsed_docs,
    load_yaml,
    make_candidate,
    paragraphs_with_spans,
    summary_stats,
    tag_question,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gold.extract")


def _heading_candidates(doc: ParsedDoc, page: ParsedPage, max_per_page: int) -> List[CandidateItem]:
    candidates: List[CandidateItem] = []
    paragraph_spans = paragraphs_with_spans(page.text)
    lines = [line for line in page.text.splitlines() if line.strip()]
    cursor = 0
    for line in lines:
        stripped = line.strip()
        pos = page.text.find(stripped, cursor)
        if pos == -1:
            pos = page.text.find(stripped)
        cursor = pos + len(stripped)
        from gold.utils import HEADING_RE

        if HEADING_RE.match(stripped):
            for paragraph, start, end in paragraph_spans:
                if start > pos:
                    question = f"According to {stripped}, what is described?"
                    candidate = make_candidate(
                        doc,
                        page,
                        question,
                        paragraph,
                        start,
                        end,
                        tags=tag_question(question, paragraph) + ["heading_para"],
                        source_type="heading_para",
                    )
                    if candidate:
                        candidate.source["heading"] = stripped
                        candidates.append(candidate)
                    break
        if len(candidates) >= max_per_page:
            break
    return candidates


def _definition_candidates(doc: ParsedDoc, page: ParsedPage) -> List[CandidateItem]:
    candidates: List[CandidateItem] = []
    for match in DEFINITION_RE.finditer(page.text):
        term = match.group(1).strip()
        snippet, start, end = extract_sentence(page.text, match.start())
        question = f"What is {term}?"
        candidate = make_candidate(
            doc,
            page,
            question,
            snippet,
            start,
            end,
            tags=tag_question(question, snippet) + ["definition"],
            source_type="definition",
        )
        if candidate:
            candidate.source["heading"] = candidate.source.get("heading", term)
            candidates.append(candidate)
    return candidates


def _table_candidates(doc: ParsedDoc, page: ParsedPage) -> List[CandidateItem]:
    candidates: List[CandidateItem] = []
    for line in page.text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.count(" ") < 1:
            continue
        if ":" in stripped or "|" in stripped or ";" in stripped:
            parts = [part.strip() for part in re.split(r"[:;\|]", stripped) if part.strip()]
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                start = page.text.find(value)
                if start != -1:
                    end = start + len(value)
                    question = f"What is {key}?"
                    candidate = make_candidate(
                        doc,
                        page,
                        question,
                        value,
                        start,
                        end,
                        tags=tag_question(question, value) + ["table"],
                        source_type="table",
                    )
                    if candidate:
                        candidate.source["heading"] = key
                        candidate.source["constraint"] = value
                        candidates.append(candidate)
    return candidates


def _caption_candidates(doc: ParsedDoc, page: ParsedPage) -> List[CandidateItem]:
    candidates: List[CandidateItem] = []
    cursor = 0
    for line in page.text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pos = page.text.find(stripped, cursor)
        if pos == -1:
            pos = page.text.find(stripped)
        cursor = pos + len(stripped)
        if CAPTION_RE.match(stripped):
            noun = stripped.split(":")[0]
            question = f"What does {noun} describe?"
            candidate = make_candidate(
                doc,
                page,
                question,
                stripped,
                pos,
                pos + len(stripped),
                tags=tag_question(question, stripped) + ["caption"],
                source_type="caption",
            )
            if candidate:
                candidate.source["heading"] = noun
                candidates.append(candidate)
    return candidates


def _metric_candidates(doc: ParsedDoc, page: ParsedPage) -> List[CandidateItem]:
    candidates: List[CandidateItem] = []
    for match in METRIC_RE.finditer(page.text):
        snippet = match.group(0)
        start = match.start()
        end = match.end()
        question = f"What is the {match.group(1)}?"
        candidate = make_candidate(
            doc,
            page,
            question,
            snippet,
            start,
            end,
            tags=tag_question(question, snippet) + ["numeric"],
            source_type="numeric",
        )
        if candidate:
            candidate.source["heading"] = candidate.source.get("heading", "metric")
            candidate.source["metric"] = snippet
            candidates.append(candidate)
    return candidates


def harvest_doc(doc: ParsedDoc, limits: dict) -> List[CandidateItem]:
    per_page = limits.get("max_candidates_per_page", 10)
    collected: List[CandidateItem] = []
    for page in doc.pages:
        page_candidates: List[CandidateItem] = []
        page_candidates.extend(_heading_candidates(doc, page, per_page))
        page_candidates.extend(_definition_candidates(doc, page))
        page_candidates.extend(_table_candidates(doc, page))
        page_candidates.extend(_caption_candidates(doc, page))
        page_candidates.extend(_metric_candidates(doc, page))
        collected.extend(dedupe_candidates(page_candidates, per_page))
    return collected


def _normalise_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _candidate_tags(base: CandidateItem) -> List[str]:
    tags = list(base.tags)
    source_type = base.source.get("type")
    if source_type:
        tags.append(source_type)
    tags.append("generic")
    # Preserve order while removing duplicates
    seen: set[str] = set()
    ordered: List[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def _apply_quality_filters(
    question: str,
    slots: Dict[str, str],
    page_text: str,
    answerspan: Tuple[int, int],
    banned: Sequence[str],
    reject_counts: Counter,
) -> bool:
    if has_banned_opening(question, banned):
        reject_counts["banned_opening"] += 1
        return False
    if not readability_bounds(question):
        reject_counts["readability"] += 1
        return False
    if not no_vague_pronouns(question):
        reject_counts["vague_pronoun"] += 1
        return False
    if not is_entity_anchored(question, slots):
        reject_counts["entity_anchor"] += 1
        return False
    if not answerability_check(page_text, answerspan, question, slots):
        reject_counts["not_answerable"] += 1
        return False
    return True


def _expand_candidate(
    base: CandidateItem,
    config: dict,
    banned: Sequence[str],
    reject_counts: Counter,
    wh_counts: Counter,
    tag_counts: Counter,
) -> List[CandidateItem]:
    slots = extract_slots_from_page(base.meta.get("page_text", ""), candidate_to_dict(base))
    page_text = base.meta.get("page_text", "")
    answerspan = (base.char_start, base.char_end)
    question_pool: Dict[str, Tuple[str, str]] = {}
    max_per_span = config.get("limits", {}).get("max_questions_per_span", 3)
    over_generate = max(max_per_span * 3, 6)
    for tag in _candidate_tags(base):
        for generated in generate_questions(tag, slots, max_per_item=over_generate):
            norm = _normalise_question(generated)
            question_pool.setdefault(norm, (generated, tag))
    # include original as last resort
    base_norm = _normalise_question(base.question)
    question_pool.setdefault(base_norm, (base.question, base.tags[0] if base.tags else base.source.get("type", "generic")))

    valid_entries: Dict[str, Tuple[str, str, str]] = {}
    for norm, (question, tag) in question_pool.items():
        if not _apply_quality_filters(question, slots, page_text, answerspan, banned, reject_counts):
            continue
        wh_type = detect_wh(question)
        valid_entries[question] = (tag, wh_type, norm)

    if not valid_entries:
        return []

    wh_targets = config.get("wh_targets") or {}
    filtered_questions: List[str] = list(valid_entries.keys())
    if wh_targets:
        filtered_questions = enforce_wh_distribution(filtered_questions, wh_targets, seed=config.get("seed", 42))
        if not filtered_questions:
            filtered_questions = list(valid_entries.keys())

    selected_questions = mmr_select(filtered_questions, page_text, k=max_per_span, lambda_=0.7)
    if not selected_questions:
        selected_questions = filtered_questions[:max_per_span]

    expanded: List[CandidateItem] = []
    for question in selected_questions:
        tag, wh_type, norm = valid_entries[question]
        wh_counts[wh_type] += 1
        tag_counts[tag] += 1
        new_id = hash_id("cand", base.id, norm)
        tags = sorted(set(base.tags) | {tag})
        meta = dict(base.meta)
        meta["wh_type"] = wh_type
        meta["slots"] = slots
        meta["generation_tag"] = tag
        expanded.append(
            CandidateItem(
                id=new_id,
                question=question,
                answer_text=base.answer_text,
                doc_id=base.doc_id,
                doc_name=base.doc_name,
                page_start=base.page_start,
                page_end=base.page_end,
                char_start=base.char_start,
                char_end=base.char_end,
                tags=tags,
                source={**base.source},
                meta=meta,
            )
        )
    return expanded


def run(parsed_dir: Path, out_path: Path, config_path: Optional[Path] = None) -> List[CandidateItem]:
    config = load_yaml(config_path) if config_path else {}
    limits = config.get("limits", {})
    parsed_dir = Path(parsed_dir)
    if not parsed_dir.exists():
        raise FileNotFoundError(parsed_dir)

    base_candidates: List[CandidateItem] = []
    for doc in load_parsed_docs(parsed_dir):
        base_candidates.extend(harvest_doc(doc, limits))
    logger.info("Extracted %d base spans", len(base_candidates))

    banned_openings = [entry.lower() for entry in config.get("banned_openings", [])]
    reject_counts: Counter = Counter()
    wh_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    expanded_candidates: List[CandidateItem] = []
    seen_questions: set[str] = set()

    for candidate in base_candidates:
        expanded = _expand_candidate(candidate, config, banned_openings, reject_counts, wh_counts, tag_counts)
        for item in expanded:
            norm = _normalise_question(item.question)
            if norm in seen_questions:
                continue
            seen_questions.add(norm)
            expanded_candidates.append(item)

    logger.info("Expanded to %d candidates after filtering", len(expanded_candidates))
    logger.info("WH distribution: %s", dict(wh_counts))
    logger.info("Tag distribution: %s", dict(tag_counts))
    logger.info("Reject reasons: %s", dict(reject_counts))

    write_jsonl(out_path, (candidate_to_dict(item) for item in expanded_candidates))
    stats = summary_stats(expanded_candidates)
    stats["reject_reasons"] = dict(reject_counts)
    stats["tag_counts"] = dict(tag_counts)
    out_stats = out_path.with_name("candidates_stats.json")
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Candidate stats written to %s", out_stats)
    return expanded_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract gold-set candidates from parsed docs.")
    parser.add_argument("--parsed", required=True, help="Directory containing parsed JSON docs.")
    parser.add_argument("--out", required=True, help="Output JSONL path for candidates.")
    parser.add_argument("--config", default="gold/config.yaml", help="Configuration YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.parsed), Path(args.out), Path(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
