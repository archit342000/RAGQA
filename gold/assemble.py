"""Assemble stratified gold set with hard negatives and stats."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from gold.models import CandidateItem, GoldItem
from gold.quality import detect_wh, enforce_wh_distribution
from gold.utils import (
    bm25_hard_negatives,
    build_bm25_index,
    candidate_to_dict,
    dedupe_candidates,
    gold_to_dict,
    load_yaml,
    read_jsonl,
    stratified_sample,
    summary_stats,
    to_gold_item,
    verify_span,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gold.assemble")


def load_candidates(paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def filter_candidates(rows: List[dict], limits: Dict[str, int]) -> List[dict]:
    max_len = limits.get("max_answer_chars", 300)
    min_len = limits.get("min_answer_chars", 5)
    filtered: List[dict] = []
    for row in rows:
        answer = row["answer_text"].strip()
        if not (min_len <= len(answer) <= max_len):
            continue
        if not answer:
            continue
        filtered.append(row)
    return filtered


def build_bm25(chunks_path: Optional[Path]) -> Optional[tuple]:
    if not chunks_path or not chunks_path.exists():
        return None
    chunks = read_jsonl(chunks_path)
    if not chunks:
        return None
    return build_bm25_index(chunks)


def _load_side_stats(path: Path) -> dict:
    candidates = [
        path.with_name(f"{path.stem}_stats.json"),
        path.with_suffix(".stats.json"),
    ]
    if "paraphrase" in path.stem:
        candidates.append(path.parent / "paraphrase_stats.json")
    else:
        candidates.append(path.parent / "candidates_stats.json")
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def _aggregate_rejects(paths: List[Path]) -> Dict[str, int]:
    reasons = Counter()
    for path in paths:
        stats_payload = _load_side_stats(path)
        for key, value in stats_payload.get("reject_reasons", {}).items():
            reasons[key] += int(value)
    for key in ("banned_opening", "vague_pronoun", "not_answerable"):
        reasons.setdefault(key, 0)
    return dict(reasons)


def _enforce_wh_targets(items: List[CandidateItem], targets: Dict[str, float], seed: int) -> List[CandidateItem]:
    if not targets or not items:
        return items
    selected_questions = enforce_wh_distribution([item.question for item in items], targets, seed)
    if not selected_questions:
        return items
    quota = Counter(selected_questions)
    filtered: List[CandidateItem] = []
    for item in items:
        if quota[item.question] > 0:
            filtered.append(item)
            quota[item.question] -= 1
    return filtered


def _apply_doc_caps(sample: List[dict], min_cap: int, max_cap: int) -> List[dict]:
    if not sample:
        return sample
    per_doc: Dict[str, List[dict]] = defaultdict(list)
    for item in sample:
        per_doc[item["doc_id"]].append(item)
    eligible: Dict[str, List[dict]] = {}
    for doc_id, items in per_doc.items():
        limited = items[:max_cap] if max_cap and max_cap > 0 else list(items)
        if min_cap and min_cap > 0 and len(limited) < min_cap:
            continue
        eligible[doc_id] = limited
    if not eligible:
        return []
    doc_indices = {doc_id: 0 for doc_id in eligible}
    filtered: List[dict] = []
    for item in sample:
        doc_id = item["doc_id"]
        if doc_id not in eligible:
            continue
        idx = doc_indices[doc_id]
        limited = eligible[doc_id]
        if idx >= len(limited):
            continue
        if limited[idx]["id"] == item["id"]:
            filtered.append(item)
            doc_indices[doc_id] += 1
    return filtered


def run(
    candidates_path: Path,
    out_path: Path,
    config_path: Optional[Path] = None,
    paraphrases_path: Optional[Path] = None,
    chunks_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
) -> List[GoldItem]:
    config = load_yaml(config_path) if config_path else {}
    quotas = config.get("quotas", {})
    limits = config.get("limits", {})
    total = quotas.get("total", 100)
    by_tag = quotas.get("by_tag", {})
    seed = config.get("seed", 42)
    hard_neg_limit = config.get("hard_negatives", 5)
    wh_targets = config.get("wh_targets") or {}
    min_doc_items = config.get("min_doc_items", 0)
    max_doc_items = config.get("max_doc_items", 0)

    candidate_files = [candidates_path]
    if paraphrases_path and paraphrases_path.exists():
        candidate_files.append(paraphrases_path)
    raw_candidates = load_candidates(candidate_files)
    logger.info("Loaded %d candidates", len(raw_candidates))

    filtered_rows = filter_candidates(raw_candidates, limits)
    logger.info("Filtered to %d candidates", len(filtered_rows))

    candidate_objs = dedupe_candidates([CandidateItem(**row) for row in filtered_rows], limits.get("max_candidates_per_page", 10))
    candidate_objs = _enforce_wh_targets(candidate_objs, wh_targets, seed)
    candidate_dicts = [candidate_to_dict(item) for item in candidate_objs]
    logger.info("Deduplicated to %d candidates after WH enforcement", len(candidate_dicts))

    sample = stratified_sample(candidate_dicts, by_tag, total, seed)
    logger.info("Sampled %d candidates for gold before doc caps", len(sample))

    sample = _apply_doc_caps(sample, min_doc_items, max_doc_items)
    logger.info("Sampled %d candidates after doc caps", len(sample))

    bm25_index = build_bm25(chunks_path)
    gold_items: List[GoldItem] = []
    for idx, cand in enumerate(sample, start=1):
        page_text = cand.get("meta", {}).get("page_text", "")
        if page_text and not verify_span(page_text, cand["char_start"], cand["char_end"], cand["answer_text"]):
            continue
        hard_negs = bm25_hard_negatives(
            cand["question"],
            cand["answer_text"],
            bm25_index,
            cand["doc_id"],
            cand["page_start"],
            hard_neg_limit,
        )
        gold_items.append(to_gold_item(cand, hard_negs, idx))

    write_jsonl(out_path, (gold_to_dict(item) for item in gold_items))
    logger.info("Wrote %d gold items to %s", len(gold_items), out_path)

    candidate_stats = summary_stats(candidate_objs)
    gold_tag_counts = Counter(tag for item in gold_items for tag in item.tags)
    gold_wh_counts = Counter(detect_wh(item.question) for item in gold_items)
    reject_reasons = _aggregate_rejects(candidate_files)

    stats = {
        "candidate_summary": candidate_stats,
        "gold": {
            "count": len(gold_items),
            "tag_counts": dict(gold_tag_counts),
            "wh_counts": dict(gold_wh_counts),
        },
        "reject_reasons": reject_reasons,
    }
    stats_path = stats_path or out_path.with_name("stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Stats written to %s", stats_path)
    return gold_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble final gold set.")
    parser.add_argument("--candidates", required=True, help="Candidates JSONL.")
    parser.add_argument("--paraphrases", help="Optional paraphrased candidates JSONL.")
    parser.add_argument("--chunks", help="Chunks JSONL for hard negatives.")
    parser.add_argument("--config", default="gold/config.yaml")
    parser.add_argument("--out", required=True, help="Output gold JSONL.")
    parser.add_argument("--stats", help="Stats JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        candidates_path=Path(args.candidates),
        out_path=Path(args.out),
        config_path=Path(args.config) if args.config else None,
        paraphrases_path=Path(args.paraphrases) if args.paraphrases else None,
        chunks_path=Path(args.chunks) if args.chunks else None,
        stats_path=Path(args.stats) if args.stats else None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
