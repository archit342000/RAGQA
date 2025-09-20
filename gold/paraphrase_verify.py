"""Optional paraphrasing with strict span verification and quality checks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import List, Optional

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore

from gold.models import CandidateItem
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
    candidate_to_dict,
    extract_slots_from_page,
    from_candidate,
    load_yaml,
    read_jsonl,
    verify_span,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gold.paraphrase")


def generate_paraphrases(question: str, allow_definition: bool, model: Optional[str], n: int = 2) -> List[str]:
    """Generate paraphrases using the configured LLM (fallback to identity)."""

    if not model or openai is None or not os.getenv("OPENAI_API_KEY"):
        return [question]
    client = openai.OpenAI()
    clause = (
        "Only start with 'What is' if the question restates a formal definition."
        if allow_definition
        else "Never start a paraphrase with 'What is'."
    )
    prompt = (
        "Rewrite the question in two diverse ways while preserving the answer span.\n"
        f"{clause}\n"
        "Use different WH-forms when possible.\n"
        "Each paraphrase must end with a question mark.\n\n"
        f"Original question: {question}"
    )
    response = client.responses.create(model=model, input=prompt, max_output_tokens=192)
    text = response.output[0].content[0].text  # type: ignore[attr-defined]
    paraphrases = [line.strip() for line in text.splitlines() if line.strip()]
    return paraphrases[:n] or [question]


def _normalise(question: str) -> str:
    return question.strip().lower()


def _passes_quality(
    question: str,
    slots: dict,
    page_text: str,
    span: tuple[int, int],
    banned: List[str],
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
    if not answerability_check(page_text, span, question, slots):
        reject_counts["not_answerable"] += 1
        return False
    return True


def run(input_path: Path, out_path: Path, model: Optional[str], config_path: Optional[Path]) -> List[dict]:
    config = load_yaml(config_path) if config_path else {}
    banned_openings = [entry.lower() for entry in config.get("banned_openings", [])]
    limits = config.get("limits", {})
    max_questions = limits.get("max_questions_per_span", 3)
    wh_targets = config.get("wh_targets") or {}

    rows = read_jsonl(input_path)
    logger.info("Loaded %d candidates for paraphrasing", len(rows))
    accepted: List[dict] = []
    seen = set()
    wh_counts = Counter()
    reject_counts = Counter()

    for row in rows:
        candidate = from_candidate(row)
        page_text = candidate.meta.get("page_text", "")
        if not verify_span(page_text, candidate.char_start, candidate.char_end, candidate.answer_text):
            continue
        slots = extract_slots_from_page(page_text, candidate_to_dict(candidate))
        allow_definition = "definition" in candidate.tags
        paraphrases = generate_paraphrases(candidate.question, allow_definition, model=model)
        accepted_local: List[str] = []
        for paraphrase in paraphrases:
            norm = _normalise(paraphrase)
            if norm in seen:
                continue
            if not _passes_quality(paraphrase, slots, page_text, (candidate.char_start, candidate.char_end), banned_openings, reject_counts):
                continue
            accepted_local.append(paraphrase)
        selected = mmr_select(accepted_local, page_text, k=max_questions, lambda_=0.7)
        for paraphrase in selected:
            norm = _normalise(paraphrase)
            if norm in seen:
                continue
            seen.add(norm)
            wh = detect_wh(paraphrase)
            wh_counts[wh] += 1
            updated = candidate_to_dict(candidate)
            updated["question"] = paraphrase
            tags = set(updated.get("tags", []))
            if paraphrase != candidate.question:
                tags.add("paraphrase")
            updated["tags"] = sorted(tags)
            meta = dict(updated.get("meta", {}))
            meta["wh_type"] = wh
            meta["slots"] = slots
            updated["meta"] = meta
            accepted.append(updated)

    if wh_targets and accepted:
        selected_questions = enforce_wh_distribution([item["question"] for item in accepted], wh_targets, seed=config.get("seed", 42))
        if selected_questions:
            quota = Counter(selected_questions)
            filtered: List[dict] = []
            for item in accepted:
                q = item["question"]
                if quota[q] > 0:
                    filtered.append(item)
                    quota[q] -= 1
            accepted = filtered
            wh_counts = Counter(item.get("meta", {}).get("wh_type", detect_wh(item["question"])) for item in accepted)

    write_jsonl(out_path, accepted)
    logger.info(
        "Paraphrased %d questions (WH distribution: %s, rejects: %s)",
        len(accepted),
        dict(wh_counts),
        dict(reject_counts),
    )
    stats_path = out_path.with_name("paraphrase_stats.json")
    stats_payload = {
        "total": len(accepted),
        "wh": dict(wh_counts),
        "reject_reasons": dict(reject_counts),
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paraphrase and verify candidate questions.")
    parser.add_argument("--in", dest="input_path", required=True, help="Input candidates JSONL.")
    parser.add_argument("--out", required=True, help="Output JSONL with paraphrased candidates.")
    parser.add_argument("--model", default="", help="LLM model identifier (optional).")
    parser.add_argument("--config", default="gold/config.yaml", help="Configuration YAML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.input_path), Path(args.out), args.model, Path(args.config))


if __name__ == "__main__":  # pragma: no cover
    main()
