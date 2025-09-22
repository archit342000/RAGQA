"""LLM-driven synthesis of question-answer candidates."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import orjson
import re
import yaml
from dotenv import load_dotenv
from tqdm.auto import tqdm

from gold.llm_client import VLLMClient
from gold.prompts import SYNTH_SYSTEM, SYNTH_USER_TEMPLATE
from gold.quality import detect_wh
from gold.spans import resolve_evidence_spans, sentence_spans as compute_sentence_spans
from gold.verify import (
    ALLOWED_WH,
    SynthItem,
    canonicalize_wh,
    parse_json_array,
    validate_synth_items,
)
from gold.window import make_windows
from gold.judge import LLMJudge, build_judge

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("gold/.cache")
def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_parsed_docs(parsed_dir: Path) -> Iterable[Dict]:
    for json_path in sorted(parsed_dir.glob("*.json")):
        with json_path.open("rb") as fh:
            yield orjson.loads(fh.read())


_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _prompt_hash(payload: dict) -> str:
    serialized = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha1(serialized).hexdigest()


def _cache_lookup(cache_dir: Path, cache_key: str) -> Optional[str]:
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        data = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return None
    response = data.get("response")
    if isinstance(response, str):
        return response
    return None


def _cache_store(cache_dir: Path, cache_key: str, response: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"response": response}
    path = cache_dir / f"{cache_key}.json"
    path.write_bytes(orjson.dumps(payload))


def _format_prompt(template: str, **values: object) -> str:
    """Safely replace ``{placeholders}`` in prompt templates."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in values:
            value = values[key]
            if value is None:
                return ""
            return str(value)
        return match.group(0)

    return _PLACEHOLDER_PATTERN.sub(replace, template)


def _build_messages(window: Dict, max_q: int, system_prompt: str) -> List[dict]:
    window_len = f"{window.get('char_len', len(window['window_text']))} chars; approx {window.get('token_count', 0)} tokens"
    user_prompt = _format_prompt(
        SYNTH_USER_TEMPLATE,
        doc_id=window["doc_id"],
        doc_name=window["doc_name"],
        page_start=window["page_start"],
        page_end=window["page_end"],
        window_len=window_len,
        max_q=max_q,
        window_text=window["window_text"],
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _sanitize_for_output(item: Dict) -> Dict:
    payload = dict(item)
    payload.pop("window_text", None)
    return payload


def _hash_item(doc_id: str, window_id: str, char_start: int, char_end: int, question: str) -> str:
    basis = f"{doc_id}|{window_id}|{char_start}|{char_end}|{question}".encode("utf-8")
    return hashlib.sha1(basis).hexdigest()


def _process_window(
    window: Dict,
    client: VLLMClient,
    config: Dict,
    system_prompt: str,
    cache_dir: Optional[Path],
    resume: bool,
    judge: Optional[LLMJudge],
) -> Tuple[List[Dict], Counter, int, int, Optional[str]]:
    max_q = config.get("limits", {}).get("max_questions_per_window", 8)
    temperature = config.get("temperature", {}).get("synth", 0.7)
    max_tokens = config.get("limits", {}).get("synth_max_tokens", 900)
    seed = config.get("seed")

    messages = _build_messages(window, max_q, system_prompt)
    hash_payload = {
        "model": client.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    prompt_hash = _prompt_hash(hash_payload)
    cache_key = hashlib.sha1(f"{window['window_id']}|{prompt_hash}".encode("utf-8")).hexdigest()

    response_text: Optional[str] = None
    if resume and cache_dir is not None:
        response_text = _cache_lookup(cache_dir, cache_key)

    if response_text is None:
        try:
            response_text = client.chat(messages, temperature, max_tokens, seed, response_format_json=True)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("LLM call failed for %s: %s", window["window_id"], exc)
            return [], Counter({"request_failed": 1}), 0, 0, str(exc)
        if resume and cache_dir is not None:
            _cache_store(cache_dir, cache_key, response_text)

    try:
        raw_items = parse_json_array(response_text)
    except ValueError:
        logger.warning("JSON parse failed for %s; skipping window", window["window_id"])
        return [], Counter({"parse_error": 1}), 0, 0, "parse_error"

    raw_count = len(raw_items)
    valid_items = validate_synth_items(raw_items)
    schema_drops = raw_count - len(valid_items)
    valid_items = valid_items[:max_q]

    drop_reasons: Counter = Counter()
    if schema_drops > 0:
        drop_reasons.update({"schema": schema_drops})

    if not valid_items:
        return [], drop_reasons, raw_count, 0, None

    sentence_spans_list = compute_sentence_spans(window["window_text"])
    kept: List[Dict] = []
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()

    for item in valid_items:
        evidence_spans = resolve_evidence_spans(item.evidence, sentence_spans_list)
        try:
            record = _process_item(item, window, sentence_spans_list, evidence_spans)
        except _DropItem as drop:
            drop_reasons.update({drop.reason: 1})
            continue
        norm_question = _normalize(record["question"])
        if norm_question in seen_questions:
            drop_reasons.update({"duplicate_question": 1})
            continue
        norm_answer = _normalize(record["answer_text"])
        if norm_answer in seen_answers:
            drop_reasons.update({"duplicate_answer": 1})
            continue

        if judge is not None:
            try:
                verdict = judge.evaluate(record, evidence_spans)
            except Exception as exc:  # pragma: no cover - defensive around LLM failures
                logger.warning(
                    "Judge evaluation failed for %s/%s: %s",
                    window["window_id"],
                    record.get("question", ""),
                    exc,
                )
                drop_reasons.update({"judge_error": 1})
                continue
            if verdict.error:
                drop_reasons.update({"judge_error": 1})
                continue
            if not verdict.passed:
                drop_reasons.update({"judge_reject": 1})
                continue

        seen_questions.add(norm_question)
        seen_answers.add(norm_answer)
        record["id"] = _hash_item(
            record["doc_id"],
            record["window_id"],
            record["char_start"],
            record["char_end"],
            record["question"],
        )
        kept.append(record)

    return kept, drop_reasons, raw_count, len(valid_items), None


class _DropItem(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _process_item(
    item: SynthItem,
    window: Dict,
    sentence_spans: Sequence[Tuple[int, int]],
    evidence_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict:
    question = item.question.strip()
    answer_text = item.answer_text.strip()
    if not answer_text:
        raise _DropItem("empty_answer")

    if not sentence_spans:
        raise _DropItem("no_sentence_spans")

    indices = [entry.index for entry in item.evidence]
    if len(indices) != len(set(indices)):
        raise _DropItem("duplicate_evidence")
    if any(idx < 0 or idx >= len(sentence_spans) for idx in indices):
        raise _DropItem("evidence_oob")
    if indices != sorted(indices):
        raise _DropItem("evidence_unsorted")

    if evidence_spans is None:
        evidence_spans = resolve_evidence_spans(item.evidence, sentence_spans)
    if not evidence_spans:
        raise _DropItem("evidence_alignment_failed")
    question_lower = question.lower()
    normalized_answer = answer_text.lower()
    stripped_answer = normalized_answer.strip(".,;:!?\"'()[]{}")
    if normalized_answer and normalized_answer in question_lower:
        raise _DropItem("answer_leak")
    if stripped_answer and stripped_answer in question_lower:
        raise _DropItem("answer_leak")
    char_start, char_end = -1, -1

    wh = canonicalize_wh(item.wh) if item.wh else detect_wh(question)
    if wh not in ALLOWED_WH:
        wh = detect_wh(question)

    evidence_entries = [entry.model_dump() for entry in item.evidence]

    record = {
        "doc_id": window["doc_id"],
        "doc_name": window["doc_name"],
        "page_start": window["page_start"],
        "page_end": window["page_end"],
        "window_id": window["window_id"],
        "question": question,
        "answer_text": answer_text,
        "wh": wh,
        "type": item.type.strip(),
        "evidence": evidence_entries,
        "tags": [],
        "char_start": char_start,
        "char_end": char_end,
        "window_text": window["window_text"],
    }
    return record


def run_synthesis(
    parsed_dir: Path,
    config: Dict,
    concurrency: int,
    resume: bool,
    out_path: Optional[Path] = None,
) -> Tuple[List[Dict], Dict]:
    window_cfg = config.get("window", {})
    max_q = config.get("limits", {}).get("max_questions_per_window", 8)
    system_prompt = _format_prompt(SYNTH_SYSTEM, max_q=max_q)

    load_dotenv()
    api_key = os.getenv(config.get("api_key_env", "VLLM_API_KEY"), "")
    if not api_key:
        raise RuntimeError("API key not found in environment")
    base_url = config.get("base_url") or os.getenv(config.get("base_url_env", "VLLM_BASE_URL"))
    if not base_url:
        raise RuntimeError("LLM base_url must be provided")

    timeout_s = config.get("timeout_s", 60)

    windows: List[Dict] = []
    for parsed_doc in _iter_parsed_docs(parsed_dir):
        doc_windows = make_windows(
            parsed_doc,
            max_pages=window_cfg.get("max_pages", 2),
            max_chars=window_cfg.get("max_chars", 12_000),
            max_tokens=window_cfg.get("max_tokens", 1_800),
        )
        windows.extend(doc_windows)

    stats = {
        "windows_total": len(windows),
        "windows_completed": 0,
        "raw_items": 0,
        "validated_items": 0,
        "kept_items": 0,
        "drop_reasons": Counter(),
        "wh_counts": Counter(),
        "type_counts": Counter(),
        "evidence_type_counts": Counter(),
        "per_doc": defaultdict(lambda: {"windows": 0, "kept": 0}),
        "errors": [],
    }

    cache_dir = _CACHE_DIR if resume else None
    all_items: List[Dict] = []

    with VLLMClient(base_url=base_url, api_key=api_key, model=config["model"], timeout_s=timeout_s) as client:
        judge = build_judge(config, client)

        worker = lambda w: _process_window(w, client, config, system_prompt, cache_dir, resume, judge)
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(worker, window): window for window in windows}
                for future in tqdm(as_completed(futures), total=len(futures), desc="synth windows"):
                    window = futures[future]
                    try:
                        kept, drops, raw_count, valid_count, error = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.exception("Window processing failed for %s", window["window_id"])
                        stats["errors"].append({"window_id": window["window_id"], "error": str(exc)})
                        continue
                    _update_stats(stats, window, kept, drops, raw_count, valid_count, error)
                    all_items.extend(kept)
        else:
            for window in tqdm(windows, desc="synth windows"):
                kept, drops, raw_count, valid_count, error = worker(window)
                _update_stats(stats, window, kept, drops, raw_count, valid_count, error)
                all_items.extend(kept)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            for item in all_items:
                fh.write(orjson.dumps(_sanitize_for_output(item)))
                fh.write(b"\n")

    stats["drop_reasons"] = dict(stats["drop_reasons"])
    stats["per_doc"] = {doc: dict(values) for doc, values in stats["per_doc"].items()}
    return all_items, stats


def _update_stats(
    stats: Dict,
    window: Dict,
    kept: Sequence[Dict],
    drops: Counter,
    raw_count: int,
    valid_count: int,
    error: Optional[str],
) -> None:
    stats["windows_completed"] += 1
    stats["raw_items"] += raw_count
    stats["validated_items"] += valid_count
    stats["kept_items"] += len(kept)
    stats["drop_reasons"].update(drops)
    if kept:
        wh_counts = stats.get("wh_counts")
        type_counts = stats.get("type_counts")
        evidence_counts = stats.get("evidence_type_counts")
        for item in kept:
            wh_value = (item.get("wh") or "").strip().lower()
            if not wh_value:
                wh_value = detect_wh(item.get("question", ""))
            if wh_counts is not None and wh_value:
                wh_counts[wh_value] += 1

            q_type = item.get("type") or ""
            q_type = q_type.strip()
            if type_counts is not None and q_type:
                type_counts[q_type] += 1

            if evidence_counts is not None:
                for ev in item.get("evidence", []) or []:
                    if isinstance(ev, dict):
                        ev_type = ev.get("type")
                    else:
                        ev_type = None
                    if isinstance(ev_type, str):
                        ev_value = ev_type.strip().lower()
                        if ev_value:
                            evidence_counts[ev_value] += 1
    doc_stats = stats["per_doc"][window["doc_id"]]
    doc_stats["windows"] += 1
    doc_stats["kept"] += len(kept)
    if error:
        stats["errors"].append({"window_id": window["window_id"], "error": error})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM synthesis over parsed documents")
    parser.add_argument("--parsed", type=Path, required=True, help="Directory of parsed document JSON files")
    parser.add_argument("--out", type=Path, required=True, help="Path to write synthesized candidates JSONL")
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", help="Reuse cached generations when available")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config(args.config)

    items, stats = run_synthesis(args.parsed, config, args.concurrency, args.resume, args.out)
    logger.info("Generated %d candidate items", len(items))
    logger.info("Drop reasons: %s", stats.get("drop_reasons"))


if __name__ == "__main__":
    main()
