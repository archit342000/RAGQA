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

from gold.align import find_span
from gold.llm_client import VLLMClient
from gold.prompts import SYNTH_SYSTEM, SYNTH_USER_TEMPLATE
from gold.quality import (
    detect_wh,
    has_banned_opening,
    is_entity_anchored,
    no_vague_pronoun,
    readability_bounds,
)
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


_SENTENCE_REGEX = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for match in _SENTENCE_REGEX.finditer(text):
        start, end = match.span()
        snippet = text[start:end].strip()
        if not snippet:
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def _resolve_evidence_spans(
    window_text: str,
    evidence: Sequence[dict],
    sentence_spans: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for entry in evidence:
        indices: List[int] = []
        if isinstance(entry, dict):
            if "index" in entry and isinstance(entry["index"], int):
                indices.append(entry["index"])
            if "indices" in entry and isinstance(entry["indices"], list):
                indices.extend(i for i in entry["indices"] if isinstance(i, int))
            snippet_key = entry.get("sentence") or entry.get("text")
            if isinstance(snippet_key, str) and snippet_key.strip():
                snippet = snippet_key.strip()
                loc = window_text.find(snippet)
                if loc != -1:
                    spans.append((loc, loc + len(snippet)))
                    continue
        elif isinstance(entry, int):
            indices.append(entry)
        for idx in indices:
            if 0 <= idx < len(sentence_spans):
                spans.append(sentence_spans[idx])
            elif 1 <= idx <= len(sentence_spans):  # tolerate 1-based indexing
                spans.append(sentence_spans[idx - 1])
    deduped: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for start, end in spans:
        key = (start, end)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


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


def _build_messages(window: Dict, max_q: int, system_prompt: str) -> List[dict]:
    window_len = f"{window.get('char_len', len(window['window_text']))} chars; approx {window.get('token_count', 0)} tokens"
    user_prompt = SYNTH_USER_TEMPLATE.format(
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
    banned = config.get("banned_openings", [])
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

    sentence_spans = _sentence_spans(window["window_text"])
    kept: List[Dict] = []
    dedup_keys: set[Tuple[str, str]] = set()

    for item in valid_items:
        evidence_spans = _resolve_evidence_spans(window["window_text"], item.evidence, sentence_spans)
        try:
            record = _process_item(item, window, sentence_spans, banned, evidence_spans)
        except _DropItem as drop:
            drop_reasons.update({drop.reason: 1})
            continue
        norm_question = " ".join(record["question"].lower().split())
        norm_answer = " ".join(record["answer_text"].lower().split())
        key = (norm_question, norm_answer)
        if key in dedup_keys:
            drop_reasons.update({"duplicate": 1})
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

        dedup_keys.add(key)
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
    banned_openings: Sequence[str],
    evidence_spans: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict:
    question = item.question.strip()
    answer_text = item.answer_text.strip()
    if not answer_text:
        raise _DropItem("empty_answer")

    if evidence_spans is None:
        evidence_spans = _resolve_evidence_spans(window["window_text"], item.evidence, sentence_spans)
    span = find_span(window["window_text"], answer_text, evidence_spans)
    if span is None:
        raise _DropItem("alignment_failed")
    char_start, char_end = span
    aligned_answer = window["window_text"][char_start:char_end]

    if not is_entity_anchored(question, aligned_answer, window["window_text"]):
        raise _DropItem("anchoring")
    if item.type.lower() != "definition" and has_banned_opening(question, banned_openings):
        raise _DropItem("banned_opening")
    if not no_vague_pronoun(question):
        raise _DropItem("vague_pronoun")
    if not readability_bounds(question):
        raise _DropItem("readability")

    wh = canonicalize_wh(item.wh) if item.wh else detect_wh(question)
    if wh not in ALLOWED_WH:
        wh = detect_wh(question)

    record = {
        "doc_id": window["doc_id"],
        "doc_name": window["doc_name"],
        "page_start": window["page_start"],
        "page_end": window["page_end"],
        "window_id": window["window_id"],
        "question": question,
        "answer_text": aligned_answer,
        "wh": wh,
        "type": item.type.strip(),
        "evidence": item.evidence,
        "tags": item.tags,
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
    system_prompt = SYNTH_SYSTEM.format(max_q=max_q)

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
