"""CLI entrypoint for LLM-driven atomic fact mining (Pass A)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from gold.llm_client import VLLMClient
from gold.prompts import MINING_SYSTEM_PROMPT, MINING_USER_PROMPT_TEMPLATE
from gold.verify import Atom, hash_atom, strict_span_match
from gold.window import make_windows

logger = logging.getLogger(__name__)
FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)


@dataclass(slots=True)
class WindowResult:
    """Container for per-window mining outcomes."""

    window: Dict[str, Any]
    raw_count: int
    atoms: List[Atom]
    drop_counts: Counter[str]
    error: Optional[str]
    used_cache: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run LLM-first mining pass")
    parser.add_argument("--parsed", required=True, help="Directory with parsed document JSON files")
    parser.add_argument(
        "--out",
        default="gold/mined_atoms.jsonl",
        help="Destination JSONL file for mined atoms",
    )
    parser.add_argument(
        "--config",
        default="gold/llm_config.yaml",
        help="Path to mining configuration YAML",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent windows to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached LLM responses instead of re-querying",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration for the miner."""

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_parsed_doc(path: Path) -> Dict[str, Any]:
    """Load a parsed document JSON file."""

    return orjson.loads(path.read_bytes())


def ensure_output_file(path: Path) -> None:
    """Ensure the output directory exists and parent directories are created."""

    path.parent.mkdir(parents=True, exist_ok=True)


def escape_braces(value: str) -> str:
    """Escape braces so they survive str.format substitution."""

    return value.replace("{", "{{").replace("}", "}}")


def render_user_prompt(window: Dict[str, Any], max_items: int) -> str:
    """Render the user prompt for a window."""

    window_text = str(window.get("window_text", ""))
    prompt = MINING_USER_PROMPT_TEMPLATE.format(
        doc_id=escape_braces(str(window.get("doc_id", "unknown"))),
        doc_name=escape_braces(str(window.get("doc_name", "unknown"))),
        page_start=window.get("page_start"),
        page_end=window.get("page_end"),
        window_chars=len(window_text),
        max_items=max_items,
        window_text=escape_braces(window_text),
    )
    return prompt


def compute_prompt_hash(system_prompt: str, user_prompt: str, max_items: int) -> str:
    """Return a stable hash for the prompt pair."""

    payload = orjson.dumps({"system": system_prompt, "user": user_prompt, "max_items": max_items})
    return hashlib.sha256(payload).hexdigest()


def cache_key_for(window: Dict[str, Any], model: str, prompt_hash: str) -> str:
    """Derive a cache key for the window and prompt."""

    payload = orjson.dumps(
        {
            "doc_id": window.get("doc_id"),
            "page_start": window.get("page_start"),
            "page_end": window.get("page_end"),
            "window_id": window.get("window_id"),
            "model": model,
            "prompt_hash": prompt_hash,
        }
    )
    return hashlib.sha1(payload).hexdigest()


def extract_json_block(text: str) -> Optional[str]:
    """Attempt to recover a JSON block enclosed in ```json fences."""

    match = FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def parse_response_payload(raw_text: str) -> Optional[Any]:
    """Parse the raw LLM response into Python data if possible."""

    try:
        return orjson.loads(raw_text)
    except orjson.JSONDecodeError:
        extracted = extract_json_block(raw_text)
        if not extracted:
            return None
        try:
            return orjson.loads(extracted)
        except orjson.JSONDecodeError:
            return None


def process_window(
    window: Dict[str, Any],
    client: VLLMClient,
    temperature: float,
    max_tokens: int,
    seed: Optional[int],
    max_items: int,
    cache_dir: Path,
    resume: bool,
) -> WindowResult:
    """Run the mining prompt for a single window."""

    user_prompt = render_user_prompt(window, max_items=max_items)
    prompt_hash = compute_prompt_hash(MINING_SYSTEM_PROMPT, user_prompt, max_items)
    cache_key = cache_key_for(window, client.model, prompt_hash)
    cache_path = cache_dir / f"{cache_key}.json"

    response_text: Optional[str] = None
    used_cache = False
    if resume and cache_path.exists():
        try:
            cached = orjson.loads(cache_path.read_bytes())
        except orjson.JSONDecodeError:
            cached = None
        if isinstance(cached, dict) and cached.get("prompt_hash") == prompt_hash:
            cached_response = cached.get("response")
            if isinstance(cached_response, str):
                response_text = cached_response
                used_cache = True

    if response_text is None:
        messages = [
            {"role": "system", "content": MINING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response_text = client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                response_format_json=True,
            )
        except Exception as exc:  # pragma: no cover - network/runtime issues
            logger.warning("Window %s failed due to LLM error: %s", window.get("window_id"), exc)
            return WindowResult(window, raw_count=0, atoms=[], drop_counts=Counter(), error="llm_error", used_cache=False)

    parsed = parse_response_payload(response_text)
    if parsed is None:
        logger.warning("Window %s returned invalid JSON", window.get("window_id"))
        return WindowResult(window, raw_count=0, atoms=[], drop_counts=Counter({"invalid_json": 1}), error="invalid_json", used_cache=used_cache)

    if not isinstance(parsed, list):
        logger.warning("Window %s returned non-list payload", window.get("window_id"))
        return WindowResult(window, raw_count=0, atoms=[], drop_counts=Counter({"non_list": 1}), error="non_list", used_cache=used_cache)

    window_text = str(window.get("window_text", ""))
    drop_counts: Counter[str] = Counter()
    valid_atoms: List[Atom] = []
    for entry in parsed:
        try:
            atom = Atom.model_validate(entry)
        except ValidationError:
            drop_counts["schema"] += 1
            continue
        if not strict_span_match(window_text, atom.char_start, atom.char_end, atom.text):
            drop_counts["span_mismatch"] += 1
            continue
        valid_atoms.append(atom)

    raw_count = len(parsed)

    if not used_cache:
        cache_payload = {
            "response": response_text,
            "prompt_hash": prompt_hash,
            "model": client.model,
            "raw_count": raw_count,
        }
        try:
            cache_path.write_bytes(orjson.dumps(cache_payload))
        except OSError as exc:  # pragma: no cover - filesystem issues
            logger.debug("Failed to write cache for %s: %s", window.get("window_id"), exc)

    return WindowResult(window, raw_count=raw_count, atoms=valid_atoms, drop_counts=drop_counts, error=None, used_cache=used_cache)


def configure_logging() -> None:
    """Initialise logging for the CLI."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> int:
    """Program entrypoint."""

    args = parse_args()
    configure_logging()
    load_dotenv()

    config_path = Path(args.config)
    config = load_config(config_path)

    base_url_env = config.get("base_url_env")
    base_url: Optional[str] = None
    if base_url_env:
        base_url = os.getenv(str(base_url_env), "").strip()
        if not base_url:
            logger.error("Environment variable %s is not set", base_url_env)
            return 1
    else:
        raw_base_url = config.get("base_url")
        if raw_base_url:
            base_url = str(raw_base_url).strip()
    model = config.get("model")
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
    temperature = float(config.get("temperature", 0.3))
    max_tokens = int(config.get("max_tokens", 800))
    max_items = int(config.get("max_items", 10))
    timeout_s = int(config.get("timeout_s", 60))
    seed = config.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    window_cfg = config.get("window", {}) or {}
    window_max_pages = int(window_cfg.get("max_pages", 2))
    window_max_chars = int(window_cfg.get("max_chars", 12_000))
    window_max_tokens = int(window_cfg.get("max_tokens", 1_800))

    api_key = os.getenv(api_key_env or "")
    if not api_key:
        logger.error("Environment variable %s is not set", api_key_env)
        return 1
    if not base_url or not model:
        logger.error("Config must provide a model and base_url via config or environment")
        return 1

    parsed_dir = Path(args.parsed)
    if not parsed_dir.exists() or not parsed_dir.is_dir():
        logger.error("Parsed directory %s does not exist", parsed_dir)
        return 1

    doc_paths = sorted(path for path in parsed_dir.glob("*.json") if path.is_file())
    if not doc_paths:
        logger.warning("No parsed documents found in %s", parsed_dir)
        return 0

    out_path = Path(args.out)
    ensure_output_file(out_path)
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    max_workers = max(1, args.concurrency)

    seen_spans: set[tuple[str, str, int, int]] = set()

    with VLLMClient(base_url=base_url, api_key=api_key, model=model, timeout_s=timeout_s) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            with out_path.open("w", encoding="utf-8") as output_file:
                for doc_path in doc_paths:
                    parsed_doc = load_parsed_doc(doc_path)
                    doc_id = str(parsed_doc.get("doc_id", doc_path.stem))
                    doc_name = str(parsed_doc.get("doc_name", doc_id))
                    windows = make_windows(
                        parsed_doc,
                        max_pages=window_max_pages,
                        max_chars=window_max_chars,
                        max_tokens=window_max_tokens,
                    )
                    if not windows:
                        logger.info("Doc %s produced no windows", doc_id)
                        continue

                    futures = [
                        executor.submit(
                            process_window,
                            window,
                            client,
                            temperature,
                            max_tokens,
                            seed,
                            max_items,
                            cache_dir,
                            args.resume,
                        )
                        for window in windows
                    ]

                    doc_raw = 0
                    doc_kept = 0
                    doc_failed = 0
                    drop_totals: Counter[str] = Counter()

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.exception("Window processing crashed: %s", exc)
                            doc_failed += 1
                            drop_totals["exception"] += 1
                            continue

                        if result.error:
                            doc_failed += 1
                            drop_totals[result.error] += 1
                            continue

                        doc_raw += result.raw_count
                        drop_totals.update(result.drop_counts)

                        for atom in result.atoms:
                            key = (result.window.get("doc_id", doc_id), result.window.get("window_id"), atom.char_start, atom.char_end)
                            if key in seen_spans:
                                drop_totals["duplicate"] += 1
                                continue
                            seen_spans.add(key)

                            record = {
                                "id": f"atom-{hash_atom(doc_id, result.window.get('window_id', 'unknown'), atom.char_start, atom.char_end)}",
                                "doc_id": doc_id,
                                "doc_name": doc_name,
                                "window_id": result.window.get("window_id"),
                                "page_start": result.window.get("page_start"),
                                "page_end": result.window.get("page_end"),
                                "kind": atom.kind,
                                "text": atom.text,
                                "char_start": atom.char_start,
                                "char_end": atom.char_end,
                                "labels": atom.labels,
                                "evidence": atom.evidence,
                                "tags": atom.tags,
                            }
                            output_file.write(orjson.dumps(record).decode("utf-8") + "\n")
                            doc_kept += 1

                    logger.info(
                        "Doc %s: windows=%d raw=%d kept=%d failed_windows=%d drops=%s",
                        doc_id,
                        len(windows),
                        doc_raw,
                        doc_kept,
                        doc_failed,
                        dict(drop_totals),
                    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
