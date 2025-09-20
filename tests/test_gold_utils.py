import json
from collections import Counter
from pathlib import Path

from gold.assemble import run as assemble_run
from gold.extract_candidates import run as extract_run
from gold.quality import answerability_check, detect_wh, enforce_wh_distribution, is_entity_anchored
from gold.utils import read_jsonl


def create_parsed_fixture(path: Path) -> Path:
    parsed_dir = path / "parsed"
    parsed_dir.mkdir()
    payload = {
        "doc_id": "docA",
        "doc_name": "docA.pdf",
        "pages": [
            {
                "page_num": 1,
                "text": (
                    "1. Retention Policy\n\n"
                    "Retention Policy is defined as keeping data for 30 days.\n"
                    "Backup window: 6 hours\n"
                    "Availability: 99.9%\n"
                    "Table 1: Service levels | Tier 1\n"
                    "Figure 2 - Architecture Diagram\n"
                    "Incident response requires escalation to tier 2.\n"
                    "Acronym RPO stands for Recovery Point Objective.\n"
                ),
            }
        ],
    }
    (parsed_dir / "docA.json").write_text(json.dumps(payload), encoding="utf-8")
    return parsed_dir


def create_config(path: Path) -> Path:
    cfg = path / "config.yaml"
    cfg.write_text(
        "limits:\n"
        "  max_candidates_per_page: 10\n"
        "  max_answer_chars: 300\n"
        "  min_answer_chars: 3\n"
        "  max_questions_per_span: 4\n"
        "quotas:\n"
        "  total: 10\n"
        "  by_tag:\n"
        "    numeric: 0.2\n"
        "    definition: 0.2\n"
        "    caption: 0.2\n"
        "    table: 0.2\n"
        "    acronym: 0.2\n"
        "wh_targets:\n"
        "  what: 0.35\n"
        "  which: 0.15\n"
        "  who: 0.02\n"
        "  when: 0.05\n"
        "  where: 0.05\n"
        "  why: 0.15\n"
        "  how: 0.15\n"
        "  how_many: 0.04\n"
        "  how_much: 0.04\n"
        "  aux: 0.05\n"
        "banned_openings:\n"
        "  - 'what is '\n"
        "  - 'what’s '\n"
        "min_doc_items: 1\n"
        "max_doc_items: 10\n",
        encoding="utf-8",
    )
    return cfg


def test_extract_candidates_enforces_quality(tmp_path: Path) -> None:
    parsed_dir = create_parsed_fixture(tmp_path)
    config = create_config(tmp_path)
    out_path = tmp_path / "candidates.jsonl"
    candidates = extract_run(parsed_dir, out_path, config)
    assert candidates
    rows = read_jsonl(out_path)
    assert rows
    wh_categories = {detect_wh(row["question"]) for row in rows}
    assert len(wh_categories) >= 5
    what_count = sum(1 for row in rows if row["question"].lower().startswith("what"))
    assert what_count <= max(1, int(0.35 * len(rows)))
    for row in rows:
        slots = row.get("meta", {}).get("slots", {})
        page_text = row.get("meta", {}).get("page_text", "")
        assert slots, "slots missing"
        assert is_entity_anchored(row["question"], slots)
        assert answerability_check(page_text, (row["char_start"], row["char_end"]), row["question"], slots)


def test_assemble_applies_wh_and_doc_caps(tmp_path: Path) -> None:
    parsed_dir = create_parsed_fixture(tmp_path)
    config = create_config(tmp_path)
    candidates_path = tmp_path / "candidates.jsonl"
    extract_run(parsed_dir, candidates_path, config)
    gold_out = tmp_path / "gold.jsonl"
    stats_out = tmp_path / "stats.json"
    gold_items = assemble_run(
        candidates_path=candidates_path,
        out_path=gold_out,
        config_path=config,
        paraphrases_path=None,
        chunks_path=None,
        stats_path=stats_out,
    )
    assert gold_items
    saved = read_jsonl(gold_out)
    assert saved
    wh_counts = Counter(detect_wh(item["question"]) for item in saved)
    assert wh_counts.get("what", 0) <= max(1, int(0.35 * len(saved)))
    stats_payload = json.loads(stats_out.read_text(encoding="utf-8"))
    assert "reject_reasons" in stats_payload
    for reason in ("banned_opening", "vague_pronoun", "not_answerable"):
        assert reason in stats_payload["reject_reasons"]


def test_enforce_wh_distribution_shapes_output() -> None:
    questions = [
        "What does the policy cover?",
        "What is the retention period?",
        "Why is encryption required?",
        "How does the backup process run?",
        "Which team handles incidents?",
        "Where is data stored?",
        "When is the review conducted?",
        "How many replicas are maintained?",
        "How much downtime is allowed?",
        "Can the system operate offline?",
    ]
    targets = {
        "what": 0.2,
        "why": 0.1,
        "how": 0.2,
        "which": 0.1,
        "where": 0.1,
        "when": 0.1,
        "how_many": 0.1,
        "how_much": 0.05,
        "aux": 0.05,
    }
    shaped = enforce_wh_distribution(questions, targets, seed=7)
    assert shaped
    wh_counts = Counter(detect_wh(q) for q in shaped)
    total = len(shaped)
    assert wh_counts.get("what", 0) <= int(0.2 * total) + 1
    assert wh_counts.get("why", 0) >= 1
    assert len(wh_counts) >= 5
