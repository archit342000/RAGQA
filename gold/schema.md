# Gold Schema Overview

## CandidateItem (intermediate)

```json
{
  "id": "cand-7e3f1c92",
  "question": "What does the retention policy describe?",
  "answer_text": "Retention Policy is defined as keeping data for 30 days.",
  "doc_id": "policy.pdf",
  "doc_name": "policy.pdf",
  "page_start": 12,
  "page_end": 12,
  "char_start": 3487,
  "char_end": 3545,
  "tags": ["definition", "numeric"],
  "source": {"type": "definition", "doc_name": "policy.pdf", "page_num": 12},
  "meta": {"page_text": "..."}
}
```

## Synthesised LLM candidate (JSON record)

LLM generation returns lightweight dictionaries rather than `CandidateItem`
instances. They include additional metadata emitted by the prompt schema.

```json
{
  "doc_id": "doc-1",
  "doc_name": "doc-1.pdf",
  "page_start": 4,
  "page_end": 4,
  "window_id": "doc-1:0",
  "question": "Which component handles nightly backups?",
  "wh": "which",
  "type": "procedural",
  "answer_text": "The storage gateway handles nightly backups.",
  "evidence": [
    {"type": "sentence", "index": 2},
    {"type": "sentence", "index": 3}
  ],
  "char_start": 812,
  "char_end": 870,
  "tags": [],
  "window_text": "..."
}
```

Each evidence object records the zero-based index of the supporting snippet
within the window alongside a coarse type hint (`sentence`, `table_cell`,
`list_item`, or `figure_caption`). Downstream metrics aggregate these `wh`,
`type`, and evidence-type values to surface prompt compliance statistics.

## GoldItem (final)

```json
{
  "id": "g-0001",
  "question": "What does the retention policy describe?",
  "answer_text": "Retention Policy is defined as keeping data for 30 days.",
  "doc_id": "policy.pdf",
  "page_start": 12,
  "page_end": 12,
  "char_start": 3487,
  "char_end": 3545,
  "tags": ["definition", "numeric"],
  "hard_negative_ids": ["chunk_42", "chunk_77"],
  "evidence": [
    {"doc_id": "policy.pdf", "page": 12, "char_start": 3487, "char_end": 3545}
  ],
  "program": null
}
```

### Tag guidance

| Tag         | When applied                                                              |
|-------------|---------------------------------------------------------------------------|
| `numeric`   | Answer contains quantities/units (%, days, GB, users, etc.).              |
| `table`     | Extracted from tabular/key-value structures.                              |
| `definition`| Matches “is/means/defined as …” regex.                                    |
| `caption`   | Derived from figure/table captions.                                       |
| `acronym`   | Answer contains ≥1 ALL CAPS token of length ≥3.                           |
| `whyhow`    | Question starts with “why” or “how”.                                      |
| `paraphrase`| Added during paraphrase verification.                                     |

All spans must satisfy `char_end > char_start` and match the underlying page
text exactly.
