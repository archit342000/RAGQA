# Evaluation Gold Schema (Standard v1.1)

The offline evaluation harness expects a JSONL or CSV file where each row
represents a question-answer pair drawn from the same domain as the chunked
documents. The minimal fields required for IR metrics are **bolded**.

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| **id** | string | yes | Unique identifier for the question. |
| **question** | string | yes | Natural-language query posed to the retriever. |
| answer_text | string | optional | Reference answer or supporting blurb (used for qualitative analysis only). |
| **doc_id** | string | yes | Identifier of the gold document containing the answer. Must match the `doc_id` emitted by the chunking pipeline. |
| **page_start** | int | yes | Inclusive page number where the answer span begins. |
| **page_end** | int | yes | Inclusive page number where the answer span ends. |
| char_start | int | optional | Character offset of the answer span within the gold document (used for context precision). |
| char_end | int | optional | Character offset of the answer span end (exclusive). |
| tags | list[str] | optional | Topical or difficulty tags, e.g. `numeric`, `table`, `paraphrase`, `acronym`, `whyhow`, `multi-hop`. |
| hard_negative_ids | list[str] | optional | Chunk identifiers that are known hard negatives for the question. |

## Notes

- Page boundaries are inclusive: `page_start <= answer <= page_end`.
- When `char_start`/`char_end` are omitted, context precision falls back to a
  page-based proxy overlap.
- Tags are used to slice metrics in the report; provide an empty list when
  unavailable.
- CSV encoding should use UTF-8 with a header row matching the field names
  above. JSONL rows should be flat objects.
- Additional fields are preserved but ignored by the harness.
