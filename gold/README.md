# LLM Gold-Set Builder

This workflow generates high-quality QA pairs directly from parsed documents
using a vLLM-hosted model. Questions are produced by the LLM, but answer spans
are aligned deterministically by the codebase.

## Quickstart

1. **Start the vLLM OpenAI server**

   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model gpt-oss-20b --host 0.0.0.0 --port 8000
   ```

2. **Provide an API key (any non-empty string works locally)**

   ```bash
   export VLLM_API_KEY="dummy"
   ```

3. **Run synthesis + gold assembly**

   ```bash
   python gold/build_gold_llm.py \
     --parsed artifacts/parsed_docs \
     --out gold/gold.jsonl \
     --config gold/llm_config.yaml \
     --concurrency 4 \
     --resume
   ```

   This command:
   - Generates per-window candidates via `llm_synthesize.py`
   - Aligns answer spans inside each window
   - Applies WH-distribution quotas and writes `gold/gold.jsonl`
   - Records run statistics in `gold/stats.json`

The optional flag `write_candidates: true` in `gold/llm_config.yaml` also saves
the pre-quota pool to `gold/candidates.jsonl` for inspection.

## Implementation Notes

- The LLM must return a JSON array of objects containing `question`, `wh`,
  `type`, `answer_text`, and `evidence`. Character offsets are computed by the
  alignment utilities and not supplied by the model.
- Answer spans are validated via exact/case-insensitive/whitespace/fuzzy
  matching. Items failing alignment, quality filters, or deduplication are
  dropped with reasons recorded in `stats.json`.
- WH quotas are enforced with an MMR-based sampler to maintain diversity while
  capping dominant categories.
