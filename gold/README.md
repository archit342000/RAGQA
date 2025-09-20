# Gold Set Utilities

This toolkit derives an in-domain evaluation dataset for the RAG system. It
operates purely on the cleaned, parsed documents already produced by the app.

## Pass A: LLM span mining quickstart

1. **Start vLLM with an OpenAI-compatible server**

   ```bash
   python -m vllm.entrypoints.openai.api_server --model <hf-id-or-path> --host 0.0.0.0 --port 8000
   ```

2. **Configure environment variables**

   The miner automatically loads `.env`. Ensure it includes the vLLM endpoint and
   API token (override with `export` if you prefer):

   ```bash
   # .env defaults
   VLLM_BASE_URL=http://localhost:8000/v1
   VLLM_API_KEY=dummy
   ```

3. **Run the mining pass**

   ```bash
    python gold/llm_mine.py \
      --parsed artifacts/parsed_docs \
      --out gold/mined_atoms.jsonl \
      --config gold/llm_config.yaml \
      --concurrency 4 \
      --resume
   ```

Notes:

- Responses are cached per window under `gold/.cache/<hash>.json` when `--resume`
  is supplied.
- Each mined atom is span-validated; invalid responses are dropped.
- If the model ignores the JSON response format hint, ensure the prompt in
  `gold/prompts.py` enforces JSON-only output.

## Pipeline

1. **Extract candidates**

   ```bash
   python gold/extract_candidates.py \
     --parsed artifacts/parsed_docs \
     --out gold/candidates.jsonl \
     --config gold/config.yaml
   ```

   Generates `CandidateItem` rows with verified answer spans and heuristic
   tags.

2. **Paraphrase (optional)**

   ```bash
   python gold/paraphrase_verify.py \
     --in gold/candidates.jsonl \
     --out gold/candidates_paraphrased.jsonl \
     --model gpt-4o-mini
   ```

   Adds paraphrased questions while re-validating the stored span.

3. **Assemble final gold set**

   ```bash
   python gold/assemble.py \
     --candidates gold/candidates.jsonl \
     --paraphrases gold/candidates_paraphrased.jsonl \
     --chunks artifacts/chunks.jsonl \
     --out gold/gold.jsonl \
     --config gold/config.yaml
   ```

   Stratifies sampling by tag, mines hard negatives, and emits
   `gold/gold.jsonl` plus `gold/stats.json`.

See `gold/schema.md` for field definitions. All scripts are deterministic given
`gold/config.yaml` and required inputs.
