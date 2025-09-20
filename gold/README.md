# Gold Set Utilities

This toolkit derives an in-domain evaluation dataset for the RAG system. It
operates purely on the cleaned, parsed documents already produced by the app.

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
