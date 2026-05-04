# Djinni LLM NER Bootstrap

This repo now contains the Stage 1 data-generation pipeline for the workflow described in the slide deck:

- pull a smaller subset from the two Hugging Face recruitment datasets
- keep only about 10,000 CVs and 10,000 job descriptions
- annotate them with a zero-shot LLM
- export span labels that can later train a custom span-based RoBERTa NER model
- keep qualification facts for the later hard-constraint scoring stage

## Files

- `prepare_hf_recruitment_subset.py`: downloads a 10k CV + 10k JD subset directly from Hugging Face.
- `prepare_seed_subset.py`: filters, cleans, and samples a balanced seed subset for manual review or LLM labeling.
- `djinni_gemini_ner.py`: Gemini-based CLI for CSV/JSONL annotation.
- `djinni_openai_ner.py`: OpenAI-based CLI for CSV/JSONL annotation with structured outputs.
- `djinni_openai_batch_ner.py`: OpenAI Batch API pipeline for cheaper large-scale Stage 1 annotation.
- `run_openai_batch_queue.ps1`: sequential runner that automatically submits one shard at a time, polls status, downloads outputs, and can finalize at the end.
- `convert_span_annotations.py`: converts manual/public span annotations into the JSONL expected by `train_span_ner.py`.
- `bootstrap_weak_annotations.py`: generates weak span annotations directly from the IT seed subset when manual Doccano labeling is not available yet.
- `train_span_ner.py`: trains a custom span-based NER model on the JSONL annotations produced by Gemini.
- `train_roberta_base.ps1`: ready-to-run PowerShell wrapper for real training with `roberta-base`.
- `onet_mapping.py`: shared Stage 2 utilities for building an O*NET index and matching extracted entities.
- `prepare_onet_index.py`: turns official O*NET exports into a compact JSONL index.
- `map_entities_to_onet.py`: maps Stage 1 entities in JSONL rows onto O*NET occupations and descriptors.
- `tests/test_djinni_gemini_ner.py`: small regression tests for span repair and chunking.
- `tests/test_prepare_hf_recruitment_subset.py`: small regression tests for Hugging Face subset normalization.
- `tests/test_prepare_seed_subset.py`: small regression tests for cleaning, filtering, and seed sampling.
- `tests/test_train_span_ner.py`: regression tests for span preprocessing and train/dev/test splitting.
- `tests/test_onet_mapping.py`: regression tests for Stage 2 O*NET indexing and mapping.

## Data Source

The subset script pulls directly from these Hugging Face datasets through the `datasets` Python library, not by scraping job pages on the web:

- `lang-uk/recruitment-dataset-candidate-profiles-english`
- `lang-uk/recruitment-dataset-job-descriptions-english`

It uses `datasets.load_dataset(..., streaming=True)` and `load_dataset_builder(...)`.

## Step 1: Prepare A 20k Subset

Install the downloader dependency first if needed:

```powershell
python -m pip install datasets
```

This creates one JSONL file with:

- 10,000 rows where `doc_type="cv"`
- 10,000 rows where `doc_type="jd"`

```powershell
python prepare_hf_recruitment_subset.py `
  --output .\hf_recruitment_subset_20k.jsonl `
  --cv-limit 10000 `
  --jd-limit 10000 `
  --overwrite
```

If you want the downloader itself to keep only IT rows and continue scanning until it collects enough accepted rows:

```powershell
python prepare_hf_recruitment_subset.py `
  --output .\hf_recruitment_it_subset_20k.jsonl `
  --cv-limit 10000 `
  --jd-limit 10000 `
  --it-only `
  --overwrite
```

That mode writes `source_offset` into each row so `--resume` can continue from the correct raw Hugging Face position later.

If you want a stricter IT-only subset that removes most ambiguous business/support rows and keeps only clearly technical roles (or rows with strong technical evidence), use:

```powershell
python prepare_hf_recruitment_subset.py `
  --output .\hf_recruitment_it_strict_subset_20k.jsonl `
  --cv-limit 10000 `
  --jd-limit 10000 `
  --it-strict `
  --overwrite
```

This mode is better when you want to reduce wasted LLM tokens before Stage 1 annotation.

If the download stops midway, rerun with `--resume`:

```powershell
python prepare_hf_recruitment_subset.py `
  --output .\hf_recruitment_subset_20k.jsonl `
  --cv-limit 10000 `
  --jd-limit 10000 `
  --resume
```

The same works for the IT-only downloader:

```powershell
python prepare_hf_recruitment_subset.py `
  --output .\hf_recruitment_it_subset_20k.jsonl `
  --cv-limit 10000 `
  --jd-limit 10000 `
  --it-only `
  --resume
```

If you have a Hugging Face token, set `HF_TOKEN` first for better Hub limits:

```powershell
$env:HF_TOKEN="YOUR_HF_TOKEN"
```

If you want a different slice, shift the window:

```powershell
python prepare_hf_recruitment_subset.py `
  --output .\hf_recruitment_subset_20k.jsonl `
  --cv-limit 10000 `
  --jd-limit 10000 `
  --cv-offset 10000 `
  --jd-offset 10000 `
  --overwrite
```

The output schema is normalized to:

- `id`
- `source_id`
- `doc_type`
- `text`
- `source_dataset`
- `source_language`

and a few useful metadata fields such as position, keyword, English level, and experience years.

## What The Script Extracts

Entity labels:

- `TECHNOLOGY`
- `JOB_ROLE`
- `SKILL`
- `WORK_ACTIVITY`
- `INDUSTRY`
- `PROJECT_TYPE`
- `DEGREE`
- `CERTIFICATION`

Qualification facts:

- `EXPERIENCE_YEARS`
- `DEGREE`
- `CERTIFICATION`

Each output record keeps:

- original `text`
- exact character spans in `entities`
- explicit degree / certification / experience facts in `qualification_facts`
- `document_type`, inferred language, and IT-domain flag

## Usage

Recommended: keep the API key outside the code.

You can pass one Gemini key, or a whole file of keys to rotate through.

Before sending anything to Gemini, prepare a cleaner and smaller seed subset:

```powershell
python prepare_seed_subset.py `
  --input .\hf_recruitment_subset_20k.jsonl `
  --seed-output .\hf_recruitment_seed_1000.jsonl `
  --cleaned-output .\hf_recruitment_cleaned.jsonl `
  --stats-output .\hf_recruitment_seed_stats.json `
  --seed-cv 500 `
  --seed-jd 500 `
  --overwrite
```

This step:

- cleans line endings, whitespace, and common mojibake
- drops rows that are too short or too sparse
- keeps English rows only
- samples a diverse, balanced seed set across CV/JD, `primary_keyword`, and length buckets

If you only want IT roles in the seed set:

```powershell
python prepare_seed_subset.py `
  --input .\hf_recruitment_subset_20k.jsonl `
  --seed-output .\hf_recruitment_seed_it_1000.jsonl `
  --cleaned-output .\hf_recruitment_cleaned_it.jsonl `
  --stats-output .\hf_recruitment_seed_it_stats.json `
  --seed-cv 500 `
  --seed-jd 500 `
  --it-only `
  --overwrite
```

For large-scale thesis annotation, the recommended OpenAI path is Batch API because it is cheaper than synchronous requests.

Batch workflow:

1. Prepare request shards and local manifests.
2. Submit batch jobs.
3. Refresh status until the batch finishes.
4. Download the output file(s).
5. Finalize them into `djinni_ner_annotations_openai_20k.jsonl`.

Example for the strict 20k IT subset:

```powershell
python djinni_openai_batch_ner.py prepare `
  --input .\hf_recruitment_it_strict_subset_20k.jsonl `
  --work-dir .\artifacts\openai_batch_strict_20k `
  --id-column id `
  --text-column text `
  --doc-type-column doc_type `
  --language-column source_language `
  --overwrite
```

## Stage 2: Map Extracted Entities To O*NET

Stage 2 assumes you already have Stage 1 output with `entities` in each JSONL row.

First build a compact local index from the official O*NET export folder:

```powershell
python prepare_onet_index.py `
  --onet-dir .\onet_db_28_3_text `
  --output .\artifacts\onet_index.jsonl `
  --overwrite
```

Then map the extracted entities onto O*NET occupations and descriptors:

```powershell
python map_entities_to_onet.py `
  --input .\djinni_ner_annotations_openai_20k_cleaned_v5.jsonl `
  --onet-index .\artifacts\onet_index.jsonl `
  --output .\artifacts\stage2_onet_mapped.jsonl `
  --min-score 0.35 `
  --overwrite
```

Each output row keeps the original Stage 1 fields plus:

- `onet_mappings`: candidate O*NET matches for each entity
- `onet_mapping_summary`: record-level summary including mapped rate and top O*NET-SOC codes

The mapper currently supports:

- `JOB_ROLE -> occupation / alternate title`
- `SKILL -> skill / knowledge / ability`
- `WORK_ACTIVITY -> work activity / task statement`
- `TECHNOLOGY -> technology skill / skill / knowledge`
- `PROJECT_TYPE -> task statement / work activity / technology skill`

Labels like `INDUSTRY`, `DEGREE`, and `CERTIFICATION` stay available for later hard-constraint logic, but they are not mapped into O*NET in this Stage 2 script yet.

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python djinni_openai_batch_ner.py submit `
  --work-dir .\artifacts\openai_batch_strict_20k
```

```powershell
python djinni_openai_batch_ner.py status `
  --work-dir .\artifacts\openai_batch_strict_20k
```

```powershell
python djinni_openai_batch_ner.py download `
  --work-dir .\artifacts\openai_batch_strict_20k
```

```powershell
python djinni_openai_batch_ner.py finalize `
  --work-dir .\artifacts\openai_batch_strict_20k `
  --output .\djinni_ner_annotations_openai_20k.jsonl `
  --require-english `
  --require-it
```

On the current strict IT subset, `prepare` produces one batch shard of about `148 MB` and `20,000` request lines, which fits under the batch file size limit.

If your OpenAI organization has a low enqueued-token queue limit, use the sequential runner below instead of manually repeating `submit -> status -> download`:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_openai_batch_queue.ps1 `
  -WorkDir .\artifacts\openai_batch_strict_20k_small `
  -PollSeconds 120 `
  -Finalize `
  -Output .\djinni_ner_annotations_openai_20k.jsonl `
  -RequireEnglish `
  -RequireIt
```

That runner only submits one shard at a time, waits for completion, downloads the result, then moves on to the next shard.

If you want a smaller synchronous run for prompt debugging or a quick pilot, use the standard OpenAI annotator:

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python djinni_openai_ner.py `
  --input .\hf_recruitment_seed_it_1000.jsonl `
  --output .\djinni_ner_annotations_openai.jsonl `
  --id-column id `
  --text-column text `
  --doc-type-column doc_type `
  --language-column source_language `
  --resume `
  --require-english `
  --require-it
```

Recommended defaults for the thesis pipeline:

- model: `gpt-4o-mini`
- prompt version: `openai_stage1_v2`
- structured output schema: exact spans + qualification facts
- chunking: `--max-chars-per-call 12000 --chunk-overlap 200`

If you already prepared a batch work directory with a more expensive model and want to switch only the shards that have not been submitted yet, retarget them in place:

```powershell
python djinni_openai_batch_ner.py retarget-model `
  --work-dir .\artifacts\openai_batch_strict_20k_small `
  --model gpt-4o-mini
```

The OpenAI prompt is designed specifically for Stage 1 of the thesis:

- weak-label IT recruitment entities for later span-based RoBERTa training
- preserve exact span text and offsets
- keep explicit qualification facts for downstream hard-constraint scoring
- separate `TECHNOLOGY` from `SKILL`
- prefer phrase-level `WORK_ACTIVITY` spans
- explicitly exclude English-level spans and methodology leakage such as `QA automation`, `Agile`, `Scrum`, `design patterns`, `OOP`, and `data analysis` from `SKILL`
- keep `INDUSTRY` only for real business sectors, not pseudo-industries like `product company`, `technology`, or `game development`
- avoid generic `PROJECT_TYPE` spans such as `application`, `app`, `project`, `web application`, `mobile app`, or `web development`
- deterministically drop any legacy `ABILITY` spans during cleanup

If you already have a completed OpenAI annotation file and want to tighten noisy labels without paying to re-run the full batch, apply the deterministic cleanup rules:

```powershell
.\.venv\Scripts\python.exe .\clean_openai_annotations.py `
  --input .\djinni_ner_annotations_openai_20k_cleaned_v3.jsonl `
  --output .\djinni_ner_annotations_openai_20k_cleaned_v4.jsonl `
  --overwrite
```

If you still want the Gemini path, the original command remains available:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_KEY"
python djinni_gemini_ner.py `
  --input .\hf_recruitment_seed_1000.jsonl `
  --output .\djinni_ner_annotations.jsonl `
  --id-column id `
  --text-column text `
  --doc-type-column doc_type `
  --language-column source_language `
  --resume `
  --require-english `
  --require-it
```

## Step 2: Train The Custom Span-Based NER Model

Once you have a non-empty annotation file, train the span classifier on top of a RoBERTa encoder:

```powershell
python -m pip install transformers

python train_span_ner.py `
  --annotations .\djinni_ner_annotations_openai_20k_cleaned_v4.jsonl `
  --output-dir .\artifacts\span_ner_roberta_base_openai_cleaned_v4_hardneg `
  --model-name roberta-base `
  --epochs 3 `
  --train-batch-size 2 `
  --eval-batch-size 2 `
  --gradient-accumulation-steps 8 `
  --max-length 256 `
  --max-span-width 8 `
  --device cuda
```

For this repo, the recommended real-training entrypoint is the wrapper below. It is preconfigured for `roberta-base` and the current weak-label dataset:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_roberta_base.ps1
```

That wrapper uses:

- `roberta-base`
- `djinni_ner_annotations_openai_20k_cleaned_v4.jsonl`
- `max-length 256`
- `max-chars-per-example 1200`
- recursive subchunking when tokenization would otherwise truncate and drop spans
- stronger negative-span sampling for weak labels (`10x` multiplier, `160` minimum negatives)
- optional automatic regeneration of `cleaned_v4` from `djinni_ner_annotations_openai_20k_cleaned_v3.jsonl`

For a quick local smoke test on CPU, use a tiny model instead:

```powershell
python train_span_ner.py `
  --annotations .\djinni_ner_annotations.jsonl `
  --output-dir .\artifacts\span_ner_smoke `
  --model-name hf-internal-testing/tiny-random-roberta `
  --epochs 1 `
  --train-batch-size 2 `
  --eval-batch-size 2 `
  --gradient-accumulation-steps 1 `
  --max-length 64 `
  --max-span-width 6 `
  --classifier-hidden-dim 64 `
  --width-embedding-dim 16
```

Training outputs:

- `span_ner.pt`: custom span-classifier checkpoint with label vocabulary and hyperparameters
- tokenizer files for the encoder
- `training_summary.json`: split sizes, feature stats, and dev/test metrics

If you want automatic key rotation, create a text file such as `gemini_keys.txt` with one key per line:

```text
AIza...
AIza...
AIza...
```

Then run:

```powershell
python djinni_gemini_ner.py `
  --input .\hf_recruitment_subset_20k.jsonl `
  --output .\djinni_ner_annotations.jsonl `
  --id-column id `
  --text-column text `
  --doc-type-column doc_type `
  --language-column source_language `
  --api-keys-file .\gemini_keys.txt `
  --resume `
  --require-english `
  --require-it
```

If all provided keys become invalid or run out of quota, the script stops immediately instead of failing the rest of the dataset. Run it again with `--resume` and a new key or updated keys file; already processed records in the output JSONL are skipped automatically.

If you annotate spans manually in Doccano or you have a public JSON/JSONL dataset with span annotations, convert it into the training format with:

```powershell
python convert_span_annotations.py `
  --input .\doccano_export.jsonl `
  --output .\djinni_ner_annotations.jsonl `
  --id-field doc_id `
  --document-type cv `
  --overwrite
```

For public datasets that use different label names, remap them during conversion:

```powershell
python convert_span_annotations.py `
  --input .\public_resume_annotations.jsonl `
  --output .\djinni_ner_annotations.jsonl `
  --document-type cv `
  --label-map-json '{\"Skills\":\"TECHNOLOGY\",\"Skill\":\"SKILL\",\"Role\":\"JOB_ROLE\"}' `
  --overwrite
```

If you do not have manual labels yet, you can bootstrap a weakly-labeled training file directly from the IT seed subset:

```powershell
python bootstrap_weak_annotations.py `
  --input .\hf_recruitment_seed_it_1000.jsonl `
  --cv-output .\doccano_export_cv.jsonl `
  --jd-output .\doccano_export_jd.jsonl `
  --combined-output .\djinni_ner_annotations.jsonl `
  --stats-output .\bootstrap_weak_annotations_stats.json `
  --overwrite
```

This writes:

- `doccano_export_cv.jsonl`
- `doccano_export_jd.jsonl`
- `djinni_ner_annotations.jsonl`
- `bootstrap_weak_annotations_stats.json`

The bootstrap rules are intentionally tightened to reduce noise:

- generic one-word roles like `Lead`, `Support`, `QA`, `Analyst`, `Architect` are filtered out
- mixed multi-role strings are split and the overly broad parent span is pruned
- weak degree rules are tightened to avoid broad `Master` / `Bachelor` false positives
- legacy `ABILITY` spans are no longer emitted by the bootstrap path

If one row contains both CV and JD text columns from some other source, pass both explicitly:

```powershell
python djinni_gemini_ner.py `
  --input .\djinni_pairs.csv `
  --output .\djinni_ner_annotations.jsonl `
  --text-columns "cv:resume_text,jd:job_description" `
  --resume `
  --require-english `
  --require-it
```

If the dataset schema is unusual, set the columns manually:

```powershell
python djinni_gemini_ner.py `
  --input .\records.jsonl `
  --output .\djinni_ner_annotations.jsonl `
  --id-column id `
  --text-column body `
  --doc-type-column source
```

## Output Shape

Each line in the output JSONL looks like:

```json
{
  "record_id": "123:resume_text",
  "document_type": "cv",
  "text": "Senior Data Scientist with 4 years of Python experience...",
  "entities": [
    {
      "label": "JOB_ROLE",
      "text": "Data Scientist",
      "start": 7,
      "end": 21,
      "normalized": "data scientist"
    }
  ],
  "qualification_facts": [
    {
      "fact_type": "EXPERIENCE_YEARS",
      "text": "4 years of Python experience",
      "start": 27,
      "end": 55,
      "operator": "=",
      "value": "4",
      "unit": "years",
      "normalized": "4 years of python experience",
      "is_mandatory": false
    }
  ]
}
```

## Notes

- The subset is taken directly from Hugging Face, specifically the two `lang-uk` recruitment datasets.
- The subset downloader now uses the `datasets` library instead of the dataset viewer HTTP API.
- `HF_TOKEN` is optional but recommended because unauthenticated Hub requests have lower rate limits.
- The seed-prep step is the recommended next stage before LLM labeling: clean first, label a smaller seed set, then scale with a trained NER model.
- The script calls Gemini with structured JSON output and retries transient failures.
- Long documents are chunked and merged back into one annotation record.
- Offsets are validated against the original text and repaired when the returned span text is correct but the offset is slightly off.
