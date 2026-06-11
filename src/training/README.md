# Training Pipeline Runbook

This directory contains the helper scripts for training small BERT-style masked
language models on MADLAD400 language groups, then evaluating embedding-space
geometry metrics. This README is written for agents that need to reproduce or
extend a training lane.

## Experiment Shape

- Data source: `allenai/madlad-400`, split `clean`, one dataset config per
  language.
- Language plan: `<max_languages>` MADLAD languages including `en`, with
  overlap against BOUQuET, FLORES+, and WMT24++ preserved as much as possible.
- Language groups: a lane-specific `--sizes` list, ending at or below
  `<max_languages>`.
- Model: 4-layer BERT-style MLM, hidden size 256, 4 attention heads,
  intermediate size 1024.
- Tokenizer: BERT WordPiece, lane-specific vocab size, trained separately for
  each language group.
- Training: one pass through the prepared data (`--num_train_epochs 1`).
- Training data is shuffled together across languages during corpus prep before
  MLM batching.

Use a new versioned lane for new full reruns: `v3`, `v4`, etc. Keep every lane
in separate paths so old runs remain comparable.

## Artifact Layout

All large data/model artifacts live under:

```bash
$DATADIR/projects/curse-of-multilinguality/training/
```

Common lane paths:

```text
language_plan_<version>_resource_ordered.json
eval_manifest_<version>.json
madlad_cache_v4tok_750m/clean/<language>.jsonl
madlad_cache_v4tok_750m/clean/<language>.manifest.json
corpora_<version>/tokenizer/n<size>.{jsonl,txt,manifest.json}
corpora_<version>/<corpus_group>/n<size>.{jsonl,txt,manifest.json}
corpora_<version>/token_deficit_report.json
tokenizers/bert_wordpiece_<vocab_size>_<version>/n<size>/
checkpoints_<version>/<corpus_group>/n<size>/
```

Metric JSON outputs stay in the repo:

```text
outputs/training_scaling_<version>/<corpus_group>/n<size>/<dataset>/<metric>.json
```

Plots stay in:

```text
misc/results_vis/plots/scaling_<version>/
misc/results_vis/plots/training_loss/
```

SLURM logs stay in:

```text
slurm_logs/%x/%j.out
slurm_logs/%x/%A_%a.out
```

## Full Pipeline

The complete pipeline is:

1. Make a frozen language plan.
2. Build or verify the reusable MADLAD raw-text cache.
3. Sample tokenizer-training text for each language group.
4. Train one tokenizer per language group.
5. Sample fixed/additive MLM corpora with the matching group tokenizer.
6. Write a token deficit report from corpus manifests.
7. Make the evaluation manifest for all model/dataset/metric jobs.
8. Launch MLM training.
9. Warm embedding caches on GPU.
10. Run metrics on CPU.
11. Plot training loss and metric scaling results.

Always ask the user for explicit permission before submitting `sbatch` jobs.

## Step 1: Language Plan

Script:

```bash
python -m src.training.make_language_plan \
  --output_path "$DATADIR/projects/curse-of-multilinguality/training/language_plan_<version>_resource_ordered.json" \
  --include_languages en \
  --num_languages <max_languages> \
  --sizes <comma_separated_group_sizes> \
  --order_by_madlad_resource \
  --resource_split clean
```

Notes:

- Eval overlap is prioritized before filling the rest of the pool.
- MADLAD resource ordering uses Hugging Face shard counts as a proxy, not exact
  token counts under the tokenizer.
- The final language in the max-size group is not guaranteed to be the lowest
  actual-token language.

SLURM examples:

```bash
sbatch --parsable slurm_scripts/training_make_language_plan_v4.sbatch
```

## Step 2: MADLAD Raw-Text Cache

To avoid repeated slow Hugging Face streaming, build a raw-text cache once for
the planned language pool. The current cache target is a clear oversample:
750M tokens per language counted with the v4 tokenizer. Later tokenizers recount
these cached rows when selecting tokenizer-training or MLM subsets.

Script:

```bash
python -m src.training.build_madlad_cache \
  --language_plan_path "$DATADIR/projects/curse-of-multilinguality/training/language_plan_<version>_resource_ordered.json" \
  --output_root "$DATADIR/projects/curse-of-multilinguality/training/madlad_cache_v4tok_750m" \
  --language_index <language_index> \
  --dataset_name allenai/madlad-400 \
  --split clean \
  --text_field text \
  --config_per_language True \
  --trust_remote_code True \
  --tokenizer_path "$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_50000_v4" \
  --max_tokens_per_language 750000000 \
  --write_index False
```

Array helper:

```bash
sbatch --parsable slurm_scripts/training_build_madlad_cache_v4tok_750m.sbatch
```

Cache layout:

```text
madlad_cache_v4tok_750m/clean/<language>.jsonl
madlad_cache_v4tok_750m/clean/<language>.manifest.json
```

Notes:

- The cache stores raw text rows, not tokenized data.
- The cache token count is only for deciding how much raw text to store.
- Sampling still counts with the tokenizer passed to `sample_madlad.py`.
- If a cached language is missing or underfilled for a future sample,
  `sample_madlad.py` falls back to Hugging Face when
  `--cache_fallback_to_hf True`.
- Exact duplicate avoidance during fallback is guaranteed for prefix sampling
  (`--sample_shuffle_buffer_size 0`). With buffer-shuffled cache sampling,
  fallback should be rare because the cache is intentionally oversized.

## Step 3: Tokenizer Corpus Sampling

Script:

```bash
python -m src.training.sample_madlad \
  --language_plan_path "$DATADIR/projects/curse-of-multilinguality/training/language_plan_<version>_resource_ordered.json" \
  --strategy tokenizer \
  --subset n<size> \
  --output_path "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/tokenizer/n<size>.jsonl" \
  --dataset_name allenai/madlad-400 \
  --split clean \
  --text_field text \
  --config_per_language True \
  --trust_remote_code True \
  --madlad_cache_root "$DATADIR/projects/curse-of-multilinguality/training/madlad_cache_v4tok_750m" \
  --cache_fallback_to_hf True \
  --tokenizer_path "$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_<vocab_size>_<seed_version>" \
  --additive_tokens_per_language <tokens_per_language> \
  --allow_underfilled True \
  --sample_shuffle_buffer_size <buffer_size> \
  --shuffle_output True
```

For a first lane with no previous tokenizer, use an older tokenizer only as the
token-counting seed, or use no tokenizer if approximate row-based sampling is
acceptable. For later lanes, using the previous lane tokenizer as the seed
avoids circularity and then trains a fresh tokenizer for the new lane.

Choose `--additive_tokens_per_language` as a lane hyperparameter. It does not
need to equal any previous lane's value.

Use `--sample_shuffle_buffer_size 0` to preserve deterministic prefix sampling
within each language. Use a positive buffer size to shuffle each language stream
before taking rows up to the token target; different `--seed` values will then
select different text per language. This is a streaming buffer shuffle, not a
perfect global shuffle unless the buffer covers the full language stream.

For per-group tokenizers, run this once for every language-group size. The
`training_sample_tokenizers_per_group.sbatch` template is an array job over
`SIZES_CSV`.

## Step 4: Per-Group Tokenizer Training

Script:

```bash
python -m src.training.train_tokenizer \
  --input_files "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/tokenizer/n<size>.txt" \
  --output_dir "$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_<vocab_size>_<version>/n<size>" \
  --vocab_size <vocab_size>
```

Each trained tokenizer directory must contain `tokenizer.json`, `vocab.txt`,
and the Hugging Face tokenizer config files before corpus sampling starts for
that group. The `training_train_tokenizers_per_group.sbatch` template is an
array job over the same `SIZES_CSV`.

## Step 5: MLM Corpus Prep

Fixed-budget example:

```bash
python -m src.training.sample_madlad \
  --language_plan_path "$DATADIR/projects/curse-of-multilinguality/training/language_plan_<version>_resource_ordered.json" \
  --strategy fixed \
  --subset n<size> \
  --output_path "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/fixed_<budget>m/n<size>.jsonl" \
  --dataset_name allenai/madlad-400 \
  --split clean \
  --text_field text \
  --config_per_language True \
  --trust_remote_code True \
  --madlad_cache_root "$DATADIR/projects/curse-of-multilinguality/training/madlad_cache_v4tok_750m" \
  --cache_fallback_to_hf True \
  --tokenizer_path "$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_<vocab_size>_<version>/n<size>" \
  --fixed_total_tokens <budget_tokens> \
  --allow_underfilled True \
  --sample_shuffle_buffer_size <buffer_size> \
  --shuffle_output True
```

Additive-budget example:

```bash
python -m src.training.sample_madlad \
  --language_plan_path "$DATADIR/projects/curse-of-multilinguality/training/language_plan_<version>_resource_ordered.json" \
  --strategy additive \
  --subset n<size> \
  --output_path "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/additive_<tokens_per_lang>m/n<size>.jsonl" \
  --tokenizer_path "$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_<vocab_size>_<version>/n<size>" \
  --additive_tokens_per_language <tokens_per_language> \
  --allow_underfilled True \
  --sample_shuffle_buffer_size <buffer_size> \
  --shuffle_output True
```

Important behavior:

- The sampler streams one language at a time until that language reaches its
  target or runs out of data.
- With `--sample_shuffle_buffer_size 0`, it takes the same prefix of rows per
  language for the same tokenizer and budget.
- With `--sample_shuffle_buffer_size > 0`, it buffer-shuffles each language
  stream with a stable seed derived from `--seed` and the language code, so
  different seeds select different rows per language.
- If `--madlad_cache_root` is set, the sampler reads cached raw rows first and
  only streams Hugging Face if the cache is missing or too small.
- It writes `.jsonl`, `.txt`, and `.manifest.json`.
- If `--shuffle_output True`, it reloads the JSONL and rewrites both JSONL/TXT
  in shuffled order. This is the step that mixes languages before training.
- Languages with insufficient data are recorded as `underfilled` in the
  manifest and the job continues when `--allow_underfilled True`.
- With per-group tokenizers, token counts and deficits are measured under that
  group's tokenizer, not a max-language shared tokenizer.

## Step 6: Token Deficit Report

Run after all corpus manifests exist:

```bash
python -m src.training.report_token_deficits \
  --manifest_glob "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/<corpus_group>/*.manifest.json" \
  --output_path "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/token_deficit_report.json"
```

Report this to the user before interpreting results. For fixed-budget runs,
`<budget_tokens>` is a lane hyperparameter and deficit is:

```text
sum(target_tokens_per_language) - sum(actual_tokens_per_language)
```

Overshoots of a few thousand tokens are normal because sampling stops after the
row that crosses the target.

## Step 7: Eval Manifest

Run after the language plan is ready. It can run before training finishes, but
the checkpoint paths it writes must match the planned training output paths.

```bash
python -m src.training.make_eval_manifest \
  --language_plan_path "$DATADIR/projects/curse-of-multilinguality/training/language_plan_<version>_resource_ordered.json" \
  --output_path "$DATADIR/projects/curse-of-multilinguality/training/eval_manifest_<version>.json" \
  --checkpoint_root "$DATADIR/projects/curse-of-multilinguality/training/checkpoints_<version>" \
  --strategies <corpus_group> \
  --sizes <comma_separated_group_sizes>
```

For fixed lanes, choose a corpus group name that includes the budget:

```text
fixed_<budget>m
```

The manifest expands every model size over `bouquet`, `floresplus`, `wmt24pp`
and all registered metrics that have enough eval languages.

## Step 8: MLM Training

Training entrypoint:

```bash
python -m src.training.train_mlm \
  --train_file "$DATADIR/projects/curse-of-multilinguality/training/corpora_<version>/<corpus_group>/n<size>.jsonl" \
  --tokenizer_path "$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_<vocab_size>_<version>/n<size>" \
  --output_dir "$DATADIR/projects/curse-of-multilinguality/training/checkpoints_<version>/<corpus_group>/n<size>" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 128 \
  --preprocessing_num_workers 8 \
  --logging_steps 1000
```

Training details:

- `train_mlm.py` tokenizes the JSONL corpus, groups text into 128-token chunks,
  shuffles the grouped chunk dataset, then trains.
- Loss logs are written to `training_loss.jsonl` inside each checkpoint.
- Checkpoints are Hugging Face compatible and evaluated with `model_type=mbert`.

## Step 9: Embedding Warmup

Warmup runs one unique checkpoint/dataset/eval-language group per array task and
populates the shared embedding cache before CPU metric jobs.

```bash
python -m src.training.run_embedding_warmup_entry \
  --manifest_path "$DATADIR/projects/curse-of-multilinguality/training/eval_manifest_<version>.json" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --output_dir "outputs/training_scaling_<version>/_embedding_warmup" \
  --batch_size 16 \
  --pooling cls \
  --device cuda
```

Warmup array size is:

```text
number_of_model_sizes x number_of_eval_datasets
```

## Step 10: Metric Evaluation

Metric jobs read the eval manifest and write JSON to `outputs/`.

```bash
python -m src.training.run_eval_manifest_entry \
  --manifest_path "$DATADIR/projects/curse-of-multilinguality/training/eval_manifest_<version>.json" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --output_dir "outputs/training_scaling_<version>" \
  --batch_size 32 \
  --pooling cls \
  --device cpu \
  --random_baseline_trials 1 \
  --random_baseline_seed 0 \
  --alignment_batch_size 64
```

The metric array size should match `len(eval_manifest["entries"])`.

## Step 11: Plotting

Training loss:

```bash
conda run -n genspace python misc/results_vis/plot_training_loss.py \
  --loss_root "$DATADIR/projects/curse-of-multilinguality/training/checkpoints_<version>" \
  --output_dir misc/results_vis/plots/training_loss/<version>
```

Metric scaling:

```bash
conda run -n genspace python misc/results_vis/plot_training_scaling.py \
  --input_dir outputs/training_scaling_<version> \
  --output_dir misc/results_vis/plots/scaling_<version> \
  --formats png pdf
```

The scaling plotter writes:

```text
misc/results_vis/plots/scaling_<version>/training_scaling_summary.csv
misc/results_vis/plots/scaling_<version>/<corpus_group>/summary.csv
misc/results_vis/plots/scaling_<version>/<corpus_group>/overview.{png,pdf}
misc/results_vis/plots/scaling_<version>/<corpus_group>/<metric>/*.{png,pdf}
```

## SLURM Chain Template

Use `sbatch --parsable` and explicit `afterok` dependencies. Ask the user for
permission before submitting.

```bash
plan_job=$(sbatch --parsable slurm_scripts/training_make_language_plan_<version>.sbatch)
cache_job=$(sbatch --parsable --dependency=afterok:${plan_job} slurm_scripts/training_build_madlad_cache_v4tok_750m.sbatch)
tok_data_job=$(sbatch --parsable --dependency=afterok:${cache_job} slurm_scripts/training_sample_tokenizers_per_group.sbatch)
tok_job=$(sbatch --parsable --dependency=afterok:${tok_data_job} slurm_scripts/training_train_tokenizers_per_group.sbatch)
data_job=$(sbatch --parsable --dependency=afterok:${tok_job} slurm_scripts/training_sample_madlad_per_group_tokenizers.sbatch)
deficit_job=$(sbatch --parsable --dependency=afterok:${data_job} slurm_scripts/training_report_deficits_<version>.sbatch)
manifest_job=$(sbatch --parsable --dependency=afterok:${deficit_job} slurm_scripts/training_make_eval_manifest_<version>.sbatch)
train_job=$(sbatch --parsable --dependency=afterok:${manifest_job} slurm_scripts/training_train_mlm_per_group_tokenizers.sbatch)
warm_job=$(sbatch --parsable --dependency=afterok:${train_job} slurm_scripts/training_warm_embeddings_<version>.sbatch)
eval_job=$(sbatch --parsable --dependency=afterok:${warm_job} slurm_scripts/training_eval_metrics_<version>.sbatch)
```

Array conventions:

- CPU arrays: no concurrency cap.
- GPU arrays: use `%10`.
- The generic per-group-tokenizer templates have `#SBATCH --array=0-10`; edit
  the upper bound if `SIZES_CSV` has a different number of groups.
- Logs: `/weka/home/nbafna1/projects/curse-of-multilinguality/slurm_logs/%x/%A_%a.out`
  for arrays, `%x/%j.out` for non-array jobs.

## Monitoring Checklist

Queue:

```bash
squeue -j <job_ids>
```

Common log scan:

```bash
rg -n "Traceback|RuntimeError|ERROR|FAILED|Exception|Killed|No such file|CUDA out of memory|### Finished" \
  slurm_logs/training_*_<version>/*.out
```

Pre-training checks:

- Every `n<size>` tokenizer directory exists and contains `tokenizer.json` and
  `vocab.txt`.
- All expected corpus manifests exist.
- `token_deficit_report.json` exists and has been reported to the user.
- Corpus JSONL/TXT shuffle temp files are gone.

Training checks:

- CUDA is visible in GPU logs.
- `training_loss.jsonl` appears under checkpoint directories.
- Each array task prints `### Finished`.

Evaluation checks:

- Warmup array finishes before eval starts.
- Metric output count matches the eval manifest entries.
- No metric JSONs are written outside `outputs/training_scaling_<version>/`.

## Historical Versioned Lanes

These are completed or queued examples, not defaults for future lanes.

- v2:
  - Fixed 500M: sizes `2,5,10,20,25,30,40,50`.
  - Additive 10M/language: sizes `2,5,10,25,50`.
  - Outputs: `outputs/training_scaling_v2`.
- v3:
  - Fresh tokenizer over all 100 languages.
  - Fixed 500M: sizes `2,5,10,15,20,25,30,40,50,75,100`.
  - Corpus group: `fixed_500m`.
- v4:
  - Fresh tokenizer over all 100 languages.
  - Fixed 300M: sizes `2,5,10,15,20,25,30,40,50,75,100`.
  - Corpus group: `fixed_300m`.
- v5:
  - Five seed replicates with shuffled data sampling.
  - Fixed 500M: sizes `2,5,10,20,30,40,50,75,100`.
  - Per-seed and per-group tokenizers:
    `tokenizers/bert_wordpiece_50000_v5/seed<k>/n<size>`.
  - Clean nested artifacts:
    `corpora_v5/seed<k>/fixed_500m/n<size>`,
    `checkpoints_v5/seed<k>/fixed_500m/n<size>`, and
    `outputs/training_scaling_v5/seed<k>/fixed_500m/n<size>`.
  - Uses the v4 100-language tokenizer only as the seed tokenizer for counting
    tokenizer-corpus sampling tokens.
  - Scaling plots aggregate means across seeds and show standard deviation.

## Environment Notes

- Default conda env in SLURM scripts: `genspace`.
- Set `DATADIR`; current cluster path is `/weka/scratch/tlippin1/nbafna1`.
- Do not install Python packages with `pip` without explicit user permission.
- Hugging Face access/network issues should be surfaced clearly; MADLAD loading
  uses `trust_remote_code=True`.
