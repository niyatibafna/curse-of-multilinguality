# Training multilingual BERT models

## Overview

We train small BERT-style masked language models on nested groups of 2 to 100
languages from MADLAD-400. The models share the same architecture but vary in
language coverage, training compute, and the distribution of training data
across languages. Each configuration is trained with five random seeds.

We then evaluate the models on the paper's multilinguality conditions:
monolingual structure, cross-lingual alignment, and non-degeneracy. See the
[metrics documentation](../metrics/README.md) for definitions.

## Training pipeline overview

The same pipeline applies to every configuration:

1. Create fixed, nested language groups and any resource-proportional token
   targets.
2. Sample tokenizer-training text for each language group.
3. Train a BERT WordPiece tokenizer for each group.
4. Sample masked-language-modeling data according to the selected compute and
   data-sampling configuration.
5. Train a 4-layer BERT encoder with masked language modeling for one epoch.

## Configurations

The experiments cross two compute settings with two data-sampling settings:

- **Fixed compute:** every language group receives 500M total training tokens.
- **Increasing compute:** each language's allocation remains fixed as languages
  are added, so the total number of training tokens increases with coverage.
- **Uniform sampling:** tokens are divided equally among languages.
- **Realistic sampling:** tokens are allocated in proportion to the amount of
  MADLAD training data for each language, measured using clean-text bytes.

All configurations use the same nested language groups of
`2,5,10,20,30,40,50,75,100` languages and five training seeds.

## Evaluation overview

We evaluate each checkpoint on multiway-parallel text from BOUQuET, FLORES+,
and WMT24++. The evaluation measures how monolingual structure and
cross-lingual alignment change as the number of training languages increases,
while also checking that the embedding space remains non-degenerate.

Two evaluation views are supported:

- `eval-all` evaluates all available training languages for each model.
- `eval-subset-n10` holds the evaluation languages fixed to the 10-language
  group as training coverage increases.

## Running the training pipeline

Replace angle-bracketed values with paths appropriate for your environment.
Repeat group-specific commands for each language-group size. Repeat data
sampling and model training for seeds 1 through 5.

1. Create the nested language groups.

   ```bash
   python -m src.training.make_language_plan \
     --output_path <language-plan.json> \
     --include_languages en \
     --num_languages 100 \
     --sizes 2,5,10,20,30,40,50,75,100 \
     --order_by_madlad_resource \
     --resource_split clean_bytes
   ```

   Realistic sampling additionally requires per-language token targets. For
   fixed compute, use `--total_tokens 500000000`. For increasing compute, use
   `--max_tokens_per_language 500000000`. Both settings apply a 10K-token
   minimum per language.

   ```bash
   python -m src.training.make_resource_token_targets \
     --language_plan_path <language-plan.json> \
     --output_path <token-targets.json> \
     --floor_tokens_per_language 10000 \
     <compute-budget-argument>
   ```

2. Sample tokenizer-training text for one language group.

   Choose the sampling arguments for the configuration: fixed-compute uniform
   uses `--strategy fixed --fixed_total_tokens 500000000`; increasing-compute
   uniform uses `--strategy additive --additive_tokens_per_language 50000000`.
   For realistic sampling, use the corresponding strategy together with
   `--target_tokens_path <token-targets.json>`.

   ```bash
   python -m src.training.sample_madlad \
     --language_plan_path <language-plan.json> \
     --subset n<size> \
     --output_path <tokenizer-corpus.jsonl> \
     --madlad_cache_root <madlad-cache-directory> \
     --tokenizer_path <seed-tokenizer-directory> \
     --sample_shuffle_buffer_size 10000 \
     --shuffle_output True \
     <sampling-arguments>
   ```

   The seed tokenizer is used only to count tokens during sampling. The sampled
   text is used to train a new tokenizer in the next step.

3. Train the language group's tokenizer.

   ```bash
   python -m src.training.train_tokenizer \
     --input_files <tokenizer-corpus.txt> \
     --output_dir <tokenizer-directory> \
     --vocab_size 50000
   ```

4. Sample masked-language-modeling corpus.

   Fixed-compute uniform sampling requires
   `--strategy fixed --fixed_total_tokens 500000000`; increasing-compute
   uniform sampling requires
   `--strategy additive --additive_tokens_per_language 50000000`. For either
   realistic-sampling configuration, use the required fixed / additive strategy and
   `--target_tokens_path <token-targets.json>` instead of an equal allocation.

   ```bash
   python -m src.training.sample_madlad \
     --language_plan_path <language-plan.json> \
     --subset n<size> \
     --output_path <training-corpus.jsonl> \
     --tokenizer_path <tokenizer-directory> \
     --madlad_cache_root <madlad-cache-directory> \
     --sample_shuffle_buffer_size 10000 \
     --shuffle_output True \
     --seed <seed> \
     <sampling-arguments>
   ```

5. Train model.

   ```bash
   python -m src.training.train_mlm \
     --train_file <training-corpus.jsonl> \
     --tokenizer_path <tokenizer-directory> \
     --output_dir <checkpoint-directory> \
     --num_train_epochs 1 \
     --per_device_train_batch_size 64 \
     --max_seq_length 128 \
     --seed <seed>
   ```

## Running the evaluation pipeline

1. Create an evaluation manifest. `<checkpoint-paths>` is a comma-separated
   list of checkpoint groups. Choose either `eval-all` or
   `eval-subset-n10` for the evaluation view.

   The manifest is a JSON file containing the checkpoint, dataset, evaluation
   languages, and metric for every evaluation to run.

   ```bash
   python -m src.training.make_eval_manifest \
     --language_plan_path <language-plan.json> \
     --output_path <evaluation-manifest.json> \
     --checkpoint_root <checkpoint-root-directory> \
     --strategies <checkpoint-paths> \
     --sizes 2,5,10,20,30,40,50,75,100 \
     --eval_stream <evaluation-view>
   ```

   The base manifest covers embedding-space metrics. Masked-language-modeling
   loss uses a derived manifest. Use `train_subset` with `eval-all` or
   `source_subset` with a fixed evaluation group.

   ```bash
   python -m src.training.make_mlm_loss_manifest \
     --base_manifest_path <evaluation-manifest.json> \
     --output_path <mlm-loss-manifest.json> \
     --sizes 2,5,10,20,30,40,50,75,100 \
     --eval_language_mode <train_subset-or-source_subset>
   ```

2. Cache checkpoint embeddings. Repeat for every unique warmup index in the
   manifest.

   ```bash
   python -m src.training.run_embedding_warmup_entry \
     --manifest_path <evaluation-manifest.json> \
     --index <warmup-index> \
     --output_dir <warmup-output-directory> \
     --batch_size 16 \
     --pooling cls \
     --device cuda
   ```

3. Evaluate every manifest entry.

   ```bash
   python -m src.training.run_eval_manifest_entry \
     --manifest_path <evaluation-manifest.json> \
     --index <entry-index> \
     --output_dir <evaluation-output-directory> \
     --batch_size 32 \
     --pooling cls \
     --device cpu
   ```
