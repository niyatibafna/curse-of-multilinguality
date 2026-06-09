## Training experiments with adding more languages

The idea is to train many multilingual models with the same token budget / additive token budget, but of varying number of languages (1, 5, 10, 25, 50, 75, 100). 

We will then compute our metrics on these models, characterising the curse of multilinguality for embedding spaces.

### Data
We will get data from MADLAD400. 

### Languages
We choose 100 languages from the dataset, including English. We'll then start with a random subset, and then add more languages to it to create incrementally bigger language subsets.

### Data sampling strategies
#### Fixed token budget
We'll have a fixed token budget of 20M, and split it equally across all languages in the group

#### Additive
We will sample 1M tokens per language and fix this.
The dataset size increases as we add more languages.

### Model and training
We'll do a 4-layer BERT-style Transformer with size hidden size 256 and 4 heads.
We'll train a tokenizer with vocabulary 50k on 100M tokens across 100 languages.
We always use the same tokenizer regardless of language group or data sampling strategy.

For each language subset, we train this model from scratch for 3 epochs.

### Optimization and other hyperparameters
We use default learning rate and other hyperparameters. 

### Checkpoints and storage
Use an appropriate place in $DATADIR as per conventions for data and models.


## Todos
### Model stream
Status: initial scripts are ready.

- `train_mlm.py` trains a 4-layer BERT masked-LM from scratch.
- Architecture defaults: hidden size 256, 4 layers, 4 heads, intermediate size
  1024, WordPiece vocab from the shared tokenizer.
- Checkpoints are saved in Hugging Face format and can be evaluated with
  `src/scripts/run_metrics.py --model_type mbert --models <checkpoint_path>`.

### Data stream
Status: initial streaming sampler is ready.

- `make_language_plan.py` writes one frozen 100-language pool with `en`
  included by default, prioritizing MADLAD languages whose codes match BOUQuET,
  FLORES+, and WMT24++ languages before filling remaining slots.
- The generated plan stores per-language evaluation coverage metadata and
  nested subsets for sizes 1, 5, 10, 25, 50, 75, and 100.
- `sample_madlad.py` streams one language at a time and stops after the target
  token budget. If a language has less text than the target under the selected
  MADLAD split, it records the language as `underfilled` in the manifest and
  continues by default.
- Corpora and manifests are written under
  `$DATADIR/projects/curse-of-multilinguality/training/corpora/`.

### Tokenizer stream
Status: initial tokenizer trainer is ready.

- `train_tokenizer.py` trains a 50k BERT WordPiece tokenizer.
- Default input:
  `$DATADIR/projects/curse-of-multilinguality/training/corpora/tokenizer/n100.txt`.
- Default output:
  `$DATADIR/projects/curse-of-multilinguality/training/tokenizers/bert_wordpiece_50000`.

### Training stream
Status: launch scripts are ready.

Launch prep:

```bash
mkdir -p /weka/home/nbafna1/projects/curse-of-multilinguality/slurm_logs/{training_make_language_plan,training_sample_madlad,training_train_tokenizer,training_train_mlm,training_make_manifest}
export DATADIR=/path/to/data
export CONDA_ENV=genspace
```

Suggested sequence:

```bash
sbatch slurm_scripts/training_make_language_plan.sbatch

# First run only array task 0 to create the tokenizer corpus.
sbatch --array=0 slurm_scripts/training_sample_madlad.sbatch
sbatch slurm_scripts/training_train_tokenizer.sbatch

# Then run array tasks 1-14 to create fixed/additive corpora using tokenizer token counts.
sbatch --array=1-14 slurm_scripts/training_sample_madlad.sbatch
sbatch slurm_scripts/training_make_manifest.sbatch
sbatch slurm_scripts/training_train_mlm.sbatch
```

Environment overrides:

- `MADLAD_DATASET`: defaults to `allenai/madlad-400`.
- `MADLAD_SPLIT`: defaults to `clean`.
- `TEXT_FIELD`: defaults to `text`.
- `CONFIG_PER_LANGUAGE`: defaults to `True`.
- `TRUST_REMOTE_CODE`: defaults to `True` for the MADLAD400 loader.
- `TOKENIZER_PATH`: defaults to the shared tokenizer path.
- `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `MAX_SEQ_LENGTH`: training job controls.

### Evaluation stream
Status: no repo changes needed.

Example:

```bash
python src/scripts/run_metrics.py \
  --models "$DATADIR/projects/curse-of-multilinguality/training/checkpoints/fixed/n25" \
  --model_type mbert \
  --datasets bouquet \
  --metrics anisotropy,comness \
  --pooling cls \
  --output_dir outputs
```

### Documentation
Use this README for instructions to agents, and status recording.

### Open launch checks

- `genspace` currently has `datasets`, `transformers`, `tokenizers`, and
  `torch`; use `CONDA_ENV` to override if needed. Do not install packages with
  `pip` without explicit permission.
- MADLAD400's active Hugging Face loader is `allenai/madlad-400` with
  `trust_remote_code=True`; the scripts default to one config per language,
  split `clean`, and text field `text`.
- English (`en`) is included in the language pool by default. Use
  `--include_languages` and `--exclude_languages` for explicit overrides.
- Confirm whether `MAX_SEQ_LENGTH=128` is acceptable for first launch, or use
  512 for more BERT-like pretraining.
