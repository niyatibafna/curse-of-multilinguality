# curse-of-multilinguality

We want to study real embedding spaces with respect to our arguments about the curse of multilinguality.

## Project Layout

- `src/data/`: dataset adapters for aligned multilingual text.
- `src/models/`: embedding model wrappers and registry.
- `src/metrics/`: metric implementations and metric math notes.
- `src/scripts/run_metrics.py`: main experiment runner.
- `misc/results_vis/`: plotting scripts.
- `tests/`: focused tests for datasets, caches, and metric math.


## Metrics

The main metric families are:

- `anisotropy`: mean off-diagonal cosine similarity across embeddings.
- `comness`: compares effective dimensionality of language variation to concept
  variation.
- `concept_language_principal_angle_overlap`: measures overlap between retained
  concept and language displacement subspaces using principal angles.
- dimensionality/growth metrics: track how concept or language subspace
  effective dimensionality changes as languages or concepts are added.

See [src/metrics/README.md](src/metrics/README.md) for definitions, formulas,
and interpretation caveats.

## Setup

Set a data/cache root:

```bash
export DATADIR=/path/to/data
```

Project caches are stored under:

```bash
$DATADIR/projects/curse-of-multilinguality/
```

`OPENAI_KEY` is required only for `openai-large`. Hugging Face auth may be
needed for gated models or datasets such as Llama and FLORES+.

## Quickstart

Run a small local smoke test:

```bash
python src/scripts/run_metrics.py \
  --models minilm \
  --datasets bouquet \
  --metrics anisotropy,comness \
  --dataset_languages eng_Latn,fra_Latn \
  --max_texts 100 \
  --batch_size 32 \
  --output_dir outputs
```

Outputs:

- dataset cache: `$DATADIR/projects/curse-of-multilinguality/multiparallel/*.jsonl`
- embedding cache: `$DATADIR/projects/curse-of-multilinguality/embeddings/*.npz`
- metric JSON: `outputs/<model>/<dataset>/<metric>.json`

## Runner Arguments

Common arguments:

- `--models`: comma-separated model registry keys.
- `--datasets`: comma-separated dataset registry keys.
- `--metrics`: comma-separated metric keys.
- `--dataset_languages`: languages/configs to download or format.
- `--eval_languages`: languages to include in metrics.
- `--max_texts`: use a prefix of cached embeddings for metric computation.
- `--layer`: hidden-state layer for transformer wrappers; default `-1`.
- `--pooling`: override model pooling. Defaults are model-specific: `cls` for
  `mbert`, `last_token` otherwise.
- `--device`: e.g. `cuda`, `cuda:0`, or `cpu`.
- `--return_details True`: include diagnostic fields in metric JSON.
- `--normalize False`: disable L2 normalization before metrics.

## Registered Models

Registry keys in `src/models/registry.py`:

- `llama`: `meta-llama/Llama-3.1-8B-Instruct`
- `mistral`: `mistralai/Ministral-8B-Instruct-2410`
- `mbert`: `bert-base-multilingual-uncased`
- `openai-large`: `text-embedding-3-large`
- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`
- `bge-small`: `BAAI/bge-small-en-v1.5`
- `bge-base`: `BAAI/bge-base-en-v1.5`
- `e5-base`: `intfloat/e5-base-v2`
- `e5-large`: `intfloat/e5-large-v2`
- `nomic`: `nomic-ai/nomic-embed-text-v1.5`

Use `--model_type` to run an arbitrary model path with an existing wrapper, for
example:

```bash
python src/scripts/run_metrics.py \
  --models meta-llama/... \
  --model_type llama \
  --datasets bouquet \
  --metrics anisotropy
```

## Registered Datasets

Registry keys in `src/data/registry.py`:

- `bouquet`, `facebook/bouquet`
- `floresplus`, `flores+`
- `wmt24++`, `wmt24pp`

All adapters convert raw rows into:

```python
{"id": "...", "data": {"language": "text"}, "metadata": {...}}
```

## Plotting

Plotting scripts read from `outputs/` and write under
`misc/results_vis/plots/`.

Examples:

```bash
python misc/results_vis/plot_comness.py
python misc/results_vis/plot_principal_angle_overlap.py
python misc/results_vis/plot_principal_angle_overlap_sweep.py
```


## Training experiments with adding more languages

The idea is to train many multilingual models with the same token budget / additive token budget, but of varying number of languages (1, 5, 10, 25, 50, 75, 100). 

We will start with monolingual BERT. 


### Data
We will get data from MADLAD400. 

### Languages
We choose 100 languages from the dataset. We'll then start with a random subset, and then add more languages to it to create incrementally bigger language subsets.

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

