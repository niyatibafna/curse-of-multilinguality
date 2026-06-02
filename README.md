# curse-of-multilinguality

Metrics for measuring how much multilingual embedding spaces spend capacity on
language variation instead of concept variation.

## Repo Structure

- `src/data/`: parallel dataset loaders and formatting to `{id, data, metadata}` rows.
- `src/models/`: embedding model wrappers and model registry.
- `src/metrics/`: metric implementations (`anisotropy`, `comness`).
- `src/scripts/run_metrics.py`: main experiment entrypoint.
- `tests/`: lightweight dataset/cache behavior tests.
- `misc/results_vis/`: plotting helpers.

## Metrics

- `anisotropy`: mean off-diagonal cosine similarity over all embeddings; high means vectors share a common direction.
- `COMness`: `d_lang / (d_lang + d_concept)`, where `d_lang` is the effective rank of same-concept cross-language displacements and `d_concept` is the effective rank of same-language concept displacements.
  - Intuition: high COMness means language identity occupies many effective dimensions relative to semantic concept variation.
  - Efficiency: This would require a matrix of (combinatorial) pairwise difference vectors of which we then find singular values. This is too large to keep in memory; instead, the code accumulates centered Gram moments and gets singular values from eigenvalues of `M_c.T @ M_c`.

## Models

Registry keys in `src/models/registry.py`:

- `llama`: `meta-llama/Llama-3.1-8B-Instruct`
- `mbert`: `bert-base-multilingual-uncased`
- `mistral`: `mistralai/Ministral-8B-Instruct-2410`
- `openai-large`: `text-embedding-3-large`
- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`
- `bge-small`: `BAAI/bge-small-en-v1.5`
- `bge-base`: `BAAI/bge-base-en-v1.5`
- `e5-base`: `intfloat/e5-base-v2`
- `e5-large`: `intfloat/e5-large-v2`
- `nomic`: `nomic-ai/nomic-embed-text-v1.5`

Use `--model_type` to run an arbitrary model path with an existing wrapper, e.g.
`--models meta-llama/... --model_type llama`.

## Datasets

Registry keys in `src/data/registry.py`:

- `bouquet`, `facebook/bouquet`
- `floresplus`, `flores+`
- `wmt24++`, `wmt24pp`

All are converted to multiparallel rows before embedding. Use
`--dataset_languages` to limit downloaded/config languages and `--eval_languages`
to limit languages used in metrics.

## Quickstart

Set data/cache root for storing caches of multiparallel data and computed embeddings:

```bash
export DATADIR=/path/to/data
```

Project data goes under:

```bash
$DATADIR/projects/curse-of-multilinguality/
```

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

Useful args:

- `--models`: comma-separated registry keys 
- `--datasets`: comma-separated dataset keys.
- `--metrics`: `anisotropy`, `comness`, or both.
- `--layer`: hidden-state layer for transformer wrappers; default `-1`.
- `--pooling`: `last_token`, `mean`, or `tokens`; default `last_token`.
- `--device`: e.g. `cuda`, `cuda:0`, `cpu`.
- `--return_details True`: include diagnostic fields in metric JSON.
- `--normalize False`: disable L2 normalization before metrics.
- `--random_baseline_trials`: random baseline samples for scaling metrics; set
  `0` to disable. Default `1`.
- `--random_baseline_seed`: seed for scaling-metric random baselines.

Env variables:

- `DATADIR`: required.
- `OPENAI_KEY`: required only for `openai-large`.
- Hugging Face auth may be needed for gated datasets/models such as FLORES+ or Llama.
