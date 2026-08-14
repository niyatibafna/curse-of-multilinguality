# There Is No Theoretical Curse of Multilinguality for Embedding Spaces

This repository contains the code for the project *There Is No Theoretical
Curse of Multilinguality for Embedding Spaces*. (TODO arxiv link)

We study whether multilingual embedding spaces can preserve high-quality
monolingual semantics and cross-lingual alignment as the number of languages
encoded increases. We will measure the empirical curse of multilinguality for embedding structure under controlled variations in training compute, data sampling, and evaluation languages.

## Repository structure

- [`src/metrics/`](src/metrics/): implements the paper's intrinsic embedding-space
  metrics. See its [README](src/metrics/README.md) for definitions and formulas.
- [`src/scripts/run_metrics.py`](src/scripts/run_metrics.py): runs metric
  evaluation and writes results to `outputs/`.
- [`src/data/`](src/data/): loads and formats multilingual evaluation datasets.
- [`src/models/`](src/models/): embedding-model wrappers and registry.
- [`src/training/`](src/training/): trains small BERT-style masked language
  models on increasingly large language sets varying compute, data, and target evaluation group settings. See its
  [README](src/training/README.md) for the full pipeline. Runs evaluation on above metrics.


## Main metrics

- **MS-NNO:** overlap between monolingual nearest-neighbor structure and a
  strong reference embedding space.
- **MS-MLM:** normalized masked-language-modeling loss, used as an indirect
  measure of monolingual quality.
- **CLA-WV:** cross-lingual alignment measured by retrieval of the correct
  translation among target-language concepts.
- **Non-degeneracy:** pairwise cosine-similarity statistics used to check that
  the embedding space does not collapse as language coverage grows.

## Trends of empirical curse of multilinguality for embedding space structure

![Main BOUQuET results](misc/paper_figures/figures/main_results_bouquet_by_evaluation.png)


If you use this code, please cite
```
TODO
```