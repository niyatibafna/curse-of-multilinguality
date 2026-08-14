# Metrics for multilingual embedding spaces

## Overview

We measure embedding-space structure using sentence-aligned multilingual text.
For each underlying text, or **concept**, the input contains one embedding per
language. This lets us study whether a model preserves monolingual semantic
structure, aligns translations across languages, and avoids representational
collapse as language coverage increases.

## Main metrics

- **MS-NNO (monolingual-structure nearest-neighbor overlap):** compares each
  concept's nearest neighbors in a target language with its nearest neighbors
  in a strong reference embedding space. The paper reports overlap at $k=20$;
  higher is better. Implemented in
  [`multilinguality_conditions.py`](multilinguality_conditions.py).
- **MS-MLM (monolingual-structure masked-language-modeling loss):** measures
  normalized masked-language-modeling loss on monolingual evaluation text,
  averaged across languages. Lower is better. Implemented in
  [`extrinsic.py`](extrinsic.py).
- **CLA-WV (cross-lingual alignment, weak view):** checks whether a source
  embedding retrieves its correct translation ahead of all other concepts in
  the target language. Higher is better. Implemented in
  [`multilinguality_conditions.py`](multilinguality_conditions.py).
- **CLA-SV (cross-lingual alignment, strong view):** checks whether a source
  embedding retrieves its correct translation ahead of different concepts
  from every language in the multilingual space. Higher is better. Implemented
  in [`multilinguality_conditions.py`](multilinguality_conditions.py).
- **Non-degeneracy:** summarizes the minimum, maximum, mean, and standard
  deviation of pairwise cosine similarities. Increasing similarity across all
  pairs may indicate that the space is collapsing. Implemented in
  [`noncollapse.py`](noncollapse.py).

## Investigating dimensionality

We also questions about how language and concepts are represented in the spaces and how that scales with language coverage.

### Constructing subspaces

Let $x_{c,\ell}$ be the embedding of concept $c$ in language $\ell$.

The **language subspace** is spanned by differences between translations of the
same concept:

$$
\mathcal{U}_{\mathrm{lang}}
= \operatorname{span}\{x_{c,\ell}-x_{c,\ell'} : \ell \ne \ell'\}.
$$

The **concept subspace** is constructed analogously from different concepts in
the same language:

$$
\mathcal{U}_{\mathrm{concept}}
= \operatorname{span}\{x_{c,\ell}-x_{c',\ell} : c \ne c'\}.
$$

The construction and scaling metrics are implemented in
[`language_subspace_dimensionality.py`](language_subspace_dimensionality.py),
[`concept_space_dimensionality.py`](concept_space_dimensionality.py), and
[`utils.py`](utils.py).

### Effective dimensionality

Let $M$ contain the centered difference vectors for a subspace, with singular
values $s_i$. We use the participation ratio

$$
d_{\mathrm{eff}}(M)
= \frac{\left(\sum_i s_i\right)^2}{\sum_i s_i^2}.
$$

This is a soft dimension count: it approaches the number of active directions
when the singular values are evenly distributed and becomes smaller when a few
directions dominate. We generally report $d_{\mathrm{eff}}/D$, where $D$ is
the ambient embedding dimension.

The available analyses measure:

- language-subspace dimension as languages are added;
- language-subspace dimension as concepts are added;
- concept-subspace dimension as languages or concepts are added; and
- concept-space dimension separately for each language.

See [`language_subspace_dimensionality.py`](language_subspace_dimensionality.py)
and [`concept_space_dimensionality.py`](concept_space_dimensionality.py).

### Concept-language overlap

To measure whether language and concept information reuse the same directions,
we retain orthonormal bases $V_c$ and $V_\ell$ that explain a chosen fraction
of each subspace's energy. If

$$
\sigma_i
= \operatorname{svd}(V_c^\top V_\ell)_i
= \cos\theta_i,
$$

then the random-adjusted overlap is

$$
O_{\mathrm{CL}}
= \frac{
\frac{1}{m}\sum_{i=1}^{m}\sigma_i^2
- \frac{\max(d_c,d_\ell)}{D}
}{
1-\frac{\max(d_c,d_\ell)}{D}
},
\qquad m=\min(d_c,d_\ell).
$$

Higher values indicate greater overlap; zero indicates overlap at the expected
level for random subspaces with the same dimensions. The paper uses bases that
capture 20% of subspace energy. Implemented in
[`interaction_between_concept_and_language.py`](interaction_between_concept_and_language.py).

## Run a metric

Use the main runner with a registered model, dataset, and metric name:

```bash
python -m src.scripts.run_metrics \
  --models <model> \
  --datasets <dataset> \
  --metrics nearest_neighbor_overlap_against_monolingual_20 \
  --dataset_languages <comma-separated-languages> \
  --eval_languages <comma-separated-languages> \
  --output_dir <output-directory>
```

Multiple metric names can be supplied as a comma-separated list. Dataset and
embedding caches require `DATADIR` to be set.
