# Metrics

This package contains diagnostics for multilingual embedding spaces. Inputs use
the common layout:

```text
X[language] -> array[num_concepts, embedding_dim]
```

Rows are aligned concepts. So `X[l, c]` means "the embedding of concept `c` in
language `l`."

Many of these metrics are concerned with dimensionality of the "concept subspace" or the "language
subspace", and the scaling of this dimensionality.

The basic way we isolate a subspace is by taking pairwise differences. For
example:

- concept geometry: keep language fixed and compare two concepts
  `X[l, c_i] - X[l, c_j]`
- language geometry: keep concept fixed and compare two languages
  `X[l_i, c] - X[l_j, c]`

The span of those difference vectors is the subspace we want to measure. The
metrics below differ mostly in which differences they collect and which axis
they scale over.

## Metrics

### `anisotropy`

Mean off-diagonal cosine similarity over all embeddings:

```text
mean_{a != b} <z_a, z_b>
```

where `z_a` are optionally normalized embedding vectors from all
language/concept pairs. High anisotropy means many embeddings share a common
direction.

### `comness` [WIP]
COMness compares language variation to concept variation:

```text
COM(X) = d_lang / (d_lang + d_concept)
```

where

```text
d_lang    = d_eff({X[l_i, c] - X[l_j, c] : c, i < j})
d_concept = d_eff({X[l, c_i] - X[l, c_j] : l, i < j})
```

Low COMness means language identity occupies few effective dimensions relative
to semantic concept variation. High COMness means language variation is
geometrically complex relative to the concept space.

### `individual_concept_dimensionality`

For each language independently, measure the effective dimension of concept
variation inside that language:

```text
d_l = d_eff({X[l, c_i] - X[l, c_j] : i < j})
```

The output contains a language -> dimension map and a descending sorted list for
plotting. This asks whether each language uses a similar-dimensional concept
space.

### `concept_space_dim_growth_by_language`

Measure how concept-space dimensionality changes as more languages are added.
Given a fixed random language order `l_1, ..., l_L`, for prefixes `L_k` compute:

```text
d(k) = d_eff({X[l, c_i] - X[l, c_j] : l in L_k, i < j})
```

The output records `(num_languages, effective_dim)` and the language order. The
desired behavior is that concept-space dimensionality should not grow much as
more languages are added.

### `language_space_dim_growth_by_language`

Measure language-space dimensionality as more languages are added. Given a fixed
random language order and prefixes `L_k`, compute same-concept cross-language
differences:

```text
d(k) = d_eff({X[l_i, c] - X[l_j, c] : c, l_i,l_j in L_k, i < j})
```

The output records `(num_languages, effective_dim)` and the language order. This
asks whether language geometry can be encoded in a small number of dimensions as
the number of languages increases.

### `language_space_growth_by_concepts`

Measure language-space dimensionality as more concepts are included. For concept
prefixes `C_k`, compute:

```text
d(k) = d_eff({X[l_i, c] - X[l_j, c] : c in C_k, i < j})
```

The desired behavior is that the language subspace should use a mostly fixed
set of dimensions across concepts, rather than growing strongly as more
concepts are observed.

## Effective Dimensionality

After constructing a subspace through difference vectors, we measure its
effective dimension. Given a centered matrix `M` with singular values `s_i`, the
default method is the stable-rank style participation ratio:

```text
d_eff(M) = (sum_i s_i)^2 / sum_i s_i^2
```

This behaves like a soft dimension count. If all singular values are equal, it
returns the number of active directions. If the spectrum is dominated by a few
directions, it returns a smaller value.

Other supported methods are:

```text
entropy:   exp(-sum_i p_i log p_i),  p_i = s_i / sum_j s_j
threshold: |{i : s_i > threshold}|
```

Dimensionality metrics usually report `d_eff / embedding_dim` by default, so the
number is a fraction of the model's embedding dimension. COMness uses
unnormalized effective dimensions internally because its final score is already
a ratio.

When `normalize=True`, each embedding vector is L2-normalized before computing
the metric. This removes vector magnitude and keeps directional geometry.

### Pairwise Displacement Trick

Many metrics need the effective dimension of a matrix of pairwise difference
vectors:

```text
M = {x_i - x_j : i < j}
```

Materializing `M` can be huge. Instead, we accumulate enough moments to recover
the singular values of the centered displacement matrix. For each group
`G = [x_0, ..., x_{n-1}]`, the code accumulates:

```text
count = n choose 2
total = sum_{i < j} (x_i - x_j)
gram  = sum_{i < j} (x_i - x_j)(x_i - x_j)^T
```

The key identity is:

```text
sum_{i < j} (x_i - x_j)(x_i - x_j)^T
  = n sum_i x_i x_i^T - (sum_i x_i)(sum_i x_i)^T
```

After all groups are accumulated, centering happens in Gram space:

```text
M_c^T M_c = M^T M - count * mean(M) mean(M)^T
```

Then:

```text
s_i = sqrt(eig_i(M_c^T M_c))
```

So we get the singular values needed for `d_eff` without storing the full
`num_displacements x embedding_dim` matrix. Memory stays at roughly
`embedding_dim x embedding_dim`.
