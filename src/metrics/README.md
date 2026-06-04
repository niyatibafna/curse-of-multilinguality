# Metrics

This package contains diagnostics for multilingual embedding spaces. Inputs use
the common layout:

$$
X[\ell] \in \mathbb{R}^{C \times D}
$$

Rows are aligned concepts. So $X[\ell, c] \in \mathbb{R}^D$ means "the
embedding of concept $c$ in language $\ell$."

Many of these metrics are concerned with dimensionality of the "concept
subspace" or the "language subspace", and the scaling of this dimensionality.

The basic way we isolate a subspace is by taking pairwise differences. For
example:

- concept geometry: keep language fixed and compare two concepts
  $X[\ell, c_i] - X[\ell, c_j]$
- language geometry: keep concept fixed and compare two languages
  $X[\ell_i, c] - X[\ell_j, c]$

The span of those difference vectors is the subspace we want to measure. The
metrics below differ mostly in which differences they collect and which axis
they scale over.

## Metrics

### `anisotropy`

Mean off-diagonal cosine similarity over all embeddings:

$$
\frac{1}{N(N-1)} \sum_{a \ne b} \langle z_a, z_b \rangle
$$

where `z_a` are optionally normalized embedding vectors from all
language/concept pairs. High anisotropy means many embeddings share a common
direction.

### `comness` [WIP]
COMness compares language variation to concept variation:

$$
\mathrm{COM}(X) =
\frac{d_{\mathrm{lang}}}{d_{\mathrm{lang}} + d_{\mathrm{concept}}}
$$

where

$$
d_{\mathrm{lang}} =
d_{\mathrm{eff}}\bigl(
\{X[\ell_i, c] - X[\ell_j, c] : c,\ i < j\}
\bigr)
$$

$$
d_{\mathrm{concept}} =
d_{\mathrm{eff}}\bigl(
\{X[\ell, c_i] - X[\ell, c_j] : \ell,\ i < j\}
\bigr)
$$

Low COMness means language identity occupies few effective dimensions relative
to semantic concept variation. High COMness means language variation is
geometrically complex relative to the concept space.

With `return_details=True`, COMness also reports a random-baseline-normalized
variant. The raw numerator and denominator use different group shapes:

- language variation uses `num_concepts` groups of size `num_languages`
- concept variation uses `num_languages` groups of size `num_concepts`

The normalized variant samples random embeddings from the full pool with each
same group shape, then compares each observed effective dimension to its matched
random baseline:

$$
\tilde d_{\mathrm{lang}} =
\frac{d_{\mathrm{lang}}}{d^{\mathrm{rand}}_{\mathrm{lang}}}
$$

$$
\tilde d_{\mathrm{concept}} =
\frac{d_{\mathrm{concept}}}{d^{\mathrm{rand}}_{\mathrm{concept}}}
$$

and reports:

$$
\mathrm{COM}_{\mathrm{normalized}}(X) =
\frac{\tilde d_{\mathrm{lang}}}
{\tilde d_{\mathrm{lang}} + \tilde d_{\mathrm{concept}}}
$$

This keeps the original COMness score available while adding a version that
controls for effective-rank growth caused by different numbers of sampled
points/displacements.

### `concept_language_principal_angle_overlap`

Measure overlap between the concept and language displacement subspaces. First
construct the same concept-displacement and language-displacement spaces used by
COMness. For each space, keep the smallest set of right singular directions
whose squared singular values explain `subspace_energy_threshold` energy
(default `0.9`). The registered sweep aliases set this threshold explicitly:

- `concept_language_principal_angle_overlap_20`: `0.2`
- `concept_language_principal_angle_overlap_50`: `0.5`
- `concept_language_principal_angle_overlap_90`: `0.9`

Then compute principal angles between the retained orthonormal bases:

$$
\sigma_i =
\mathrm{svd}\left(V_{\mathrm{concept}}^\top V_{\mathrm{language}}\right)_i
= \cos(\theta_i)
$$

The main raw score is:

$$
\mathrm{overlap}_{\mathrm{raw}} =
\frac{1}{m} \sum_{i=1}^m \sigma_i^2,
\qquad
m = \min(k_{\mathrm{concept}}, k_{\mathrm{language}})
$$

where $k_{\mathrm{concept}}$ and $k_{\mathrm{language}}$ are the retained
subspace dimensions. Low values mean the retained concept and language
subspaces are close to orthogonal; high values mean they reuse directions.

Raw overlap must be interpreted with the retained dimensions. For two random
subspaces of dimensions $k_{\mathrm{concept}}$ and $k_{\mathrm{language}}$ in
$\mathbb{R}^D$,

$$
\mathbb{E}\left[\mathrm{tr}(P_{\mathrm{concept}}P_{\mathrm{language}})\right]
=
\frac{k_{\mathrm{concept}} k_{\mathrm{language}}}{D}
$$

so the expected mean squared principal-angle cosine is:

$$
\mathrm{overlap}_{\mathrm{rand}} =
\frac{
\mathbb{E}\left[\mathrm{tr}(P_{\mathrm{concept}}P_{\mathrm{language}})\right]
}{
\min(k_{\mathrm{concept}}, k_{\mathrm{language}})
}
=
\frac{\max(k_{\mathrm{concept}}, k_{\mathrm{language}})}{D}
$$

The metric therefore also reports an excess-over-random score:

$$
\mathrm{overlap}_{\mathrm{adjusted}} =
\frac{
\mathrm{overlap}_{\mathrm{raw}} - \mathrm{overlap}_{\mathrm{rand}}
}{
1 - \mathrm{overlap}_{\mathrm{rand}}
}
$$

This is `0` when overlap matches random subspaces with the same dimensions and
`1` for perfect overlap. Values below `0` mean less overlap than that matched
random baseline.

The output includes `principal_angle_cosines`, `principal_angles_degrees`,
`mean_squared_cosine`, `random_expected_mean_squared_cosine`,
`adjusted_mean_squared_cosine`, `max_cosine`, retained subspace dimensions, and
retained energy.

### `individual_concept_dimensionality`

For each language independently, measure the effective dimension of concept
variation inside that language:

$$
d_\ell =
d_{\mathrm{eff}}\bigl(
\{X[\ell, c_i] - X[\ell, c_j] : i < j\}
\bigr)
$$

The output contains a language -> dimension map and a descending sorted list for
plotting. This asks whether each language uses a similar-dimensional concept
space.

### `concept_space_dim_growth_by_language`

Measure how concept-space dimensionality changes as more languages are added.
Given a fixed random language order $\ell_1, \ldots, \ell_L$, for prefixes
$L_k$ compute:

$$
d(k) =
d_{\mathrm{eff}}\bigl(
\{X[\ell, c_i] - X[\ell, c_j] : \ell \in L_k,\ i < j\}
\bigr)
$$

The output records `(num_languages, effective_dim)` and the language order. The
desired behavior is that concept-space dimensionality should not grow much as
more languages are added.

### `concept_space_dim_growth_by_concept`

Measure how concept-space dimensionality changes as more aligned concepts are
included. For concept prefixes $C_k$, compute same-language concept
differences:

$$
d(k) =
d_{\mathrm{eff}}\bigl(
\{X[\ell, c_i] - X[\ell, c_j] : \ell,\ c_i,c_j \in C_k,\ i < j\}
\bigr)
$$

The output records `(num_concepts, effective_dim)`. This asks how quickly the
concept subspace fills out as more semantic items are observed.

### `language_space_dim_growth_by_language`

Measure language-space dimensionality as more languages are added. Given a fixed
random language order and prefixes $L_k$, compute same-concept cross-language
differences:

$$
d(k) =
d_{\mathrm{eff}}\bigl(
\{
X[\ell_i, c] - X[\ell_j, c] :
c,\ \ell_i,\ell_j \in L_k,\ i < j
\}
\bigr)
$$

The output records `(num_languages, effective_dim)` and the language order. This
asks whether language geometry can be encoded in a small number of dimensions as
the number of languages increases.

### `language_space_growth_by_concepts`

Measure language-space dimensionality as more concepts are included. For concept
prefixes $C_k$, compute:

$$
d(k) =
d_{\mathrm{eff}}\bigl(
\{
X[\ell_i, c] - X[\ell_j, c] :
c \in C_k,\ i < j
\}
\bigr)
$$

The desired behavior is that the language subspace should use a mostly fixed
set of dimensions across concepts, rather than growing strongly as more
concepts are observed.

## Effective Dimensionality

After constructing a subspace through difference vectors, we measure its
effective dimension. Given a centered matrix $M$ with singular values $s_i$, the
default method is the stable-rank style participation ratio:

$$
d_{\mathrm{eff}}(M) =
\frac{\bigl(\sum_i s_i\bigr)^2}{\sum_i s_i^2}
$$

This behaves like a soft dimension count. If all singular values are equal, it
returns the number of active directions. If the spectrum is dominated by a few
directions, it returns a smaller value.

Other supported methods are:

Entropy effective rank:

$$
d_{\mathrm{entropy}}(M) =
\exp\bigl(-\sum_i p_i \log p_i\bigr),
\qquad
p_i = \frac{s_i}{\sum_j s_j}
$$

Threshold rank:

$$
d_{\mathrm{threshold}}(M) =
|\{i : s_i > \tau\}|
$$

Dimensionality metrics usually report $d_{\mathrm{eff}} / D$ by default, so the
number is a fraction of the model's embedding dimension. COMness uses
unnormalized effective dimensions internally because its final scores are
ratios.

When `normalize=True`, each embedding vector is L2-normalized before computing
the metric. This removes vector magnitude and keeps directional geometry.

### Random baseline normalization

The scaling metrics and COMness details also report random baselines. This is
meant to separate real subspace growth from the generic effect of giving the
effective-rank calculation more points.

For each observed prefix, the metric samples random embeddings from the full
language/concept pool and arranges them into groups with the same sizes as the
observed groups. It then computes effective dimensionality with the exact same
pairwise-displacement calculation.

For example, if `language_space_dim_growth_by_language` is evaluating a prefix
of 5 languages over 500 concepts, the observed calculation uses 500 groups of
size 5, one group per concept. The random baseline uses the same grouping shape:
500 random groups of size 5.

Scaling rows keep the original `effective_dim` field and add:

- `random_effective_dim_mean`
- `random_effective_dim_std`
- `random_baseline_trials`
- `effective_dim_ratio = effective_dim / random_effective_dim_mean`

Use `random_baseline_trials` to control the number of random samples. Set it to
0 to disable the baseline fields. Use `random_baseline_seed` for reproducible
sampling.

### Making this efficient

Many metrics need the effective dimension of a matrix of pairwise difference
vectors, computed between points belonging to a single group. A group could be
all language variants of a single concept, or all concepts in a single
language. The full matrix would contain displacement vectors from all groups,
and we need its singular values.
We note that each group contributes only its own within-group
differences:

$$
M^{(g)} = \{x_i^{(g)} - x_j^{(g)} : i < j\}
$$

The full displacement matrix is the row-wise stack of those group matrices:

$$
M =
\begin{bmatrix}
M^{(1)} \\
M^{(2)} \\
\vdots \\
M^{(G)}
\end{bmatrix}
$$

We want the singular values of the centered displacement matrix $M_c$, because
those singular values are what define $d_{\mathrm{eff}}$. To get them, we use
the fact that the eigenvalues of

$$
M_c^\top M_c
$$

are the squared singular values of $M_c$:

$$
s_i = \sqrt{\lambda_i(M_c^\top M_c)}
$$

So the goal is to construct $M_c^\top M_c$ directly. This is the useful Gram
matrix because it is only `embedding_dim x embedding_dim`. The alternative,
$M_c M_c^\top$, would be `num_displacements x num_displacements`, which is the
large object we are trying to avoid.

First ignore centering and construct $M^\top M$. Since $M$ is a vertical stack,

$$
\begin{aligned}
M^\top M
&=
\begin{bmatrix}
(M^{(1)})^\top & (M^{(2)})^\top & \cdots & (M^{(G)})^\top
\end{bmatrix}
\begin{bmatrix}
M^{(1)} \\
M^{(2)} \\
\vdots \\
M^{(G)}
\end{bmatrix}
\end{aligned}
$$

so

$$M^\top M = \sum_g (M^{(g)})^\top M^{(g)}$$

There are no cross-group terms here. (Cross terms would appear in $MM^\top$, not
in $M^\top M$. Both have the same non-zero eigenvalues but we'll use this for convenience.) Since we only need $M^\top M$, each group can be processed
independently and added to a running total.

For one group $G = [x_0, \ldots, x_{n-1}]$,

$$
(M^{(g)})^\top M^{(g)} =
\sum_{i < j} (x_i - x_j)(x_i - x_j)^\top
$$

So we do not need to store the entire displacement matrix; we can collect its
Gram matrix.

There is also a closed-form contribution for each group:

$$
\begin{aligned}
\sum_{i < j} (x_i - x_j)(x_i - x_j)^\top
&= n \sum_i x_i x_i^\top \\
&\quad - \bigl(\sum_i x_i\bigr)\bigl(\sum_i x_i\bigr)^\top
\end{aligned}
$$

This lets us add one group’s contribution to $M^\top M$ using only the group’s
sum and second moment, without constructing the pairwise displacement rows.

The code also accumulates the number and sum of all displacement rows:

$$
\mathrm{count} = \sum_g {n_g \choose 2}
$$

$$
\mathrm{total} =
\sum_g \sum_{i < j} (x_i^{(g)} - x_j^{(g)})
$$

These are needed to center the full stacked matrix. If

$$
\mu_M = \frac{\mathrm{total}}{\mathrm{count}},
$$

then centering in Gram space gives:

$$
M_c^\top M_c =
M^\top M -
\mathrm{count} \cdot \mu_M \mu_M^\top
$$

Finally, recover the singular values from the eigenvalues:

$$
s_i = \sqrt{\lambda_i(M_c^\top M_c)}
$$

So we get the singular values needed for $d_{\mathrm{eff}}$ without storing the
full `num_displacements x embedding_dim` matrix. Memory stays at roughly
`embedding_dim x embedding_dim`.
