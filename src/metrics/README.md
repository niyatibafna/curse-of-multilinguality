# Metrics

This package contains diagnostics for multilingual embedding spaces. Inputs use
the common layout:

$$
X[\ell] \in \mathbb{R}^{C \times D}
$$

Rows are aligned concepts. So $X[\ell, c] \in \mathbb{R}^D$ means "the
embedding of concept $c$ in language $\ell$."

Many of these metrics are concerned with dimensionality of the "concept subspace" or the "language
subspace", and the scaling of this dimensionality.

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
unnormalized effective dimensions internally because its final score is already
a ratio.

When `normalize=True`, each embedding vector is L2-normalized before computing
the metric. This removes vector magnitude and keeps directional geometry.

### Making this efficient

Many metrics need the effective dimension of a matrix of pairwise difference
vectors. In the grouped case, each group contributes only its own within-group
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

So don't need to store the entire matrix, we can just loop over all difference vectors and collect the Gram matrix. 

But there's actually a closed form solution to find the contribution of each group.
The key identity is:

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
