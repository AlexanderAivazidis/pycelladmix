"""KNN aggregation of transcripts and rank-k NMF factorisation.

Mirrors R ``run_knn_nmf`` / ``knn_count_matrix`` / ``knn_adjacency_matrix``.

Default backend is **deterministic JAX NMF** (multiplicative-update or projected
gradient), with the 30 random restarts vectorised via ``jax.vmap`` so they run
in parallel on GPU. An optional ``mode="bayesian"`` switches to a numpyro
Poisson-NMF model with Gamma priors fit by SVI — this is *not* a faithful
reproduction of the R algorithm, but a probabilistic alternative that gives
uncertainty over factor loadings.
"""

from __future__ import annotations
