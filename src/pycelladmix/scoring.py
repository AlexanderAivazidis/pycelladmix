"""Per-cell Bayesian admixture scoring.

Mirrors R ``estimate_contamination_scores`` and supporting helpers
(``estimate_cell_adjacency``, ``estimate_cell_type_adjacency``,
``estimate_gene_prob_per_type``, ``estimate_correlation_preservation``).

Inputs: an scRNA-seq reference, the spatial counts, and a cell-type
adjacency matrix derived from Delaunay triangulation of cell centroids.

Closed-form posterior P(contaminated | gene, cell-type) — implemented in JAX
for vectorisation across cells and genes. Cell-type adjacency comes from
:mod:`scipy.spatial.Delaunay`.
"""

from __future__ import annotations
