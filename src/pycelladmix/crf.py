"""Per-molecule factor assignment via a chain conditional random field.

Mirrors R ``run_crf_all`` / ``run_crf``. Inference uses loopy belief propagation
over the molecule-KNN graph; node potentials come from NMF gene loadings,
edge potentials from a single tunable label-agreement parameter.

Implementation in JAX: per-cell belief propagation is JIT-compiled and
``vmap``-ed over cells, so cells are processed in parallel on GPU rather than
serially as in the R reference.
"""

from __future__ import annotations
