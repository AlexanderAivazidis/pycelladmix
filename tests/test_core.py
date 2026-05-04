"""End-to-end tests for pycelladmix.core."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from pycelladmix import run_celladmix
from pycelladmix.core import CellAdmixResult


def test_run_celladmix_no_scoring(synthetic_transcripts: pd.DataFrame):
    """Without an scRNA-seq reference, NMF + CRF should run end-to-end."""
    res = run_celladmix(
        synthetic_transcripts,
        k=3,
        h_nmf=10,
        h_crf=10,
        nmf_runs=3,
        nmf_iter=100,
        crf_iter=50,
    )
    assert isinstance(res, CellAdmixResult)
    assert res.nmf.W.shape[1] == 3
    assert res.crf["factor"].between(1, 3).all()
    assert res.contamination is None


def test_run_celladmix_with_scoring(synthetic_transcripts: pd.DataFrame):
    """With a synthetic RNA reference, the full pipeline returns admixture scores."""
    rng = np.random.default_rng(0)
    types = sorted(synthetic_transcripts["celltype"].unique())
    n_genes = synthetic_transcripts["gene"].nunique()
    n_rna_per_type = 30
    rna_cells: list[str] = []
    rna_types: list[str] = []
    rna_counts = np.zeros((n_genes, len(types) * n_rna_per_type), dtype=np.float32)
    gene_names = sorted(synthetic_transcripts["gene"].unique())
    type_markers = {ct: [g for g in gene_names if g.startswith(f"{ct}_")] for ct in types}
    for ti, ct in enumerate(types):
        marker_idx = [gene_names.index(g) for g in type_markers[ct]]
        for j in range(n_rna_per_type):
            cell_idx = ti * n_rna_per_type + j
            rna_cells.append(f"rna_{ct}_{j}")
            rna_types.append(ct)
            rna_counts[marker_idx, cell_idx] = rng.poisson(5, size=len(marker_idx))
    annot_rna = pd.Series(rna_types, index=rna_cells)

    res = run_celladmix(
        synthetic_transcripts,
        k=3,
        h_nmf=10,
        h_crf=10,
        nmf_runs=3,
        nmf_iter=100,
        crf_iter=50,
        cm_rna=sparse.csr_matrix(rna_counts),
        annot_rna=annot_rna,
        rna_gene_names=gene_names,
        signal_thres=0.1,
        min_expr_frac=0.05,
        adjust_scores=False,
    )
    assert res.contamination is not None
    assert res.cell_type_adjacency is not None
    fracs = res.contamination.cell_admixture_fractions
    assert (fracs >= 0).all()
    assert (fracs <= 1).all()
    assert len(fracs) > 0
