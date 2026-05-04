"""Tests for pycelladmix.scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from pycelladmix.scoring import (
    estimate_cell_adjacency,
    estimate_cell_type_adjacency,
    estimate_contamination_scores,
    estimate_correlation_preservation,
    estimate_gene_prob_per_type,
)


def test_gene_prob_per_type_shapes():
    rng = np.random.default_rng(0)
    n_cells, n_genes = 20, 10
    cm = sparse.csr_matrix(rng.poisson(1.5, size=(n_genes, n_cells)).astype(np.float32))
    types = pd.Series(["A"] * 10 + ["B"] * 10, index=[f"c{i}" for i in range(n_cells)])
    out = estimate_gene_prob_per_type(cm, types)
    assert out.shape == (2, n_genes)
    assert (out.values >= 0).all() and (out.values <= 1).all()
    assert sorted(out.index.tolist()) == ["A", "B"]


def test_cell_adjacency_simple_grid():
    """Cells on a regular grid should produce expected nearest-neighbour adjacencies."""
    rows = []
    cell_id = 0
    for i in range(3):
        for j in range(3):
            cx, cy = i * 10, j * 10
            ct = "A" if (i + j) % 2 == 0 else "B"
            for k in range(8):
                rows.append(
                    {
                        "x": cx + np.cos(k) * 0.5,
                        "y": cy + np.sin(k) * 0.5,
                        "cell_type": ct,
                        "cell": f"cell_{cell_id}",
                    }
                )
            cell_id += 1
    df = pd.DataFrame(rows)
    adj = estimate_cell_adjacency(df, edge_max_mad=10.0)
    assert "cell_s" in adj.columns and "cell_e" in adj.columns
    assert (adj["cell_s"] != adj["cell_e"]).all()
    seen = set(zip(adj["cell_s"], adj["cell_e"], strict=True))
    assert ("cell_0", "cell_1") in seen or ("cell_1", "cell_0") in seen
    typ_mat = estimate_cell_type_adjacency(adj)
    assert set(typ_mat.index) == {"A", "B"}


def test_contamination_scores_recovers_admixture():
    """Synthetic admixture: cells of type A get B-marker counts;
    scoring should give type-A cells higher admixture than type-B cells."""
    rng = np.random.default_rng(0)
    n_genes = 30
    n_markers_a = 15
    n_cells_per_type = 40
    a_cells = [f"a{i}" for i in range(n_cells_per_type)]
    b_cells = [f"b{i}" for i in range(n_cells_per_type)]
    cells = a_cells + b_cells
    annot = pd.Series(["A"] * n_cells_per_type + ["B"] * n_cells_per_type, index=cells)
    gene_names = [f"g{i}" for i in range(n_genes)]
    n_genes_per_block = n_markers_a

    cm_rna = np.zeros((n_genes, len(cells)), dtype=np.float32)
    for j, c in enumerate(cells):
        if annot[c] == "A":
            cm_rna[:n_genes_per_block, j] = rng.poisson(5, size=n_genes_per_block)
        else:
            cm_rna[n_genes_per_block:, j] = rng.poisson(5, size=n_genes - n_genes_per_block)

    cm_spatial = cm_rna.copy()
    for j, c in enumerate(cells):
        if annot[c] == "A":
            cm_spatial[n_genes_per_block:, j] = rng.poisson(3, size=n_genes - n_genes_per_block)

    K = pd.DataFrame([[10.0, 5.0], [5.0, 10.0]], index=["A", "B"], columns=["A", "B"])
    res = estimate_contamination_scores(
        cm_rna=sparse.csr_matrix(cm_rna),
        cm_spatial=sparse.csr_matrix(cm_spatial),
        annot_rna=annot,
        annot_spatial=annot,
        cell_type_adj_mat=K,
        gene_names=gene_names,
        p_c=0.25,
        signal_thres=0.1,
        min_expr_frac=0.05,
        adjust=False,
    )
    a_mean = res.cell_admixture_fractions.loc[a_cells].mean()
    b_mean = res.cell_admixture_fractions.loc[b_cells].mean()
    assert a_mean > b_mean
    assert a_mean > 0.05


def test_correlation_preservation_identity():
    rng = np.random.default_rng(0)
    cm = sparse.csr_matrix(rng.poisson(2.0, size=(10, 80)).astype(np.float32))
    cors = estimate_correlation_preservation(cm, cm, gene_names=[f"g{i}" for i in range(10)])
    assert (cors.dropna() > 0.99).all()
