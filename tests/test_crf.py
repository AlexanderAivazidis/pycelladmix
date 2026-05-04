"""Tests for pycelladmix.crf."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

from pycelladmix.crf import (
    build_molecule_graph,
    loopy_bp_max_product,
    run_crf_all,
)
from pycelladmix.nmf import run_knn_nmf


def test_build_molecule_graph_disconnected():
    """Edges from different cells must not connect."""
    rng = np.random.default_rng(0)
    rows = []
    for cell_idx in range(2):
        for mol in range(20):
            rows.append(
                {
                    "x": rng.normal() + cell_idx * 1000,
                    "y": rng.normal(),
                    "z": rng.normal(),
                    "gene": f"g{mol % 4}",
                    "cell": f"c{cell_idx}",
                    "mol_id": f"m_{cell_idx}_{mol}",
                }
            )
    df = pd.DataFrame(rows)
    senders, receivers, reverse_idx, mol_ids = build_molecule_graph(df, h=5)
    cell_of_idx = np.repeat([0, 1], 20)
    assert (cell_of_idx[senders] == cell_of_idx[receivers]).all()
    np.testing.assert_array_equal(senders, receivers[reverse_idx])
    np.testing.assert_array_equal(receivers, senders[reverse_idx])


def test_loopy_bp_smooths_to_uniform_label():
    """A connected graph with strong agreement should collapse to one label."""
    n = 30
    k = 3
    rng = np.random.default_rng(0)
    log_node_pot = jnp.asarray(rng.normal(0, 0.1, size=(n, k)))
    t = 0.01
    A = jnp.full((k, k), t).at[jnp.diag_indices(k)].set(1.0 - (k - 1) * t)
    log_edge_pot = jnp.log(A)
    senders = []
    receivers = []
    for i in range(n):
        for j in (i - 1, i + 1):
            if 0 <= j < n:
                senders.append(min(i, j))
                receivers.append(max(i, j))
    pairs = sorted(set(zip(senders, receivers, strict=True)))
    s = np.empty(2 * len(pairs), dtype=np.int32)
    r = np.empty(2 * len(pairs), dtype=np.int32)
    for e, (a, b) in enumerate(pairs):
        s[2 * e] = a
        r[2 * e] = b
        s[2 * e + 1] = b
        r[2 * e + 1] = a
    rev = np.arange(2 * len(pairs), dtype=np.int32) ^ 1
    labels = loopy_bp_max_product(
        log_node_pot,
        log_edge_pot,
        jnp.asarray(s),
        jnp.asarray(r),
        jnp.asarray(rev),
        n_iter=50,
    )
    assert len(np.unique(np.asarray(labels))) == 1


def test_run_crf_all_end_to_end(synthetic_transcripts: pd.DataFrame):
    """Full NMF + CRF pipeline on synthetic data; output covers all retained molecules."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=3, n_iter=100, weighted=True)
    crf_res = run_crf_all(synthetic_transcripts, res, num_nn=10, n_iter=50)
    assert isinstance(crf_res, pd.DataFrame)
    assert set(crf_res.columns) == {"mol_id", "factor"}
    assert crf_res["factor"].between(1, 3).all()
    assert len(crf_res) == len(res.mol_ids)
    factor_counts = crf_res["factor"].value_counts()
    assert (factor_counts > 0).all(), "All factors should have at least one assigned molecule"


def test_run_crf_all_groups_by_celltype(synthetic_transcripts: pd.DataFrame):
    """CRF assignments should correlate strongly with the underlying cell type label."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=5, n_iter=200, weighted=True)
    crf_res = run_crf_all(synthetic_transcripts, res, num_nn=10, n_iter=100)
    df = synthetic_transcripts.set_index("mol_id").loc[crf_res["mol_id"]]
    crf_res = crf_res.copy()
    crf_res["celltype"] = df["celltype"].to_numpy()
    cont = pd.crosstab(crf_res["celltype"], crf_res["factor"])
    purity = cont.max(axis=1).sum() / cont.sum().sum()
    assert purity > 0.7, f"Per-cell-type CRF purity {purity:.2f} is too low"
