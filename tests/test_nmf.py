"""Tests for pycelladmix.nmf."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pycelladmix.nmf import KNNNMFResult, fit_nmf, get_knn_counts_all, run_knn_nmf


def test_get_knn_counts_all_shape(synthetic_transcripts: pd.DataFrame):
    X, mol_ids = get_knn_counts_all(synthetic_transcripts, h=10, include_self=True)
    n_genes = synthetic_transcripts["gene"].nunique()
    assert X.shape[1] == n_genes
    assert X.shape[0] == len(mol_ids)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    assert (row_sums == 11).all()


def test_get_knn_counts_all_skips_small_cells():
    rng = np.random.default_rng(1)
    rows = []
    for cell_idx in range(3):
        for mol in range(5):
            rows.append(
                {
                    "x": rng.normal(),
                    "y": rng.normal(),
                    "z": rng.normal(),
                    "gene": f"g{rng.integers(0, 4)}",
                    "cell": f"c{cell_idx}",
                    "mol_id": f"m_{cell_idx}_{mol}",
                }
            )
    for cell_idx in range(3, 5):
        for mol in range(50):
            rows.append(
                {
                    "x": rng.normal(),
                    "y": rng.normal(),
                    "z": rng.normal(),
                    "gene": f"g{rng.integers(0, 4)}",
                    "cell": f"c{cell_idx}",
                    "mol_id": f"m_{cell_idx}_{mol}",
                }
            )
    df = pd.DataFrame(rows)
    X, mol_ids = get_knn_counts_all(df, h=10, include_self=True)
    assert X.shape[0] == 100  # only the two big cells contribute, 50 each
    assert all(mid.startswith("m_3") or mid.startswith("m_4") for mid in mol_ids)


def test_fit_nmf_decreases_error():
    rng = np.random.default_rng(0)
    n, k_true, g = 60, 3, 12
    W_true = rng.uniform(0, 1, size=(n, k_true))
    H_true = rng.uniform(0, 1, size=(k_true, g))
    X = (W_true @ H_true).astype(np.float32) + rng.normal(0, 0.05, size=(n, g)).astype(np.float32)
    X = np.maximum(X, 0)
    W, H, _, err = fit_nmf(X, k=k_true, n_runs=4, n_iter=200, seed=0, weighted=False)
    assert W.shape == (n, k_true)
    assert H.shape == (k_true, g)
    err_init = float(np.sum((X - W @ H) ** 2))
    err_baseline = float(np.sum((X - X.mean()) ** 2))
    assert err_init < err_baseline
    assert err > 0
    assert (W >= 0).all()
    assert (H >= 0).all()


def test_run_knn_nmf_end_to_end(synthetic_transcripts: pd.DataFrame):
    res = run_knn_nmf(
        synthetic_transcripts,
        k=3,
        h=10,
        n_runs=3,
        n_iter=100,
        seed=0,
        weighted=True,
    )
    assert isinstance(res, KNNNMFResult)
    assert res.W.shape[1] == 3
    assert res.H.shape[0] == 3
    assert res.W.shape[0] == len(res.mol_ids)
    assert res.H.shape[1] == len(res.gene_names)
    assert (np.asarray(res.W) >= 0).all()
    assert (np.asarray(res.H) >= 0).all()
    assert res.reconstruction_error > 0


def test_run_knn_nmf_recovers_celltype_structure(synthetic_transcripts: pd.DataFrame):
    """With 3 simulated cell types, rank-3 NMF factors should align to cell types."""
    res = run_knn_nmf(
        synthetic_transcripts,
        k=3,
        h=10,
        n_runs=5,
        n_iter=200,
        seed=0,
        weighted=True,
    )
    df = synthetic_transcripts.set_index("mol_id").loc[res.mol_ids]
    W = np.asarray(res.W)
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    df_w = pd.DataFrame(W_norm, index=res.mol_ids)
    df_w["celltype"] = df["celltype"].to_numpy()
    means_per_ct = df_w.groupby("celltype").mean()
    assigned = means_per_ct.idxmax(axis=1).nunique()
    assert assigned == 3, f"Expected 3 distinct dominant factors, got {assigned}"
