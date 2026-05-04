"""Tests for the subsample-and-project scale workflow:
- pycelladmix.nmf.project_per_molecule_loadings
- pycelladmix.annotation.aggregate_to_admix_prior
- pycelladmix.correction.cleaned_counts_matrix / admixture_count_summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from pycelladmix.annotation import aggregate_to_admix_prior, run_enrichment_test
from pycelladmix.correction import admixture_count_summary, cleaned_counts_matrix
from pycelladmix.nmf import project_per_molecule_loadings, run_knn_nmf
from pycelladmix.preprocessing import transcript_counts_matrix


def test_project_per_molecule_loadings_shape_and_signs(synthetic_transcripts: pd.DataFrame):
    """Projection on the same df used for fit should give non-negative loadings of correct shape."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=3, n_iter=200, weighted=True)
    W_full, mol_ids = project_per_molecule_loadings(
        synthetic_transcripts, H=res.H, gene_names=list(res.gene_names), h=10
    )
    assert W_full.shape == (len(mol_ids), 3)
    assert np.all(np.asarray(W_full) >= 0)
    assert len(mol_ids) >= len(res.mol_ids) - 5  # roughly the same set


def test_project_per_molecule_loadings_subsample_then_full(synthetic_transcripts: pd.DataFrame):
    """Fit NMF on a subsample, then project on the full df. Loadings should still cluster by celltype."""
    rng = np.random.default_rng(0)
    sub_cells = rng.choice(synthetic_transcripts["cell"].unique(), size=80, replace=False)
    df_sub = synthetic_transcripts[synthetic_transcripts["cell"].isin(sub_cells)].copy()
    res = run_knn_nmf(df_sub, k=3, h=10, n_runs=3, n_iter=200, weighted=True)

    W_full, mol_ids = project_per_molecule_loadings(
        synthetic_transcripts, H=res.H, gene_names=list(res.gene_names), h=10
    )
    assert W_full.shape[1] == 3

    # Per-cell-type mean of W: each cell type's mean should peak on a different factor.
    df_idx = synthetic_transcripts.set_index("mol_id").loc[mol_ids]
    df_w = pd.DataFrame(np.asarray(W_full))
    df_w["celltype"] = df_idx["celltype"].to_numpy()
    means = df_w.groupby("celltype").mean()
    assert means.idxmax(axis=1).nunique() == 3, "Each cell type should align to a distinct factor"


def test_aggregate_to_admix_prior_masks_per_celltype(synthetic_transcripts: pd.DataFrame):
    """Cells of type A get zero loading on factors flagged only for type B's admixture."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=3, n_iter=200, weighted=True)
    W_full, mol_ids = project_per_molecule_loadings(
        synthetic_transcripts, H=res.H, gene_names=list(res.gene_names), h=10
    )
    admixture_factors = {"A": [1], "B": [2]}
    W_admix, H_admix, cell_names, _, admix_factor_ids = aggregate_to_admix_prior(
        synthetic_transcripts,
        W_per_molecule=W_full,
        mol_ids=mol_ids,
        H=res.H,
        gene_names=res.gene_names,
        admixture_factors=admixture_factors,
    )
    assert W_admix.shape == (len(cell_names), 2)
    assert H_admix.shape == (2, len(res.gene_names))
    np.testing.assert_array_equal(admix_factor_ids, np.array([1, 2]))

    cell_to_type = synthetic_transcripts.drop_duplicates("cell").set_index("cell")["celltype"]
    types = np.asarray([cell_to_type[c] for c in cell_names])

    col_factor_1 = 0
    col_factor_2 = 1
    assert np.all(W_admix[types == "B", col_factor_1] == 0)  # B cells don't get factor-1 admix
    assert np.all(W_admix[types == "A", col_factor_2] == 0)  # A cells don't get factor-2 admix
    assert np.all(W_admix[types == "C", :] == 0)  # C has no admixture flagged at all
    assert (W_admix[types == "A", col_factor_1] > 0).any()
    assert (W_admix[types == "B", col_factor_2] > 0).any()


def test_aggregate_to_admix_prior_with_real_enrichment(synthetic_transcripts: pd.DataFrame):
    """Use the marker-enrichment test to derive admixture_factors, then aggregate."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=5, n_iter=200, weighted=True)
    types = sorted(synthetic_transcripts["celltype"].unique())
    markers = {
        ct: [g for g in synthetic_transcripts["gene"].unique() if g.startswith(f"{ct}_")]
        for ct in types
    }

    enr = run_enrichment_test(res, markers, n_perm=300, p_thresh=0.1, seed=0)
    sig = enr[enr["significant"]].copy()
    home = (
        sig.sort_values("observed_frac", ascending=False)
        .drop_duplicates("factor")
        .rename(columns={"cell_type": "home"})[["factor", "home"]]
    )
    sig = sig.merge(home, on="factor", how="left")
    foreign = sig[sig["cell_type"] != sig["home"]]
    admixture_factors = {
        ct: g["factor"].astype(int).tolist() for ct, g in foreign.groupby("cell_type")
    }

    if not admixture_factors:
        admixture_factors = {types[0]: [int(home.iloc[1]["factor"])]}

    W_full, mol_ids = project_per_molecule_loadings(
        synthetic_transcripts, H=res.H, gene_names=list(res.gene_names), h=10
    )
    W_admix, H_admix, cell_names, _, _ = aggregate_to_admix_prior(
        synthetic_transcripts,
        W_per_molecule=W_full,
        mol_ids=mol_ids,
        H=res.H,
        gene_names=res.gene_names,
        admixture_factors=admixture_factors,
    )
    assert W_admix.shape[0] == len(cell_names)
    assert W_admix.shape[1] >= 1
    assert (W_admix >= 0).all()


def test_cleaned_counts_matrix_subtraction():
    rng = np.random.default_rng(0)
    n_genes, n_cells = 5, 7
    observed = rng.poisson(8, size=(n_genes, n_cells)).astype(np.float64)
    W = rng.uniform(0, 1, size=(n_cells, 2)).astype(np.float64)
    H = rng.uniform(0, 1, size=(2, n_genes)).astype(np.float64)

    cleaned = cleaned_counts_matrix(observed, W, H, cell_axis="columns").toarray()
    assert cleaned.shape == observed.shape
    assert (cleaned >= 0).all()
    assert (cleaned <= observed + 1e-9).all()

    pred = np.minimum(W @ H, observed.T).clip(min=0).T
    expected = np.maximum(observed - pred, 0.0)
    np.testing.assert_allclose(cleaned, expected, atol=1e-6)


def test_cleaned_counts_matrix_zero_admix_is_identity():
    rng = np.random.default_rng(1)
    obs = sparse.csr_matrix(rng.poisson(3, size=(4, 5)).astype(np.float64))
    W = np.zeros((5, 2))
    H = np.zeros((2, 4))
    cleaned = cleaned_counts_matrix(obs, W, H).toarray()
    np.testing.assert_array_equal(cleaned, obs.toarray())


def test_admixture_count_summary_has_required_columns(synthetic_transcripts: pd.DataFrame):
    cm, gene_names, cell_names = transcript_counts_matrix(synthetic_transcripts)
    n_cells = cm.shape[1]
    rng = np.random.default_rng(0)
    W_admix = rng.uniform(0, 0.5, size=(n_cells, 2))
    H_admix = rng.uniform(0, 0.5, size=(2, len(gene_names)))
    summary = admixture_count_summary(W_admix, H_admix, cm)
    for col in ["observed_total", "admix_predicted", "admix_removed", "frac_removed"]:
        assert col in summary.columns
    assert (summary["admix_removed"] <= summary["observed_total"] + 1e-9).all()
