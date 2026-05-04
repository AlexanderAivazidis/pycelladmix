"""Tests for pycelladmix.utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pycelladmix.utils import (
    cell_knn_count_matrix,
    cell_sizes,
    cells_with_min_molecules,
    gene_index,
    knn_indices_xyz,
    normalize_loadings,
    sparse_corr,
    validate_transcripts,
)


def test_validate_transcripts_missing_column():
    df = pd.DataFrame({"x": [0], "y": [0], "z": [0], "gene": ["A"], "cell": ["c1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_transcripts(df)


def test_validate_transcripts_duplicate_mol_id():
    df = pd.DataFrame(
        {
            "x": [0, 0],
            "y": [0, 0],
            "z": [0, 0],
            "gene": ["A", "B"],
            "cell": ["c1", "c1"],
            "mol_id": ["m1", "m1"],
        }
    )
    with pytest.raises(ValueError, match="must be unique"):
        validate_transcripts(df)


def test_cell_sizes_and_filter(synthetic_transcripts: pd.DataFrame):
    sizes = cell_sizes(synthetic_transcripts)
    assert (sizes == 80).all()
    kept = cells_with_min_molecules(synthetic_transcripts, 50)
    assert len(kept) == synthetic_transcripts["cell"].nunique()
    kept_high = cells_with_min_molecules(synthetic_transcripts, 100)
    assert len(kept_high) == 0


def test_gene_index_consistency(synthetic_transcripts: pd.DataFrame):
    idx, names = gene_index(synthetic_transcripts)
    assert (idx >= 0).all()
    assert idx.max() < len(names)
    recovered = names[idx]
    np.testing.assert_array_equal(recovered, synthetic_transcripts["gene"].to_numpy())


def test_knn_indices_excludes_self():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    idx = knn_indices_xyz(coords, k=2)
    for i, row in enumerate(idx):
        assert i not in row.tolist()
    assert idx.shape == (4, 2)


def test_cell_knn_count_matrix_shape(synthetic_transcripts: pd.DataFrame):
    df_cell = synthetic_transcripts[synthetic_transcripts["cell"] == "cell_0"]
    gene_names = np.asarray(sorted(synthetic_transcripts["gene"].unique()))
    cm = cell_knn_count_matrix(df_cell, k=10, gene_names=gene_names, include_self=True)
    assert isinstance(cm, sparse.csr_matrix)
    assert cm.shape == (len(df_cell), len(gene_names))
    row_sums = np.asarray(cm.sum(axis=1)).ravel()
    np.testing.assert_array_equal(row_sums, np.full(len(df_cell), 11))  # k + self


def test_normalize_loadings_gene():
    H = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = np.asarray(normalize_loadings(H, by="gene"))
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_normalize_loadings_factor():
    H = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = np.asarray(normalize_loadings(H, by="factor"))
    np.testing.assert_allclose(out.sum(axis=0), 1.0, atol=1e-6)


def test_sparse_corr_matches_numpy():
    rng = np.random.default_rng(0)
    X_dense = rng.poisson(2.0, size=(8, 50)).astype(np.float64)
    expected = np.corrcoef(X_dense)
    X_sparse = sparse.csr_matrix(X_dense)
    got = sparse_corr(X_sparse)
    np.testing.assert_allclose(got, expected, atol=1e-6)
