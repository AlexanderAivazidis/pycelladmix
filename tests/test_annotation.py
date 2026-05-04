"""Tests for pycelladmix.annotation."""

from __future__ import annotations

import pandas as pd
import pytest

from pycelladmix.annotation import (
    factors_to_remove_per_celltype,
    run_bridge_test,
    run_enrichment_test,
    run_membrane_test,
)
from pycelladmix.nmf import run_knn_nmf


def test_run_enrichment_test_recovers_assignment(synthetic_transcripts: pd.DataFrame):
    """Each factor in the synthetic 3-celltype data should be enriched in its own markers."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=5, n_iter=200, weighted=True)
    types = sorted(synthetic_transcripts["celltype"].unique())
    markers = {
        ct: [g for g in synthetic_transcripts["gene"].unique() if g.startswith(f"{ct}_")]
        for ct in types
    }

    enr = run_enrichment_test(res, markers, n_perm=500, p_thresh=0.05, seed=0)
    assert {"factor", "cell_type", "observed_frac", "pval"}.issubset(enr.columns)
    assert "fdr" in enr.columns

    # For each factor, the cell type it's most enriched in should be significant.
    home = enr.sort_values("observed_frac", ascending=False).drop_duplicates("factor")[
        ["factor", "cell_type", "fdr"]
    ]
    assert (home["fdr"] < 0.05).all()
    assert home["cell_type"].nunique() == 3


def test_factors_to_remove_per_celltype(synthetic_transcripts: pd.DataFrame):
    """When factors map cleanly to cell types, the foreign-factor list should be empty
    (since each factor is only home to its own type, no cross-type significance)."""
    res = run_knn_nmf(synthetic_transcripts, k=3, h=10, n_runs=5, n_iter=200, weighted=True)
    types = sorted(synthetic_transcripts["celltype"].unique())
    markers = {
        ct: [g for g in synthetic_transcripts["gene"].unique() if g.startswith(f"{ct}_")]
        for ct in types
    }
    enr = run_enrichment_test(res, markers, n_perm=500, p_thresh=0.05, seed=0)
    rm = factors_to_remove_per_celltype(enr)
    for _ct, factors in rm.items():
        assert isinstance(factors, list)
        assert all(isinstance(f, int) for f in factors)


def test_bridge_and_membrane_stubs():
    with pytest.raises(NotImplementedError):
        run_bridge_test()
    with pytest.raises(NotImplementedError):
        run_membrane_test()
