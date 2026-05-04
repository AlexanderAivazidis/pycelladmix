"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_transcripts() -> pd.DataFrame:
    """A small synthetic transcript-level dataframe.

    50 cells across 3 cell types, each with 80 molecules sampled from a
    cell-type-specific gene distribution. 30 genes total (10 markers per
    type). 3D coordinates are scattered around per-cell centroids on a grid.
    """
    rng = np.random.default_rng(0)

    cell_types = ["A", "B", "C"]
    n_cells_per_type = 50
    n_mols_per_cell = 80
    n_markers_per_type = 10

    gene_names = []
    type_to_genes = {}
    for ct in cell_types:
        genes = [f"{ct}_g{i}" for i in range(n_markers_per_type)]
        type_to_genes[ct] = genes
        gene_names.extend(genes)

    rows = []
    cell_id = 0
    for ct in cell_types:
        for _ in range(n_cells_per_type):
            cx, cy, cz = rng.uniform(0, 100, size=3)
            for _ in range(n_mols_per_cell):
                if rng.random() < 0.85:
                    g = rng.choice(type_to_genes[ct])
                else:
                    g = rng.choice(gene_names)
                rows.append(
                    {
                        "x": cx + rng.normal(0, 2.0),
                        "y": cy + rng.normal(0, 2.0),
                        "z": cz + rng.normal(0, 0.5),
                        "gene": g,
                        "cell": f"cell_{cell_id}",
                        "celltype": ct,
                        "mol_id": f"m_{cell_id}_{_}",
                    }
                )
            cell_id += 1
    df = pd.DataFrame(rows)
    df["mol_id"] = [f"m{i:06d}" for i in range(len(df))]
    return df
