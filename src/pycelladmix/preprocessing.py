"""Input preparation: gene filtering, cell-type subsampling, transcript-table validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .utils import gene_index, validate_transcripts


def transcript_counts_matrix(
    df: pd.DataFrame,
    gene_names: list[str] | np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build a ``(n_genes, n_cells)`` counts matrix from a transcript dataframe.

    Returns the sparse counts matrix, the gene-name array (rows), and the
    cell-id array (columns).
    """
    validate_transcripts(df)
    if gene_names is None:
        gene_names = np.asarray(sorted(df["gene"].unique()), dtype=object)
    cell_names = np.asarray(sorted(df["cell"].unique()), dtype=object)
    g_idx, _ = gene_index(df, all_genes=gene_names)
    c_idx = pd.Categorical(df["cell"], categories=list(cell_names)).codes.astype(np.int64)
    data = np.ones(len(df), dtype=np.float32)
    cm = sparse.coo_matrix((data, (g_idx, c_idx)), shape=(len(gene_names), len(cell_names))).tocsr()
    return cm, gene_names, cell_names


def get_high_exp_genes(df: pd.DataFrame, frac_thresh: float = 0.1) -> np.ndarray:
    """Genes expressed in at least ``frac_thresh`` of cells of *some* cell type.

    Mirrors R ``get_high_exp_genes``. Requires ``celltype`` column in ``df``.
    """
    if "celltype" not in df.columns:
        raise ValueError("`df` must have a `celltype` column.")
    cm, gene_names, cell_names = transcript_counts_matrix(df)
    cell_to_type = df.drop_duplicates("cell").set_index("cell")["celltype"].to_dict()
    types = np.asarray([cell_to_type[c] for c in cell_names])
    keep_mask = np.zeros(len(gene_names), dtype=bool)
    for ct in np.unique(types):
        mask = types == ct
        sub = cm[:, mask]
        frac = np.asarray((sub > 0).sum(axis=1)).ravel() / max(int(mask.sum()), 1)
        keep_mask |= frac >= frac_thresh
    return gene_names[keep_mask]


def get_high_var_genes(df: pd.DataFrame, n_hvgs: int = 1000) -> np.ndarray:
    """Top-``n_hvgs`` highly variable genes by log-normalised variance.

    Approximation of Seurat's ``FindVariableFeatures(selection.method="vst")``:
    log1p-normalise counts per cell to total counts, then rank genes by variance.
    Mirrors R ``get_high_var_genes`` in spirit but not in exact ranking.
    """
    cm, gene_names, _ = transcript_counts_matrix(df)
    n_hvgs = min(n_hvgs, cm.shape[0])
    cm_dense = cm.toarray().astype(np.float32)
    col_totals = cm_dense.sum(axis=0)
    col_totals = np.where(col_totals > 0, col_totals, 1.0)
    log_norm = np.log1p(cm_dense / col_totals[None, :] * 1e4)
    var = log_norm.var(axis=1)
    top_idx = np.argsort(var)[::-1][:n_hvgs]
    return np.asarray(gene_names[top_idx], dtype=object)


def subset_genes(
    df: pd.DataFrame,
    top_g_thresh: float | None = None,
    n_hvgs: int | None = None,
) -> pd.DataFrame:
    """Subset transcript dataframe to a chosen gene set.

    Mirrors R ``subset_genes``. If ``top_g_thresh`` is set, restrict to
    high-expression genes (per-cell-type frac > threshold); if ``n_hvgs`` is
    set, then further restrict to the top variable genes among the survivors.
    Adds a sentinel ``gene_sub=True`` column to flag downstream code that
    NMF was fit on a subset.
    """
    out = df.copy()
    if top_g_thresh is not None:
        keep = get_high_exp_genes(out, frac_thresh=top_g_thresh)
        out = out[out["gene"].isin(keep)]
    if n_hvgs is not None:
        keep = get_high_var_genes(out, n_hvgs=n_hvgs)
        out = out[out["gene"].isin(keep)]
    out["gene_sub"] = True
    return out


def balance_cell_types(
    df: pd.DataFrame,
    cell_annot: pd.DataFrame | pd.Series | None = None,
    num_cells_samp: int = 5000,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Multinomial subsample of cells with library-size-balanced probabilities.

    Mirrors R ``samp_ct_equal``. Requires either ``celltype`` column in ``df``
    or a ``cell_annot`` mapping cell id → cell type.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    if cell_annot is None:
        if "celltype" not in df.columns:
            raise ValueError("Need either a `celltype` column or `cell_annot` argument.")
        cell_annot = df.drop_duplicates("cell").set_index("cell")["celltype"]
    elif isinstance(cell_annot, pd.DataFrame):
        if "celltype" not in cell_annot.columns:
            raise ValueError("`cell_annot` DataFrame must have a `celltype` column.")
        cell_annot = cell_annot["celltype"]

    ct_counts = cell_annot.value_counts()
    total_lib = (
        df["celltype"].value_counts()
        if "celltype" in df.columns
        else (
            df.groupby("cell")
            .size()
            .rename("n")
            .to_frame()
            .join(cell_annot.rename("celltype"), how="left")
            .groupby("celltype")["n"]
            .sum()
        )
    )
    av_lib = total_lib / ct_counts.reindex(total_lib.index).fillna(1)
    probs = 1.0 / av_lib
    probs = probs / probs.sum()

    keep_cells: list[str] = []
    for ct, p in probs.items():
        target = int(round(num_cells_samp * float(p)))
        cells_of_type = cell_annot.index[cell_annot == ct].to_numpy()
        if target >= len(cells_of_type):
            keep_cells.extend(cells_of_type.tolist())
        elif target > 0:
            keep_cells.extend(rng.choice(cells_of_type, size=target, replace=False).tolist())
    return df[df["cell"].isin(keep_cells)].copy()


__all__ = [
    "balance_cell_types",
    "get_high_exp_genes",
    "get_high_var_genes",
    "subset_genes",
    "transcript_counts_matrix",
]
