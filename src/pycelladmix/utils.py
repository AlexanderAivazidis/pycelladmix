"""Shared low-level helpers: KNN graph construction, sparse correlation, batched NNLS."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float, Int
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

REQUIRED_TX_COLUMNS = ("x", "y", "z", "gene", "cell", "mol_id")


def validate_transcripts(df: pd.DataFrame) -> None:
    """Check that ``df`` has the columns expected of a transcript-level dataframe."""
    missing = [c for c in REQUIRED_TX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Transcript dataframe is missing required columns: {missing}. "
            f"Required: {REQUIRED_TX_COLUMNS}."
        )
    if df["mol_id"].duplicated().any():
        raise ValueError("`mol_id` must be unique across the transcript dataframe.")


def cell_sizes(df: pd.DataFrame) -> pd.Series:
    """Number of molecules per cell, indexed by cell id."""
    return df.groupby("cell", sort=False).size()


def cells_with_min_molecules(df: pd.DataFrame, min_molecules: int) -> np.ndarray:
    """Return cell ids that have at least ``min_molecules`` molecules."""
    sizes = cell_sizes(df)
    return sizes.index[sizes >= min_molecules].to_numpy()


def gene_index(
    df: pd.DataFrame, all_genes: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``df['gene']`` to integer indices.

    Returns ``(gene_idx, gene_names)`` where ``gene_idx[i]`` is the integer
    index of molecule ``i``'s gene within ``gene_names``.
    """
    if all_genes is None:
        gene_names = np.asarray(sorted(df["gene"].unique()), dtype=object)
    else:
        gene_names = np.asarray(list(all_genes), dtype=object)
    name_to_idx = {g: i for i, g in enumerate(gene_names.tolist())}
    gene_idx = df["gene"].map(name_to_idx).to_numpy()
    if np.any(pd.isna(gene_idx)):
        unknown = sorted(set(df["gene"]) - set(gene_names.tolist()))
        raise ValueError(f"Genes in df missing from `all_genes`: {unknown[:5]} (+more)")
    return gene_idx.astype(np.int64), gene_names


def knn_indices_xyz(
    coords: Float[np.ndarray, "n 3"], k: int, include_self: bool = False
) -> Int[np.ndarray, "n k"]:
    """k-nearest neighbour indices on a 3D point cloud.

    Parameters
    ----------
    coords
        ``(n, 3)`` array of x/y/z coordinates.
    k
        Number of neighbours per point. Self-loops are excluded by default
        (matching R ``FNN::get.knn``); set ``include_self=True`` to mark each
        point as its own first neighbour.
    """
    n = coords.shape[0]
    if k >= n:
        raise ValueError(f"Requested k={k} neighbours but only {n} points available.")
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    _, idx = nn.kneighbors(coords)
    if include_self:
        idx[:, 0] = np.arange(n)
        return idx[:, : k + 1] if k + 1 <= idx.shape[1] else idx
    return idx[:, 1:]


def cell_knn_count_matrix(
    df_cell: pd.DataFrame,
    k: int,
    gene_names: Sequence[str],
    include_self: bool = True,
) -> sparse.csr_matrix:
    """KNN-aggregated count matrix for a single cell.

    For each of the ``n`` molecules in the cell, sum one-hot gene labels of its
    ``k`` nearest neighbours (and itself, if ``include_self``). Mirrors R
    ``knn_count_matrix``.

    Returns
    -------
    Sparse ``(n_molecules, n_genes)`` matrix in CSR format.
    """
    coords = df_cell[["x", "y", "z"]].to_numpy(dtype=np.float64)
    n = coords.shape[0]
    if k >= n:
        raise ValueError(f"Cell has {n} molecules but k={k} neighbours requested.")

    nn_idx = knn_indices_xyz(coords, k=k, include_self=False)

    gene_idx, _ = gene_index(df_cell, all_genes=gene_names)
    k_eff = nn_idx.shape[1] + (1 if include_self else 0)

    rows = np.repeat(np.arange(n), k_eff)
    if include_self:
        nb = np.column_stack([np.arange(n), nn_idx]).reshape(-1)
    else:
        nb = nn_idx.reshape(-1)
    cols = gene_idx[nb]
    data = np.ones_like(rows, dtype=np.float32)

    n_genes = len(gene_names)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n_genes)).tocsr()


def stack_cell_count_matrices(
    matrices_per_cell: list[sparse.csr_matrix],
    mol_ids_per_cell: list[np.ndarray],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Stack per-cell KNN count matrices and concatenate molecule ids."""
    if not matrices_per_cell:
        raise ValueError("No cells provided.")
    X = sparse.vstack(matrices_per_cell, format="csr")
    mol_ids = np.concatenate(mol_ids_per_cell)
    return X, mol_ids


def sparse_corr(X: sparse.csr_matrix | sparse.csc_matrix) -> Float[np.ndarray, "n n"]:
    """Pearson correlation between rows of a sparse matrix.

    Mirrors R ``sparse_cor``. Computes ``cov(X) / (std(X) std(X)^T)`` with
    sparse-friendly intermediates. Returns a dense ``(n_rows, n_rows)`` matrix.
    """
    X = X.tocsr().astype(np.float64)
    n_rows, n_cols = X.shape
    means = np.asarray(X.mean(axis=1)).ravel()
    sq = X.multiply(X)
    sums_sq = np.asarray(sq.sum(axis=1)).ravel()
    var = sums_sq / n_cols - means**2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    cov = (X @ X.T).toarray() / n_cols - np.outer(means, means)
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    return corr


def batched_nnls(A: Float[Array, "m k"], B: Float[Array, "m n"]) -> Float[np.ndarray, "k n"]:
    """Solve ``A x_j ≈ b_j`` with non-negative ``x_j`` for each column ``b_j`` of B.

    Used in :func:`pycelladmix.nmf.project_nmf` to extend NMF gene loadings to
    genes outside the HVG subset. Each column is solved independently with
    :func:`scipy.optimize.nnls`.
    """
    from scipy.optimize import nnls

    A_np = np.asarray(A, dtype=np.float64)
    B_np = np.asarray(B, dtype=np.float64)
    k = A_np.shape[1]
    n = B_np.shape[1]
    out = np.zeros((k, n), dtype=np.float64)
    for j in range(n):
        out[:, j], _ = nnls(A_np, B_np[:, j])
    return out


@partial(jax.jit, static_argnames=("by",))
def normalize_loadings(H: Float[Array, "k g"], by: str = "gene") -> Float[Array, "k g"]:
    """Normalise NMF gene loadings.

    Mirrors R ``run_crf_all`` ``normalize.by`` argument:

    * ``"gene"``: each factor's loadings sum to 1 across genes (column sum to 1
      in the R convention where loadings are gene × factor; here factor × gene
      so we normalise across the gene axis = axis 1).
    * ``"factor"``: each gene's loadings sum to 1 across factors.
    * ``"gene.factor"``: apply factor normalisation, then gene normalisation.

    Notes
    -----
    Loadings are returned with shape ``(k, g)`` where ``k`` is the number of
    factors and ``g`` is the number of genes — the transpose of R's convention.
    """
    eps = 1e-12
    if by == "gene":
        return H / (jnp.sum(H, axis=1, keepdims=True) + eps)
    if by == "factor":
        return H / (jnp.sum(H, axis=0, keepdims=True) + eps)
    if by == "gene.factor":
        H1 = H / (jnp.sum(H, axis=0, keepdims=True) + eps)
        return H1 / (jnp.sum(H1, axis=1, keepdims=True) + eps)
    raise ValueError(f"Unknown normalisation '{by}'. Expected 'gene', 'factor', or 'gene.factor'.")


__all__ = [
    "REQUIRED_TX_COLUMNS",
    "batched_nnls",
    "cell_knn_count_matrix",
    "cell_sizes",
    "cells_with_min_molecules",
    "gene_index",
    "knn_indices_xyz",
    "normalize_loadings",
    "sparse_corr",
    "stack_cell_count_matrices",
    "validate_transcripts",
]
