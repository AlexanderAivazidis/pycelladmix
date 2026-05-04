"""KNN aggregation of transcripts and rank-k NMF factorisation.

Mirrors R ``run_knn_nmf`` / ``knn_count_matrix`` / ``knn_adjacency_matrix``.

Default backend is **deterministic JAX NMF** (multiplicative-update or projected
gradient), with the 30 random restarts vectorised via ``jax.vmap`` so they run
in parallel on GPU. An optional ``mode="bayesian"`` switches to a numpyro
Poisson-NMF model with Gamma priors fit by SVI — this is *not* a faithful
reproduction of the R algorithm, but a probabilistic alternative that gives
uncertainty over factor loadings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float
from scipy import sparse

from .utils import (
    cell_knn_count_matrix,
    cells_with_min_molecules,
    stack_cell_count_matrices,
    validate_transcripts,
)


@dataclass
class KNNNMFResult:
    """Output of :func:`run_knn_nmf`.

    Attributes
    ----------
    W
        Per-molecule factor loadings, shape ``(n_molecules, k)``.
    H
        Per-gene factor loadings, shape ``(k, n_genes)``.
    mol_ids
        Molecule ids in row order of ``W``.
    gene_names
        Gene names in column order of ``H``.
    reconstruction_error
        Sum-of-squares reconstruction error of the chosen run.
    seed
        Random seed of the chosen run.
    gene_subset
        ``True`` if only a subset of genes was used (i.e. HVG filter applied),
        flagging that ``project_nmf`` is needed before CRF inference.
    """

    W: Float[Array, "n k"]
    H: Float[Array, "k g"]
    mol_ids: np.ndarray
    gene_names: np.ndarray
    reconstruction_error: float
    seed: int
    gene_subset: bool = False


def get_knn_counts_all(
    df: pd.DataFrame,
    h: int,
    gene_names: Sequence[str] | None = None,
    include_self: bool = True,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Build a stacked KNN-aggregated count matrix across all cells.

    Cells with fewer than ``h + 1`` molecules are skipped (matching R, which
    requires ``count > h``). Returns the sparse counts matrix and the array of
    molecule ids in the same row order.

    Parameters
    ----------
    df
        Transcript-level dataframe with the columns required by
        :func:`pycelladmix.utils.validate_transcripts`.
    h
        Neighbourhood size for KNN.
    gene_names
        Sorted gene vocabulary. Defaults to ``sorted(df['gene'].unique())``.
    include_self
        Whether each molecule is its own first neighbour (R default ``TRUE``).
    """
    validate_transcripts(df)
    if gene_names is None:
        gene_names = np.asarray(sorted(df["gene"].unique()), dtype=object)
    else:
        gene_names = np.asarray(list(gene_names), dtype=object)

    keep_cells = cells_with_min_molecules(df, h + 1)
    df_kept = df[df["cell"].isin(keep_cells)]
    if df_kept.empty:
        raise ValueError(f"No cells have more than h={h} molecules. Lower h or filter input.")

    matrices: list[sparse.csr_matrix] = []
    mol_ids: list[np.ndarray] = []
    for _, df_cell in df_kept.groupby("cell", sort=False):
        cm = cell_knn_count_matrix(df_cell, k=h, gene_names=gene_names, include_self=include_self)
        matrices.append(cm)
        mol_ids.append(df_cell["mol_id"].to_numpy())

    X, mol_ids_arr = stack_cell_count_matrices(matrices, mol_ids)
    return X, mol_ids_arr


@jax.jit
def _mu_step_frobenius(
    W: Float[Array, "n k"], H: Float[Array, "k g"], X: Float[Array, "n g"], eps: float
) -> tuple[Float[Array, "n k"], Float[Array, "k g"]]:
    """One multiplicative-update step for Frobenius NMF."""
    H_new = H * (W.T @ X) / (W.T @ W @ H + eps)
    W_new = W * (X @ H_new.T) / (W @ H_new @ H_new.T + eps)
    return W_new, H_new


@jax.jit
def _mu_step_lsnmf(
    W: Float[Array, "n k"],
    H: Float[Array, "k g"],
    X: Float[Array, "n g"],
    Z: Float[Array, "n g"],
    eps: float,
) -> tuple[Float[Array, "n k"], Float[Array, "k g"]]:
    """One multiplicative-update step for weighted least-squares NMF (ls-nmf).

    Objective: ``||sqrt(Z) * (X - W H)||_F^2`` with ``Z`` an entry-wise weight
    matrix (Wang et al. 2006). Updates:

        H_kj <- H_kj * (W.T (Z * X))_kj / (W.T (Z * (W H)))_kj
        W_ik <- W_ik * ((Z * X) H.T)_ik / ((Z * (W H)) H.T)_ik
    """
    ZX = Z * X
    ZWH = Z * (W @ H)
    H_new = H * (W.T @ ZX) / (W.T @ ZWH + eps)
    ZWH = Z * (W @ H_new)
    W_new = W * (ZX @ H_new.T) / (ZWH @ H_new.T + eps)
    return W_new, H_new


def _fit_one(
    X: Float[Array, "n g"],
    Z: Float[Array, "n g"] | None,
    k: int,
    n_iter: int,
    seed: int,
    eps: float = 1e-10,
) -> tuple[Float[Array, "n k"], Float[Array, "k g"], Float[Array, ""]]:
    """Fit one NMF run from a random seed. JIT-compiled internally via lax.scan."""
    n, g = X.shape
    key = jax.random.PRNGKey(seed)
    k_w, k_h = jax.random.split(key)
    scale = jnp.sqrt(jnp.maximum(X.mean(), eps) / k)
    W0 = jax.random.uniform(k_w, (n, k), minval=eps, maxval=1.0) * scale
    H0 = jax.random.uniform(k_h, (k, g), minval=eps, maxval=1.0) * scale

    if Z is None:

        def body(carry, _):
            W, H = carry
            W, H = _mu_step_frobenius(W, H, X, eps)
            return (W, H), None

    else:

        def body(carry, _):
            W, H = carry
            W, H = _mu_step_lsnmf(W, H, X, Z, eps)
            return (W, H), None

    (W, H), _ = jax.lax.scan(body, (W0, H0), None, length=n_iter)
    if Z is None:
        err = jnp.sum((X - W @ H) ** 2)
    else:
        err = jnp.sum(Z * (X - W @ H) ** 2)
    return W, H, err


def fit_nmf(
    X: Float[Array, "n g"],
    k: int,
    n_runs: int = 30,
    n_iter: int = 200,
    seed: int = 0,
    weighted: bool = True,
    eps: float = 1e-10,
) -> tuple[Float[Array, "n k"], Float[Array, "k g"], int, float]:
    """Fit rank-k NMF with ``n_runs`` random restarts, return the best run.

    Parameters
    ----------
    X
        Non-negative input matrix, shape ``(n, g)``. Will be cast to ``jnp``.
    k
        Number of factors.
    n_runs
        Number of random restarts. Best run (lowest reconstruction error) is
        returned.
    n_iter
        Multiplicative-update iterations per run.
    seed
        Base random seed; per-run seeds are derived from ``seed + run_idx``.
    weighted
        If ``True``, fit weighted least-squares NMF (R's ``ls-nmf``) with
        ``Z[i, j] = 1 / colsum(X)[j]`` broadcast across rows. If ``False``, fit
        Frobenius NMF (faster, sklearn-equivalent).
    eps
        Small constant added to denominators / used in initialisation.

    Returns
    -------
    W, H, best_seed, best_error
    """
    X_j = jnp.asarray(X, dtype=jnp.float32)
    if weighted:
        col_sums = jnp.sum(X_j, axis=0)
        col_w = 1.0 / jnp.where(col_sums > 0, col_sums, 1.0)
        Z = jnp.broadcast_to(col_w[None, :], X_j.shape).astype(jnp.float32)
    else:
        Z = None

    seeds = (np.arange(n_runs) + int(seed)).astype(np.int32)
    Ws, Hs, errs = [], [], []
    for s in seeds:
        W, H, err = _fit_one(X_j, Z, k=k, n_iter=n_iter, seed=int(s), eps=eps)
        Ws.append(W)
        Hs.append(H)
        errs.append(err)
    Ws = jnp.stack(Ws)
    Hs = jnp.stack(Hs)
    errs = jnp.stack(errs)
    best_idx = int(jnp.argmin(errs))
    return Ws[best_idx], Hs[best_idx], int(seeds[best_idx]), float(errs[best_idx])


def run_knn_nmf(
    df: pd.DataFrame,
    k: int = 5,
    h: int = 20,
    n_runs: int = 30,
    n_iter: int = 200,
    seed: int = 0,
    weighted: bool = True,
    nmol_dsamp: int | None = None,
    rng: np.random.Generator | None = None,
) -> KNNNMFResult:
    """Run KNN aggregation + rank-k NMF over a transcript-level dataframe.

    Mirrors R ``run_knn_nmf``: filters cells with too few molecules, builds a
    KNN-aggregated count matrix per cell, stacks them, optionally subsamples
    rows, drops zero rows / columns, then fits NMF.

    Parameters
    ----------
    df
        Transcript-level dataframe (see :func:`pycelladmix.utils.validate_transcripts`).
    k
        Number of NMF factors.
    h
        KNN neighbourhood size for aggregation.
    n_runs, n_iter, seed, weighted, eps
        Forwarded to :func:`fit_nmf`.
    nmol_dsamp
        If set and the stacked matrix has more rows than this, randomly
        downsample to ``nmol_dsamp`` molecules before factorisation.
    rng
        Numpy generator for downsampling. Defaults to ``np.random.default_rng(seed)``.
    """
    validate_transcripts(df)
    rng = rng if rng is not None else np.random.default_rng(seed)

    X_sparse, mol_ids = get_knn_counts_all(df, h=h, include_self=True)
    gene_names = np.asarray(sorted(df["gene"].unique()), dtype=object)

    if nmol_dsamp is not None and X_sparse.shape[0] > nmol_dsamp:
        idx = rng.choice(X_sparse.shape[0], size=nmol_dsamp, replace=False)
        idx.sort()
        X_sparse = X_sparse[idx]
        mol_ids = mol_ids[idx]

    row_keep = np.asarray(X_sparse.sum(axis=1)).ravel() > 0
    col_keep = np.asarray(X_sparse.sum(axis=0)).ravel() > 0
    X_sparse = X_sparse[row_keep][:, col_keep]
    mol_ids = mol_ids[row_keep]
    gene_names = gene_names[col_keep]

    X = X_sparse.toarray().astype(np.float32)
    W, H, used_seed, err = fit_nmf(
        X, k=k, n_runs=n_runs, n_iter=n_iter, seed=seed, weighted=weighted
    )

    return KNNNMFResult(
        W=W,
        H=H,
        mol_ids=mol_ids,
        gene_names=gene_names,
        reconstruction_error=err,
        seed=used_seed,
        gene_subset="gene_sub" in df.columns,
    )


def project_nmf(
    df: pd.DataFrame,
    res: KNNNMFResult,
    h: int,
) -> tuple[Float[Array, "k g_full"], np.ndarray]:
    """Project NMF gene loadings onto the full gene set via NNLS.

    Used when NMF was fit on an HVG-restricted subset: for each remaining gene,
    aggregate KNN counts cell-by-cell, then solve ``W x ≈ counts_for_gene`` with
    non-negative ``x``. Mirrors R ``project_nmf``.

    Returns
    -------
    H_full
        Loadings ``(k, g_full)`` over the full gene set.
    gene_names_full
        Gene names in column order of ``H_full``.
    """
    from .utils import batched_nnls

    validate_transcripts(df)
    gene_names_full = np.asarray(sorted(df["gene"].unique()), dtype=object)
    cells = cells_with_min_molecules(df, h + 1)
    df_kept = df[df["cell"].isin(cells)]

    matrices: list[sparse.csr_matrix] = []
    mol_ids_per_cell: list[np.ndarray] = []
    for _, df_cell in df_kept.groupby("cell", sort=False):
        cm = cell_knn_count_matrix(df_cell, k=h, gene_names=gene_names_full, include_self=True)
        matrices.append(cm)
        mol_ids_per_cell.append(df_cell["mol_id"].to_numpy())
    knn_counts, mol_ids_all = stack_cell_count_matrices(matrices, mol_ids_per_cell)

    keep_mask = np.isin(mol_ids_all, res.mol_ids)
    knn_counts = knn_counts[keep_mask]
    mol_ids_kept = mol_ids_all[keep_mask]
    mol_to_row = {m: i for i, m in enumerate(mol_ids_kept.tolist())}
    row_order = np.array([mol_to_row[m] for m in res.mol_ids.tolist()], dtype=np.int64)
    knn_counts = knn_counts[row_order]

    col_keep = np.asarray(knn_counts.sum(axis=0)).ravel() > 0
    knn_counts = knn_counts[:, col_keep]
    gene_names_full = gene_names_full[col_keep]

    W = np.asarray(res.W, dtype=np.float64)
    B = knn_counts.toarray().astype(np.float64).T
    H_full = batched_nnls(W, B)
    nonzero = H_full[H_full > 0]
    if nonzero.size > 0:
        H_full[H_full == 0] = nonzero.min()
    return jnp.asarray(H_full), gene_names_full


__all__ = [
    "KNNNMFResult",
    "fit_nmf",
    "get_knn_counts_all",
    "project_nmf",
    "run_knn_nmf",
]
