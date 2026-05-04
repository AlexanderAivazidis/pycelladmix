"""Counts-matrix correction by subtraction of expected admixture.

The simplest "v1" sequential workflow: run :func:`pycelladmix.run_celladmix` (or
the subsample workflow), get ``(W_admix, H_admix)`` from
:func:`pycelladmix.annotation.aggregate_to_admix_prior`, then subtract the
predicted admixture counts ``W_admix @ H_admix`` from the observed
gene-by-cell matrix and clamp at zero. The result is a "cleaned" counts
matrix ready to feed downstream pipelines (Stormi, scanpy, etc.).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def cleaned_counts_matrix(
    observed: np.ndarray | sparse.csr_matrix | sparse.csc_matrix,
    W_admix: np.ndarray,
    H_admix: np.ndarray,
    *,
    cell_axis: str = "columns",
    cell_names: Sequence[str] | None = None,
    cells_in_W: Sequence[str] | None = None,
    cap_at_observed: bool = True,
) -> sparse.csr_matrix:
    """Subtract expected admixture from observed counts and clamp at zero.

    Parameters
    ----------
    observed
        Observed counts matrix. Either ``(n_genes, n_cells)`` (default,
        ``cell_axis="columns"``) or ``(n_cells, n_genes)``
        (``cell_axis="rows"``). Sparse or dense.
    W_admix
        ``(n_cells, k_admix)`` per-cell admixture loadings (from
        :func:`pycelladmix.annotation.aggregate_to_admix_prior`).
    H_admix
        ``(k_admix, n_genes)`` factor-by-gene admixture basis.
    cell_axis
        ``"columns"`` (default; matches scanpy/scverse `(genes, cells)`
        convention) or ``"rows"``.
    cell_names
        Cell ids in the column (or row) order of ``observed``. If both
        ``cell_names`` and ``cells_in_W`` are provided, the function reorders
        ``W_admix`` so its rows align with ``observed``'s cell axis.
    cells_in_W
        Cell ids in the row order of ``W_admix``.
    cap_at_observed
        If ``True`` (default) the predicted admixture counts are also capped
        at ``observed`` per ``(cell, gene)`` so subtraction never produces
        negative counts even before the post-hoc clamp; this matches the
        intuition that admixture cannot exceed what was actually observed.

    Returns
    -------
    Sparse CSR counts matrix in the same orientation as ``observed``.
    """
    if cell_axis not in {"columns", "rows"}:
        raise ValueError("cell_axis must be 'columns' or 'rows'")

    obs = sparse.csr_matrix(observed) if not sparse.issparse(observed) else observed.tocsr()
    if cell_axis == "columns":
        n_genes, n_cells = obs.shape
    else:
        n_cells, n_genes = obs.shape

    W = np.asarray(W_admix, dtype=np.float64)
    H = np.asarray(H_admix, dtype=np.float64)
    if W.shape[1] != H.shape[0]:
        raise ValueError(
            f"W_admix has shape {W.shape} but H_admix has shape {H.shape}; "
            "their inner dimensions (k_admix) must match."
        )
    if H.shape[1] != n_genes:
        raise ValueError(
            f"H_admix has {H.shape[1]} gene columns but `observed` has {n_genes} genes."
        )

    if cell_names is not None and cells_in_W is not None:
        cell_to_idx = {c: i for i, c in enumerate(list(cells_in_W))}
        try:
            order = np.asarray([cell_to_idx[c] for c in cell_names])
        except KeyError as e:
            raise ValueError(f"Cell {e!r} from `cell_names` is not present in `cells_in_W`.") from e
        W = W[order]
    if W.shape[0] != n_cells:
        raise ValueError(f"W_admix has {W.shape[0]} cell rows but `observed` has {n_cells} cells.")

    admix_pred = np.maximum(W @ H, 0.0)  # (n_cells, n_genes)

    if cell_axis == "columns":
        obs_dense = obs.toarray()  # (n_genes, n_cells)
        admix_pred_T = admix_pred.T  # (n_genes, n_cells)
        if cap_at_observed:
            admix_pred_T = np.minimum(admix_pred_T, obs_dense)
        cleaned = obs_dense - admix_pred_T
    else:
        obs_dense = obs.toarray()  # (n_cells, n_genes)
        if cap_at_observed:
            admix_pred = np.minimum(admix_pred, obs_dense)
        cleaned = obs_dense - admix_pred

    cleaned = np.maximum(cleaned, 0.0)
    return sparse.csr_matrix(cleaned)


def admixture_count_summary(
    W_admix: np.ndarray,
    H_admix: np.ndarray,
    observed: np.ndarray | sparse.csr_matrix | sparse.csc_matrix,
    *,
    cell_axis: str = "columns",
) -> pd.DataFrame:
    """Per-cell summary of total admixture removed by :func:`cleaned_counts_matrix`.

    Returns a DataFrame indexed by cell row with columns:

    * ``observed_total``: total observed counts in this cell.
    * ``admix_predicted``: total predicted admixture counts (before clamping).
    * ``admix_removed``: total admixture actually subtracted (post-clamp).
    * ``frac_removed``: ``admix_removed / observed_total`` (NaN if zero).
    """
    obs = sparse.csr_matrix(observed) if not sparse.issparse(observed) else observed.tocsr()
    if cell_axis == "columns":
        observed_total = np.asarray(obs.sum(axis=0)).ravel()
    else:
        observed_total = np.asarray(obs.sum(axis=1)).ravel()

    pred = np.maximum(np.asarray(W_admix) @ np.asarray(H_admix), 0.0)
    admix_predicted = pred.sum(axis=1)
    cleaned = cleaned_counts_matrix(observed, W_admix, H_admix, cell_axis=cell_axis)
    if cell_axis == "columns":
        cleaned_total = np.asarray(cleaned.sum(axis=0)).ravel()
    else:
        cleaned_total = np.asarray(cleaned.sum(axis=1)).ravel()
    admix_removed = observed_total - cleaned_total
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(observed_total > 0, admix_removed / observed_total, np.nan)

    return pd.DataFrame(
        {
            "observed_total": observed_total,
            "admix_predicted": admix_predicted,
            "admix_removed": admix_removed,
            "frac_removed": frac,
        }
    )


__all__ = ["admixture_count_summary", "cleaned_counts_matrix"]
