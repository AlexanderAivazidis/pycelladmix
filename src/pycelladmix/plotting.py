"""Visualisation helpers: NMF loading heatmaps, before/after score panels, annotation grids."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .nmf import KNNNMFResult


def plot_nmf_loadings(
    res: KNNNMFResult,
    gene_order: Sequence[str] | None = None,
    *,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap of normalised NMF gene loadings (factors × genes).

    Each row is a factor with loadings normalised to sum to 1 across genes
    (i.e. column-stochastic per the cellAdmix convention).
    """
    H = np.asarray(res.H)
    H = H / (H.sum(axis=1, keepdims=True) + 1e-12)
    gene_names = list(res.gene_names)
    if gene_order is not None:
        idx = [gene_names.index(g) for g in gene_order if g in gene_names]
        H = H[:, idx]
        gene_names = [gene_names[i] for i in idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(gene_names) * 0.15), 0.5 + 0.4 * H.shape[0]))
    im = ax.imshow(H, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(gene_names)))
    ax.set_xticklabels(gene_names, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(H.shape[0]))
    ax.set_yticklabels([f"factor_{i + 1}" for i in range(H.shape[0])])
    ax.set_xlabel("gene")
    ax.set_ylabel("NMF factor")
    plt.colorbar(im, ax=ax, label="normalised loading")
    return ax


def plot_admixture_distribution(
    scores: pd.Series,
    group_labels: pd.Series | None = None,
    *,
    bins: int = 30,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of per-cell admixture scores, optionally split by a categorical label.

    Useful for comparing scores between *known* contaminated and clean cells in
    a benchmark dataset, or between cell types in a real run.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    if group_labels is None:
        ax.hist(scores.dropna(), bins=bins, color="#1f77b4", alpha=0.85)
    else:
        joined = pd.concat([scores.rename("score"), group_labels.rename("group")], axis=1).dropna()
        for g, sub in joined.groupby("group"):
            ax.hist(sub["score"], bins=bins, alpha=0.55, label=f"{g} (n={len(sub)})")
        ax.legend()
    ax.set_xlabel("per-cell admixture fraction")
    ax.set_ylabel("# cells")
    return ax


def plot_factor_assignment_heatmap(
    crf_df: pd.DataFrame,
    cell_to_type: pd.Series,
    *,
    normalize: str = "index",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Contingency heatmap of CRF factor assignments per cell type.

    Parameters
    ----------
    crf_df
        Output of :func:`pycelladmix.crf.run_crf_all` (columns ``mol_id``, ``factor``).
    cell_to_type
        Per-molecule cell-type label (length = ``len(crf_df)`` and aligned
        in order, *or* a Series indexed by ``mol_id`` so we can join).
    normalize
        Forwarded to :func:`pandas.crosstab` (``"index"``, ``"columns"``, ``False``).
    """
    if cell_to_type.index.equals(pd.Index(crf_df["mol_id"])):
        types = cell_to_type.to_numpy()
    elif len(cell_to_type) == len(crf_df):
        types = np.asarray(cell_to_type)
    else:
        types = cell_to_type.reindex(crf_df["mol_id"]).to_numpy()

    cont = pd.crosstab(types, crf_df["factor"], normalize=normalize)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * cont.shape[1] + 2, 0.4 * cont.shape[0] + 1))
    im = ax.imshow(cont.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(cont.shape[1]))
    ax.set_xticklabels([f"f{c}" for c in cont.columns])
    ax.set_yticks(np.arange(cont.shape[0]))
    ax.set_yticklabels(cont.index)
    ax.set_xlabel("CRF factor")
    ax.set_ylabel("cell type")
    plt.colorbar(im, ax=ax, label="fraction" if normalize else "count")
    return ax


__all__ = [
    "plot_admixture_distribution",
    "plot_factor_assignment_heatmap",
    "plot_nmf_loadings",
]
