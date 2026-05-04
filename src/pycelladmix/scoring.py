"""Per-cell Bayesian admixture scoring.

Mirrors R ``estimate_contamination_scores`` and supporting helpers
(``estimate_cell_adjacency``, ``estimate_cell_type_adjacency``,
``estimate_gene_prob_per_type``, ``estimate_correlation_preservation``).

Inputs: an scRNA-seq reference, the spatial counts, and a cell-type
adjacency matrix derived from Delaunay triangulation of cell centroids.

Closed-form posterior P(contaminated | gene, cell-type) — implemented in JAX
for vectorisation across cells and genes. Cell-type adjacency comes from
:mod:`scipy.spatial.Delaunay`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float
from scipy import sparse
from scipy.spatial import Delaunay

REQUIRED_ADJACENCY_COLUMNS = ("x", "y", "cell_type", "cell")


@dataclass
class ContaminationResult:
    """Output of :func:`estimate_contamination_scores`.

    Attributes
    ----------
    contamination_probs
        Per-gene, per-cell-type contamination probability,
        ``(n_genes, n_celltypes)`` indexed by gene names / cell-type names.
    cell_admixture_fractions
        Per-cell admixture fraction, indexed by cell id.
    """

    contamination_probs: pd.DataFrame
    cell_admixture_fractions: pd.Series


def estimate_cell_adjacency(
    df_spatial: pd.DataFrame,
    edge_max_mad: float = 4.0,
    random_shift: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Cell-cell adjacency via Delaunay triangulation of molecule positions.

    Mirrors R ``estimate_cell_adjacency`` (single-grid variant — drops the
    ``n.cores``-driven grid splitting since SciPy's ``Delaunay`` is fast
    enough on a single core for whole-tissue datasets, and the grid
    splitting was a CPU-parallelisation hack rather than an algorithmic
    requirement).

    Parameters
    ----------
    df_spatial
        Molecule-level dataframe with columns ``x``, ``y``, ``cell_type``, ``cell``.
    edge_max_mad
        Edges with length above ``median + edge_max_mad * MAD`` are dropped to
        cull spurious long Delaunay edges across tissue voids.
    random_shift
        Small uniform jitter added to coordinates before triangulation, to
        break ties from molecules at exactly equal coordinates.

    Returns
    -------
    DataFrame with columns ``cell_s``, ``cell_e``, ``cts``, ``cte`` (source /
    target cell + cell-type) for each unique cell-cell adjacency. Both
    directions are included.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    missing = [c for c in REQUIRED_ADJACENCY_COLUMNS if c not in df_spatial.columns]
    if missing:
        raise ValueError(f"`df_spatial` missing required columns: {missing}.")

    df = df_spatial.dropna(subset=["cell_type"]).copy()
    df["x"] = df["x"].to_numpy() + rng.uniform(-random_shift, random_shift, size=len(df))
    df["y"] = df["y"].to_numpy() + rng.uniform(-random_shift, random_shift, size=len(df))

    coords = df[["x", "y"]].to_numpy()
    tri = Delaunay(coords)
    simplices = tri.simplices  # (n_tri, 3)
    pairs = np.concatenate(
        [simplices[:, [0, 1]], simplices[:, [0, 2]], simplices[:, [1, 2]]], axis=0
    )
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)

    is_, ie = pairs[:, 0], pairs[:, 1]
    dx = coords[is_, 0] - coords[ie, 0]
    dy = coords[is_, 1] - coords[ie, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) * 1.4826
    keep = dist < (med + edge_max_mad * mad)
    is_, ie = is_[keep], ie[keep]

    cells = df["cell"].to_numpy()
    types = df["cell_type"].to_numpy()
    keep_cells = cells[is_] != cells[ie]
    is_, ie = is_[keep_cells], ie[keep_cells]

    fwd = pd.DataFrame(
        {
            "cell_s": cells[is_],
            "cell_e": cells[ie],
            "cts": types[is_],
            "cte": types[ie],
        }
    )
    bwd = pd.DataFrame(
        {
            "cell_s": cells[ie],
            "cell_e": cells[is_],
            "cts": types[ie],
            "cte": types[is_],
        }
    )
    return pd.concat([fwd, bwd], ignore_index=True).drop_duplicates()


def estimate_cell_type_adjacency(adj_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cell-cell adjacency into cell-type × cell-type matrix.

    Mirrors R ``estimate_cell_type_adjacency``: count source-cells with a given
    (cts, cte) edge and average per source cell.

    Returns
    -------
    DataFrame ``(n_celltypes, n_celltypes)`` indexed by source / target type
    names. Missing entries are filled with zero. Symmetric in expectation but
    not guaranteed bit-exact (independent groupbys).
    """
    counts = (
        adj_df.groupby(["cell_s", "cts", "cte"], observed=True)
        .size()
        .reset_index(name="n")
        .groupby(["cts", "cte"], observed=True)["n"]
        .mean()
        .reset_index()
        .pivot(index="cte", columns="cts", values="n")
    )
    types = sorted(set(counts.index) | set(counts.columns))
    counts = counts.reindex(index=types, columns=types).fillna(0.0)
    return counts


def estimate_gene_prob_per_type(
    cm: sparse.csr_matrix | sparse.csc_matrix | np.ndarray,
    cell_to_type: pd.Series,
    expr_thres: float = 0.0,
    use_counts: bool = False,
) -> pd.DataFrame:
    """Per-cell-type gene expression probability.

    Parameters
    ----------
    cm
        Counts matrix, shape ``(n_genes, n_cells)``. Sparse or dense.
    cell_to_type
        Mapping cell id → cell type, length ``n_cells``. Index ordering must
        match the columns of ``cm``.
    expr_thres
        Threshold above which a gene counts as "expressed" (default 0).
    use_counts
        If ``True``, return relative gene-count fractions per type. If ``False``
        (default, R default), return mean expression-indicator per type.

    Returns
    -------
    DataFrame ``(n_celltypes, n_genes)``. Indexed by cell-type name.
    """
    if isinstance(cm, np.ndarray):
        cm = sparse.csr_matrix(cm)
    if cm.shape[1] != len(cell_to_type):
        raise ValueError(
            f"cm has {cm.shape[1]} cell columns but cell_to_type has {len(cell_to_type)} entries."
        )
    types = sorted(cell_to_type.unique().tolist())
    out = np.zeros((len(types), cm.shape[0]), dtype=np.float64)
    cells_arr = cell_to_type.to_numpy()
    for i, ct in enumerate(types):
        mask = cells_arr == ct
        sub = cm[:, mask]
        if use_counts:
            out[i] = np.asarray(sub.sum(axis=1)).ravel()
        else:
            out[i] = np.asarray((sub > expr_thres).sum(axis=1)).ravel() / max(mask.sum(), 1)
    if use_counts:
        col_sums = out.sum(axis=0, keepdims=True)
        out = np.where(col_sums > 0, out / col_sums, 0.0)
    gene_names = (
        cm.indices  # not really gene names; expect caller to assign columns
        if False
        else None
    )
    df = pd.DataFrame(out, index=types)
    if gene_names is not None:
        df.columns = gene_names
    return df


def _gene_scores(
    prob: Float[Array, "t g"],
    K: Float[Array, "t t"],
    type_idx: int,
) -> tuple[Float[Array, " g"], Float[Array, " g"]]:
    """Per-gene contamination ``cont`` and reference ``ref`` scores.

    For target type ``u = type_idx``:

        cont_fracs[v] = K[v, u] / sum_{x != u} K[x, u]   for v != u
        cont[g]       = sum_{v != u} prob[v, g] * cont_fracs[v]
        ref[g]        = prob[u, g]
    """
    n_types = K.shape[0]
    mask = jnp.arange(n_types) != type_idx
    col = K[:, type_idx]
    denom = jnp.sum(jnp.where(mask, col, 0.0))
    cont_fracs = jnp.where(mask, col / jnp.where(denom > 0, denom, 1.0), 0.0)
    cont = jnp.einsum("v,vg->g", cont_fracs, prob)
    ref = prob[type_idx]
    return cont, ref


def _gene_contamination_probs(
    cont: Float[Array, " g"],
    ref: Float[Array, " g"],
    p_cu: float,
) -> Float[Array, " g"]:
    """Posterior P(contaminated | gene) given combined cont / ref scores."""
    num = p_cu * cont
    den = p_cu * cont + (1.0 - p_cu) * ref
    return jnp.where(den > 0, num / den, 0.0)


def estimate_contamination_scores(
    cm_rna: sparse.csr_matrix | sparse.csc_matrix | np.ndarray,
    cm_spatial: sparse.csr_matrix | sparse.csc_matrix | np.ndarray,
    annot_rna: pd.Series,
    annot_spatial: pd.Series,
    cell_type_adj_mat: pd.DataFrame,
    gene_names: Sequence[str],
    p_c: float | None = 0.25,
    signal_thres: float = 0.25,
    min_expr_frac: float = 0.05,
    use_counts: bool = False,
    adjust: bool = True,
    exclude_cell_types: Sequence[str] | None = None,
) -> ContaminationResult:
    """Bayesian per-cell admixture scoring.

    Closed-form posterior P(contaminated | gene, cell-type) computed for each
    cell type, applied to per-cell counts to produce a scalar admixture
    fraction per cell. Mirrors R ``estimate_contamination_scores``.

    Parameters
    ----------
    cm_rna, cm_spatial
        Counts matrices ``(n_genes, n_cells)`` for the scRNA-seq reference and
        the spatial dataset. Must have the same gene order (caller's
        responsibility — restrict to common genes before calling).
    annot_rna, annot_spatial
        ``cell_id → cell_type`` mappings, indexed in the same order as the
        columns of ``cm_rna`` / ``cm_spatial`` respectively.
    cell_type_adj_mat
        Cell-type × cell-type adjacency, e.g. from
        :func:`estimate_cell_type_adjacency`. Index / columns are cell-type names.
    gene_names
        Names of the genes (rows of ``cm_rna`` / ``cm_spatial``).
    p_c
        Prior weight on contamination — multiplied by ``1 - K[u,u]/sum(K[:,u])``
        to give per-celltype prior. ``None`` falls back to a flat 0.5 prior.
    signal_thres
        Per-gene posterior threshold above which a gene contributes to the
        per-cell admixture fraction.
    min_expr_frac
        A gene must have expression-fraction above this in *some* cell type to
        count as informative.
    use_counts
        Forwarded to :func:`estimate_gene_prob_per_type`.
    adjust
        If ``True``, subtract per-celltype baseline ``cont_fracs_null`` from
        each cell's score (R default).
    exclude_cell_types
        Cell types to skip entirely.

    Returns
    -------
    :class:`ContaminationResult` with per-gene contamination probabilities
    (indexed gene × celltype) and per-cell admixture fractions.
    """
    if isinstance(cm_rna, np.ndarray):
        cm_rna = sparse.csr_matrix(cm_rna)
    if isinstance(cm_spatial, np.ndarray):
        cm_spatial = sparse.csr_matrix(cm_spatial)
    if cm_rna.shape[0] != cm_spatial.shape[0]:
        raise ValueError("cm_rna and cm_spatial must have the same number of genes (rows).")
    if len(gene_names) != cm_rna.shape[0]:
        raise ValueError("gene_names length must equal n_genes.")

    types = list(cell_type_adj_mat.index)
    if exclude_cell_types is not None:
        types = [t for t in types if t not in set(exclude_cell_types)]
    K_df = cell_type_adj_mat.loc[types, types]
    K = jnp.asarray(K_df.to_numpy(), dtype=jnp.float32)

    prob_df = estimate_gene_prob_per_type(cm_rna, annot_rna, use_counts=use_counts)
    prob_df.columns = list(gene_names)
    prob_df = prob_df.reindex(types).fillna(0.0)
    prob = jnp.asarray(prob_df.to_numpy(), dtype=jnp.float32)

    expr_frac_mask = jnp.max(prob, axis=0) > min_expr_frac

    diag = jnp.diag(K)
    col_sums = jnp.sum(K, axis=0)
    prior_cont = jnp.where(col_sums > 0, 1.0 - diag / col_sums, 0.0)

    cont_probs_cols: list[np.ndarray] = []
    cont_fracs_null: dict[str, float] = {}
    for u, ct in enumerate(types):
        cont, ref = _gene_scores(prob, K, u)
        p_cu = 0.5 if p_c is None else float(p_c) * float(prior_cont[u])
        gc = _gene_contamination_probs(cont, ref, p_cu)
        gc = jnp.where((gc > signal_thres) & expr_frac_mask, gc, 0.0)
        cont_probs_cols.append(np.asarray(gc))

        ref_native = prob[u]
        denom = jnp.sum(ref_native)
        null = jnp.where(denom > 0, jnp.sum(gc * ref_native) / denom, 0.0)
        cont_fracs_null[ct] = float(null)

    cont_probs = pd.DataFrame(
        np.stack(cont_probs_cols, axis=1),
        index=list(gene_names),
        columns=types,
    )

    cm_csc = cm_spatial.tocsc()
    cell_ids = list(annot_spatial.index)
    type_of_cell = annot_spatial.to_numpy()
    cell_scores: dict[str, float] = {}
    col_sums_cells = np.asarray(cm_csc.sum(axis=0)).ravel()
    for ct in types:
        mask = type_of_cell == ct
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        gprobs = cont_probs[ct].to_numpy()
        sub = cm_csc[:, idx]
        weighted = np.asarray(sub.multiply(gprobs[:, None]).sum(axis=0)).ravel()
        denom = col_sums_cells[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(denom > 0, weighted / denom, 0.0)
        if adjust:
            scores = np.maximum(scores - cont_fracs_null[ct], 0.0)
        for j, cell_idx in enumerate(idx):
            cell_scores[cell_ids[cell_idx]] = float(scores[j])

    fractions = pd.Series(cell_scores, name="admixture_fraction")
    return ContaminationResult(contamination_probs=cont_probs, cell_admixture_fractions=fractions)


def estimate_correlation_preservation(
    cm1: sparse.csr_matrix | np.ndarray,
    cm2: sparse.csr_matrix | np.ndarray,
    gene_names: Sequence[str],
) -> pd.Series:
    """Per-gene correlation between gene-gene correlations of two count matrices.

    Mirrors R ``estimate_correlation_preservation``: useful for verifying that
    contamination-correction did not destroy gene-gene correlation structure.
    """
    from .utils import sparse_corr

    if isinstance(cm1, np.ndarray):
        cm1 = sparse.csr_matrix(cm1)
    if isinstance(cm2, np.ndarray):
        cm2 = sparse.csr_matrix(cm2)
    keep = (np.asarray(cm1.sum(axis=1)).ravel() > 0) & (np.asarray(cm2.sum(axis=1)).ravel() > 0)
    cm1 = cm1[keep]
    cm2 = cm2[keep]
    names = np.asarray(gene_names)[keep]
    cors1 = sparse_corr(cm1)
    cors2 = sparse_corr(cm2)
    np.fill_diagonal(cors1, np.nan)
    np.fill_diagonal(cors2, np.nan)
    out = np.empty(len(names))
    for i in range(len(names)):
        a = cors1[:, i]
        b = cors2[:, i]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 2:
            out[i] = np.nan
        else:
            out[i] = float(np.corrcoef(a[m], b[m])[0, 1])
    return pd.Series(out, index=names, name="correlation_preservation")


__all__ = [
    "ContaminationResult",
    "estimate_cell_adjacency",
    "estimate_cell_type_adjacency",
    "estimate_contamination_scores",
    "estimate_correlation_preservation",
    "estimate_gene_prob_per_type",
]
