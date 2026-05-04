"""Factor → cell-type annotation: deciding which factors are admixture.

Three complementary tests, each mirroring a function in the R package:

- bridge test (boundary-crossing permutation; ``run_bridge_test``)
- membrane test (microscopy-signal line integral; ``run_memb_test``)
- enrichment test (vs known marker genes; ``get_enr``)

A false-positive filter (``check_f_rm``) reconciles the three before
returning the final removal list.

Permutation tests run all replicates in a single JAX ``vmap`` so they can run
in parallel on GPU.

Status notes
------------
* :func:`run_enrichment_test` is implemented and matches the R semantics
  (column-stochastic factor loadings, per-cell-type permutation null,
  per-factor FDR).
* :func:`run_bridge_test` and :func:`run_membrane_test` are stubs at the moment
  — they require additional inputs (cell-cell adjacency for bridge, microscopy
  images for membrane) that the average user can pre-compute. PRs welcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array, Float
from scipy.stats import false_discovery_control

from .nmf import KNNNMFResult


def _normalise_markers(
    markers: pd.DataFrame | Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Coerce marker input to a long DataFrame with columns gene, marker_of."""
    if isinstance(markers, pd.DataFrame):
        if set(markers.columns) >= {"gene", "marker_of"}:
            return markers[["gene", "marker_of"]].copy()
        raise ValueError("`markers` DataFrame must have columns 'gene' and 'marker_of'.")
    rows: list[dict] = []
    for ct, genes in markers.items():
        for g in genes:
            rows.append({"gene": g, "marker_of": ct})
    return pd.DataFrame(rows)


def _perm_null_for_celltype(
    V: Float[Array, "g k"],
    n_c: int,
    n_perm: int,
    seed: int,
) -> Float[Array, "perm k"]:
    """Permutation null: sum of factor loadings over ``n_c`` random genes, ``n_perm`` times."""
    n_genes = V.shape[0]
    keys = jax.random.split(jax.random.PRNGKey(seed), n_perm)

    def one(key):
        perm = jax.random.permutation(key, n_genes)
        return jnp.sum(V[perm[:n_c]], axis=0)

    return jax.vmap(one)(keys)


def run_enrichment_test(
    res: KNNNMFResult,
    markers: pd.DataFrame | Mapping[str, Sequence[str]],
    n_perm: int = 10000,
    p_thresh: float = 0.1,
    adj_pvals: bool = True,
    seed: int = 0,
    min_markers_per_type: int = 3,
) -> pd.DataFrame:
    """Marker-enrichment permutation test (mirror of R ``get_enr``).

    For each (factor, cell-type) pair, compute the sum of column-normalised
    NMF gene loadings over the cell-type's marker genes, then compare to a
    null distribution built by sampling ``n_c`` random genes from the same
    factor's loadings ``n_perm`` times.

    Parameters
    ----------
    res
        Output of :func:`pycelladmix.nmf.run_knn_nmf`. The test runs on
        ``res.H`` directly; if NMF was fit on an HVG subset, only the
        retained genes contribute (no projection).
    markers
        Either a long DataFrame with columns ``gene``, ``marker_of`` (one row
        per gene-celltype assignment) or a mapping ``celltype → list of genes``.
    n_perm, p_thresh, adj_pvals
        Number of permutations, significance threshold, whether to apply
        Benjamini–Hochberg FDR correction per factor (R default ``TRUE``).
    seed
        Base PRNG seed for permutations.
    min_markers_per_type
        Cell types with fewer than this many markers (after restricting to
        genes present in ``res``) are dropped — too few markers makes the
        permutation null degenerate.

    Returns
    -------
    DataFrame with columns ``factor``, ``cell_type``, ``observed_frac``,
    ``pval``, and (if ``adj_pvals``) ``fdr``. Each row is a (factor, cell-type)
    pair.
    """
    H = np.asarray(res.H)  # (k, g)
    H = H / (H.sum(axis=1, keepdims=True) + 1e-12)
    V = jnp.asarray(H.T, dtype=jnp.float32)  # (g, k)
    gene_to_idx = {g: i for i, g in enumerate(res.gene_names.tolist())}

    markers_df = _normalise_markers(markers)
    markers_df = markers_df[markers_df["gene"].isin(gene_to_idx)]
    counts = markers_df["marker_of"].value_counts()
    keep_types = counts.index[counts >= min_markers_per_type].tolist()
    markers_df = markers_df[markers_df["marker_of"].isin(keep_types)]
    if markers_df.empty:
        raise ValueError(
            f"No cell types retained — need >= {min_markers_per_type} markers "
            "present in the NMF gene set per cell type."
        )

    k = V.shape[1]
    rows: list[dict] = []
    for ti, ct in enumerate(sorted(markers_df["marker_of"].unique())):
        marker_genes = markers_df[markers_df["marker_of"] == ct]["gene"].tolist()
        idx = jnp.asarray([gene_to_idx[g] for g in marker_genes], dtype=jnp.int32)
        observed = jnp.sum(V[idx], axis=0)  # (k,)

        null = _perm_null_for_celltype(V, n_c=len(marker_genes), n_perm=n_perm, seed=seed + ti)
        # P(null > observed) — strict greater-than, R semantics.
        gt = (null > observed[None, :]).mean(axis=0)  # (k,)
        pvals = np.asarray(gt)
        # Avoid zero p-values (R replaces with 1/n_perm).
        pvals = np.where(pvals == 0, 1.0 / n_perm, pvals)

        for f in range(k):
            rows.append(
                {
                    "factor": f + 1,  # 1-indexed to match R
                    "cell_type": ct,
                    "observed_frac": float(observed[f]),
                    "pval": float(pvals[f]),
                }
            )

    df = pd.DataFrame(rows)
    if adj_pvals:
        df["fdr"] = np.nan
        for f in df["factor"].unique():
            sel = df["factor"] == f
            df.loc[sel, "fdr"] = false_discovery_control(df.loc[sel, "pval"].to_numpy())
        df["significant"] = df["fdr"] < p_thresh
    else:
        df["significant"] = df["pval"] < p_thresh
    return df.sort_values(["factor", "cell_type"]).reset_index(drop=True)


def factors_to_remove_per_celltype(
    enrichment_df: pd.DataFrame,
    rule: str = "non_self",
) -> dict[str, list[int]]:
    """Heuristic: from an enrichment-test DataFrame, pick which factors to remove per cell type.

    ``rule="non_self"`` (the only currently implemented rule, matching the R
    convention): for each cell type, return the factors that are *significantly
    enriched in marker genes of OTHER cell types* — those are the ones that
    most likely represent admixed signal coming into this cell type from
    elsewhere. The cell-type's own dominant factor is kept as native expression.
    """
    if rule != "non_self":
        raise ValueError(f"Unknown rule '{rule}'. Only 'non_self' is implemented.")
    sig = enrichment_df[enrichment_df["significant"]].copy()
    # For each factor, the cell type with the strongest enrichment is its "home".
    home = (
        sig.sort_values("observed_frac", ascending=False)
        .drop_duplicates("factor")[["factor", "cell_type"]]
        .rename(columns={"cell_type": "home_celltype"})
    )
    sig = sig.merge(home, on="factor", how="left")
    foreign = sig[sig["cell_type"] != sig["home_celltype"]]
    return {
        ct: sorted(group["factor"].unique().tolist()) for ct, group in foreign.groupby("cell_type")
    }


def run_bridge_test(*args, **kwargs):
    """Boundary-crossing permutation test (R ``run_bridge_test``).

    Not yet implemented — requires per-cell adjacency, a fairly heavy port.
    Tracking issue: if you need this, please open a GitHub issue or PR.
    """
    raise NotImplementedError(
        "run_bridge_test is on the roadmap. For now use run_enrichment_test "
        "or compute admixture scores directly with run_celladmix(...)."
    )


def run_membrane_test(*args, **kwargs):
    """Membrane-signal line-integral test (R ``run_memb_test``).

    Not yet implemented — requires registered microscopy images. Tracking
    issue: if you need this, please open a GitHub issue or PR.
    """
    raise NotImplementedError(
        "run_membrane_test requires registered microscopy images and is on "
        "the roadmap. For now use run_enrichment_test."
    )


def aggregate_to_admix_prior(
    df: pd.DataFrame,
    W_per_molecule: Float[Array, "n k"],
    mol_ids: np.ndarray,
    H: Float[Array, "k g"],
    gene_names: Sequence[str],
    admixture_factors: Mapping[str, Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-molecule factor loadings into a per-cell admixture prior.

    For the Stormi spatial-prior integration. Given:

    * ``W_per_molecule``: soft per-molecule loadings (from
      :func:`pycelladmix.nmf.run_knn_nmf` or
      :func:`pycelladmix.nmf.project_per_molecule_loadings`),
    * ``H``: the factor-by-gene basis (typically frozen from the subsample fit),
    * ``admixture_factors``: a ``cell_type -> list[factor_id]`` mapping (1-indexed
      factor ids, matching :func:`factors_to_remove_per_celltype`),

    this returns:

    1. ``W_admix``: per-cell loading on each admixture factor,
       shape ``(n_cells, k_admix)``. For cell ``c`` of cell-type ``t``, entry
       ``W_admix[c, j]`` is the *sum* over molecules ``i`` in ``c`` of
       ``W_per_molecule[i, factor_idx_of_j]``, but **only if factor ``j`` is
       flagged as admixture for cell-type ``t``**; otherwise zero.
    2. ``H_admix``: the rows of ``H`` corresponding to factors that appear in
       any admixture set, shape ``(k_admix, n_genes)``.

    The product ``W_admix @ H_admix`` gives expected admixture counts per cell
    per gene and can be subtracted from observed counts via
    :func:`pycelladmix.correction.cleaned_counts_matrix`, or fed into Stormi as
    a refinable prior.

    Parameters
    ----------
    df
        Transcript-level dataframe. Must have a ``celltype`` column so we
        know which admixture-factor list applies to each cell.
    W_per_molecule
        ``(n_molecules, k)`` soft per-molecule factor loadings.
    mol_ids
        Molecule ids in row order of ``W_per_molecule``.
    H
        ``(k, n_genes)`` factor-by-gene basis.
    gene_names
        Gene names matching the columns of ``H``.
    admixture_factors
        ``cell_type -> list[int]`` (1-indexed factor ids).

    Returns
    -------
    W_admix : ndarray, ``(n_cells, k_admix)``
    H_admix : ndarray, ``(k_admix, n_genes)``
    cell_names : ndarray of cell ids in row order of ``W_admix``
    gene_names_out : ndarray (== ``np.asarray(gene_names)``, for convenience)
    admix_factor_ids : ndarray of 1-indexed factor ids in column order of
        ``W_admix`` and row order of ``H_admix``
    """
    if "celltype" not in df.columns:
        raise ValueError("`df` must have a `celltype` column for per-cell-type masking.")

    admix_factor_ids = np.asarray(
        sorted({int(f) for fs in admixture_factors.values() for f in fs}), dtype=np.int64
    )
    if admix_factor_ids.size == 0:
        raise ValueError("`admixture_factors` does not contain any factor ids.")
    H_np = np.asarray(H, dtype=np.float64)
    H_admix = H_np[admix_factor_ids - 1]  # convert 1-indexed to 0-indexed

    W_np = np.asarray(W_per_molecule, dtype=np.float32)
    if W_np.shape[0] != len(mol_ids):
        raise ValueError(
            f"W_per_molecule has {W_np.shape[0]} rows but mol_ids has {len(mol_ids)} entries."
        )
    df_idx = df.set_index("mol_id").reindex(np.asarray(mol_ids))
    if df_idx["cell"].isna().any():
        missing = int(df_idx["cell"].isna().sum())
        raise ValueError(
            f"{missing} mol_ids in `mol_ids` are not present in `df`. "
            "Did you pass the right transcript dataframe?"
        )

    cell_of_mol = df_idx["cell"].to_numpy()

    cell_names = np.asarray(sorted(pd.unique(df_idx["cell"])))
    cell_to_idx = {c: i for i, c in enumerate(cell_names.tolist())}
    factor_to_col = {int(f): j for j, f in enumerate(admix_factor_ids.tolist())}

    cell_type = (
        df.drop_duplicates("cell").set_index("cell")["celltype"].reindex(cell_names).to_numpy()
    )

    n_cells = len(cell_names)
    k_admix = admix_factor_ids.size
    W_admix = np.zeros((n_cells, k_admix), dtype=np.float64)

    cell_idx_per_mol = np.asarray([cell_to_idx[c] for c in cell_of_mol])

    for j, factor_id in enumerate(admix_factor_ids.tolist()):
        col_W = W_np[:, int(factor_id) - 1]  # 1-indexed -> 0-indexed
        contrib = np.zeros(n_cells, dtype=np.float64)
        np.add.at(contrib, cell_idx_per_mol, col_W)
        W_admix[:, j] = contrib

    keep_mask = np.zeros((n_cells, k_admix), dtype=bool)
    for t, factor_list in admixture_factors.items():
        rows = np.flatnonzero(cell_type == t)
        if rows.size == 0:
            continue
        cols = np.asarray(
            [factor_to_col[int(f)] for f in factor_list if int(f) in factor_to_col],
            dtype=np.int64,
        )
        if cols.size == 0:
            continue
        keep_mask[np.ix_(rows, cols)] = True
    W_admix = np.where(keep_mask, W_admix, 0.0)

    return (
        W_admix.astype(np.float32),
        H_admix.astype(np.float32),
        cell_names,
        np.asarray(list(gene_names)),
        admix_factor_ids.astype(np.int64),
    )


__all__ = [
    "aggregate_to_admix_prior",
    "factors_to_remove_per_celltype",
    "run_bridge_test",
    "run_enrichment_test",
    "run_membrane_test",
]
