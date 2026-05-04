"""Factor → cell-type annotation: deciding which factors are admixture.

Three complementary tests, each mirroring a function in the R package:

- bridge test (boundary-crossing permutation; ``run_bridge_test``)
- membrane test (microscopy-signal line integral; ``run_memb_test``)
- enrichment test (vs known marker genes; ``get_enr``)

A false-positive filter (``check_f_rm``) reconciles the three before
returning the final removal list.

Permutation tests run all replicates in a single JAX ``vmap`` on GPU — the
single largest speed-up over the R reference, which loops permutations on CPU.

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


__all__ = [
    "factors_to_remove_per_celltype",
    "run_bridge_test",
    "run_enrichment_test",
    "run_membrane_test",
]
