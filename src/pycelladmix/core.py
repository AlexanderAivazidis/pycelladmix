"""High-level user-facing API.

Entry point :func:`run_celladmix` orchestrates the full pipeline:
preprocessing → KNN-NMF → CRF smoothing → admixture scoring.

Annotation tests (bridge / membrane / enrichment) are exposed in
:mod:`pycelladmix.annotation`; the orchestrator does not require them by
default since they need either microscopy images (membrane test) or
explicit marker-gene sets (enrichment test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .crf import run_crf_all
from .nmf import KNNNMFResult, run_knn_nmf
from .preprocessing import transcript_counts_matrix
from .scoring import (
    ContaminationResult,
    estimate_cell_adjacency,
    estimate_cell_type_adjacency,
    estimate_contamination_scores,
)
from .utils import validate_transcripts


@dataclass
class CellAdmixResult:
    """Output of :func:`run_celladmix`.

    Attributes
    ----------
    nmf
        Output of the KNN-NMF stage.
    crf
        DataFrame ``mol_id → factor`` from the CRF stage.
    contamination
        Output of the Bayesian admixture scoring stage. ``None`` when no
        scRNA-seq reference is provided.
    cell_type_adjacency
        Cell-type × cell-type adjacency matrix used for scoring. ``None`` if
        scoring was skipped.
    """

    nmf: KNNNMFResult
    crf: pd.DataFrame
    contamination: ContaminationResult | None = None
    cell_type_adjacency: pd.DataFrame | None = None


def run_celladmix(
    df: pd.DataFrame,
    *,
    k: int = 5,
    h_nmf: int = 20,
    h_crf: int = 10,
    nmf_runs: int = 30,
    nmf_iter: int = 200,
    crf_iter: int = 200,
    same_label_ratio: float = 5.0,
    seed: int = 0,
    weighted_nmf: bool = True,
    cm_rna: object | None = None,
    annot_rna: pd.Series | None = None,
    rna_gene_names: list[str] | None = None,
    p_c: float = 0.25,
    signal_thres: float = 0.25,
    min_expr_frac: float = 0.05,
    adjust_scores: bool = True,
) -> CellAdmixResult:
    """End-to-end cellAdmix pipeline.

    Stages:

    1. **KNN-NMF** (:func:`pycelladmix.nmf.run_knn_nmf`) on the transcript
       table. Random restarts run on GPU via JAX.
    2. **CRF** (:func:`pycelladmix.crf.run_crf_all`) MAP-decodes a per-molecule
       factor label via loopy belief propagation.
    3. **Admixture scoring** (:func:`pycelladmix.scoring.estimate_contamination_scores`),
       *if* a reference scRNA-seq counts matrix and annotation are provided.
       Otherwise the CRF labels alone are returned and the user can compute
       scores separately.

    Parameters
    ----------
    df
        Transcript-level dataframe with the columns expected by
        :func:`pycelladmix.utils.validate_transcripts`. A ``celltype`` column
        is required for the scoring stage.
    k, h_nmf, h_crf
        NMF rank, KNN size for NMF aggregation, KNN size for CRF graph.
    nmf_runs, nmf_iter, crf_iter
        Number of NMF random restarts, NMF MU iterations per run, BP
        iterations per CRF run.
    same_label_ratio, seed, weighted_nmf
        Forwarded to the corresponding stages.
    cm_rna, annot_rna, rna_gene_names
        Reference scRNA-seq counts (``(n_genes, n_cells)``), per-cell type
        annotation, and gene names. If any of these is ``None``, scoring is
        skipped.
    p_c, signal_thres, min_expr_frac, adjust_scores
        Forwarded to :func:`pycelladmix.scoring.estimate_contamination_scores`.

    Returns
    -------
    :class:`CellAdmixResult`
    """
    validate_transcripts(df)

    nmf_res = run_knn_nmf(
        df,
        k=k,
        h=h_nmf,
        n_runs=nmf_runs,
        n_iter=nmf_iter,
        seed=seed,
        weighted=weighted_nmf,
    )
    crf_res = run_crf_all(
        df,
        nmf_res,
        num_nn=h_crf,
        same_label_ratio=same_label_ratio,
        n_iter=crf_iter,
    )

    contamination: ContaminationResult | None = None
    cell_type_adjacency: pd.DataFrame | None = None
    if cm_rna is not None and annot_rna is not None and rna_gene_names is not None:
        if "celltype" not in df.columns:
            raise ValueError(
                "Scoring requires a `celltype` column in `df` (per-molecule cell-type label)."
            )
        df_for_adj = df.rename(columns={"celltype": "cell_type"})
        cell_adj = estimate_cell_adjacency(df_for_adj)
        cell_type_adjacency = estimate_cell_type_adjacency(cell_adj)

        cm_spatial, spatial_genes, spatial_cells = transcript_counts_matrix(df)
        common = sorted(set(spatial_genes) & set(rna_gene_names))
        if not common:
            raise ValueError("No common genes between RNA reference and spatial dataset.")
        spatial_idx = np.asarray([list(spatial_genes).index(g) for g in common])
        rna_idx = np.asarray([list(rna_gene_names).index(g) for g in common])
        cm_spatial_sub = cm_spatial[spatial_idx]
        from scipy import sparse as _sp

        cm_rna_sub = _sp.csr_matrix(cm_rna)[rna_idx]

        cell_to_type = df.drop_duplicates("cell").set_index("cell")["celltype"]
        annot_spatial = pd.Series(
            [cell_to_type.get(c, None) for c in spatial_cells], index=list(spatial_cells)
        ).dropna()

        contamination = estimate_contamination_scores(
            cm_rna=cm_rna_sub,
            cm_spatial=cm_spatial_sub,
            annot_rna=annot_rna,
            annot_spatial=annot_spatial,
            cell_type_adj_mat=cell_type_adjacency,
            gene_names=common,
            p_c=p_c,
            signal_thres=signal_thres,
            min_expr_frac=min_expr_frac,
            adjust=adjust_scores,
        )

    return CellAdmixResult(
        nmf=nmf_res,
        crf=crf_res,
        contamination=contamination,
        cell_type_adjacency=cell_type_adjacency,
    )


__all__ = ["CellAdmixResult", "run_celladmix"]
