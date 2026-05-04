"""pycelladmix: identify and correct cell admixtures in imaging-based spatial transcriptomics.

Python port of https://github.com/kharchenkolab/cellAdmix (Mitchel et al. 2025).
"""

from . import annotation, correction, plotting, preprocessing
from .annotation import aggregate_to_admix_prior
from .core import CellAdmixResult, run_celladmix
from .correction import cleaned_counts_matrix
from .crf import run_crf_all
from .nmf import KNNNMFResult, project_per_molecule_loadings, run_knn_nmf
from .scoring import (
    ContaminationResult,
    estimate_cell_adjacency,
    estimate_cell_type_adjacency,
    estimate_contamination_scores,
)

__version__ = "0.1.0"

__all__ = [
    "CellAdmixResult",
    "ContaminationResult",
    "KNNNMFResult",
    "__version__",
    "aggregate_to_admix_prior",
    "annotation",
    "cleaned_counts_matrix",
    "correction",
    "estimate_cell_adjacency",
    "estimate_cell_type_adjacency",
    "estimate_contamination_scores",
    "plotting",
    "preprocessing",
    "project_per_molecule_loadings",
    "run_celladmix",
    "run_crf_all",
    "run_knn_nmf",
]
