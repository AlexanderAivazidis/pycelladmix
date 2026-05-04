"""pycelladmix: identify and correct cell admixtures in imaging-based spatial transcriptomics.

Python port of https://github.com/kharchenkolab/cellAdmix (Mitchel et al. 2025).
"""

from . import annotation, plotting, preprocessing
from .core import CellAdmixResult, run_celladmix
from .crf import run_crf_all
from .nmf import KNNNMFResult, run_knn_nmf
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
    "annotation",
    "estimate_cell_adjacency",
    "estimate_cell_type_adjacency",
    "estimate_contamination_scores",
    "plotting",
    "preprocessing",
    "run_celladmix",
    "run_crf_all",
    "run_knn_nmf",
]
