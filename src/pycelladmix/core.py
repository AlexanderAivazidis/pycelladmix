"""High-level user-facing API.

Entry point :func:`run_celladmix` orchestrates the full pipeline:
preprocessing → KNN-NMF → CRF smoothing → factor annotation → admixture scoring.
"""

from __future__ import annotations
