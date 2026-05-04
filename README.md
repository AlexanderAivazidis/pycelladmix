# pycelladmix

Python port of [`cellAdmix`](https://github.com/kharchenkolab/cellAdmix): identify and correct cell admixtures in imaging-based spatial transcriptomics data.

> **Independent community port — not endorsed by the original `cellAdmix` authors yet.** This is a separate code base maintained by Alexander Aivazidis. The original R package by Mitchel, Petukhov, Gao, and Kharchenko remains the reference implementation; please cite their paper. If you have a question about the *algorithm*, the R package and its authors are the authoritative source. Issues with this Python port go to [this repository's issue tracker](https://github.com/AlexanderAivazidis/pycelladmix/issues).

> **Status: alpha.** API is stabilising; expect minor breaking changes before 0.2.

## What it does

In imaging-based spatial transcriptomics (Xenium, MERSCOPE, CosMx, …), each detected molecule is assigned to a cell based on a cell-segmentation mask. Segmentation errors cause molecules to be assigned to the wrong cell — **admixture** — biasing downstream cell-type calls and differential-expression analyses.

`cellAdmix` factorises spatial neighbourhoods of molecules with NMF, smooths factor assignments with a CRF, and tests which factors represent admixture (vs. native expression) using boundary-crossing, membrane-signal, and marker-enrichment statistics. Per-cell admixture scores are then computed against an scRNA-seq reference and a spatial cell-type adjacency graph.

`pycelladmix` is a Python re-implementation built around `AnnData` for compatibility with the [scverse](https://scverse.org) ecosystem (`scanpy`, `squidpy`, `spatialdata`). The numerically heavy stages (NMF factorisation, loopy belief propagation for CRF inference, permutation tests) are written in **JAX** and **numpyro**, JIT-compiled, and `vmap`-ed across cells / restarts / permutations so they can run on GPU. Performance has not been benchmarked against the R reference; head-to-head numbers are pending.

The [original R package](https://github.com/kharchenkolab/cellAdmix) is the reference implementation — please cite their paper and use it if you want the canonical behaviour.

## Installation

```bash
# CPU-only (works anywhere; fine for small datasets and trying it out):
pip install pycelladmix

# GPU (recommended for real datasets — pulls jax[cuda12], Linux x86_64 only):
pip install "pycelladmix[gpu]"

# from source with uv:
git clone https://github.com/AlexanderAivazidis/pycelladmix
cd pycelladmix
uv sync                 # CPU
uv sync --extra gpu     # GPU
```

## Quickstart

```python
import pycelladmix

# `df` is a per-molecule dataframe with columns: x, y, z, gene, cell, celltype, mol_id
# Reference scRNA-seq counts (genes × cells), per-cell type annotation, and gene names
# are optional — without them the pipeline runs NMF + CRF only and the user can compute
# scores separately.

res = pycelladmix.run_celladmix(
    df,
    k=8,                    # NMF rank
    h_nmf=20,               # KNN neighbourhood size for NMF aggregation
    h_crf=10,               # KNN neighbourhood size for the CRF graph
    cm_rna=cm_rna,          # optional: scRNA-seq reference counts
    annot_rna=annot_rna,    # optional: cell id → cell type
    rna_gene_names=genes,   # optional: gene names for cm_rna
)

# Results:
res.nmf.W                                       # (n_molecules, k) factor loadings
res.nmf.H                                       # (k, n_genes)    gene loadings
res.crf                                         # mol_id → factor (CRF MAP labels)
res.contamination.cell_admixture_fractions      # per-cell admixture score
res.contamination.contamination_probs           # per-gene per-cell-type P(contamination)
```

Three executed end-to-end tutorials are bundled, including a validation on real MERFISH data:

- [`notebooks/01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) — synthetic 3-cell-type data with simulated admixture; verifies the contaminated cells are recovered.
- [`notebooks/02_factor_annotation.ipynb`](notebooks/02_factor_annotation.ipynb) — marker-enrichment permutation test for identifying admixture factors *without* an scRNA-seq reference.
- [`notebooks/03_real_data_merfish.ipynb`](notebooks/03_real_data_merfish.ipynb) — real MERFISH mouse hypothalamus data (Moffitt et al. 2018, fetched via `squidpy.datasets.merfish`); validates that the unsupervised pipeline recovers the published cell-class structure.

## Credits

This package is a Python port of the R package [`cellAdmix`](https://github.com/kharchenkolab/cellAdmix) by Mitchel, Petukhov, Gao, and Kharchenko. If you use `pycelladmix`, please cite the original paper:

> Mitchel J., Petukhov V., Gao T., Kharchenko P. *cellAdmix: identifying and correcting cell admixtures in spatial transcriptomics.* bioRxiv 2025.

The original R implementation is MIT-licensed; this port follows the same license. All algorithmic credit belongs to the original authors; bugs and Pythonisms are mine.

## License

MIT — see [LICENSE](LICENSE).
