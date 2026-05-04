# pycelladmix

Python port of [`cellAdmix`](https://github.com/kharchenkolab/cellAdmix) — identify and correct cell admixtures in imaging-based spatial transcriptomics.

```{warning}
**Alpha.** API is stabilising; expect minor breaking changes before 0.1.
```

The pipeline runs in four stages, all numerically heavy ones in **JAX** (GPU-first):

1. **KNN-NMF** — for each cell, build a KNN-aggregated count matrix; stack across cells; factor with rank-k NMF (Frobenius or weighted `ls-nmf`). 30 random restarts run in parallel via `vmap`.
2. **CRF** — assign each molecule to one of the `k` factors using max-product loopy belief propagation over the molecule-KNN graph. Per-cell graphs are unioned into one disconnected graph for a single JIT-compiled BP routine.
3. **Annotation** — marker-enrichment permutation test flags which factors look like admixture from a foreign cell type. (Bridge / membrane tests are on the roadmap.)
4. **Bayesian scoring** — closed-form posterior P(contaminated | gene, cell-type) from an scRNA-seq reference + Delaunay-derived spatial cell-type adjacency, applied to spatial counts to give a per-cell admixture fraction.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
tutorials/01_quickstart
tutorials/02_factor_annotation
tutorials/03_real_data_merfish
api
```

## Acknowledgements

This package is a Python port of the R package `cellAdmix` by Mitchel, Petukhov, Gao, and Kharchenko. All algorithmic credit belongs to the original authors.
