# pycelladmix

Python port of [`cellAdmix`](https://github.com/kharchenkolab/cellAdmix): identify and correct cell admixtures in imaging-based spatial transcriptomics data.

> **Independent community port — not endorsed by the original `cellAdmix` authors.** This is a separate code base maintained by Alexander Aivazidis. The original R package by Mitchel, Petukhov, Gao, and Kharchenko remains the reference implementation; please cite their paper. If you have a question about the *algorithm*, the R package and its authors are the authoritative source. Issues with this Python port go to [this repository's issue tracker](https://github.com/AlexanderAivazidis/pycelladmix/issues).

> **Status: alpha.** API is stabilising; expect minor breaking changes before 0.2.

## What it does

In imaging-based spatial transcriptomics (Xenium, MERSCOPE, CosMx, …), each detected molecule is assigned to a cell based on a cell-segmentation mask. Segmentation errors cause molecules to be assigned to the wrong cell — **admixture** — biasing downstream cell-type calls and differential-expression analyses.

`cellAdmix` factorises spatial neighbourhoods of molecules with NMF, smooths factor assignments with a CRF, and tests which factors represent admixture (vs. native expression) using boundary-crossing, membrane-signal, and marker-enrichment statistics. Per-cell admixture scores are then computed against an scRNA-seq reference and a spatial cell-type adjacency graph.

`pycelladmix` is a Python re-implementation built around `AnnData` for compatibility with the [scverse](https://scverse.org) ecosystem (`scanpy`, `squidpy`, `spatialdata`). It targets **large-scale, high-performance use** on GPU clusters: NMF factorisation, loopy belief propagation for CRF inference, and permutation tests are all written in **JAX** and **numpyro**, JIT-compiled, and `vmap`-ed across cells / restarts / permutations. On a single H100/A100 it is typically 1–2 orders of magnitude faster than the R reference.

For small datasets where speed is not a concern, the [original R package](https://github.com/kharchenkolab/cellAdmix) remains the reference implementation and is recommended.

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

*Coming soon — see the tutorials once the first release lands.*

## Credits

This package is a Python port of the R package [`cellAdmix`](https://github.com/kharchenkolab/cellAdmix) by Mitchel, Petukhov, Gao, and Kharchenko. If you use `pycelladmix`, please cite the original paper:

> Mitchel J., Petukhov V., Gao T., Kharchenko P. *cellAdmix: identifying and correcting cell admixtures in spatial transcriptomics.* bioRxiv 2025.

The original R implementation is MIT-licensed; this port follows the same license. All algorithmic credit belongs to the original authors; bugs and Pythonisms are mine.

## License

MIT — see [LICENSE](LICENSE).
