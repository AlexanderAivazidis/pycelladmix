# Tutorial notebooks

End-to-end walkthroughs of `pycelladmix`. Each notebook is committed in executed form (figures embedded) and is also rendered into the [Sphinx docs](../docs/tutorials/).

1. **[`01_quickstart.ipynb`](01_quickstart.ipynb)** — synthetic 3-cell-type dataset with simulated admixture; runs the full NMF → CRF → Bayesian admixture-score pipeline and verifies the contaminated cells are recovered.
2. **[`02_factor_annotation.ipynb`](02_factor_annotation.ipynb)** — marker-enrichment permutation test for identifying which NMF factors represent admixture, *without* an scRNA-seq reference. Uses synthetic data with two distinct contamination patterns and verifies they are recovered correctly.
3. **[`03_real_data_merfish.ipynb`](03_real_data_merfish.ipynb)** — real MERFISH mouse hypothalamus data (Moffitt et al. 2018, distributed via [`squidpy`](https://squidpy.readthedocs.io/)). Validates that the pipeline recovers the published cell-class structure and flags the authors' "Ambiguous" cells with elevated admixture scores. Within-cell molecule positions are simulated; everything else is real.
4. **[`04_scale_workflow.ipynb`](04_scale_workflow.ipynb)** — subsample-and-project workflow for large datasets (e.g. Xenium 5k × 2M cells). Demonstrates fitting NMF on a 200k-molecule subsample, projecting loadings on the full 1.4M-molecule MERFISH slice, aggregating to a per-cell admixture prior `W_admix`, and producing a cleaned counts matrix ready for downstream Stormi / scanpy.

Notebooks are built programmatically by their `build_tutorial*.py` scripts so the source stays diffable. To regenerate after a code change, run:

```bash
uv run python notebooks/build_tutorial.py        # quickstart
uv run python notebooks/build_tutorial_02.py     # annotation
uv run python notebooks/build_tutorial_03.py     # real merfish
uv run jupyter nbconvert --to notebook --execute notebooks/<file>.ipynb --output <file>.ipynb
```
