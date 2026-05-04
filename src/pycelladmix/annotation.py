"""Factor → cell-type annotation: deciding which factors are admixture.

Three complementary tests, each mirroring a function in the R package:

- bridge test (boundary-crossing permutation; ``run_bridge_test``)
- membrane test (microscopy-signal line integral; ``run_memb_test``)
- enrichment test (vs known marker genes; ``get_enr``)

A false-positive filter (``check_f_rm``) reconciles the three before
returning the final removal list.

Permutation tests run all replicates in a single JAX ``vmap`` on GPU — the
single largest speed-up over the R reference, which loops permutations on CPU.
"""

from __future__ import annotations
