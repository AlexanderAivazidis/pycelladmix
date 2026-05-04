"""Smoke tests: package imports and core dependencies are usable."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro

import pycelladmix


def test_version_is_string() -> None:
    assert isinstance(pycelladmix.__version__, str)
    assert pycelladmix.__version__.count(".") >= 2


def test_jax_runtime() -> None:
    """JAX is installed and can run a trivial JIT'd op."""
    f = jax.jit(lambda x: x * 2.0)
    assert float(f(jnp.array(3.0))) == 6.0


def test_numpyro_imports() -> None:
    """numpyro is installed."""
    assert hasattr(numpyro, "sample")
