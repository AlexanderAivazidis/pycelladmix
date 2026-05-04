# Installation

## From PyPI

```bash
# CPU-only (works on any platform):
pip install pycelladmix

# GPU (recommended for real datasets, Linux x86_64 only):
pip install "pycelladmix[gpu]"
```

The default install pulls plain JAX (CPU). Add the `[gpu]` extra to bring in `jax[cuda12]` for GPU acceleration. The package is intended primarily for high-performance use on GPU clusters; for small-scale work the [original R package](https://github.com/kharchenkolab/cellAdmix) is also a fine choice.

## Requirements

- Python 3.10 – 3.12.
- For the `[gpu]` extra: Linux x86_64 with a **CUDA 12-capable NVIDIA GPU** (H100 / A100 / RTX 6000 / L40s).

## From source

```bash
git clone https://github.com/AlexanderAivazidis/pycelladmix
cd pycelladmix
uv sync                 # CPU
uv sync --extra gpu     # GPU
```

## Development install

```bash
uv sync --all-extras
uv run pytest
```
