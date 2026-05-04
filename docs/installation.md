# Installation

`pycelladmix` is GPU-first: the default install pulls JAX with the CUDA 12 wheel on Linux x86_64. The package is intended for high-performance use on GPU clusters; for small-scale work the [original R package](https://github.com/kharchenkolab/cellAdmix) is recommended.

## Requirements

- Linux x86_64 with a **CUDA 12-capable NVIDIA GPU** (H100 / A100 / RTX 6000 / L40s).
- Python 3.10 – 3.12.

## From PyPI

*Not yet released.*

```bash
pip install pycelladmix
```

## From source

```bash
git clone https://github.com/AlexanderAivazidis/pycelladmix
cd pycelladmix
uv sync
```

## Development install

```bash
uv sync --all-extras
uv run pytest
```
