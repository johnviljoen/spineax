# Spineax (SParse lINear Solvers in JAX)

This repo integrates existing sparse linear solvers into JAX. I currently 
feature a single GPU-based linear solver (with plans to implement more - starting with amgx):
- cuDSS

For those that need sparsity pattern detection for jax jacobians/hessians I also offer this [(in progress) package](https://github.com/johnviljoen/jax2sympy).

> NOTE: This is part of an ongoing project to create a batch solved IPOPT clone in JAX that we will publish in the coming days/weeks (I am open sourcing this package ahead of time due to significant interest in this solver being integrated into JAX, to allow others to get started on their own research projects with it!).

> Do not hesitate to reach out if you need modifications to this for your own research work - I would love to collaborate :).

## cuDSS

I expose ***most*** features of cuDSS (as of 0.7.0) to JAX with ***zero-copy arrays*** and ***full FFI jit/vmap integration*** with ***XLA state management*** including custom batching functionality to expose more information than cuDSS currently supports.

This currently supports:
- ✅ ***zero-copies between JAX and cuDSS***
- ✅ ***full FFI jit/vmap integration***
- ✅ ***all*** cuDSS datatypes (F32, F64, C64, C128)
- ✅ ***all*** cuDSS solvers (general, symmetric, symmetric positive defnite, hermitian, hermitian positive definite)
- ✅ ***all*** cuDSS outputs ([see example](examples/cudss/outputs.py))
- ✅ upper/lower triangular and full sparse matrix definitions

This currently lacks:

- ❌ Differentiation through cuDSS solvers is not currently supported (fairly easy to implement if people need it)
- ❌ Does not support full retrieval of all auxillary information from batched system (a cuDSS limitation as of 0.7.0)
- ❌ vmap over heterogeneous sparsity patterns in batch is currently not supported (the C++ code is written, but the clean vmap implementation is not)

Caveats:

- Currently on the first call to the solve function we perform METIS reordering, analysis, factorization, and solve. Then on subsequent calls we perform only warm refactorization and solve. If there is demand for only a solve or only a refactorization I can support individual calls for these components later.

### Examples

* [JAX transformation composability](examples/cudss/composability.py)
* [Testing all available datatypes](examples/cudss/datatypes.py)
* [Seeing all cuDSS auxilliary outputs](examples/cudss/outputs.py)
* [Testing all available solvers (general, sym, herm, spd, hpd)](examples/cudss/solver_types.py)


# Installation

Requirements:
* For cuDSS support an NVIDIA GPU of Pascal generation and newer is required
* Only linux is currently supported
* conda (recommended) or pip venv virtual environment. ***WARNING: UV is currently failing***.

```bash
conda create -n spineax pip
conda activate spineax

# for cuda 13
pip install "scikit-build-core>=0.5" nanobind "jax[cuda13]>=0.8.0" nvidia-cudss-cu13
pip install --no-build-isolation "spineax[cuda13] @ git+ssh://git@github.com/johnviljoen/spineax.git"

# for cuda 12
pip install "scikit-build-core>=0.5" nanobind "jax[cuda12]>=0.5.0" nvidia-cudss-cu12
pip install --no-build-isolation "spineax[cuda12] @ git+ssh://git@github.com/johnviljoen/spineax.git"
```

## Troubleshooting

### Build fails with `__cudaGetKernel was not declared`

This happens when your system's CUDA toolkit version doesn't match the pip-installed CUDA runtime.
You can skip building the optional `pbatch_solve` module (builds by default):

```bash
# CUDA 12
pip install "scikit-build-core>=0.5" nanobind "jax[cuda12]>=0.5.0" nvidia-cudss-cu12
pip install --no-build-isolation --config-settings=cmake.args="-DBUILD_PBATCH_SOLVE=OFF" .[cuda12]

# CUDA 13
pip install "scikit-build-core>=0.5" nanobind "jax[cuda13]>=0.8.0" nvidia-cudss-cu13
pip install --no-build-isolation --config-settings=cmake.args="-DBUILD_PBATCH_SOLVE=OFF" .[cuda13]

# Citation

TODO



