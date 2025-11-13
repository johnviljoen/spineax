# Spineax (SParse lINear Solvers in JAX)

This repo integrates existing sparse linear solvers into JAX. I currently 
feature a single GPU-based linear solver (with plans to implement more, including CPU-based ones in the future):
- cuDSS

For those that need sparsity pattern detection for jax jacobians/hessians I also offer this [(in progress) package](https://github.com/johnviljoen/jax2sympy).

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
pip install git+ssh://git@github.com/johnviljoen/spineax.git
```





