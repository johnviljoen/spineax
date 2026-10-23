# Spineax (SParse lINear Solvers in JAX)

This repo integrates existing sparse linear solvers into JAX. I currently 
feature a single GPU-based linear solver (with plans to implement more):
- cuDSS

For those that need sparsity pattern detection for jax jacobians/hessians I also offer this [package](https://github.com/johnviljoen/jax2sympy).

I built this repo as part of a project to GPU-batch solve many IPOPT optimizations in [jaxipm](https://github.com/johnviljoen/jaxipm).

## cuDSS

I expose ***most*** features of cuDSS to JAX with ***zero-copy arrays*** and ***full FFI jit/vmap integration*** including custom batching functionality to expose more information than cuDSS currently supports.

This currently supports:
- ***zero-copies between JAX and cuDSS***
- ***full FFI jit/vmap/grad integration*** ([example](examples/cudss/composability.py))
- ***all*** cuDSS ***datatypes*** (F32, F64, C64, C128) ([example](examples/cudss/datatypes.py))
- ***all*** cuDSS ***solvers*** (general, symmetric, symmetric positive defnite, hermitian, hermitian positive definite) ([example](examples/cudss/solver_types.py))
- ***all*** cuDSS ***outputs*** ([example](examples/cudss/outputs.py), even in the batched case!)
- Batches of ***heterogeneous sparsity patterns***, and even ***heterogeneous sizes***! ([example](examples/cudss/heterogeneous_batch.py))

We have also added a ***new Lineax-based API***, which is now the recommended method of interfacing with spineax ([example](examples/cudss/lineax_solver.py)).

# Installation

Requirements:
* An NVIDIA GPU of **Turing generation (compute capability 7.5) or newer**
* **CUDA 13**
* **cuDSS 1.0 or newer**
* **Python 3.12 or newer**
* Linux x86-64 only

pip:
```bash
pip install spineax
```

uv:
```bash
uv pip install spineax
```

> Using a uv-managed project instead? Just `uv add spineax`.


# Citation

```
@article{viljoen2026scaling,
  title={Scaling Nonlinear Optimization: Many Problems One GPU},
  author={Viljoen, John and Haffner, Johanna and Tomizuka, Masayoshi and Mehr, Negar},
  journal={arXiv preprint arXiv:2606.26341},
  year={2026}
}
```



