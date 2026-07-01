"""Persistent cuDSS factorization: factor once, solve many right-hand sides.

Thin JAX wrappers over the ``factor_solve`` native module:

    token = factorize(csr_values, csr_offsets, csr_columns, ...)   # int32[1]
    x     = solve_with(token, b)                                   # reuse factors

``token`` is an ordinary JAX array, so it threads through ``jax.jit`` and a
``custom_vjp`` backward pass; the data dependency it creates makes XLA run every
``solve_with`` after its ``factorize``. For a SYMMETRIC operator ``J = Jᵀ`` the
same token also solves the adjoint (``λ = J⁻ᵀ v = J⁻¹ v``) with no refactorization.

The factorizations live in a process-global LRU cache on the C++ side (capacity
from ``SPINEAX_FACTOR_CACHE``, default 8); a token whose factorization has been
evicted raises at solve time.
"""

import jax
import jax.numpy as jnp

# Initialize the CUDA context before importing the native module (matches solver.py).
jax.devices()

from spineax import factor_solve as _fs   # native nanobind module

# Register the FFI targets (plain handlers, no instantiate/state).
jax.ffi.register_ffi_target(
    "spineax_factorize_f32", _fs.factorize_handler_f32(), platform="CUDA")
jax.ffi.register_ffi_target(
    "spineax_factorize_f64", _fs.factorize_handler_f64(), platform="CUDA")
jax.ffi.register_ffi_target(
    "spineax_solve_with_f32", _fs.solve_handler_f32(), platform="CUDA")
jax.ffi.register_ffi_target(
    "spineax_solve_with_f64", _fs.solve_handler_f64(), platform="CUDA")


from functools import lru_cache


# raw FFI calls ---------------------------------------------------------------
def _ffi_factorize(csr_values, csr_offsets, csr_columns, device_id, mtype_id, mview_id):
    if csr_values.dtype == jnp.float64:
        name = "spineax_factorize_f64"
    elif csr_values.dtype == jnp.float32:
        name = "spineax_factorize_f32"
    else:
        raise ValueError(f"factorize: unsupported dtype {csr_values.dtype}")
    call = jax.ffi.ffi_call(
        name, jax.ShapeDtypeStruct((1,), jnp.int32), has_side_effect=True)
    return call(csr_values, csr_offsets.astype(jnp.int32), csr_columns.astype(jnp.int32),
                device_id=int(device_id), mtype_id=int(mtype_id), mview_id=int(mview_id))


def _ffi_solve(token, b_values, device_id):
    if b_values.dtype == jnp.float64:
        name = "spineax_solve_with_f64"
    elif b_values.dtype == jnp.float32:
        name = "spineax_solve_with_f32"
    else:
        raise ValueError(f"solve_with: unsupported dtype {b_values.dtype}")
    call = jax.ffi.ffi_call(
        name, jax.ShapeDtypeStruct(b_values.shape, b_values.dtype), has_side_effect=True)
    return call(token.astype(jnp.int32), b_values, device_id=int(device_id))


# vmap-aware wrappers ---------------------------------------------------------
# Static config (device/matrix type/view) is baked via a closure so custom_vmap
# only ever sees array arguments.
@lru_cache(maxsize=None)
def _make_factorize(device_id, mtype_id, mview_id):
    @jax.custom_batching.custom_vmap
    def fac(csr_values, csr_offsets, csr_columns):
        return _ffi_factorize(csr_values, csr_offsets, csr_columns,
                              device_id, mtype_id, mview_id)

    @fac.def_vmap
    def _fac_vmap(axis_size, in_batched, csr_values, csr_offsets, csr_columns):
        # Reached only when the MATRIX itself is batched (e.g. design-variable
        # vmap): distinct matrices have distinct factorizations, so there is no
        # sharing — factorize each independently.
        vb, ob, cb = in_batched
        vals = csr_values if vb else jnp.broadcast_to(csr_values, (axis_size,) + csr_values.shape)
        offs = csr_offsets if ob else jnp.broadcast_to(csr_offsets, (axis_size,) + csr_offsets.shape)
        cols = csr_columns if cb else jnp.broadcast_to(csr_columns, (axis_size,) + csr_columns.shape)
        out = jax.lax.map(
            lambda t: _ffi_factorize(t[0], t[1], t[2], device_id, mtype_id, mview_id),
            (vals, offs, cols))
        return out, True

    return fac


@lru_cache(maxsize=None)
def _make_solve(device_id):
    @jax.custom_batching.custom_vmap
    def sol(token, b_values):
        return _ffi_solve(token, b_values, device_id)

    @sol.def_vmap
    def _sol_vmap(axis_size, in_batched, token, b_values):
        tok_b, b_b = in_batched
        b = b_values if b_b else jnp.broadcast_to(b_values, (axis_size,) + b_values.shape)
        if not tok_b:
            # ONE shared factorization, a batch of RHS. A JAX ``(B, n)`` row-major
            # array is bit-identical to a cuDSS column-major ``(n, B)`` dense
            # matrix, so a single multi-RHS SOLVE handles the whole batch with no
            # refactorization — this is the transparent ``jax.vmap`` fast path.
            return _ffi_solve(token, b, device_id), True
        # Batched tokens (distinct factorizations): solve each with its own token.
        out = jax.lax.map(lambda tb: _ffi_solve(tb[0], tb[1], device_id), (token, b))
        return out, True

    return sol


def factorize(csr_values, csr_offsets, csr_columns,
              device_id=0, mtype_id=1, mview_id=1):
    """Factorize a CSR matrix and return a token referencing the cached factors.

    Parameters
    ----------
    csr_values, csr_offsets, csr_columns : CSR triple (offsets/columns int32).
    mtype_id : 0 general, 1 symmetric, 2 hermitian, 3 spd, 4 hpd.
    mview_id : 0 full, 1 upper, 2 lower.

    Returns
    -------
    token : int32[1] JAX array referencing the cached factorization.

    Under ``jax.vmap`` a batched matrix is factorized per batch element (distinct
    factorizations); an *unbatched* matrix is naturally hoisted out of the batch
    by JAX and factorized once — the case that pairs with a batched RHS to give
    transparent factor-once / solve-many.
    """
    return _make_factorize(int(device_id), int(mtype_id), int(mview_id))(
        csr_values, csr_offsets, csr_columns)


def solve_with(token, b_values, device_id=0):
    """Solve ``A x = b`` reusing the factorization referenced by ``token``.

    ``b_values`` is a single RHS ``(n,)``. Under ``jax.vmap`` with an unbatched
    token and a batched RHS, the whole batch is solved by one multi-RHS cuDSS
    SOLVE (no refactorization); with batched tokens it falls back to solving each
    independently.
    """
    return _make_solve(int(device_id))(token, b_values)
