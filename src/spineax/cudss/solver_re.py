"""
solver_re stands for Return Everything
"""

import functools as ft
import jax
import jax.core
import jax.extend.core
from jax.interpreters import mlir, batching
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np

# Force JAX to initialize CUDA context BEFORE importing my C++ functions!!!!!!!!
jax.devices()

# Import the functions that return pointers from our compiled C++
from spineax import single_solve_re # , batch_solve_re, pbatch_solve_re

# primitives ===================================================================
# single
solve_single_f32_re_p = jax.extend.core.Primitive("solve_single_f32_re")
solve_single_f32_re_p.multiple_results = True
solve_single_f64_re_p = jax.extend.core.Primitive("solve_single_f64_re")
solve_single_f64_re_p.multiple_results = True
solve_single_c64_re_p = jax.extend.core.Primitive("solve_single_c64_re")
solve_single_c64_re_p.multiple_results = True
solve_single_c128_re_p = jax.extend.core.Primitive("solve_single_c128_re")
solve_single_c128_re_p.multiple_results = True


# implementations ==============================================================
@solve_single_f32_re_p.def_impl
def solve_single_f32_re_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_f32_re", *args, **kwargs)
@solve_single_f64_re_p.def_impl
def solve_single_f64_re_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_f64_re", *args, **kwargs)
@solve_single_c64_re_p.def_impl
def solve_single_c64_re_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_c64_re", *args, **kwargs)
@solve_single_c128_re_p.def_impl
def solve_single_c128_re_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_c128_re", *args, **kwargs)

def general_single_solve_impl(
        name, 
        b_values, 
        csr_values, 
        offsets,
        columns,
        device_id, 
        mtype_id, 
        mview_id
    ):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # x
            jax.ShapeDtypeStruct((), jnp.int64), # lu_nnz
            jax.ShapeDtypeStruct((), jnp.int32), # npivots
            jax.ShapeDtypeStruct((2,), jnp.int32), # inertia
            jax.ShapeDtypeStruct(b_values.shape, jnp.int32), # perm_reorder_row
            jax.ShapeDtypeStruct(b_values.shape, jnp.int32), # perm_reorder_col
            jax.ShapeDtypeStruct(b_values.shape, jnp.int32), # perm_row
            jax.ShapeDtypeStruct(b_values.shape, jnp.int32), # perm_col
            jax.ShapeDtypeStruct(b_values.shape, jnp.int32), # perm_matching
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # diag
            jax.ShapeDtypeStruct(b_values.shape, jnp.float32), # scaled row
            jax.ShapeDtypeStruct(b_values.shape, jnp.float32), # scaled col
            jax.ShapeDtypeStruct((1023,), jnp.int32),   # elimination tree
            jax.ShapeDtypeStruct((), jnp.int32),   # nsuperpanels
            jax.ShapeDtypeStruct((2,), jnp.int64),   # schur shape
        ),
        has_side_effect=True
    )

    out = call(
        b_values, 
        csr_values, 
        offsets,
        columns,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

    return out

# # Compute inertia instead of returning diag and perm
# batch_size = 1
# matrix_dim = b_values.shape[0]
# inertia = compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim)
# return [x, inertia[0]]  # Return solution and inertia for single batch

# registrations and lowerings ==================================================

try:
    from jax._src.lib import jaxlib_extension_version
    _NEW_FFI_API = jaxlib_extension_version >= 381
except ImportError:
    _NEW_FFI_API = False


def register_ffi(name: str, func, *, type: str, platform: str = "CUDA"):
    handler = getattr(func, f"handler_{type}")()
    state_dict = getattr(func, f"state_dict_{type}")()
    type_id = getattr(func, f"type_id_{type}")()
    if _NEW_FFI_API:
        jax.ffi.register_ffi_type(name, state_dict, platform=platform)
    else:
        jax.ffi.register_ffi_type_id(name, type_id, platform=platform)
    # order matters, ffi_target needs to be registered after type
    jax.ffi.register_ffi_target(name, handler, platform=platform)

# single
register_ffi("solve_single_f32_re", single_solve_re, type="f32")
register_ffi("solve_single_f64_re", single_solve_re, type="f64")
register_ffi("solve_single_c64_re", single_solve_re, type="c64")
register_ffi("solve_single_c128_re", single_solve_re, type="c128")

solve_single_f32_re_low = mlir.lower_fun(solve_single_f32_re_impl, multiple_results=True)
mlir.register_lowering(solve_single_f32_re_p, solve_single_f32_re_low)
solve_single_f64_re_low = mlir.lower_fun(solve_single_f64_re_impl, multiple_results=True)
mlir.register_lowering(solve_single_f64_re_p, solve_single_f64_re_low)
solve_single_c64_re_low = mlir.lower_fun(solve_single_c64_re_impl, multiple_results=True)
mlir.register_lowering(solve_single_c64_re_p, solve_single_c64_re_low)
solve_single_c128_re_low = mlir.lower_fun(solve_single_c128_re_impl, multiple_results=True)
mlir.register_lowering(solve_single_c128_re_p, solve_single_c128_re_low)


# abstract evaluations =========================================================
@solve_single_f32_re_p.def_abstract_eval
@solve_single_f64_re_p.def_abstract_eval
@solve_single_c64_re_p.def_abstract_eval
@solve_single_c128_re_p.def_abstract_eval
def solve_aval(
        b_values, 
        csr_values, 
        offsets,
        columns,
        device_id, 
        mtype_id, 
        mview_id
    ):
    return [
            jax.core.ShapedArray(b_values.shape, b_values.dtype),   # x
            jax.core.ShapedArray((), jnp.int64), # lu_nnz
            jax.core.ShapedArray((), jnp.int32), # npivots
            jax.core.ShapedArray((2,), jnp.int32), # inertia
            jax.core.ShapedArray(b_values.shape, jnp.int32), # perm_reorder_row
            jax.core.ShapedArray(b_values.shape, jnp.int32), # perm_reorder_col
            jax.core.ShapedArray(b_values.shape, jnp.int32), # perm_row
            jax.core.ShapedArray(b_values.shape, jnp.int32), # perm_col
            jax.core.ShapedArray(b_values.shape, jnp.int32), # perm_matching
            jax.core.ShapedArray(b_values.shape, b_values.dtype),   # diag
            jax.core.ShapedArray(b_values.shape, jnp.float32), # scaled row
            jax.core.ShapedArray(b_values.shape, jnp.float32), # scaled col
            jax.core.ShapedArray((1023,), jnp.int32),   # elimination tree
            jax.core.ShapedArray((), jnp.int32),   # nsuperpanels
            jax.core.ShapedArray((2,), jnp.int64),   # schur shape
        ]


# single solve interface =======================================================
@ft.partial(
    jax.jit, static_argnames=[
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def solve(
        b_values, 
        csr_values, 
        offsets,
        columns,
        device_id, 
        mtype_id, 
        mview_id
    ):
    if csr_values.dtype == jnp.float32:
        print(f"solving with float32")
        solver = solve_single_f32_re_p
    elif csr_values.dtype == jnp.float64:
        print(f"solving with float64")
        solver = solve_single_f64_re_p
    elif csr_values.dtype == jnp.complex64:
        solver = solve_single_c64_re_p
    elif csr_values.dtype == jnp.complex128:
        solver = solve_single_c128_re_p
    else:
        raise ValueError(f"Unsupported dtype: {csr_values.dtype}")

    return solver.bind(
        b_values, 
        csr_values, 
        offsets,
        columns,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

# state handling ---------------------------------------------------------------

class CuDSSSolverRE:
    def __init__(self, csr_offsets, csr_columns, device_id, mtype_id, mview_id):

        self._solve_fn = ft.partial(solve,
            offsets=csr_offsets,
            columns=csr_columns,
            device_id=device_id,
            mtype_id=mtype_id,
            mview_id=mview_id
        )

    def __call__(self, b, csr_values):
        return self._solve_fn(b, csr_values)   