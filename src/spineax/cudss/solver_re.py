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
        offsets_ptr,
        offsets_size,
        columns_ptr,
        columns_size,
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
        offsets_ptr = offsets_ptr,
        offsets_size = offsets_size,
        columns_ptr = columns_ptr,
        columns_size = columns_size,
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

# single
jax.ffi.register_ffi_target("solve_single_f32_re", single_solve_re.handler_f32(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_f32_re", single_solve_re.type_id_f32(), platform="CUDA")
jax.ffi.register_ffi_target("solve_single_f64_re", single_solve_re.handler_f64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_f64_re", single_solve_re.type_id_f64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_single_c64_re", single_solve_re.handler_c64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_c64_re", single_solve_re.type_id_c64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_single_c128_re", single_solve_re.handler_c128(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_c128_re", single_solve_re.type_id_c128(), platform="CUDA")

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
        offsets_ptr,
        offsets_size,
        columns_ptr,
        columns_size,
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
        "offsets_ptr",
        "offsets_size",
        "columns_ptr",
        "columns_size",
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def solve(
        b_values, 
        csr_values, 
        offsets_ptr,
        offsets_size,
        columns_ptr,
        columns_size,
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
        offsets_ptr = offsets_ptr,
        offsets_size = offsets_size,
        columns_ptr = columns_ptr,
        columns_size = columns_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

# state handling ---------------------------------------------------------------

class CuDSSSolverRE:
    def __init__(self, csr_offsets, csr_columns, device_id, mtype_id, mview_id):
        self._csr_offsets = csr_offsets # hold reference to structures alive in state
        self._csr_columns = csr_columns # hold reference to structures alive in state

        self._solve_fn = ft.partial(solve,
            offsets_ptr=self._csr_offsets.unsafe_buffer_pointer(),
            offsets_size=self._csr_offsets.size,
            columns_ptr=self._csr_columns.unsafe_buffer_pointer(),
            columns_size=self._csr_columns.size,
            device_id=device_id,
            mtype_id=mtype_id,
            mview_id=mview_id
        )

    def __call__(self, b, csr_values):
        return self._solve_fn(b, csr_values)   

if __name__ == "__main__":
    
    import jax.experimental.sparse as jsparse

    # example usage
    # -------------
    M1 = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ])
    M2 = M1 * 0.9

    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0])
    b2 = b1 * 1.1

    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0])
    m2 = M2 + M2.T - jnp.diag(M2) * jnp.eye(M2.shape[0])
    true_x1 = jnp.linalg.solve(m1, b1)
    true_x2 = jnp.linalg.solve(m2, b2)

    LHS1 = jsparse.BCSR.fromdense(M1)
    LHS2 = jsparse.BCSR.fromdense(M2)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data
    csr_offsets2, csr_columns2, csr_values2 = LHS2.indptr, LHS2.indices, LHS2.data

    assert all(csr_offsets1 == csr_offsets2)
    assert all(csr_columns1 == csr_columns2)

    batch_size = 2
    offsets_batch = jnp.vstack([csr_offsets1, csr_offsets2])
    columns_batch = jnp.vstack([csr_columns1, csr_columns2])
    csr_values = jnp.vstack([csr_values1, csr_values2])
    device_id = 0; mtype_id = 1; mview_id = 1
    b = jnp.vstack([b1, b2])

    # instantiate solve
    solver = CuDSSSolverRE(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    x, lu_nnz, npivots, inertia, perm_reorder_row, perm_reorder_col, perm_row, \
    perm_col, perm_matching, diag, scale_row, scale_col, elimination_tree, \
    nsuperpanels, schur_shape = solver(b[0], csr_values[0])

    pass
