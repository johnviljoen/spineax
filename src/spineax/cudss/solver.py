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
from spineax import single_solve, batch_solve, pbatch_solve

# primitives ===================================================================
# single
solve_single_f32_p = jax.extend.core.Primitive("solve_single_f32")
solve_single_f32_p.multiple_results = True
solve_single_f64_p = jax.extend.core.Primitive("solve_single_f64")
solve_single_f64_p.multiple_results = True
solve_single_c64_p = jax.extend.core.Primitive("solve_single_c64")
solve_single_c64_p.multiple_results = True
solve_single_c128_p = jax.extend.core.Primitive("solve_single_c128")
solve_single_c128_p.multiple_results = True

# batch
solve_batch_f32_p = jax.extend.core.Primitive("solve_batch_f32")
solve_batch_f32_p.multiple_results = True
solve_batch_f64_p = jax.extend.core.Primitive("solve_batch_f64")
solve_batch_f64_p.multiple_results = True
solve_batch_c64_p = jax.extend.core.Primitive("solve_batch_c64")
solve_batch_c64_p.multiple_results = True
solve_batch_c128_p = jax.extend.core.Primitive("solve_batch_c128")
solve_batch_c128_p.multiple_results = True

# pseudo batch
solve_pbatch_f32_p = jax.extend.core.Primitive("solve_pbatch_f32")
solve_pbatch_f32_p.multiple_results = True
solve_pbatch_f64_p = jax.extend.core.Primitive("solve_pbatch_f64")
solve_pbatch_f64_p.multiple_results = True
solve_pbatch_c64_p = jax.extend.core.Primitive("solve_pbatch_c64")
solve_pbatch_c64_p.multiple_results = True
solve_pbatch_c128_p = jax.extend.core.Primitive("solve_pbatch_c128")
solve_pbatch_c128_p.multiple_results = True

# Helper function to compute inertia from diag and perm
def compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim):
    # Reorder diagonal according to permutation
    inv_perm = jnp.argsort(perm)
    diag_original_order = diag[inv_perm]
    out = diag_original_order.reshape([batch_size, matrix_dim])

    # cuDSS pivoting threshold seems to be 1e-13. everything above this on
    # plus or minus side seems to reliably indicate that particular inertia value.
    threshold = 1e-13
    positive = jnp.sum(out > threshold, axis=1)
    negative = jnp.sum(out < -threshold, axis=1)

    return jnp.stack([positive, negative], axis=1, dtype=jnp.int32)

# attempts at figuring out better estimates of inertia from diag and perm
# def compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim, pivot_tol=1e-8, static_pivot_value=1e-8):
#     """Compute inertia (number of positive, negative, zero eigenvalues) from LDL^T factorization.

#     When cuDSS performs LDL^T factorization with static pivoting, huge negative values
#     (< -1e10) can appear in the diagonal for rank-deficient KKT matrices with H≈0.

#     Mathematical basis:
#         For KKT matrices [H A^T; A 0] with H≈0 (zero or tiny regularization):
#         - The huge negatives encode m = number of constraints
#         - Eigenvalue structure follows the saddle-point: m positive, m negative, (n-m) zero
#         - This is mathematically consistent with the rank-deficient structure

#         For standard matrices (no huge negatives):
#         - Use threshold-based counting with threshold = 1e-13 (cuDSS's static pivot value)

#         For PSD H matrices (detected post-hoc):
#         - Static pivots present, standard counting shows zero=0, small next value
#         - Mark as singular with zero=1

#     Args:
#         diag: Diagonal values from LDL^T factorization
#         perm: Permutation array from factorization
#         batch_size: Number of matrices in batch
#         matrix_dim: Dimension of each matrix (n+m for KKT systems)

#     Returns:
#         Array of shape (batch_size, 3) containing [positive, negative, zero] counts
#     """
#     # Reorder diagonal according to permutation
#     inv_perm = jnp.argsort(perm)
#     diag_original_order = diag[inv_perm]
#     out = diag_original_order.reshape([batch_size, matrix_dim])

#     # Count huge negative values in reordered diagonal
#     huge_neg_count = jnp.sum(out < -1e10, axis=1)
    
#     # Use standard threshold
#     threshold = 1e-13
#     positive = jnp.sum(out > threshold, axis=1)
#     negative = jnp.sum(out < -threshold, axis=1)
#     zero = matrix_dim - positive - negative

#     # Correction: For every 2 huge negative values, we need to add 1 to positive
#     # This accounts for the special encoding cuDSS uses
#     correction = (huge_neg_count + 1) // 2  # Round up division

#     positive = positive + correction
#     zero = zero - correction

#     return jnp.stack([positive, negative, zero], axis=1, dtype=jnp.int32)

# implementations ==============================================================
@solve_single_f32_p.def_impl
def solve_single_f32_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_f32", *args, **kwargs)
@solve_single_f64_p.def_impl
def solve_single_f64_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_f64", *args, **kwargs)
@solve_single_c64_p.def_impl
def solve_single_c64_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_c64", *args, **kwargs)
@solve_single_c128_p.def_impl
def solve_single_c128_impl(*args, **kwargs):
    return general_single_solve_impl("solve_single_c128", *args, **kwargs)

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
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # diag
            jax.ShapeDtypeStruct(b_values.shape, jnp.int32),        # perm_reorder_row
        ),
        has_side_effect=True
    )

    x, diag, perm = call(
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

    # Compute inertia instead of returning diag and perm
    batch_size = 1
    matrix_dim = b_values.shape[0]
    inertia = compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim)
    return [x, inertia[0]]  # Return solution and inertia for single batch

@solve_batch_f32_p.def_impl
def solve_batch_f32_impl(*args, **kwargs):
    return general_batch_solve_impl("solve_batch_f32", *args, **kwargs)
@solve_batch_f64_p.def_impl
def solve_batch_f64_impl(*args, **kwargs):
    return general_batch_solve_impl("solve_batch_f64", *args, **kwargs)
@solve_batch_c64_p.def_impl
def solve_batch_c64_impl(*args, **kwargs):
    return general_batch_solve_impl("solve_batch_c64", *args, **kwargs)
@solve_batch_c128_p.def_impl
def solve_batch_c128_impl(*args, **kwargs):
    return general_batch_solve_impl("solve_batch_c128", *args, **kwargs)

def general_batch_solve_impl(
        name, 
        b_values, 
        csr_values, 
        batch_size,
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
            jax.ShapeDtypeStruct((b_values.size,), b_values.dtype),   # diag
            jax.ShapeDtypeStruct((b_values.size,), jnp.int32),        # perm_reorder_row
        ),
        has_side_effect=True
    )

    x, diag, perm = call(
        b_values, 
        csr_values, 
        batch_size = batch_size,
        offsets_ptr = offsets_ptr,
        offsets_size = offsets_size,
        columns_ptr = columns_ptr,
        columns_size = columns_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

    # Compute inertia instead of returning diag and perm
    matrix_dim = b_values.shape[1]  # Assuming b_values shape is (batch_size, n)
    inertia = compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim)
    return [x, inertia]


@solve_pbatch_f32_p.def_impl
def solve_pbatch_f32_impl(*args, **kwargs):
    return general_pbatch_solve_impl("solve_pbatch_f32", *args, **kwargs)
@solve_pbatch_f64_p.def_impl
def solve_pbatch_f64_impl(*args, **kwargs):
    return general_pbatch_solve_impl("solve_pbatch_f64", *args, **kwargs)
@solve_pbatch_c64_p.def_impl
def solve_pbatch_c64_impl(*args, **kwargs):
    return general_pbatch_solve_impl("solve_pbatch_c64", *args, **kwargs)
@solve_pbatch_c128_p.def_impl
def solve_pbatch_c128_impl(*args, **kwargs):
    return general_pbatch_solve_impl("solve_pbatch_c128", *args, **kwargs)

def general_pbatch_solve_impl(
        name, 
        b_values, 
        csr_values, 
        batch_size,
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
            jax.ShapeDtypeStruct((b_values.size,), b_values.dtype),   # diag
            jax.ShapeDtypeStruct((b_values.size,), jnp.int32),        # perm_reorder_row
        ),
        has_side_effect=True
    )

    x, diag, perm = call(
        b_values, 
        csr_values, 
        batch_size = batch_size,
        offsets_ptr = offsets_ptr,
        offsets_size = offsets_size,
        columns_ptr = columns_ptr,
        columns_size = columns_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

    # Compute inertia instead of returning diag and perm
    matrix_dim = b_values.shape[1]  # Assuming b_values shape is (batch_size, n)
    inertia = compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim)
    jax.debug.print("inertia: {}", inertia)
    return [x, inertia]

# registrations and lowerings ==================================================

# single
jax.ffi.register_ffi_target("solve_single_f32", single_solve.handler_f32(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_f32", single_solve.type_id_f32(), platform="CUDA")
jax.ffi.register_ffi_target("solve_single_f64", single_solve.handler_f64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_f64", single_solve.type_id_f64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_single_c64", single_solve.handler_c64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_c64", single_solve.type_id_c64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_single_c128", single_solve.handler_c128(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_single_c128", single_solve.type_id_c128(), platform="CUDA")

solve_single_f32_low = mlir.lower_fun(solve_single_f32_impl, multiple_results=True)
mlir.register_lowering(solve_single_f32_p, solve_single_f32_low)
solve_single_f64_low = mlir.lower_fun(solve_single_f64_impl, multiple_results=True)
mlir.register_lowering(solve_single_f64_p, solve_single_f64_low)
solve_single_c64_low = mlir.lower_fun(solve_single_c64_impl, multiple_results=True)
mlir.register_lowering(solve_single_c64_p, solve_single_c64_low)
solve_single_c128_low = mlir.lower_fun(solve_single_c128_impl, multiple_results=True)
mlir.register_lowering(solve_single_c128_p, solve_single_c128_low)

# batch
jax.ffi.register_ffi_target("solve_batch_f32", batch_solve.handler_f32(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_batch_f32", batch_solve.type_id_f32(), platform="CUDA")
jax.ffi.register_ffi_target("solve_batch_f64", batch_solve.handler_f64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_batch_f64", batch_solve.type_id_f64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_batch_c64", batch_solve.handler_c64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_batch_c64", batch_solve.type_id_c64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_batch_c128", batch_solve.handler_c128(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_batch_c128", batch_solve.type_id_c128(), platform="CUDA")

solve_batch_f32_low = mlir.lower_fun(solve_batch_f32_impl, multiple_results=True)
mlir.register_lowering(solve_batch_f32_p, solve_batch_f32_low)
solve_batch_f64_low = mlir.lower_fun(solve_batch_f64_impl, multiple_results=True)
mlir.register_lowering(solve_batch_f64_p, solve_batch_f64_low)
solve_batch_c64_low = mlir.lower_fun(solve_batch_c64_impl, multiple_results=True)
mlir.register_lowering(solve_batch_c64_p, solve_batch_c64_low)
solve_batch_c128_low = mlir.lower_fun(solve_batch_c128_impl, multiple_results=True)
mlir.register_lowering(solve_batch_c128_p, solve_batch_c128_low)

# psuedo batch
jax.ffi.register_ffi_target("solve_pbatch_f32", pbatch_solve.handler_f32(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_pbatch_f32", pbatch_solve.type_id_f32(), platform="CUDA")
jax.ffi.register_ffi_target("solve_pbatch_f64", pbatch_solve.handler_f64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_pbatch_f64", pbatch_solve.type_id_f64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_pbatch_c64", pbatch_solve.handler_c64(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_pbatch_c64", pbatch_solve.type_id_c64(), platform="CUDA")
jax.ffi.register_ffi_target("solve_pbatch_c128", pbatch_solve.handler_c128(), platform="CUDA")
jax.ffi.register_ffi_type_id("solve_pbatch_c128", pbatch_solve.type_id_c128(), platform="CUDA")

solve_pbatch_f32_low = mlir.lower_fun(solve_pbatch_f32_impl, multiple_results=True)
mlir.register_lowering(solve_pbatch_f32_p, solve_pbatch_f32_low)
solve_pbatch_f64_low = mlir.lower_fun(solve_pbatch_f64_impl, multiple_results=True)
mlir.register_lowering(solve_pbatch_f64_p, solve_pbatch_f64_low)
solve_pbatch_c64_low = mlir.lower_fun(solve_pbatch_c64_impl, multiple_results=True)
mlir.register_lowering(solve_pbatch_c64_p, solve_pbatch_c64_low)
solve_pbatch_c128_low = mlir.lower_fun(solve_pbatch_c128_impl, multiple_results=True)
mlir.register_lowering(solve_pbatch_c128_p, solve_pbatch_c128_low)

# abstract evaluations =========================================================
@solve_single_f32_p.def_abstract_eval
@solve_single_f64_p.def_abstract_eval
@solve_single_c64_p.def_abstract_eval
@solve_single_c128_p.def_abstract_eval
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
            jax.core.ShapedArray(b_values.shape, b_values.dtype),       # x
            jax.core.ShapedArray((2,), jnp.int32),                      # inertia [positive, negative]
        ]

@solve_batch_f32_p.def_abstract_eval
@solve_batch_f64_p.def_abstract_eval
@solve_batch_c64_p.def_abstract_eval
@solve_batch_c128_p.def_abstract_eval
def solve_batch_aval(
        b_values, 
        csr_values, 
        batch_size,
        offsets_ptr,
        offsets_size,
        columns_ptr,
        columns_size,
        device_id, 
        mtype_id, 
        mview_id
    ):
    return [
            jax.core.ShapedArray(b_values.shape, b_values.dtype),       # x
            jax.core.ShapedArray((batch_size, 2), jnp.int32),           # inertia [positive, negative]
        ]

@solve_pbatch_f32_p.def_abstract_eval
@solve_pbatch_f64_p.def_abstract_eval
@solve_pbatch_c64_p.def_abstract_eval
@solve_pbatch_c128_p.def_abstract_eval
def solve_pbatch_aval(
        b_values, 
        csr_values, 
        batch_size,
        offsets_ptr,
        offsets_size,
        columns_ptr,
        columns_size,
        device_id, 
        mtype_id, 
        mview_id
    ):
    return [
            jax.core.ShapedArray(b_values.shape, b_values.dtype),       # x
            jax.core.ShapedArray((batch_size, 2), jnp.int32),           # inertia [positive, negative]
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
        solver = solve_single_f32_p
    elif csr_values.dtype == jnp.float64:
        print(f"solving with float64")
        solver = solve_single_f64_p
    elif csr_values.dtype == jnp.complex64:
        solver = solve_single_c64_p
    elif csr_values.dtype == jnp.complex128:
        solver = solve_single_c128_p
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

# manual batch solve interface =================================================
@ft.partial(
    jax.jit, static_argnames=[
        "batch_size",
        "offsets_ptr",
        "offsets_size",
        "columns_ptr",
        "columns_size",
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def batch_solve(
        b_values, 
        csr_values, 
        batch_size,
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
        solver = solve_batch_f32_p
    elif csr_values.dtype == jnp.float64:
        print(f"solving with float64")
        solver = solve_batch_f64_p
    elif csr_values.dtype == jnp.complex64:
        solver = solve_batch_c64_p
    elif csr_values.dtype == jnp.complex128:
        solver = solve_batch_c128_p
    else:
        raise ValueError(f"Unsupported dtype: {csr_values.dtype}")

    return solver.bind(
        b_values, 
        csr_values, 
        batch_size = batch_size,
        offsets_ptr = offsets_ptr,
        offsets_size = offsets_size,
        columns_ptr = columns_ptr,
        columns_size = columns_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )


# manual psuedo batch solve interface =================================================
@ft.partial(
    jax.jit, static_argnames=[
        "batch_size",
        "offsets_ptr",
        "offsets_size",
        "columns_ptr",
        "columns_size",
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def pbatch_solve(
        b_values, 
        csr_values, 
        batch_size,
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
        solver = solve_pbatch_f32_p
    elif csr_values.dtype == jnp.float64:
        print(f"solving with float64")
        solver = solve_pbatch_f64_p
    elif csr_values.dtype == jnp.complex64:
        solver = solve_pbatch_c64_p
    elif csr_values.dtype == jnp.complex128:
        solver = solve_pbatch_c128_p
    else:
        raise ValueError(f"Unsupported dtype: {csr_values.dtype}")

    return solver.bind(
        b_values, 
        csr_values, 
        batch_size = batch_size,
        offsets_ptr = offsets_ptr,
        offsets_size = offsets_size,
        columns_ptr = columns_ptr,
        columns_size = columns_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

# vmap batch solve interface ===================================================

def solve_single_f32_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_f64_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_c64_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_c128_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)

vmap_using_pseudo_batch = True

def general_solve_vmap(
    vector_arg_values: tuple[Array, Array],     # [b_values, csr_values]
    batch_axes: tuple[int | None, int | None],  # [b_values, csr_values]
    **kwargs                                    # static params
) -> Array:
    
    b_values, csr_values = vector_arg_values
    a_b, a_val = batch_axes

    # guards
    if any(ax is None for ax in (a_val, a_b)):
        raise NotImplementedError("Only batched csr_values and b_values are supported right now")
    
    # the non-batched path
    if a_val is None and a_b is None:
        return solve(*vector_arg_values, **kwargs), (None, None)

    # if only one of the sets of values are batched
    elif a_val or a_b is None:
        raise NotImplementedError("Both csr_values and b_values must be batched")
    
    # the batched path binding
    elif a_val is not None and a_b is not None and vmap_using_pseudo_batch is False:
        if csr_values.dtype == jnp.float32:
            solver = solve_batch_f32_p
        elif csr_values.dtype == jnp.float64:
            solver = solve_batch_f64_p
        elif csr_values.dtype == jnp.complex64:
            solver = solve_batch_c64_p
        elif csr_values.dtype == jnp.complex128:
            solver = solve_batch_c128_p
        else:
            raise ValueError(f"Unsupported dtype: {csr_values.dtype}")
        return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **kwargs), (0,0)
    elif a_val is not None and a_b is not None and vmap_using_pseudo_batch is True:
        if csr_values.dtype == jnp.float32:
            solver = solve_pbatch_f32_p
        elif csr_values.dtype == jnp.float64:
            solver = solve_pbatch_f64_p
        elif csr_values.dtype == jnp.complex64:
            solver = solve_pbatch_c64_p
        elif csr_values.dtype == jnp.complex128:
            solver = solve_pbatch_c128_p
        else:
            raise ValueError(f"Unsupported dtype: {csr_values.dtype}")

        return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **kwargs), (0,0)
    
    else:
        raise NotImplementedError("This path should not be possible")

batching.primitive_batchers[solve_single_f32_p] = solve_single_f32_vmap
batching.primitive_batchers[solve_single_f64_p] = solve_single_f64_vmap
batching.primitive_batchers[solve_single_c64_p] = solve_single_c64_vmap
batching.primitive_batchers[solve_single_c128_p] = solve_single_c128_vmap

# vmap of vmap
def solve_batch_vmap(vector_arg_values, batch_axes, **kwargs):
    """Handle vmap of already-batched solve"""
    b_values, csr_values = vector_arg_values
    a_b, a_val = batch_axes
    
    if a_b is None and a_val is None:
        # Not actually batching
        return batch_solve(*vector_arg_values, **kwargs), (None, None)
    
    # Flatten nested batches
    batch_size1 = b_values.shape[0]
    batch_size2 = b_values.shape[1]
    total_batch = batch_size1 * batch_size2
    
    b_flat = b_values.reshape(total_batch, -1)
    csr_flat = csr_values.reshape(total_batch, -1)
    
    # the non-batched path
    if a_val is None and a_b is None:
        return solve(*vector_arg_values, **kwargs), (None, None)

    # if only one of the sets of values are batched
    elif a_val or a_b is None:
        raise NotImplementedError("Both csr_values and b_values must be batched")
    
    elif a_val is not None and a_b is not None and vmap_using_pseudo_batch is False:
        if csr_values.dtype == jnp.float32:
            solver = solve_batch_f32_p
        elif csr_values.dtype == jnp.float64:
            solver = solve_batch_f64_p
        elif csr_values.dtype == jnp.complex64:
            solver = solve_batch_c64_p
        elif csr_values.dtype == jnp.complex128:
            solver = solve_batch_c128_p
        else:
            raise ValueError(f"Unsupported dtype: {csr_values.dtype}")
        return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **kwargs), (0,0)
    elif a_val is not None and a_b is not None and vmap_using_pseudo_batch is True:
        if csr_values.dtype == jnp.float32:
            solver = solve_pbatch_f32_p
        elif csr_values.dtype == jnp.float64:
            solver = solve_pbatch_f64_p
        elif csr_values.dtype == jnp.complex64:
            solver = solve_pbatch_c64_p
        elif csr_values.dtype == jnp.complex128:
            solver = solve_pbatch_c128_p
        else:
            raise ValueError(f"Unsupported dtype: {csr_values.dtype}")

    # we are replacing this
    kwargs.__delitem__("batch_size")

    x_flat, inertia_flat = solver.bind(
        b_flat, csr_flat, 
        batch_size=total_batch, 
        **kwargs
    )
    
    # Reshape back
    x = x_flat.reshape(batch_size1, batch_size2, -1)
    inertia = inertia_flat.reshape(batch_size1, batch_size2, 2)
    
    return (x, inertia), (0, 0)

batching.primitive_batchers[solve_batch_f32_p] = solve_batch_vmap
batching.primitive_batchers[solve_batch_f64_p] = solve_batch_vmap
batching.primitive_batchers[solve_pbatch_f32_p] = solve_batch_vmap
batching.primitive_batchers[solve_pbatch_f64_p] = solve_batch_vmap

# create python side composable class to ensure validity of the columns and offsets
class CuDSSSolver:
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
    solver = CuDSSSolver(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    # call it - dispatches single solve by default
    test1, in1 = solver(b[0], csr_values[0])

    # call it in vmap/jit
    test2, in2 = jax.jit(jax.vmap(solver))(b, csr_values)

    # unlimited composability in jit/vmap
    b_ = jnp.stack([jnp.stack([b,b]), jnp.stack([b,b])])
    csr_values_ = jnp.stack([jnp.stack([csr_values, csr_values]), jnp.stack([csr_values, csr_values])])
    test3, in3 = jax.jit(jax.vmap(jax.vmap(jax.vmap(solver))))(b_, csr_values_)

    # see difference between dense solves and cuDSS
    pass


