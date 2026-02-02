import functools as ft
import jax
import jax.core
import jax.extend.core
from jax.interpreters import mlir, batching
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import equinox as eqx

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
        csr_offsets,
        csr_columns,
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
        csr_offsets,
        csr_columns,
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
        csr_offsets,
        csr_columns,
        batch_size,
        device_id, 
        mtype_id, 
        mview_id
    ):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # x
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),       # inertia
        ),
        has_side_effect=True
    )

    x, inertia = call(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        batch_size = batch_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

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
        csr_offsets,
        csr_columns,
        batch_size,
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
        csr_offsets,
        csr_columns,
        batch_size = batch_size,
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
register_ffi("solve_single_f32", single_solve, type="f32")
register_ffi("solve_single_f64", single_solve, type="f64")
register_ffi("solve_single_c64", single_solve, type="c64")
register_ffi("solve_single_c128", single_solve, type="c128")

solve_single_f32_low = mlir.lower_fun(solve_single_f32_impl, multiple_results=True)
mlir.register_lowering(solve_single_f32_p, solve_single_f32_low)
solve_single_f64_low = mlir.lower_fun(solve_single_f64_impl, multiple_results=True)
mlir.register_lowering(solve_single_f64_p, solve_single_f64_low)
solve_single_c64_low = mlir.lower_fun(solve_single_c64_impl, multiple_results=True)
mlir.register_lowering(solve_single_c64_p, solve_single_c64_low)
solve_single_c128_low = mlir.lower_fun(solve_single_c128_impl, multiple_results=True)
mlir.register_lowering(solve_single_c128_p, solve_single_c128_low)

# batch
register_ffi("solve_batch_f32", batch_solve, type="f32")
register_ffi("solve_batch_f64", batch_solve, type="f64")
register_ffi("solve_batch_c64", batch_solve, type="c64")
register_ffi("solve_batch_c128", batch_solve, type="c128")

solve_batch_f32_low = mlir.lower_fun(solve_batch_f32_impl, multiple_results=True)
mlir.register_lowering(solve_batch_f32_p, solve_batch_f32_low)
solve_batch_f64_low = mlir.lower_fun(solve_batch_f64_impl, multiple_results=True)
mlir.register_lowering(solve_batch_f64_p, solve_batch_f64_low)
solve_batch_c64_low = mlir.lower_fun(solve_batch_c64_impl, multiple_results=True)
mlir.register_lowering(solve_batch_c64_p, solve_batch_c64_low)
solve_batch_c128_low = mlir.lower_fun(solve_batch_c128_impl, multiple_results=True)
mlir.register_lowering(solve_batch_c128_p, solve_batch_c128_low)

# psuedo batch
register_ffi("solve_pbatch_f32", pbatch_solve, type="f32")
register_ffi("solve_pbatch_f64", pbatch_solve, type="f64")
register_ffi("solve_pbatch_c64", pbatch_solve, type="c64")
register_ffi("solve_pbatch_c128", pbatch_solve, type="c128")

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
        csr_offsets,
        csr_columns,
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
        csr_offsets,
        csr_columns,
        batch_size,
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
        csr_offsets,
        csr_columns,
        batch_size,
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
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def solve(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
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
        csr_offsets,
        csr_columns,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

# manual batch solve interface =================================================
@ft.partial(
    jax.jit, static_argnames=[
        "batch_size",
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def batch_solve(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        batch_size,
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
        csr_offsets,
        csr_columns,
        batch_size = batch_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )


# manual psuedo batch solve interface =================================================
@ft.partial(
    jax.jit, static_argnames=[
        "batch_size",
        "device_id",
        "mtype_id",
        "mview_id"
    ]
)
def pbatch_solve(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        batch_size,
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
        csr_offsets,
        csr_columns,
        batch_size = batch_size,
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

vmap_using_pseudo_batch = False

def general_solve_vmap(
    vector_arg_values: tuple[Array, Array],     # [b_values, csr_values]
    batch_axes: tuple[int | None, int | None],  # [b_values, csr_values]
    **kwargs                                    # static params
) -> Array:

    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes

    # Handle spurious batch axes on sparsity patterns.
    # This happens when the solve is inside a jax.lax.switch that gets vmapped -
    # JAX broadcasts all branch inputs to have batch dimensions, even constants.
    # Since sparsity patterns are the same across all batch elements, extract first.
    if a_off is not None:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=a_off, keepdims=False)
        a_off = None
    if a_col is not None:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=a_col, keepdims=False)
        a_col = None

    # Update vector_arg_values with the corrected sparsity patterns
    vector_arg_values = (b_values, csr_values, csr_offsets, csr_columns)

    # guards (these should never trigger now since we handle spurious batch axes above)
    if any(ax is not None for ax in (a_off, a_col)):
        raise NotImplementedError("don't support batches of heterogeneous sparsity patterns yet (its coming tho...)")

    if all(ax is None for ax in (a_val, a_b)):
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
        return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **kwargs), (0, 0)
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

        return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **kwargs), (0, 0)
    
    else:
        raise NotImplementedError("This path should not be possible")

batching.primitive_batchers[solve_single_f32_p] = solve_single_f32_vmap
batching.primitive_batchers[solve_single_f64_p] = solve_single_f64_vmap
batching.primitive_batchers[solve_single_c64_p] = solve_single_c64_vmap
batching.primitive_batchers[solve_single_c128_p] = solve_single_c128_vmap

# vmap of vmap
def solve_batch_vmap(vector_arg_values, batch_axes, **kwargs):
    """Handle vmap of already-batched solve"""
    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes

    # Handle spurious batch axes on sparsity patterns (same fix as general_solve_vmap)
    if a_off is not None:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=a_off, keepdims=False)
        a_off = None
    if a_col is not None:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=a_col, keepdims=False)
        a_col = None
    vector_arg_values = (b_values, csr_values, csr_offsets, csr_columns)

    if any(ax is not None for ax in (a_off, a_col)):
        raise NotImplementedError("don't support batches of heterogeneous sparsity patterns yet (its coming tho...)")

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
        # Remove batch_size from kwargs if present (happens with nested vmap)
        kwargs = {k: v for k, v in kwargs.items() if k != "batch_size"}
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

    # Remove batch_size from kwargs if present (happens with nested vmap)
    kwargs = {k: v for k, v in kwargs.items() if k != "batch_size"}

    x_flat, inertia_flat = solver.bind(
        b_flat, csr_flat, csr_offsets, csr_columns,
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
class CuDSSSolver(eqx.Module):
    """Sparse linear solver wrapper that marks sparsity pattern as static for vmap."""
    csr_offsets: Array = eqx.field(static=True)
    csr_columns: Array = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)

    def __init__(self, csr_offsets, csr_columns, device_id, mtype_id, mview_id):
        self.csr_offsets = csr_offsets
        self.csr_columns = csr_columns
        self.device_id = device_id
        self.mtype_id = mtype_id
        self.mview_id = mview_id

    def __call__(self, b, csr_values):
        return solve(b, csr_values,
            csr_offsets=self.csr_offsets,
            csr_columns=self.csr_columns,
            device_id=self.device_id,
            mtype_id=self.mtype_id,
            mview_id=self.mview_id
        )   
        1