import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import pytest

from spineax.cudss.solver import CuDSSSolver
from spineax.cudss.solver_re import CuDSSSolverRE


jax.config.update("jax_enable_x64", True)


def _require_gpu():
    if not jax.devices("gpu"):
        pytest.skip("CUDA device required for cuDSS tests")


def _base_system(dtype=jnp.float32):
    M1 = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=dtype,
    )
    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=dtype)
    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0], dtype=dtype)
    true_x1 = jnp.linalg.solve(m1, b1)
    return M1, b1, m1, true_x1


def test_cudss_composability():
    _require_gpu()

    M1, b1, m1, true_x1 = _base_system(jnp.float32)
    M2 = M1 * 0.9
    b2 = b1 * 1.1
    m2 = M2 + M2.T - jnp.diag(M2) * jnp.eye(M2.shape[0], dtype=M2.dtype)
    true_x2 = jnp.linalg.solve(m2, b2)

    LHS1 = jsparse.BCSR.fromdense(M1)
    LHS2 = jsparse.BCSR.fromdense(M2)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data
    csr_offsets2, csr_columns2, csr_values2 = LHS2.indptr, LHS2.indices, LHS2.data

    assert jnp.all(csr_offsets1 == csr_offsets2)
    assert jnp.all(csr_columns1 == csr_columns2)

    csr_values = jnp.vstack([csr_values1, csr_values2])
    b = jnp.vstack([b1, b2])
    device_id = 0
    mtype_id = 1
    mview_id = 1

    solver = CuDSSSolver(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    test1, inertia1 = solver(b[0], csr_values[0])
    test2, inertia2 = jax.jit(jax.vmap(solver))(b, csr_values)

    b_ = jnp.stack([jnp.stack([b, b]), jnp.stack([b, b])])
    csr_values_ = jnp.stack([jnp.stack([csr_values, csr_values]), jnp.stack([csr_values, csr_values])])
    test3, inertia3 = jax.jit(jax.vmap(jax.vmap(jax.vmap(solver))))(b_, csr_values_)

    assert jnp.allclose(test1, true_x1, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(test2, jnp.stack([true_x1, true_x2]), rtol=1e-5, atol=1e-5)

    expected3 = jnp.stack([jnp.stack([jnp.stack([true_x1, true_x2])] * 2)] * 2)
    assert jnp.allclose(test3, expected3, rtol=1e-5, atol=1e-5)

    assert inertia1.shape == (2,)
    assert inertia2.shape == (2, 2)
    assert inertia3.shape == (2, 2, 2, 2)


@pytest.mark.parametrize(
    "dtype",
    [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128],
)
def test_cudss_datatypes(dtype):
    _require_gpu()

    _M1, b1, m1, true_x1 = _base_system(dtype)
    LHS1 = jsparse.BCSR.fromdense(m1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    device_id = 0
    mtype_id = 1
    mview_id = 0
    solver = CuDSSSolver(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    x, inertia = solver(b1, csr_values1)

    assert x.shape == b1.shape
    assert inertia.shape == (2,)
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mtype_id", list(range(5)))
def test_cudss_solver_types(mtype_id):
    _require_gpu()

    _M1, b1, m1, true_x1 = _base_system(jnp.float32)
    LHS1 = jsparse.BCSR.fromdense(m1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    device_id = 0
    mview_id = 0
    solver = CuDSSSolver(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    x, inertia = solver(b1, csr_values1)

    assert x.shape == b1.shape
    assert inertia.shape == (2,)
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)


def test_cudss_outputs():
    _require_gpu()

    M1, b1, _m1, true_x1 = _base_system(jnp.float32)
    LHS1 = jsparse.BCSR.fromdense(M1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    device_id = 0
    mtype_id = 1
    mview_id = 1
    solver = CuDSSSolverRE(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    (
        x,
        lu_nnz,
        npivots,
        inertia,
        perm_reorder_row,
        perm_reorder_col,
        perm_row,
        perm_col,
        perm_matching,
        diag,
        scale_row,
        scale_col,
        elimination_tree,
        nsuperpanels,
        schur_shape,
    ) = solver(b1, csr_values1)

    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)
    assert lu_nnz > 0
    assert npivots >= 0
    assert inertia.shape == (2,)
    assert perm_reorder_row.shape == b1.shape
    assert perm_reorder_col.shape == b1.shape
    assert perm_row.shape == b1.shape
    assert perm_col.shape == b1.shape
    assert perm_matching.shape == b1.shape
    assert diag.shape == b1.shape
    assert scale_row.shape == b1.shape
    assert scale_col.shape == b1.shape
    assert elimination_tree.shape == (1023,)
    assert nsuperpanels.shape == ()
    assert schur_shape.shape == (2,)
