import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver

def test_composability():

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
    print(f"difference between cudss and cusolver in single solve: {jnp.linalg.norm(test1 - true_x1)}")

    print(f"difference between cudss and cusolver in vmap solve: {jnp.linalg.norm(test2 - jnp.stack([true_x1, true_x2]))}")

test_composability()