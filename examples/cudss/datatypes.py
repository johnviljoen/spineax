"""Example: the API supports f32/f64/c64/c128."""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax import cudss

jax.config.update("jax_enable_x64", True)


def test_datatypes(dtype):
    # example usage
    # -------------
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

    LHS1 = jsparse.BCSR.fromdense(m1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    # full symmetric matrix passed in, full view
    token = cudss.analyze(
        csr_values1, csr_offsets1, csr_columns1, mtype_id=1, mview_id=0
    )
    token = cudss.factorize(token, csr_values1)
    x = cudss.solve(token, b1)
    inertia = cudss.inertia(cudss.query(token))

    # check out the values of the various things!
    print(f"x: {x}")
    print(f"inertia: {inertia}")
    print(f"max err vs dense solve: {jnp.max(jnp.abs(x - true_x1)):.2e}")


dtypes = [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]

for dtype in dtypes:
    test_datatypes(dtype)
