import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver
jax.config.update("jax_enable_x64", True)

def test_datatypes(dtype):

    # example usage
    # -------------
    M1 = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ], dtype=dtype)

    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=dtype)

    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0], dtype=dtype)
    true_x1 = jnp.linalg.solve(m1, b1)

    LHS1 = jsparse.BCSR.fromdense(M1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    device_id = 0; mtype_id = 1; mview_id = 1

    # instantiate solve
    solver = CuDSSSolver(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    x, inertia = solver(b1, csr_values1)

    # check out the values of the various things!
    print(f"x: {x}")
    print(f"inertia: {inertia}")

dtypes = [
    jnp.float32,
    jnp.float64,
    jnp.complex64,
    jnp.complex128
]

for dtype in dtypes:
    test_datatypes(dtype)
