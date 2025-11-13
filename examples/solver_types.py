import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver


def test_datatypes(mtype_id):

    # example usage
    # -------------
    M1 = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ])

    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0])

    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0])
    true_x1 = jnp.linalg.solve(m1, b1)

    LHS1 = jsparse.BCSR.fromdense(m1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    device_id = 0 # run on GPU 0
    mview_id = 0 # we are passing the whole LHS matrix in FULL

    # instantiate solve
    solver = CuDSSSolver(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    x, inertia = solver(b1, csr_values1)

    # check out the values of the various things!
    print(f"x: {x}")
    print(f"inertia: {inertia}")

mtypes = [
    "general",
    "symmetric",
    "hermitian",
    "symmetric_positive_definite",
    "hermitian_positive_definite"
]

for mtype_id, mtype in enumerate(mtypes):
    print(f"testing: {mtype}")
    test_datatypes(mtype_id)