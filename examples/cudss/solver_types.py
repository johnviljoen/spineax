"""Example: all five cuDSS matrix types through the token API, plus the
per-factorization cuDSS knobs (reordering algorithm, factor storage)."""

import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np
from spineax import cudss


def test_solver_types(mtype):
    # example usage
    # -------------
    M1 = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )

    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0])

    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0])
    true_x1 = jnp.linalg.solve(m1, b1)

    LHS1 = jsparse.BCSR.fromdense(m1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    # we are passing the whole LHS matrix in FULL (mview_id=0), on GPU 0
    token = cudss.analyze(
        csr_values1,
        csr_offsets1,
        csr_columns1,
        mtype_id=mtype,
        mview_id="full",
        device_id=0,
    )
    token = cudss.factorize(token, csr_values1)
    x = cudss.solve(token, b1)

    # check out the values of the various things!
    print(f"x: {x}")
    print(f"max err vs dense solve: {jnp.max(jnp.abs(x - true_x1)):.2e}")


def _random_symmetric(n=200):
    # large enough that the reordering's effect on factor fill is visible
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    A = A + A.T + n * np.eye(n)
    mask = (np.abs(A) > 1.2) | np.eye(n, dtype=bool)
    A = jnp.asarray(np.where(mask, A, 0.0))
    return A, jsparse.BCSR.fromdense(A), jnp.asarray(rng.standard_normal(n))


def test_reordering_types(reordering):
    # the reordering changes factor fill (lu_nnz from query), not the answer
    A, sp, b = _random_symmetric()
    token = cudss.analyze(
        sp.data, sp.indptr, sp.indices, mtype_id=1, mview_id=0, reordering=reordering
    )
    token = cudss.factorize(token, sp.data)
    x = cudss.solve(token, b)
    lu_nnz = int(cudss.query(token)["lu_nnz"][0])
    print(f"lu_nnz: {lu_nnz}")
    print(f"max err vs dense solve: {jnp.max(jnp.abs(A @ x - b)):.2e}")


def test_memory_types(memory):
    # "hybrid" = host+device factors, for factorizations bigger than VRAM
    A, sp, b = _random_symmetric()
    token = cudss.analyze(
        sp.data, sp.indptr, sp.indices, mtype_id=1, mview_id=0, memory=memory
    )
    token = cudss.factorize(token, sp.data)
    x = cudss.solve(token, b)
    print(f"max err vs dense solve: {jnp.max(jnp.abs(A @ x - b)):.2e}")


mtypes = [
    "general",
    "symmetric",
    "hermitian",
    "symmetric_positive_definite",
    "hermitian_positive_definite",
]

reordering_types = [
    "default",
    "btf_colamd",
    "colamd",
    "amd",
    "nested_dissection",
    "none",
]

memory_types = ["device", "hybrid"]

for mtype in mtypes:
    print(f"testing: {mtype}")
    test_solver_types(mtype)

for reordering in reordering_types:
    print(f"testing reordering: {reordering}")
    test_reordering_types(reordering)

for memory in memory_types:
    print(f"testing memory: {memory}")
    test_memory_types(memory)
