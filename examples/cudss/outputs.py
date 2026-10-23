"""Example: read EVERYTHING cuDSS knows about a factorization via query().

query(token) is the one data door: every cuDSS data item is returned
unconditionally (zero-filled where cuDSS declines for this matrix type /
config), and you take what you need — e.g. inertia(data) for the per-block
[positive, negative] eigenvalue counts.
"""

import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax import cudss


def test_outputs():
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

    LHS1 = jsparse.BCSR.fromdense(M1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    # upper triangular view of a symmetric matrix
    token = cudss.analyze(
        csr_values1, csr_offsets1, csr_columns1, mtype_id=1, mview_id=1
    )
    token = cudss.factorize(token, csr_values1)
    x = cudss.solve(token, b1)

    data = cudss.query(token)

    # check out the values of the various things!
    print(f"x: {x} (max err vs dense: {jnp.max(jnp.abs(x - true_x1)):.2e})")
    print(f"lu_nnz (Number of non-zero entries in LU factors): {data['lu_nnz']}")
    print(
        f"npivots (Number of pivots encountered during factorization): {data['npivots']}"
    )
    print(
        f"inertia (cuDSS's own positive/negative indices of inertia for the system matrix A (two integer values), random behaviour if zero eigenvalues present): {data['inertia']}"
    )
    print(
        f"perm_reorder_row (Row permutation P after reordering such that A[P,Q] is factorized): {data['perm_reorder_row']}"
    )
    print(
        f"perm_reorder_col (Column permutation Q after reordering such that A[P,Q] is factorized): {data['perm_reorder_col']}"
    )
    print(
        f"perm_row (Final row permutation P (includes effects of both reordering and pivoting) which is applied to the original right-hand side of the system in the form b_new = b_old * P) (only supported with alg 1,2 used for reordering): {data['perm_row']}"
    )
    print(
        f"perm_col (Final column permutation Q (includes effects of both reordering and pivoting) which is applied to transform the solution of the permuted system into the original solution x_old = x_new * Q^-1) (only supported with alg 1,2 used for reordering): {data['perm_col']}"
    )
    print(
        f"perm_matching (Matching (column) permutation Q such that A[:,Q] is reordered and then factorized) (requires matching to be enabled): {data['perm_matching']}"
    )
    print(f"diag (Diagonal of the factorized matrix): {data['diag']}")
    print(
        f"scale_row (Row scaling the factorized matrix (corresponding to the rows of the original matrix)) (requires matching to be enabled): {data['scale_row']}"
    )
    print(
        f"scale_col (Column scaling the factorized matrix (corresponding to the columns of the original matrix)) (requires matching to be enabled): {data['scale_col']}"
    )
    print(
        f"nd_partition_tree (cuDSS >= 0.8 successor to the elimination tree): {data['nd_partition_tree'][:16]}..."
    )
    print(
        f"nsuperpanels (Number of superpanels in the matrix) (enabled by default): {data['nsuperpanels']}"
    )
    print(f"schur_shape (disabled by default): {data['schur_shape']}")

    # spineax's per-block LDL^T inertia, sign counts of the block-aligned diag
    print(f"inertia (spineax, per-block from diag): {cudss.inertia(data)}")


test_outputs()
