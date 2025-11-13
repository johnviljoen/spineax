import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver_re import CuDSSSolverRE

def test_outputs():

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

    LHS1 = jsparse.BCSR.fromdense(M1)
    csr_offsets1, csr_columns1, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    device_id = 0; mtype_id = 1; mview_id = 1

    # instantiate solve
    solver = CuDSSSolverRE(csr_offsets1, csr_columns1, device_id, mtype_id, mview_id)

    x, lu_nnz, npivots, inertia, perm_reorder_row, perm_reorder_col, perm_row, \
    perm_col, perm_matching, diag, scale_row, scale_col, elimination_tree, \
    nsuperpanels, schur_shape = solver(b1, csr_values1)

    # check out the values of the various things!
    print(f"x: {x}")
    print(f"lu_nnz (Number of non-zero entries in LU factors): {lu_nnz}")
    print(f"npivots (Number of pivots encountered during factorization): {npivots}")
    print(f"inertia (Positive and negative indices of inertia for the system matrix A (two integer values), random behaviour if zero eigenvalues present): {inertia}")
    print(f"perm_reorder_row (Row permutation P after reordering such that A[P,Q] is factorized): {perm_reorder_row}")
    print(f"perm_reorder_col (Column permutation Q after reordering such that A[P,Q] is factorized): {perm_reorder_col}")
    print(f"perm_row (Final row permutation P (includes effects of both reordering and pivoting) which is applied to the original right-hand side of the system in the form b_new = b_old * P) (only supported with alg 1,2 used for reordering): {perm_row}")
    print(f"perm_col (Final column permutation Q (includes effects of both reordering and pivoting) which is applied to transform the solution of the permuted system into the original solution x_old = x_new * Q^-1) (only supported with alg 1,2 used for reordering): {perm_col}")
    print(f"perm_matching (Matching (column) permutation Q such that A[:,Q] is reordered and then factorized) (requires matching to be enabled): {perm_matching}")
    print(f"diag (Diagonal of the factorized matrix): {diag}")
    print(f"scale_row (Row scaling the factorized matrix (corresponding to the rows of the original matrix)) (requires matching to be enabled): {scale_row}")
    print(f"scale_col (Column scaling the factorized matrix (corresponding to the columns of the original matrix)) (requires matching to be enabled): {scale_col}")
    print(f"elimination_tree (enabled always): {elimination_tree}")
    print(f"nsuperpanels (Number of superpanels in the matrix) (enabled by default): {nsuperpanels}")
    print(f"schur_shape (disabled by default): {schur_shape}")

test_outputs()