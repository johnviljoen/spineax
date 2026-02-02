"""Example: Solve multiple systems with the same matrix but different right-hand sides."""
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax.cudss.solver import CuDSSSolver

def test_batched_rhs():
    # Single matrix A
    A = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ], dtype=jnp.float32)

    # Batch of right-hand sides (4 different RHS vectors)
    b_batch = jnp.array([
        [7.0, 12.0, 25.0, 4.0, 13.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
    ], dtype=jnp.float32)

    # Symmetrize and get reference solution
    A_sym = A + A.T - jnp.diag(A) * jnp.eye(A.shape[0], dtype=jnp.float32)
    true_x = jax.vmap(lambda b: jnp.linalg.solve(A_sym, b))(b_batch)

    # Convert to CSR
    LHS = jsparse.BCSR.fromdense(A)
    csr_offsets, csr_columns, csr_values = LHS.indptr, LHS.indices, LHS.data

    device_id = 0; mtype_id = 1; mview_id = 1  # symmetric, upper triangular

    # Create solver (matrix structure is static)
    solver = CuDSSSolver(csr_offsets, csr_columns, device_id, mtype_id, mview_id)

    # Solve batch: vmap over b only, csr_values stays the same
    x_batch, inertia_batch = jax.vmap(lambda b: solver(b, csr_values))(b_batch)

    print(f"Batch size: {b_batch.shape[0]}")
    print(f"Solution shape: {x_batch.shape}")
    print(f"Max error vs reference: {jnp.max(jnp.abs(x_batch - true_x)):.2e}")

if __name__ == "__main__":
    test_batched_rhs()
