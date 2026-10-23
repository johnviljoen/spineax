"""Example: solve multiple right-hand sides against ONE factorization.

A JAX (B, n) row-major stack of RHS is bit-identical to a cuDSS column-major
(n, B) dense matrix, so the whole batch is ONE cuDSS SOLVE — factor once,
solve many. The same fast path fires under jax.vmap with an unbatched token.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax import cudss


def test_batched_rhs():
    # Single matrix A
    A = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float32,
    )

    # Batch of right-hand sides (4 different RHS vectors)
    b_batch = jnp.array(
        [
            [7.0, 12.0, 25.0, 4.0, 13.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=jnp.float32,
    )

    # Symmetrize and get reference solution
    A_sym = A + A.T - jnp.diag(A) * jnp.eye(A.shape[0], dtype=jnp.float32)
    true_x = jax.vmap(lambda b: jnp.linalg.solve(A_sym, b))(b_batch)

    # Convert to CSR (upper triangle, symmetric view)
    LHS = jsparse.BCSR.fromdense(A)
    csr_offsets, csr_columns, csr_values = LHS.indptr, LHS.indices, LHS.data

    # Factor ONCE
    token = cudss.analyze(csr_values, csr_offsets, csr_columns, mtype_id=1, mview_id=1)
    token = cudss.factorize(token, csr_values)

    # One multi-RHS SOLVE for the whole stack
    x_batch = cudss.solve(token, b_batch)

    # ... and the identical vmap form (same single cuDSS call underneath)
    x_vmap = jax.vmap(lambda b: cudss.solve(token, b))(b_batch)

    print(f"Batch size: {b_batch.shape[0]}")
    print(f"Solution shape: {x_batch.shape}")
    print(f"Max error vs reference: {jnp.max(jnp.abs(x_batch - true_x)):.2e}")
    print(f"Max diff direct vs vmap: {jnp.max(jnp.abs(x_batch - x_vmap)):.2e}")


if __name__ == "__main__":
    test_batched_rhs()
