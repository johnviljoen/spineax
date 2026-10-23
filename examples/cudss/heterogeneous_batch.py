"""Example: heterogeneous batches — different sparsity patterns, and even
different SIZES, in one factorization.

spineax's batching model is literally "a batch of systems IS one bigger
block-diagonal sparse system", and nothing requires the blocks to be alike:

- different PATTERNS, same size: stack per-block offsets/columns to
  (B, n+1)/(B, nnz) — through the explicit batch door or vmap — and the
  whole batch is analyzed as ONE system. query/inertia still split per
  block.
- different SIZES: assemble the block-diagonal CSR yourself (a few lines of
  numpy) and mint ONE single-system token for it — one analyze, one
  factorize, one solve for the whole collection, sliced at the block
  boundaries. Per-block inertia still works, because cuDSS returns the
  LDL^T diagonal in INPUT order.
"""

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from spineax import cudss

rng = np.random.default_rng(7)


# helpers ----------------------------------------------------------------------
def dense_of_upper(values, offsets, columns, n):
    """Dense symmetric matrix from an upper-triangle CSR (mview_id=1)."""
    U = np.zeros((n, n))
    for i in range(n):
        for k in range(int(offsets[i]), int(offsets[i + 1])):
            U[i, int(columns[k])] = values[k]
    return U + U.T - np.diag(np.diag(U))


def tridiag_upper(n):
    """Tridiagonal SPD: diag 4, superdiag 1. Upper nnz = 2n - 1."""
    offsets = np.concatenate([np.arange(0, 2 * (n - 1), 2), [2 * n - 2, 2 * n - 1]])
    columns = np.concatenate(
        [np.stack([np.arange(n - 1), np.arange(1, n)], 1).ravel(), [n - 1]]
    )
    values = np.concatenate([np.tile([4.0, 1.0], n - 1), [4.0]])
    return values, offsets, columns


def arrow_upper(n):
    """Arrowhead SPD: dense first row, diagonal rest. Upper nnz = 2n - 1."""
    offsets = np.concatenate([[0], np.arange(n, 2 * n)])
    columns = np.concatenate([np.arange(n), np.arange(1, n)])
    values = np.concatenate([[float(n)], np.ones(n - 1), np.full(n - 1, 2.0)])
    return values, offsets, columns


def indefinite_upper(n):
    """Small dense symmetric INDEFINITE block (for the inertia demo)."""
    A = rng.standard_normal((n, n))
    A = A + A.T
    mask = np.triu(np.ones((n, n), bool))
    offsets = np.concatenate([[0], np.cumsum(mask.sum(1))])
    return A[mask], offsets, np.nonzero(mask)[1]


def test_heterogeneous_patterns():
    """Same size, different structure: a tridiagonal and an arrowhead system
    (both upper nnz = 2n-1) stacked into one (B, ...) explicit batch."""
    print("=" * 70)
    print("TEST 1: one batch, two different sparsity patterns")
    print("=" * 70)
    n = 40
    systems = [tridiag_upper(n), arrow_upper(n)]

    vals = jnp.asarray(np.stack([s[0] for s in systems]))
    offs = jnp.asarray(np.stack([s[1] for s in systems]), jnp.int32)
    cols = jnp.asarray(np.stack([s[2] for s in systems]), jnp.int32)
    b = jnp.asarray(rng.standard_normal((2, n)))

    token = cudss.analyze(vals, offs, cols, mtype_id=1, mview_id=1)
    token = cudss.factorize(token, vals)
    x = cudss.solve(token, b)
    inertia = cudss.inertia(cudss.query(token), batch_size=2)

    for i, name in enumerate(["tridiagonal", "arrowhead  "]):
        A = dense_of_upper(*systems[i], n)
        err = np.linalg.norm(A @ np.asarray(x[i]) - np.asarray(b[i]))
        print(f"  {name}  residual = {err:.2e}   inertia = {np.asarray(inertia[i])}")
        assert err < 1e-10
    cudss.release(token)
    print()


def block_diag_csr(blocks):
    """Assemble [(values, offsets, columns), ...] of sizes n_i into the CSR
    of the one block-diagonal system. Returns block boundaries too."""
    ns = [len(b[1]) - 1 for b in blocks]
    bounds = np.concatenate([[0], np.cumsum(ns)])
    values = np.concatenate([b[0] for b in blocks])
    columns = np.concatenate([np.asarray(b[2]) + lo for b, lo in zip(blocks, bounds)])
    offsets = np.concatenate(
        [[0], np.cumsum(np.concatenate([np.diff(b[1]) for b in blocks]))]
    )
    return values, offsets, columns, bounds


def test_heterogeneous_sizes():
    """Different sizes: n=40 tridiagonal + n=25 arrowhead + n=8 indefinite
    dense, factored and solved as ONE system."""
    print("=" * 70)
    print("TEST 2: one factorization, three different SIZES (40, 25, 8)")
    print("=" * 70)
    blocks = [tridiag_upper(40), arrow_upper(25), indefinite_upper(8)]
    values, offsets, columns, bounds = block_diag_csr(blocks)
    b = rng.standard_normal(bounds[-1])

    token = cudss.analyze(
        jnp.asarray(values),
        jnp.asarray(offsets, jnp.int32),
        jnp.asarray(columns, jnp.int32),
        mtype_id=1,
        mview_id=1,
    )
    token = cudss.factorize(token, jnp.asarray(values))
    x = np.asarray(cudss.solve(token, jnp.asarray(b)))

    # per-block inertia at heterogeneous sizes: slice the INPUT-ORDERED
    # LDL^T diagonal at the block boundaries and count signs
    diag = np.asarray(cudss.query(token)["diag"])

    for i, name in enumerate(["tridiag n=40", "arrow   n=25", "indef   n=8 "]):
        lo, hi = bounds[i], bounds[i + 1]
        A = dense_of_upper(*blocks[i], hi - lo)
        err = np.linalg.norm(A @ x[lo:hi] - b[lo:hi])
        d = diag[lo:hi]
        inr = [int((d > 1e-13).sum()), int((d < -1e-13).sum())]
        eigs = np.linalg.eigvalsh(A)
        expected = [int((eigs > 0).sum()), int((eigs < 0).sum())]
        print(f"  {name}  residual = {err:.2e}   inertia = {inr} (expected {expected})")
        assert err < 1e-10 and inr == expected
    cudss.release(token)
    print()


if __name__ == "__main__":
    test_heterogeneous_patterns()
    test_heterogeneous_sizes()
    print("heterogeneous batch example completed.")
