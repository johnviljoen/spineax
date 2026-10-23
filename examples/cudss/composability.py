"""Example: the API composes with jit, vmap, and grad.

A batch under vmap is ONE block-diagonal cuDSS system: vmap(analyze) mints a
single registry entry and every downstream phase is one batched call. vmap
composes: nested vmap peels one axis per level into the same ONE
block-diagonal system — a (2, 2, 2) batch IS the flattened 8-batch — and
autodiff-added axes peel through the same rules, so grad-of-grad under vmap
is just more solves against the same factors.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from spineax import cudss


def test_composability():
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
    csr_offsets, csr_columns, csr_values1 = LHS1.indptr, LHS1.indices, LHS1.data

    def token_solve(values, b):
        token = cudss.analyze(values, csr_offsets, csr_columns, mtype_id=1, mview_id=1)
        token = cudss.factorize(token, values)
        return cudss.solve(token, b)

    # single solve, eager
    x1 = token_solve(csr_values1, b1)

    # jit + 3rd-order vmap: a (2, 2, 2) batch of scaled systems, each vmap
    # level peeling one axis into the same ONE block-diagonal factorization
    scales = 1.0 + 0.1 * jnp.arange(8.0, dtype=M1.dtype).reshape(2, 2, 2)
    vals = scales[..., None] * csr_values1
    b = jnp.broadcast_to(b1, (2, 2, 2) + b1.shape)
    x = jax.jit(jax.vmap(jax.vmap(jax.vmap(token_solve))))(vals, b)
    # (s*A)^-1 b = A^-1 b / s: the reference needs no per-block solve
    true_x = true_x1 / scales[..., None]

    print(
        f"difference between cudss and cusolver in single solve: {jnp.linalg.norm(x1 - true_x1)}"
    )
    print(
        f"difference between cudss and cusolver in (2,2,2) nested-vmap solve: {jnp.linalg.norm(x - true_x)}"
    )

    assert jnp.allclose(x1, true_x1, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(x, true_x, rtol=1e-4, atol=1e-4)

    # vmap(grad(grad(...))): loss(s) = sum((s*A)^-1 b) = sum(A^-1 b) / s has
    # the analytic second derivative 2 * sum(A^-1 b) / s^3.
    # Reverse-over-reverse differentiation reduces to extra solves against
    # the same factors, and vmap over s batches them block-diagonally.
    def loss(s):
        return jnp.sum(token_solve(s * csr_values1, b1))

    s = jnp.array([0.5, 1.0, 2.0, 4.0], dtype=M1.dtype)
    d2 = jax.jit(jax.vmap(jax.grad(jax.grad(loss))))(s)
    d2_true = 2.0 * jnp.sum(true_x1) / s**3

    print(
        f"vmap(grad(grad(loss))) max err vs analytic: {jnp.max(jnp.abs(d2 - d2_true))}"
    )
    assert jnp.allclose(d2, d2_true, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_composability()
    print("composability example completed.")
