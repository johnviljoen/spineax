"""Example: spineax's default user API — the lineax solver.

``CSROperator`` + ``CuDSS`` (both part of
``spineax.cudss``, defined in solver.py next to the token machinery they
wrap) give two interoperating styles:

- lineax protocol: ``lx.linear_solve(operator, b, solver)`` — following
  lineax's own convention (``lx.Cholesky``/``lx.LU`` factorize in ``init``
  and only substitute in ``compute``), init = analyze + factorize and
  compute = solve, with the factorized ``FactorToken`` as the state. Many
  right-hand sides against one factorization reuse lineax's ``state=``
  argument, exactly as with the built-in solvers.
- explicit phases: because ``FactorToken`` is a pytree, each cuDSS phase is
  a token-threading method on the solver — the "true control" tier for
  IPM/Newton loops where the analyze/factorize/refactorize boundaries are
  yours, plus ``query`` for every cuDSS data item (inertia included).
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import lineax as lx

from spineax import cudss
from spineax.cudss import CSROperator, CuDSS

jax.config.update("jax_enable_x64", True)


def test_lineax_solver():
    key = jax.random.PRNGKey(0)
    n = 50
    # banded symmetric positive definite test matrix, full CSR pattern
    dense = jnp.zeros((n, n), dtype=jnp.float64)
    dense = dense.at[jnp.arange(n), jnp.arange(n)].set(4.0)
    dense = dense.at[jnp.arange(n - 1), jnp.arange(1, n)].set(-1.0)
    dense = dense.at[jnp.arange(1, n), jnp.arange(n - 1)].set(-1.0)
    sp = jsparse.BCSR.fromdense(dense)
    b = jax.random.normal(key, (n,), dtype=jnp.float64)

    operator = CSROperator(sp.data, sp.indptr, sp.indices, lx.symmetric_tag)
    solver = CuDSS()

    # explicit phase style: each cuDSS phase is a method call, and the
    # analyze/factorize boundary is yours
    token = solver.analyze(operator)  # ANALYSIS (once)
    token = solver.factorize(token, operator)  # FACTORIZATION
    x = solver.solve(token, b)  # SOLVE
    err = jnp.max(jnp.abs(operator.mv(x) - b))
    print(f"explicit phases:           max |Ax-b| = {err:.2e}")
    assert err < 1e-12

    # query: every cuDSS data item for this factorization in one dict —
    # derive what you need from it, e.g. the inertia check an IPM runs
    # before paying for a solve
    data = solver.query(token)
    inr = cudss.inertia(data)
    print(
        f"query -> inertia:          [pos, neg] = {inr} "
        f"(lu_nnz {int(data['lu_nnz'][0])}, npivots {int(data['npivots'][0])})"
    )
    assert inr[0] == n and inr[1] == 0  # positive definite

    # one-shot lineax style: lx.linear_solve runs init (analyze+factorize)
    # + compute (solve) itself on the SAME solver object. Each un-stated
    # call mints a fresh registry entry (lineax is designed to re-init every
    # time) whose factors occupy device memory — release it via the token
    # that lx.Solution hands back, so one-shot solves cannot evict your
    # long-lived tokens from the LRU. (Outside jit only; under jit the LRU
    # is the backstop.)
    sol = lx.linear_solve(operator, b, solver)
    err = jnp.max(jnp.abs(operator.mv(sol.value) - b))
    cudss.release(sol.state)
    print(
        f"one-shot linear_solve:     max |Ax-b| = {err:.2e} "
        f"(entry released, registry: {cudss.registry_size()})"
    )
    assert err < 1e-12

    # the styles interoperate: the factorized token IS lineax's state, so
    # it slots into state= — one factorization, many right-hand sides
    for i in range(3):
        bi = jax.random.normal(jax.random.PRNGKey(i), (n,), dtype=jnp.float64)
        xi = lx.linear_solve(operator, bi, solver, state=token).value
        err = jnp.max(jnp.abs(operator.mv(xi) - bi))
        print(f"state-reuse solve rhs {i}:   max |Ax-b| = {err:.2e}")
        assert err < 1e-12

    # Newton-style value update: refactorize reuses the pivot order and the
    # analysis from the explicit token — no re-analyze, no new registry entry
    new_values = sp.data * 2.0
    new_operator = CSROperator(new_values, sp.indptr, sp.indices, lx.symmetric_tag)
    token = solver.refactorize(token, new_operator)
    x2 = solver.solve(token, b)
    err = jnp.max(jnp.abs(new_operator.mv(x2) - b))
    print(f"after value update:        max |Ax-b| = {err:.2e}")
    assert err < 1e-12

    # autodiff: lineax differentiates through linear_solve; the backward
    # pass solves with the transposed state — the same token, since the
    # matrix is symmetric (this is the custom_vjp adjoint-reuse pattern).
    # init runs inside the trace here, so each call mints one entry that
    # cannot be released eagerly — the LRU evicts them as they age out.
    @jax.jit
    def loss(values):
        op = CSROperator(values, sp.indptr, sp.indices, lx.symmetric_tag)
        x = lx.linear_solve(op, b, solver).value
        return jnp.sum(x**2)

    grad_sparse = jax.grad(loss)(sp.data)

    def dense_loss(A):
        return jnp.sum(jnp.linalg.solve(A, b) ** 2)

    grad_dense_full = jax.grad(dense_loss)(dense)
    rows = jnp.repeat(jnp.arange(n), jnp.diff(sp.indptr))
    err = jnp.max(jnp.abs(grad_sparse - grad_dense_full[rows, sp.indices]))
    print(f"grad through linear_solve: max err vs dense = {err:.2e}")
    assert err < 1e-10

    print("lineax example completed.")


test_lineax_solver()
