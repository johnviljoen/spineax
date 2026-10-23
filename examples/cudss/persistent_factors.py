"""Example: persistent factors — factor once, solve many, refactor, self-heal.

Every token id names ONE immutable numeric state: ``refactorize`` returns a
fresh id, and using a superseded (or LRU-evicted, or released) state
transparently rebuilds its factors from the token's own CSR arrays — every
phase, ``query`` included. ``rebuild_count()`` counts the heals — a rising
count means SPINEAX_FACTOR_CACHE is too small.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import lineax as lx

from spineax import cudss
from spineax.cudss import CSROperator, CuDSS

jax.config.update("jax_enable_x64", True)


def test_persistent_factors():
    n = 50
    dense = jnp.zeros((n, n), dtype=jnp.float64)
    dense = dense.at[jnp.arange(n), jnp.arange(n)].set(4.0)
    dense = dense.at[jnp.arange(n - 1), jnp.arange(1, n)].set(-1.0)
    dense = dense.at[jnp.arange(1, n), jnp.arange(n - 1)].set(-1.0)
    sp = jsparse.BCSR.fromdense(dense)
    op = CSROperator(sp.data, sp.indptr, sp.indices, lx.symmetric_tag)
    solver = CuDSS()

    # factor once (analyze + factorize), solve many against the state
    state = solver.init(op, {})
    for seed in range(3):
        b = jax.random.normal(jax.random.PRNGKey(seed), (n,), jnp.float64)
        x = lx.linear_solve(op, b, solver, state=state).value
        assert jnp.linalg.norm(dense @ x - b) < 1e-12

    # refactor: new values, reused pivot order, FRESH id — the old state
    # is superseded, not overwritten
    new_op = CSROperator(2.0 * sp.data, sp.indptr, sp.indices, lx.symmetric_tag)
    new_state = solver.refactorize(state, new_op)
    b = jnp.ones((n,), jnp.float64)
    x2 = lx.linear_solve(new_op, b, solver, state=new_state).value
    assert jnp.linalg.norm(2.0 * dense @ x2 - b) < 1e-12

    # the old state still answers for ITS OWN matrix: first use rebuilds
    # (one analyze + factorize), later uses hit the healed entry
    r0 = cudss.rebuild_count()
    x1 = lx.linear_solve(op, b, solver, state=state).value
    assert jnp.linalg.norm(dense @ x1 - b) < 1e-12
    assert cudss.rebuild_count() == r0 + 1
    lx.linear_solve(op, b, solver, state=state)
    assert cudss.rebuild_count() == r0 + 1

    print("persistent factors: solve-many, refactor, self-heal OK")


if __name__ == "__main__":
    test_persistent_factors()
