"""
Example: the canonical token pattern for a Newton/IPM-style loop.

Everything the token API was designed for, in one loop over a BATCH of
symmetric indefinite (KKT) systems held as ONE block-diagonal factorization:

  - analyze ONCE at setup (structure never changes);
  - each iteration: refactorize-or-not as ordinary lax.cond over the token
  - per-block inertia from the one data door (query -> inertia) to drive
    regularization, checked BEFORE paying for a solve;
  - one block SOLVE for the whole batch.

The "problem" is synthetic: each block is a KKT matrix K_i(rho) with a
regularization rho_i, and the loop bumps rho_i wherever the inertia is not
the expected (n_x, n_c) — the standard IPM inertia-correction pattern.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from spineax import cudss


def make_kkt_batch(n_x=60, n_c=20, batch_size=16, seed=0):
    """KKT batch with shared sparsity; some blocks need regularization.

    Returns the shared upper-triangle pattern, a values-assembly function
    values(rho) -> (B, nnz), and the rhs batch.
    """
    rng = np.random.default_rng(seed)
    n = n_x + n_c

    Hs, Js = [], []
    for i in range(batch_size):
        Q, _ = np.linalg.qr(rng.standard_normal((n_x, n_x)))
        if i % 4 == 0:
            # rank-deficient H: needs rho > 0 for correct inertia
            eigs = np.concatenate(
                [np.zeros(n_x // 2), np.linspace(1, 5, n_x - n_x // 2)]
            )
        else:
            eigs = np.linspace(0.5, 5.0, n_x)
        Hs.append(Q @ np.diag(eigs) @ Q.T)
        Js.append(rng.standard_normal((n_c, n_x)))

    def dense_kkt(i, rho):
        K = np.zeros((n, n))
        K[:n_x, :n_x] = Hs[i] + rho * np.eye(n_x)
        K[n_x:, :n_x] = Js[i]
        K[:n_x, n_x:] = Js[i].T
        return K

    # shared pattern: the full dense upper triangle (explicit zeros included,
    # so every block shares one structure regardless of its values)
    mask = np.triu(np.ones((n, n), dtype=bool))
    columns = jnp.asarray(np.nonzero(mask)[1].astype(np.int32))
    offsets = jnp.asarray(
        np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32)
    )

    K0 = np.stack([np.triu(dense_kkt(i, 0.0)) for i in range(batch_size)])
    Keye = np.zeros((n, n))
    Keye[:n_x, :n_x] = np.eye(n_x)
    Keye = np.triu(Keye)

    K0_vals = jnp.asarray(K0[:, mask])  # (B, nnz) at rho = 0
    eye_vals = jnp.asarray(Keye[mask])  # (nnz,) the d/drho direction

    def values(rho):  # rho: (B,) -> (B, nnz)
        return K0_vals + rho[:, None] * eye_vals[None, :]

    b = jnp.asarray(rng.standard_normal((batch_size, n)))
    return offsets, columns, values, b, n_x, n_c


def main():
    batch_size = 16
    offsets, columns, values, b, n_x, n_c = make_kkt_batch(batch_size=batch_size)
    expected = jnp.array([n_x, n_c])  # correct KKT inertia: (n_x pos, n_c neg)

    rho = jnp.zeros(batch_size)
    vals = values(rho)

    # 1. analyze ONCE: one block-diagonal entry for the whole batch
    token = cudss.analyze(vals, offsets, columns, mtype_id=1, mview_id=1)
    # 2. first factorization (fresh pivoting)
    token = cudss.factorize(token, vals)

    factorize_j = jax.jit(cudss.factorize)
    solve_j = jax.jit(cudss.solve)

    print(
        f"batch of {batch_size} KKT systems, n = {n_x + n_c}, expected inertia ({n_x}, {n_c})"
    )
    for it in range(4):
        # 3. per-block inertia BEFORE solving (the one data door)
        inertia = cudss.inertia(cudss.query(token), batch_size=batch_size)
        wrong = jnp.any(inertia != expected[None, :], axis=1)
        n_wrong = int(wrong.sum())
        print(
            f"iter {it}: {n_wrong}/{batch_size} blocks with wrong inertia"
            f"{'' if n_wrong == 0 else ' -> bumping rho and refactorizing'}"
        )

        if n_wrong > 0:
            # 4. inertia correction: bump rho ONLY on the offending blocks.
            #    This is a FACTORIZE, not a refactorize: refactorization
            #    reuses the pivot order chosen for the old (singular) matrix,
            #    which is exactly what regularization is trying to escape.
            #    refactorize is for value drift within one matrix character
            #    (the per-iteration case below); fresh pivoting is the price
            #    of changing the character.
            rho = jnp.where(wrong, jnp.maximum(rho * 10.0, 1e-4), rho)
            vals = values(rho)
            token = factorize_j(token, vals)
            continue

        # 5. inertia is right everywhere: ONE block solve for the whole batch
        x = solve_j(token, b)
        print(
            f"        solved; ||x|| per block in "
            f"[{float(jnp.linalg.norm(x, axis=1).min()):.3f}, "
            f"{float(jnp.linalg.norm(x, axis=1).max()):.3f}]"
        )
        break

    # The same refactorize-or-not decision as traced control flow: a token is
    # a value, so lax.cond can carry it through either branch inside jit.
    @jax.jit
    def step(token, vals, b, needs_refactor):
        token = jax.lax.cond(
            needs_refactor,
            lambda t: cudss.refactorize(t, vals),
            lambda t: t,
            token,
        )
        return cudss.solve(token, b), token

    x2, token = step(token, vals, b, False)  # skip branch: reuse factors
    x3, token = step(token, vals, b, True)  # take branch: fresh refactorization
    print(
        f"lax.cond step: skip-vs-take solution diff {float(jnp.abs(x2 - x3).max()):.2e} "
        "(same values -> same solution)"
    )


if __name__ == "__main__":
    main()
    print("newton_ipm_loop example completed.")
