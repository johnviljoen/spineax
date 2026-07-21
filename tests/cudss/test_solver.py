"""Tests for the token-based factorization API (docs/token_design.md)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from spineax import cudss


def _require_gpu():
    if not jax.devices("gpu"):
        pytest.skip("CUDA device required for cuDSS tests")


def _sym_system(n=50, dtype=jnp.float64, seed=0, shift=None):
    """Random symmetric matrix (upper-triangle CSR) + dense reference.

    With shift=None the matrix is diagonally-dominant PD; a scalar shift
    controls the spectrum for indefinite tests.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A = A + A.T + (n if shift is None else shift) * np.eye(n)
    upper = np.triu(A)
    # CSR of the dense upper triangle: offsets/columns of explicit entries
    mask = np.triu(np.ones((n, n), dtype=bool))
    columns = np.nonzero(mask)[1].astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32)
    values = upper[mask]
    return (
        jnp.asarray(values, dtype=dtype),
        jnp.asarray(offsets),
        jnp.asarray(columns),
        jnp.asarray(A, dtype=dtype),
    )


def _general_complex_system(n=30, dtype=jnp.complex128, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = A + n * np.eye(n)
    columns = np.tile(np.arange(n, dtype=np.int32), n)
    offsets = (np.arange(n + 1, dtype=np.int32) * n)
    values = A.reshape(-1)
    return (
        jnp.asarray(values, dtype=dtype),
        jnp.asarray(offsets),
        jnp.asarray(columns),
        jnp.asarray(A, dtype=dtype),
    )


_TOL = {
    jnp.float32: 1e-4,
    jnp.float64: 1e-10,
    jnp.complex64: 1e-3,
    jnp.complex128: 1e-10,
}


def _rel_err(A, x, b):
    A, x, b = np.asarray(A), np.asarray(x), np.asarray(b)
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)


# correctness ==================================================================
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_analyze_factorize_solve(dtype):
    _require_gpu()
    values, offsets, columns, A = _sym_system(dtype=dtype)
    b = jnp.asarray(np.random.default_rng(2).standard_normal(A.shape[0]), dtype=dtype)

    token = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
    assert token.kind == "single"
    assert token.n == A.shape[0]

    token = cudss.factorize(token, values)
    inertia = cudss.inertia(cudss.query(token))
    np.testing.assert_array_equal(np.asarray(inertia), [A.shape[0], 0])  # PD

    x = cudss.solve(token, b)
    assert _rel_err(A, x, b) < _TOL[dtype]


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_complex_general(dtype):
    _require_gpu()
    values, offsets, columns, A = _general_complex_system(dtype=dtype)
    b = jnp.asarray(
        np.random.default_rng(3).standard_normal(A.shape[0])
        + 1j * np.random.default_rng(4).standard_normal(A.shape[0]),
        dtype=dtype,
    )
    token = cudss.analyze(values, offsets, columns, mtype_id=0, mview_id=0)
    token = cudss.factorize(token, values)
    x = cudss.solve(token, b)
    assert _rel_err(A, x, b) < _TOL[dtype]


def test_jit_full_chain():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(5).standard_normal(A.shape[0]))

    @jax.jit
    def full(values, b):
        t = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, values)
        return cudss.solve(t, b), cudss.inertia(cudss.query(t))

    x, inertia = full(values, b)
    assert _rel_err(A, x, b) < 1e-10
    np.testing.assert_array_equal(np.asarray(inertia), [A.shape[0], 0])


def test_explicit_typed_ffi_api_version():
    """Every emitted spineax custom call states api_version=4 EXPLICITLY.

    Regression test for the jaxipm-scale compile failure: jax.ffi.ffi_call
    lowers typed-FFI calls as api_version=1 + an mhlo.backend_config
    side-attribute and trusts XLA's StableHLO->HLO import to restore the
    typed version — which does not happen on sufficiently large modules,
    making every call dispatch as a LEGACY custom call (NOT_FOUND at
    compile). spineax emits the op itself with the typed form stated in
    full; nothing is left to restore and the handler ABI (typed FFI, api 4)
    is pinned even if jax's default version moves.
    """
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(6).standard_normal(A.shape[0]))

    def full(values, b):
        t = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, values)
        return cudss.solve(t, b, ir_nsteps=1), cudss.inertia(cudss.query(t))

    txt = jax.jit(full).lower(values, b).as_text()
    calls = [l for l in txt.splitlines() if "spineax_token" in l]
    assert len(calls) >= 4  # analyze, factorize, query, solve(s)
    assert all("api_version = 4" in l for l in calls)
    assert "mhlo.backend_config" not in txt
    # analyze's static config rides the backend_config DICT attribute
    (al,) = [l for l in calls if "analyze" in l]
    assert "mtype_id = 1" in al and "batch_size = 1" in al


def test_refactorize():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(6).standard_normal(A.shape[0]))

    token = cudss.analyze(values, offsets, columns)
    token = cudss.factorize(token, values)
    token = cudss.refactorize(token, values * 2.0)
    x = cudss.solve(token, b)
    assert _rel_err(2 * np.asarray(A), x, b) < 1e-10
    inertia = cudss.inertia(cudss.query(token))
    np.testing.assert_array_equal(np.asarray(inertia), [A.shape[0], 0])


# multi-RHS and vmap ===========================================================
def test_multirhs_stack():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    B = jnp.asarray(np.random.default_rng(7).standard_normal((6, A.shape[0])))

    token = cudss.analyze(values, offsets, columns)
    token = cudss.factorize(token, values)
    X = cudss.solve(token, B)  # one multi-RHS SOLVE
    for i in range(B.shape[0]):
        assert _rel_err(A, X[i], B[i]) < 1e-10


def test_vmap_multirhs_fast_path():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    B = jnp.asarray(np.random.default_rng(8).standard_normal((6, A.shape[0])))

    token = cudss.analyze(values, offsets, columns)
    token = cudss.factorize(token, values)
    X_vmap = jax.vmap(lambda b: cudss.solve(token, b))(B)
    X_direct = cudss.solve(token, B)
    np.testing.assert_allclose(np.asarray(X_vmap), np.asarray(X_direct), rtol=1e-14)


def test_vmap_batch_is_block_diagonal(monkeypatch):
    _require_gpu()
    # capacity is re-read from the env each call; raise it so LRU eviction
    # cannot mask the entry count this test asserts on
    monkeypatch.setenv("SPINEAX_FACTOR_CACHE", "64")
    values, offsets, columns, A = _sym_system()
    vals_batch = jnp.stack([values, values * 3.0])
    b = jnp.asarray(np.random.default_rng(9).standard_normal(A.shape[0]))
    b_batch = jnp.stack([b, b])

    size_before = cudss.registry_size()
    tokens = jax.vmap(lambda v: cudss.analyze(v, offsets, columns))(vals_batch)
    # ONE block-diagonal entry, its id broadcast across the batch
    assert tokens.id.shape == (2, 1)
    ids = np.asarray(tokens.id)
    assert ids[0, 0] == ids[1, 0]
    assert cudss.registry_size() - size_before == 1

    tokens = jax.vmap(cudss.factorize)(tokens, vals_batch)
    inertias = cudss.inertia(cudss.query(tokens), batch_size=2)
    np.testing.assert_array_equal(np.asarray(inertias), [[A.shape[0], 0]] * 2)
    xs = jax.vmap(cudss.solve)(tokens, b_batch)
    assert _rel_err(A, xs[0], b) < 1e-10
    assert _rel_err(3 * np.asarray(A), xs[1], b) < 1e-10

    # vmap-minted tokens work eagerly (outside vmap) too: batch-shaped args
    tokens2 = cudss.factorize(tokens, vals_batch * 2.0)
    xs2 = cudss.solve(tokens2, b_batch)
    assert _rel_err(2 * np.asarray(A), xs2[0], b) < 1e-10
    assert _rel_err(6 * np.asarray(A), xs2[1], b) < 1e-10


def test_explicit_batch_door():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    n = A.shape[0]
    vals = jnp.stack([values, values * 2.0, values * 5.0])
    token = cudss.analyze(vals, offsets, columns)
    assert token.kind == "pbatch"
    assert token.batch_size == 3
    token = cudss.factorize(token, vals)
    inertia = cudss.inertia(cudss.query(token), batch_size=3)
    np.testing.assert_array_equal(np.asarray(inertia), [[n, 0]] * 3)

    B = jnp.asarray(np.random.default_rng(20).standard_normal((3, n)))
    X = cudss.solve(token, B)
    for i, s in enumerate([1.0, 2.0, 5.0]):
        assert _rel_err(s * np.asarray(A), X[i], B[i]) < 1e-10

    token = cudss.refactorize(token, vals * 3.0)
    X = cudss.solve(token, B)
    for i, s in enumerate([3.0, 6.0, 15.0]):
        assert _rel_err(s * np.asarray(A), X[i], B[i]) < 1e-10


def test_vmap_batched_patterns():
    _require_gpu()
    # two systems, SAME shapes but DIFFERENT sparsity patterns (the "general
    # case": the whole block is analyzed, no shared structure)
    n = 4
    offs = jnp.asarray([0, 2, 3, 4, 5], dtype=jnp.int32)  # 5 nnz, upper view
    cols_a = jnp.asarray([0, 1, 1, 2, 3], dtype=jnp.int32)  # diag + (0,1)
    cols_b = jnp.asarray([0, 2, 1, 2, 3], dtype=jnp.int32)  # diag + (0,2)
    vals = jnp.asarray([4.0, 1.0, 3.0, 5.0, 2.0], dtype=jnp.float64)

    def dense(cols):
        A = np.zeros((n, n))
        v = np.asarray(vals)
        k = 0
        for i in range(n):
            for j in range(int(offs[i]), int(offs[i + 1])):
                A[i, int(cols[j])] = v[k]
                k += 1
        return A + A.T - np.diag(np.diag(A))

    cols_batch = jnp.stack([cols_a, cols_b])
    tokens = jax.vmap(lambda c: cudss.analyze(vals, offs, c))(cols_batch)
    tokens = jax.vmap(lambda t: cudss.factorize(t, vals))(tokens)
    b = jnp.asarray(np.random.default_rng(21).standard_normal(n))
    xs = jax.vmap(lambda t: cudss.solve(t, b))(tokens)
    assert _rel_err(dense(cols_a), xs[0], b) < 1e-10
    assert _rel_err(dense(cols_b), xs[1], b) < 1e-10


def test_vmap_factorize_unbatched_token_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = cudss.analyze(values, offsets, columns)
    vals_batch = jnp.stack([values, values * 2.0])
    with pytest.raises(ValueError, match="vmap\\(analyze\\)"):
        jax.vmap(lambda v: cudss.factorize(token, v))(vals_batch)


# control flow and autodiff ====================================================
def test_lax_cond_refactorize():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(10).standard_normal(A.shape[0]))

    @jax.jit
    def step(token, vals, b, do_refactor):
        token = jax.lax.cond(
            do_refactor,
            lambda t: cudss.refactorize(t, vals),
            lambda t: t,
            token,
        )
        return cudss.solve(token, b), token

    token = cudss.analyze(values, offsets, columns)
    token = cudss.factorize(token, values)

    x, token = step(token, values * 4.0, b, False)
    assert _rel_err(A, x, b) < 1e-10  # skip branch: old factors
    x, token = step(token, values * 4.0, b, True)
    assert _rel_err(4 * np.asarray(A), x, b) < 1e-10  # take branch: new factors


def test_custom_vjp_adjoint_reuse(monkeypatch):
    _require_gpu()
    # capacity is re-read from the env each call; raise it so LRU eviction
    # cannot mask the entry count this test asserts on
    monkeypatch.setenv("SPINEAX_FACTOR_CACHE", "64")
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(11).standard_normal(A.shape[0]))

    @jax.custom_vjp
    def token_solve(vals, b):
        t = cudss.analyze(vals, offsets, columns)
        t = cudss.factorize(t, vals)
        return cudss.solve(t, b)

    def fwd(vals, b):
        t = cudss.analyze(vals, offsets, columns)
        t = cudss.factorize(t, vals)
        return cudss.solve(t, b), t  # token threads through residuals

    def bwd(t, v):
        # symmetric: lambda = A^-T v = A^-1 v, reusing the forward factors
        return (None, cudss.solve(t, v))

    token_solve.defvjp(fwd, bwd)

    size_before = cudss.registry_size()
    grad_b = jax.jit(jax.grad(lambda v, b: token_solve(v, b).sum(), argnums=1))(values, b)
    expected = np.linalg.solve(np.asarray(A), np.ones(A.shape[0]))
    np.testing.assert_allclose(np.asarray(grad_b), expected, rtol=1e-10)
    # forward+backward share ONE factorization
    assert cudss.registry_size() - size_before == 1


# inertia ======================================================================
def test_inertia_indefinite():
    _require_gpu()
    # shift=0: random symmetric, genuinely indefinite
    values, offsets, columns, A = _sym_system(n=40, shift=0.0, seed=12)
    eigs = np.linalg.eigvalsh(np.asarray(A))
    expected = [int((eigs > 0).sum()), int((eigs < 0).sum())]

    token = cudss.analyze(values, offsets, columns)
    token = cudss.factorize(token, values)
    inertia = cudss.inertia(cudss.query(token))
    np.testing.assert_array_equal(np.asarray(inertia), expected)


def test_inertia_heterogeneous_batch_attribution():
    """Per-block inertia must land on the RIGHT blocks.

    Regression test: sign counts are computed on the block-aligned diag
    directly; reordering by perm_reorder_row first misattributes blocks in
    heterogeneous batches (its cross-block structure does not match the diag
    layout). Homogeneous batches cannot catch this — every block looks alike.
    """
    _require_gpu()
    n = 40
    rng = np.random.default_rng(30)
    # distinct expected inertia per block: PD, indefinite, PD, indefinite
    blocks, expected = [], []
    for i in range(4):
        A = rng.standard_normal((n, n))
        A = A + A.T + (n if i % 2 == 0 else 0.0) * np.eye(n)
        eigs = np.linalg.eigvalsh(A)
        expected.append([int((eigs > 0).sum()), int((eigs < 0).sum())])
        blocks.append(A)
    assert expected[0] != expected[1]  # the test is vacuous otherwise

    mask = np.triu(np.ones((n, n), dtype=bool))
    columns = jnp.asarray(np.nonzero(mask)[1].astype(np.int32))
    offsets = jnp.asarray(
        np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32))
    vals = jnp.asarray(np.stack([np.triu(A)[mask] for A in blocks]))

    token = cudss.analyze(vals, offsets, columns, mtype_id=1, mview_id=1)
    token = cudss.factorize(token, vals)
    inertia = cudss.inertia(cudss.query(token), batch_size=4)
    np.testing.assert_array_equal(np.asarray(inertia), expected)


def test_query_inertia_under_vmap():
    """query composes with vmap: ONE block query, input-ordered fields (diag,
    scales) split per block, block-global fields broadcast. Per-element
    inertia(query(token)) is what an IPM's inertia-correction loop reads
    inside a vmapped solver."""
    _require_gpu()
    n = 40
    rng = np.random.default_rng(32)
    blocks, expected = [], []
    for i in range(4):  # heterogeneous: PD, indefinite, PD, indefinite
        A = rng.standard_normal((n, n))
        A = A + A.T + (n if i % 2 == 0 else 0.0) * np.eye(n)
        eigs = np.linalg.eigvalsh(A)
        expected.append([int((eigs > 0).sum()), int((eigs < 0).sum())])
        blocks.append(A)
    assert expected[0] != expected[1]

    mask = np.triu(np.ones((n, n), dtype=bool))
    columns = jnp.asarray(np.nonzero(mask)[1].astype(np.int32))
    offsets = jnp.asarray(
        np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32))
    vals = jnp.asarray(np.stack([np.triu(A)[mask] for A in blocks]))

    @jax.jit
    @jax.vmap
    def per_block(v):
        t = cudss.analyze(v, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, v)
        data = cudss.query(t)
        return cudss.inertia(data), data["diag"], data["lu_nnz"]

    inr, diag, lu_nnz = per_block(vals)
    np.testing.assert_array_equal(np.asarray(inr), expected)
    assert diag.shape == (4, n)  # input-ordered field: split per block
    # diag is the LDL^T D in input order — sign pattern per block, not the
    # matrix diagonal; sanity-check block attribution via the sign counts
    assert lu_nnz.shape == (4, 1)  # block-global field: broadcast
    assert len(set(np.asarray(lu_nnz).ravel().tolist())) == 1


def test_query_inertia_under_nested_vmap():
    """Nested vmap composes: a (2, 2) heterogeneous batch is still ONE block
    query, and the input-ordered fields unwind one axis per vmap level with
    per-SYSTEM granularity (not per-outer-level)."""
    _require_gpu()
    n = 40
    rng = np.random.default_rng(33)
    blocks, expected = [], []
    for i in range(4):  # heterogeneous: PD, indefinite, PD, indefinite
        A = rng.standard_normal((n, n))
        A = A + A.T + (n if i % 2 == 0 else 0.0) * np.eye(n)
        eigs = np.linalg.eigvalsh(A)
        expected.append([int((eigs > 0).sum()), int((eigs < 0).sum())])
        blocks.append(A)
    assert expected[0] != expected[1]

    mask = np.triu(np.ones((n, n), dtype=bool))
    columns = jnp.asarray(np.nonzero(mask)[1].astype(np.int32))
    offsets = jnp.asarray(
        np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32))
    vals = jnp.asarray(
        np.stack([np.triu(A)[mask] for A in blocks])).reshape(2, 2, -1)

    @jax.jit
    @jax.vmap
    @jax.vmap
    def per_block(v):
        t = cudss.analyze(v, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, v)
        data = cudss.query(t)
        return cudss.inertia(data), data["diag"], data["lu_nnz"]

    inr, diag, lu_nnz = per_block(vals)
    assert inr.shape == (2, 2, 2)
    np.testing.assert_array_equal(np.asarray(inr).reshape(4, 2), expected)
    assert diag.shape == (2, 2, n)  # split axis per level, per-system blocks
    assert lu_nnz.shape == (2, 2, 1)  # block-global: broadcast at every level


def test_explicit_batch_door_under_vmap():
    """The explicit (B, nnz) door itself vmaps: the door routes through the
    same custom_vmap wrapper, so an outer vmap peels into one
    (B_outer * B)-block system instead of hitting the raw FFI."""
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    n = A.shape[0]
    scales = jnp.asarray([[1.0, 2.0], [0.5, 4.0]])
    vals = scales[..., None] * values  # (2, 2, nnz)
    b = jnp.asarray(np.random.default_rng(22).standard_normal((2, 2, n)))

    def solve_pair(v2, b2):  # v2: (2, nnz) — explicit batch door
        t = cudss.analyze(v2, offsets, columns)
        t = cudss.factorize(t, v2)
        return cudss.solve(t, b2)

    X = jax.vmap(solve_pair)(vals, b)
    assert X.shape == (2, 2, n)
    for i in range(2):
        for j in range(2):
            assert _rel_err(float(scales[i, j]) * np.asarray(A),
                            X[i, j], b[i, j]) < 1e-10


def test_grad_under_nested_vmap():
    """Reverse mode composes through nested vmap: grads wrt values and rhs of
    a (2, 2)-batched solve match the dense reference (the autodiff-added
    axes peel through the same recursive rules as the system axes)."""
    _require_gpu()
    n = 12
    values, offsets, columns, _ = _sym_system(n=n)
    scales = jnp.asarray([[1.0, 2.0], [0.5, 4.0]], dtype=jnp.float64)
    vals = scales[..., None] * values
    b = jnp.asarray(np.random.default_rng(23).standard_normal((2, 2, n)))

    def loss(vals, b):
        def one(v, bb):
            t = cudss.analyze(v, offsets, columns)
            t = cudss.factorize(t, v)
            return cudss.solve(t, bb)
        return jnp.sum(jnp.sin(jax.vmap(jax.vmap(one))(vals, b)))

    iu = np.triu_indices(n)  # _sym_system stores the full upper triangle

    def dense_loss(vals, b):
        def one(v, bb):
            U = jnp.zeros((n, n), v.dtype).at[iu].set(v)
            return jnp.linalg.solve(U + U.T - jnp.diag(jnp.diag(U)), bb)
        return jnp.sum(jnp.sin(jax.vmap(jax.vmap(one))(vals, b)))

    g_vals, g_b = jax.grad(loss, argnums=(0, 1))(vals, b)
    g_vals_d, g_b_d = jax.grad(dense_loss, argnums=(0, 1))(vals, b)
    np.testing.assert_allclose(np.asarray(g_vals), np.asarray(g_vals_d),
                               atol=1e-9)
    np.testing.assert_allclose(np.asarray(g_b), np.asarray(g_b_d), atol=1e-9)


def test_grad_through_checkpoint():
    """Reverse mode through jax.checkpoint/remat (issue #18): the phase chain
    must stay effect-free at the jaxpr level — remat partial-eval rejects any
    effectful equation — while XLA-level ordering rides on has_side_effect
    and the token id dataflow."""
    _require_gpu()
    values, offsets, columns, _ = _sym_system(n=20)
    b = jnp.asarray(np.random.default_rng(29).standard_normal(20))

    def chain(v):
        t = cudss.analyze(v, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, v)
        return cudss.solve(t, b)

    loss = lambda v: jnp.sum(chain(v) ** 2)
    loss_remat = lambda v: jnp.sum(jax.checkpoint(chain)(v) ** 2)

    assert np.isfinite(float(loss_remat(values)))  # forward through remat
    g = jax.grad(loss_remat)(values)               # used to raise on FfiEffect
    g_ref = jax.grad(loss)(values)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), atol=1e-12)
    # and jitted, where partial-eval actually splits the jaxpr
    g_jit = jax.jit(jax.grad(loss_remat))(values)
    np.testing.assert_allclose(np.asarray(g_jit), np.asarray(g_ref),
                               atol=1e-12)


# error handling ===============================================================
def test_solve_before_factorize_raises():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(15).standard_normal(A.shape[0]))
    token = cudss.analyze(values, offsets, columns)
    with pytest.raises(Exception, match="requires a factorized token"):
        cudss.solve(token, b).block_until_ready()


def test_refactorize_before_factorize_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = cudss.analyze(values, offsets, columns)
    with pytest.raises(Exception, match="requires a factorized token"):
        cudss.refactorize(token, values).id.block_until_ready()


def test_values_size_mismatch_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = cudss.analyze(values, offsets, columns)
    with pytest.raises(ValueError, match="nnz"):
        cudss.factorize(token, values[:-1])


def test_dtype_mismatch_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system(dtype=jnp.float64)
    token = cudss.analyze(values, offsets, columns)
    with pytest.raises(ValueError, match="dtype"):
        cudss.factorize(token, values.astype(jnp.float32))


def test_structure_tampering_raises():
    """A token's offsets/columns leaves are immutable by contract.

    Sizes alone cannot catch a swapped same-shaped pattern, and cuDSS would
    silently factorize it against the WRONG analysis/pivot order — so every
    phase verifies the structure CONTENT against the fingerprint stored at
    analysis and must fail loudly here.
    """
    import dataclasses

    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(22).standard_normal(A.shape[0]))
    token = cudss.analyze(values, offsets, columns)
    token = cudss.factorize(token, values)

    tampered = dataclasses.replace(token, columns=jnp.roll(token.columns, 1))
    with pytest.raises(Exception, match="fingerprint"):
        cudss.solve(tampered, b).block_until_ready()
    with pytest.raises(Exception, match="fingerprint"):
        cudss.factorize(tampered, values).id.block_until_ready()


# ported from the legacy CuDSSSolver/CuDSSSolverRE suite ======================
def _legacy_base_system(dtype=jnp.float32):
    M1 = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=dtype,
    )
    b1 = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=dtype)
    m1 = M1 + M1.T - jnp.diag(M1) * jnp.eye(M1.shape[0], dtype=dtype)
    true_x1 = jnp.linalg.solve(m1, b1)
    return M1, b1, m1, true_x1


def _csr_of(M):
    import jax.experimental.sparse as jsparse

    bcsr = jsparse.BCSR.fromdense(M)
    return bcsr.indptr, bcsr.indices, bcsr.data


def test_legacy_composability():
    _require_gpu()
    M1, b1, _, true_x1 = _legacy_base_system(jnp.float32)
    M2 = M1 * 0.9
    b2 = b1 * 1.1
    m2 = M2 + M2.T - jnp.diag(M2) * jnp.eye(M2.shape[0], dtype=M2.dtype)
    true_x2 = jnp.linalg.solve(m2, b2)

    offsets, columns, values1 = _csr_of(M1)
    _, _, values2 = _csr_of(M2)
    values = jnp.vstack([values1, values2])
    b = jnp.vstack([b1, b2])

    def token_solve(values, b):
        t = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, values)
        return cudss.solve(t, b)

    x1 = token_solve(values[0], b[0])
    x2 = jax.jit(jax.vmap(token_solve))(values, b)

    assert jnp.allclose(x1, true_x1, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(x2, jnp.stack([true_x1, true_x2]), rtol=1e-5, atol=1e-5)


def test_legacy_nested_vmap():
    # nested vmap composes: every custom_vmap rule collapses one axis and
    # recurses, so a (2, 2) batch is the same ONE block-diagonal system as
    # the flattened (4,) batch
    _require_gpu()
    M1, b1, _, _ = _legacy_base_system(jnp.float32)
    offsets, columns, values1 = _csr_of(M1)
    scales = jnp.array([[1.0, 2.0], [0.5, 4.0]], dtype=jnp.float32)
    values = scales[..., None] * values1
    b = jnp.broadcast_to(b1, (2, 2) + b1.shape)

    def token_solve(values, b):
        t = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, values)
        return cudss.solve(t, b)

    x = jax.jit(jax.vmap(jax.vmap(token_solve)))(values, b)
    x_flat = jax.jit(jax.vmap(token_solve))(
        values.reshape(4, -1), jnp.broadcast_to(b1, (4,) + b1.shape))
    assert x.shape == (2, 2) + b1.shape
    np.testing.assert_allclose(np.asarray(x).reshape(4, -1),
                               np.asarray(x_flat), rtol=1e-5)


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_legacy_datatypes(dtype):
    _require_gpu()
    _, b1, m1, true_x1 = _legacy_base_system(dtype)
    offsets, columns, values = _csr_of(m1)  # full symmetric matrix, mview full

    token = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=0)
    token = cudss.factorize(token, values)
    x = cudss.solve(token, b1)

    assert x.shape == b1.shape
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mtype_id", list(range(5)))
def test_legacy_solver_types(mtype_id):
    _require_gpu()
    _, b1, m1, true_x1 = _legacy_base_system(jnp.float32)
    offsets, columns, values = _csr_of(m1)

    token = cudss.analyze(values, offsets, columns, mtype_id=mtype_id, mview_id=0)
    token = cudss.factorize(token, values)
    x = cudss.solve(token, b1)

    assert x.shape == b1.shape
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)


def test_query():
    _require_gpu()
    # port of the CuDSSSolverRE "return everything" test
    M1, b1, _, true_x1 = _legacy_base_system(jnp.float32)
    offsets, columns, values = _csr_of(M1)  # upper triangle, mview upper
    n = b1.shape[0]

    token = cudss.analyze(values, offsets, columns, mtype_id=1, mview_id=1)
    token = cudss.factorize(token, values)
    x = cudss.solve(token, b1)
    assert jnp.allclose(x, true_x1, rtol=1e-5, atol=1e-5)

    out = cudss.query(token)
    assert out["lu_nnz"][0] > 0
    assert out["npivots"][0] >= 0
    assert out["inertia"].shape == (2,)
    for key in ("perm_reorder_row", "perm_reorder_col", "perm_row",
                "perm_col", "perm_matching", "scale_row", "scale_col"):
        assert out[key].shape == (n,), key
    assert out["diag"].shape == (n,)
    assert out["diag"].dtype == jnp.float32
    assert out["nd_partition_tree"].shape[0] > 0
    assert out["nsuperpanels"].shape == (1,)
    assert out["schur_shape"].shape == (2,)

    # inertia() over the same data agrees with a fresh query after refactorize
    inr1 = cudss.inertia(out)
    token = cudss.factorize(token, values)
    inr2 = cudss.inertia(cudss.query(token))
    np.testing.assert_array_equal(np.asarray(inr1), np.asarray(inr2))


def test_query_before_factorize_raises():
    _require_gpu()
    values, offsets, columns, _ = _sym_system()
    token = cudss.analyze(values, offsets, columns)
    with pytest.raises(Exception, match="requires a factorized token"):
        jax.block_until_ready(cudss.query(token))


# autodiff through the raw explicit phases ====================================
def test_grad_raw_explicit_phases():
    """jax.grad through analyze -> factorize -> solve, no lineax, no manual
    custom_vjp: the numeric phases pass the values tangent through and
    solve's built-in vjp does the implicit-function-theorem math. Upper
    triangular storage: one stored entry (i, j) is BOTH A_ij and A_ji."""
    _require_gpu()
    n = 20
    values, offsets, columns, _ = _sym_system(n=n, seed=14)
    b = jnp.asarray(np.random.default_rng(24).standard_normal(n))
    rows = jnp.repeat(jnp.arange(n), jnp.diff(offsets))

    def loss(vals, b):
        t = cudss.analyze(vals, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, vals)
        return jnp.sum(cudss.solve(t, b) ** 2)

    gv, gb = jax.jit(jax.grad(loss, argnums=(0, 1)))(values, b)

    def dense_loss(vals, b):
        U = jnp.zeros((n, n)).at[rows, columns].set(vals)
        A = U + U.T - jnp.diag(jnp.diag(U))
        return jnp.sum(jnp.linalg.solve(A, b) ** 2)

    gv_d, gb_d = jax.grad(dense_loss, argnums=(0, 1))(values, b)
    np.testing.assert_allclose(np.asarray(gv), np.asarray(gv_d),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(gb), np.asarray(gb_d),
                               rtol=1e-9, atol=1e-9)


def test_grad_raw_full_view_and_batch():
    """Gradient conventions for full-pattern storage and the explicit batch
    door (block-diagonal gather path), against dense references."""
    _require_gpu()
    import jax.experimental.sparse as jsparse

    n = 12
    *_, A = _sym_system(n=n, seed=15)
    sp = jsparse.BCSR.fromdense(jnp.asarray(A))
    rows = jnp.repeat(jnp.arange(n), jnp.diff(sp.indptr))
    b = jnp.asarray(np.random.default_rng(25).standard_normal(n))

    # full view (mview 0): every stored entry is one independent A_ij
    def loss(vals, b):
        t = cudss.analyze(vals, sp.indptr, sp.indices, mtype_id=1, mview_id=0)
        t = cudss.factorize(t, vals)
        return jnp.sum(cudss.solve(t, b) ** 2)

    gv = jax.grad(loss)(sp.data, b)

    def dense_loss(vals, b):
        Ad = jnp.zeros((n, n)).at[rows, sp.indices].set(vals)
        return jnp.sum(jnp.linalg.solve(Ad, b) ** 2)

    gv_d = jax.grad(dense_loss)(sp.data, b)
    np.testing.assert_allclose(np.asarray(gv), np.asarray(gv_d),
                               rtol=1e-9, atol=1e-9)

    # explicit batch door: per-block gradients through ONE block solve
    vals_batch = jnp.stack([sp.data, sp.data * 2.0])
    b_batch = jnp.stack([b, b * 3.0])

    def batch_loss(vals, bb):
        t = cudss.analyze(vals, sp.indptr, sp.indices, mtype_id=1, mview_id=0)
        t = cudss.factorize(t, vals)
        return jnp.sum(cudss.solve(t, bb) ** 2)

    gv_b, gb_b = jax.jit(jax.grad(batch_loss, argnums=(0, 1)))(
        vals_batch, b_batch)

    def dense_batch_loss(vals, bb):
        out = 0.0
        for k in range(2):
            Ad = jnp.zeros((n, n)).at[rows, sp.indices].set(vals[k])
            out = out + jnp.sum(jnp.linalg.solve(Ad, bb[k]) ** 2)
        return out

    gv_bd, gb_bd = jax.grad(dense_batch_loss, argnums=(0, 1))(
        vals_batch, b_batch)
    np.testing.assert_allclose(np.asarray(gv_b), np.asarray(gv_bd),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(gb_b), np.asarray(gb_bd),
                               rtol=1e-9, atol=1e-9)


def test_grad_general_mtype():
    """Reverse mode through a GENERAL (mtype 0) solve: cuDSS has no
    transpose solve, so the backward pass factorizes A^T on the fly."""
    _require_gpu()
    n = 12
    rng = np.random.default_rng(26)
    A = rng.standard_normal((n, n)) + n * np.eye(n)  # nonsymmetric
    columns = jnp.asarray(np.tile(np.arange(n, dtype=np.int32), n))
    offsets = jnp.asarray(np.arange(n + 1, dtype=np.int32) * n)
    rows = jnp.repeat(jnp.arange(n), n)
    vals = jnp.asarray(A.reshape(-1))
    b = jnp.asarray(rng.standard_normal(n))

    def loss(v, b):
        t = cudss.analyze(v, offsets, columns, mtype_id=0, mview_id=0)
        t = cudss.factorize(t, v)
        return jnp.sum(cudss.solve(t, b) ** 2)

    gv, gb = jax.jit(jax.grad(loss, argnums=(0, 1)))(vals, b)

    def dense_loss(v, b):
        Ad = jnp.zeros((n, n)).at[rows, columns].set(v)
        return jnp.sum(jnp.linalg.solve(Ad, b) ** 2)

    gv_d, gb_d = jax.grad(dense_loss, argnums=(0, 1))(vals, b)
    np.testing.assert_allclose(np.asarray(gv), np.asarray(gv_d),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(gb), np.asarray(gb_d),
                               rtol=1e-9, atol=1e-9)


def test_grad_hermitian_complex():
    """Reverse mode through a hermitian (mtype 2) complex solve: the adjoint
    reuses the forward factors via A^T x = c <=> x = conj(A^-1 conj(c))."""
    _require_gpu()
    n = 12
    rng = np.random.default_rng(27)
    H = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = H + H.conj().T + 2 * n * np.eye(n)
    mask = np.triu(np.ones((n, n), bool))
    offsets = jnp.asarray(
        np.concatenate([[0], np.cumsum(mask.sum(1))]).astype(np.int32))
    columns = jnp.asarray(np.nonzero(mask)[1].astype(np.int32))
    rows = jnp.asarray(np.nonzero(mask)[0])
    vals = jnp.asarray(H[mask])
    b = jnp.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))

    def loss(v, b):
        t = cudss.analyze(v, offsets, columns, mtype_id=2, mview_id=1)
        t = cudss.factorize(t, v)
        return jnp.sum(jnp.abs(cudss.solve(t, b)) ** 2)

    gv, gb = jax.jit(jax.grad(loss, argnums=(0, 1)))(vals, b)

    def dense_loss(v, b):
        # the stored diagonal is A_ii AS-IS (real for hermitian input);
        # subtracting diag(conj(U)) keeps that convention in the reference
        U = jnp.zeros((n, n), jnp.complex128).at[rows, columns].set(v)
        Ad = U + U.conj().T - jnp.diag(jnp.diag(U).conj())
        return jnp.sum(jnp.abs(jnp.linalg.solve(Ad, b)) ** 2)

    gv_d, gb_d = jax.grad(dense_loss, argnums=(0, 1))(vals, b)
    np.testing.assert_allclose(np.asarray(gv), np.asarray(gv_d),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(gb), np.asarray(gb_d),
                               rtol=1e-9, atol=1e-9)


def test_higher_order_derivatives():
    """Arbitrary-order differentiation: custom_linear_solve's rules recurse,
    the numeric-phase identity rules recurse, and the solve vmap rule
    recurses (jacfwd-of-grad nests one vmap per order onto the FFI call).
    Hessian, reverse-over-reverse and third order against dense refs."""
    _require_gpu()
    n = 10
    values, offsets, columns, _ = _sym_system(n=n, seed=28)
    b = jnp.asarray(np.random.default_rng(29).standard_normal(n))
    rows = jnp.repeat(jnp.arange(n), jnp.diff(offsets))

    def loss(v, b):
        t = cudss.analyze(v, offsets, columns, mtype_id=1, mview_id=1)
        t = cudss.factorize(t, v)
        return jnp.sum(cudss.solve(t, b) ** 2)

    def dense_loss(v, b):
        U = jnp.zeros((n, n)).at[rows, columns].set(v)
        A = U + U.T - jnp.diag(jnp.diag(U))
        return jnp.sum(jnp.linalg.solve(A, b) ** 2)

    # forward-over-reverse (jax.hessian), wrt values and wrt b
    Hv = jax.jit(jax.hessian(loss))(values, b)
    Hv_d = jax.hessian(dense_loss)(values, b)
    np.testing.assert_allclose(np.asarray(Hv), np.asarray(Hv_d),
                               rtol=1e-8, atol=1e-8)
    Hb = jax.jit(jax.hessian(loss, argnums=1))(values, b)
    Hb_d = jax.hessian(dense_loss, argnums=1)(values, b)
    np.testing.assert_allclose(np.asarray(Hb), np.asarray(Hb_d),
                               rtol=1e-8, atol=1e-8)

    # reverse-over-reverse
    gg = jax.grad(lambda bb: jax.grad(loss, argnums=1)(values, bb).sum())(b)
    gg_d = jax.grad(
        lambda bb: jax.grad(dense_loss, argnums=1)(values, bb).sum())(b)
    np.testing.assert_allclose(np.asarray(gg), np.asarray(gg_d),
                               rtol=1e-9, atol=1e-9)

    # third order along a direction, values entering through factorize
    dv = jnp.asarray(np.random.default_rng(31).standard_normal(values.shape))
    d3 = jax.grad(jax.grad(jax.grad(
        lambda s: loss(values + s * dv, b))))(0.0)
    d3_d = jax.grad(jax.grad(jax.grad(
        lambda s: dense_loss(values + s * dv, b))))(0.0)
    np.testing.assert_allclose(float(d3), float(d3_d), rtol=1e-7)


# lineax adapter (the default user API, defined in solver.py) ================
def test_lineax_adapter():
    import lineax as lx

    _require_gpu()
    *_, A = _sym_system(n=40, shift=0.0, seed=13)
    # full-pattern operator from the dense symmetric matrix
    import jax.experimental.sparse as jsparse

    sp = jsparse.BCSR.fromdense(jnp.asarray(A))
    operator = cudss.CSROperator(sp.data, sp.indptr, sp.indices,
                                 lx.symmetric_tag)
    assert lx.is_symmetric(operator)
    assert operator.transpose() is operator
    solver = cudss.CuDSS()
    b = jnp.asarray(np.random.default_rng(23).standard_normal(A.shape[0]))

    # explicit phases incl. query; symmetric tag resolves to mtype 1 (LDL)
    token = solver.analyze(operator)
    assert token.mtype_id == 1
    token = solver.factorize(token, operator)
    x = solver.solve(token, b)
    assert _rel_err(A, x, b) < 1e-10
    eigs = np.linalg.eigvalsh(np.asarray(A))
    inr = cudss.inertia(solver.query(token))
    np.testing.assert_array_equal(
        np.asarray(inr), [int((eigs > 0).sum()), int((eigs < 0).sum())])

    # lineax protocol, interoperating with the explicit token as state=
    x2 = lx.linear_solve(operator, b, solver, state=token).value
    np.testing.assert_allclose(np.asarray(x2), np.asarray(x), rtol=1e-12)
    sol = lx.linear_solve(operator, b, solver)
    assert _rel_err(A, sol.value, b) < 1e-10
    assert cudss.release(sol.state) is True


def test_lineax_general_operator():
    """Untagged CSROperator = general (mtype 0): transpose is a real CSR
    transpose, and gradients through lx.linear_solve pay for one transposed
    factorization in the backward pass (no cuDSS transpose solve)."""
    import lineax as lx
    import jax.experimental.sparse as jsparse

    _require_gpu()
    n = 12
    rng = np.random.default_rng(33)
    A = rng.standard_normal((n, n)) + n * np.eye(n)  # nonsymmetric
    sp = jsparse.BCSR.fromdense(jnp.asarray(A))
    b = jnp.asarray(rng.standard_normal(n))
    solver = cudss.CuDSS()

    operator = cudss.CSROperator(sp.data, sp.indptr, sp.indices)
    assert not lx.is_symmetric(operator)
    np.testing.assert_allclose(np.asarray(operator.transpose().as_matrix()),
                               np.asarray(A).T, rtol=1e-15)

    token = solver.analyze(operator)
    assert token.mtype_id == 0
    sol = lx.linear_solve(operator, b, solver)
    np.testing.assert_allclose(np.asarray(sol.value),
                               np.linalg.solve(np.asarray(A), np.asarray(b)),
                               rtol=1e-9)
    cudss.release(sol.state)

    # gradient through lx.linear_solve (exercises solver.transpose -> A^T
    # factorization) vs dense reference
    def loss(vals, bb):
        op = cudss.CSROperator(vals, sp.indptr, sp.indices)
        return jnp.sum(lx.linear_solve(op, bb, solver).value ** 2)

    gv, gb = jax.grad(loss, argnums=(0, 1))(sp.data, b)
    rows = jnp.repeat(jnp.arange(n), jnp.diff(sp.indptr))

    def dense_loss(vals, bb):
        Ad = jnp.zeros((n, n)).at[rows, sp.indices].set(vals)
        return jnp.sum(jnp.linalg.solve(Ad, bb) ** 2)

    gv_d, gb_d = jax.grad(dense_loss, argnums=(0, 1))(sp.data, b)
    np.testing.assert_allclose(np.asarray(gv), np.asarray(gv_d),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(gb), np.asarray(gb_d),
                               rtol=1e-9, atol=1e-9)


# lifetime =====================================================================
def test_release_then_self_heal():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(16).standard_normal(A.shape[0]))
    token = cudss.factorize(cudss.analyze(values, offsets, columns), values)
    cudss.solve(token, b).block_until_ready()

    assert cudss.release(token) is True
    assert cudss.release(token) is False  # second release is a no-op
    # release frees the factors, not the token: the next solve rebuilds
    r0 = cudss.rebuild_count()
    x = cudss.solve(token, b)
    assert _rel_err(A, x, b) < _TOL[jnp.float64]
    assert cudss.rebuild_count() == r0 + 1
    cudss.solve(token, b).block_until_ready()  # healed in place: no rebuild
    assert cudss.rebuild_count() == r0 + 1


def test_lru_eviction_self_heals():
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(17).standard_normal(A.shape[0]))
    cap = cudss.cache_capacity()

    victim = cudss.factorize(cudss.analyze(values, offsets, columns), values)
    # cap fresh entries push the untouched victim out
    for _ in range(cap):
        cudss.analyze(values, offsets, columns).id.block_until_ready()
    assert cudss.registry_size() <= cap
    r0 = cudss.rebuild_count()
    x = cudss.solve(victim, b)  # evicted -> rebuilt from the token's arrays
    assert _rel_err(A, x, b) < _TOL[jnp.float64]
    assert cudss.rebuild_count() == r0 + 1
    # query heals too: evict again, then read diagnostics off the rebuild
    for _ in range(cap):
        cudss.analyze(values, offsets, columns).id.block_until_ready()
    r1 = cudss.rebuild_count()
    inr = cudss.inertia(cudss.query(victim))
    np.testing.assert_array_equal(np.asarray(inr), [A.shape[0], 0])  # PD
    assert cudss.rebuild_count() == r1 + 1


def test_cudss_config_knobs():
    """reordering/memory knobs reach cuDSS, and survive eviction self-heal
    (they are part of the token's rebuild recipe like mtype/mview)."""
    _require_gpu()
    values, offsets, columns, A = _sym_system(n=80)
    b = jnp.asarray(np.random.default_rng(19).standard_normal(A.shape[0]))

    lu = {}
    for alg in ("default", "none"):  # fill-reducing vs natural order
        t = cudss.factorize(
            cudss.analyze(values, offsets, columns, reordering=alg), values)
        assert _rel_err(A, cudss.solve(t, b), b) < _TOL[jnp.float64]
        lu[alg] = int(np.asarray(cudss.query(t)["lu_nnz"])[0])
    assert lu["none"] >= lu["default"]

    # hybrid host+device factors, evicted and healed with the same config
    t = cudss.factorize(
        cudss.analyze(values, offsets, columns, reordering="none",
                      memory="hybrid"), values)
    for _ in range(cudss.cache_capacity()):
        cudss.analyze(values, offsets, columns).id.block_until_ready()
    assert _rel_err(A, cudss.solve(t, b), b) < _TOL[jnp.float64]
    assert int(np.asarray(cudss.query(t)["lu_nnz"])[0]) == lu["none"]

    with pytest.raises(ValueError, match="unknown reordering"):
        cudss.analyze(values, offsets, columns, reordering="bogus")
    with pytest.raises(Exception, match="invalid reordering_id"):
        # raw enum ints pass through; out-of-range is the C++ backstop
        cudss.analyze(values, offsets, columns,
                      reordering=9).id.block_until_ready()


def test_numeric_phases_rename():
    """Every id names one immutable numeric state: factorize/refactorize
    return fresh ids, and a superseded token self-heals to ITS OWN values —
    the stale-token wrong-answer class is gone."""
    _require_gpu()
    values, offsets, columns, A = _sym_system()
    b = jnp.asarray(np.random.default_rng(18).standard_normal(A.shape[0]))

    t_an = cudss.analyze(values, offsets, columns)
    t0 = cudss.factorize(t_an, values)
    t1 = cudss.refactorize(t0, 2.0 * values)
    ids = {int(np.asarray(t.id)[0]) for t in (t_an, t0, t1)}
    assert len(ids) == 3

    # t1 owns the entry; solving stale t0 rebuilds A's factors, not 2A's
    x1 = cudss.solve(t1, b)
    x0 = cudss.solve(t0, b)
    assert _rel_err(2.0 * A, x1, b) < _TOL[jnp.float64]
    assert _rel_err(A, x0, b) < _TOL[jnp.float64]


def test_concurrent_refactors_branch_from_one_token():
    from concurrent.futures import ThreadPoolExecutor

    _require_gpu()
    values, offsets, columns, A = _sym_system(n=80, seed=35)
    b = jnp.asarray(np.random.default_rng(36).standard_normal(A.shape[0]))
    token = cudss.factorize(cudss.analyze(values, offsets, columns), values)

    # Compile before dispatching the same resident state from two host threads.
    warm = cudss.refactorize(token, 1.5 * values)
    warm.id.block_until_ready()
    token = cudss.factorize(cudss.analyze(values, offsets, columns), values)

    def branch(scale):
        result = cudss.refactorize(token, scale * values)
        result.id.block_until_ready()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        branches = list(pool.map(branch, (2.0, 3.0)))

    for scale, result in zip((2.0, 3.0), branches):
        assert _rel_err(scale * A, cudss.solve(result, b), b) < _TOL[jnp.float64]
    assert _rel_err(A, cudss.solve(token, b), b) < _TOL[jnp.float64]
