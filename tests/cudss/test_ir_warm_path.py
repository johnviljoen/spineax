"""Regression: cuDSS warm solves with iterative refinement and fresh buffers.

Before the owned-buffer fix, the second call into ``solve()`` crashed
with ``CUDA_ERROR_ILLEGAL_ADDRESS`` whenever XLA reallocated the input
arrays between solves — the normal case in any JAX loop that
re-materialises operands.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from spineax.cudss.solver import solve as cudss_solve


def _require_gpu():
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU available")


def _poisson_csr(n_side):
    """5-point Poisson stencil on n_side × n_side grid (SPD)."""
    n = n_side * n_side
    rows, cols, vals = [], [], []
    for i in range(n_side):
        for j in range(n_side):
            r = i * n_side + j
            rows.append(r); cols.append(r); vals.append(4.0)
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n_side and 0 <= nj < n_side:
                    rows.append(r); cols.append(ni * n_side + nj); vals.append(-1.0)
    order = np.lexsort((np.asarray(cols), np.asarray(rows)))
    rows = np.asarray(rows)[order]
    cols = np.asarray(cols, dtype=np.int32)[order]
    vals = np.asarray(vals, dtype=np.float64)[order]
    offsets = np.zeros(n + 1, dtype=np.int32)
    np.add.at(offsets[1:], rows, 1)
    np.cumsum(offsets, out=offsets)
    return offsets, cols, vals, n


def _gpu_used_mib():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip().split("\n")[0])


def test_fresh_buffer_warm_solves():
    """Re-materialise input arrays each call (so XLA allocates fresh GPU
    buffers) and check that repeated solves both succeed and stay accurate."""
    _require_gpu()
    offsets, cols, vals, n = _poisson_csr(60)
    A_dense = np.zeros((n, n))
    A_dense[np.repeat(np.arange(n), np.diff(offsets)), cols] = vals
    b_np = np.ones(n)
    x_ref = np.linalg.solve(A_dense, b_np)

    for _ in range(5):
        x, _ = cudss_solve(
            jnp.asarray(b_np),
            jnp.asarray(vals),
            jnp.asarray(offsets),
            jnp.asarray(cols),
            device_id=0, mtype_id=3, mview_id=0,
        )
        jax.block_until_ready(x)

    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-9, atol=1e-12)


def test_warm_solves_dont_leak_memory():
    """50 fresh-buffer warm solves: GPU memory must not grow."""
    _require_gpu()
    offsets, cols, vals, n = _poisson_csr(60)
    b_np = np.ones(n)

    mem_first = None
    for i in range(50):
        x, _ = cudss_solve(
            jnp.asarray(b_np),
            jnp.asarray(vals),
            jnp.asarray(offsets),
            jnp.asarray(cols),
            device_id=0, mtype_id=3, mview_id=0,
        )
        jax.block_until_ready(x)
        if i == 0:
            mem_first = _gpu_used_mib()
    mem_last = _gpu_used_mib()

    growth = (mem_last - mem_first) / 49.0
    assert abs(growth) < 2.0, f"GPU memory grew by {growth:+.2f} MiB / solve"
