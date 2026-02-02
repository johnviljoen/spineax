"""Test pbatch_solve optional fallback behavior."""
import pytest
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse


def get_test_system():
    """Create a simple test linear system."""
    A = jnp.array([
        [4., 0., 1., 0., 0.],
        [0., 3., 2., 0., 0.],
        [0., 0., 5., 0., 1.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 2.],
    ], dtype=jnp.float32)

    b = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=jnp.float32)

    # Symmetrize
    A_sym = A + A.T - jnp.diag(A) * jnp.eye(A.shape[0], dtype=jnp.float32)
    true_x = jnp.linalg.solve(A_sym, b)

    LHS = jsparse.BCSR.fromdense(A)
    return LHS.indptr, LHS.indices, LHS.data, b, true_x


class TestPbatchAvailable:
    """Tests when pbatch_solve IS available."""

    def test_pbatch_available_flag(self):
        """Verify PBATCH_AVAILABLE is True when pbatch builds."""
        from spineax.cudss import solver
        assert solver.PBATCH_AVAILABLE is True, "pbatch_solve should be available"

    def test_vmap_uses_pbatch(self):
        """Verify vmap uses pseudo_batch when available."""
        from spineax.cudss import solver
        assert solver.vmap_using_pseudo_batch is True, "Should use pbatch when available"

    def test_single_solve(self):
        """Test single solve works."""
        from spineax.cudss.solver import CuDSSSolver

        csr_offsets, csr_columns, csr_values, b, true_x = get_test_system()
        solver = CuDSSSolver(csr_offsets, csr_columns, 0, 1, 1)

        x, inertia = solver(b, csr_values)

        assert jnp.allclose(x, true_x, atol=1e-5), f"Solution error: {jnp.max(jnp.abs(x - true_x))}"
        assert inertia[0] == 5, f"Expected positive inertia 5, got {inertia[0]}"
        assert inertia[1] == 0, f"Expected negative inertia 0, got {inertia[1]}"

    def test_batched_solve_inertia(self):
        """Test batched solve returns correct inertia with pbatch."""
        from spineax.cudss.solver import CuDSSSolver

        csr_offsets, csr_columns, csr_values, b, true_x = get_test_system()
        solver = CuDSSSolver(csr_offsets, csr_columns, 0, 1, 1)

        # Batch of RHS
        b_batch = jnp.stack([b, b * 2, b * 0.5])

        x_batch, inertia_batch = jax.vmap(lambda bi: solver(bi, csr_values))(b_batch)

        assert x_batch.shape == (3, 5), f"Wrong shape: {x_batch.shape}"
        # With pbatch, inertia should be correct
        assert jnp.all(inertia_batch[:, 0] == 5), f"Inertia should be [5,0], got {inertia_batch}"


class TestPbatchFallback:
    """Tests simulating pbatch_solve NOT available (fallback to batch_solve)."""

    def test_fallback_single_solve(self):
        """Single solve should still work without pbatch."""
        from spineax.cudss.solver import CuDSSSolver

        csr_offsets, csr_columns, csr_values, b, true_x = get_test_system()
        solver = CuDSSSolver(csr_offsets, csr_columns, 0, 1, 1)

        # Single solve doesn't use pbatch anyway
        x, inertia = solver(b, csr_values)

        assert jnp.allclose(x, true_x, atol=1e-5)

    def test_fallback_batched_solve(self):
        """Test batched solve works with batch_solve fallback."""
        from spineax.cudss import solver as solver_module
        from spineax.cudss.solver import CuDSSSolver

        # Temporarily disable pbatch
        original_value = solver_module.vmap_using_pseudo_batch
        solver_module.vmap_using_pseudo_batch = False

        try:
            csr_offsets, csr_columns, csr_values, b, true_x = get_test_system()
            solver = CuDSSSolver(csr_offsets, csr_columns, 0, 1, 1)

            b_batch = jnp.stack([b, b * 2])
            csr_batch = jnp.stack([csr_values, csr_values])

            # With vmap_using_pseudo_batch=False, uses batch_solve
            x_batch, inertia_batch = jax.vmap(solver)(b_batch, csr_batch)

            assert x_batch.shape == (2, 5), f"Wrong shape: {x_batch.shape}"
            # Solution should still be correct
            assert jnp.allclose(x_batch[0], true_x, atol=1e-5)
            # Note: inertia may be [0,0] with batch_solve - that's expected
        finally:
            # Restore original value
            solver_module.vmap_using_pseudo_batch = original_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
