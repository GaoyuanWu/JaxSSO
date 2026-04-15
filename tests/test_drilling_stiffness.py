"""Tests for the lstsq solver option.

A clamped plate under vertical load is solved with nx=5, 6, and 7.
The sparse solver produces singular matrices at certain mesh sizes due to
near-zero drilling DOF eigenvalues in the augmented Lagrangian system.
The lstsq solver handles all cases robustly.
"""

import numpy as np
import pytest
from JaxSSO import model as jm


def _build_clamped_plate(nx: int) -> jm.Model:
    """Build an nx×nx quad plate clamped on the left edge, loaded on the right."""
    m = jm.Model()
    for j in range(nx + 1):
        for i in range(nx + 1):
            x, y = i / nx, j / nx
            z = 0.05 * x * (1.0 - x)
            m.add_node(j * (nx + 1) + i, x, y, z)

    etag = 0
    for j in range(nx):
        for i in range(nx):
            p0 = j * (nx + 1) + i
            m.add_quad(etag, p0, p0 + 1, p0 + 1 + (nx + 1), p0 + (nx + 1),
                       0.01, 2.1e11, 0.3, 1.0, 1.0)
            etag += 1

    for j in range(nx + 1):
        m.add_support(j * (nx + 1), [1, 1, 1, 1, 1, 1])
    for j in range(nx + 1):
        m.add_nodal_load(j * (nx + 1) + nx, [0, 0, -1000, 0, 0, 0])

    m.model_ready()
    return m


class TestSparseSolverSingularity:
    """The sparse solver fails at certain mesh sizes because the augmented
    system with Lagrange multipliers becomes exactly singular due to
    near-zero drilling DOF eigenvalues."""

    @pytest.mark.parametrize("nx", [5, 6, 7])
    def test_sparse_fails_at_some_sizes(self, nx: int) -> None:
        m = _build_clamped_plate(nx)
        m.solve(which_solver="sparse", enforce_scipy_sparse=True)
        se = m.strain_energy()
        if nx in (6, 7):
            assert np.isnan(se), (
                f"Expected NaN for nx={nx} with sparse solver, got {se}"
            )
        else:
            assert not np.isnan(se)
            assert se > 0


class TestLstsqSolver:
    """The lstsq solver handles near-singular systems gracefully by
    computing the minimum-norm least-squares solution."""

    @pytest.mark.parametrize("nx", [5, 6, 7])
    def test_lstsq_always_works(self, nx: int) -> None:
        m = _build_clamped_plate(nx)
        m.solve(which_solver="lstsq")
        se = m.strain_energy()
        assert not np.isnan(se), f"NaN for nx={nx} with lstsq"
        assert se > 0

    def test_strain_energy_increases_with_mesh_size(self) -> None:
        """More load nodes → higher total strain energy."""
        se_prev = 0.0
        for nx in [3, 4, 5, 6, 7, 8]:
            m = _build_clamped_plate(nx)
            m.solve(which_solver="lstsq")
            se = m.strain_energy()
            assert se > se_prev, f"SE did not increase at nx={nx}: {se} <= {se_prev}"
            se_prev = se
