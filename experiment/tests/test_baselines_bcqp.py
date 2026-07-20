"""Unit tests for the MPRGP bound-constrained-QP baseline."""

from __future__ import annotations

import numpy as np
import pytest

from nncg_note.baselines_bcqp import mprgp, power_iteration_lambda_max
from nncg_note.problems import make_problem


@pytest.mark.parametrize("kappa", [1e2, 1e4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mprgp_recovers_planted_optimum(kappa, seed):
    """MPRGP reaches the known planted optimum on the synthetic family."""
    A, b, x_star, _ = make_problem(200, kappa, seed=seed)
    lam = float(np.linalg.eigvalsh(A)[-1])
    res = mprgp(lambda v: A @ v, b, len(b), lam_max=lam, tol=1e-9, maxit=100000)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-5
    assert np.all(res.x >= 0.0)                          # feasible-point method


def test_mprgp_estimates_lambda_max_when_omitted():
    """Without lam_max, MPRGP falls back to power iteration and still converges."""
    A, b, x_star, _ = make_problem(120, 1e3, seed=0)
    res = mprgp(lambda v: A @ v, b, len(b), tol=1e-9, maxit=100000)
    assert res.converged
    assert np.max(np.abs(res.x - x_star)) < 1e-5


def test_mprgp_hits_iteration_cap():
    """A tight cap returns the best iterate flagged as not converged."""
    A, b, _, _ = make_problem(200, 1e6, seed=0)
    lam = float(np.linalg.eigvalsh(A)[-1])
    res = mprgp(lambda v: A @ v, b, len(b), lam_max=lam, tol=1e-12, maxit=2)
    assert not res.converged
    assert res.iters == 2


def test_power_iteration_matches_dense_eig():
    """Power iteration recovers lambda_max of an SPD matrix."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((60, 60))
    A = M @ M.T + np.eye(60)
    est = power_iteration_lambda_max(lambda v: A @ v, 60, iters=500)
    assert est == pytest.approx(float(np.linalg.eigvalsh(A)[-1]), rel=1e-3)


def test_power_iteration_zero_operator_returns_early():
    """A zero operator yields a zero iterate; the guard returns the last estimate."""
    assert power_iteration_lambda_max(lambda v: np.zeros_like(v), 10) == 0.0
