"""Tests for the ``minvar`` drop-in: problem construction, the nncg and
baselines solve hooks, shrinkage utilities, warm starts, and balance systems.

The invariant everywhere is that all solvers minimise the same program, so they
must agree on the minimiser (to each method's tolerance), land on the feasible
set (``w >= 0``, ``B w = c``), and reproduce the return contracts the experiment
scripts unpack.
"""

from __future__ import annotations

import numpy as np
import pytest

from minvar.minvar import (
    MinVarProblem,
    lw_alpha_and_target,
    lw_alpha_and_target_hard,
    oas_alpha_and_target,
    rmt_target_and_alpha,
)


@pytest.fixture
def problem(returns):
    alpha, target = lw_alpha_and_target(returns)
    return MinVarProblem(returns, alpha=alpha, target=target)


# ---------------------------------------------------------------------------
# Feasibility and cross-solver agreement (budget 1^T w = 1)
# ---------------------------------------------------------------------------


def test_budget_solution_is_feasible(problem):
    w, outer, inner = problem.solve_cg()
    assert abs(w.sum() - 1.0) < 1e-8
    assert w.min() >= -1e-9
    assert outer >= 1 and inner >= 1


def test_all_solvers_agree(problem):
    w_cg = problem.solve_cg()[0]
    w_kkt = problem.solve_kkt()[0]
    w_pcg = problem.solve_pcg()[0]
    w_cla = problem.solve_clarabel()[0]
    w_osqp = problem.solve_osqp()[0]
    w_prox = problem.solve_proximal()[0]
    # exact/direct solvers agree to machine precision
    assert np.linalg.norm(w_kkt - w_cg) < 1e-6
    assert np.linalg.norm(w_pcg - w_cg) < 1e-6
    # iterative baselines agree to their stopping tolerance
    assert np.linalg.norm(w_cla - w_cg) < 1e-3
    assert np.linalg.norm(w_osqp - w_cg) < 1e-3
    assert np.linalg.norm(w_prox - w_cg) < 1e-3


def test_return_contracts(problem):
    assert len(problem.solve_cg()) == 3  # (w, outer, inner)
    assert len(problem.solve_kkt()) == 2  # (w, outer)
    assert len(problem.solve_pcg()) == 2  # (w, inner)
    assert len(problem.solve_clarabel()) == 2  # (w, iters)
    assert len(problem.solve_osqp()) == 2
    assert len(problem.solve_cvxpy()) == 2
    assert len(problem.solve_proximal()) == 2
    assert len(problem.solve_fista()) == 2


def test_cvxpy_reference_agrees_with_cg(problem):
    # The CVXPY reference solves the same program via its modeling layer, so
    # both backends must agree with the exact CG solution to solver tolerance.
    w_cg = problem.solve_cg()[0]
    w_clarabel, it_c = problem.solve_cvxpy()
    w_osqp, it_o = problem.solve_cvxpy(backend="osqp")
    assert np.linalg.norm(w_clarabel - w_cg) < 1e-3
    assert np.linalg.norm(w_osqp - w_cg) < 1e-3
    assert abs(w_clarabel.sum() - 1.0) < 1e-6
    assert it_c >= 1 and it_o >= 1


def test_proximal_and_fista_are_the_simplex_baseline(problem):
    # Both first-order hooks route to Duchi's simplex FISTA (documented alias).
    assert np.linalg.norm(problem.solve_proximal()[0] - problem.solve_fista()[0]) < 1e-9


# ---------------------------------------------------------------------------
# Warm starts
# ---------------------------------------------------------------------------


def test_cg_warm_reduces_outer_iterations(problem):
    w_cold, outer_cold, _inner, warm = problem.solve_cg_warm()
    w_warm, outer_warm, _inner2, _warm2 = problem.solve_cg_warm(warm_start=warm)
    assert np.linalg.norm(w_warm - w_cold) < 1e-8
    assert outer_warm <= outer_cold


def test_kkt_warm_round_trip(problem):
    w_cold, _outer, warm = problem.solve_kkt_warm()
    w_warm, outer_warm, _warm2 = problem.solve_kkt_warm(warm_start=warm)
    assert np.linalg.norm(w_warm - w_cold) < 1e-8
    assert outer_warm <= 2


# ---------------------------------------------------------------------------
# Balance systems (B w = c) and the return tilt (rho, mu)
# ---------------------------------------------------------------------------


def test_balance_system_satisfied(returns):
    alpha, target = lw_alpha_and_target(returns)
    n = returns.shape[1]
    b_eq = np.zeros((2, n))
    b_eq[0, : n // 2] = 1.0
    b_eq[1, n // 2 :] = 1.0
    c_eq = np.array([0.6, 0.4])
    prob = MinVarProblem(returns, alpha=alpha, target=target, B=b_eq, c=c_eq)
    w_cg = prob.solve_cg()[0]
    w_cla = prob.solve_clarabel()[0]
    assert np.abs(b_eq @ w_cg - c_eq).max() < 1e-8
    assert w_cg.min() >= -1e-9
    assert np.linalg.norm(w_cg - w_cla) < 1e-3


def test_first_order_baseline_rejects_balance_system(returns):
    n = returns.shape[1]
    b_eq = np.ones((1, n))
    prob = MinVarProblem(returns, B=b_eq, c=np.array([1.0]))
    with pytest.raises(NotImplementedError):
        prob.solve_proximal()


def test_return_tilt_changes_solution(returns):
    alpha, target = lw_alpha_and_target(returns)
    mu = np.random.default_rng(3).uniform(0.0, 1e-3, returns.shape[1])
    base = MinVarProblem(returns, alpha=alpha, target=target)
    tilted = MinVarProblem(returns, alpha=alpha, target=target, rho=5.0, mu=mu)
    w0 = base.solve_cg()[0]
    w1, _, _ = tilted.solve_cg()
    assert abs(w1.sum() - 1.0) < 1e-8
    assert np.linalg.norm(w1 - w0) > 1e-4  # the tilt actually moved the portfolio
    # cross-check the tilted problem against a baseline
    assert np.linalg.norm(w1 - tilted.solve_clarabel()[0]) < 1e-3


def test_b_without_c_raises(returns):
    with pytest.raises(ValueError, match="together"):
        MinVarProblem(returns, B=np.ones((1, returns.shape[1])))


# ---------------------------------------------------------------------------
# RMT low-rank target: Woodbury (FactorOperator) == dense Cholesky
# ---------------------------------------------------------------------------


def test_rmt_woodbury_matches_cholesky(returns):
    target, lr_factors, k, alpha = rmt_target_and_alpha(returns)
    assert alpha == 1.0
    assert k >= 1
    woodbury = MinVarProblem(returns, alpha=1.0, target=target, target_lr=lr_factors).solve_kkt()[0]
    cholesky = MinVarProblem(returns, alpha=1.0, target=target).solve_kkt()[0]
    assert np.linalg.norm(woodbury - cholesky) < 1e-8
    assert abs(woodbury.sum() - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Shrinkage utilities
# ---------------------------------------------------------------------------


def test_lw_alpha_in_unit_interval(returns):
    alpha, target = lw_alpha_and_target(returns)
    n = returns.shape[1]
    assert 0.0 <= alpha <= 1.0
    assert target.shape == (n, n)
    # scaled-identity target: bar_lambda * I
    assert np.allclose(target, np.diag(np.diag(target)))
    assert np.allclose(np.diag(target), target[0, 0])


def test_oas_alpha_in_unit_interval(returns):
    alpha, _ = oas_alpha_and_target(returns)
    assert 0.0 <= alpha <= 1.0


def test_lw_hard_passes_alpha_through(returns):
    alpha, target = lw_alpha_and_target_hard(returns, alpha=0.5)
    assert alpha == 0.5
    _, lw_target = lw_alpha_and_target(returns)
    assert np.allclose(target, lw_target)  # same scaled-identity target


def test_rmt_correlation_cleaning(returns):
    target, (mu_bar, u_k, delta_k), k, alpha = rmt_target_and_alpha(returns)
    n = returns.shape[1]
    assert alpha == 1.0
    assert u_k.shape == (n, k)
    assert delta_k.shape == (k,)
    assert np.all(delta_k > 0)  # signal eigenvalues exceed the noise floor
    # C0 is scalar-identity-plus-low-rank on the correlation: min eigenvalue is mu_bar
    assert np.linalg.eigvalsh(target).min() == pytest.approx(mu_bar, rel=1e-6)
    assert np.allclose(target, mu_bar * np.eye(n) + u_k @ np.diag(delta_k) @ u_k.T)
    # cleaning is on the correlation, not the covariance (unit-scale vs variance-scale diagonal)
    cov = returns.T @ returns / returns.shape[0]
    assert not np.allclose(np.diag(target), np.diag(cov))


# ---------------------------------------------------------------------------
# Operator-construction branches (exercise every A = 2*Sigma path)
# ---------------------------------------------------------------------------


def test_wrong_shape_balance_matrix_raises(returns):
    n = returns.shape[1]
    with pytest.raises(ValueError, match="B must have shape"):
        MinVarProblem(returns, B=np.ones((1, n + 1)), c=np.array([1.0]))


def test_low_rank_target_matrix_free_cg(returns):
    # target_lr feeds a FactorOperator into the composite (cg/pcg) operator.
    target, lr, _k, _a = rmt_target_and_alpha(returns)
    w_cg = MinVarProblem(returns, alpha=1.0, target=target, target_lr=lr).solve_cg()[0]
    w_dense = MinVarProblem(returns, alpha=1.0, target=target).solve_cg()[0]
    assert abs(w_cg.sum() - 1.0) < 1e-8
    assert np.linalg.norm(w_cg - w_dense) < 1e-6


def test_low_rank_target_dense_exact_below_alpha_one(returns):
    # target_lr with alpha < 1 keeps the data term, so the exact path densifies
    # 2*Sigma (the _dense_matrix low-rank branch) rather than using Woodbury.
    target, lr, _k, _a = rmt_target_and_alpha(returns)
    w = MinVarProblem(returns, alpha=0.5, target=target, target_lr=lr).solve_kkt()[0]
    w_ref = MinVarProblem(returns, alpha=0.5, target=target).solve_kkt()[0]
    assert np.linalg.norm(w - w_ref) < 1e-8  # low-rank and dense targets coincide


def test_no_target_exact_uses_gram_operator(returns):
    # No shrinkage: the exact operator is a GramOperator on the scaled data.
    w, outer = MinVarProblem(returns).solve_kkt()
    assert abs(w.sum() - 1.0) < 1e-8
    assert w.min() >= -1e-9


def test_cvxpy_reconstructs_low_rank_target(returns):
    # solve_cvxpy with only low-rank factors (no dense target) reconstructs the
    # dense target from them for the Cholesky split.
    _target, lr, _k, _a = rmt_target_and_alpha(returns)
    w, iters = MinVarProblem(returns, alpha=1.0, target=None, target_lr=lr).solve_cvxpy()
    assert abs(w.sum() - 1.0) < 1e-6
    assert iters >= 1


def test_cvxpy_no_target(returns):
    # No shrinkage target: the CVXPY objective is the plain least-squares form.
    w, iters = MinVarProblem(returns).solve_cvxpy()
    assert abs(w.sum() - 1.0) < 1e-6
    assert iters >= 1


def test_cvxpy_return_tilt(returns):
    # rho != 0 adds the -rho * mu^T w return term to the CVXPY objective.
    alpha, target = lw_alpha_and_target(returns)
    mu = np.random.default_rng(4).uniform(0.0, 1e-3, returns.shape[1])
    w, _ = MinVarProblem(returns, alpha=alpha, target=target, rho=5.0, mu=mu).solve_cvxpy()
    assert abs(w.sum() - 1.0) < 1e-6
    assert np.linalg.norm(w - MinVarProblem(returns, alpha=alpha, target=target).solve_cvxpy()[0]) > 1e-4
