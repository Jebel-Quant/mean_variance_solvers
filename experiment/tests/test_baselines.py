"""Tests for the vendored alternative NNQP solvers in ``baselines.py``.

Each solver of the bound-only program ``min_{x>=0} 1/2 x^T A x - b^T x`` is
checked against the planted optimum and cross-checked against the others; the
equality-augmented (``B x = c``) path of OSQP/Clarabel and the exact simplex
projection are exercised separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from cvx.linalg import DenseOperator

import baselines
from baselines import (
    BaselineResult,
    _project_simplex,
    ones_row,
    solve_clarabel,
    solve_duchi,
    solve_fista,
    solve_lawson_hanson,
    solve_osqp,
)


def _op(a):
    """Wrap a dense SPD array as the SymmetricOperator the baselines expect."""
    return DenseOperator(a)


# ---------------------------------------------------------------------------
# Bound-only program: every solver recovers the planted optimum
# ---------------------------------------------------------------------------

BOUND_SOLVERS = [solve_osqp, solve_clarabel, solve_lawson_hanson, solve_fista]


@pytest.mark.parametrize("solver", BOUND_SOLVERS, ids=lambda s: s.__name__)
def test_recovers_planted_optimum(planted_nnqp, solver):
    a, b, x_star = planted_nnqp
    res = solver(_op(a), b)
    assert isinstance(res, BaselineResult)
    assert np.linalg.norm(res.x - x_star) < 1e-3
    assert res.iters >= 0
    assert res.time_s >= 0.0
    # feasibility: non-negativity holds to tolerance
    assert res.x.min() > -1e-6


def test_bound_solvers_agree(planted_nnqp):
    a, b, _ = planted_nnqp
    xs = [solver(_op(a), b).x for solver in BOUND_SOLVERS]
    ref = xs[0]
    for x in xs[1:]:
        assert np.linalg.norm(x - ref) < 1e-3


def test_lawson_hanson_matches_direct_solution(planted_nnqp):
    # On the planted support the KKT solution is the unconstrained solve there.
    a, b, x_star = planted_nnqp
    res = solve_lawson_hanson(_op(a), b)
    assert res.status == "solved"
    assert np.linalg.norm(res.x - x_star) < 1e-8  # active-set is essentially exact


# ---------------------------------------------------------------------------
# Equality-augmented program: OSQP / Clarabel with B x = c
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", [solve_osqp, solve_clarabel], ids=lambda s: s.__name__)
def test_equality_constraint_satisfied(planted_nnqp, solver):
    a, b, _ = planted_nnqp
    n = a.shape[0]
    b_eq = ones_row(n)  # 1^T x = 1 budget
    c_eq = np.array([1.0])
    res = solver(_op(a), b, b_eq=b_eq, c_eq=c_eq)
    assert abs(float((b_eq @ res.x)[0]) - 1.0) < 1e-6
    assert res.x.min() > -1e-6
    assert res.lam is not None and res.lam.shape == (1,)


def test_osqp_clarabel_agree_on_budget(planted_nnqp):
    a, b, _ = planted_nnqp
    n = a.shape[0]
    b_eq, c_eq = ones_row(n), np.array([1.0])
    x_osqp = solve_osqp(_op(a), b, b_eq=b_eq, c_eq=c_eq).x
    x_cla = solve_clarabel(_op(a), b, b_eq=b_eq, c_eq=c_eq).x
    assert np.linalg.norm(x_osqp - x_cla) < 1e-4


@pytest.mark.parametrize("solver", [solve_osqp, solve_clarabel], ids=lambda s: s.__name__)
def test_b_eq_without_c_eq_raises(planted_nnqp, solver):
    a, b, _ = planted_nnqp
    with pytest.raises(ValueError, match="together"):
        solver(_op(a), b, b_eq=ones_row(a.shape[0]), c_eq=None)


# ---------------------------------------------------------------------------
# Simplex projection and Duchi FISTA
# ---------------------------------------------------------------------------


def test_project_simplex_is_feasible():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(50)
    for beta in (1.0, 3.5):
        p = _project_simplex(v, beta)
        assert p.min() >= 0.0
        assert abs(p.sum() - beta) < 1e-10


def test_project_simplex_idempotent_on_simplex():
    v = np.array([0.2, 0.3, 0.5])
    assert np.allclose(_project_simplex(v, 1.0), v)


def test_project_simplex_rejects_nonpositive_beta():
    with pytest.raises(ValueError, match="beta"):
        _project_simplex(np.ones(3), 0.0)


def test_duchi_lands_on_simplex(planted_nnqp):
    a, b, _ = planted_nnqp
    res = solve_duchi(_op(a), b, beta=1.0)
    assert abs(res.x.sum() - 1.0) < 1e-6
    assert res.x.min() >= -1e-9


def test_duchi_matches_clarabel_on_simplex(planted_nnqp):
    # Clarabel with the 1^T x = 1 equality solves the same simplex program.
    a, b, _ = planted_nnqp
    n = a.shape[0]
    x_duchi = solve_duchi(_op(a), b, beta=1.0).x
    x_cla = solve_clarabel(_op(a), b, b_eq=ones_row(n), c_eq=np.array([1.0])).x
    assert np.linalg.norm(x_duchi - x_cla) < 1e-3


def test_ones_row_shape():
    r = ones_row(7)
    assert r.shape == (1, 7)
    assert np.all(r == 1.0)


def test_fista_step_override_still_converges(planted_nnqp):
    a, b, x_star = planted_nnqp
    lam_max = float(np.linalg.eigvalsh(a)[-1])
    res = solve_fista(_op(a), b, step=1.0 / lam_max)
    assert np.linalg.norm(res.x - x_star) < 1e-3
