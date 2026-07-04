"""Shared pytest fixtures and path setup for the experiment test suite.

The modules under test (``baselines``, ``minvar``, ``simulate``, ``util``) live
in the ``experiment/`` directory one level up, exactly as the scripts import
them; putting that directory on ``sys.path`` lets the tests import them the same
way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))


@pytest.fixture
def planted_nnqp():
    """A small planted-optimum NNQP ``min_{x>=0} 1/2 x^T A x - b^T x``.

    Returns ``(A, b, x_star)`` where ``A`` is SPD with condition number ~50 and
    ``x_star >= 0`` is the unique minimiser: a support is positive, its
    complement zero, and ``b = A x_star - s_star`` with ``s_star`` the positive
    reduced gradient off the support (so complementary slackness holds exactly).
    """
    rng = np.random.default_rng(0)
    n = 12
    eig = np.geomspace(1.0, 50.0, n)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * eig) @ q.T
    a = 0.5 * (a + a.T)

    supp = rng.permutation(n)[:6]
    x_star = np.zeros(n)
    x_star[supp] = rng.uniform(0.5, 1.5, size=6)
    off = np.array([i for i in range(n) if i not in supp])
    s_star = np.zeros(n)
    s_star[off] = rng.uniform(0.5, 1.5, size=off.size)
    b = a @ x_star - s_star
    return a, b, x_star


@pytest.fixture
def returns():
    """A small demeaned synthetic return matrix (T=200, n=40)."""
    from simulate import simulate_equity_returns

    return simulate_equity_returns(40, 200, rng=0)
