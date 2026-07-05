"""Tests for the nncg test-problem generators (make_problem, shaw, phillips)."""

from __future__ import annotations

import numpy as np
import pytest

from nncg_note.problems import make_problem, phillips, shaw


def test_make_problem_planted_optimum_is_feasible_and_complementary():
    a, b, x_star, s_star = make_problem(30, kappa=1e3, seed=0)
    assert np.allclose(a, a.T)  # symmetric
    assert x_star.min() >= 0.0 and s_star.min() >= 0.0
    assert np.allclose(x_star * s_star, 0.0)  # complementary slackness
    assert np.allclose(b, a @ x_star - s_star)


def test_shaw_returns_consistent_triple():
    m, x, d = shaw(64)
    assert m.shape == (64, 64)
    assert np.allclose(m, m.T)
    assert x.min() > 0.0  # two Gaussian humps, strictly positive
    assert np.allclose(d, m @ x)


def test_phillips_returns_consistent_triple():
    m, x, d = phillips(64)
    assert m.shape == (64, 64)
    assert x.min() >= 0.0
    assert np.allclose(d, m @ x)


def test_phillips_requires_multiple_of_four():
    with pytest.raises(ValueError, match="multiple of 4"):
        phillips(30)
