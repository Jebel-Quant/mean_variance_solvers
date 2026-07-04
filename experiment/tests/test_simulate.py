"""Tests for the vendored ``simulate_equity_returns`` factor-model generator."""

from __future__ import annotations

import doctest

import numpy as np

import simulate
from simulate import simulate_equity_returns


def test_shape_and_orientation():
    x = simulate_equity_returns(100, 200, rng=0)
    assert x.shape == (200, 100)  # (T, n)


def test_columns_are_demeaned():
    x = simulate_equity_returns(60, 300, rng=1)
    assert np.abs(x.mean(axis=0)).max() < 1e-13


def test_reproducible_with_seed():
    a = simulate_equity_returns(40, 120, rng=7)
    b = simulate_equity_returns(40, 120, rng=7)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = simulate_equity_returns(40, 120, rng=1)
    b = simulate_equity_returns(40, 120, rng=2)
    assert not np.array_equal(a, b)


def test_market_factor_dominates_spectrum():
    # A market factor with universally positive loadings should produce one
    # eigenvalue far above the idiosyncratic bulk.
    x = simulate_equity_returns(120, 600, k=5, rng=3)
    cov = (x.T @ x) / x.shape[0]
    eigs = np.sort(np.linalg.eigvalsh(cov))[::-1]
    assert eigs[0] > 5 * eigs[1]


def test_default_factor_count():
    # k defaults to max(3, n // 10); n=200 -> 20 factors, still a valid matrix.
    x = simulate_equity_returns(200, 400, rng=0)
    assert x.shape == (400, 200)


def test_docstring_examples():
    results = doctest.testmod(simulate, verbose=False)
    assert results.failed == 0
