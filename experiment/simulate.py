"""Simulate equity returns from a latent factor model.

Vendored into the experiment folder so the figure/table scripts own their
synthetic-data generator directly rather than importing it from
``fast_minimum_variance.data``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_equity_returns"]


def simulate_equity_returns(
    n: int,
    T: int,  # noqa: N803
    *,
    k: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate a TxN demeaned equity return matrix with latent factor structure.

    Returns are generated from the model

        X = F @ B.T + E

    where F (TxK) are factor returns, B (NxK) are factor loadings, and E (TxN)
    is idiosyncratic noise.  The first factor is a market factor with universally
    positive loadings and high variance; the remaining k-1 factors are
    style/industry factors with sparse loadings.  This produces a covariance
    spectrum qualitatively similar to equity universes: a dominant market
    eigenvalue, a handful of secondary factor eigenvalues, and a long tail of
    near-equal idiosyncratic eigenvalues.

    Parameters
    ----------
    n:
        Number of assets.
    T:
        Number of time periods (trading days).
    k:
        Number of latent factors.  Defaults to ``max(3, n // 10)``.
    rng:
        Random state — a :class:`numpy.random.Generator`, an integer seed,
        or ``None`` (non-reproducible).

    Returns:
    -------
    X : ndarray of shape (T, n)
        Demeaned return matrix.  Each column has zero mean.

    Examples:
    --------
    >>> X = simulate_equity_returns(100, 200, k=5, rng=0)
    >>> X.shape
    (200, 100)
    >>> bool(abs(X.mean(axis=0)).max() < 1e-14)
    True
    """
    rng = np.random.default_rng(rng)
    if k is None:
        k = max(3, n // 10)

    # Factor volatilities (daily): market ~1 %, style factors ~0.5 %
    factor_vols = np.concatenate([[0.01], np.full(k - 1, 0.005)])

    # Factor returns: T x k
    F = rng.standard_normal((T, k)) * factor_vols  # noqa: N806

    # Factor loadings: n x k
    # Market: all assets have positive exposure in [0.4, 0.8]
    # Style:  sparse (~50 % non-zero), drawn from N(0, 0.2)
    B = np.zeros((n, k))  # noqa: N806
    B[:, 0] = rng.uniform(0.4, 0.8, size=n)
    for j in range(1, k):
        mask = rng.random(n) < 0.5
        B[mask, j] = rng.standard_normal(int(mask.sum())) * 0.2

    # Idiosyncratic volatility: uniform in [0.5 %, 1.5 %] per asset
    idio_vols = rng.uniform(0.005, 0.015, size=n)
    E = rng.standard_normal((T, n)) * idio_vols  # noqa: N806

    X: np.ndarray = F @ B.T + E  # noqa: N806
    X -= X.mean(axis=0)  # noqa: N806
    return X
