"""Test-problem generators for the "Non-Negative Conjugate Gradients" study.

The planted-optimum synthetic SPD family and Hansen's ill-posed NNLS test
problems (``shaw``, ``phillips``) that the experiment_nncg* scripts solve. These
live here because they are inputs to the study, not part of the :mod:`nncg`
solver package the scripts now call — the solvers themselves come from
``nncg`` (and ``nncg.krylov``), with dense operands wrapped in
``cvx.linalg.DenseOperator``.
"""

from __future__ import annotations

import numpy as np


def make_problem(n, kappa, support_frac=0.5, seed=0):
    """A = Q diag(lambda) Q^T with condition number kappa and a planted optimum.

    Returns (A, b, x_star, s_star). The spectrum is geometric on [1, kappa], so
    lambda_min = 1 and lambda_max = kappa. A support of size round(support_frac*n)
    is chosen; x* is positive there and zero elsewhere, s* is zero there and
    positive elsewhere, and b = A x* - s*. Then s* = A x* - b is the reduced
    gradient, complementary slackness holds coordinate-wise, and x* is the unique
    minimiser of min_{x>=0} 1/2 x^T A x - b^T x.
    """
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, kappa, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = (Q * eig) @ Q.T
    A = 0.5 * (A + A.T)

    k = max(1, int(round(support_frac * n)))
    perm = rng.permutation(n)
    supp = perm[:k]

    x_star = np.zeros(n)
    x_star[supp] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    off = perm[k:]
    s_star[off] = rng.uniform(0.5, 1.5, size=n - k)
    b = A @ x_star - s_star
    return A, b, x_star, s_star


def shaw(n):
    """The `shaw` test problem from Hansen's Regularization Tools.

    A one-dimensional image-restoration model: a first-kind Fredholm integral
    equation discretised by the midpoint rule on (-pi/2, pi/2). Returns the
    symmetric kernel matrix M (n x n), the exact solution x (two Gaussian
    humps, strictly positive, hence a genuine non-negative target), and the
    right-hand side d = M x. M is numerically rank-deficient, so the Gram
    operator A = M^T M is numerically singular; see experiment_nncg_regu.py.
    """
    h = np.pi / n
    i = np.arange(1, n + 1)
    s = (i - 0.5) * h - np.pi / 2                    # midpoints in (-pi/2, pi/2)
    S, T = np.meshgrid(s, s, indexing="ij")
    co = np.cos(S) + np.cos(T)
    u = np.pi * (np.sin(S) + np.sin(T))
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.where(u == 0.0, 1.0, np.sin(u) / u)
    M = (co ** 2) * (sinc ** 2) * h
    a1, c1, t1 = 2.0, 6.0, 0.8                        # Hansen's canonical params
    a2, c2, t2 = 1.0, 2.0, -0.5
    x = a1 * np.exp(-c1 * (s - t1) ** 2) + a2 * np.exp(-c2 * (s - t2) ** 2)
    return M, x, M @ x


def phillips(n):
    """The `phillips` test problem from Hansen's Regularization Tools.

    A deconvolution with a symmetric banded Toeplitz kernel M. The exact
    solution phi(t) = 1 + cos(pi t/3) on |t| <= 3 (zero outside) is
    non-negative with a known support, so about half the components sit at the
    bound: a genuinely large, structured active set, and (with light noise) a
    signal the solver should recover. Returns (M, x, d = M x); n must be a
    multiple of 4.
    """
    if n % 4 != 0:
        raise ValueError("phillips: n must be a multiple of 4")
    h = 12.0 / n
    n4 = n // 4
    c = np.cos(np.arange(-1, n4 + 1) * 4.0 * np.pi / n)
    r = np.zeros(n)
    r[:n4] = h + 9.0 / (h * np.pi ** 2) * (2 * c[1:n4 + 1] - c[:n4] - c[2:n4 + 2])
    r[n4] = h / 2.0 + 9.0 / (h * np.pi ** 2) * (np.cos(4.0 * np.pi / n) - 1.0)
    diff = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    M = r[diff]                                       # symmetric Toeplitz kernel
    t = (np.arange(1, n + 1) - 0.5) * h - 6.0         # grid on (-6, 6)
    x = np.where(np.abs(t) < 3.0, 1.0 + np.cos(np.pi * t / 3.0), 0.0)
    return M, x, M @ x
