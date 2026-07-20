"""Matrix-free bound-constrained-QP baselines for "Non-Negative Conjugate Gradients".

The paper distinguishes its complementary-basis loop from the *feasible-point*
active-set family (Bertsekas' projected Newton, More-Toraldo GPCG, Dostal's
MPRGP). This module supplies MPRGP itself as a benchmark competitor, so the
comparison in Section 7 is head-to-head rather than only verbal.

MPRGP -- Modified Proportioning with Reduced Gradient Projections (Dostal and
Schoberl, 2005) -- solves

    min_{x >= 0}  1/2 x^T A x - b^T x,   A SPD,

using only matrix-vector products with ``A``, so it runs in the same
matrix-free regime as the paper's method. It interleaves conjugate-gradient
steps on the current free face with expansion steps (a projected fixed-step
gradient move that grows the active set) and proportioning steps (a
chopped-gradient move that releases active constraints), switched by Dostal's
proportioning test. It is a feasible-point method: every iterate satisfies
``x >= 0``.

``osqp`` and Clarabel are imported by the benchmark scripts directly; only
MPRGP needs implementing here.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BcqpResult:
    """Result of a bound-constrained-QP solve."""

    x: np.ndarray
    iters: int
    converged: bool


def power_iteration_lambda_max(matvec, n, iters=100, seed=0):
    """Estimate the largest eigenvalue of the SPD operator ``matvec``."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = matvec(v)
        lam = float(v @ w)
        nw = np.linalg.norm(w)
        if nw == 0.0:
            return lam
        v = w / nw
    return lam


def mprgp(matvec, b, n, lam_max=None, tol=1e-8, maxit=100000, gamma=1.0, x0=None):
    """MPRGP for ``min_{x>=0} 1/2 x^T A x - b^T x`` with ``A`` given by ``matvec``.

    Args:
        matvec: linear operator ``v -> A v`` for the SPD matrix ``A``.
        b: right-hand side (length ``n``).
        n: problem dimension.
        lam_max: an upper estimate of ``lambda_max(A)``; the fixed step is
            ``abar = 1 / lam_max`` in ``(0, 2/lambda_max]``. Estimated by power
            iteration if omitted.
        tol: stop when the projected-gradient norm falls below ``tol * ||b||``.
        maxit: iteration cap.
        gamma: proportioning constant ``Gamma > 0`` (Dostal's ``Gamma``).
        x0: feasible starting point (default ``0``).

    Returns:
        A :class:`BcqpResult`; ``iters`` counts outer MPRGP iterations.
    """
    b = np.asarray(b, dtype=float)
    if lam_max is None:
        lam_max = power_iteration_lambda_max(matvec, n)
    abar = 1.0 / lam_max                                  # fixed step in (0, 2/lambda_max]

    x = np.zeros(n) if x0 is None else np.maximum(np.asarray(x0, dtype=float), 0.0)
    g = matvec(x) - b                                     # gradient A x - b

    def free_grad(x, g):
        return np.where(x > 0.0, g, 0.0)                 # phi

    def chopped_grad(x, g):
        return np.where(x <= 0.0, np.minimum(g, 0.0), 0.0)  # beta

    phi = free_grad(x, g)
    p = phi.copy()
    stop = tol * max(np.linalg.norm(b), 1.0)

    it = 0
    while it < maxit:
        beta = chopped_grad(x, g)
        if np.linalg.norm(phi + beta) <= stop:
            return BcqpResult(x=x, iters=it, converged=True)
        it += 1

        # reduced free gradient phi_tilde_i = min(phi_i, x_i / abar) on the free set
        phi_tilde = np.where(x > 0.0, np.minimum(phi, x / abar), 0.0)
        if beta @ beta <= gamma * gamma * (phi_tilde @ phi):
            # Proportional: CG step, or expansion when the CG step is infeasible.
            Ap = matvec(p)
            pAp = float(p @ Ap)
            acg = float(g @ p) / pAp
            pos = p > 0.0
            afeas = np.min(x[pos] / p[pos]) if np.any(pos) else np.inf
            if acg <= afeas:
                x = x - acg * p
                g = g - acg * Ap
                phi = free_grad(x, g)
                beta_cg = float(phi @ Ap) / pAp
                p = phi - beta_cg * p
            else:
                x = np.maximum(x - afeas * p, 0.0)       # move to the boundary
                g = matvec(x) - b
                phi = free_grad(x, g)
                x = np.maximum(x - abar * phi, 0.0)      # projected fixed-step gradient
                g = matvec(x) - b
                phi = free_grad(x, g)
                p = phi.copy()
        else:
            # Proportioning: chopped-gradient step releases active constraints.
            d = beta
            Ad = matvec(d)
            acg = float(g @ d) / float(d @ Ad)
            x = x - acg * d
            g = g - acg * Ad
            phi = free_grad(x, g)
            p = phi.copy()

    return BcqpResult(x=x, iters=it, converged=False)
