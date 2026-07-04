"""Core algorithms for the "Non-Negative Conjugate Gradients" note.

Implements, directly from the paper's definitions and with NumPy only:

  * make_problem     -- the planted-optimum synthetic SPD family (Section 7.2)
  * shaw / phillips  -- Hansen's ill-posed NNLS test problems (Section 7.2)
  * cg / pcg         -- (Jacobi-preconditioned) conjugate gradients (Prop. 3.1)
  * solve_nnqp       -- the active-set / block-principal-pivoting loop (Algorithm 1)
  * solve_nnqp_op    -- its matrix-free variant, A supplied as an operator (Section 3)
  * solve_nnqp_eq    -- its equality-augmented variant, Bx = c (Section 3)
  * kkt_violation    -- the KKT certificate of Section 2

Shared by experiment_nncg.py (the synthetic study) and
experiment_nncg_bench.py (the external-solver benchmark).
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


def cg(matvec, rhs, tol=1e-8, maxit=100000, x0=None):
    """Solve (matvec) x = rhs by conjugate gradients. Returns (x, iterations).

    x0 warm-starts the iteration: the initial residual is rhs - matvec(x0), so
    a good guess cuts the count by the log of the initial error (Section 5.4).
    """
    if x0 is None:
        x = np.zeros_like(rhs)
        r = rhs.copy()
    else:
        x = x0.copy()
        r = rhs - matvec(x)
    p = r.copy()
    rs = float(r @ r)
    bnorm = float(np.linalg.norm(rhs))
    if bnorm == 0.0:
        return x, 0
    for it in range(1, maxit + 1):
        Ap = matvec(p)
        alpha = rs / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) / bnorm <= tol:
            return x, it
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, maxit


def pcg(matvec, rhs, dinv, tol=1e-8, maxit=100000):
    """Jacobi-preconditioned CG: M^{-1} = diag(dinv). Returns (x, iterations)."""
    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = dinv * r
    p = z.copy()
    rz = float(r @ z)
    bnorm = float(np.linalg.norm(rhs))
    if bnorm == 0.0:
        return x, 0
    for it in range(1, maxit + 1):
        Ap = matvec(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) / bnorm <= tol:
            return x, it
        z = dinv * r
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit


def solve_nnqp(A, b, tol=1e-8, cg_tol=1e-10, p_max=3, inner="cg", track=False,
               cg_maxit=100000, max_outer=None, warm=None):
    """Minimise 1/2 x^T A x - b^T x over x >= 0 by the active-set loop.

    Each free-block solve is matrix-free CG on v -> A_F v; A is never refactorised.
    inner="exact" swaps in a direct dense solve (used to verify the inexactness
    lemma: the CG-driven loop must visit the same free sets); inner="pcg" uses
    Jacobi-preconditioned CG on diag(A_F). track=True records the free-set
    trajectory. max_outer caps the outer loop (used by the rank-deficient panel,
    where alpha=0 is expected to fail); "converged" reports whether the KKT exit
    was reached. warm=(free_mask, x_prev) starts the loop from a previous solve's
    free set and warm-starts each CG call from x_prev restricted to the current
    free set (Section 5.4). Returns a dict of x and the loop counters, plus the
    final free mask under "free".
    """
    n = len(b)
    if warm is None:
        free = np.ones(n, dtype=bool)  # F = {1..n} initially
        x_guess = None
    else:
        free = warm[0].copy()
        x_guess = warm[1]
    x = np.zeros(n)
    N_bar = n + 1
    patience = p_max
    outer = inner_total = fallback = 0
    traj = [] if track else None
    converged = True

    while True:
        if max_outer is not None and outer >= max_outer:
            converged = False
            break
        idx = np.flatnonzero(free)
        if track:
            traj.append(tuple(idx.tolist()))
        A_FF = A[np.ix_(idx, idx)]  # sliced view drives the matrix-free mat-vec
        x0 = x_guess[idx] if x_guess is not None else None
        if inner == "exact":
            xF, k_step = np.linalg.solve(A_FF, b[idx]), 1
        elif inner == "pcg":
            xF, k_step = pcg(lambda v: A_FF @ v, b[idx],
                             1.0 / np.diag(A_FF), tol=cg_tol, maxit=cg_maxit)
        else:
            xF, k_step = cg(lambda v: A_FF @ v, b[idx], tol=cg_tol,
                            maxit=cg_maxit, x0=x0)
        outer += 1
        inner_total += k_step

        x = np.zeros(n)
        x[idx] = xF
        if x_guess is not None:
            x_guess = x  # warm mode: newest iterate seeds the next reduced solve
        s = A @ x - b  # reduced gradient s_i = (Ax)_i - b_i

        prim = np.flatnonzero(free & (x < -tol))            # D: free but negative
        dual = np.flatnonzero((~free) & (s < -tol))         # V: bound but s<0
        viol = np.concatenate([prim, dual])
        N = viol.size
        if N == 0:
            break  # KKT satisfied -> unique global minimiser

        if N < N_bar or patience > 0:  # fast path: progress, or patience remains
            if N < N_bar:
                N_bar = N
                patience = p_max
            else:
                patience -= 1
            free[prim] = False  # batch exchange: drop all D, add all V
            free[dual] = True
        else:  # anti-cycling fallback: single Bland least-index pivot
            fallback += 1
            i_star = int(viol.min())
            free[i_star] = not free[i_star]

    return {"x": x, "outer": outer, "inner": inner_total, "fallback": fallback,
            "traj": traj, "converged": converged, "free": free}


def solve_nnqp_op(matvec, b, tol=1e-8, cg_tol=1e-10, p_max=3,
                  cg_maxit=100000, max_outer=None):
    """Matrix-free active-set loop: identical logic to solve_nnqp, but A is
    supplied only as an operator `matvec(x) = A @ x` and is never assembled.

    The free-block solve restricts the operator to the current free set by
    embedding the reduced vector into full space, applying matvec, and reading
    back the free coordinates -- so no n-by-n matrix, and no O(n^2) storage,
    is ever formed. This is the operator abstraction of Section 3: the same
    loop and termination proof run on any object that supplies a mat-vec (a
    Gram operator M^T(M v), a Kronecker blur, a diagonal-plus-low-rank factor).
    """
    n = len(b)
    free = np.ones(n, dtype=bool)
    x = np.zeros(n)
    N_bar = n + 1
    patience = p_max
    outer = inner_total = fallback = 0
    converged = True

    while True:
        if max_outer is not None and outer >= max_outer:
            converged = False
            break
        idx = np.flatnonzero(free)

        def matvec_F(vF, idx=idx):                    # A_F v, matrix-free
            v = np.zeros(n)
            v[idx] = vF
            return matvec(v)[idx]

        xF, k_step = cg(matvec_F, b[idx], tol=cg_tol, maxit=cg_maxit)
        outer += 1
        inner_total += k_step

        x = np.zeros(n)
        x[idx] = xF
        s = matvec(x) - b                             # reduced gradient

        prim = np.flatnonzero(free & (x < -tol))
        dual = np.flatnonzero((~free) & (s < -tol))
        viol = np.concatenate([prim, dual])
        N = viol.size
        if N == 0:
            break
        if N < N_bar or patience > 0:
            if N < N_bar:
                N_bar = N
                patience = p_max
            else:
                patience -= 1
            free[prim] = False
            free[dual] = True
        else:
            fallback += 1
            i_star = int(viol.min())
            free[i_star] = not free[i_star]

    return {"x": x, "outer": outer, "inner": inner_total, "fallback": fallback,
            "converged": converged, "free": free}


def solve_nnqp_eq(A, b, B, c, tol=1e-8, cg_tol=1e-10, p_max=3):
    """Active-set loop with a general linear equality constraint B x = c.

    On each free set the saddle system
        A_F x_F - B_F^T lambda = b_F,   B_F x_F = c
    is solved by eliminating lambda in R^p through the p-by-p Schur complement
    S = B_F A_F^{-1} B_F^T (Section 3): v0 = A_F^{-1} b_F and the columns of
    V1 = A_F^{-1} B_F^T are p+1 matrix-free CG solves sharing the operator A_F,
    then lambda solves S lambda = B_F v0 - c and x_F = v0 + V1 lambda. The single
    normalisation 1^T x = beta is the p=1, B=1^T case. The reduced gradient used
    by the dual test is s = A x - b - B^T lambda.
    """
    n = len(b)
    p = B.shape[0]
    free = np.ones(n, dtype=bool)
    x = np.zeros(n)
    lam = np.zeros(p)
    N_bar = n + 1
    patience = p_max
    outer = inner_total = fallback = 0

    while True:
        idx = np.flatnonzero(free)
        A_FF = A[np.ix_(idx, idx)]
        B_F = B[:, idx]
        v0, k0 = cg(lambda v: A_FF @ v, b[idx], tol=cg_tol)
        V1 = np.zeros((idx.size, p))
        k_cols = 0
        for j in range(p):
            V1[:, j], kj = cg(lambda v: A_FF @ v, B_F[j], tol=cg_tol)
            k_cols += kj
        outer += 1
        inner_total += k0 + k_cols

        S = B_F @ V1                              # p-by-p Schur complement, SPD
        lam = np.linalg.solve(S, c - B_F @ v0)    # multiplier in R^p
        xF = v0 + V1 @ lam                        # x_F = A_F^{-1}(b_F + B_F^T lambda)
        x = np.zeros(n)
        x[idx] = xF
        s = A @ x - b - B.T @ lam                 # reduced gradient

        prim = np.flatnonzero(free & (x < -tol))
        dual = np.flatnonzero((~free) & (s < -tol))
        viol = np.concatenate([prim, dual])
        N = viol.size
        if N == 0:
            break
        if N < N_bar or patience > 0:
            if N < N_bar:
                N_bar = N
                patience = p_max
            else:
                patience -= 1
            free[prim] = False
            free[dual] = True
        else:
            fallback += 1
            i_star = int(viol.min())
            free[i_star] = not free[i_star]

    return {"x": x, "lam": lam, "outer": outer, "inner": inner_total, "fallback": fallback}


def kkt_violation(A, b, x):
    """max of negativity and complementarity violations of the KKT system."""
    s = A @ x - b
    return float(max(np.max(-x, initial=0.0), np.max(-s, initial=0.0),
                     np.max(np.abs(x * s), initial=0.0)))
