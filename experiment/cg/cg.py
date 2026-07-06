"""Self-contained min-variance problem builder for the experiment scripts.

Replaces the ``fast_minimum_variance`` dependency: the figure/table scripts now
build the long-only minimum-variance program here and solve it with only two
libraries — :mod:`nncg` (the matrix-free active-set NNQP method under study) and
:mod:`baselines` (the external comparison solvers) — over ``cvx.linalg``
operators, with shrinkage estimated through scikit-learn.

The problem is the equality-augmented non-negative quadratic program

    min_w  w^T Sigma w - rho * mu^T w    s.t.  B w = c,  w >= 0,

with ``Sigma = (1 - alpha)/T * X^T X + alpha * target`` the (optionally
shrunk) covariance and the default constraint the budget ``1^T w = 1``. In the
NNQP form ``min 1/2 x^T A x - b^T x`` this is ``A = 2*Sigma`` and ``b = rho*mu``;
the factor of two is folded into the operator coefficients so the minimiser is
identical.

:class:`CGProblem` mirrors the method names and return contracts of the
former ``fast_minimum_variance._MinVarProblem`` so the experiment scripts only
swap their import lines:

    solve_cg()             -> (w, outer, inner)      nncg, inner="cg"
    solve_pcg()            -> (w, iters)             nncg, inner="pcg"
    solve_kkt()            -> (w, outer)             nncg, inner="exact"
    solve_cg_warm(warm)    -> (w, outer, inner, warm)
    solve_kkt_warm(warm)   -> (w, outer, warm)
    solve_clarabel()       -> (w, iters)             baselines.solve_clarabel
    solve_osqp()           -> (w, iters)             baselines.solve_osqp
    solve_cvxpy(backend)   -> (w, iters)             CVXPY modeling -> Clarabel/OSQP
    solve_proximal()       -> (w, iters)             baselines.solve_duchi
    solve_fista()          -> (w, iters)             baselines.solve_duchi

``solve_cvxpy`` is the ground-truth reference the papers compare against: it
goes through CVXPY's problem-construction layer (hence its
problem-construction overhead relative to the direct-API ``solve_clarabel`` /
``solve_osqp`` baselines), then dispatches to the same Clarabel/OSQP backend.

The warm state is the ``(free_mask, x)`` pair :meth:`nncg.ActiveSetSolver.solve_eq`
consumes and :class:`nncg.Result` exposes as ``(.free, .x)``.

Note on the first-order rows: the min-variance program is simplex-constrained
(``1^T w = 1``), so its accelerated projected-gradient baseline is FISTA with
the exact simplex projection — :func:`baselines.solve_duchi`. The orthant-only
:func:`baselines.solve_fista` would return ``w = 0`` here, so both the
``solve_proximal`` and ``solve_fista`` hooks route to ``solve_duchi``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cvx.linalg import DenseOperator, FactorOperator, GramOperator, SumOperator
from numpy.typing import NDArray
from sklearn.covariance import ledoit_wolf, oas

import nncg
from common import baselines

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
Warm = tuple[NDArray[np.bool_], Vector]

__all__ = [
    "CGProblem",
    "lw_alpha_and_target",
    "lw_alpha_and_target_hard",
    "oas_alpha_and_target",
    "rmt_target_and_alpha",
]


# ---------------------------------------------------------------------------
# Shrinkage intensity and target utilities (pure NumPy + scikit-learn)
# ---------------------------------------------------------------------------


def lw_alpha_and_target(x: Matrix) -> tuple[float, Matrix]:
    """Return ``(alpha_lw, target)`` for Ledoit-Wolf scaled-identity shrinkage.

    ``x`` must be column-demeaned. The target is ``bar_lambda * I`` with
    ``bar_lambda = ||x||_F^2 / (n * T)`` the mean per-asset variance; the
    intensity is scikit-learn's Ledoit-Wolf estimate.
    """
    _, alpha = ledoit_wolf(x, assume_centered=True)
    t, n = x.shape
    bar_lam = float(np.linalg.norm(x, "fro") ** 2) / (n * t)
    return float(alpha), bar_lam * np.eye(n)


def lw_alpha_and_target_hard(x: Matrix, alpha: float = 0.5) -> tuple[float, Matrix]:
    """Return ``(alpha, target)`` for scaled-identity shrinkage at a fixed ``alpha``.

    Same ``bar_lambda * I`` target as :func:`lw_alpha_and_target`, but the
    intensity is passed in rather than estimated.
    """
    t, n = x.shape
    bar_lam = float(np.linalg.norm(x, "fro") ** 2) / (n * t)
    return alpha, bar_lam * np.eye(n)


def oas_alpha_and_target(x: Matrix) -> tuple[float, Matrix]:
    """Return ``(alpha_oas, target)`` for Oracle-Approximating-Shrinkage.

    The same ``bar_lambda * I`` target as Ledoit-Wolf, with scikit-learn's OAS
    intensity (Chen et al. 2010), which has lower MSE when ``n/T`` is
    non-negligible.
    """
    _, alpha = oas(x, assume_centered=True)
    t, n = x.shape
    bar_lam = float(np.linalg.norm(x, "fro") ** 2) / (n * t)
    return float(alpha), bar_lam * np.eye(n)


def rmt_target_and_alpha(x: Matrix) -> tuple[Matrix, tuple[float, Matrix, Vector], int, float]:
    """RMT-clipped *correlation* target ``C0`` with ``alpha = 1``.

    Random-matrix cleaning is applied to the sample correlation
    ``C = D^{-1/2} Sigma D^{-1/2}`` (``D = diag(Sigma)``), whose Marchenko-Pastur
    bulk edge is ``(1 + sqrt(n/T))^2`` (unit variance for standardised returns).
    Correlation eigenvalues above the edge are kept (signal); the rest are
    clipped to the trace-preserving noise floor ``mu_bar = (n - sum lambda_k)/(n - k)``,
    giving the scalar-identity-plus-low-rank cleaned correlation

        C0 = mu_bar * I + U_k @ diag(lambda_k - mu_bar) @ U_k^T.

    The minimum-variance problem is solved in the standardised coordinates
    ``y = D^{1/2} w`` against ``C0`` (the paper's change of variables): because
    ``w >= 0 <=> y >= 0`` and ``w^T Sigma0 w = y^T C0 y`` this is exact, and it
    keeps the well-conditioned scalar-identity Woodbury path. The caller maps the
    solution back with ``w = D^{-1/2} y``. Returns ``(C0, lr_factors, k, 1.0)``
    with ``lr_factors = (mu_bar, U_k, delta_k)``.
    """
    t, n = x.shape
    cov = (x.T @ x) / t
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)  # sample correlation (unit diagonal)
    mp_upper = (1.0 + np.sqrt(n / t)) ** 2  # MP edge, sigma^2 = 1 for a correlation

    eigs, vecs = np.linalg.eigh(corr)  # ascending
    signal = eigs > mp_upper
    k = int(signal.sum())
    eigs_k = eigs[signal]
    vecs_k = vecs[:, signal]

    mu_bar = float((n - eigs_k.sum()) / (n - k))  # trace-preserving bulk mean (tr C = n)
    delta_k = eigs_k - mu_bar
    target = mu_bar * np.eye(n) + vecs_k @ np.diag(delta_k) @ vecs_k.T
    lr_factors = (mu_bar, vecs_k, delta_k)
    return target, lr_factors, k, 1.0


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


def _clip_and_renormalize(w: Vector) -> Vector:
    """Clip weights to ``[0, inf)`` and renormalize to sum to 1."""
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else w


@dataclass(frozen=True)
class CGProblem:
    """Long-only minimum-variance program, solved by ``nncg`` and ``baselines``.

    Fields mirror the former ``fast_minimum_variance`` problem so the scripts
    construct it unchanged:

    Attributes:
        X: Column-demeaned ``(T, n)`` return matrix.
        target: Optional ``(n, n)`` shrinkage target ``T0``.
        alpha: Shrinkage intensity in ``[0, 1]``.
        rho: Return-tilt coefficient (``0`` for pure minimum variance).
        mu: Optional ``(n,)`` expected-return vector (used when ``rho != 0``).
        target_lr: Optional ``(bar_lambda, U_k, delta_k)`` low-rank factors of a
            scalar-identity-plus-low-rank target ``bar_lambda*I + U_k diag(delta_k) U_k^T``,
            applied through a :class:`~cvx.linalg.FactorOperator` (never formed at
            ``n x n``).
        B: Optional ``(p, n)`` balance system; ``None`` is the budget ``1^T w = 1``.
        c: Optional ``(p,)`` balance targets.
    """

    X: Matrix
    target: Matrix | None = None
    alpha: float = 0.0
    rho: float = 0.0
    mu: Vector | None = None
    target_lr: tuple[float, Matrix, Vector] | None = None
    B: Matrix | None = None
    c: Vector | None = None

    def __post_init__(self) -> None:
        """Validate the balance-system shapes."""
        if (self.B is None) != (self.c is None):
            raise ValueError("B and c must be supplied together")
        if self.B is not None and self.B.shape[1] != self.n:
            raise ValueError(f"B must have shape (p, {self.n}), got {self.B.shape}")

    @property
    def t(self) -> int:
        """Number of observations (rows of X)."""
        return int(self.X.shape[0])

    @property
    def n(self) -> int:
        """Number of assets (columns of X)."""
        return int(self.X.shape[1])

    # -- problem data ------------------------------------------------------

    def _operator(self) -> SumOperator:
        """Build ``A = 2*Sigma`` as a cvx-linalg operator (nothing formed n x n).

        ``Sigma = (1 - alpha)/T * X^T X + alpha * target``; the data term carries
        the full weight when no target is supplied. The factor of two folds the
        NNQP's ``1/2`` into the coefficients so ``min 1/2 w^T A w`` reproduces
        ``min w^T Sigma w``.
        """
        has_target = self.target_lr is not None or self.target is not None
        c_data = (1.0 - self.alpha) if has_target else 1.0
        terms: list[tuple[float, object]] = [(2.0 * c_data / self.t, GramOperator(self.X))]
        if self.target_lr is not None:
            bar_lam, u_k, delta_k = self.target_lr
            terms.append(
                (2.0 * self.alpha, FactorOperator(np.full(self.n, bar_lam), u_k, np.diag(delta_k)))
            )
        elif self.target is not None:
            terms.append((2.0 * self.alpha, DenseOperator(self.target)))
        return SumOperator(terms)

    def _dense_matrix(self) -> Matrix:
        """Materialise ``A = 2*Sigma`` as a dense array (for the direct solve)."""
        has_target = self.target_lr is not None or self.target is not None
        c_data = (1.0 - self.alpha) if has_target else 1.0
        mat = (2.0 * c_data / self.t) * (self.X.T @ self.X)
        if self.target_lr is not None:
            bar_lam, u_k, delta_k = self.target_lr
            mat = mat + 2.0 * self.alpha * (bar_lam * np.eye(self.n) + u_k @ np.diag(delta_k) @ u_k.T)
        elif self.target is not None:
            mat = mat + 2.0 * self.alpha * self.target
        return mat

    def _exact_operator(self) -> object:
        """Return a single operator supporting the direct (``"exact"``) solve.

        The active-set direct path needs ``solve_free``/``rcond_free``, which the
        composite :class:`~cvx.linalg.SumOperator` does not provide. Three cases
        collapse to one structured operator:

        * a diagonal-plus-low-rank target at ``alpha = 1`` (the data term drops
          out) -> :class:`~cvx.linalg.FactorOperator`, the Woodbury fast path;
        * no target -> ``GramOperator`` on the scaled data factor;
        * otherwise -> a dense ``DenseOperator`` (assemble-then-Cholesky).
        """
        if self.target_lr is not None and self.alpha == 1.0:
            bar_lam, u_k, delta_k = self.target_lr
            coeff = 2.0 * self.alpha
            return FactorOperator(np.full(self.n, coeff * bar_lam), u_k, np.diag(coeff * delta_k))
        if self.target is None and self.target_lr is None:
            return GramOperator(np.sqrt(2.0 / self.t) * self.X)
        return DenseOperator(self._dense_matrix())

    def _linear_term(self) -> Vector:
        """The NNQP linear term ``b = rho * mu`` (zeros for pure minimum variance)."""
        if self.rho != 0.0 and self.mu is not None:
            return self.rho * np.asarray(self.mu, dtype=float)
        return np.zeros(self.n)

    def _equality(self) -> tuple[Matrix, Vector]:
        """Return ``(B, c)`` — the balance system, or the budget ``1^T w = 1``."""
        if self.B is not None:
            return np.asarray(self.B, dtype=float), np.asarray(self.c, dtype=float)
        return np.ones((1, self.n)), np.array([1.0])

    def _project(self, w: Vector, project: bool) -> Vector:
        """Clip-and-renormalize to the budget when ``project`` and no balance system."""
        if project and self.B is None:
            return _clip_and_renormalize(w)
        return w

    # -- nncg: the matrix-free active-set method under study ----------------

    def _solve_nncg(self, inner: str, warm: Warm | None) -> nncg.Result:
        """Run ``ActiveSetSolver.solve_eq`` on the operator with the given inner solver.

        The matrix-free ``"cg"``/``"pcg"`` paths use the composite operator; the
        direct ``"exact"`` path needs a single structured operator (see
        :meth:`_exact_operator`).
        """
        a = self._exact_operator() if inner == "exact" else self._operator()
        b = self._linear_term()
        b_eq, c_eq = self._equality()
        inner_solver = {"cg": nncg.CG, "pcg": nncg.Jacobi, "exact": nncg.Exact}[inner]()
        solver = nncg.ActiveSetSolver(inner_solver)
        return solver.solve_eq(a, b, b_eq, c_eq, warm=warm)

    def solve_cg(self, *, project: bool = True) -> tuple[Vector, int, int]:
        """Matrix-free CG active-set solve; return ``(w, outer, inner)``."""
        r = self._solve_nncg("cg", None)
        return self._project(r.x, project), r.outer, r.inner

    def solve_pcg(self, *, project: bool = True) -> tuple[Vector, int]:
        """Jacobi-preconditioned CG active-set solve; return ``(w, inner)``."""
        r = self._solve_nncg("pcg", None)
        return self._project(r.x, project), r.inner

    def solve_kkt(self, *, project: bool = True) -> tuple[Vector, int]:
        """Direct (exact) inner-solve active-set; return ``(w, outer)``."""
        r = self._solve_nncg("exact", None)
        return self._project(r.x, project), r.outer

    def solve_cg_warm(
        self, *, warm_start: Warm | None = None, project: bool = True
    ) -> tuple[Vector, int, int, Warm]:
        """Warm-started CG solve; return ``(w, outer, inner, warm)``."""
        r = self._solve_nncg("cg", warm_start)
        return self._project(r.x, project), r.outer, r.inner, (r.free, r.x)

    def solve_kkt_warm(
        self, *, warm_start: Warm | None = None, project: bool = True
    ) -> tuple[Vector, int, Warm]:
        """Warm-started direct solve; return ``(w, outer, warm)``."""
        r = self._solve_nncg("exact", warm_start)
        return self._project(r.x, project), r.outer, (r.free, r.x)

    # -- baselines: external comparison solvers -----------------------------

    def _operator_and_terms(self) -> tuple[object, Vector, Matrix, Vector]:
        """Return ``(A, b, B, c)`` for the baseline solvers."""
        return self._operator(), self._linear_term(), *self._equality()

    def solve_clarabel(self, *, project: bool = True) -> tuple[Vector, int]:
        """Clarabel interior-point baseline; return ``(w, iters)``."""
        a, b, b_eq, c_eq = self._operator_and_terms()
        res = baselines.solve_clarabel(a, b, b_eq, c_eq)
        return self._project(res.x, project), res.iters

    def solve_osqp(self, *, project: bool = True) -> tuple[Vector, int]:
        """OSQP ADMM baseline; return ``(w, iters)``."""
        a, b, b_eq, c_eq = self._operator_and_terms()
        res = baselines.solve_osqp(a, b, b_eq, c_eq)
        return self._project(res.x, project), res.iters

    def _dense_target(self) -> Matrix:
        """The shrinkage target as a dense matrix (reconstructed from low-rank factors)."""
        if self.target is not None:
            return self.target
        bar_lam, u_k, delta_k = self.target_lr
        return bar_lam * np.eye(self.n) + u_k @ np.diag(delta_k) @ u_k.T

    def solve_cvxpy(self, *, backend: str = "clarabel", project: bool = True) -> tuple[Vector, int]:
        """Ground-truth reference solve through CVXPY's modeling layer; ``(w, iters)``.

        The min-variance objective is written in least-squares form and CVXPY
        dispatches it to Clarabel (default) or OSQP (``backend="osqp"``). Unlike
        the direct-API :meth:`solve_clarabel` / :meth:`solve_osqp` baselines this
        pays CVXPY's problem-construction overhead — the reference the papers
        measure that overhead against. ``iters`` is the backend's iteration count.
        """
        import cvxpy as cp

        w = cp.Variable(self.n)
        if self.target is not None or self.target_lr is not None:
            # target = M = chol @ chol.T, so ||chol.T w||^2 = w^T M w
            chol = np.linalg.cholesky(self._dense_target())
            objective = (1.0 - self.alpha) * cp.sum_squares(self.X @ w) / self.t + self.alpha * cp.sum_squares(
                chol.T @ w
            )
        else:
            objective = cp.sum_squares(self.X @ w) / self.t
        if self.rho != 0.0 and self.mu is not None:
            objective = objective - self.rho * (np.asarray(self.mu, dtype=float) @ w)

        constraints = [self.B @ w == self.c, w >= 0] if self.B is not None else [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.OSQP if backend == "osqp" else cp.CLARABEL)
        if w.value is None:  # pragma: no cover - defensive: CVXPY backend failure
            raise RuntimeError("CVXPY solver failed to find a solution")
        return self._project(np.asarray(w.value, dtype=float), project), int(problem.solver_stats.num_iters or 0)

    def solve_proximal(self, *, project: bool = True) -> tuple[Vector, int]:
        """Accelerated projected-gradient baseline on the simplex; ``(w, iters)``.

        FISTA with the exact simplex projection (:func:`baselines.solve_duchi`),
        the constrained first-order method for the ``1^T w = 1`` program. Only
        the budget problem is supported.
        """
        return self._solve_duchi(project)

    def solve_fista(self, *, project: bool = True) -> tuple[Vector, int]:
        """Accelerated projected-gradient baseline; alias of :meth:`solve_proximal`.

        The min-variance program is simplex-constrained, so its FISTA baseline
        is the simplex-projected :func:`baselines.solve_duchi`; the orthant-only
        ``baselines.solve_fista`` is not applicable (it would return ``w = 0``).
        """
        return self._solve_duchi(project)

    def _solve_duchi(self, project: bool) -> tuple[Vector, int]:
        """Shared FISTA-on-simplex baseline for the budget program."""
        if self.B is not None:
            raise NotImplementedError("first-order simplex baseline supports the budget constraint only")
        a = self._operator()
        b = self._linear_term()
        res = baselines.solve_duchi(a, b, beta=1.0)
        return self._project(res.x, project), res.iters
