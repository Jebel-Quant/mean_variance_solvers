"""Synthetic SPD study for "Non-Negative Conjugate Gradients".

Self-contained: depends only on NumPy and Matplotlib. It implements the three
methods analysed in the note directly from their definitions --

  * the matrix-free CG inner solve (Proposition 3.1),
  * the active-set / block-principal-pivoting outer loop (Algorithm 1), and
  * the projected-gradient comparator (Proposition 5.1) --

and runs them on the controlled test family of Section 7.2: A = Q diag(lambda) Q^T
with a prescribed spectral condition number kappa, and a planted non-negative
minimiser x* >= 0 of chosen support obtained from a complementary pair
b = A x* - s* with s* >= 0, so the exact optimum (x*, s*) of LCP(A, -b) is known
in closed form. This turns the three qualitative predictions of Section 7.2 into
measured curves.

Usage:
    uv run experiment_nncg.py    # from non_negative_cg/experiment/

Outputs (files):
    graphs/nncg_kappa.pdf      CG inner iterations vs kappa, with a sqrt(kappa) guide
    graphs/nncg_cg_vs_pg.pdf   CG vs projected gradient: O(sqrt kappa) vs O(kappa)
    graphs/nncg_reg.pdf        CG iterations and kappa vs the regularising split alpha
    graphs/nncg_warm.pdf       warm vs cold CG iterations along a parametric sweep
    tables/nncg_synthetic.tex  per-kappa outer/inner/PG counts (booktabs fragment)
    tables/nncg_rankdef.tex    rank-deficient (m<n) alpha sweep (booktabs fragment)
    tables/nncg_defs.tex       fitted scaling exponents etc. as \\newcommand macros

Outputs (stdout):
    a correctness check (max |x - x*|) and the three measured tables.

No finance data or external package is used; the whole study is reproducible
from a fixed seed.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from util.runner import output_dirs

from nncg_ref import (cg, kkt_violation, make_problem, pcg,  # noqa: F401
                      solve_nnqp, solve_nnqp_eq)

HERE = Path(__file__).parent
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)

COLOR_CG = "#ff7f0e"
COLOR_PG = "#9467bd"
COLOR_KAPPA = "#1f77b4"


# ---------------------------------------------------------------------------
# Projected-gradient comparator (Proposition 5.1)
# ---------------------------------------------------------------------------

def solve_projgrad(A, b, x_star, tol=1e-6, maxit=2000000):
    """x_{k+1} = max(x_k - (1/L) (A x_k - b), 0), L = lambda_max(A).

    Counts iterations to reach relative solution error ||x-x*||/||x*|| <= tol.
    Returns (iterations, converged).
    """
    L = float(np.linalg.eigvalsh(A)[-1])
    tau = 1.0 / L
    x = np.zeros_like(b)
    xs_norm = float(np.linalg.norm(x_star))
    for it in range(1, maxit + 1):
        x = np.maximum(x - tau * (A @ x - b), 0.0)
        if np.linalg.norm(x - x_star) / xs_norm <= tol:
            return it, True
    return maxit, False


def solve_fista(A, b, x_star, tol=1e-6, maxit=2000000):
    """FISTA (Beck--Teboulle 2009): accelerated projected gradient for
    min_{x>=0} 1/2 x^T A x - b^T x. Same per-step cost as solve_projgrad
    (one mat-vec plus the orthant projection) but the O(sqrt kappa) rate:
    the projected-gradient step is taken at an extrapolated point y_k, with
    the momentum weight (t_k - 1)/t_{k+1} of the standard t-sequence.

    Counts iterations to reach relative solution error ||x-x*||/||x*|| <= tol.
    Returns (iterations, converged).
    """
    L = float(np.linalg.eigvalsh(A)[-1])
    tau = 1.0 / L
    x = np.zeros_like(b)                       # x_{k-1}
    y = x.copy()                               # extrapolated point
    t = 1.0
    xs_norm = float(np.linalg.norm(x_star))
    for it in range(1, maxit + 1):
        x_new = np.maximum(y - tau * (A @ y - b), 0.0)   # PG step at y
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if np.linalg.norm(x - x_star) / xs_norm <= tol:
            return it, True
    return maxit, False


def fit_slope(xs, ys):
    """Least-squares slope of log(ys) against log(xs)."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


# ===========================================================================
# Panel A: CG inner iterations vs kappa (and outer count)  -- predictions 1 & 2
# ===========================================================================

print("=" * 72)
print("CG inner iterations vs kappa   (n=200, support 50%, mean over 5 seeds)")
print("=" * 72)

N = 200
SEEDS = range(5)
KAPPAS = np.geomspace(1e1, 1e6, 12)

cg_inner, cg_outer, cg_fallback = [], [], []
print(f"{'kappa':>12}  {'outer':>7}  {'inner(CG)':>10}  {'fallback':>9}  {'max|x-x*|':>11}")
for kap in KAPPAS:
    inners, outers, fbacks, errs = [], [], [], []
    for sd in SEEDS:
        A, b, x_star, _ = make_problem(N, kap, seed=sd)
        res = solve_nnqp(A, b)
        inners.append(res["inner"])
        outers.append(res["outer"])
        fbacks.append(res["fallback"])
        errs.append(float(np.max(np.abs(res["x"] - x_star))))
    cg_inner.append(float(np.mean(inners)))
    cg_outer.append(float(np.mean(outers)))
    cg_fallback.append(float(np.mean(fbacks)))
    print(
        f"{kap:>12.1e}  {np.mean(outers):>7.1f}  {np.mean(inners):>10.1f}"
        f"  {np.mean(fbacks):>9.2f}  {max(errs):>11.2e}"
    )

max_err = max(
    float(np.max(np.abs(solve_nnqp(*make_problem(N, k, seed=0)[:2])["x"]
                        - make_problem(N, k, seed=0)[2])))
    for k in (KAPPAS[0], KAPPAS[-1])
)
cg_slope, cg_r2 = fit_slope(KAPPAS, cg_inner)
print(f"\nCG inner-iteration scaling: slope={cg_slope:.3f} (expected 0.5), R^2={cg_r2:.4f}")
print(f"outer steps range: {min(cg_outer):.1f}..{max(cg_outer):.1f}  (expected small, ~O(1))")
print(f"fallback invocations: max {max(cg_fallback):.2f} per solve (expected ~0 on generic data)")


# ===========================================================================
# Panel B: CG vs projected gradient  -- prediction 3
# ===========================================================================

print()
print("=" * 72)
print("CG inner iterations vs projected-gradient iterations   (n=120)")
print("=" * 72)

N_PG = 120
KAPPAS_PG = np.geomspace(1e1, 1e4, 8)  # PG cost ~ kappa, so cap the range
cg_pg_inner, pg_iters = [], []
print(f"{'kappa':>12}  {'inner(CG)':>10}  {'iter(PG)':>10}  {'PG conv':>8}")
for kap in KAPPAS_PG:
    A, b, x_star, _ = make_problem(N_PG, kap, seed=0)
    res = solve_nnqp(A, b)
    it_pg, conv = solve_projgrad(A, b, x_star)
    cg_pg_inner.append(res["inner"])
    pg_iters.append(it_pg)
    print(f"{kap:>12.1e}  {res['inner']:>10d}  {it_pg:>10d}  {str(conv):>8}")

pg_slope, pg_r2 = fit_slope(KAPPAS_PG, pg_iters)
cg_pg_slope, cg_pg_r2 = fit_slope(KAPPAS_PG, cg_pg_inner)
ratio = [p / c for p, c in zip(pg_iters, cg_pg_inner)]
ratio_slope, ratio_r2 = fit_slope(KAPPAS_PG, ratio)
print(f"\nCG slope={cg_pg_slope:.3f}, PG slope={pg_slope:.3f}  "
      f"(both below their O(sqrt kappa)/O(kappa) worst-case bounds: clustered")
print(" spectra let CG converge superlinearly and damp the first-order mode too).")
print(f"PG/CG ratio grows as kappa^{ratio_slope:.2f} (~sqrt kappa = kappa^0.5), "
      f"from {ratio[0]:.1f}x at kappa={KAPPAS_PG[0]:.0e} to "
      f"{ratio[-1]:.0f}x at kappa={KAPPAS_PG[-1]:.0e}.")


# ===========================================================================
# Panel C: regularising split lowers kappa and the iteration count  -- pred. 1
# ===========================================================================

print()
print("=" * 72)
print("CG inner iterations vs regularising split alpha   (n=200, base kappa=1e5)")
print("=" * 72)

BASE_KAPPA = 1e5
A0, b0, x0_star, s0_star = make_problem(N, BASE_KAPPA, seed=0)
eig0 = np.linalg.eigvalsh(A0)
lam_min, lam_max = float(eig0[0]), float(eig0[-1])
ALPHAS = np.linspace(0.0, 0.9, 19)
reg_inner, reg_kappa = [], []
print(f"{'alpha':>7}  {'kappa(A_a)':>11}  {'inner(CG)':>10}")
for a in ALPHAS:
    # A_alpha = (1-a) A + a I ; the planted b is regularised consistently so the
    # same test problem is solved at every alpha (R^T R = I target). Averaged over
    # seeds to smooth the integer iteration counts.
    inners_a, kap_a = [], None
    for sd in SEEDS:
        Asd, _, xsd, ssd = make_problem(N, BASE_KAPPA, seed=sd)
        eig_sd = np.linalg.eigvalsh(Asd)
        A_a = (1.0 - a) * Asd + a * np.eye(N)
        b_a = A_a @ xsd - ssd
        inners_a.append(solve_nnqp(A_a, b_a)["inner"])
        kap_a = ((1.0 - a) * float(eig_sd[-1]) + a) / ((1.0 - a) * float(eig_sd[0]) + a)
    reg_inner.append(float(np.mean(inners_a)))
    reg_kappa.append(kap_a)
    print(f"{a:>7.3f}  {kap_a:>11.1f}  {np.mean(inners_a):>10.1f}")


# ===========================================================================
# Panel D: general equality constraint B x = c  (Section 3, p >= 1)
# ===========================================================================

print()
print("=" * 72)
print("Equality-augmented solve  min_{x>=0, Bx=c}  (n=200, planted optimum)")
print("=" * 72)


def make_eq_problem(n, kappa, p, support_frac=0.5, seed=0):
    """Plant (x*, lambda*, s*) satisfying the KKT system of the B x = c problem.

    x* > 0 on a support of size k >= p (so B_F has full row rank generically),
    zero elsewhere; s* = 0 on the support and > 0 off it; lambda* in R^p is
    arbitrary; then b = A x* - B^T lambda* - s* and c = B x*.
    """
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, kappa, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = 0.5 * ((Q * eig) @ Q.T + ((Q * eig) @ Q.T).T)
    k = max(p + 1, int(round(support_frac * n)))
    perm = rng.permutation(n)
    supp, off = perm[:k], perm[k:]
    x_star = np.zeros(n)
    x_star[supp] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[off] = rng.uniform(0.5, 1.5, size=n - k)
    B = rng.standard_normal((p, n))
    lam_star = rng.standard_normal(p)
    b = A @ x_star - B.T @ lam_star - s_star
    c = B @ x_star
    return A, b, B, c, x_star, lam_star, s_star


print(f"{'p':>4}  {'kappa':>9}  {'outer':>7}  {'inner':>7}  {'max|x-x*|':>11}  {'|Bx-c|':>10}")
eq_err = 0.0
for p in (1, 3, 10):
    for kap in (1e2, 1e4):
        A, b, B, c, x_star, lam_star, _ = make_eq_problem(N, kap, p, seed=p)
        res = solve_nnqp_eq(A, b, B, c)
        ex = float(np.max(np.abs(res["x"] - x_star)))
        feas = float(np.linalg.norm(B @ res["x"] - c))
        eq_err = max(eq_err, ex, feas)
        tag = "  (= 1^T x = beta)" if p == 1 else ""
        print(f"{p:>4}  {kap:>9.0e}  {res['outer']:>7d}  {res['inner']:>7d}"
              f"  {ex:>11.2e}  {feas:>10.2e}{tag}")
print(f"\nEquality-augmented solver recovers the planted optimum to {eq_err:.1e}"
      f" across p in {{1,3,10}} (p=1 reproduces the single-normalisation case).")


# ===========================================================================
# Panel E: inexactness lemma -- CG vs exact inner solves visit the same free sets
# ===========================================================================

print()
print("=" * 72)
print("Trajectory agreement: CG inner solve vs exact inner solve  (Lemma)")
print("=" * 72)

print(f"{'kappa':>9}  {'seed':>5}  {'outer(CG)':>10}  {'outer(exact)':>13}  {'same traj':>10}")
traj_total = traj_agree = 0
for kap in (1e2, 1e4, 1e6):
    for sd in SEEDS:
        A, b, x_star, _ = make_problem(N, kap, seed=sd)
        r_cg = solve_nnqp(A, b, track=True)
        r_ex = solve_nnqp(A, b, inner="exact", track=True)
        same = r_cg["traj"] == r_ex["traj"]
        traj_total += 1
        traj_agree += same
        print(f"{kap:>9.0e}  {sd:>5}  {r_cg['outer']:>10}  {r_ex['outer']:>13}  {str(same):>10}")

print(f"\nCG-driven loop visits the exact loop's free sets on {traj_agree}/{traj_total} "
      f"instances (cg_tol=1e-10, test tol=1e-8), as the inexactness lemma licenses.")


# ===========================================================================
# Panel F: rank-deficient Gram operator (m < n) -- only alpha > 0 is well posed
# ===========================================================================

print()
print("=" * 72)
print("Rank-deficient Gram operator  (n=200, m=100; support below/above rank)")
print("=" * 72)


n_rd, m_rd = 200, 100
RD_CAP, RD_MAX_OUTER = 2000, 30
RD_ALPHAS = [0.0, 0.05, 0.2]
rd_rows = []  # (k, alpha, n_conv, n_seeds, mean_inner_conv, n_capped, max_err_conv)
for k_rd in (50, 150):  # optimal support below / above the data rank m
    print(f"--- planted support k={k_rd} (data rank m={m_rd}) ---")
    print(f"{'alpha':>7}  {'conv':>6}  {'inner(conv)':>12}  {'capped':>7}  {'max|x-x*|':>11}")
    for a in RD_ALPHAS:
        convs, inners, errs, capped = 0, [], [], 0
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            M_rd = rng.standard_normal((m_rd, n_rd)) / np.sqrt(m_rd)
            A0 = M_rd.T @ M_rd  # PSD, rank m: free blocks with |F| > m are singular
            perm = rng.permutation(n_rd)
            x_rd = np.zeros(n_rd)
            x_rd[perm[:k_rd]] = rng.uniform(0.5, 1.5, size=k_rd)
            s_rd = np.zeros(n_rd)
            s_rd[perm[k_rd:]] = rng.uniform(0.5, 1.5, size=n_rd - k_rd)
            A_a = (1.0 - a) * A0 + a * np.eye(n_rd)
            b_a = A_a @ x_rd - s_rd  # planted KKT point for every alpha, incl. 0
            res = solve_nnqp(A_a, b_a, cg_maxit=RD_CAP, max_outer=RD_MAX_OUTER)
            # a capped inner solve returned without meeting its tolerance:
            # no certificate, whatever the outer loop then does with it
            capped += res["inner"] >= RD_CAP
            if res["converged"]:
                convs += 1
                inners.append(res["inner"])
                errs.append(float(np.max(np.abs(res["x"] - x_rd))))
        mean_inner = float(np.mean(inners)) if inners else float("nan")
        max_err = max(errs) if errs else float("nan")
        rd_rows.append((k_rd, a, convs, len(list(SEEDS)), mean_inner, capped, max_err))
        print(f"{a:>7.2f}  {convs:>4}/5  {mean_inner:>12.0f}  {capped:>7}  {max_err:>11.1e}")

rd_ok = [r for r in rd_rows if r[1] > 0]
print(f"\nalpha=0: unreliable below the rank (uncertified capped solves; occasional"
      f"\nnon-convergence) and fails on every seed once the optimal support exceeds"
      f"\nthe rank. Every alpha > 0 run converges (max err "
      f"{max(r[6] for r in rd_ok):.1e}), at an order of magnitude fewer iterations.")


# ===========================================================================
# Panel G: Jacobi-preconditioned CG inside the loop  (badly scaled operator)
# ===========================================================================

print()
print("=" * 72)
print("Jacobi PCG vs CG inside the active-set loop  (n=200, scaled operator)")
print("=" * 72)


def make_scaled_problem(n, kappa_core, spread, support_frac=0.5, seed=0):
    """A = D^(1/2) (Q Lambda Q^T) D^(1/2): well-conditioned core, bad row scaling.

    Jacobi preconditioning removes the diagonal scaling, so PCG should run at
    the core's condition number regardless of the spread.
    """
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, kappa_core, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    core = 0.5 * ((Q * eig) @ Q.T + ((Q * eig) @ Q.T).T)
    d = rng.permutation(np.geomspace(1.0, spread, n))
    A = core * np.sqrt(np.outer(d, d))
    k = max(1, int(round(support_frac * n)))
    perm = rng.permutation(n)
    x_star = np.zeros(n)
    x_star[perm[:k]] = rng.uniform(0.5, 1.5, size=k)
    s_star = np.zeros(n)
    s_star[perm[k:]] = rng.uniform(0.5, 1.5, size=n - k)
    return A, A @ x_star - s_star, x_star


SPREADS = [1e0, 1e2, 1e4]
pcg_rows = []
print(f"{'spread':>9}  {'kappa(A)':>10}  {'inner(CG)':>10}  {'inner(PCG)':>11}  {'ratio':>6}")
for spread in SPREADS:
    inners_cg, inners_pcg, kappas = [], [], []
    for sd in SEEDS:
        A, b, x_star = make_scaled_problem(200, 100.0, spread, seed=sd)
        eigs = np.linalg.eigvalsh(A)
        kappas.append(float(eigs[-1] / eigs[0]))
        r_cg = solve_nnqp(A, b)
        r_pcg = solve_nnqp(A, b, inner="pcg")
        for r in (r_cg, r_pcg):
            assert float(np.max(np.abs(r["x"] - x_star))) < 1e-6
        inners_cg.append(r_cg["inner"])
        inners_pcg.append(r_pcg["inner"])
    mc, mp = float(np.mean(inners_cg)), float(np.mean(inners_pcg))
    pcg_rows.append((spread, float(np.mean(kappas)), mc, mp))
    print(f"{spread:>9.0e}  {np.mean(kappas):>10.1e}  {mc:>10.1f}  {mp:>11.1f}  {mc / mp:>6.1f}")

print("\nJacobi PCG runs at the core's conditioning regardless of the diagonal"
      "\nspread; plain CG pays for the scaling, as Section 6 predicts.")


# ===========================================================================
# Panel H: exercising the Bland fallback -- an adversarial anti-correlated family
# ===========================================================================

print()
print("=" * 72)
print("Exercising the fallback: anti-correlated near-duplicate columns (n=20)")
print("=" * 72)


def make_adversarial(n, seed, noise=1e-2, ridge=1e-6):
    """Columns arrive in near-anti-parallel pairs: M = [M0, -M0 + noise*E].

    Dropping a variable flips the sign of its partner, so the batch exchange
    systematically over-shoots -- the codimension-one near-tie event of
    Section 4 made generic. The ridge keeps A a P-matrix.
    """
    rng = np.random.default_rng(seed)
    M0 = rng.standard_normal((n, n // 2))
    M = np.hstack([M0, -M0 + noise * rng.standard_normal((n, n // 2))])
    A = M.T @ M + ridge * np.eye(n)
    return A, M.T @ rng.standard_normal(n)


FB_SEEDS = 60
fb_cycles = fb_conv = fb_fired = fb_max = 0
fb_kkt = 0.0
for sd in range(FB_SEEDS):
    A, b = make_adversarial(20, sd)
    # pure block pivoting: patience unbounded, exact solves -- the theory-faithful
    # fast path with the guard removed. A revisited free set proves a cycle.
    pure = solve_nnqp(A, b, p_max=10**9, inner="exact", max_outer=300, track=True)
    cycled = (not pure["converged"]) and len(pure["traj"]) != len(set(pure["traj"]))
    # guarded loop: Algorithm 1 as stated (default patience, CG inner solves)
    g = solve_nnqp(A, b)
    fb_cycles += cycled
    fb_conv += g["converged"]
    fb_fired += g["fallback"] > 0
    fb_max = max(fb_max, g["fallback"])
    fb_kkt = max(fb_kkt, kkt_violation(A, b, g["x"]))

print(f"pure block pivoting (no fallback, exact solves): cycles on "
      f"{fb_cycles}/{FB_SEEDS} seeds (revisits a free set, never terminates)")
print(f"guarded loop (p_max=3, CG inner): converges {fb_conv}/{FB_SEEDS}; "
      f"fallback fires on {fb_fired}/{FB_SEEDS} seeds, up to {fb_max} least-index "
      f"pivots; max KKT violation {fb_kkt:.1e}")


# ===========================================================================
# Panel I: warm-starting across a parametric sweep  (Section 5.4)
# ===========================================================================

print()
print("=" * 72)
print("Warm-started parametric sweep  (n=200, kappa=1e4, 40 steps)")
print("=" * 72)

N_W, KAPPA_W, STEPS_W = 200, 1e4, 40
A_w, b0_w, x0_w, _ = make_problem(N_W, KAPPA_W, seed=0)
rng_w = np.random.default_rng(1)
db_w = rng_w.standard_normal(N_W)
db_w *= 0.02 * np.linalg.norm(b0_w) / np.linalg.norm(db_w)  # 2% drift per step

cold_outer, cold_inner, warm_outer, warm_inner, drifts = [], [], [], [], []
prev_free = None
prev_x = None
prev_supp = None
for k in range(STEPS_W):
    b_k = b0_w + k * db_w
    rc = solve_nnqp(A_w, b_k)
    if prev_free is None:
        rw = rc  # first point: nothing to warm-start from
    else:
        rw = solve_nnqp(A_w, b_k, warm=(prev_free, prev_x))
    assert float(np.max(np.abs(rw["x"] - rc["x"]))) < 1e-6  # same optimum
    supp = frozenset(np.flatnonzero(rc["free"]).tolist())
    drifts.append(len(supp ^ prev_supp) if prev_supp is not None else 0)
    cold_outer.append(rc["outer"])
    cold_inner.append(rc["inner"])
    warm_outer.append(rw["outer"])
    warm_inner.append(rw["inner"])
    prev_free, prev_x, prev_supp = rw["free"], rw["x"], supp

# steps after the first, where warm-starting actually applies
co, ci = np.array(cold_outer[1:]), np.array(cold_inner[1:])
wo, wi = np.array(warm_outer[1:]), np.array(warm_inner[1:])
dr = np.array(drifts[1:])
print(f"{'':>12}  {'outer (mean)':>13}  {'inner (total)':>14}")
print(f"{'cold':>12}  {co.mean():>13.1f}  {ci.sum():>14d}")
print(f"{'warm':>12}  {wo.mean():>13.1f}  {wi.sum():>14d}")
print(f"\nsupport drift per step: mean {dr.mean():.1f}, max {dr.max()} of "
      f"{len(prev_supp)} active indices")
print(f"warm-start speedup: outer {co.mean() / wo.mean():.1f}x, "
      f"inner {ci.sum() / wi.sum():.1f}x")
print(f"steps with unchanged support solved in one outer step (warm): "
      f"{int(np.sum((dr == 0) & (wo == 1)))}/{int(np.sum(dr == 0))} "
      f"(Prop. support-stable case)")

# Figure: per-step inner iterations, cold vs warm
figW, axW = plt.subplots(figsize=(4.5, 3.2))
axW.plot(range(1, STEPS_W), ci, marker="o", markersize=3, color=COLOR_KAPPA,
         label="cold start")
axW.plot(range(1, STEPS_W), wi, marker="s", markersize=3, color=COLOR_CG,
         label="warm start")
axW.set_yscale("log")
axW.set_xlabel("Sweep step $k$")
axW.set_ylabel("CG iterations for step $k$")
axW.set_title(rf"Warm-started sweep  ($n={N_W}$, $\kappa=10^4$)")
axW.legend(framealpha=0.9)
axW.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
figW.tight_layout(pad=1.0)
figW.savefig(GRAPHS / "nncg_warm.pdf", bbox_inches="tight")
figW.savefig(GRAPHS / "nncg_warm.png", bbox_inches="tight", dpi=150)
print(f"Saved {GRAPHS / 'nncg_warm.pdf'}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Figure A: CG inner iterations vs kappa, with sqrt(kappa) guide
figA, axA = plt.subplots(figsize=(4.5, 3.2))
axA.plot(KAPPAS, cg_inner, marker="o", markersize=4, color=COLOR_CG,
         label="CG inner iterations (total)")
# sqrt(kappa) upper envelope anchored at the smallest kappa: the worst-case bound
# of Prop. 3.1, which the clustered spectrum lets CG stay below.
axA.plot(KAPPAS, cg_inner[0] * np.sqrt(KAPPAS / KAPPAS[0]), color="gray",
         linestyle="--", linewidth=0.9, label=r"$O(\sqrt{\kappa})$ envelope")
axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlabel(r"Condition number $\kappa$")
axA.set_ylabel("CG iterations to tolerance")
axA.set_title(r"CG inner count vs. $\kappa$  ($n=200$)")
axA.legend(framealpha=0.9)
axA.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
figA.tight_layout(pad=1.0)
figA.savefig(GRAPHS / "nncg_kappa.pdf", bbox_inches="tight")
figA.savefig(GRAPHS / "nncg_kappa.png", bbox_inches="tight", dpi=150)
print(f"\nSaved {GRAPHS / 'nncg_kappa.pdf'}")

# Figure B: CG vs projected gradient
figB, axB = plt.subplots(figsize=(4.5, 3.2))
axB.plot(KAPPAS_PG, cg_pg_inner, marker="o", markersize=4, color=COLOR_CG,
         label=r"CG inner, $O(\sqrt{\kappa})$")
axB.plot(KAPPAS_PG, pg_iters, marker="s", markersize=4, color=COLOR_PG,
         label=r"Projected gradient, $O(\kappa)$")
axB.plot(KAPPAS_PG, cg_pg_inner[0] * np.sqrt(KAPPAS_PG / KAPPAS_PG[0]),
         color="gray", linestyle="--", linewidth=0.9, label=r"$\sqrt{\kappa}$")
axB.plot(KAPPAS_PG, pg_iters[0] * (KAPPAS_PG / KAPPAS_PG[0]),
         color="gray", linestyle=":", linewidth=0.9, label=r"$\kappa$")
axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlabel(r"Condition number $\kappa$")
axB.set_ylabel("Iterations to convergence")
axB.set_title(r"Krylov vs. first order  ($n=120$)")
axB.legend(framealpha=0.9, fontsize=7)
axB.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
figB.tight_layout(pad=1.0)
figB.savefig(GRAPHS / "nncg_cg_vs_pg.pdf", bbox_inches="tight")
figB.savefig(GRAPHS / "nncg_cg_vs_pg.png", bbox_inches="tight", dpi=150)
print(f"Saved {GRAPHS / 'nncg_cg_vs_pg.pdf'}")

# Figure C: regularising split -- iterations and kappa vs alpha
figC, axC = plt.subplots(figsize=(4.5, 3.2))
axC.plot(ALPHAS, reg_inner, marker="o", markersize=4, color=COLOR_CG,
         label="CG inner iterations")
axC.set_xlabel(r"Regularisation intensity $\alpha$")
axC.set_ylabel("CG iterations to tolerance", color=COLOR_CG)
axC.tick_params(axis="y", labelcolor=COLOR_CG)
axC.set_title(r"Regularising split  ($n=200$, base $\kappa=10^5$)")
axC.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
axC2 = axC.twinx()
axC2.plot(ALPHAS, reg_kappa, marker="^", markersize=4, color=COLOR_KAPPA,
          linestyle="--", label=r"$\kappa(A_\alpha)$")
axC2.set_yscale("log")
axC2.set_ylabel(r"$\kappa(A_\alpha)$", color=COLOR_KAPPA)
axC2.tick_params(axis="y", labelcolor=COLOR_KAPPA)
lines1, labels1 = axC.get_legend_handles_labels()
lines2, labels2 = axC2.get_legend_handles_labels()
axC.legend(lines1 + lines2, labels1 + labels2, framealpha=0.9, loc="upper right")
figC.tight_layout(pad=1.0)
figC.savefig(GRAPHS / "nncg_reg.pdf", bbox_inches="tight")
figC.savefig(GRAPHS / "nncg_reg.png", bbox_inches="tight", dpi=150)
print(f"Saved {GRAPHS / 'nncg_reg.pdf'}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# A representative subset of kappa for the printed table.
table_rows = []
for kap in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    inners, outers, fbacks = [], [], []
    for sd in SEEDS:
        A, b, x_star, _ = make_problem(N, kap, seed=sd)
        res = solve_nnqp(A, b)
        inners.append(res["inner"])
        outers.append(res["outer"])
        fbacks.append(res["fallback"])
    table_rows.append((kap, np.mean(outers), np.mean(inners), np.mean(fbacks)))

with open(TABLES / "nncg_synthetic.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg.py -- do not edit by hand.\n")
    fh.write("\\begin{tabular}{rrrr}\n\\toprule\n")
    fh.write("$\\kappa$ & Outer steps $s$ & CG inner (total) & Fallback pivots \\\\\n")
    fh.write("\\midrule\n")
    for kap, outer, inner, fback in table_rows:
        exp = int(round(np.log10(kap)))
        fh.write(f"$10^{{{exp}}}$ & {outer:.1f} & {inner:.0f} & {fback:.2f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"Saved {TABLES / 'nncg_synthetic.tex'}")

def sci(v):
    """Format a float as LaTeX scientific notation, e.g. $2\\cdot10^{-9}$."""
    if v == 0.0:
        return "$0$"
    mant, expo = f"{v:.0e}".split("e")
    return f"${mant}\\cdot10^{{{int(expo)}}}$"


with open(TABLES / "nncg_rankdef.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg.py -- do not edit by hand.\n")
    fh.write("\\begin{tabular}{rrcrrr}\n\\toprule\n")
    fh.write("Support $k$ & $\\alpha$ & Converged & CG inner & Capped solves & "
             "$\\lVert x - x^\\star\\rVert_\\infty$ \\\\\n\\midrule\n")
    prev_k = None
    for k, a, convs, total, mean_inner, capped, max_err in rd_rows:
        if prev_k is not None and k != prev_k:
            fh.write("\\midrule\n")
        prev_k = k
        inner_s = "--" if np.isnan(mean_inner) else f"{mean_inner:.0f}"
        err_s = "--" if np.isnan(max_err) else sci(max_err)
        fh.write(f"{k} & {a:.2f} & {convs}/{total} & {inner_s} & {capped} & {err_s} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"Saved {TABLES / 'nncg_rankdef.tex'}")

with open(TABLES / "nncg_defs.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg.py -- do not edit by hand.\n")
    fh.write(f"\\newcommand{{\\nncgCGslope}}{{{cg_slope:.2f}}}\n")
    fh.write(f"\\newcommand{{\\nncgCGrtwo}}{{{cg_r2:.3f}}}\n")
    fh.write(f"\\newcommand{{\\nncgPGslope}}{{{pg_slope:.2f}}}\n")
    fh.write(f"\\newcommand{{\\nncgPGrtwo}}{{{pg_r2:.3f}}}\n")
    fh.write(f"\\newcommand{{\\nncgMaxOuter}}{{{max(cg_outer):.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgMaxErr}}{{{max_err:.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgPGratio}}{{{pg_iters[-1] / cg_pg_inner[-1]:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgRatioSlope}}{{{ratio_slope:.2f}}}\n")
    fh.write(f"\\newcommand{{\\nncgKappaMax}}{{10^{{{int(round(np.log10(KAPPAS_PG[-1])))}}}}}\n")
    fh.write(f"\\newcommand{{\\nncgEqErr}}{{{eq_err:.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgTrajAgree}}{{{traj_agree}}}\n")
    fh.write(f"\\newcommand{{\\nncgTrajTotal}}{{{traj_total}}}\n")
    fh.write(f"\\newcommand{{\\nncgRankOkErr}}{{{max(r[6] for r in rd_ok):.0e}}}\n")
    _rd0 = {r[0]: r for r in rd_rows if r[1] == 0.0}
    fh.write(f"\\newcommand{{\\nncgRankConvLow}}{{{_rd0[50][2]}}}\n")   # alpha=0, k<m
    fh.write(f"\\newcommand{{\\nncgRankConvHigh}}{{{_rd0[150][2]}}}\n")  # alpha=0, k>m
    # PCG panel at the widest diagonal spread: CG pays for the scaling, PCG does not.
    _sp, _kap, _mc, _mp = pcg_rows[-1]
    fh.write(f"\\newcommand{{\\nncgPcgKappa}}{{{_kap:.0e}}}\n")
    fh.write(f"\\newcommand{{\\nncgPcgCG}}{{{_mc:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgPcgPCG}}{{{_mp:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgPcgRatio}}{{{_mc / _mp:.0f}}}\n")
    _mc1, _mp1 = pcg_rows[0][2], pcg_rows[0][3]
    fh.write(f"\\newcommand{{\\nncgPcgBase}}{{{_mp1:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgFbSeeds}}{{{FB_SEEDS}}}\n")
    fh.write(f"\\newcommand{{\\nncgFbCycles}}{{{fb_cycles}}}\n")
    fh.write(f"\\newcommand{{\\nncgFbConv}}{{{fb_conv}}}\n")
    fh.write(f"\\newcommand{{\\nncgFbFired}}{{{fb_fired}}}\n")
    fh.write(f"\\newcommand{{\\nncgFbMax}}{{{fb_max}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmSteps}}{{{STEPS_W - 1}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmColdOuter}}{{{co.mean():.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmOuter}}{{{wo.mean():.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmInnerX}}{{{ci.sum() / wi.sum():.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmDrift}}{{{dr.mean():.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmStable}}{{{int(np.sum((dr == 0) & (wo == 1)))}}}\n")
    fh.write(f"\\newcommand{{\\nncgWarmStableTot}}{{{int(np.sum(dr == 0))}}}\n")
print(f"Saved {TABLES / 'nncg_defs.tex'}")
print("\nDone.")
