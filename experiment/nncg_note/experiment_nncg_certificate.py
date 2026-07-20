"""Exact certificate vs feasible-point tolerance for "Non-Negative Conjugate Gradients".

The active-set loop terminates with an *exact* KKT certificate: it identifies the
optimal active set combinatorially and returns a dual-feasible, complementary
$(x, s)$ pair to machine precision. A feasible-point method such as MPRGP only
drives the KKT residual below a requested tolerance -- it cannot certify the
active set, and on near-degenerate problems (variables whose primal or dual
margin is tiny) it mislabels them unless run to extreme tolerance.

This script makes the gap quantitative. We plant a bound-constrained QP whose
optimum has a geometric *ladder* of margins, from O(1) down to near-zero, so a
tolerance-based method must resolve ever-smaller gaps to get the support right.
We then compare, over several seeds:

  * the active-set loop with an exact inner solve (Algorithm 1), and
  * MPRGP at a sweep of stopping tolerances,

on (i) support-identification errors against the planted optimum, and (ii) the
KKT natural residual ||min(x, s)||_inf, which is zero exactly at a KKT point.

Usage:
    uv run python -m nncg_note.experiment_nncg_certificate   # from experiment/

Outputs:
    graphs/nncg_certificate.pdf        support errors + KKT residual vs tolerance
    tables/nncg_certificate_defs.tex   headline numbers as \\newcommand macros

NumPy + Matplotlib only (MPRGP from nncg_note.baselines_bcqp).
"""


from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cvx.linalg import DenseOperator

from common.util.runner import SMOKE, output_dirs

from nncg import ActiveSetConfig, ActiveSetSolver, Exact
from nncg_note.baselines_bcqp import mprgp

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

COLOR_MP = "#8c564b"
COLOR_AS = "#ff7f0e"

N = 40 if SMOKE else 200
KAPPA = 1e3
SEEDS = 2 if SMOKE else 5
TOLS = [1e-4, 1e-8, 1e-12] if SMOKE else [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
MIN_MARGIN = 1e-8
CLASSIFY_TOL = 1e-10         # "held" if x_i > CLASSIFY_TOL; below the smallest margin
                            # so a correctly recovered small-margin variable is not
                            # counted as an error (the exact solve recovers it to ~1e-14)


def planted(seed):
    """A near-degenerate planted BCQP: geometric margin ladder from 1 to MIN_MARGIN."""
    rng = np.random.default_rng(seed)
    eig = np.geomspace(1.0, KAPPA, N)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    A = 0.5 * ((Q * eig) @ Q.T + ((Q * eig) @ Q.T).T)
    k = N // 2
    perm = rng.permutation(N)
    supp, off = perm[:k], perm[k:]
    margins = np.geomspace(MIN_MARGIN, 1.0, k)
    x_star = np.zeros(N)
    x_star[supp] = margins                       # primal margins on the support
    s_star = np.zeros(N)
    s_star[off] = np.geomspace(MIN_MARGIN, 1.0, N - k)   # dual margins off it
    b = A @ x_star - s_star
    return A, b, x_star, float(eig[-1])


def kkt_natural_residual(A, b, x):
    """||min(x, s)||_inf with s = A x - b; zero exactly at a KKT point."""
    s = A @ x - b
    return float(np.max(np.abs(np.minimum(x, s))))


def support_errors(x, x_star):
    """Count variables whose held/active classification differs from the optimum."""
    true_active = x_star == 0.0
    pred_active = x <= CLASSIFY_TOL
    return int(np.sum(pred_active != true_active))


print("=" * 78)
print(f"Exact certificate vs MPRGP tolerance  (n={N}, kappa={KAPPA:.0e}, "
      f"{SEEDS} seeds, margins down to {MIN_MARGIN:.0e})")
print("=" * 78)

as_err, as_resid, as_outer = [], [], []
mp_err = {t: [] for t in TOLS}
mp_resid = {t: [] for t in TOLS}
mp_iters = {t: [] for t in TOLS}
mp_dual = {t: [] for t in TOLS}

for seed in range(SEEDS):
    A, b, x_star, lam_max = planted(seed)
    op = DenseOperator(A)
    r = ActiveSetSolver(Exact(), ActiveSetConfig(tol=1e-10, max_outer=200)).solve(op, b)
    as_err.append(support_errors(r.x, x_star))
    as_resid.append(kkt_natural_residual(A, b, r.x))
    as_outer.append(r.outer)
    for t in TOLS:
        mp = mprgp(lambda v: A @ v, b, N, lam_max=lam_max, tol=t, maxit=2_000_000)
        mp_err[t].append(support_errors(mp.x, x_star))
        mp_resid[t].append(kkt_natural_residual(A, b, mp.x))
        mp_iters[t].append(mp.iters)
        mp_dual[t].append(float((A @ mp.x - b).min()))

as_err_m = float(np.mean(as_err))
as_resid_m = float(np.mean(as_resid))
as_outer_m = float(np.mean(as_outer))
print(f"active-set (exact): support errors {as_err_m:.1f}, KKT residual {as_resid_m:.1e}, "
      f"outer steps {as_outer_m:.1f}")
print(f"{'MPRGP tol':>10}{'supp_err':>10}{'KKT_resid':>12}{'min_s':>12}{'iters':>9}")
for t in TOLS:
    print(f"{t:>10.0e}{np.mean(mp_err[t]):>10.1f}{np.mean(mp_resid[t]):>12.1e}"
          f"{np.mean(mp_dual[t]):>12.1e}{np.mean(mp_iters[t]):>9.0f}")

# tolerance (and cost) at which MPRGP first matches the active set's exact support
tol_zero = next((t for t in TOLS if np.mean(mp_err[t]) < 0.5), None)
iters_zero = np.mean(mp_iters[tol_zero]) if tol_zero is not None else float("nan")
tol_hi = 1e-6 if 1e-6 in TOLS else TOLS[1]


# ---------------------------------------------------------------------------
# Figure: support errors (left) and KKT residual (right) vs MPRGP tolerance
# ---------------------------------------------------------------------------

tols = np.array(TOLS)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 2.9))

axA.semilogx(tols, [np.mean(mp_err[t]) for t in TOLS], marker="P", markersize=5,
             color=COLOR_MP, label="MPRGP")
axA.axhline(as_err_m, linestyle="--", linewidth=1.0, color=COLOR_AS,
            label="active set (exact)")
axA.set_xlabel("MPRGP stopping tolerance")
axA.set_ylabel(f"support errors (of $n={N}$)")
axA.set_title("Exact support identification")
axA.legend(framealpha=0.9)
axA.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

axB.loglog(tols, [np.mean(mp_resid[t]) for t in TOLS], marker="P", markersize=5,
           color=COLOR_MP, label="MPRGP")
axB.axhline(as_resid_m, linestyle="--", linewidth=1.0, color=COLOR_AS,
            label="active set (exact)")
axB.set_xlabel("MPRGP stopping tolerance")
axB.set_ylabel(r"KKT residual $\|\min(x,s)\|_\infty$")
axB.set_title("Certificate quality")
axB.legend(framealpha=0.9)
axB.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)

fig.tight_layout(pad=0.8)
fig.savefig(GRAPHS / "nncg_certificate.pdf", bbox_inches="tight")
fig.savefig(GRAPHS / "nncg_certificate.png", bbox_inches="tight", dpi=150)
print(f"\nSaved {GRAPHS / 'nncg_certificate.pdf'}")


# ---------------------------------------------------------------------------
# Headline macros
# ---------------------------------------------------------------------------

def texpow(t):
    return f"10^{{{int(round(np.log10(t)))}}}"


def sci(x):
    """Format a signed value as bare LaTeX math ``m\\times10^{e}`` (wrap in $ in prose)."""
    if x == 0.0:
        return "0"
    sign = "-" if x < 0 else ""
    a = abs(x)
    e = int(np.floor(np.log10(a)))
    return f"{sign}{a / 10.0 ** e:.1f}\\times10^{{{e}}}"


with open(TABLES / "nncg_certificate_defs.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_certificate.py -- do not edit by hand.\n")
    fh.write(f"\\newcommand{{\\nncgCertN}}{{{N}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertSeeds}}{{{SEEDS}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertMinMargin}}{{{texpow(MIN_MARGIN)}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertASouter}}{{{as_outer_m:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertASresid}}{{{sci(as_resid_m)}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertMPerrHi}}{{{np.mean(mp_err[tol_hi]):.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertMPtolHi}}{{{texpow(tol_hi)}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertMPresidHi}}{{{sci(float(np.mean(mp_resid[tol_hi])))}}}\n")
    fh.write(f"\\newcommand{{\\nncgCertMPdualHi}}{{{sci(float(np.mean(mp_dual[tol_hi])))}}}\n")
    if tol_zero is not None:
        fh.write(f"\\newcommand{{\\nncgCertMPtolZero}}{{{texpow(tol_zero)}}}\n")
        fh.write(f"\\newcommand{{\\nncgCertMPitZero}}{{{iters_zero:.0f}}}\n")
print(f"Saved {TABLES / 'nncg_certificate_defs.tex'}")
print("\nDone.")
