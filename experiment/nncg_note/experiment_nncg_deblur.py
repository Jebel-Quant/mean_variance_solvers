"""Matrix-free 2-D image deblurring for "Non-Negative Conjugate Gradients".

Demonstrates the matrix-free operator abstraction (Section 3) at a scale where
the Gram matrix cannot be formed, and -- the point of this script -- times the
matrix-free active-set loop head-to-head against the dense baselines of the
external benchmark (Lawson-Hanson, Clarabel) as the problem grows.

A separable Gaussian blur B = K (x) K acts on an N-by-N image (n = N^2
unknowns); applied as X -> K X K^T it costs O(N^3) per product and never
materialises the n-by-n operator A = B^T B, whose dense storage would be
O(n^2). Images are non-negative, so the reconstruction is a non-negative
least-squares problem, solved by the matrix-free active-set loop
(ActiveSetSolver) with a ridge split for the P-matrix property.

The dense baselines need the explicit Gram operator, which here is available in
closed form without ever forming B: with G = K^T K,

    A = B^T B = (K (x) K)^T (K (x) K) = (K^T K) (x) (K^T K) = G (x) G,

so A_alpha = (1-alpha) (G (x) G) + alpha I. That n-by-n matrix is what
Lawson-Hanson must Cholesky-factorise and Clarabel must store and factorise
each interior-point step -- O(n^2) memory and O(n^3) work. Both are run only
while the dense Gram stays under a small storage budget; past it they are
omitted, their O(n^3) cost already dwarfing the matrix-free solve (a factor of
tens at n = 4096) and the dense A itself reaching tens of GB at the largest
matrix-free sizes.

Usage:
    uv run python -m nncg_note.experiment_nncg_deblur   # from the experiment/ directory

Outputs:
    graphs/nncg_deblur.pdf         true / blurred+noise / reconstructed panels
    graphs/nncg_deblur_bench.pdf   wall-clock vs n: matrix-free NNCG vs dense baselines
    tables/nncg_deblur.tex         per-size wall-clock and dense-A memory (booktabs)
    tables/nncg_deblur_defs.tex    headline numbers as \\newcommand macros

The matrix-free solve is NumPy only; the dense baselines add SciPy and
Clarabel, and matplotlib is used solely to render the figures.
"""


from __future__ import annotations

import time
from pathlib import Path

import clarabel
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import osqp
import scipy.sparse as sp
from cvx.linalg import SymmetricOperator
from scipy.linalg import solve_triangular
from scipy.optimize import nnls as scipy_nnls

from common.util.runner import SMOKE, output_dirs

from nncg import CG, ActiveSetConfig, ActiveSetSolver, KrylovConfig
from nncg_note.baselines_bcqp import mprgp

HERE = Path(__file__).resolve().parents[1]  # experiment/ root
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

SIGMA = 2.0
ALPHA = 1e-3                                          # ridge split weight
NOISE_REL = 1e-3


def gaussian_blur(nside, sigma):
    """Row-normalised 1-D Gaussian Toeplitz blur; the 2-D blur is K (x) K."""
    i = np.arange(nside)
    d = np.abs(np.subtract.outer(i, i))
    K = np.exp(-(d ** 2) / (2.0 * sigma ** 2))
    return K / K.sum(axis=1, keepdims=True)


def test_image(nside):
    """The Shepp--Logan phantom: a standard, recognisable non-negative test image.

    A sum of ellipses (the modified-contrast coefficients), non-negative after
    clipping, hence a genuine non-negative deconvolution target with a large
    black background that the non-negativity constraints recover.
    """
    # (intensity, semi-axis a, semi-axis b, centre x0, centre y0, angle deg)
    ellipses = [
        (1.00, .69, .92, 0.0, 0.0, 0.0),
        (-.80, .6624, .8740, 0.0, -.0184, 0.0),
        (-.20, .1100, .3100, .22, 0.0, -18.0),
        (-.20, .1600, .4100, -.22, 0.0, 18.0),
        (.10, .2100, .2500, 0.0, .35, 0.0),
        (.10, .0460, .0460, 0.0, .10, 0.0),
        (.10, .0460, .0460, 0.0, -.10, 0.0),
        (.10, .0460, .0230, -.08, -.605, 0.0),
        (.10, .0230, .0230, 0.0, -.606, 0.0),
        (.10, .0230, .0460, .06, -.605, 0.0),
    ]
    axis = np.linspace(-1.0, 1.0, nside)
    xx, yy = np.meshgrid(axis, -axis)
    X = np.zeros((nside, nside))
    for intensity, a, b, x0, y0, phi in ellipses:
        t = np.deg2rad(phi)
        xr = (xx - x0) * np.cos(t) + (yy - y0) * np.sin(t)
        yr = -(xx - x0) * np.sin(t) + (yy - y0) * np.cos(t)
        X[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += intensity
    return np.clip(X, 0.0, None)


class BlurOperator(SymmetricOperator):
    """The SPD deblur operator A_alpha as a matrix-free cvx.linalg operator.

    Wraps the ``matvec`` closure so :class:`nncg.ActiveSetSolver` can drive it
    without ever assembling the ``n x n`` matrix. ``block_matvec`` embeds the
    reduced vector into full space, applies the full operator, and reads back
    the active rows -- the reduced action the active-set CG needs. The
    direct-solve hooks are unused (inner="cg" only).
    """

    def __init__(self, apply, dim):
        self._apply = apply
        self._dim = dim

    @property
    def n(self):
        return self._dim

    def matvec(self, v):
        return self._apply(np.asarray(v, dtype=float))

    def block_matvec(self, rows, cols, v):
        full = np.zeros(self._dim)
        full[cols] = v
        return self.matvec(full)[rows]

    def restricted(self, idx):
        """Free-block view of this operator: matvec via block_matvec(idx, idx, ·).

        There is no cheaper "pre-sliced" representation for a matrix-free blur
        (unlike a Gram operator's factor columns): every application still
        re-embeds into full space and reads back the free rows. This just
        satisfies nncg's operator protocol, which builds a restricted view
        once per free set rather than calling block_matvec per CG iteration.
        """
        idx = np.asarray(idx)
        return BlurOperator(lambda v: self.block_matvec(idx, idx, v), idx.size)

    def rcond_free(self, idx):
        raise NotImplementedError("BlurOperator supports matrix-free CG only")

    def solve_free(self, idx, rhs):
        raise NotImplementedError("BlurOperator supports matrix-free CG only")


def make_instance(nside):
    """Build (K, x_true, b) for the N-by-N deblurring instance."""
    K = gaussian_blur(nside, SIGMA)
    X_true = test_image(nside)
    D = K @ X_true @ K.T                              # blurred image, B x
    rng = np.random.default_rng(0)
    D_noisy = D + NOISE_REL * np.linalg.norm(D) * rng.standard_normal((nside, nside)) / nside
    b = (K.T @ D_noisy @ K).ravel()                  # B^T d
    return K, X_true.ravel(), D_noisy, b


def matvec_for(K, nside):
    """A_alpha x = (1-alpha) B^T B x + alpha x, matrix-free via K X K^T."""
    def matvec(xv):
        Xm = xv.reshape(nside, nside)
        Y = K @ Xm @ K.T                             # B x
        Z = K.T @ Y @ K                              # B^T (B x)
        return (1.0 - ALPHA) * Z.ravel() + ALPHA * xv
    return matvec


def dense_gram(K):
    """A_alpha = (1-alpha) (G (x) G) + alpha I with G = K^T K, without forming B."""
    G = K.T @ K
    n = K.shape[0] ** 2
    return (1.0 - ALPHA) * np.kron(G, G) + ALPHA * np.eye(n)


def solve_matrix_free(K, nside, b):
    """Time the paper's matrix-free active-set + CG loop on the deblur operator."""
    n = nside * nside
    t0 = time.perf_counter()
    res = ActiveSetSolver(
        CG(KrylovConfig(tol=1e-8, maxit=500)), ActiveSetConfig(max_outer=40)
    ).solve(BlurOperator(matvec_for(K, nside), n), b)
    return res, time.perf_counter() - t0


def solve_clarabel(A, b):
    """Clarabel interior point on min 1/2 x^T A x - b^T x s.t. x >= 0."""
    n = len(b)
    P = sp.triu(sp.csc_matrix(A), format="csc")
    Ac = -sp.identity(n, format="csc")
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    t0 = time.perf_counter()
    sol = clarabel.DefaultSolver(P, -b, Ac, np.zeros(n),
                                 [clarabel.NonnegativeConeT(n)], settings).solve()
    return np.asarray(sol.x), time.perf_counter() - t0


def solve_osqp(A, b):
    """OSQP (ADMM) on min 1/2 x^T A x - b^T x s.t. x >= 0."""
    n = len(b)
    solver = osqp.OSQP()
    solver.setup(P=sp.triu(sp.csc_matrix(A), format="csc"), q=-b,
                 A=sp.identity(n, format="csc"), l=np.zeros(n),
                 u=np.full(n, np.inf), verbose=False)
    t0 = time.perf_counter()
    sol = solver.solve()
    return np.asarray(sol.x), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Size sweep: matrix-free NNCG vs the dense baselines
# ---------------------------------------------------------------------------

# Full run: NNCG across a range that reaches a dense-infeasible regime (n up to
# 65536, dense A = 34 GB). Dense baselines are run only while the Gram matrix
# stays under DENSE_MEM_CAP_GB. Smoke run: tiny sizes straddling the cap so both
# the dense and dense-skipped code paths are exercised in seconds.
NSIDES = [16, 32] if SMOKE else [32, 48, 64, 96, 128, 192, 256]
DENSE_MEM_CAP_GB = 1e-3 if SMOKE else 0.25
RECON_NSIDE = 16 if SMOKE else 128                   # size shown in the panel figure


def dense_gb(n):
    return n * n * 8 / 1e9


print("=" * 78)
print("Matrix-free deblurring: wall-clock vs n  (NNCG vs dense Lawson-Hanson / Clarabel)")
print(f"sigma = {SIGMA}, ridge alpha = {ALPHA:.0e}, noise = {NOISE_REL:.0e}, "
      f"dense cap = {DENSE_MEM_CAP_GB:g} GB")
print("=" * 78)
print(f"{'N':>5}{'n':>8}{'denseA(GB)':>12}{'NNCG(s)':>10}{'MPRGP(s)':>10}"
      f"{'LH(s)':>10}{'Clarabel(s)':>12}{'OSQP(s)':>10}{'outer':>7}{'inner':>7}")

rows = []           # per-size records for the table/figure
recon = {}          # reconstruction (x, x_true) at RECON_NSIDE for the panel figure
for nside in NSIDES:
    n = nside * nside
    K, x_true, D_noisy, b = make_instance(nside)

    res, t_mf = solve_matrix_free(K, nside, b)
    if nside == RECON_NSIDE:
        recon = {"x": res.x, "D_noisy": D_noisy,
                 "X_true": x_true.reshape(nside, nside)}

    # MPRGP is also matrix-free, so it runs at every size alongside NNCG.
    t0 = time.perf_counter()
    mprgp(matvec_for(K, nside), b, n, tol=1e-8, maxit=200000)
    t_mprgp = time.perf_counter() - t0

    t_lh = t_cl = t_osqp = np.nan
    if dense_gb(n) <= DENSE_MEM_CAP_GB:
        A = dense_gram(K)
        # Lawson-Hanson: the Cholesky factorisation is part of the solve cost.
        t0 = time.perf_counter()
        L = np.linalg.cholesky(A)
        d = solve_triangular(L, b, lower=True)
        scipy_nnls(L.T, d)
        t_lh = time.perf_counter() - t0
        _, t_cl = solve_clarabel(A, b)
        _, t_osqp = solve_osqp(A, b)
        del A

    rows.append({"nside": nside, "n": n, "dense_gb": dense_gb(n),
                 "t_mf": t_mf, "t_mprgp": t_mprgp, "t_lh": t_lh, "t_cl": t_cl,
                 "t_osqp": t_osqp, "outer": res.outer, "inner": res.inner})
    print(f"{nside:>5}{n:>8}{dense_gb(n):>12.2f}{t_mf:>10.3f}{t_mprgp:>10.3f}"
          f"{t_lh:>10.3f}{t_cl:>12.3f}{t_osqp:>10.3f}{res.outer:>7}{res.inner:>7}")


# ---------------------------------------------------------------------------
# Reconstruction figure (true / blurred+noise / reconstructed) at RECON_NSIDE
# ---------------------------------------------------------------------------

nside = RECON_NSIDE
X_hat = recon["x"].reshape(nside, nside)
X_true = recon["X_true"]
vmax = float(X_true.max())


def psnr(est, ref, peak):
    """Peak signal-to-noise ratio (dB) of ``est`` against ``ref``."""
    mse = float(np.mean((est - ref) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(peak ** 2 / mse)


psnr_blur = psnr(recon["D_noisy"], X_true, vmax)
psnr_recon = psnr(X_hat, X_true, vmax)
panels = [("True $x^\\star$", X_true),
          (f"Blurred $+$ noise ({psnr_blur:.1f} dB)", recon["D_noisy"]),
          (f"Reconstructed NNCG ({psnr_recon:.1f} dB)", X_hat)]
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))
for ax, (title, img) in zip(axes, panels):
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout(pad=0.6)
fig.savefig(GRAPHS / "nncg_deblur.pdf", bbox_inches="tight")
fig.savefig(GRAPHS / "nncg_deblur.png", bbox_inches="tight", dpi=150)
print(f"\nSaved {GRAPHS / 'nncg_deblur.pdf'}")


# ---------------------------------------------------------------------------
# Benchmark figure: wall-clock vs n (dense curves stop at the feasibility wall)
# ---------------------------------------------------------------------------

ns = [r["n"] for r in rows]
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.plot(ns, [r["t_mf"] for r in rows], marker="D", markersize=4, color="#ff7f0e",
        label="Active set $+$ CG (matrix-free)")
ax.plot(ns, [r["t_mprgp"] for r in rows], marker="P", markersize=4, color="#8c564b",
        label="MPRGP (matrix-free)")
dense_ns = [r["n"] for r in rows if np.isfinite(r["t_lh"])]
if dense_ns:
    ax.plot(dense_ns, [r["t_lh"] for r in rows if np.isfinite(r["t_lh"])],
            marker="o", markersize=4, color="#1f77b4", label="Lawson--Hanson (dense)")
    ax.plot(dense_ns, [r["t_cl"] for r in rows if np.isfinite(r["t_cl"])],
            marker="^", markersize=4, color="#2ca02c", label="Clarabel (dense)")
    ax.plot(dense_ns, [r["t_osqp"] for r in rows if np.isfinite(r["t_osqp"])],
            marker="v", markersize=4, color="#9467bd", label="OSQP (dense)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Problem size $n = N^2$")
ax.set_ylabel("Wall-clock time (s)")
ax.set_title("Matrix-free deblurring vs dense baselines")
ax.legend(framealpha=0.9, fontsize=7)
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
fig.tight_layout(pad=1.0)
fig.savefig(GRAPHS / "nncg_deblur_bench.pdf", bbox_inches="tight")
fig.savefig(GRAPHS / "nncg_deblur_bench.png", bbox_inches="tight", dpi=150)
print(f"Saved {GRAPHS / 'nncg_deblur_bench.pdf'}")


# ---------------------------------------------------------------------------
# Table: per-size wall-clock and dense-A memory
# ---------------------------------------------------------------------------

def fmt(t):
    return f"{t:.2f}" if np.isfinite(t) else "---"


with open(TABLES / "nncg_deblur.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_deblur.py -- do not edit by hand.\n")
    fh.write("\\begin{tabular}{rrrrrrrr}\n\\toprule\n")
    fh.write("$N$ & $n = N^2$ & dense $A$ (GB) & LH (s) & Clarabel (s) & OSQP (s) "
             "& MPRGP (s) & NNCG (s) \\\\\n\\midrule\n")
    for r in rows:
        fh.write(f"{r['nside']} & {r['n']} & {r['dense_gb']:.2f} & "
                 f"{fmt(r['t_lh'])} & {fmt(r['t_cl'])} & {fmt(r['t_osqp'])} & "
                 f"{r['t_mprgp']:.2f} & {r['t_mf']:.2f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"Saved {TABLES / 'nncg_deblur.tex'}")


# ---------------------------------------------------------------------------
# Headline macros
# ---------------------------------------------------------------------------

recon_row = next(r for r in rows if r["nside"] == RECON_NSIDE)
big = rows[-1]                                        # largest matrix-free size
dense_rows = [r for r in rows if np.isfinite(r["t_lh"])]
head = dense_rows[-1]                                 # largest size the baselines reached
active = int(np.sum(recon["x"] <= 1e-8))
recov = float(np.max(np.abs(recon["x"] - recon["X_true"].ravel())))

with open(TABLES / "nncg_deblur_defs.tex", "w") as fh:
    fh.write("% Generated by experiment_nncg_deblur.py -- do not edit by hand.\n")
    # Reconstruction panel (unchanged interface used by Section 7).
    fh.write(f"\\newcommand{{\\nncgDeblurNside}}{{{RECON_NSIDE}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurDim}}{{{recon_row['n']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurOuter}}{{{recon_row['outer']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurInner}}{{{recon_row['inner']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurActive}}{{{active}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurDenseGB}}{{{recon_row['dense_gb']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurRecov}}{{{recov:.1e}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurPSNRblur}}{{{psnr_blur:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurPSNR}}{{{psnr_recon:.1f}}}\n")
    # Largest matrix-free size and its (never-formed) dense footprint.
    fh.write(f"\\newcommand{{\\nncgDeblurMaxNside}}{{{big['nside']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurMaxDim}}{{{big['n']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurMaxTime}}{{{big['t_mf']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurMaxMPRGPTime}}{{{big['t_mprgp']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurMaxDenseGB}}{{{big['dense_gb']:.0f}}}\n")
    # Largest size all three solvers reached, and the matrix-free speedups there.
    fh.write(f"\\newcommand{{\\nncgDeblurHeadNside}}{{{head['nside']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurHeadDim}}{{{head['n']}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurHeadMFTime}}{{{head['t_mf']:.2f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurHeadLHTime}}{{{head['t_lh']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurHeadIPTime}}{{{head['t_cl']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurHeadLHx}}{{{head['t_lh'] / head['t_mf']:.0f}}}\n")
    fh.write(f"\\newcommand{{\\nncgDeblurHeadIPx}}{{{head['t_cl'] / head['t_mf']:.0f}}}\n")
print(f"Saved {TABLES / 'nncg_deblur_defs.tex'}")
print("\nDone.")
