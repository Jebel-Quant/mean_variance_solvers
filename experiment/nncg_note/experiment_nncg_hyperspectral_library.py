"""Library-scale FCLS unmixing: the same comparison at n=498, not n=14.

experiment_nncg_hyperspectral.py runs per-pixel FCLS abundance unmixing
against n = 14 VCA-extracted endmembers, where NNCG's wall-clock edge over
Clarabel (interior-point) is small (see \\nncgHsSpeedupWall there): at n=14,
factorising the 14x14 Gram matrix is already so cheap that neither Clarabel's
per-iteration cost nor NNCG's matrix-free Krylov step has room to
differentiate -- both solves are dominated by Python/binding overhead, not
by the linear algebra the paper's argument is actually about.

This script reruns the identical QP

    min_x  1/2 x^T A x - b_i^T x   s.t.  x >= 0,  1^T x = 1,   A = M^T M,

against the full USGS 1995 spectral library instead of scene-extracted
endmembers: M has n = 498 candidate mineral signatures (not 14), the
standard over-complete dictionary of the sparse-unmixing literature
(Iordache & Bioucas-Dias and successors benchmark against this exact set on
this exact Cuprite scene). At this scale Clarabel's per-pixel factorisation
cost grows like n^3 (498/14)^3 =~ 4.0e4 times the flops of the n=14 case,
while NNCG's inner CG step stays matrix-free and linear in n per iteration --
this is the regime the paper's "avoid forming/factorising A" argument is
about, unlike the n=14 case where it is moot.

Because a single solve is now far more expensive, this script does not loop
over the whole scene: it takes a contiguous raster-order block of
--n-sample pixels (a spatial neighbourhood, so the warm-start mechanism of
Prop. 5.2 still applies) and runs NNCG (cold and warm), Clarabel and FISTA
on that identical sample -- an honest apples-to-apples timing at this scale,
rather than an extrapolation from a separately-sized sample as in the n=14
script.

A = M^T M is rank <= (number of clean bands), 188 here, strictly less than
n = 498: with more library columns than bands, A is only positive
*semi*-definite, not the A > 0 this paper's solvers assume throughout --
unlike the n=14 VCA case, where scene-extracted endmembers keep n well below
the band count and A is generically full rank. This is the ridge/Tikhonov
case flagged in the paper's abstract: --ridge-frac (default 1e-6, applied as
a fraction of A's largest eigenvalue, itself read off M's singular values
without ever forming A) adds ridge * I to secure A > 0. NNCG takes the ridge
through cvx.linalg.GramOperator(M, ridge), which never forms the n x n
matrix even with the ridge added; Clarabel and FISTA need an explicit dense
matrix regardless, so they receive M.T @ M + ridge * I directly.

On the real Cuprite scene against the real USGS library, every pixel's
solution has support size 1 (one active library member), regardless of
ridge -- checked over --ridge-frac from 1e-3 to 1e-6 with no change. This
is not a regularisation artefact: the winning library column typically
matches the pixel spectrum at ~0.996 cosine correlation already, so the QP
has no incentive to blend it with anything else. With 498 mostly-distinct
mineral signatures (a modest ~51 near-duplicate pairs out of ~124,000), a
generic Cuprite pixel is well explained by one dominant match -- a real
property of this scene/library pair, not a mixture-unmixing showcase. The
timing comparison below (NNCG vs Clarabel vs FISTA) is unaffected by this --
all three solve the identical QP -- but the recovered abundances themselves
are effectively 1-nearest-library-neighbour assignments here, not sparse
mixtures.

Data: the real USGS 1995 spectral library (498 real mineral reflectance
spectra over the raw 224-band AVIRIS grid, the same wavelength grid as the
Cuprite cube) is not redistributed here. It looks for a local copy at
--library (default DATA/USGS_1995_Library.mat, a scipy.io.loadmat-readable
file with a `datalib` array of shape (224, n+3) -- the first 3 columns are
wavelength/resolution/channel-number metadata, not spectra) and, absent
that, falls back to a synthetic random library of the same shape so the
script still runs end to end offline. A real copy is mirrored (for research
use) at:
    https://raw.githubusercontent.com/ricardoborsoi/MUA_SparseUnmixing/master/USGS_1995_Library.mat
originally the USGS splib06 library resampled to the AVIRIS band grid by
Iordache, Bioucas-Dias & Plaza. The Cuprite scene itself is loaded exactly
as in experiment_nncg_hyperspectral.py (--data / --synthetic).

Usage:
    uv run python -m nncg_note.experiment_nncg_hyperspectral_library
    uv run python -m nncg_note.experiment_nncg_hyperspectral_library --data path/to/cuprite.mat --library path/to/USGS_1995_Library.mat

Outputs:
    graphs/nncg_hyperspectral_library.pdf        outer-iteration + support-size histograms
    tables/nncg_hyperspectral_library_defs.tex   headline numbers as \\newcommand macros
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cvx.linalg import GramOperator
from common.util.runner import SMOKE, output_dirs

from nncg_note.experiment_nncg_hyperspectral import (
    load_scene,
    prune_bands,
    fista_fcls,
    run_clarabel_fcls,
    snake_order,
    unmix_image,
)

HERE = Path(__file__).resolve().parents[1]  # experiment/ root (data, graphs, tables)
GRAPHS, TABLES = output_dirs(HERE)
DATA = HERE / "data"

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

N_SIG_SYNTHETIC = 498  # match the real library's signature count
N_RAW_BANDS = 224


def load_library(path):
    """Load the USGS 1995 spectral library from a local .mat file; None if absent.

    `datalib` is (224, n+3): the first 3 columns are metadata (wavelengths in
    microns, resolution, channel number), not reflectance spectra -- see the
    module docstring. Returns (224, n) real signatures.
    """
    if not path.exists():
        return None
    from scipy.io import loadmat
    mat = loadmat(path)
    return mat["datalib"][:, 3:].astype(float)


def synthetic_library(n_sig=N_SIG_SYNTHETIC, n_bands=N_RAW_BANDS, seed=1):
    """A synthetic stand-in library: n_sig smoothed random reflectance-like
    curves over n_bands, in the same style as experiment_nncg_hyperspectral's
    synthetic endmembers, just at library scale rather than 14 endmembers."""
    rng = np.random.default_rng(seed)
    band = np.arange(n_bands)
    lib = np.zeros((n_bands, n_sig))
    for k in range(n_sig):
        base = 0.3 + 0.5 * rng.random()
        curve = base * np.ones(n_bands)
        n_features = rng.integers(2, 5)
        for _ in range(n_features):
            centre = rng.uniform(0, n_bands)
            width = rng.uniform(8, 25)
            depth = rng.uniform(0.1, 0.4)
            curve -= depth * np.exp(-0.5 * ((band - centre) / width) ** 2)
        lib[:, k] = np.clip(curve, 0.02, None)
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA / "cuprite.mat"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--library", default=str(DATA / "USGS_1995_Library.mat"))
    ap.add_argument("--n-sample", type=int, default=500,
                     help="contiguous raster-order pixels to unmix against the library")
    ap.add_argument("--ridge-frac", type=float, default=1e-6,
                     help="ridge added to A = M'M, as a fraction of A's largest eigenvalue; "
                          "needed because n (library size) typically exceeds the band count, "
                          "making the un-ridged A only positive semi-definite. On the real "
                          "Cuprite scene/USGS library pair, support size is ~1 regardless of "
                          "this value over 1e-3..1e-6 (see module docstring) -- kept small "
                          "here since it isn't buying anything, not because a larger value "
                          "was observed to hurt")
    ap.add_argument("--fista-tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if SMOKE:  # keep the smoke run to seconds: a tiny sample against a tiny library
        args.n_sample = min(args.n_sample, 20)

    cube, _, _ = load_scene(args)  # ground truth (synthetic-only) isn't meaningful against a library dictionary
    rows, cols, m = cube.shape
    Y = cube.reshape(-1, m).T
    N = Y.shape[1]

    lib_path = Path(args.library)
    lib = load_library(lib_path)
    if lib is None:
        print(f"No local library at {lib_path}; using a synthetic random library instead. "
              "See the module docstring for how to fetch the real USGS_1995_Library.mat.")
        lib = synthetic_library(n_sig=N_SIG_SYNTHETIC if not SMOKE else 40, seed=args.seed)
    M = prune_bands(lib.T).T  # (bands, n_sig) -- same water-absorption removal as the cube
    if M.shape[0] != m:
        raise ValueError(f"library has {M.shape[0]} clean bands but the scene has {m}; "
                          "both must start from the same raw 224-band AVIRIS grid")
    n_sig = M.shape[1]
    print(f"Library: {n_sig} signatures over {m} clean bands")

    order = snake_order(rows, cols)[: min(args.n_sample, N)]
    n_sample = len(order)
    print(f"Unmixing {n_sample} pixels (contiguous raster block) against n={n_sig} library signatures...")

    # A = M'M has rank <= m (clean bands) < n_sig here, so it is only PSD --
    # a ridge is needed to reach the A > 0 this solver family assumes. Sized
    # off M's singular values, so forming the dense n_sig x n_sig A isn't
    # needed just to pick it.
    lambda_max = float(np.linalg.svd(M, compute_uv=False)[0] ** 2)
    ridge = args.ridge_frac * lambda_max
    rank_A = int(np.linalg.matrix_rank(M))
    print(f"A = M'M: {n_sig}x{n_sig}, rank <= {rank_A} (bands={m}) -- "
          f"ridge={ridge:.3g} ({args.ridge_frac:.0e} x lambda_max={lambda_max:.3g})")

    A_op = GramOperator(M, ridge=ridge)  # matrix-free even with the ridge: never forms n_sig x n_sig
    A_reg = M.T @ M + ridge * np.eye(n_sig)  # Clarabel/FISTA need an explicit dense matrix regardless
    Bmat = np.ones((1, n_sig))
    c = np.array([1.0])
    Bpix = M.T @ Y[:, order]  # (n_sig, n_sample); only the sampled pixels' rhs
    local_order = np.arange(n_sample)  # Bpix's own columns are already in raster order

    print("Unmixing (cold start: every pixel from scratch)...")
    X_cold, outer_cold, inner_cold, t_cold = unmix_image(
        Bpix, A_op, Bmat, c, local_order, warm_start=False, label="cold")
    print(f"  {t_cold:.2f}s, total outer iters = {outer_cold.sum()}")

    print("Unmixing (warm start: free set carried along the raster scan)...")
    X_warm, outer_warm, inner_warm, t_warm = unmix_image(
        Bpix, A_op, Bmat, c, local_order, warm_start=True, label="warm")
    print(f"  {t_warm:.2f}s, total outer iters = {outer_warm.sum()}")

    agree = float(np.max(np.abs(X_cold - X_warm)))
    print(f"Cold vs warm start agree to {agree:.1e} (same active-set fixed points)")

    # Clarabel and FISTA on the *same* sample -- at this scale the sample is
    # the comparison itself, not an extrapolation proxy for a full-scene run.
    t0 = time.perf_counter()
    for j in range(n_sample):
        run_clarabel_fcls(A_reg, Bpix[:, j], n_sig)
    t_clarabel = time.perf_counter() - t0
    print(f"Clarabel on {n_sample} pixels: {t_clarabel:.2f}s ({1000 * t_clarabel / n_sample:.2f} ms/pixel)")

    L = lambda_max + ridge  # shared Lipschitz const of A_reg's gradient
    fista_iters = np.zeros(n_sample, dtype=int)
    t0 = time.perf_counter()
    for j in range(n_sample):
        _, fista_iters[j] = fista_fcls(A_reg, Bpix[:, j], L, tol=args.fista_tol)
    t_fista = time.perf_counter() - t0
    print(f"FISTA on {n_sample} pixels: {t_fista:.2f}s, "
          f"mean {fista_iters.mean():.0f} / max {fista_iters.max()} iterations "
          f"to residual < {args.fista_tol:.0e}")

    frac_single = float(np.mean(outer_warm == 1))
    speedup_outer = float(outer_cold.sum() / max(outer_warm.sum(), 1))
    speedup_wall = float(t_clarabel / max(t_warm, 1e-9))
    speedup_fista = float(t_fista / max(t_warm, 1e-9))
    support_size = np.sum(X_warm > 1e-6, axis=0)  # active (nonzero-abundance) library members per pixel

    # -----------------------------------------------------------------------
    # Figure: outer-iteration histogram (clipped, with outliers reported
    # rather than silently stretching the axis -- see the n=14 script, where
    # a few pathological pixels made this panel unreadable) + support-size
    # histogram, the diagnostic that's actually informative at library scale.
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))

    cap = max(20, int(np.percentile(np.concatenate([outer_cold, outer_warm]), 99)))
    n_over_cold = int((outer_cold > cap).sum())
    n_over_warm = int((outer_warm > cap).sum())
    bins = np.arange(0, cap + 2) - 0.5
    axes[0].hist(np.clip(outer_cold, 0, cap), bins=bins, alpha=0.6, label="cold start", color="#d62728")
    axes[0].hist(np.clip(outer_warm, 0, cap), bins=bins, alpha=0.6, label="warm start", color="#2ca02c")
    axes[0].set_xlabel(f"outer (active-set) iterations, clipped to {cap}")
    axes[0].set_ylabel("pixels")
    if n_over_cold or n_over_warm:
        axes[0].set_title(f"{n_over_cold} cold / {n_over_warm} warm pixels >{cap} (clipped)", fontsize=7)
    axes[0].legend()

    axes[1].hist(support_size, bins=np.arange(0, support_size.max() + 2) - 0.5, color="#1f77b4")
    axes[1].set_xlabel(f"active library members per pixel (of n={n_sig})")
    axes[1].set_ylabel("pixels")
    fig.tight_layout()
    fig.savefig(GRAPHS / "nncg_hyperspectral_library.pdf")
    print(f"\nSaved {GRAPHS / 'nncg_hyperspectral_library.pdf'}")

    # -----------------------------------------------------------------------
    # LaTeX macros
    # -----------------------------------------------------------------------
    with open(TABLES / "nncg_hyperspectral_library_defs.tex", "w") as fh:
        fh.write("% Generated by experiment_nncg_hyperspectral_library.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\nncgHsLibPixels}}{{{n_sample}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibBands}}{{{m}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibSignatures}}{{{n_sig}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibRank}}{{{rank_A}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibRidgeFrac}}{{{args.ridge_frac:.0e}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibOuterCold}}{{{int(outer_cold.sum())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibOuterWarm}}{{{int(outer_warm.sum())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibFracSingle}}{{{frac_single * 100:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibSpeedupOuter}}{{{speedup_outer:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibSpeedupWall}}{{{speedup_wall:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibAgree}}{{{agree:.0e}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibSpeedupFista}}{{{speedup_fista:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibFistaIters}}{{{int(round(fista_iters.mean()))}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibFistaItersMax}}{{{int(fista_iters.max())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibSupportMean}}{{{support_size.mean():.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsLibSupportMax}}{{{int(support_size.max())}}}\n")
    print(f"Saved {TABLES / 'nncg_hyperspectral_library_defs.tex'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
