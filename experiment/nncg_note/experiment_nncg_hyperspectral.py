"""AVIRIS Cuprite hyperspectral unmixing warm start for "Non-Negative Conjugate Gradients".

Per-pixel fully-constrained least-squares (FCLS) abundance unmixing is the
p=1 equality-augmented QP of Section 3 at the smallest possible scale: for
n = 14 endmembers the Gram matrix A = M^T M in R^{14x14} is tiny, and every
one of the N ~ 47,750 pixels of the classic 250x191 AVIRIS Cuprite sub-scene
poses

    min_x  1/2 x^T A x - b_i^T x   s.t.  x >= 0,  1^T x = 1,   b_i = M^T d_i.

This script instantiates the raster-scan warm start of Section 5 (Prop. 5.2):
neighbouring pixels usually share the same mineral mix, so passing pixel
(i-1)'s (free mask, solution) pair into solve_nnqp_eq's `warm` argument for
pixel i lets many pixels certify in a single outer iteration instead of
re-searching the active set from scratch (cold start). On the offline
synthetic proxy scene (see
below, tuned for realistic per-pixel mineral contrast) the effect is real but
modest: roughly a 1.1-1.2x reduction in total outer iterations, well short of
the 1.6-1.7x an earlier, unrealistically flat-abundance version of this scene
showed. The mechanism is active-set overlap between neighbours, which is
driven by *spatial* coherence of the dominant material, not by how peaked
each pixel's own mixture is -- a scene tuned only to look right per-pixel can
overstate the warm start's benefit. The script also times a per-pixel
Clarabel interior-point baseline on a random sample (looping it over all
47,750 pixels is representative of "processes every pixel cold" but too slow
to run in full here); on this realistic-contrast scene the two are roughly
matched (NNCG is not reliably faster here -- see \\nncgHsSpeedupWall), unlike
the flat-abundance scene where NNCG had a clear wall-clock edge. The FISTA
comparison below is the more telling one at this problem size.

Endmembers M in R^{188x14} are extracted from the scene itself by Vertex
Component Analysis (Nascimento & Bioucas-Dias, 2005): project the (mean-
centred) radiance onto its leading endmember-count-many singular directions,
then repeatedly pick the pixel with the largest projection onto a direction
orthogonal to the vertices already found. This is the "high-SNR" branch of
VCA (no explicit SNR test / affine augmentation), a reasonable simplification
for a well-illuminated mineral scene like Cuprite and standard in tutorials.

Besides Clarabel, the script also benchmarks FISTA (Beck & Teboulle, 2009):
accelerated projected gradient onto the probability simplex, the standard
first-order baseline for FCLS unmixing. Its per-iteration cost is a single
14-by-14 mat-vec plus an O(n log n) simplex projection -- cheaper per step
than NNCG's active-set/CG loop, but it needs many steps (no finite active-set
termination), timed to the same fixed-point-residual tolerance on the same
pixel sample as the Clarabel baseline.

Quality against ground truth: the synthetic proxy scene's generative
abundances and endmember spectra are known exactly (they're what the cube
was built from), so on that scene the recovered abundances can be scored,
not just cross-checked for cold/warm self-consistency. VCA's extracted
endmembers are matched to the generative spectra by a Hungarian assignment
on spectral angle before scoring -- VCA recovers the right vertices, but not
necessarily in the generative endmember order. Real Cuprite has no exact
per-pixel abundance ground truth (only coarse USGS Tricorder mineral maps),
so this quality check only runs on the synthetic scene.

Data: this script does not embed a redistribution of the AVIRIS Cuprite
imagery (usage terms vary by mirror). It looks for a local cube at
--data (default DATA/cuprite.mat, any key holding a 3-D array) and, if none
is found, prints instructions and falls back to a synthetic proxy scene of
the same shape (250x191 pixels, 224 bands, 14 endmembers) built from smooth
softmax-of-Gaussian-fields abundances -- non-negative, sum-to-one, and
spatially coherent by construction, so the warm-start mechanic and its
headline numbers reproduce end to end with no network access. Real Cuprite
data (flight f970619t01p02_r02) is distributed by NASA/JPL's AVIRIS project,
https://aviris.jpl.nasa.gov/data/free_data.html; several unmixing papers also
redistribute a pre-cropped 250x191x224 cube (commonly named cuprite_ref.mat
or similar) -- point --data at any such file.

Usage:
    uv run python -m nncg_note.experiment_nncg_hyperspectral                # real data if found, else synthetic
    uv run python -m nncg_note.experiment_nncg_hyperspectral --synthetic     # force the offline proxy scene
    uv run python -m nncg_note.experiment_nncg_hyperspectral --data path.mat # use a specific cube

Outputs:
    graphs/nncg_hyperspectral.pdf        abundance map, error map (synthetic only) + outer-iteration histogram
    tables/nncg_hyperspectral_defs.tex   headline numbers as \\newcommand macros
"""


from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from cvx.linalg import DenseOperator
from common.util.runner import SMOKE, output_dirs

from nncg import solve_nnqp_eq

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

N_ROWS, N_COLS, N_BANDS = 250, 191, 224
if SMOKE:  # tiny synthetic scene so the end-to-end smoke test stays fast
    N_ROWS, N_COLS = 40, 30
N_ENDMEMBERS = 14
BAD_BANDS_1INDEXED = [(1, 2), (104, 113), (148, 167), (221, 224)]  # water absorption / low SNR
ABUNDANCE_TEMP = 40.0  # softmax sharpness: mean dominant-endmember fraction ~0.74 (real-mineral-map-like)


def prune_bands(cube):
    """Drop the water-absorption / low-SNR bands, 224 -> ~188 clean bands.

    cube is (rows, cols, bands); returns the pruned cube.
    """
    bad = set()
    for lo, hi in BAD_BANDS_1INDEXED:
        bad.update(range(lo - 1, hi))
    keep = [i for i in range(cube.shape[-1]) if i not in bad]
    return cube[..., keep]


# ---------------------------------------------------------------------------
# Data: a local Cuprite cube if available, else a spatially-coherent proxy
# ---------------------------------------------------------------------------

def find_band_axis(shape, target_pixels=N_ROWS * N_COLS):
    """Identify the band axis of a 3-D cube of unknown orientation.

    Picks whichever axis, when excluded, leaves the other two multiplying
    closest to the known 250x191 = 47,750 pixel count of this sub-scene --
    unambiguous for the canonical shape, where a naive "largest axis" or
    "closest to 200" guess mis-fires (250 > 224, and |224-200| > |191-200|).
    """
    shape = list(shape)
    products = []
    for i in range(3):
        others = [shape[j] for j in range(3) if j != i]
        products.append(abs(others[0] * others[1] - target_pixels))
    return int(np.argmin(products))


def load_local_cuprite(path):
    """Load a user-supplied cube from a .mat/.npy file; None if not present."""
    if not path.exists():
        return None
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        from scipy.io import loadmat
        mat = loadmat(path)
        arr = max((v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 3),
                  key=np.size)
    band_axis = find_band_axis(arr.shape)
    cube = np.moveaxis(arr, band_axis, -1).astype(float)
    print(f"Loaded local cube {path}, shape {cube.shape} (band axis -> last)")
    return cube


def synthetic_cuprite(seed=0):
    """A synthetic proxy scene with the same shape and genuine spatial coherence.

    Endmember spectra are smoothed random curves (non-negative, absorption-
    feature-like dips) over N_BANDS bands. Abundances are the softmax, across
    endmembers, of N_ENDMEMBERS independent Gaussian-blurred random fields --
    softmax lands automatically on the simplex (>=0, sums to 1) and inherits
    the fields' smoothness, so neighbouring pixels share nearly the same
    dominant material exactly as in a real mineral scene.

    Note on realism: bands (188) vastly outnumber endmembers (14), so FCLS
    recovery is a well-conditioned, overdetermined regression -- even a cold
    active-set search rarely needs many iterations, and any given synthetic
    proxy is at best a conservative stand-in for real spectral/spatial
    structure. Numbers reported by this script are measured on this proxy,
    not asserted a priori; pass --data to run on a real Cuprite cube instead.

    Returns (cube, abundances, spectra): abundances (rows, cols, k) and
    spectra (k, N_BANDS) are the exact generative ground truth, kept so the
    caller can score recovered abundances against a known answer -- unlike
    real Cuprite, which has no exact per-pixel abundance ground truth.
    """
    rng = np.random.default_rng(seed)

    band = np.arange(N_BANDS)
    spectra = np.zeros((N_ENDMEMBERS, N_BANDS))
    for k in range(N_ENDMEMBERS):
        base = 0.3 + 0.5 * rng.random()
        curve = base * np.ones(N_BANDS)
        n_features = rng.integers(2, 5)
        for _ in range(n_features):
            centre = rng.uniform(0, N_BANDS)
            width = rng.uniform(8, 25)
            depth = rng.uniform(0.1, 0.4)
            curve -= depth * np.exp(-0.5 * ((band - centre) / width) ** 2)
        spectra[k] = np.clip(curve, 0.02, None)

    def blurred_field():
        coarse = rng.standard_normal((N_ROWS, N_COLS))
        # separable Gaussian blur via repeated small-kernel convolution (no scipy.ndimage dependency)
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        k /= k.sum()
        field = coarse
        for _ in range(6):
            field = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, field)
            field = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, field)
        return field

    fields = np.stack([blurred_field() for _ in range(N_ENDMEMBERS)], axis=-1)  # rows, cols, k
    abundances = np.exp(ABUNDANCE_TEMP * fields)
    abundances /= abundances.sum(axis=-1, keepdims=True)                        # simplex, spatially smooth

    cube = abundances @ spectra                                                 # rows, cols, bands
    noise = 0.01 * rng.standard_normal(cube.shape) * cube.mean()
    cube = np.clip(cube + noise, 0.0, None)
    return cube, abundances, spectra


def load_scene(args):
    cube = None if args.synthetic else load_local_cuprite(Path(args.data))
    if cube is not None:
        return prune_bands(cube), None, None
    print(f"No local cube at {args.data} (or --synthetic requested); "
          "using the offline synthetic proxy scene. See the module "
          "docstring for how to point --data at a real Cuprite cube.")
    cube, abundances, spectra = synthetic_cuprite()
    return prune_bands(cube), abundances, prune_bands(spectra)


# ---------------------------------------------------------------------------
# Vertex Component Analysis (Nascimento & Bioucas-Dias, 2005), high-SNR branch
# ---------------------------------------------------------------------------

def vca(Y, n_endmembers, seed=0):
    """Extract endmember spectra as n_endmembers extremal columns of Y.

    Y is (bands, pixels). Projects onto the leading n_endmembers left
    singular vectors of Y (a linear subspace containing the mixing simplex
    for well-illuminated, high-SNR data), then greedily picks, for a
    sequence of random directions orthogonal to the vertices already found,
    the pixel with the largest projection -- the VCA vertex search.
    """
    rng = np.random.default_rng(seed)
    Ud, _, _ = np.linalg.svd(Y, full_matrices=False)
    Ud = Ud[:, :n_endmembers]
    Yp = Ud.T @ Y                                     # (n_endmembers, pixels)

    A = np.zeros((n_endmembers, n_endmembers))
    A[0, 0] = 1.0
    indices = []
    for i in range(n_endmembers):
        w = rng.standard_normal(n_endmembers)
        f = w - A @ (np.linalg.pinv(A) @ w)
        f /= np.linalg.norm(f)
        v = f @ Yp
        idx = int(np.argmax(np.abs(v)))
        A[:, i] = Yp[:, idx]
        indices.append(idx)
    return Y[:, indices], indices


def match_endmembers(M_est, spectra_true):
    """Match VCA-extracted endmember columns of M_est (bands, k) to the
    generative spectra rows of spectra_true (k, bands) by spectral angle.

    VCA recovers the right vertices but in an arbitrary order, so scoring
    recovered abundances against ground truth needs a permutation first.
    Cosine similarity + a Hungarian assignment (scipy.optimize.
    linear_sum_assignment) gives the optimal one-to-one matching. Returns
    perm with perm[i] the M_est column matching spectra_true row i.
    """
    from scipy.optimize import linear_sum_assignment
    est = M_est / np.linalg.norm(M_est, axis=0, keepdims=True)
    true = spectra_true / np.linalg.norm(spectra_true, axis=1, keepdims=True)
    sim = true @ est                                       # (k, k) cosine similarities
    row, col = linear_sum_assignment(-sim)
    return col[np.argsort(row)]


# ---------------------------------------------------------------------------
# FISTA FCLS baseline (accelerated projected gradient onto the simplex)
# ---------------------------------------------------------------------------

def project_simplex(v):
    """Euclidean projection of v onto {x >= 0, sum(x) = 1} (Held et al., 1974).

    Sort-and-threshold: the projection is v shifted down by a scalar theta and
    clipped at 0, with theta chosen so the result sums to 1.
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    rho = idx[u - css / idx > 0][-1]
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def fista_fcls(A, b, L, tol=1e-6, max_iter=2000):
    """FISTA (Beck & Teboulle, 2009) on min 1/2 x^T A x - b^T x s.t. x in simplex.

    L is the gradient's Lipschitz constant (the largest eigenvalue of A);
    since A = M^T M is shared by every pixel it is computed once for the
    whole scene, not per call. Stops on the fixed-point residual
    ||x - P(x - grad(x)/L)||_inf, the exact stationarity certificate for a
    projected-gradient step at any step size > 0 -- comparable in spirit to
    solve_nnqp_eq's KKT-based stopping rule. Returns (x, n_iter).
    """
    x = np.full(A.shape[0], 1.0 / A.shape[0])
    y = x.copy()
    t = 1.0
    for k in range(1, max_iter + 1):
        grad = A @ y - b
        x_new = project_simplex(y - grad / L)
        resid = np.max(np.abs(x_new - project_simplex(x_new - (A @ x_new - b) / L)))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if resid < tol:
            return x, k
    return x, max_iter


# ---------------------------------------------------------------------------
# Clarabel FCLS baseline (interior point, "processes every pixel cold")
# ---------------------------------------------------------------------------

def run_clarabel_fcls(A, b, n_endmembers):
    import clarabel
    P = sp.triu(sp.csc_matrix(A), format="csc")
    rows = sp.vstack([sp.csc_matrix(np.ones((1, n_endmembers))),
                       -sp.identity(n_endmembers, format="csc")], format="csc")
    rhs = np.concatenate([[1.0], np.zeros(n_endmembers)])
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(
        P, -b, rows, rhs,
        [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n_endmembers)], settings)
    return np.asarray(solver.solve().x)


# ---------------------------------------------------------------------------
# Raster/snake pixel order: consecutive pixels are spatial neighbours
# ---------------------------------------------------------------------------

def snake_order(n_rows, n_cols):
    order = []
    for r in range(n_rows):
        cols = range(n_cols) if r % 2 == 0 else range(n_cols - 1, -1, -1)
        order.extend(r * n_cols + c for c in cols)
    return np.array(order)


def unmix_image(Y, A_op, B, c, order, warm_start):
    """Solve the FCLS QP for every pixel in `order`; optionally warm-start
    each solve's free set and iterate from the previous pixel's solution
    (Prop. 5.2), via the nncg package's `warm=(free_mask, x_prev)` argument."""
    n_end = B.shape[1]
    n_pix = Y.shape[1]
    X = np.zeros((n_end, n_pix))
    outer = np.zeros(n_pix, dtype=int)
    inner = np.zeros(n_pix, dtype=int)
    warm = None
    t0 = time.perf_counter()
    for i in order:
        res = solve_nnqp_eq(A_op, Y[:, i], B, c, warm=warm if warm_start else None)
        X[:, i] = res.x
        outer[i] = res.outer
        inner[i] = res.inner
        if warm_start:
            warm = (res.free, res.x)
    elapsed = time.perf_counter() - t0
    return X, outer, inner, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA / "cuprite.mat"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--clarabel-sample", type=int, default=1500,
                     help="pixels sampled for the Clarabel/FISTA per-pixel timing baselines")
    ap.add_argument("--fista-tol", type=float, default=1e-6,
                     help="fixed-point-residual tolerance for the FISTA baseline")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if SMOKE:  # cap the per-pixel baseline sample to keep the smoke run quick
        args.clarabel_sample = min(args.clarabel_sample, 80)

    cube, true_abundances, true_spectra = load_scene(args)
    rows, cols, m = cube.shape
    Y = cube.reshape(-1, m).T                                        # (m, N)
    N = Y.shape[1]
    print(f"Scene: {rows}x{cols} pixels, {m} clean bands, N = {N} spectra")

    M, endmember_px = vca(Y, N_ENDMEMBERS, seed=args.seed)
    print(f"VCA endmembers: {N_ENDMEMBERS} extracted from pixel indices {endmember_px[:5]}...")

    A = M.T @ M
    A_op = DenseOperator(A)                                           # shared across every pixel's solve
    Bmat = np.ones((1, N_ENDMEMBERS))
    c = np.array([1.0])
    Bpix = M.T @ Y                                                    # b_i columns, all pixels at once

    order = snake_order(rows, cols)

    print("Unmixing (cold start: every pixel from scratch)...")
    X_cold, outer_cold, inner_cold, t_cold = unmix_image(Bpix, A_op, Bmat, c, order, warm_start=False)
    print(f"  {t_cold:.2f}s, total outer iters = {outer_cold.sum()}")

    print("Unmixing (warm start: free set carried along the raster scan)...")
    X_warm, outer_warm, inner_warm, t_warm = unmix_image(Bpix, A_op, Bmat, c, order, warm_start=True)
    print(f"  {t_warm:.2f}s, total outer iters = {outer_warm.sum()}")

    agree = float(np.max(np.abs(X_cold - X_warm)))
    print(f"Cold vs warm start agree to {agree:.1e} (same active-set fixed points)")

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(N, size=min(args.clarabel_sample, N), replace=False)

    t0 = time.perf_counter()
    for i in sample:
        run_clarabel_fcls(A, Bpix[:, i], N_ENDMEMBERS)
    t_clarabel_sample = time.perf_counter() - t0
    t_clarabel_full = t_clarabel_sample * N / sample.size
    print(f"Clarabel on {sample.size} sampled pixels: {t_clarabel_sample:.2f}s "
          f"(extrapolated full scene: {t_clarabel_full:.1f}s)")

    L = float(np.linalg.eigvalsh(A)[-1])                              # shared Lipschitz const
    fista_iters = np.zeros(sample.size, dtype=int)
    t0 = time.perf_counter()
    for j, i in enumerate(sample):
        _, fista_iters[j] = fista_fcls(A, Bpix[:, i], L, tol=args.fista_tol)
    t_fista_sample = time.perf_counter() - t0
    t_fista_full = t_fista_sample * N / sample.size
    print(f"FISTA on {sample.size} sampled pixels: {t_fista_sample:.2f}s "
          f"(extrapolated full scene: {t_fista_full:.1f}s), "
          f"mean {fista_iters.mean():.0f} / max {fista_iters.max()} iterations "
          f"to residual < {args.fista_tol:.0e}")

    frac_single = float(np.mean(outer_warm == 1))
    speedup_outer = float(outer_cold.sum() / max(outer_warm.sum(), 1))
    speedup_wall = float(t_clarabel_full / max(t_warm, 1e-9))
    speedup_fista = float(t_fista_full / max(t_warm, 1e-9))

    # -----------------------------------------------------------------------
    # Quality against ground truth (synthetic scene only -- real Cuprite has
    # no exact per-pixel abundance ground truth to compare against)
    # -----------------------------------------------------------------------
    abund_rmse = abund_mae = None
    error_map = None
    if true_abundances is not None:
        perm = match_endmembers(M, true_spectra)
        X_matched = X_warm[perm]                                      # align to generative order
        X_true = true_abundances.reshape(-1, N_ENDMEMBERS).T
        diff = X_matched - X_true
        abund_rmse = float(np.sqrt(np.mean(diff ** 2)))
        abund_mae = float(np.mean(np.abs(diff)))
        error_map = np.mean(np.abs(diff), axis=0).reshape(rows, cols)
        print(f"Abundance recovery vs ground truth: RMSE {abund_rmse:.3e}, "
              f"MAE {abund_mae:.3e} (after Hungarian endmember matching)")

    # -----------------------------------------------------------------------
    # Figure: dominant-endmember abundance map, [ground-truth error map,]
    # outer-iteration histogram
    # -----------------------------------------------------------------------
    n_panels = 3 if error_map is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(3.3 * n_panels, 2.8))
    dominant = X_warm.argmax(axis=0).reshape(rows, cols)
    axes[0].imshow(dominant, cmap="tab20", interpolation="nearest")
    axes[0].set_title("dominant endmember per pixel")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    next_ax = 1
    if error_map is not None:
        im = axes[1].imshow(error_map, cmap="viridis", interpolation="nearest")
        axes[1].set_title("mean abs. abundance error")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        next_ax = 2

    bins = np.arange(1, max(outer_cold.max(), outer_warm.max()) + 2) - 0.5
    axes[next_ax].hist(outer_cold, bins=bins, alpha=0.6, label="cold start", color="#d62728")
    axes[next_ax].hist(outer_warm, bins=bins, alpha=0.6, label="warm start", color="#2ca02c")
    axes[next_ax].set_xlabel("outer (active-set) iterations")
    axes[next_ax].set_ylabel("pixels")
    axes[next_ax].legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "nncg_hyperspectral.pdf")
    print(f"\nSaved {GRAPHS / 'nncg_hyperspectral.pdf'}")

    # -----------------------------------------------------------------------
    # LaTeX macros
    # -----------------------------------------------------------------------
    with open(TABLES / "nncg_hyperspectral_defs.tex", "w") as fh:
        fh.write("% Generated by experiment_nncg_hyperspectral.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\nncgHsPixels}}{{{N}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsBands}}{{{m}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsEndmembers}}{{{N_ENDMEMBERS}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsOuterCold}}{{{int(outer_cold.sum())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsOuterWarm}}{{{int(outer_warm.sum())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsFracSingle}}{{{frac_single * 100:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsSpeedupOuter}}{{{speedup_outer:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsSpeedupWall}}{{{speedup_wall:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsAgree}}{{{agree:.0e}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsSpeedupFista}}{{{speedup_fista:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsFistaIters}}{{{int(round(fista_iters.mean()))}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsFistaItersMax}}{{{int(fista_iters.max())}}}\n")
        if abund_rmse is not None:
            fh.write(f"\\newcommand{{\\nncgHsAbundRmse}}{{{abund_rmse:.2e}}}\n")
            fh.write(f"\\newcommand{{\\nncgHsAbundMae}}{{{abund_mae:.2e}}}\n")
    print(f"Saved {TABLES / 'nncg_hyperspectral_defs.tex'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
