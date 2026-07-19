"""Urban hyperspectral unmixing: the same FCLS comparison, a visually legible scene.

experiment_nncg_hyperspectral.py's Cuprite mineral map is a poor visualization:
VCA-extracted endmembers have no names, and the dominant-endmember map is a
speckled mix of near-indistinguishable minerals. Urban (HYDICE sensor, 307x307
pixels, 162 clean bands after sensor/atmospheric band removal) is the standard
alternative in the unmixing literature specifically because its 6 materials --
Asphalt, Grass, Tree, Roof, Metal, Dirt -- are visually distinct on the ground
and the recovered dominant-material map reads as an actual aerial photo (roads,
lawns, rooftops), not a speckle field.

Endmembers are the per-class mean spectrum of a bundled 6-class reference
library (Asphalt/Grass/Tree/Roof/Metal/Dirt), not VCA -- Urban's library
already ships with named material classes, so there is no need to extract and
then guess what each vertex represents. n = 6 is even smaller than the n = 14
Cuprite case, so this script is not about the NNCG-vs-Clarabel wall-clock gap
(see experiment_nncg_hyperspectral_library.py for that, at n = 498); it is
about producing a dominant-material map worth looking at.

Same QP, same solvers as the other two scripts in this trio:

    min_x  1/2 x^T A x - b_i^T x   s.t.  x >= 0,  1^T x = 1,   A = M^T M,

with A = M^T M generically full rank here (6 << 162 bands), so no ridge is
needed, unlike the library-scale script.

Data: this script does not embed a redistribution of the Urban scene. It looks
for a local copy at --data (default DATA/Urban_R162.mat, a scipy.io.loadmat
file with `Y` (bands x pixels, uint16, scaled to `maxValue`), `nRow`, `nCol`)
and a matching --library (default DATA/spectral_library_urban.mat, with
`lib1`..`lib6` per-class candidate spectra and `material_names`). Absent
either, falls back to a small synthetic scene/library pair of the same shape
so the script still runs end to end offline. Real copies (for research use)
are mirrored at:
    https://raw.githubusercontent.com/ricardoborsoi/MUA_SparseUnmixing/master/real_data/Urban_R162.mat
    https://raw.githubusercontent.com/ricardoborsoi/MUA_SparseUnmixing/master/real_data/spectral_library_urban.mat

Usage:
    uv run python -m nncg_note.experiment_nncg_hyperspectral_urban
    uv run python -m nncg_note.experiment_nncg_hyperspectral_urban --data path/to/Urban_R162.mat --library path/to/spectral_library_urban.mat

Outputs:
    graphs/nncg_hyperspectral_urban.pdf        dominant-material map + outer-iteration histogram
    tables/nncg_hyperspectral_urban_defs.tex   headline numbers as \\newcommand macros
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cvx.linalg import DenseOperator
from common.util.runner import SMOKE, output_dirs

from nncg_note.experiment_nncg_hyperspectral import (
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
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)

N_ROWS, N_COLS, N_BANDS = 307, 307, 162
if SMOKE:  # tiny synthetic scene so the end-to-end smoke test stays fast
    N_ROWS, N_COLS = 40, 30
MATERIAL_NAMES = ["Asphalt", "Grass", "Tree", "Roof", "Metal", "Dirt"]
N_MATERIALS = len(MATERIAL_NAMES)


def load_urban(path):
    """Load the real Urban scene from a local .mat file; None if absent.

    `Y` is (bands, pixels) on a 0..maxValue integer scale -- rescaled to the
    library's native 0..1 reflectance here so the two are in the same units.
    """
    if not path.exists():
        return None
    from scipy.io import loadmat
    mat = loadmat(path)
    Y = mat["Y"].astype(float) / float(mat["maxValue"].item())
    return Y, int(mat["nRow"].item()), int(mat["nCol"].item())


def load_urban_library(path):
    """Load the bundled 6-class Urban reference library; None if absent.

    Returns M (bands, N_MATERIALS), each column the mean spectrum of that
    material's candidate signatures (lib1..lib6), matched to MATERIAL_NAMES
    by the file's own `material_names` (checked, not assumed).
    """
    if not path.exists():
        return None
    from scipy.io import loadmat
    mat = loadmat(path)
    names = [str(mat["material_names"][0, i][0]) for i in range(mat["material_names"].shape[1])]
    if names != MATERIAL_NAMES:
        raise ValueError(f"library material order {names} != expected {MATERIAL_NAMES}")
    return np.column_stack([mat[f"lib{i + 1}"].mean(axis=1) for i in range(N_MATERIALS)])


def synthetic_urban(seed=0):
    """A synthetic (scene, library) pair of the same shape, for offline runs.

    Endmember spectra are smoothed random curves (same style as the other two
    scripts' synthetic fallbacks); abundances are a softmax of Gaussian-blurred
    random fields, so neighbouring pixels share a dominant material as in a
    real urban scene (buildings and roads are spatially contiguous).
    """
    rng = np.random.default_rng(seed)
    band = np.arange(N_BANDS)
    spectra = np.zeros((N_MATERIALS, N_BANDS))
    for k in range(N_MATERIALS):
        base = 0.1 + 0.4 * rng.random()
        curve = base * np.ones(N_BANDS)
        for _ in range(rng.integers(2, 5)):
            centre, width, depth = rng.uniform(0, N_BANDS), rng.uniform(8, 20), rng.uniform(0.05, 0.2)
            curve -= depth * np.exp(-0.5 * ((band - centre) / width) ** 2)
        spectra[k] = np.clip(curve, 0.01, None)

    def blurred_field():
        coarse = rng.standard_normal((N_ROWS, N_COLS))
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        k /= k.sum()
        field = coarse
        for _ in range(6):
            field = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, field)
            field = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, field)
        return field

    fields = np.stack([blurred_field() for _ in range(N_MATERIALS)], axis=-1)
    abundances = np.exp(30.0 * fields)
    abundances /= abundances.sum(axis=-1, keepdims=True)
    cube = abundances @ spectra  # rows, cols, bands
    Y = cube.reshape(-1, N_BANDS).T
    return Y, N_ROWS, N_COLS, spectra.T  # M is (bands, N_MATERIALS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATA / "Urban_R162.mat"))
    ap.add_argument("--library", default=str(DATA / "spectral_library_urban.mat"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--clarabel-sample", type=int, default=1500,
                     help="pixels sampled for the Clarabel/FISTA per-pixel timing baselines")
    ap.add_argument("--fista-tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if SMOKE:
        args.clarabel_sample = min(args.clarabel_sample, 80)

    scene = None if args.synthetic else load_urban(Path(args.data))
    M = None if args.synthetic else load_urban_library(Path(args.library))
    if scene is None or M is None:
        print(f"No local Urban data at {args.data} / {args.library} (or --synthetic requested); "
              "using the offline synthetic proxy scene. See the module docstring for how to "
              "point --data/--library at the real files.")
        Y, rows, cols, M = synthetic_urban(seed=args.seed)
    else:
        Y, rows, cols = scene
    n = M.shape[1]
    N = Y.shape[1]
    print(f"Scene: {rows}x{cols} pixels, {M.shape[0]} bands, N = {N} spectra, {n} materials")

    A = M.T @ M
    A_op = DenseOperator(A)  # n=6 << bands, generically full rank -- no ridge needed
    Bmat = np.ones((1, n))
    c = np.array([1.0])
    Bpix = M.T @ Y

    order = snake_order(rows, cols)

    print("Unmixing (cold start: every pixel from scratch)...")
    X_cold, outer_cold, inner_cold, t_cold = unmix_image(
        Bpix, A_op, Bmat, c, order, warm_start=False, label="cold")
    print(f"  {t_cold:.2f}s, total outer iters = {outer_cold.sum()}")

    print("Unmixing (warm start: free set carried along the raster scan)...")
    X_warm, outer_warm, inner_warm, t_warm = unmix_image(
        Bpix, A_op, Bmat, c, order, warm_start=True, label="warm")
    print(f"  {t_warm:.2f}s, total outer iters = {outer_warm.sum()}")

    agree = float(np.max(np.abs(X_cold - X_warm)))
    print(f"Cold vs warm start agree to {agree:.1e} (same active-set fixed points)")

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(N, size=min(args.clarabel_sample, N), replace=False)

    t0 = time.perf_counter()
    for i in sample:
        run_clarabel_fcls(A, Bpix[:, i], n)
    t_clarabel_sample = time.perf_counter() - t0
    t_clarabel_full = t_clarabel_sample * N / sample.size
    print(f"Clarabel on {sample.size} sampled pixels: {t_clarabel_sample:.2f}s "
          f"(extrapolated full scene: {t_clarabel_full:.1f}s)")

    L = float(np.linalg.eigvalsh(A)[-1])
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
    # Figure: dominant-material map (named, discrete colormap + legend) and a
    # clipped outer-iteration histogram (outliers reported, not hidden -- see
    # the n=14 Cuprite script, where a few pathological pixels once made this
    # panel unreadable by stretching the axis).
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    dominant = X_warm.argmax(axis=0).reshape(rows, cols)
    cmap = plt.get_cmap("tab10", N_MATERIALS)
    im = axes[0].imshow(dominant, cmap=cmap, vmin=-0.5, vmax=N_MATERIALS - 0.5, interpolation="nearest")
    axes[0].set_title("dominant material per pixel")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(N_MATERIALS)]
    axes[0].legend(handles, MATERIAL_NAMES, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=3, frameon=False)

    cap = max(20, int(np.percentile(np.concatenate([outer_cold, outer_warm]), 99)))
    n_over_cold = int((outer_cold > cap).sum())
    n_over_warm = int((outer_warm > cap).sum())
    bins = np.arange(0, cap + 2) - 0.5
    axes[1].hist(np.clip(outer_cold, 0, cap), bins=bins, alpha=0.6, label="cold start", color="#d62728")
    axes[1].hist(np.clip(outer_warm, 0, cap), bins=bins, alpha=0.6, label="warm start", color="#2ca02c")
    axes[1].set_xlabel(f"outer (active-set) iterations, clipped to {cap}")
    axes[1].set_ylabel("pixels")
    if n_over_cold or n_over_warm:
        axes[1].set_title(f"{n_over_cold} cold / {n_over_warm} warm pixels >{cap} (clipped)", fontsize=7)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(GRAPHS / "nncg_hyperspectral_urban.pdf")
    print(f"\nSaved {GRAPHS / 'nncg_hyperspectral_urban.pdf'}")

    # -----------------------------------------------------------------------
    # LaTeX macros
    # -----------------------------------------------------------------------
    with open(TABLES / "nncg_hyperspectral_urban_defs.tex", "w") as fh:
        fh.write("% Generated by experiment_nncg_hyperspectral_urban.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbPixels}}{{{N}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbBands}}{{{M.shape[0]}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbMaterials}}{{{n}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbOuterCold}}{{{int(outer_cold.sum())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbOuterWarm}}{{{int(outer_warm.sum())}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbFracSingle}}{{{frac_single * 100:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbSpeedupOuter}}{{{speedup_outer:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbSpeedupWall}}{{{speedup_wall:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbAgree}}{{{agree:.0e}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbSpeedupFista}}{{{speedup_fista:.1f}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbFistaIters}}{{{int(round(fista_iters.mean()))}}}\n")
        fh.write(f"\\newcommand{{\\nncgHsUrbFistaItersMax}}{{{int(fista_iters.max())}}}\n")
    print(f"Saved {TABLES / 'nncg_hyperspectral_urban_defs.tex'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
