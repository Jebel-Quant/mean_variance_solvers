r"""What a random change of variables does to the dual method, and what it does not.

Substituting ``x = T y`` for an invertible ``T`` carries ``(G, a, C, b)`` to
``(T^T G T, T^T a, T^T C, b)``, and then

    C'^T G'^-1 C' = C^T T T^-1 G^-1 T^-T T^T C = C^T G^-1 C,

so the dual Hessian -- the only matrix the method ever factorises -- is unchanged
on every working set. Section 7's proposition draws the consequences: the active
set, the multipliers and the whole walk are invariant, while
``kappa(G') ~ kappa(G) kappa(T)^2`` is free to be anything. That makes ``{T}`` a
family of problems whose answer is known at arbitrary conditioning, without a
reference solver, which is what this script uses it for.

Three questions, in the order the paper asks them:

  * does the implementation reproduce the invariance -- same active set, same
    multipliers, same iteration count -- and how far does it hold as ``kappa(G')``
    grows past what double precision can carry in ``x``;
  * where is the conditioning boundary at which the certificate starts rejecting
    the guessed set, and what does the fast path then cost, given that a rejected
    guess is paid for and thrown away;
  * what a guess that is going to be rejected should cost, given that the two
    mistakes available are not symmetric: skipping a guess that would have been
    certified forfeits the whole speedup, while attempting one that gets rejected
    forfeits only the guess.

The third is the only one that changes what the code should do, and it is the one
that came out against the obvious rule. A conditioning threshold read off the
Cholesky factor is free, but it does not separate the two populations -- see
``cheap_kappa`` -- and the asymmetry above means a wrong skip costs several times
a wrong attempt. What does work is remembering a rejection that has actually
happened, which is free, cannot be wrong about the past, and is exactly the
setting of Section 8: a family of programs sharing ``G``.

Usage:
    uv run python -m quadprog_note.experiment_qp_transform   # from experiment/

Outputs:
    graphs/quadprog_transform.pdf        accuracy, verdict and cost vs kappa(G')
    tables/quadprog_transform_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + Matplotlib + cvx-quadprog.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common.util.runner import SMOKE, output_dirs

from cvx.quadprog import solve_qp
from cvx.quadprog import _pdas

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

# kappa(T), not kappa(G'): the transform's condition number is what the
# experiment sets, and kappa(G') ~ kappa(G) kappa(T)^2 is what it produces.
# The smoke grid is three points rather than eight, but it still straddles the
# boundary: a grid that never reaches the rejecting regime would leave the
# branch this script exists to measure unexercised.
KAPPAS = [1.0, 1e4, 1e7] if SMOKE else [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
N_ACCURACY = 60 if SMOKE else 200
INSTANCES = 2 if SMOKE else 10

# Timing is a separate, smaller sweep: best-of-REPS at two conditionings only,
# one on each side of the boundary the accuracy sweep locates.
TIME_SIZES = [100] if SMOKE else [100, 200, 400]
TIME_INSTANCES = 1 if SMOKE else 3
REPS = 1 if SMOKE else 5

BASES = ("box", "dense $C$")

#: Number of linear terms in the family the verdict cache is measured on. The
#: family shares G, as an efficient frontier or a rolling rebalance does, so one
#: rejection is evidence about the rest of it.
FAMILY_TERMS = 4 if SMOKE else 20


def hessian(rng, n):
    """Return the moderately conditioned Hessian the other experiments use."""
    b_mat = rng.standard_normal((n, n))
    return b_mat @ b_mat.T + n * np.eye(n)


def base_problem(name, rng, n):
    """Return (G, a, C, b) for the base problem, before any transformation."""
    g = hessian(rng, n)
    a = rng.standard_normal(n)
    xu = np.linalg.solve(g, a)

    if name == "box":
        shift = rng.standard_normal(n) * 0.5
        lo = xu - np.abs(rng.standard_normal(n)) + shift
        hi = xu + np.abs(rng.standard_normal(n)) + shift
        return g, a, np.hstack([np.eye(n), -np.eye(n)]), np.concatenate([lo, -hi])

    if name == "dense $C$":
        m = max(2, n // 2)
        c = rng.standard_normal((n, m))
        return g, a, c, c.T @ xu + rng.standard_normal(m) * 0.5

    raise ValueError(name)


def transform(rng, n, kappa):
    """Return an invertible ``T`` with condition number ``kappa``.

    Two independent random orthogonal factors around a graded diagonal, so that
    no eigenvector of ``G`` is aligned with a singular direction of ``T`` and the
    transformed problem retains no trace of the coordinate system it came from.
    """
    if kappa == 1.0:
        return np.linalg.qr(rng.standard_normal((n, n)))[0]
    q1 = np.linalg.qr(rng.standard_normal((n, n)))[0]
    q2 = np.linalg.qr(rng.standard_normal((n, n)))[0]
    return q1 @ np.diag(np.logspace(0, np.log10(kappa), n)) @ q2


def apply_transform(t, g, a, c):
    """Return the transformed data of the substitution ``x = T y`` (``b`` is fixed)."""
    return t.T @ g @ t, t.T @ a, t.T @ c


def cheap_kappa(g):
    """Estimate ``kappa(G)`` from the Cholesky factor the solver computes anyway.

    ``max_i L_ii^2 / min_i L_ii^2`` costs nothing beyond a factorisation already
    performed, and it tracks the conditioning well enough to order the cells of
    the sweep. It is not, however, a classifier: the experiment reports its value
    at the worst accepted guess and at the best rejected one, and they come out
    the wrong way round. That is what rules out a static threshold, and it is
    measured here rather than assumed either way.
    """
    d = np.diag(np.linalg.cholesky(g)) ** 2
    return float(d.max() / d.min())


def kkt_residual(g, a, c, b, x, lagrangian) -> float:
    """Return the scaled sup-norm KKT residual, as the certificate measures it."""
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    slack = c.T @ x - b
    lam = np.asarray(lagrangian)
    return max(
        float(np.max(np.abs(g @ x - a - c @ lam))),
        max(0.0, -float(np.min(slack))),
        max(0.0, -float(np.min(lam))),
        float(np.max(np.abs(lam * slack))),
    ) / scale


class Verdicts:
    """Collect the certificate's verdicts from the shipped fast path."""

    def __init__(self) -> None:
        self.seen: list[bool] = []
        self._orig = None

    def __enter__(self):
        self._orig = _pdas._certified

        def certified(*args, **kwargs):
            verdict = self._orig(*args, **kwargs)
            self.seen.append(bool(verdict))
            return verdict

        _pdas._certified = certified
        return self

    def __exit__(self, *exc):
        _pdas._certified = self._orig
        return False


def accuracy_sweep():
    """Return one record per (base, kappa, instance) of the transformed solve."""
    rows = []
    for base in BASES:
        for kappa in KAPPAS:
            for i in range(INSTANCES):
                # Seeded arithmetically from the cell, as the other scripts are,
                # so the table reproduces across runs and machines.
                seed = (BASES.index(base) * 1_000_000) + int(np.log10(kappa) * 1_000) + i
                rng = np.random.default_rng(seed)
                g, a, c, b = base_problem(base, rng, N_ACCURACY)
                ref = solve_qp(g, a, c, b)

                t = transform(rng, N_ACCURACY, kappa)
                gt, at, ct = apply_transform(t, g, a, c)
                walk = solve_qp(gt, at, ct, b)
                with Verdicts() as verdicts:
                    solve_qp(gt, at, ct, b, fast=True)

                lam_ref = np.asarray(ref.lagrangian)
                rows.append({
                    "base": base,
                    "kappa_t": kappa,
                    "kappa_g": float(np.linalg.cond(gt)),
                    "est": cheap_kappa(gt),
                    "set_agrees": set(walk.iact.tolist()) == set(ref.iact.tolist()),
                    "iters_agree": int(walk.iterations[0]) == int(ref.iterations[0]),
                    "err_x": float(np.linalg.norm(t @ walk.x - ref.x)
                                   / max(1.0, np.linalg.norm(ref.x))),
                    "err_lam": float(np.linalg.norm(np.asarray(walk.lagrangian) - lam_ref)
                                     / max(1.0, np.linalg.norm(lam_ref))),
                    "resid": kkt_residual(gt, at, ct, b, walk.x, walk.lagrangian),
                    "certified": all(verdicts.seen) and bool(verdicts.seen),
                })
            last = rows[-1]
            print(f"{base:<12} kappa(T)={kappa:7.0e} kappa(G')={last['kappa_g']:8.2e} "
                  f"est={last['est']:8.2e} set={last['set_agrees']!s:<5} "
                  f"err={last['err_x']:.2e} certified={last['certified']}")
    return rows


def timed(fn):
    """Return (result, best wall time in ms) over REPS calls."""
    best, out = np.inf, None
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best * 1e3


def timing_sweep(kappa_good, kappa_bad):
    """Return per-(size, regime) timings of the walk, the fast path and the rule.

    The third column is what the skip rule would cost: the free estimate decides,
    and above the threshold the walk runs directly rather than after a guess that
    is going to be rejected.
    """
    cells = {}
    for n in TIME_SIZES:
        for label, kappa in (("well conditioned", kappa_good), ("ill conditioned", kappa_bad)):
            walk_ms, fast_ms, rule_ms = [], [], []
            for i in range(TIME_INSTANCES):
                rng = np.random.default_rng(500_000 + n * 100 + i)
                g, a, c, b = base_problem("box", rng, n)
                t = transform(rng, n, kappa)
                gt, at, ct = apply_transform(t, g, a, c)

                _, tw = timed(lambda: solve_qp(gt, at, ct, b))
                _, tf = timed(lambda: solve_qp(gt, at, ct, b, fast=True))
                # The rule pays for the estimate either way; it is one Cholesky
                # the solver performs regardless, so it is charged nothing extra.
                skipped = cheap_kappa(gt) > SKIP_ABOVE
                walk_ms.append(tw)
                fast_ms.append(tf)
                rule_ms.append(tw if skipped else tf)
            cells[(n, label)] = {
                "walk": float(np.median(walk_ms)),
                "fast": float(np.median(fast_ms)),
                "rule": float(np.median(rule_ms)),
            }
            c_ = cells[(n, label)]
            print(f"n={n:<5} {label:<17} walk={c_['walk']:7.3f} ms  "
                  f"fast={c_['fast']:7.3f} ms  rule={c_['rule']:7.3f} ms")
    return cells


def rejection_rates(rows):
    """Return {kappa_t: pooled fraction of guesses the certificate rejected}."""
    return {k: 1.0 - np.mean([r["certified"] for r in rows if r["kappa_t"] == k])
            for k in KAPPAS}


def band(rows):
    """Return the boundary as the band it is, not as a point.

    Three numbers: the largest kappa(G') at which a guess was still certified,
    the smallest at which one was rejected, and the smallest at which the
    majority were. On this evidence the first two come out in that order, so the
    two populations overlap by about two decades and the bases disagree inside
    the overlap. That is the finding, and the reason the cost argument is about
    expected cost rather than about locating a threshold.
    """
    good = [r["kappa_g"] for r in rows if r["certified"]]
    bad = [r["kappa_g"] for r in rows if not r["certified"]]
    rates = rejection_rates(rows)
    majority = [k for k in KAPPAS if rates[k] >= 0.5]
    kappa_majority = min(majority) if majority else float("nan")
    g_majority = np.median([r["kappa_g"] for r in rows
                            if r["kappa_t"] == kappa_majority]) if majority else float("nan")
    est_good = max((r["est"] for r in rows if r["certified"]), default=float("nan"))
    est_bad = min((r["est"] for r in rows if not r["certified"]), default=float("nan"))
    return {
        "last_certified": max(good, default=float("nan")),
        "first_rejected": min(bad, default=float("nan")),
        "kappa_majority": kappa_majority,
        "g_majority": float(g_majority),
        "est_good": est_good,
        "est_bad": est_bad,
    }


def timed(fn):
    """Return (result, best wall time in ms) over REPS calls."""
    best, out = np.inf, None
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best * 1e3


def timing_sweep(kappa_good, kappa_bad):
    """Return per-(size, regime) timings of the exact walk and the fast path.

    Two conditionings, one on each side of the band, so that the two mistakes a
    skip rule can make can be priced against each other: forfeiting a guess that
    would have been certified costs the well-conditioned gap, and attempting one
    that gets rejected costs the ill-conditioned one.
    """
    cells = {}
    for n in TIME_SIZES:
        for label, kappa in (("well conditioned", kappa_good),
                             ("ill conditioned", kappa_bad)):
            walk_ms, fast_ms = [], []
            for i in range(TIME_INSTANCES):
                rng = np.random.default_rng(500_000 + n * 100 + i)
                g, a, c, b = base_problem("box", rng, n)
                t = transform(rng, n, kappa)
                gt, at, ct = apply_transform(t, g, a, c)
                _, tw = timed(lambda: solve_qp(gt, at, ct, b))
                _, tf = timed(lambda: solve_qp(gt, at, ct, b, fast=True))
                walk_ms.append(tw)
                fast_ms.append(tf)
            cells[(n, label)] = {"walk": float(np.median(walk_ms)),
                                 "fast": float(np.median(fast_ms))}
            c_ = cells[(n, label)]
            print(f"n={n:<5} {label:<17} walk={c_['walk']:7.3f} ms  "
                  f"fast={c_['fast']:7.3f} ms")
    return cells


def family_sweep(kappa_bad):
    """Price the verdict cache on a family of programs sharing an ill-conditioned G.

    G, C and b are fixed and only the linear term varies, which is Section 8's
    setting. Three policies: guess on every member; guess until the certificate
    rejects once and walk thereafter; never guess. The cache is worth having only
    if a rejection on one member predicts rejection on the rest, so the verdicts
    are recorded as well as the times.
    """
    n = TIME_SIZES[-1]
    rng = np.random.default_rng(909_090)
    g, a, c, b = base_problem("box", rng, n)
    t = transform(rng, n, kappa_bad)
    gt, _, ct = apply_transform(t, g, a, c)
    terms = [t.T @ rng.standard_normal(n) for _ in range(FAMILY_TERMS)]

    verdicts = []
    for term in terms:
        with Verdicts() as v:
            solve_qp(gt, term, ct, b, fast=True)
        verdicts.append(bool(v.seen) and all(v.seen))

    always = sum(timed(lambda: solve_qp(gt, term, ct, b, fast=True))[1] for term in terms)
    never = sum(timed(lambda: solve_qp(gt, term, ct, b))[1] for term in terms)

    # The cache guesses until a rejection, then stops for the rest of the family.
    cached, stopped = 0.0, False
    for term, certified in zip(terms, verdicts):
        if stopped:
            cached += timed(lambda: solve_qp(gt, term, ct, b))[1]
            continue
        cached += timed(lambda: solve_qp(gt, term, ct, b, fast=True))[1]
        stopped = not certified

    rejected = sum(1 for v in verdicts if not v)
    print(f"\nfamily of {FAMILY_TERMS} linear terms at n={n}, kappa(T)={kappa_bad:.0e}: "
          f"{rejected}/{FAMILY_TERMS} guesses rejected")
    print(f"  always guess {always:8.1f} ms   verdict cache {cached:8.1f} ms   "
          f"never guess {never:8.1f} ms")
    return {"n": n, "always": always, "cached": cached, "never": never,
            "rejected": rejected, "terms": FAMILY_TERMS}


def figure(rows, cells, fam, kappa_bad) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    style = {"box": ("#1f77b4", "o", "-"), "dense $C$": ("#2ca02c", "^", "-.")}
    for base, (colour, marker, dash) in style.items():
        sel = [r for r in rows if r["base"] == base]
        xs = sorted({r["kappa_g"] for r in sel})
        med = [np.median([r["err_x"] for r in sel if r["kappa_g"] == x]) for x in xs]
        ax_a.plot(xs, med, color=colour, marker=marker, linestyle=dash,
                  markersize=4, label=f"{base}: error in $x$")

    # The error a backward-stable solve is entitled to, for reference.
    xs_all = sorted({r["kappa_g"] for r in rows})
    ax_a.plot(xs_all, [x * np.finfo(float).eps for x in xs_all],
              color="0.4", linestyle=":", linewidth=1, label=r"$\kappa(G')\,\varepsilon$")

    # The transition is a band; shade it rather than drawing a line through it.
    rates = rejection_rates(rows)
    mixed = [k for k in KAPPAS if 0.0 < rates[k] < 1.0]
    if mixed:
        lo = np.median([r["kappa_g"] for r in rows if r["kappa_t"] == min(mixed)])
        hi = np.median([r["kappa_g"] for r in rows if r["kappa_t"] == max(mixed)])
        ax_a.axvspan(lo, hi, color="#d62728", alpha=0.10,
                     label="certificate rejects some")

    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"$\kappa(G')$")
    ax_a.set_ylabel("relative error in $x$")
    ax_a.set_title("Same problem, worse coordinates")
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax_a.legend(framealpha=0.9, loc="upper left")

    labels = ["always\nguess", "verdict\ncache", "never\nguess"]
    values = [fam["always"], fam["cached"], fam["never"]]
    ax_b.bar(labels, values, color=["#d62728", "#2ca02c", "#1f77b4"], width=0.6)
    for i, v in enumerate(values):
        ax_b.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax_b.set_ylabel(f"ms for {fam['terms']} solves")
    ax_b.set_ylim(0, max(values) * 1.18)
    ax_b.set_title(rf"A family sharing an ill-conditioned $G$ "
                   rf"($n = {fam['n']}$, $\kappa(T) = 10^{{{int(np.log10(kappa_bad))}}}$)")
    ax_b.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout(pad=0.8)
    fig.savefig(GRAPHS / "quadprog_transform.pdf", bbox_inches="tight")
    fig.savefig(GRAPHS / "quadprog_transform.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'quadprog_transform.pdf'}")


def sci(x) -> str:
    """Format a positive value as bare LaTeX math ``m\times10^{e}``.

    A grid that never crosses the boundary leaves the rejecting side undefined;
    that prints as ``n/a`` rather than aborting the run, so a reduced sweep still
    produces a file, and the absence is visible in it.
    """
    if x == 0.0:
        return "0"
    if not np.isfinite(x):
        return "\\mathrm{n/a}"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10.0**exponent
    if round(mantissa, 1) >= 10.0:  # 9.99e-16 must print as 1.0e-15
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"{mantissa:.1f}\\times 10^{{{exponent}}}"


def emit(rows, cells, fam, bnd) -> None:
    n_big = TIME_SIZES[-1]
    ill = cells[(n_big, "ill conditioned")]
    well = cells[(n_big, "well conditioned")]
    worst = max(rows, key=lambda r: r["kappa_g"])
    certified_rows = [r for r in rows if r["certified"]]

    # The two mistakes a skip rule can make, priced at the largest size measured.
    false_skip = well["walk"] - well["fast"]
    missed_skip = ill["fast"] - ill["walk"]

    path = TABLES / "quadprog_transform_defs.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by quadprog_note/experiment_qp_transform.py"
                 " -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\qpTfN}}{{{N_ACCURACY}}}\n")
        fh.write(f"\\newcommand{{\\qpTfInstances}}{{{INSTANCES}}}\n")
        fh.write(f"\\newcommand{{\\qpTfSolves}}{{{len(rows)}}}\n")
        fh.write(f"\\newcommand{{\\qpTfKappaHi}}{{{sci(worst['kappa_g'])}}}\n")
        fh.write(f"\\newcommand{{\\qpTfKappaTHi}}{{10^{{{int(np.log10(KAPPAS[-1]))}}}}}\n")
        # The rejected fraction across the grid, in order: the transition read as
        # a sequence rather than as a threshold.
        rates = rejection_rates(rows)
        fh.write("\\newcommand{\\qpTfRates}{"
                 + ", ".join(f"{100 * rates[k]:.0f}" for k in KAPPAS) + "}\n")
        # The invariance, as counts: nothing here is a tolerance.
        fh.write(f"\\newcommand{{\\qpTfSetAgree}}{{"
                 f"{100 * np.mean([r['set_agrees'] for r in rows]):.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfIterAgree}}{{"
                 f"{100 * np.mean([r['iters_agree'] for r in rows]):.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfErrHi}}{{{sci(worst['err_x'])}}}\n")
        fh.write(f"\\newcommand{{\\qpTfLamErrHi}}{{{sci(worst['err_lam'])}}}\n")
        # The band, and the free estimate's failure to resolve it.
        fh.write(f"\\newcommand{{\\qpTfLastGood}}{{{sci(bnd['last_certified'])}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFirstBad}}{{{sci(bnd['first_rejected'])}}}\n")
        fh.write(f"\\newcommand{{\\qpTfMajority}}{{{sci(bnd['g_majority'])}}}\n")
        fh.write(f"\\newcommand{{\\qpTfEstGood}}{{{sci(bnd['est_good'])}}}\n")
        fh.write(f"\\newcommand{{\\qpTfEstBad}}{{{sci(bnd['est_bad'])}}}\n")
        if certified_rows:
            fh.write(f"\\newcommand{{\\qpTfAcceptWorst}}{{"
                     f"{sci(max(r['resid'] for r in certified_rows))}}}\n")
        # What a guess costs on each side of the band.
        fh.write(f"\\newcommand{{\\qpTfTimeN}}{{{n_big}}}\n")
        fh.write(f"\\newcommand{{\\qpTfWalkIll}}{{{ill['walk']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFastIll}}{{{ill['fast']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfWalkWell}}{{{well['walk']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFastWell}}{{{well['fast']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfWellSpeedup}}{{"
                 f"{well['walk'] / max(well['fast'], 1e-12):.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfIllPenalty}}{{"
                 f"{100 * (ill['fast'] / max(ill['walk'], 1e-12) - 1):.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFalseSkipCost}}{{{false_skip:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfMissedSkipCost}}{{{missed_skip:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfCostRatio}}{{"
                 f"{false_skip / max(missed_skip, 1e-12):.1f}}}\n")
        # The verdict cache, on a family sharing G.
        fh.write(f"\\newcommand{{\\qpTfFamilyTerms}}{{{fam['terms']}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFamilyRejected}}{{{fam['rejected']}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFamilyAlways}}{{{fam['always']:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFamilyCached}}{{{fam['cached']:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFamilyNever}}{{{fam['never']:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpTfFamilySaving}}{{"
                 f"{100 * (1 - fam['cached'] / max(fam['always'], 1e-12)):.0f}}}\n")
    print(f"Saved {path}")


def main() -> None:
    rows = accuracy_sweep()
    bnd = band(rows)
    rates = rejection_rates(rows)
    print("\nrejection rate by kappa(T): "
          + "  ".join(f"{k:.0e}:{100 * rates[k]:.0f}%" for k in KAPPAS))
    print(f"last certified guess at kappa(G') = {bnd['last_certified']:.2e}; "
          f"first rejection at {bnd['first_rejected']:.2e}; "
          f"majority rejected from {bnd['g_majority']:.2e}")
    print(f"free estimate: {bnd['est_good']:.2e} at the worst accepted guess, "
          f"{bnd['est_bad']:.2e} at the best rejected one")

    # Time one conditioning on each side of the band: the mildest one at which
    # the certificate rejects the majority of guesses is the informative one.
    kappa_bad = bnd["kappa_majority"] if np.isfinite(bnd["kappa_majority"]) else KAPPAS[-1]
    cells = timing_sweep(KAPPAS[0], kappa_bad)
    fam = family_sweep(kappa_bad)

    figure(rows, cells, fam, kappa_bad)
    emit(rows, cells, fam, bnd)
    print("\nDone.")


if __name__ == "__main__":
    main()
