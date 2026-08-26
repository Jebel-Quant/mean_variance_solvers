r"""The solver on a real covariance, which is not the matrix the other tests use.

Every other experiment here draws its Hessian as ``G = BB^T + nI`` for Gaussian
``B``. That is positive definite by construction and moderately conditioned, and
it is the wrong shape in a way that matters: a well-conditioned Hessian has a
minimiser that spreads across coordinates, so few bounds bind and the active set
stays small. A sample covariance of asset returns is close to low rank plus
diagonal, badly conditioned, and its long-only minimum-variance optimum
concentrates into a handful of names -- which means nearly every bound binds and
the active set is nearly the whole constraint set.

So the synthetic families understate the very quantity the paper's argument turns
on. This script measures the same things on real data:

  * the conditioning, and how far it is from the synthetic case;
  * the size of the active set, and the exact walk's iteration count, which grows
    with it;
  * whether the guessed active set still certifies when the system it induces is
    badly conditioned;
  * whether the KKT residual survives that conditioning;
  * a frontier sweep, which is the warm-start machinery's actual use case: a
    family of programs differing only in the linear term.

Two datasets, so that the numbers are not a property of one market: daily returns
for the S&P 500 (n = 494) and the FTSE 100 (n = 87).

Usage:
    uv run python -m quadprog_note.experiment_qp_portfolio   # from experiment/

Inputs:
    data/sp500_pct_returns.parquet
    data/ftse100_pct_returns.parquet

Outputs:
    graphs/quadprog_portfolio.pdf        frontier, and per-point cost warm vs cold
    tables/quadprog_portfolio.tex        one row per dataset per solver
    tables/quadprog_portfolio_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + pandas + Matplotlib + cvx-quadprog.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.util.runner import SMOKE, output_dirs

from cvx.quadprog import Sweep, solve_qp
from cvx.quadprog import _pdas

from quadprog_note.experiment_qp_compare import kkt_residual, run_clarabel, run_osqp

HERE = Path(__file__).resolve().parents[1]
GRAPHS, TABLES = output_dirs(HERE)

mpl.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})

DATASETS = [
    ("S\\&P~500", "data/sp500_pct_returns.parquet"),
    ("FTSE~100", "data/ftse100_pct_returns.parquet"),
]
FRONTIER_POINTS = 5 if SMOKE else 50
REPEATS = 1 if SMOKE else 3

try:
    import quadprog as quadprog_c
except ImportError:  # pragma: no cover - depends on the host
    quadprog_c = None


def load(path, cap):
    """Return (Sigma, mu, shape, dropped) from daily percentage returns, annualised.

    Two filters, and the second is not cosmetic. Columns with missing data go
    first. Then columns of zero sample variance: the FTSE series carries one, an
    instrument quoted flat across the whole window, and a single such column makes
    the sample covariance singular -- rank 86 of 87 -- which a strictly convex
    program does not admit. The solver is right to refuse it, and does. Dropping
    the degenerate asset is the modelling fix rather than a workaround, since an
    asset with no variance has no place in a variance-minimising portfolio.
    """
    frame = pd.read_parquet(HERE / path).dropna(axis=1, how="any")
    returns_all = frame.to_numpy(dtype=float) / 100.0
    keep = returns_all.std(axis=0) > 0.0
    dropped = int((~keep).sum())
    frame = frame.loc[:, keep]
    if cap and frame.shape[1] > cap:
        frame = frame.iloc[:, :cap]
    returns = frame.to_numpy(dtype=float) / 100.0
    sigma = np.cov(returns, rowvar=False) * 252.0
    mu = returns.mean(axis=0) * 252.0
    return sigma, mu, frame.shape, dropped


def long_only(sigma, mu, rho):
    """Return (G, a, C, b, meq) for min 1/2 w'Sigma w - rho mu'w, 1'w = 1, w >= 0."""
    n = sigma.shape[0]
    c = np.hstack([np.ones((n, 1)), np.eye(n)])
    b = np.concatenate([[1.0], np.zeros(n)])
    return sigma, rho * mu, c, b, 1


def best_of(fn, repeats):
    best, out = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


class RepairCounter:
    """Count the fast path's set repairs without reimplementing its loop."""

    def __enter__(self):
        self.count = 0
        self._orig = _pdas._repair

        def repair(*args, **kwargs):
            self.count += 1
            return self._orig(*args, **kwargs)

        _pdas._repair = repair
        return self

    def __exit__(self, *exc):
        _pdas._repair = self._orig
        return False


def analyse(label, path):
    """Measure one dataset at the minimum-variance corner of the frontier."""
    cap = 60 if SMOKE else None
    sigma, mu, shape, dropped = load(path, cap)
    n = sigma.shape[0]
    eigs = np.linalg.eigvalsh(sigma)
    kappa = float(eigs[-1] / eigs[0])

    g, a, c, b, meq = long_only(sigma, mu, 0.0)  # minimum variance: rho = 0

    t_exact, sol = best_of(lambda: solve_qp(g, a, c, b, meq), REPEATS)
    active = len(sol.iact)
    outer = int(sol.iterations[0])
    resid_exact = kkt_residual(g, a, c, b, meq, sol.x)
    nonzero = int(np.sum(sol.x > 1e-10))

    t_fast, sol_fast = best_of(lambda: solve_qp(g, a, c, b, meq, fast=True), REPEATS)
    with RepairCounter() as counter:
        attempt = _pdas.attempt(g, a, c, b, meq)
    certified = attempt is not None
    resid_fast = kkt_residual(g, a, c, b, meq, sol_fast.x)

    rows = {
        "cvx-quadprog": (t_exact, resid_exact),
        "cvx-quadprog, \\texttt{fast}": (t_fast, resid_fast),
    }
    if quadprog_c is not None:
        t_ref, out = best_of(
            lambda: quadprog_c.solve_qp(g.copy(), a.copy(), c, b, meq)[0], REPEATS)
        rows["\\texttt{quadprog} (C)"] = (t_ref, kkt_residual(g, a, c, b, meq, out))
    for name, fn in (("OSQP", run_osqp), ("Clarabel", run_clarabel)):
        try:
            seconds, out = best_of(lambda: fn(g, a, c, b, meq), REPEATS)
            rows[name] = (seconds, kkt_residual(g, a, c, b, meq, out))
        except Exception as exc:
            print(f"    {name} failed: {type(exc).__name__}: {exc}")
            rows[name] = (float("nan"), float("nan"))

    print(f"{label}: n = {n}, T = {shape[0]}, kappa = {kappa:.2e}"
          + (f", dropped {dropped} zero-variance" if dropped else ""))
    print(f"  active {active}/{n + 1}, nonzero weights {nonzero}, "
          f"exact outer {outer}, fast repairs {counter.count}, certified {certified}")
    for name, (seconds, resid) in rows.items():
        print(f"  {name:<30} {seconds * 1e3:>9.1f} ms  resid {resid:.2e}")

    return {
        "label": label, "n": n, "T": shape[0], "kappa": kappa, "active": active,
        "nonzero": nonzero, "outer": outer, "repairs": counter.count,
        "certified": certified, "rows": rows, "sigma": sigma, "mu": mu,
        "dropped": dropped,
    }


def frontier(sigma, mu, points):
    """Sweep the risk-aversion weight, warm and cold, and return the frontier."""
    n = sigma.shape[0]
    c = np.hstack([np.ones((n, 1)), np.eye(n)])
    b = np.concatenate([[1.0], np.zeros(n)])
    scale = float(np.max(np.abs(mu))) or 1.0
    rhos = np.linspace(0.0, 2.0 / scale, points)

    sweep = Sweep(sigma, c, b, meq=1)
    start = time.perf_counter()
    warm = [sweep.solve(rho * mu) for rho in rhos]
    t_warm = (time.perf_counter() - start) / points

    start = time.perf_counter()
    for rho in rhos:
        solve_qp(sigma, rho * mu, c, b, 1)
    t_cold = (time.perf_counter() - start) / points

    vol = [float(np.sqrt(s.x @ sigma @ s.x)) for s in warm]
    ret = [float(mu @ s.x) for s in warm]
    names = [int(np.sum(s.x > 1e-10)) for s in warm]
    return {"rhos": rhos, "vol": vol, "ret": ret, "names": names,
            "t_warm": t_warm, "t_cold": t_cold,
            "hits": sweep.hits, "misses": sweep.misses}


def figure(results, front) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax_a.plot(np.array(front["vol"]) * 100, np.array(front["ret"]) * 100,
              color="#1f77b4", marker="o", markersize=3, linewidth=1)
    ax_a.set_xlabel("annualised volatility (\\%)")
    ax_a.set_ylabel("annualised return (\\%)")
    ax_a.set_title(f"Long-only frontier, {results[0]['label'].replace(chr(92) + '&', '&')}")
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    ax_b.plot(np.array(front["vol"]) * 100, front["names"],
              color="#d62728", marker="s", markersize=3, linewidth=1)
    ax_b.set_xlabel("annualised volatility (\\%)")
    ax_b.set_ylabel("names held")
    ax_b.set_title("Holdings along the frontier")
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout(pad=0.8)
    fig.savefig(GRAPHS / "quadprog_portfolio.pdf", bbox_inches="tight")
    fig.savefig(GRAPHS / "quadprog_portfolio.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'quadprog_portfolio.pdf'}")


def sci(x) -> str:
    if not np.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10.0**exponent
    if round(mantissa) >= 10:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"{mantissa:.0f}\\times 10^{{{exponent}}}"


def emit(results, front) -> None:
    lines = []
    for res in results:
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{res['label']}}}, "
                     f"$n = {res['n']}$, $T = {res['T']}$, "
                     f"$\\kappa(\\Sigma) = {sci(res['kappa'])}$, "
                     f"{res['active']} active}} \\\\\n")
        for name, (seconds, resid) in res["rows"].items():
            ms = f"{seconds * 1e3:.1f}" if np.isfinite(seconds) else "--"
            lines.append(f"\\quad {name} & {ms} & ${sci(resid)}$ \\\\\n")
        lines.append("\\addlinespace\n")

    path = TABLES / "quadprog_portfolio.tex"
    path.write_text(
        "% Generated by quadprog_note/experiment_qp_portfolio.py -- do not edit by hand.\n"
        f"\\def\\quadprogPortfolioRows{{%\n{''.join(lines)}}}\n")
    print(f"Saved {path}")

    big = results[0]
    path = TABLES / "quadprog_portfolio_defs.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by quadprog_note/experiment_qp_portfolio.py -- do not edit by hand.\n")
        for res, tag in zip(results, ("Sp", "Ftse")):
            fh.write(f"\\newcommand{{\\qpPf{tag}N}}{{{res['n']}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}T}}{{{res['T']}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}Kappa}}{{{sci(res['kappa'])}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}Active}}{{{res['active']}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}Names}}{{{res['nonzero']}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}Outer}}{{{res['outer']}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}Repairs}}{{{res['repairs']}}}\n")
            fh.write(f"\\newcommand{{\\qpPf{tag}Resid}}"
                     f"{{{sci(res['rows']['cvx-quadprog'][1])}}}\n")
        speed = big["rows"]["cvx-quadprog"][0] / big["rows"]["cvx-quadprog, \\texttt{fast}"][0]
        fh.write(f"\\newcommand{{\\qpPfFastSpeed}}{{{speed:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpPfPoints}}{{{FRONTIER_POINTS}}}\n")
        fh.write(f"\\newcommand{{\\qpPfWarm}}{{{front['t_warm'] * 1e3:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpPfCold}}{{{front['t_cold'] * 1e3:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpPfSweepSpeed}}"
                 f"{{{front['t_cold'] / front['t_warm']:.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpPfHits}}{{{front['hits']}}}\n")
        fh.write(f"\\newcommand{{\\qpPfMisses}}{{{front['misses']}}}\n")
        fh.write(f"\\newcommand{{\\qpPfNamesLo}}{{{min(front['names'])}}}\n")
        fh.write(f"\\newcommand{{\\qpPfNamesHi}}{{{max(front['names'])}}}\n")
    print(f"Saved {path}")


def main() -> None:
    results = [analyse(label, path) for label, path in DATASETS]
    front = frontier(results[0]["sigma"], results[0]["mu"], FRONTIER_POINTS)
    print(f"\nfrontier: {FRONTIER_POINTS} points, warm {front['t_warm'] * 1e3:.1f} ms/pt, "
          f"cold {front['t_cold'] * 1e3:.1f} ms/pt, "
          f"hits {front['hits']}, misses {front['misses']}, "
          f"names {min(front['names'])}-{max(front['names'])}")
    figure(results, front)
    emit(results, front)
    print("\nDone.")


if __name__ == "__main__":
    main()
