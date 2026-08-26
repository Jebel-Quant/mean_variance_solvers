r"""How well the guessed active set does, and how the certificate separates.

Section 7 of the quadprog note argues that a primal--dual active set has no
finite-termination theorem for general constraints ``C^T x >= b``, because the
working-set system ``C_A^T G^-1 C_A`` is a principal submatrix of ``G^-1`` only
when the constraints are bounds. Losing that costs the P-matrix property, and with
it the guarantee that every principal pivot is defined. The section then argues
the method is nonetheless sound, because strict convexity makes the KKT conditions
sufficient, so every candidate can be certified before it is returned.

Both halves are claims about behaviour, and neither is provable. This script
measures them, on the shipped implementation and by instrumenting the shipped
loop rather than reimplementing it: ``_repair``, ``_working_set_solve`` and
``_certified`` are wrapped with counters, so what is counted is what runs.

Four questions:

  * how often the guess is certifiable, by problem family and size;
  * how many set repairs it takes, and whether that grows with ``n``;
  * how often a guessed set is rank deficient -- the failure the P-matrix
    property rules out for bounds and does not rule out for general ``C``;
  * how far apart the certificate's two populations sit: the KKT residual of the
    candidates it accepts against that of the candidates it rejects.

The last of these is the one that decides whether the certificate's tolerance is
delicate or not. If the populations are separated by orders of magnitude, any
threshold between them works and nothing is tuned.

Usage:
    uv run python -m quadprog_note.experiment_qp_pdas   # from experiment/

Outputs:
    graphs/quadprog_pdas.pdf        certified fraction and repairs vs n
    tables/quadprog_pdas_defs.tex   headline numbers as \newcommand macros

NumPy + SciPy + Matplotlib + cvx-quadprog.
"""

from __future__ import annotations

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

SIZES = [12, 40] if SMOKE else [12, 25, 50, 100, 200]
INSTANCES = 3 if SMOKE else 30

# Colours chosen to stay distinguishable in greyscale as well as in colour.
FAMILY_STYLE = {
    "box": ("#1f77b4", "o", "-"),
    "budget + bounds": ("#d62728", "s", "--"),
    "dense $C$": ("#2ca02c", "^", "-."),
    "with equalities": ("#9467bd", "D", ":"),
    "duplicated $C$": ("#8c564b", "v", (0, (3, 1, 1, 1))),
}


def hessian(rng, n):
    """Return a moderately conditioned positive definite Hessian."""
    b_mat = rng.standard_normal((n, n))
    return b_mat @ b_mat.T + n * np.eye(n)


def family_problem(name, rng, n):
    """Return (G, a, C, b, meq) for one of the four constraint shapes."""
    g = hessian(rng, n)
    a = rng.standard_normal(n)
    xu = np.linalg.solve(g, a)

    if name == "box":
        # Bounds straddling the unconstrained minimum, so roughly half bind.
        lo = xu - np.abs(rng.standard_normal(n))
        hi = xu + np.abs(rng.standard_normal(n))
        shift = rng.standard_normal(n) * 0.5
        lo, hi = lo + shift, hi + shift
        c = np.hstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([lo, -hi])
        return g, a, c, b, 0

    if name == "budget + bounds":
        # The long-only portfolio shape: one dense equality plus non-negativity.
        c = np.hstack([np.ones((n, 1)), np.eye(n)])
        b = np.concatenate([[1.0], np.zeros(n)])
        return g, a, c, b, 1

    if name == "dense $C$":
        m = max(2, n // 2)
        c = rng.standard_normal((n, m))
        b = c.T @ xu + rng.standard_normal(m) * 0.5
        return g, a, c, b, 0

    if name == "duplicated $C$":
        # Rank deficiency is what the P-matrix property rules out for bounds and
        # does not rule out here, but a random dense C almost never produces a
        # dependent guess. Repeating columns makes the failure mode reachable:
        # a guess that names both copies gives a singular working-set system.
        m = max(4, n // 2)
        base = rng.standard_normal((n, m // 2))
        c = np.hstack([base, base])
        b = c.T @ xu + np.tile(rng.standard_normal(m // 2), 2) * 0.5
        return g, a, c, b, 0

    if name == "with equalities":
        m = max(4, n // 2)
        meq = max(1, m // 4)
        c = rng.standard_normal((n, m))
        b = c.T @ xu + rng.standard_normal(m) * 0.5
        return g, a, c, b, meq

    raise ValueError(name)


class Instrument:
    """Wrap the fast path's internals so the shipped loop reports on itself."""

    def __init__(self) -> None:
        self.repairs = 0
        self.singular = 0
        self.accepted: list[float] = []
        self.rejected: list[float] = []
        self._orig = {}

    def __enter__(self):
        self._orig = {
            "_repair": _pdas._repair,
            "_working_set_solve": _pdas._working_set_solve,
            "_certified": _pdas._certified,
        }

        def repair(*args, **kwargs):
            self.repairs += 1
            return self._orig["_repair"](*args, **kwargs)

        def working_set_solve(*args, **kwargs):
            out = self._orig["_working_set_solve"](*args, **kwargs)
            if out is None:
                self.singular += 1
            return out

        def certified(g, a, c, b, meq, x, lagrangian):
            verdict = self._orig["_certified"](g, a, c, b, meq, x, lagrangian)
            (self.accepted if verdict else self.rejected).append(
                kkt_residual(g, a, c, b, meq, x, lagrangian)
            )
            return verdict

        _pdas._repair = repair
        _pdas._working_set_solve = working_set_solve
        _pdas._certified = certified
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(_pdas, name, fn)
        return False


def kkt_residual(g, a, c, b, meq, x, lagrangian) -> float:
    """Return the scaled sup-norm KKT residual of a candidate primal-dual pair.

    The same five conditions the certificate checks, reduced to one number by
    taking the worst, so that accepted and rejected candidates are comparable on a
    single axis.
    """
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    slack = c.T @ x - b
    parts = [
        float(np.max(np.abs(g @ x - a - c @ lagrangian))),
        float(np.max(np.abs(slack[:meq]), initial=0.0)),
        max(0.0, -float(np.min(slack[meq:], initial=0.0))),
        max(0.0, -float(np.min(lagrangian[meq:], initial=0.0))),
        float(np.max(np.abs(lagrangian[meq:] * slack[meq:]), initial=0.0)),
    ]
    return max(parts) / scale


def run():
    """Return per-cell results and the pooled certificate populations."""
    cells = {}
    accepted: list[float] = []
    rejected: list[float] = []
    singular_by_family = dict.fromkeys(FAMILY_STYLE, 0)
    attempts_by_family = dict.fromkeys(FAMILY_STYLE, 0)

    for family in FAMILY_STYLE:
        for n in SIZES:
            certified = 0
            repairs: list[int] = []
            outer: list[int] = []
            for i in range(INSTANCES):
                # Deterministic across runs: PYTHONHASHSEED randomises str hashing,
                # so a hash-derived seed would make the paper's numbers irreproducible.
                seed = (list(FAMILY_STYLE).index(family) * 1_000_000) + (n * 1_000) + i
                rng = np.random.default_rng(seed)
                g, a, c, b, meq = family_problem(family, rng, n)

                # The exact walk, for the iteration count the fast path competes with.
                try:
                    outer.append(int(solve_qp(g, a, c, b, meq).iterations[0]))
                except ValueError:
                    continue  # infeasible draw; not what this experiment is about

                with Instrument() as inst:
                    found = _pdas.attempt(g, a, c, b, meq)
                attempts_by_family[family] += 1
                singular_by_family[family] += inst.singular
                accepted.extend(inst.accepted)
                rejected.extend(inst.rejected)
                if found is not None:
                    certified += 1
                    repairs.append(inst.repairs)

            cells[(family, n)] = {
                "certified": certified / max(1, len(outer)),
                "repairs_mean": float(np.mean(repairs)) if repairs else float("nan"),
                "repairs_max": max(repairs) if repairs else 0,
                "outer_mean": float(np.mean(outer)) if outer else float("nan"),
                "instances": len(outer),
            }
            print(f"{family:<18} n={n:<5} certified={cells[(family, n)]['certified']:.0%} "
                  f"repairs={cells[(family, n)]['repairs_mean']:.1f} "
                  f"exact outer={cells[(family, n)]['outer_mean']:.0f}")

    return cells, accepted, rejected, singular_by_family, attempts_by_family


def log_x_ticks(ax, sizes) -> None:
    """Label a log x-axis at the sizes actually measured, and nowhere else.

    A log axis spanning a narrow range -- 12 to 200 here -- puts minor ticks at
    2, 3, 4, 6 times each decade, and at figure width their labels overlap into an
    illegible smear. The sizes are a short discrete list, so label exactly those
    and silence the minor formatter.
    """
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2)


def figure(cells) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for family, (colour, marker, dash) in FAMILY_STYLE.items():
        xs = SIZES
        ax_a.plot(xs, [100 * cells[(family, n)]["certified"] for n in xs],
                  color=colour, marker=marker, linestyle=dash, markersize=4, label=family)
        ax_b.plot(xs, [cells[(family, n)]["repairs_mean"] for n in xs],
                  color=colour, marker=marker, linestyle=dash, markersize=4, label=family)

    ax_a.set_xscale("log")
    log_x_ticks(ax_a, SIZES)
    ax_a.set_xlabel("$n$")
    ax_a.set_ylabel("certified (\\%)")
    ax_a.set_ylim(-5, 105)
    ax_a.set_title("Guesses the certificate accepts")
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax_a.legend(framealpha=0.9, loc="lower right")

    ax_b.set_xscale("log")
    log_x_ticks(ax_b, SIZES)
    ax_b.set_xlabel("$n$")
    ax_b.set_ylabel("set repairs to convergence")
    ax_b.set_ylim(bottom=0)
    ax_b.set_title("Repairs, against an exact walk that grows with $n$")
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    fig.tight_layout(pad=0.8)
    fig.savefig(GRAPHS / "quadprog_pdas.pdf", bbox_inches="tight")
    fig.savefig(GRAPHS / "quadprog_pdas.png", bbox_inches="tight", dpi=150)
    print(f"\nSaved {GRAPHS / 'quadprog_pdas.pdf'}")


def sci(x) -> str:
    """Format a positive value as bare LaTeX math ``m\\times10^{e}``."""
    if x == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / 10.0**exponent
    if round(mantissa, 1) >= 10.0:  # 9.99e-16 must print as 1.0e-15
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return f"{mantissa:.1f}\\times 10^{{{exponent}}}"


def emit(cells, accepted, rejected, singular, attempts) -> None:
    biggest = SIZES[-1]
    all_repairs = [cells[(f, n)]["repairs_mean"] for f in FAMILY_STYLE for n in SIZES]
    all_repairs = [r for r in all_repairs if not np.isnan(r)]
    max_repairs = max(cells[(f, n)]["repairs_max"] for f in FAMILY_STYLE for n in SIZES)

    # The claim Section 7 makes about the exact walk's cost at n = 100.
    ref_n = 100 if 100 in SIZES else biggest
    budget_outer = cells[("budget + bounds", ref_n)]["outer_mean"]

    small = min(SIZES)
    certified_small = min(cells[(f, small)]["certified"] for f in FAMILY_STYLE)
    certified_big = min(cells[(f, biggest)]["certified"] for f in FAMILY_STYLE)

    path = TABLES / "quadprog_pdas_defs.tex"
    with open(path, "w") as fh:
        fh.write("% Generated by quadprog_note/experiment_qp_pdas.py -- do not edit by hand.\n")
        fh.write(f"\\newcommand{{\\qpPdasInstances}}{{{INSTANCES}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasFamilies}}{{{len(FAMILY_STYLE)}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasSizes}}{{{', '.join(str(n) for n in SIZES)}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasRepairsLo}}{{{min(all_repairs):.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasRepairsHi}}{{{max(all_repairs):.1f}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasRepairsMax}}{{{max_repairs}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasRefN}}{{{ref_n}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasBudgetOuter}}{{{budget_outer:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasNsmall}}{{{small}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasNbig}}{{{biggest}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasCertSmall}}{{{100 * certified_small:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasCertBig}}{{{100 * certified_big:.0f}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasAccepted}}{{{len(accepted)}}}\n")
        fh.write(f"\\newcommand{{\\qpPdasRejected}}{{{len(rejected)}}}\n")
        if accepted:
            fh.write(f"\\newcommand{{\\qpPdasAcceptWorst}}{{{sci(max(accepted))}}}\n")
        if rejected:
            fh.write(f"\\newcommand{{\\qpPdasRejectBest}}{{{sci(min(rejected))}}}\n")
        for family, key in (("box", "Box"), ("budget + bounds", "Budget"),
                            ("dense $C$", "Dense"), ("with equalities", "Eq"),
                            ("duplicated $C$", "Dup")):
            rate = singular[family] / max(1, attempts[family])
            fh.write(f"\\newcommand{{\\qpPdasSingular{key}}}{{{100 * rate:.1f}}}\n")
    print(f"Saved {path}")

    print(f"\ncertificate: {len(accepted)} accepted, worst residual "
          f"{max(accepted):.2e}" if accepted else "no accepted candidates")
    if rejected:
        print(f"             {len(rejected)} rejected, best residual {min(rejected):.2e}")
    print("rank-deficient working-set solves, by family:")
    for family in FAMILY_STYLE:
        print(f"  {family:<18} {singular[family]:>4} over {attempts[family]:>4} attempts")


def main() -> None:
    cells, accepted, rejected, singular, attempts = run()
    figure(cells)
    emit(cells, accepted, rejected, singular, attempts)
    print("\nDone.")


if __name__ == "__main__":
    main()
