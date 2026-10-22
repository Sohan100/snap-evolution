"""Tables for the write-up of the sweep, from runs/summary.csv (python -m snapevo summary first).

    python cluster/sweep_tables.py

prints, as markdown:

1. ratio    the baseline-to-SNAP T ratio along SNAP's lower envelope, with the
            fallbacks drawn, as its range per model and time (the "drawn" column
            of snapevo compare, envelope rows only)
2. Tqt      T per qubit per unit time at a per-site infidelity of 1e-6 and 1e-4,
            drawn, against t, for SNAP and the baseline (log-log along each envelope)
3. depths   per budget with two compiles, T per qubit at each depth and which won,
            and the wall of each model and time (the largest jump in T per qubit
            from one budget to the next tighter one, cheapest depth per budget)
4. fallback the baseline's drawn infidelity over its success branch and its bound
            over its success branch, median and largest per model and time
"""

from __future__ import annotations

import csv
import os
import statistics
from pathlib import Path

from snapevo.cli import _envelope, _interp

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("SNAPEVO_RUNS", REPO / "runs"))
NAMES = {"xxz_d0.25_hx1.6": r"$\Delta = 0.25$, $h_x = 1.6$", "xxz_d0.25_hx0": r"$\Delta = 0.25$, $h_x = 0$"}


def f(r, k):
    return float(r[k]) if r.get(k, "") != "" else None


def drawn_c(r):
    """Per-site infidelity with the fallbacks drawn, SNAP's full bound where it has no draws."""
    return f(r, "c_sampled") if r.get("c_sampled") else (f(r, "c_total") if r["method"] == "snap" else None)


def tex(b: float) -> str:
    m, e = f"{b:.0e}".split("e")
    e = int(e)
    return f"$10^{{{e}}}$" if m == "1" else f"${m}\\times10^{{{e}}}$"


def main() -> None:
    rows = list(csv.DictReader((RUNS / "summary.csv").open()))
    groups = {}
    for r in rows:
        groups.setdefault((NAMES[r["model"]], float(r["t"])), []).append(r)
    keys = sorted(groups)

    print("### Ratio along SNAP's envelope, fallbacks drawn\n")
    print("| model | $t$ | SNAP envelope points | compared | ratio range | at the loose end | at the tight end |")
    print("|---|---|---|---|---|---|---|")
    for key in keys:
        g = groups[key]
        snap = [(drawn_c(r), f(r, "Tq")) for r in g if r["method"] == "snap" and f(r, "Tq") > 0 and drawn_c(r)]
        base = [(drawn_c(r), f(r, "Tq")) for r in g if r["method"] == "baseline" and f(r, "Tq") > 0 and drawn_c(r)]
        env_s, env_b = _envelope(snap), _envelope(base)
        ratios = [(x, _interp(env_b, x) / y) for x, y in env_s if _interp(env_b, x) is not None]
        if ratios:
            lo, hi = min(q for _, q in ratios), max(q for _, q in ratios)
            loose, tight = max(ratios)[1], min(ratios)[1]
            print(f"| {key[0]} | {key[1]:g} | {len(env_s)} | {len(ratios)} | {lo:.1f} to {hi:.1f} | "
                  f"{loose:.1f} | {tight:.1f} |")
        else:
            print(f"| {key[0]} | {key[1]:g} | {len(env_s)} | 0 | - | - | - |")

    print("\n### T per qubit per unit time at fixed per-site infidelity, fallbacks drawn\n")
    for c in (1e-6, 1e-4):
        print(f"At $c = {tex(c)[1:-1]}$ ($1 - F \\approx {100 * c:.0e}$ at $N = 100$):\n")
        times = sorted({k[1] for k in keys})
        print("| model | method | " + " | ".join(f"$t = {t:g}$" for t in times) + " |")
        print("|---|---|" + "---|" * len(times))
        for m in ("A", "B"):
            for method, name in (("snap", "SNAP"), ("baseline", "baseline")):
                cells = []
                for t in times:
                    g = groups.get((m, t), [])
                    env = _envelope([(drawn_c(r), f(r, "Tq")) for r in g
                                     if r["method"] == method and f(r, "Tq") > 0 and drawn_c(r)])
                    y = _interp(env, c)
                    cells.append("-" if y is None else f"{y / t:.1f}")
                print(f"| {m} | {name} | " + " | ".join(cells) + " |")
        print()

    print("### Two compile depths per budget\n")
    print("| model | $t$ | budget | shallow $D$, T/q | deep $D$, T/q | won |")
    print("|---|---|---|---|---|---|")
    walls = []
    for key in keys:
        snap = [r for r in groups[key] if r["method"] == "snap"]
        best = {}
        for b in sorted({f(r, "budget") for r in snap}, reverse=True):
            pts = sorted((r for r in snap if f(r, "budget") == b), key=lambda r: int(r["D"]))
            best[b] = min(pts, key=lambda r: f(r, "Tq"))
            if len(pts) == 2:
                s, d = pts
                qs, qd = f(s, "Tq"), f(d, "Tq")
                won = "tie" if abs(qd - qs) <= 0.005 * max(qs, qd, 1e-9) else ("deep" if qd < qs else "shallow")
                print(f"| {key[0]} | {key[1]:g} | {tex(b)} | {s['D']}, {f(s, 'Tq'):.1f} | "
                      f"{d['D']}, {f(d, 'Tq'):.1f} | {won} |")
        seq = [(b, f(r, "Tq"), r["D"]) for b, r in sorted(best.items(), reverse=True) if f(r, "Tq") > 0]
        jumps = [(q1 / q0, b0, b1, q0, q1, d0, d1) for (b0, q0, d0), (b1, q1, d1) in zip(seq, seq[1:])
                 if b1 <= 1.0001e-4]
        if jumps:
            walls.append((key, max(jumps), seq[-1]))
    print("\n### Walls (largest jump in T per qubit between neighbouring budgets, from $10^{-4}$ down)\n")
    print("| model | $t$ | from | to | T/q | jump | tightest budget reached |")
    print("|---|---|---|---|---|---|---|")
    for key, (q, b0, b1, q0, q1, d0, d1), last in walls:
        print(f"| {key[0]} | {key[1]:g} | {tex(b0)} ($D = {d0}$) | {tex(b1)} ($D = {d1}$) | "
              f"{q0:.0f} to {q1:.0f} | {q:.2f}x | {tex(last[0])} |")

    print("\n### Baseline: drawn and bound against the success branch\n")
    print("| model | $t$ | points drawn | drawn / success, median | largest | bound / success, median | largest |")
    print("|---|---|---|---|---|---|---|")
    for key in keys:
        b = [r for r in groups[key] if r["method"] == "baseline" and r.get("infid_sampled")
             and f(r, "infid") < 0.999]
        if not b:
            continue
        d = [f(r, "infid_sampled") / f(r, "infid") for r in b]
        bd = [min(f(r, "infid_total"), 1.0) / f(r, "infid") for r in b]
        print(f"| {key[0]} | {key[1]:g} | {len(b)} | {statistics.median(d):.3f} | {max(d):.2f} | "
              f"{statistics.median(bd):.2f} | {max(bd):.1f} |")


if __name__ == "__main__":
    main()
