"""Figures of the translation-invariant (infinite-chain) results: SNAP against Trotter + KN.

    python cluster/ti_figures.py [--times 1,2,4,6] [--out runs/figures/ti]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0. Every error is the infinite-chain
per-site infidelity c_inf of runs/ti/imps_Lu8_n32/ (ti_imps_run.py pack): the mean over 32 random product inputs
with an 8-site repeating cell, of the success branch (the KN fallbacks are not drawn). Every T count is the bulk
circuit's T per qubit (the result file's Tq less its T_edges_per_qubit): the translation-invariant circuit has no
ends. No open-chain number enters.

    ti_tradeoff   T per qubit against c_inf, per field and t; SNAP's points and lower envelope, the Trotter + KN
                  points (marker by product-formula order) and their lower envelope
    ti_ratio      Trotter + KN T over SNAP T at equal c_inf, along SNAP's envelope (log-log interpolation on the
                  baseline's envelope), per field, one line per t
    ti_time       T per qubit per unit time at c_inf = 1e-4 and 1e-6 against t, per field
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ti_ring_eval as tr                                                   # noqa: E402

FIELD_NAME = {"1.6": r"$h_x = 1.6$", "0": r"$h_x = 0$"}
SNAP_C, BASE_C = "#2a78d6", "#eb6834"         # categorical slots 1 and 2 of the reference palette
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER_MARK = {1: "o", 2: "s", 4: "^"}
T_RAMP = ["#a9c9f0", "#6ea5e6", "#2a78d6", "#0f4f9e", "#0a3266"]   # one hue, light to dark, for t


def envelope(pts):
    """Lower envelope of (c, T): each kept point beats every cheaper one in c."""
    env, best = [], math.inf
    for c, T in sorted(pts, key=lambda p: (p[1], p[0])):
        if c < best:
            env.append((c, T))
            best = c
    return sorted(env)


def interp(env, c):
    """T on the envelope at c, log-log between neighbours; None outside its range."""
    if not env or c < env[0][0] or c > env[-1][0]:
        return None
    for (c0, T0), (c1, T1) in zip(env, env[1:]):
        if c0 <= c <= c1:
            if c1 == c0:
                return min(T0, T1)
            f = math.log(c / c0) / math.log(c1 / c0)
            return math.exp(math.log(T0) + f * (math.log(T1) - math.log(T0)))
    return env[0][1]


def load(times) -> dict:
    """{(field, t): [point]} with c_inf, its standard error, bulk T per qubit, method and product-formula order."""
    out = {}
    for t in times:
        for f in tr.FIELDS:
            path = tr.RUNS / "ti" / "imps_Lu8_n32" / f"{tr.TAGS[f]}_t{t}.json"
            if not path.exists():
                continue
            pts = []
            for r in json.loads(path.read_text())["points"]:
                if r["n"] < 32:
                    continue
                p = next(q for q in tr.points((t,), (r["method"],)) if q["stem"] == r["stem"] and q["field"] == f)
                rec = json.loads(Path(p["eval_file"] or p["file"]).read_text())
                Tq = float(rec["Tq"]) - float(rec.get("T_edges_per_qubit") or 0.0)
                if Tq <= 0:
                    continue
                pts.append({"method": r["method"], "c": r["c_inf"], "se": r["c_inf_se"], "T": Tq, "D": r["D"],
                            "order": p.get("order"), "stem": r["stem"]})
            out[(f, t)] = pts
    return out


def style(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(which="both", colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def fig_tradeoff(data, times, out: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(times), figsize=(3.1 * len(times), 6.2), squeeze=False)
    for i, f in enumerate(tr.FIELDS):
        for j, t in enumerate(times):
            ax = axes[i][j]
            style(ax)
            pts = data.get((f, t), [])
            sn = [p for p in pts if p["method"] == "snap"]
            bs = [p for p in pts if p["method"] == "baseline"]
            for o, m in ORDER_MARK.items():
                q = [p for p in bs if p["order"] == o]
                ax.scatter([p["c"] for p in q], [p["T"] for p in q], s=22, marker=m, facecolor="none",
                           edgecolor=BASE_C, lw=1.0, alpha=0.75, zorder=2)
            ax.scatter([p["c"] for p in sn], [p["T"] for p in sn], s=18, color=SNAP_C, edgecolor="white",
                       lw=0.6, alpha=0.8, zorder=3)
            for q, col in ((bs, BASE_C), (sn, SNAP_C)):
                e = envelope([(p["c"], p["T"]) for p in q])
                if e:
                    ax.plot(*zip(*e), color=col, lw=2.0, zorder=4, drawstyle="default")
            ax.set_title(f"{FIELD_NAME[f]}, $t = {t}$", fontsize=9, color=TEXT)
            if i == 1:
                ax.set_xlabel(r"per-site infidelity $c_\infty$", fontsize=9, color=TEXT)
            if j == 0:
                ax.set_ylabel("T per qubit", fontsize=9, color=TEXT)
            if i == 0 and j == 0:
                e = envelope([(p["c"], p["T"]) for p in sn])
                eb = envelope([(p["c"], p["T"]) for p in bs])
                if e:
                    ax.annotate("SNAP", e[len(e) // 2], textcoords="offset points", xytext=(-26, -14),
                                color=TEXT, fontsize=8)
                if eb:
                    ax.annotate("Trotter + KN", eb[len(eb) // 2], textcoords="offset points", xytext=(4, 6),
                                color=TEXT, fontsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=SNAP_C, marker="o", lw=2, ms=5, label="SNAP (points, lower envelope)"),
               Line2D([], [], color=BASE_C, lw=2, label="Trotter + KN, lower envelope")]
    handles += [Line2D([], [], color=BASE_C, marker=m, ls="none", mfc="none", ms=5, label=f"Trotter order {o} + KN")
                for o, m in ORDER_MARK.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(r"XXZ chain, $\Delta = 0.25$, infinite chain (translation invariant): T cost against per-site "
                 r"infidelity", fontsize=10, color=TEXT)
    fig.text(0.5, 0.935, "SNAP from the current ring cells (L-BFGS-trained); Hessian (trust-region Newton) "
             "retraining in progress", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ti_tradeoff.{ext}", dpi=170)
    plt.close(fig)


def ratios(data, f, t):
    pts = data.get((f, t), [])
    es = envelope([(p["c"], p["T"]) for p in pts if p["method"] == "snap"])
    eb = envelope([(p["c"], p["T"]) for p in pts if p["method"] == "baseline"])
    return [(c, interp(eb, c) / T) for c, T in es if interp(eb, c)]


def fig_ratio(data, times, out: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    ramp = T_RAMP[-len(times):] if len(times) <= len(T_RAMP) else T_RAMP
    for ax, f in zip(axes, tr.FIELDS):
        style(ax)
        ax.set_yscale("linear")
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        for t, col in zip(times, ramp):
            r = ratios(data, f, t)
            if r:
                ax.plot(*zip(*r), color=col, lw=2.0, marker="o", ms=4, label=f"$t = {t}$")
        ax.set_title(FIELD_NAME[f], fontsize=9, color=TEXT)
        ax.set_xlabel(r"per-site infidelity $c_\infty$", fontsize=9, color=TEXT)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("T(Trotter + KN) / T(SNAP)\nat equal $c_\\infty$", fontsize=9, color=TEXT)
    fig.suptitle("SNAP's T saving on the infinite chain (current ring cells, L-BFGS-trained)", fontsize=10, color=TEXT)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ti_ratio.{ext}", dpi=170)
    plt.close(fig)


def fig_time(data, times, out: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, f in zip(axes, tr.FIELDS):
        style(ax)
        ax.set_xscale("linear")
        for c, ls in ((1e-4, "-"), (1e-6, "--")):
            for meth, col, lab in (("snap", SNAP_C, "SNAP"), ("baseline", BASE_C, "Trotter + KN")):
                xs, ys = [], []
                for t in times:
                    e = envelope([(p["c"], p["T"]) for p in data.get((f, t), []) if p["method"] == meth])
                    T = interp(e, c)
                    if T:
                        xs.append(t)
                        ys.append(T / t)
                if xs:
                    exp = int(round(math.log10(c)))
                    ax.plot(xs, ys, color=col, ls=ls, lw=2.0, marker="o", ms=4,
                            label=f"{lab}, $c_\\infty = 10^{{{exp}}}$")
        ax.set_title(FIELD_NAME[f], fontsize=9, color=TEXT)
        ax.set_xlabel("$t$", fontsize=9, color=TEXT)
        ax.set_xticks(times)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("T per qubit per unit time", fontsize=9, color=TEXT)
    fig.suptitle("Cost per unit time on the infinite chain (current ring cells, L-BFGS-trained)", fontsize=10,
                 color=TEXT)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ti_time.{ext}", dpi=170)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", default="1,2,4,6")
    ap.add_argument("--out", default=str(tr.RUNS / "figures" / "ti"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    times = [int(v) for v in a.times.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(times)
    fig_tradeoff(data, times, out)
    fig_ratio(data, times, out)
    fig_time(data, times, out)
    for f in tr.FIELDS:
        for t in times:
            r = [x for _, x in ratios(data, f, t)]
            if r:
                print(f"h_x = {f}, t = {t}: ratio median {np.median(r):.2f}, range {min(r):.2f}-{max(r):.2f}, "
                      f"{len(r)} SNAP envelope points")
    print(f"wrote {out}/ti_tradeoff, ti_ratio, ti_time (.pdf, .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
