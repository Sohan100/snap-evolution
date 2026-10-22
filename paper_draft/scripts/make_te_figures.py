"""Figures and Table I of the time-evolution section (Sec. V), from the open-chain arm only.

    .venv/bin/python paper_draft/scripts/make_te_figures.py [--refresh]

Reads the frozen copy paper_draft/data_snapshot/runs/ of the run records, so a rerun gives the same figures and
numbers. --refresh first replaces that copy with the current records from runs/; after it, recheck every
number in the text against te_text_numbers.txt.

Writes, next to section_5_time_evolution.tex in paper_draft/:
    fig_te_tradeoff.pdf    T per qubit against the per-site infidelity, h_x = 1.6 and 0, t = 1, 2 and 4
    fig_te_floors.pdf      floor of the trained circuits against depth, every t
    fig_te_time.pdf        T per qubit per unit time against t at eps = 1e-3 and 3e-4
    tab_te_ratio.tex       Table I, product formula T over SNAP T at equal eps
    fig_te_data.csv        every plotted point, the table view of the three figures
    fig_te_sources.txt     every input file with its modification time (the open-chain runs are still adding
                           points, so this is the snapshot the figures show)

Data. No ring-trained circuit is used anywhere.
    SNAP       RUNS/<tag>/t<t>/chain/ladder.json and chain/newton/eval_*.json: trained, compiled, realized and
               evaluated on the 100-site open chain (cluster/chain_train.py, cluster/chain_snap.py --reopt newton)
    baseline   RUNS/<tag>/t<t>/baseline_o*.json: product formulas, not trained, evaluated on the same chain
    Liu        LIT/mps_final/<tag>/t<t>/*.json: orderings of Liu et al. (arXiv:2610.03672), N = 100 MPS,
               success branch only, not yet independently verified
    rigorous   LIT/rigorous/best.csv: guaranteed per-site infidelity at N = 100

The x value of a measured point is the per-site infidelity 1 - (1 - infid)^(1/N) with the KN fallbacks drawn.
A SNAP point that is not drawn yet uses its fallback bound instead, which can only overstate its error, and a
baseline point that was not drawn is left out (snap_advantage_vs_t.csv's rule). Each series is its lower
envelope: a point is kept only if no cheaper point has a lower infidelity.
"""

import csv
import glob
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
LIT = Path("/pscratch/sd/s/sohan100/snapevo-scratch/litbase")
OUT = Path(__file__).resolve().parent.parent
# The open-chain runs are still adding points, so the figures, the table and the numbers in the text all read a
# frozen copy of the run records. --refresh copies the current records into it first.
SNAPSHOT = OUT / "data_snapshot"
RUNS = SNAPSHOT / "runs"

MODELS = (("xxz_d0.25_hx1.6", 1.6, r"$h_x = 1.6$"), ("xxz_d0.25_hx0", 0.0, r"$h_x = 0$"))
TIMES = (1, 2, 4)
N = 100
OUR_FAMILIES = {"BW10/o1", "BW10/o2", "BW10/o4", "BW10/XEO/o1", "BW10/XEO/o2", "BW10/XEO/o4"}

# Okabe-Ito, as in the paper's other figures. Checked with the dataviz validator (Python port): all-pairs CVD
# Delta E >= 11.0, normal-vision >= 15.6. Orange is below 3:1 on white, so every series also has its own marker.
SNAP_C, PF_C, LIT_C, RIG_C = "#0072B2", "#D55E00", "#009E73", "#E69F00"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# ordinal ramp for t (matplotlib Blues 0.45, 0.73, 1): adjacent OKLCH Delta L >= 0.08, lightest 2.1:1 on white
T_RAMP = {1: "#7fb9da", 2: "#3686c0", 4: "#08306b"}

COL, TEXT = 3.40, 7.06  # revtex prx column and text widths, inches

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.3, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

SOURCES = set()


def load(p):
    SOURCES.add(Path(p))
    with open(p) as fh:
        return json.load(fh)


def envelope(pts):
    """Lower envelope of (eps, T) points: each kept point beats every cheaper one in eps."""
    env, best = [], float("inf")
    for x, y in sorted(pts, key=lambda p: (p[1], p[0])):
        if x < best:
            env.append((x, y))
            best = x
    return sorted(env)


def interp(env, x):
    """T on the envelope at eps = x, log-log between neighbours; None outside its range."""
    if not env or x < env[0][0] or x > env[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(env, env[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return min(y0, y1)
            f = math.log(x / x0) / math.log(x1 / x0)
            return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
    return env[0][1]


def per_site(infid):
    return 1.0 - (1.0 - min(infid, 1.0)) ** (1.0 / N)


UNDRAWN = set()  # SNAP evaluations still placed at their fallback bound


def snap_points(tag, t):
    """(eps, T per qubit) of every evaluated open-chain SNAP point; drawn eps, else the bound."""
    out = []
    for p in sorted(glob.glob(str(RUNS / tag / f"t{t}" / "chain" / "newton" / "eval_*.json"))):
        e = load(p)
        if "infid_sampled" not in e:
            UNDRAWN.add(p)
        x = per_site(e["infid_sampled"]) if "infid_sampled" in e else per_site(e["infid_total"])
        if e["Tq"] > 0:
            out.append((x, e["Tq"]))
    return out


def baseline_points(tag, t):
    out = []
    for p in sorted(glob.glob(str(RUNS / tag / f"t{t}" / "baseline_o*.json"))):
        e = load(p)
        if "mps" in e and "infid_sampled" in e and e["Tq"] > 0:
            out.append((per_site(e["infid_sampled"]), e["Tq"]))
    return out


SOURCES.add(LIT / "rigorous" / "best.csv")
RIG = list(csv.DictReader((LIT / "rigorous" / "best.csv").open()))


def series(tag, hx, t):
    """Envelopes of the four series at one field and time."""
    out = {"snap": envelope(snap_points(tag, t)), "baseline": envelope(baseline_points(tag, t))}
    lit = []
    for p in glob.glob(str(LIT / "mps_final" / tag / f"t{t:g}" / "*.json")):
        d = load(p)
        if d["family"] not in OUR_FAMILIES and d["Tq"] > 0:
            lit.append((d["c_site"], d["Tq"]))
    out["lit"] = envelope(lit)
    best = {}
    for r in RIG:
        if float(r["hx"]) == hx and int(r["N"]) == N and float(r["t"]) == t:
            c = float(r["c"])
            best[c] = min(best.get(c, float("inf")), float(r["Tq"]))
    out["rig"] = sorted(best.items())
    return out


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


DATA = []  # (figure, field, t, series, x, y)

SPEC = (  # key, color, marker, face, linestyle, legend label
    ("rig", RIG_C, "D", RIG_C, (0, (4, 2)), "Rigorous bound, order 4"),
    ("lit", LIT_C, "^", "white", "-", "Orderings of Liu et al., measured"),
    ("baseline", PF_C, "s", "white", "-", "Product formulas, measured"),
    ("snap", SNAP_C, "o", SNAP_C, "-", "SNAP"),
)
ARROW_EPS = 1e-3


def fig_tradeoff():
    fig, axes = plt.subplots(2, 3, figsize=(TEXT, 4.4), sharex=True, sharey=True)
    ratios = []
    for i, (tag, hx, mname) in enumerate(MODELS):
        for j, t in enumerate(TIMES):
            ax = axes[i][j]
            style(ax)
            env = series(tag, hx, t)
            for key, color, marker, face, ls, label in SPEC:
                e = env[key]
                if not e:
                    continue
                xs, ys = zip(*e)
                ax.plot(xs, ys, color=color, ls=ls, marker=marker, ms=3.8 if key != "rig" else 3.2, mfc=face,
                        mec=color, mew=0.9, markevery=2 if key == "rig" else 1, label=label, zorder=3)
                DATA.extend(("tradeoff", hx, t, key, x, y) for x, y in e)
            # the gap at ARROW_EPS, the error every panel reaches
            ts, tb = interp(env["snap"], ARROW_EPS), interp(env["baseline"], ARROW_EPS)
            if ts and tb:
                ax.annotate("", xy=(ARROW_EPS, ts * 1.12), xytext=(ARROW_EPS, tb / 1.12),
                            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7, shrinkA=0, shrinkB=0))
                # below SNAP and toward smaller eps, where the curves rise away from it
                ax.text(ARROW_EPS * 0.8, ts / 1.35, rf"$\times{tb / ts:.1f}$", color=INK, fontsize=7.5,
                        ha="left", va="top")
                ratios.append((hx, t, tb / ts))
            letter = "abcdef"[3 * i + j]
            ax.text(0.04, 0.95, rf"$\mathbf{{({letter})}}$  {mname}, $t = {t}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.5)
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 1:
                ax.set_xlabel(r"error per site $\epsilon$")
            if j == 0:
                ax.set_ylabel("T count per qubit")
    ax = axes[0][0]
    ax.set_xlim(3e-1, 1e-10)
    ax.set_ylim(1.0, 1e5)
    ax.set_xticks([1e-2, 1e-4, 1e-6, 1e-8])
    # No direct labels: the curves are too close for them. Each series has its own marker and fill, so the
    # legend does not rely on color.
    handles = [Line2D([], [], color=c, ls=ls, marker=m, mfc=f, mec=c, mew=0.9, ms=3.8, label=lab)
               for _, c, m, f, ls, lab in SPEC[::-1]]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), handlelength=2.6,
               columnspacing=1.6)
    fig.tight_layout(h_pad=0.4, w_pad=0.5)
    fig.savefig(OUT / "fig_te_tradeoff.pdf")
    fig.savefig(OUT / "fig_te_tradeoff.png", dpi=200)
    plt.close(fig)
    return ratios


def ladder_records(tag, t):
    d = load(RUNS / tag / f"t{t}" / "chain" / "ladder.json")
    for k in ("results", "depths", "ladder", "records"):
        if isinstance(d.get(k), list):
            return d[k]
    raise KeyError(f"no depth records in {tag} t={t} ladder.json: keys {sorted(d)}")


def fig_floors():
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.6), sharex=True, sharey=True)
    for ax, (tag, hx, mname), letter in zip(axes, MODELS, "ab"):
        style(ax)
        for t in TIMES:
            pts = sorted((int(r["D"]), float(r["c_train"])) for r in ladder_records(tag, t))
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=T_RAMP[t], marker="o", ms=3.0, mfc=T_RAMP[t], mec=T_RAMP[t], lw=1.1, zorder=3)
            ax.text(xs[-1] + 0.5, ys[-1], rf"$t = {t}$", fontsize=7, color=MUTED, va="center", ha="left")
            DATA.extend(("floors", hx, t, "floor", x, y) for x, y in pts)
        ax.set_yscale("log")
        ax.text(0.97, 0.95, rf"$\mathbf{{({letter})}}$  {mname}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5)
        ax.set_ylabel(r"floor $\epsilon_0$")
    axes[1].set_xlabel(r"depth $D$ (sublayers)")
    axes[0].set_xlim(0, 20.5)
    axes[0].set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    axes[0].set_ylim(3e-11, 1.0)
    axes[0].set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1])
    fig.tight_layout(h_pad=0.3)
    fig.savefig(OUT / "fig_te_floors.pdf")
    fig.savefig(OUT / "fig_te_floors.png", dpi=200)
    plt.close(fig)


TIME_EPS = (1e-3, 3e-4)


def per_time(tag, hx):
    """T per qubit per unit time on the SNAP and baseline envelopes at TIME_EPS, by t."""
    out = {}
    for t in TIMES:
        env = series(tag, hx, t)
        for c in TIME_EPS:
            for key in ("snap", "baseline"):
                v = interp(env[key], c)
                if v:
                    out.setdefault((key, c), []).append((t, v / t))
    return out


def fig_time():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.35), sharey=True)
    for ax, (tag, hx, mname), letter in zip(axes, MODELS, "ab"):
        style(ax)
        for (key, c), pts in sorted(per_time(tag, hx).items()):
            color = SNAP_C if key == "snap" else PF_C
            marker = "o" if key == "snap" else "s"
            tight = c < 1e-3
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=color, ls="--" if tight else "-", marker=marker, ms=3.6,
                    mfc="white" if tight else color, mec=color, mew=1.0, lw=1.1, zorder=3)
            DATA.extend(("time", hx, t, f"{key}_{c:.0e}", t, y) for t, y in pts)
        ax.set_yscale("log")
        ax.set_xticks(list(TIMES))
        ax.set_xlim(0.6, 4.4)
        ax.text(0.04, 0.96, rf"$\mathbf{{({letter})}}$  {mname}", transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5)
        ax.set_xlabel(r"time $t$")
    axes[0].set_ylabel("T count per qubit per unit time")
    axes[0].set_ylim(8, 600)
    handles = [
        Line2D([], [], color=SNAP_C, marker="o", mfc=SNAP_C, mec=SNAP_C, ms=3.6, label="SNAP"),
        Line2D([], [], color=PF_C, marker="s", mfc=PF_C, mec=PF_C, ms=3.6, label="product formula"),
        Line2D([], [], color=MUTED, ls="-", label=r"$\epsilon = 10^{-3}$"),
        Line2D([], [], color=MUTED, ls="--", marker="o", mfc="white", mec=MUTED, ms=3.6,
               label=r"$\epsilon = 3 \times 10^{-4}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.13), handlelength=2.4,
               columnspacing=1.2)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(OUT / "fig_te_time.pdf")
    fig.savefig(OUT / "fig_te_time.png", dpi=200)
    plt.close(fig)


TABLE_EPS = (1e-2, 1e-3, 1e-4)


def table():
    """Table I: product-formula T over SNAP T at equal eps (drawn, else bound for SNAP), every t, both fields."""
    cells = {}
    for tag, hx, _ in MODELS:
        for t in TIMES:
            env = series(tag, hx, t)
            for c in TABLE_EPS:
                ts, tb = interp(env["snap"], c), interp(env["baseline"], c)
                cells[(hx, t, c)] = None if not (ts and tb) else tb / ts
    fmt = lambda v: "--" if v is None else f"{v:.1f}"
    head = " & ".join(rf"$10^{{{int(round(math.log10(c)))}}}$" for c in TABLE_EPS)
    bound_note = (r" SNAP points whose fallback branches are not drawn yet sit at their fallback bound, which can"
                  r" only overstate their error." if UNDRAWN else "")
    lines = [
        "% Generated by paper_draft/scripts/make_te_figures.py; do not edit by hand.",
        r"\begin{table}[t]",
        r"\caption{Ratio of the T count of the cheapest measured product formula to that of SNAP, at equal error"
        r" per site $\epsilon$ on 100 sites, with the fallback branches of the product formulas drawn."
        + bound_note + r" A dash marks a time at which the compiled circuits do not yet reach that $\epsilon$.}",
        r"\label{tab:te_ratio}",
        r"\begin{ruledtabular}",
        r"\begin{tabular}{ccccccc}",
        r" & \multicolumn{3}{c}{$h_x = 1.6$} & \multicolumn{3}{c}{$h_x = 0$} \\",
        rf"$t$ & {head} & {head} \\",
        r"\hline",
    ]
    for t in TIMES:
        row = [fmt(cells[(hx, t, c)]) for _, hx, _ in MODELS for c in TABLE_EPS]
        lines.append(f"{t} & " + " & ".join(row) + r" \\")
    lines += [r"\end{tabular}", r"\end{ruledtabular}", r"\end{table}", ""]
    (OUT / "tab_te_ratio.tex").write_text("\n".join(lines))
    return cells


def refresh_snapshot():
    """Replace the frozen copy with the current open-chain and baseline records from the repo's runs/."""
    import shutil
    if RUNS.exists():
        shutil.rmtree(RUNS)
    n = 0
    for tag, _, _ in MODELS:
        for t in TIMES:
            src = REPO / "runs" / tag / f"t{t}"
            files = ([src / "chain" / "ladder.json"] + sorted((src / "chain" / "newton").glob("eval_*.json"))
                     + sorted(src.glob("baseline_o*.json")))
            for f in files:
                dst = RUNS / f.relative_to(REPO / "runs")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    (SNAPSHOT / "SNAPSHOT.txt").write_text(
        f"Copied {n} run records from {REPO / 'runs'} at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.\n")
    print(f"snapshot: {n} files")


def text_numbers():
    """Every number the text of Sec. V quotes, from the same snapshot as the figures."""
    import statistics as st
    lines = [(SNAPSHOT / "SNAPSHOT.txt").read_text().strip()]
    say = lines.append
    evals = {(tag, t): [load(p) for p in sorted(glob.glob(str(RUNS / tag / f"t{t}" / "chain" / "newton" /
                                                               "eval_*.json")))]
             for tag, _, _ in MODELS for t in TIMES}
    bases = {(tag, t): [e for e in (load(p) for p in sorted(glob.glob(str(RUNS / tag / f"t{t}" /
                                                                          "baseline_o*.json"))))
                        if "mps" in e and "infid_sampled" in e and e["Tq"] > 0]
             for tag, _, _ in MODELS for t in TIMES}
    x_snap = lambda e: per_site(e["infid_sampled"] if "infid_sampled" in e else e["infid_total"])
    x_base = lambda e: per_site(e["infid_sampled"])
    grid = [10 ** (-1 - 0.05 * i) for i in range(181)]
    decades = (1e-3, 1e-4, 1e-5, 1e-6)
    say(f"SNAP circuits: {sum(len(v) for v in evals.values())}")
    for tag, hx, _ in MODELS:
        say(f"== h_x = {hx}")
        allr, rig_s, rig_b, trot, brot, rrat = [], [], [], [], [], []
        for t in TIMES:
            env = series(tag, hx, t)
            r = [(interp(env["baseline"], c) / interp(env["snap"], c), c) for c in grid
                 if interp(env["snap"], c) and interp(env["baseline"], c)]
            allr += [(a, c, t) for a, c in r]
            r3 = interp(env["baseline"], 1e-3) / interp(env["snap"], 1e-3)
            say(f"t={t}: SNAP reaches eps {env['snap'][0][0]:.1e}; ratio at 1e-3 {r3:.2f}; "
                f"min {min(r)[0]:.2f} at {min(r)[1]:.0e}, max {max(r)[0]:.2f} at {max(r)[1]:.0e}")
            for c in decades:
                s, b, g = interp(env["snap"], c), interp(env["baseline"], c), interp(env["rig"], c)
                if s and g:
                    rig_s.append(g / s)
                if b and g:
                    rig_b.append(g / b)
            for c in (1e-2, 1e-3, 1e-4, 1e-5):
                cs = [e for e in evals[(tag, t)] if x_snap(e) <= c]
                cb = [e for e in bases[(tag, t)] if x_base(e) <= c]
                if cs and cb:
                    s = min(cs, key=lambda e: e["Tq"])
                    b = min(cb, key=lambda e: e["Tq"])
                    trot.append(s["T_per_rotation"])
                    brot.append(b["T_per_rotation"])
                    rrat.append(s["rotations_per_qubit"] / b["rotations_per_qubit"])
            cl = [e["n_free"] / (e["n_free"] + e["n_t"] + e["n_rus"] + e["n_kn"]) for e in evals[(tag, t)]
                  if e["budget"] == 1e-3]
            say(f"   Clifford share at budget 1e-3: median {st.median(cl):.2f} over {len(cl)} depths")
        say(f"ratio over all eps and t: {min(allr)[0]:.2f} (t={min(allr)[2]}, {min(allr)[1]:.0e}) to "
            f"{max(allr)[0]:.2f} (t={max(allr)[2]}, {max(allr)[1]:.0e})")
        say(f"rigorous/SNAP, eps 1e-3..1e-6: {min(rig_s):.1f} to {max(rig_s):.1f}; "
            f"rigorous/PF: {min(rig_b):.1f} to {max(rig_b):.1f}")
        say(f"T per rotation, eps 1e-2..1e-5: SNAP {min(trot):.2f} to {max(trot):.2f}, "
            f"PF {min(brot):.2f} to {max(brot):.2f}; SNAP/PF rotations {min(rrat):.2f} to {max(rrat):.2f}")
    ev = [e for v in evals.values() for e in v]
    kn = lambda e: e["n_kn"] / (e["n_free"] + e["n_t"] + e["n_rus"] + e["n_kn"])
    loose = [e for e in ev if e["budget"] >= 1e-4]
    tight = [e for e in ev if e["budget"] < 1e-4]
    say(f"unsnapped: {sum(kn(e) > 0 for e in loose)} of {len(loose)} circuits at budget >= 1e-4; "
        f"{sum(kn(e) > 0 for e in tight)} of {len(tight)} below, max share {max(map(kn, tight), default=0):.2f}")
    ho = [e["c_site"] / e["C_train_real"] for e in ev if e.get("C_train_real")]
    say(f"held-out/training at N = 100: median {st.median(ho):.3f}, range {min(ho):.3f} to {max(ho):.3f}")
    bs = sorted(per_site(e["infid_total"]) / per_site(e["infid"]) for e in ev if e["infid"] < 1)
    say(f"bound/success for SNAP: median {st.median(bs):.3f}, 90th percentile {bs[int(0.9 * len(bs))]:.3f}; "
        f"drawn: {sum('infid_sampled' in e for e in ev)} of {len(ev)}")
    say(f"end-gate share of T: max {max(e['T_edges_per_qubit'] / e['Tq'] for e in ev if e['Tq'] > 0):.3f}")
    bd = sorted(x_base(e) / per_site(e["infid"]) for v in bases.values() for e in v if e["infid"] < 1)
    say(f"PF drawn/success: median {st.median(bd):.3f}")
    for tag, hx, _ in MODELS:
        env = series(tag, hx, 4)
        s, b = interp(env["snap"], 3e-4), interp(env["baseline"], 3e-4)
        say(f"whole chain h_x={hx} t=4 eps=3e-4: SNAP {s * N:.3g}, PF {b * N:.3g}")
    (OUT / "te_text_numbers.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    import sys
    if "--refresh" in sys.argv or not RUNS.exists():
        refresh_snapshot()
    ratios = fig_tradeoff()
    fig_floors()
    fig_time()
    cells = table()
    with (OUT / "fig_te_data.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "h_x", "t", "series", "x", "y"])
        w.writerows(DATA)
    with (OUT / "fig_te_sources.txt").open("w") as fh:
        fh.write(f"# inputs of make_te_figures.py, run {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        for p in sorted(SOURCES):
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))}  {p}\n")
    for hx, t, r in ratios:
        print(f"tradeoff arrow h_x={hx} t={t}: x{r:.2f} at eps = {ARROW_EPS:.0e}")
    for (hx, t, c), v in sorted(cells.items()):
        print(f"table h_x={hx} t={t} eps={c:.0e}: {'-' if v is None else f'{v:.2f}'}")
    for tag, hx, _ in MODELS:
        for (key, c), pts in sorted(per_time(tag, hx).items()):
            print(f"per unit time h_x={hx} {key} eps={c:.0e}: " + ", ".join(f"t={t}:{y:.1f}" for t, y in pts))
    print(f"{len(SOURCES)} input files listed in fig_te_sources.txt")
    text_numbers()


if __name__ == "__main__":
    main()
