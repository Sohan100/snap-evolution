"""Publication figures for the time-evolution section: SNAP against Trotter + KN, Trotter + GridSynth (and Gibbs-Cincio)
on the infinite chain.

    python cluster/ti_paper_figs.py [--times 1,2,3,4] [--arms snap,baseline,gs] [--out runs/figures/ti/paper]

--arms picks the arms drawn: snap, baseline (Trotter + KN), gs (Trotter + GridSynth), gc (Gibbs-Cincio); the default is
SNAP against both Trotter arms (Sohan, 26 Oct 2026: SNAP against Trotter + KN, then "lets add it to all the plots" for
Trotter + GridSynth, the standard pipeline Isaac asked for).

Models: XXZ chain, Delta = 0.25, J = 1 (spin units), at h_x = 0 and h_x = 1.6. Every number is cluster/ti_tcount.load's:
the squared infidelity per site 1 - F = 1 - mean_k |lambda_k| on the 100 random chi = 2 iMPS test inputs, with
fallbacks, against the expected T count per qubit.

Arms (4 training inputs throughout):
  - SNAP: the Newton-trained ladder compiled by SNAP (imps_snap/).
  - Trotter + KN: product formulas of orders 1, 2, 4 synthesized by KN (cluster/ti_trotter.py).
  - Trotter + GridSynth: the same formulas and precisions, every rotation synthesized by GridSynth (Ross-Selinger),
    deterministic (cluster/ti_trotter_gs.py).
  - Gibbs-Cincio: every rotation synthesized at one precision on the L-BFGS-trained ladder (imps_uniform_lbfgs/); where
    that arm has no points yet, the same synthesis on the Newton-trained ladder (imps_uniform/), which gives the same
    T counts wherever both exist (t = 1, 2).
Each arm is drawn by its Pareto front: the circuits no cheaper circuit of the same arm beats in 1 - F, thinned to
those that improve on the next cheaper kept one by more than THIN (10%; 1 - F has a 3-5% standard error over the test
inputs, and the Gibbs-Cincio arm otherwise stacks its depths at one precision into columns), over 1 - F >= C_MIN
(1e-6, SNAP's floor). The ratio panels give T_arm / T_SNAP at each SNAP front circuit, T_arm the cheapest circuit of
that arm at least as accurate.

Figures: tcount_t<T>.(pdf|png) per time (T per qubit and the ratios, both fields), rotation_t<T>.(pdf|png) per time
(T per rotation), tcount_qubit.(pdf|png) and tcount_rotation.(pdf|png) for all times (one row per time). T per
rotation is the expected T count over the rotation gates of the circuit before compilation, the same count for every
arm: every rotation slot of the trained circuit for SNAP and Gibbs-Cincio (a slot SNAP sets to a Clifford counts, at
0 T), every rotation of the product formula for Trotter + KN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cluster"))
import ti_tcount as tt                                                      # noqa: E402

FIELDS = (("xxz_d0.25_hx0", r"$h_x = 0$"), ("xxz_d0.25_hx1.6", r"$h_x = 1.6$"))
# label, colour, marker: the reference palette's categorical slots 1-3, each with its own marker
STYLE = {"snap": ("SNAP", "#2a78d6", "o"),
         "gc": ("Gibbs–Cincio", "#eb6834", "D"),
         "baseline": ("Trotter + KN", "#1baf7a", "s"),
         "gs": ("Trotter + GridSynth", "#eda100", "^")}
ORDER = ("gs", "baseline", "gc", "snap")                                   # drawing order: SNAP on top
SHOW = ["snap", "baseline", "gs"]                                          # the arms drawn (--arms)
LEGEND = ("snap", "baseline", "gs", "gc")
XLABEL = r"Squared infidelity per site, $1 - F$"
THIN = 0.9
C_MIN = 1e-6          # every arm drawn over 1 - F >= 1e-6 only (SNAP's floor)


def _rc():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2, "ytick.minor.size": 2,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.6, "lines.markersize": 5,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def arms(tag: str, t: int) -> dict:
    """{arm: Pareto front [(1 - F, T per qubit, T per rotation)]} for snap, gc, baseline and gs."""
    d = tt.load(tag, t)
    gc = d["uniform_lbfgs"] or d["uniform"]
    out = {}
    for name, pts in (("snap", d["snap"]), ("gc", gc), ("baseline", d["baseline"]), ("gs", d["gridsynth"])):
        pts = [p for p in pts if p["c"] >= C_MIN and p["T_per_qubit"] > 0]
        front, last = set(), float("inf")
        for c, T in sorted(tt.envelope([(p["c"], p["T_per_qubit"]) for p in pts]), key=lambda q: q[1]):
            if c < THIN * last:
                front.add((c, T))
                last = c
        rows = sorted({(p["c"], p["T_per_qubit"], p["T_per_rotation"]) for p in pts
                       if (p["c"], p["T_per_qubit"]) in front})
        out[name] = rows
    out["gc_source"] = "L-BFGS ladder" if d["uniform_lbfgs"] else "Newton ladder"
    return out


def ratio(snap: list, other: list) -> list:
    """[(1 - F of the SNAP circuit, T_other / T_SNAP)]: the cheapest other circuit at least as accurate."""
    out = []
    for c, T, _ in snap:
        ok = [To for co, To, _ in other if co <= c]
        if ok:
            out.append((c, min(ok) / T))
    return out


def _series(ax, x, y, name, lines: bool = True, **kw):
    lab, col, mk = STYLE[name]
    ax.plot(x, y, color=col, marker=mk, mfc="white", mec=col, mew=1.2, ms=4.5 if mk == "D" else 5, label=lab,
            ls="-" if lines else "none", zorder=3 if name == "snap" else 2, **kw)


def _draw(ax, a: dict, key: int, lines: bool = True) -> None:
    for name in ORDER:
        if name in SHOW and a[name]:
            _series(ax, [r[0] for r in a[name]], [r[key] for r in a[name]], name, lines)


def _legend(fig, plt, y: float, lines: bool = True) -> None:
    handles = [plt.Line2D([], [], color=STYLE[n][1], marker=STYLE[n][2], mfc="white", mec=STYLE[n][1], mew=1.2,
                          ls="-" if lines else "none", label=STYLE[n][0]) for n in LEGEND if n in SHOW]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=2.2 if lines else 1.0, columnspacing=2.0)


def _ratio_label() -> str:
    others = [n for n in ("gc", "baseline", "gs") if n in SHOW]
    name = r"\mathrm{Trotter}" if others == ["baseline"] else (r"\mathrm{GC}" if others == ["gc"] else r"\mathrm{method}")
    return rf"$T_{{{name}}}\,/\,T_{{\mathrm{{SNAP}}}}$"


def _box(ax):
    ax.set_xscale("log")
    ax.set_xlim(left=0.6 * C_MIN)
    ax.minorticks_on()


def per_time(t: int, out: Path) -> bool:
    plt = _rc()
    from matplotlib.ticker import LogLocator, NullFormatter
    data = [arms(tag, t) for tag, _ in FIELDS]
    if not any(a["snap"] for a in data):
        return False
    fig = plt.figure(figsize=(7.0, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=(2.5, 1.0), hspace=0.12, wspace=0.26,
                          left=0.085, right=0.985, top=0.83, bottom=0.11)
    for j, ((tag, fname), a) in enumerate(zip(FIELDS, data)):
        ax = fig.add_subplot(gs[0, j])
        _draw(ax, a, 1)
        _box(ax)
        ax.set_yscale("log")
        ax.set_title(f"({chr(97 + j)}) {fname}")
        if j == 0:
            ax.set_ylabel(r"Expected $T$ gates per qubit")
        axr = fig.add_subplot(gs[1, j], sharex=ax)
        ax.tick_params(labelbottom=False)
        axr.axhline(1.0, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
        for name in [n for n in ("gs", "baseline", "gc") if n in SHOW]:
            r = ratio(a["snap"], a[name])
            if r:
                _series(axr, [x for x, _ in r], [y for _, y in r], name)
        _box(axr)
        axr.set_yscale("log")
        axr.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
        axr.yaxis.set_minor_formatter(NullFormatter())
        axr.set_xlabel(XLABEL)
        if j == 0:
            axr.set_ylabel(_ratio_label())
    _legend(fig, plt, 0.935)
    fig.suptitle(rf"XXZ chain, $t = {t}$", y=0.985, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"tcount_t{t}.{ext}")
    plt.close(fig)
    return True


def rotation_time(t: int, out: Path) -> bool:
    """T per rotation of each arm's front circuits at time t, both fields side by side."""
    plt = _rc()
    from matplotlib.ticker import MaxNLocator
    data = [arms(tag, t) for tag, _ in FIELDS]
    if not any(a["snap"] for a in data):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.74, bottom=0.15, wspace=0.22)
    for j, ((tag, fname), a) in enumerate(zip(FIELDS, data)):
        ax = axes[j]
        _draw(ax, a, 2, lines=False)
        _box(ax)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.set_title(f"({chr(97 + j)}) {fname}")
        ax.set_xlabel(XLABEL)
        if j == 0:
            ax.set_ylabel(r"Expected $T$ gates per rotation")
    _legend(fig, plt, 0.905, lines=False)
    fig.suptitle(rf"XXZ chain, $t = {t}$", y=0.985, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"rotation_t{t}.{ext}")
    plt.close(fig)
    return True


def all_times(times, out: Path, key: int, ylabel: str, stem: str, logy: bool, lines: bool = True) -> None:
    plt = _rc()
    from matplotlib.ticker import MaxNLocator
    rows = [t for t in times if any(arms(tag, t)["snap"] for tag, _ in FIELDS)]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.0, 0.6 + 2.35 * len(rows)), squeeze=False)
    for i, t in enumerate(rows):
        for j, (tag, fname) in enumerate(FIELDS):
            ax = axes[i][j]
            _draw(ax, arms(tag, t), key, lines)
            _box(ax)
            ax.set_title(rf"({chr(97 + 2 * i + j)}) {fname}, $t = {t}$", fontsize=9.5, pad=4)
            if logy:
                ax.set_yscale("log")
            else:
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if i == len(rows) - 1:
                ax.set_xlabel(XLABEL)
            if j == 0:
                ax.set_ylabel(ylabel)
    _legend(fig, plt, 1.0, lines)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / (0.6 + 2.35 * len(rows))), h_pad=0.8, w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", default="1,2,3,4")
    ap.add_argument("--arms", default="snap,baseline,gs",
                    help="any of snap, baseline (Trotter + KN), gs (Trotter + GridSynth), gc (Gibbs-Cincio)")
    ap.add_argument("--out", default=str(REPO / "runs" / "figures" / "ti" / "paper"))
    a = ap.parse_args()
    SHOW[:] = [x for x in a.arms.split(",") if x]
    times = [int(x) for x in a.times.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in times:
        if per_time(t, out) and "gc" in SHOW:
            src = {fname: arms(tag, t)["gc_source"] for tag, fname in FIELDS}
            print(f"t = {t}: Gibbs-Cincio from the {src}")
        rotation_time(t, out)
    all_times(times, out, 1, r"Expected $T$ gates per qubit", "tcount_qubit", True)
    all_times(times, out, 2, r"Expected $T$ gates per rotation", "tcount_rotation", False, lines=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
