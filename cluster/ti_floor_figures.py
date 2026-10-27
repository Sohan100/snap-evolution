"""Figures of the infinite-chain training ladders (snapevo.imps_train): floors against depth, and generalization.

    python cluster/ti_floor_figures.py [--out runs/figures/ti]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0. Every cost is per site,
c = 1 - mean_k |lambda0_k| on the 2-site cell (Gibbs and Cincio's C3 per cell is about 2c). Reads
runs/<tag>/t<T>/imps_train{,_n8,_lbfgs,_n8_lbfgs}/ladder.json and, for depths still training, their checkpoints.

    ti_floors   per field (rows) and t (columns): the test cost (100 random chi = 2 iMPS inputs, solid) and the
                training cost (dashed) of each finished depth, for Newton on 4 and 8 training inputs and L-BFGS on
                4; open markers are depths still training (best training cost so far); gray dotted is first-order
                Trotter at the same depth (the "trotter" start's initial cost, training inputs); the horizontal
                line is the ladder target
    ti_generalization   test / training cost against depth, Newton on 4 and on 8 training inputs
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.lines import Line2D                                         # noqa: E402
from matplotlib.ticker import MaxNLocator                                   # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIELDS = (("1.6", "xxz_d0.25_hx1.6", r"$h_x = 1.6$"), ("0", "xxz_d0.25_hx0", r"$h_x = 0$"))
TIMES = (1, 2, 3, 4)
ARMS = (("imps_train", "Newton, 4 inputs", "#2a78d6"),          # categorical slots 1-4 of the reference palette
        ("imps_train_n8", "Newton, 8 inputs", "#eb6834"),           # (lines only, the adjacent-pair case)
        ("imps_train_lbfgs", "L-BFGS, 4 inputs", "#1baf7a"),
        ("imps_train_n8_lbfgs", "L-BFGS, 8 inputs", "#eda100"))
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(tag: str, t: int, arm: str):
    d = REPO / "runs" / tag / f"t{t}" / arm
    if not (d / "ladder.json").exists():
        return None
    doc = json.loads((d / "ladder.json").read_text())
    res = sorted(doc["results"], key=lambda r: r["D"])
    done = {r["D"] for r in res}
    running = {}
    target = doc.get("target") or 3.3e-7
    if any(r["C_train"] <= target for r in res):             # the ladder has stopped: its partial depths are abandoned
        return {"res": res, "running": running, "target": target}
    for ck in glob.glob(str(d / "ckpt" / "D*_*.json")):
        D = int(Path(ck).name.split("_")[0][1:])
        if D not in done and D > max(done, default=0):
            c = json.loads(Path(ck).read_text())["c"]
            running[D] = min(c, running.get(D, c))
    return {"res": res, "running": running, "target": target}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for s in ax.spines.values():
        s.set_color(GRID)


def floors(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharey=True, facecolor=SURFACE)
    target = None
    for i, (_f, tag, fname) in enumerate(FIELDS):
        for j, t in enumerate(TIMES):
            ax = axes[i, j]
            style(ax)
            trot = {}
            for arm, _lab, col in ARMS:
                L = load(tag, t, arm)
                if L is None:
                    continue
                target = target or L["target"]
                D = [r["D"] for r in L["res"]]
                ax.plot(D, [r["C_val"] for r in L["res"]], "-o", color=col, lw=2, ms=4.5, mec=SURFACE, mew=0.8)
                ax.plot(D, [r["C_train"] for r in L["res"]], "--", color=col, lw=1.1)
                for Dr, c in L["running"].items():
                    ax.plot([Dr], [c], "o", ms=7, mfc="none", mec=col, mew=1.6)
                for r in L["res"]:
                    if "trotter" in r["runs"]:
                        trot.setdefault(r["D"], r["runs"]["trotter"]["c_start"])
            if trot:
                Ds = sorted(trot)
                ax.plot(Ds, [trot[d] for d in Ds], ":", color=MUTED, lw=1.4)
            ax.axhline(target or 3.3e-7, color=MUTED, lw=0.9, ls=(0, (6, 3)))
            ax.set_yscale("log")
            ax.set_ylim(1e-11, 0.5)
            if i == 0:
                ax.set_title(f"t = {t}", color=TEXT, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{fname}\ncost per site", color=TEXT, fontsize=9)
            if i == 1:
                ax.set_xlabel("depth D (sublayers)", color=TEXT, fontsize=9)
    axes[0, 0].annotate(f"target {target or 3.3e-7:.1e}", xy=(0.97, target or 3.3e-7), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", ha="right", va="bottom", color=MUTED, fontsize=7.5)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4.5, label=lab) for _a, lab, c in ARMS]
    handles += [Line2D([], [], color=MUTED, lw=2, label="test (100 inputs)"),
                Line2D([], [], color=MUTED, lw=1.1, ls="--", label="training"),
                Line2D([], [], color=MUTED, lw=0, marker="o", ms=7, mfc="none", mew=1.6, label="depth still training"),
                Line2D([], [], color=MUTED, lw=1.4, ls=":", label="1st-order Trotter, same depth")]
    fig.suptitle("Infinite-chain training floors, XXZ $\\Delta = 0.25$ (2-site cell, $\\chi = 2$ random iMPS inputs)",
                 color=TEXT, fontsize=11, y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=8, frameon=False, fontsize=7.5, labelcolor=TEXT,
               bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ti_floors.{ext}", dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def generalization(out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4), sharey=True, facecolor=SURFACE)
    for i, (_f, tag, fname) in enumerate(FIELDS):
        for j, t in enumerate(TIMES):
            ax = axes[i, j]
            style(ax)
            for arm, _lab, col in ARMS[:2]:
                L = load(tag, t, arm)
                if L is None:
                    continue
                D = [r["D"] for r in L["res"]]
                ax.plot(D, [r["C_val"] / r["C_train"] for r in L["res"]], "-o", color=col, lw=2, ms=4.5, mec=SURFACE,
                        mew=0.8)
            ax.axhline(1, color=MUTED, lw=0.9)
            ax.set_ylim(0.8, 4)
            if i == 0:
                ax.set_title(f"t = {t}", color=TEXT, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{fname}\ntest / training", color=TEXT, fontsize=9)
            if i == 1:
                ax.set_xlabel("depth D (sublayers)", color=TEXT, fontsize=9)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=4.5, label=lab) for _a, lab, c in ARMS[:2]]
    fig.suptitle("Generalization: test cost over training cost at each finished depth", color=TEXT, fontsize=11,
                 y=0.995)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5, labelcolor=TEXT,
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"ti_generalization.{ext}", dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "runs" / "figures" / "ti"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    floors(out)
    generalization(out)
    print(f"wrote {out}/ti_floors.(png|pdf), {out}/ti_generalization.(png|pdf)")


if __name__ == "__main__":
    main()
