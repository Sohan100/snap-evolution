"""T counts against per-site error on the infinite chain, every arm translation invariant (Sohan, 24 Oct 2026).

    python cluster/ti_tcount.py run  [--times 1] [--workers 30]    # baselines: runs/ti/imps_chi2/<tag>_t<T>.json
    python cluster/ti_tcount.py plot [--times 1] [--out runs/figures/ti]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0.

Arms, all evaluated the same way: the per-site cost on the 100 test inputs of the iMPS ladders (random chi = 2 iMPS on
the 2-site cell, imps_train.make_input("test", k), references by imps_train.ref_path), with every synthesized
rotation falling back on its own with its probability (imps_snap.fallback_terms, the expectation over independent
draws; a draw itself would break translation invariance), and on the success branch:
  - SNAP: the compiles of the ladders against the infinite-chain cost (imps_snap.ti_point;
    runs/<tag>/t<T>/imps_snap[_n8]/eval_*.json), with 4 and 8 training inputs;
  - Gibbs and Cincio's procedure (arXiv:2506.02105, Sec. III.4): every rotation of a trained ladder at one KN
    precision (cluster/imps_snap.py uniform). Exactly theirs on the L-BFGS ladders (runs/<tag>/t<T>/imps_uniform_lbfgs/,
    filled diamonds), and the same synthesis on the Newton ladders (imps_uniform/, open diamonds);
  - Trotter + KN: cluster/ti_trotter.py's scan, chosen and scored on the infinite chain only
    (runs/ti/imps_trotter/<tag>_t<T>.json). ``run`` below rescores the older points that baseline.scan's ring screen
    picked (runs/ti/imps_chi2/); they are not plotted, since a comparison method must be TI from end to end.
T per qubit and per rotation are the records' (no edges on the infinite chain). Points whose realized training
cost is over their budget (SNAP's over_budget) are left out.

plot: ti_tcount_qubit (T per qubit, each arm's lower envelope) and ti_tcount_rotation, against the error with
fallbacks, per field (columns) and t (rows); and ti_tcount_optimizer_qubit / _rotation, SNAP on the ladders trained
by Newton (the protocol) against SNAP on the L-BFGS ladders (the comparison arm), 4 and 8 training inputs, every
step after training the same, so a difference in T is the training optimizer's.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy loads
    os.environ.setdefault(_v, "1")

import argparse                                                             # noqa: E402
import concurrent.futures as cf                                             # noqa: E402
import json                                                                 # noqa: E402
import math                                                                 # noqa: E402
import multiprocessing as mp                                                # noqa: E402
import sys                                                                  # noqa: E402
import time                                                                 # noqa: E402
from pathlib import Path                                                    # noqa: E402

import numpy as np                                                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cluster"))
OUT = REPO / "runs" / "ti" / "imps_chi2"
N_TEST = 100

_W: dict = {}


def _init(field: str, t: float) -> None:
    from snapevo.hamiltonian import XXZ
    _W.clear()
    _W.update(model=XXZ(0.25, float(field)), t=float(t), io={})


def _io(k: int):
    from snapevo import imps as im
    from snapevo import imps_train as it
    if k not in _W["io"]:
        phi = it.make_input("test", k)
        path = it.ref_path(_W["model"], _W["t"], "test", k)
        psi, _ = im.reference_state(_W["model"], _W["t"], phi, it.DT, it.CHI_MAX, it.CUTOFF, path)
        _W["io"][k] = (phi, psi)
    return _W["io"][k]


def _task(args):
    from snapevo.imps_snap import fallback_terms
    i, flayers, k = args
    phi, psi = _io(k)
    return i, k, fallback_terms(flayers, phi, psi)


def _baseline_layers(p: dict, model, t: float) -> list:
    """imps_snap.fallback_layers of a Trotter + KN point (its KN records by angle, as fallbacks.sample_baseline)."""
    import ti_ring_eval as tr
    from snapevo.baseline import trotter_circuit
    from snapevo.imps_snap import fallback_layers
    from snapevo.synthesis import synthesize
    rec = json.loads(Path(p["file"]).read_text())
    circ = trotter_circuit(model, float(t), int(rec["order"]), int(rec["steps"]))
    real = tr.circuit(p)
    synth = {k: (v if "U0" in v else synthesize(float(k), float(rec["eps"]), unitaries=True))
             for k, v in rec["synth"].items()}

    def rec_for(l, s):
        return synth.get(f"{float(round(circ.layers[l][1][s], 15)):.15f}")

    return fallback_layers(real.cell.name, real.layers, rec_for)


def run(times, workers: int) -> None:
    import ti_ring_eval as tr
    from snapevo.hamiltonian import XXZ
    OUT.mkdir(parents=True, exist_ok=True)
    for t in times:
        for f, tag in tr.TAGS.items():
            model = XXZ(0.25, float(f))
            pts = [p for p in tr._baselines(f, t) if p["eval_file"]]
            t0 = time.time()
            fls = [_baseline_layers(p, model, t) for p in pts]
            print(f"{tag} t={t}: {len(pts)} product formulas, branch unitaries ready in {time.time() - t0:.0f}s",
                  flush=True)
            tasks = [(i, fl, k) for i, fl in enumerate(fls) for k in range(N_TEST)]
            res = [[None] * N_TEST for _ in pts]
            with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                        initargs=(f, t)) as ex:
                for i, k, r in ex.map(_task, tasks, chunksize=4):
                    res[i][k] = r
            out = []
            for p, rr in zip(pts, res):
                rec = json.loads(Path(p["eval_file"]).read_text())
                la = np.array([r["log_abs"] for r in rr])
                lf = la + np.array([r["kn"] + r["lib"] for r in rr])
                c_s, c_f = -np.expm1(la), -np.expm1(lf)
                out.append({"method": "baseline", "stem": p["stem"], "D": p["D"], "order": p["order"],
                            "steps": p["steps"], "eps": p["eps"],
                            "c": float(c_s.mean()), "se": float(c_s.std(ddof=1) / math.sqrt(N_TEST)),
                            "c_fallbacks": float(c_f.mean()), "c_fallbacks_se": float(c_f.std(ddof=1) / math.sqrt(N_TEST)),
                            "fallbacks_per_cell": float(rr[0]["n_kn"]), "n": N_TEST, "c_open100": rec.get("c_site"),
                            "T_per_qubit": float(rec["Tq"]) - float(rec.get("T_edges_per_qubit") or 0.0),
                            "T_per_rotation": rec.get("T_per_rotation"),
                            "rotations_per_qubit": rec.get("rotations_per_qubit")})
            doc = {"model": tag, "t": t, "metric": "per site on the 100 random chi = 2 iMPS test inputs: c_fallbacks "
                   "with independent fallbacks (imps_snap.fallback_terms), c on the success branch", "points": out}
            (OUT / f"{tag}_t{t}.json").write_text(json.dumps(doc, indent=1))
            print(f"{tag} t={t}: {len(out)} points in {time.time() - t0:.0f}s -> {OUT / f'{tag}_t{t}.json'}",
                  flush=True)


# --------------------------------------------------------------------------- figures

ARMS = (("snap", "SNAP, 4 inputs", "#2a78d6", "o", True),          # categorical slots 1-4 of the reference palette
        ("snap_n8", "SNAP, 8 inputs", "#1baf7a", "o", True),
        ("uniform_lbfgs", "Gibbs-Cincio (L-BFGS training, uniform precision), 4 inputs", "#eda100", "D", True),
        ("uniform", "Gibbs-Cincio synthesis on the Newton ladders, 4 inputs", "#eda100", "D", False),
        ("baseline", "Trotter + KN", "#eb6834", "s", False))
# colour: optimizer; marker: inputs. L-BFGS open and dashed over Newton filled and solid: at t = 1 the two land on
# the same points, and solid-over-solid hid the Newton curves.
COMPARE = (("snap", "SNAP, Newton-trained, 4 inputs", "#2a78d6", "o", True),
           ("snap_n8", "SNAP, Newton-trained, 8 inputs", "#2a78d6", "^", True),
           ("snap_lbfgs", "SNAP, L-BFGS-trained, 4 inputs", "#eb6834", "o", False, "--"),
           ("snap_n8_lbfgs", "SNAP, L-BFGS-trained, 8 inputs", "#eb6834", "^", False, "--"))
SUBS = {"snap": "imps_snap", "snap_n8": "imps_snap_n8", "uniform": "imps_uniform",
        "snap_lbfgs": "imps_snap_lbfgs", "snap_n8_lbfgs": "imps_snap_n8_lbfgs", "uniform_lbfgs": "imps_uniform_lbfgs"}
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FIELD_NAME = {"xxz_d0.25_hx1.6": r"$h_x = 1.6$", "xxz_d0.25_hx0": r"$h_x = 0$"}


def envelope(pts):
    """Lower envelope of (c, T): each kept point beats every cheaper one in c."""
    env, best = [], math.inf
    for c, T in sorted(pts, key=lambda q: (q[1], q[0])):
        if c < best:
            env.append((c, T))
            best = c
    return sorted(env)


def load(tag: str, t: int) -> dict:
    """{arm: [{c, T_per_qubit, T_per_rotation}]} with c the error with fallbacks."""
    d = REPO / "runs" / tag / f"t{t}"
    out = {a: [] for a in list(SUBS) + ["baseline", "gridsynth"]}
    for arm, sub in SUBS.items():
        for f in sorted((d / sub).glob("eval_*.json")):
            e = json.loads(f.read_text())
            if e.get("over_budget") or "c_fallbacks" not in e:
                continue
            out[arm].append({"c": e["c_fallbacks"], "c_s": e["c"], "T_per_qubit": e["Tq"],
                             "T_per_rotation": e["T_per_rotation"]})
    path = REPO / "runs" / "ti" / "imps_trotter" / f"{tag}_t{t}.json"     # cluster/ti_trotter.py: the TI-native scan
    if path.exists():
        for p in json.loads(path.read_text())["points"]:
            if p["method"] == "baseline" and "c_fallbacks" in p:
                out["baseline"].append({"c": p["c_fallbacks"], "c_s": p["c"], "T_per_qubit": p["T_per_qubit"],
                                        "T_per_rotation": p["T_per_rotation"]})
    path = REPO / "runs" / "ti" / "imps_trotter_gs" / f"{tag}_t{t}.json"  # cluster/ti_trotter_gs.py: Trotter + GridSynth
    if path.exists():
        for p in json.loads(path.read_text())["points"]:
            out["gridsynth"].append({"c": p["c_fallbacks"], "c_s": p["c"], "T_per_qubit": p["T_per_qubit"],
                                     "T_per_rotation": p["T_per_rotation"]})
    return out


def plot(times, out: Path) -> None:
    _plot(times, out, ARMS, "ti_tcount")
    _plot(times, out, COMPARE, "ti_tcount_optimizer")


def ratios(t: int, tag: str, a: str = "snap_lbfgs", b: str = "snap", levels=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6)) -> dict:
    """T per qubit of arm a over arm b at equal error (cheapest point of each at or below each level)."""
    data = load(tag, t)
    out = {}
    for c in levels:
        ta = [p["T_per_qubit"] for p in data[a] if p["c"] <= c]
        tb = [p["T_per_qubit"] for p in data[b] if p["c"] <= c]
        if ta and tb:
            out[c] = (min(ta), min(tb), min(ta) / min(tb))
    return out


def _plot(times, out: Path, arms, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out.mkdir(parents=True, exist_ok=True)
    tags = list(FIELD_NAME)
    for key, ylab, fname, logy in (("T_per_qubit", "T gates per qubit", f"{stem}_qubit", True),
                                    ("T_per_rotation", "T gates per rotation", f"{stem}_rotation", False)):
        fig, axes = plt.subplots(len(times), 2, figsize=(10.5, 4.4 * len(times)), squeeze=False, facecolor=SURFACE)
        for i, t in enumerate(times):
            for j, tag in enumerate(tags):
                ax = axes[i][j]
                ax.set_facecolor(SURFACE)
                data = load(tag, t)
                for arm, _lab, col, mk, filled, *ls in arms:
                    pts = [p for p in data[arm] if p.get(key) is not None and p[key] > 0 and p["c"] > 0]
                    if not pts:
                        continue
                    ax.scatter([p["c"] for p in pts], [p[key] for p in pts], s=26 if filled else 46, marker=mk,
                               facecolor=col if filled else "none", edgecolor=SURFACE if filled else col,
                               lw=0.6 if filled else 1.2, zorder=3, alpha=0.9)
                    if key == "T_per_qubit":
                        e = envelope([(p["c"], p[key]) for p in pts])
                        ax.step([c for c, _ in e], [T for _, T in e], where="post", color=col,
                                lw=2 if not ls else 1.4, ls=ls[0] if ls else "-", zorder=2)
                ax.set_xscale("log")
                if logy:
                    ax.set_yscale("log")
                ax.grid(True, which="major", color=GRID, lw=0.6)
                ax.tick_params(which="both", colors=MUTED, labelsize=8)
                for s_ in ax.spines.values():
                    s_.set_color(GRID)
                ax.set_title(f"{FIELD_NAME[tag]}, t = {t}", color=TEXT, fontsize=10)
                ax.set_xlabel("error per site on the infinite chain, with fallbacks (100 random $\\chi = 2$ iMPS "
                              "inputs)", color=TEXT, fontsize=8)
                if j == 0:
                    ax.set_ylabel(ylab, color=TEXT, fontsize=9)
        handles = [Line2D([], [], color=col, lw=(2 if not ls else 1.4) if key == "T_per_qubit" else 0,
                          ls=ls[0] if ls else "-", marker=mk, ms=6 if filled else 7,
                          mfc=col if filled else "none", mec=SURFACE if filled else col, label=lab)
                   for _a, lab, col, mk, filled, *ls in arms]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, labelcolor=TEXT,
                   bbox_to_anchor=(0.5, 0.995))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{fname}.{ext}", dpi=170, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
    print(f"wrote {out}/{stem}_qubit.(png|pdf), {out}/{stem}_rotation.(png|pdf)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "plot"))
    ap.add_argument("--times", default="1")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--out", default=str(REPO / "runs" / "figures" / "ti"))
    a = ap.parse_args()
    times = [int(x) for x in a.times.split(",")]
    if a.cmd == "run":
        run(times, a.workers)
    else:
        plot(times, Path(a.out))


if __name__ == "__main__":
    main()
