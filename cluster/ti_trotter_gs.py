"""Trotter + GridSynth on the infinite chain: the product formulas of cluster/ti_trotter.py with every rotation
synthesized by GridSynth (Ross-Selinger, pygridsynth) instead of KN, scored on the TI system only.

    python cluster/ti_trotter_gs.py FIELD T [--workers 30]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = FIELD (1.6 or 0). Output runs/ti/imps_trotter_gs/<tag>_t<T>.json.

The standard pipeline most readers know (Isaac, 26 Oct 2026: "Trotter + GridSynth, which is what most people would know
as the standard instead of KN"). Same formulas and precisions as the Trotter + KN scan: every (order, steps, eps) of
runs/ti/imps_trotter/<tag>_t<T>.json, with GridSynth's operator-norm tolerance eps / 2 (KN's eps is a diamond-norm
budget, about twice the operator-norm distance), up to a global phase. Each non-Clifford+T rotation is replaced by
GridSynth's exact Clifford+T unitary, off-diagonal part included (fallbacks.slot_gates places a 2x2 unitary in any Rz
slot), and the circuit is scored on the ladders' 100 test inputs by imps_snap.fallback_terms with no fallback instances:
GridSynth is deterministic. T per qubit counts GridSynth's T gates (an odd multiple of pi/4 costs 1, a multiple of pi/2
costs 0, as baseline.realize_baseline); rotations per qubit as realize_baseline.
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
import warnings                                                             # noqa: E402
from pathlib import Path                                                    # noqa: E402

import numpy as np                                                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cluster"))
import ti_tcount as tt                                                      # noqa: E402
import ti_trotter as ttr                                                    # noqa: E402

OUT = REPO / "runs" / "ti" / "imps_trotter_gs"
SLOT = {"a": 0, "b": 1, "c": 2, "e": 3, "f": 4}


def _gs(args):
    """(theta, eps) -> GridSynth's unitary for Rz(theta) within eps (operator norm, up to phase) and its T count."""
    import mpmath
    from pygridsynth.config import GridsynthConfig
    from pygridsynth.gridsynth import gridsynth
    from snapevo.synthesis import _fast_t_count
    theta, eps = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U = gridsynth(mpmath.mpf(repr(theta)), mpmath.mpf(repr(eps)), cfg=GridsynthConfig(up_to_phase=True))
    M = np.array(U.to_complex_matrix.tolist(), dtype=complex)
    return theta, eps, [[[z.real, z.imag] for z in row] for row in M], int(_fast_t_count(U))


def _mat(rec) -> np.ndarray:
    return np.array([[complex(*z) for z in row] for row in rec], dtype=complex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field")
    ap.add_argument("t", type=float)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()

    from snapevo import fallbacks as fb
    from snapevo.baseline import QUARTER, _is_exact, trotter_circuit
    from snapevo.hamiltonian import XXZ

    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, float(a.field))

    def log(s):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g} trotter+gs] {s}", flush=True)

    src = json.loads((REPO / "runs" / "ti" / "imps_trotter" / f"{model.tag}_t{t}.json").read_text())
    plan = sorted({(int(p["order"]), int(p["steps"]), float(p["eps"])) for p in src["points"]
                   if p["method"] == "baseline"})
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / f"gridsynth_{model.tag}_t{t}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = lambda th, e: f"{th:.15f}|{e:.6e}"                               # noqa: E731
    need = sorted({(float(th), eps / 2) for order, r, eps in plan
                   for _p, ang in trotter_circuit(model, t, order, r).layers for th in ang
                   if not _is_exact(float(th))} - {tuple(map(float, k.split("|"))) for k in cache})
    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=tt._init,
                                initargs=(a.field, float(t))) as ex:
        t0 = time.time()
        for th, e, M, T in ex.map(_gs, need, chunksize=1):
            cache[key(th, e)] = {"U": M, "T": T}
        cache_path.write_text(json.dumps(cache))
        log(f"{len(plan)} formulas, {len(need)} GridSynth syntheses in {time.time() - t0:.0f}s")

        def gate_layers(circ, eps: float | None):
            """fallback_terms layers of circ: exact Rz everywhere (eps None) or GridSynth's at eps / 2."""
            out = []
            for par, ang in circ.layers:
                r = fb._layer_rots(circ.cell.name, np.asarray(ang, dtype=float), lambda s: None)
                U = {}
                if eps is not None:
                    for name, s in SLOT.items():
                        th = float(ang[s])
                        if not _is_exact(th):
                            U[name] = _mat(cache[key(th, eps / 2)]["U"])
                out.append((int(par), fb._gate(circ.cell.name, r, U), []))
            return out

        # the slot decomposition reproduces the circuit's own gates: exact Rz through fallbacks._gate
        circ = trotter_circuit(model, t, plan[0][0], plan[0][1])
        g_cell = [circ.cell.gate(np.asarray(ang, dtype=float)) for _p, ang in circ.layers]
        g_slot = [g for _p, g, _i in gate_layers(circ, None)]
        dev = max(float(np.min([np.abs(x - y * ph).max() for ph in (1, -1, 1j, -1j)])) for x, y in zip(g_cell, g_slot))
        log(f"slot decomposition vs cell gates: max deviation {dev:.1e} (up to a global phase)")
        if dev > 1e-10:
            raise SystemExit("the slot decomposition does not reproduce the cell's gates")

        part = OUT / f"{model.tag}_t{t}.partial.json"
        done = json.loads(part.read_text()) if part.exists() else {}
        recs = []
        t0 = time.time()
        for order, r, eps in plan:
            circ = trotter_circuit(model, t, order, r)
            Tq = rot = 0.0
            for _p, ang in circ.layers:
                for s, th in enumerate(ang):
                    th = float(th)
                    m = circ.cell.multiplicity[s]
                    if abs(np.remainder(th + np.pi, 2 * np.pi) - np.pi) > 1e-12:
                        rot += 0.5 * m
                    if _is_exact(th):
                        Tq += 0.5 * m * (0.0 if round(th / QUARTER) % 2 == 0 else 1.0)
                    else:
                        Tq += 0.5 * m * cache[key(th, eps / 2)]["T"]
            k = f"{order}_{r}_{eps:.1e}"
            if k not in done:
                done[k] = ttr._scores(ex, [gate_layers(circ, eps)], log)[0]
                part.write_text(json.dumps(done))
            sc = done[k]
            c_tr = next(p["c_trotter"] for p in src["points"] if p["method"] == "baseline" and
                        (p["order"], p["steps"], p["eps"]) == (order, r, eps))
            recs.append({"method": "gridsynth", "order": order, "steps": r, "eps_kn": eps, "eps_gridsynth": eps / 2,
                         "c_trotter": c_tr, "depth": circ.depth, "T_per_qubit": Tq, "rotations_per_qubit": rot,
                         "T_per_rotation": Tq / rot if rot else None, "c": sc["c"], "se": sc["se"],
                         "c_fallbacks": sc["c"], "c_fallbacks_se": sc["se"]})
        log(f"scored {len(recs)} formulas in {time.time() - t0:.0f}s")
    doc = {"model": model.tag, "t": t, "system": "infinite chain, 2-site cell",
           "metric": "per site on the 100 random chi = 2 iMPS test inputs; GridSynth is deterministic, so c_fallbacks = c",
           "synthesis": "pygridsynth (Ross-Selinger), operator-norm tolerance eps_kn / 2, up to a global phase",
           "points": recs}
    path = OUT / f"{model.tag}_t{t}.json"
    path.write_text(json.dumps(doc, indent=1))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
