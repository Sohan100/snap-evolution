"""Trotter + KN on the infinite chain: product formulas scanned, synthesized and scored on the TI system only.

    python cluster/ti_trotter.py FIELD T [--workers 30] [--c-floor 1e-7] [--r-max 512]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = FIELD (1.6 or 0). Output runs/ti/imps_trotter/<tag>_t<T>.json.

Every comparison method is built and evaluated on the translation-invariant system and nothing else (Sohan, 24 Oct
2026, an imperative): this replaces the product formulas picked by baseline.scan's 10-site ring screen. The rule is
baseline.scan's with every number on the infinite chain: the per-site cost on the ladders' 100 test inputs (random
chi = 2 iMPS on the 2-site cell, imps_train.make_input("test", k), references by imps_train.ref_path).

  - For each order 1, 2, 4 (baseline.trotter_circuit: even bonds, odd bonds and the field, Suzuki), r grows (1, 2, 3,
    4, then x1.5) until the Trotter-only cost is below c_floor (order 1: below a tenth of one order-2 step's) or r
    reaches r_max.
  - Each (order, r) is synthesized (baseline.realize_baseline: every distinct angle once with KN) at the first grid
    precision at or below eps* = sqrt(c_trotter / 3) / (0.45 R), R the rotations per qubit, and the two above it (r = 1:
    every looser one too).
  - Each realized circuit is scored on the test inputs on the success branch (c) and with every KN rotation falling
    back on its own (c_fallbacks; imps_snap.fallback_terms, the expectation that keeps translation invariance).
T per qubit and per rotation are realize_baseline's (no edges on the infinite chain).
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
import ti_tcount as tt                                                      # noqa: E402

OUT = REPO / "runs" / "ti" / "imps_trotter"
N_TEST = 100


def _synth(args):
    from snapevo.synthesis import synthesize
    theta, eps = args
    synthesize(float(theta), float(eps), unitaries=True)
    return theta, eps


def _scores(ex, flayers_list, log) -> list[dict]:
    """Per circuit: c (success) and c_fallbacks over the test inputs, from ti_tcount._task (fallback_terms)."""
    tasks = [(i, fl, k) for i, fl in enumerate(flayers_list) for k in range(N_TEST)]
    res = [[None] * N_TEST for _ in flayers_list]
    for i, k, r in ex.map(tt._task, tasks, chunksize=1):
        res[i][k] = r
    out = []
    for rr in res:
        la = np.array([r["log_abs"] for r in rr])
        lf = la + np.array([r["kn"] + r["lib"] for r in rr])
        c_s, c_f = -np.expm1(la), -np.expm1(lf)
        out.append({"c": float(c_s.mean()), "se": float(c_s.std(ddof=1) / math.sqrt(N_TEST)),
                    "c_fallbacks": float(c_f.mean()), "c_fallbacks_se": float(c_f.std(ddof=1) / math.sqrt(N_TEST)),
                    "fallbacks_per_cell": float(rr[0]["n_kn"])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field")
    ap.add_argument("t", type=float)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--c-floor", type=float, default=1e-7, dest="c_floor")
    ap.add_argument("--r-max", type=int, default=512, dest="r_max")
    a = ap.parse_args()

    from snapevo.baseline import EPS_GRID, ORDERS, realize_baseline, rotation_tally, trotter_circuit
    from snapevo.hamiltonian import XXZ
    from snapevo.imps_snap import fallback_layers
    from snapevo.synthesis import synthesize

    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, float(a.field))

    def log(s):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g} trotter] {s}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    nofb = lambda _l, _s: None                                               # noqa: E731
    with cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=tt._init,
                                initargs=(a.field, float(t))) as ex:
        # 1) the Trotter-only cost of each (order, r), on the infinite chain
        c_step2 = _scores(ex, [fallback_layers("xxz+x", trotter_circuit(model, t, 2, 1).layers, nofb)], log)[0]["c"]
        plan = []
        for order in ORDERS:
            stop = max(a.c_floor, c_step2 / 10) if order == 1 else a.c_floor
            r = 1
            while True:
                circ = trotter_circuit(model, t, order, r)
                c_tr = _scores(ex, [fallback_layers("xxz+x", circ.layers, nofb)], log)[0]["c"]
                R = sum(rotation_tally(circ).values())
                eps_star = math.sqrt(c_tr / 3) / (0.45 * R) if R else EPS_GRID[0]
                looser = [e for e in EPS_GRID if e > eps_star]
                tighter = [e for e in EPS_GRID if e <= eps_star]
                for eps in (looser if r == 1 else looser[-2:]) + tighter[:1]:
                    plan.append((order, r, eps, c_tr))
                log(f"order {order}, r = {r}: Trotter-only c {c_tr:.3e} (depth {circ.depth}), eps* {eps_star:.1e}")
                if c_tr < stop or r >= a.r_max:
                    break
                r = r + 1 if r < 4 else int(round(r * 1.5))
        # 2) every distinct angle at its precision, with its branch unitaries (KN's cache, parallel)
        need = sorted({(float(th), eps) for order, r, eps, _ in plan
                       for th in rotation_tally(trotter_circuit(model, t, order, r))})
        t0 = time.time()
        list(ex.map(_synth, need, chunksize=1))
        log(f"{len(plan)} formulas, {len(need)} syntheses in {time.time() - t0:.0f}s")
        # 3) realize and score each on the infinite chain
        recs, fls = [], []
        for order, r, eps, c_tr in plan:
            circ = trotter_circuit(model, t, order, r)
            rb = realize_baseline(circ, eps)
            syn = {k: (v if "U0" in v else synthesize(float(k), eps, unitaries=True)) for k, v in rb["synth"].items()}
            rec_for = (lambda l, s, circ=circ, syn=syn: syn.get(f"{float(round(circ.layers[l][1][s], 15)):.15f}"))
            fls.append(fallback_layers("xxz+x", rb["circuit"].layers, rec_for))
            recs.append({"method": "baseline", "order": order, "steps": r, "eps": eps, "c_trotter": c_tr,
                         "depth": circ.depth, "T_per_qubit": rb["Tq"], "rotations_per_qubit": rb["rotations_per_qubit"],
                         "T_per_rotation": rb["Tq"] / rb["rotations_per_qubit"] if rb["rotations_per_qubit"] else None,
                         "n_distinct": rb["n_distinct"]})
        # one formula at a time (100 tasks), each saved as it is scored, so a rerun resumes and memory stays small
        part = OUT / f"{model.tag}_t{t}.partial.json"
        done = json.loads(part.read_text()) if part.exists() else {}
        t0 = time.time()
        for rec, fl in zip(recs, fls):
            key = f"{rec['order']}_{rec['steps']}_{rec['eps']:.1e}"
            if key not in done:
                done[key] = _scores(ex, [fl], log)[0]
                part.write_text(json.dumps(done))
            rec.update(done[key])
        log(f"scored {len(recs)} realized formulas in {time.time() - t0:.0f}s")
    doc = {"model": model.tag, "t": t, "system": "infinite chain, 2-site cell", "c_floor": a.c_floor,
           "metric": "per site on the 100 random chi = 2 iMPS test inputs: c on the success branch, c_fallbacks with "
                     "independent KN fallbacks (imps_snap.fallback_terms)", "points": recs}
    path = OUT / f"{model.tag}_t{t}.json"
    path.write_text(json.dumps(doc, indent=1))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
