"""Newton-polish trained ladder depths, so every t gets the same recipe (L-BFGS from three starts, then trust-region
Newton on the best), without touching cells.json.

    python cluster/polish_ladder.py --tag xxz_d0.25_hx1.6 --t 6 --depths 97,105 [--maxiter 50]

The sweep's ladders got the Newton polish (train.py, ``polish``) only at t <= 4 and only at the depths trained after it
was added; the t >= 6 ladders stalled under L-BFGS alone, with equal floors at consecutive depths. This applies the same
50 trust-exact steps (exact Hessian, JAX forward-over-reverse) to existing depths and writes
runs/<tag>/t<t>/cells_polished.json: one entry per depth with the old and new C_train and C_val, the new angles, and
the seconds. An entry is kept only when it lowers C_train. Depths already in the file are skipped, so runs resume.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.optimize as sopt

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--depths", required=True, help="comma-separated, or 'all'")
    ap.add_argument("--maxiter", type=int, default=50)
    args = ap.parse_args(argv)

    from snapevo import io
    from snapevo.cells import get_cell
    from snapevo.circuits import Layered, TRAIN_SET, local_infidelity, state_pairs
    from snapevo.hamiltonian import XXZ
    from snapevo.jaxcost import make_cost, make_hessian
    from snapevo.train import L_TRAIN, VAL_SET

    tdir = REPO / "runs" / args.tag / f"t{args.t}"
    doc = io.read_json(tdir / "cells.json")
    m = doc["model"]
    model, cell, t, N = XXZ(delta=m["delta"], hx=m["hx"], J=m["J"]), get_cell(doc["cell"]), float(doc["t"]), doc["N"]
    assert N == L_TRAIN, (N, L_TRAIN)
    tin, tg = state_pairs(model, N, t, TRAIN_SET[1], TRAIN_SET[0])
    vin, vtg = state_pairs(model, N, t, VAL_SET[1], VAL_SET[0])
    out = tdir / "cells_polished.json"
    pol = io.read_json(out) if out.exists() else {"model": m, "cell": doc["cell"], "t": t, "N": N, "maxiter": args.maxiter,
                                                  "results": []}
    have = {int(r["D"]) for r in pol["results"]}
    byD = {int(r["D"]): r for r in doc["results"]}
    depths = sorted(byD) if args.depths == "all" else [int(v) for v in args.depths.split(",")]
    for D in depths:
        if D in have:
            continue
        r = byD[D]
        x0 = np.asarray(r["params"], dtype=float).reshape(-1)
        f = make_cost(cell, N, tin, tg, D)
        hess = make_hessian(cell, N, tin, tg, D, method="jax")
        t0 = time.time()
        res = sopt.minimize(f, x0, jac=True, hess=hess, method="trust-exact",
                            options={"maxiter": args.maxiter, "gtol": 1e-16})
        sec = time.time() - t0
        better = bool(res.fun < r["C_train"])
        x = res.x if better else x0
        c_val = local_infidelity(Layered.brickwork(cell, x).apply(vin, N), vtg, N)
        print(f"[{time.strftime('%H:%M:%S')}] {args.tag} t={args.t} D={D}: C_train {r['C_train']:.3e} -> {res.fun:.3e} "
              f"(nit {res.nit}, {sec:.0f} s), C_val {r['C_val']:.3e} -> {c_val:.3e}{'' if better else ', kept the old'}",
              flush=True)
        pol["results"].append({"D": D, "params": np.asarray(x).reshape(D, cell.n_angles).tolist(),
                               "C_train_old": r["C_train"], "C_val_old": r["C_val"], "C_train": float(min(res.fun, r["C_train"])),
                               "C_val": float(c_val), "nit": int(res.nit), "seconds": sec, "improved": better,
                               "start": r.get("start", "") + ("+newton" if better else "")})
        pol["results"].sort(key=lambda q: q["D"])
        io.write_json(out, pol, indent=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
