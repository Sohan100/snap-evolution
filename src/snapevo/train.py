"""Train the brickwork depth ladder for one model and time.

Each depth starts from three points and keeps the best after L-BFGS-B:
the first-order Trotter angles, the previous depth padded with identity
sublayers ("warm", which wins at depth), and the Trotter angles plus 0.1 rad
noise. Training uses 12 product states (seed 0) at N = 10; the recorded
``C_val`` uses 32 more (seed 999). Held-out C comes from circuits.held_out
and the reported error from evaluate.py, never from here.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.optimize as sopt

from snapevo import io
from snapevo.cells import Cell
from snapevo.circuits import Layered, TRAIN_SET, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ

__all__ = ["train_ladder", "load_cell", "cell_records"]

VAL_SET = (999, 32)
L_TRAIN = 10


def cell_records(path: Path) -> list[dict]:
    return io.read_json(path)["results"]


def load_cell(path: Path, D: int) -> tuple[np.ndarray, dict]:
    """Flat trained angles at depth D and the file header (model, cell, t, N)."""
    doc = io.read_json(path)
    for r in doc["results"]:
        if int(r["D"]) == D:
            return np.asarray(r["params"], dtype=float).reshape(-1), doc
    raise KeyError(f"{path} has no D={D}; available: {sorted(r['D'] for r in doc['results'])}")


def train_ladder(model: XXZ, cell: Cell, t: float, depths: list[int], out: Path,
                 maxiter: int = 6000, log=print) -> Path:
    """Train every depth in ``depths`` (ascending), resuming any already in ``out``."""
    from snapevo.jaxcost import make_cost

    N = L_TRAIN
    na = cell.n_angles
    tin, tg = state_pairs(model, N, t, TRAIN_SET[1], TRAIN_SET[0])
    vin, vtg = state_pairs(model, N, t, VAL_SET[1], VAL_SET[0])
    header = {"model": model.to_dict(), "cell": cell.name, "t": t, "N": N,
              "train_set": list(TRAIN_SET), "val_set": list(VAL_SET)}
    results = []
    if Path(out).exists():
        doc = io.read_json(out)
        if {k: doc[k] for k in ("model", "cell", "t")} != {k: header[k] for k in ("model", "cell", "t")}:
            raise ValueError(f"{out} belongs to a different model, cell or time")
        results = doc["results"]
    opts = {"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14, "maxfun": 400000}
    prev = None
    for D in sorted(depths):
        done = [r for r in results if int(r["D"]) == D]
        if done:
            prev = np.asarray(done[0]["params"])
            continue
        lower = [r for r in results if int(r["D"]) < D]
        if prev is None and lower:
            prev = np.asarray(max(lower, key=lambda r: r["D"])["params"])
        f = make_cost(cell, N, tin, tg, D)
        starts = {"trotter": cell.trotter_angles(model, t, D)}
        if prev is not None and prev.shape[0] < D:
            starts["warm"] = np.vstack([prev, np.zeros((D - prev.shape[0], na))]).reshape(-1)
        starts["random"] = starts["trotter"] + 0.1 * np.random.default_rng(D).normal(size=na * D)
        best, best_name = None, None
        for name, x0 in starts.items():
            t0 = time.time()
            res = sopt.minimize(f, x0, jac=True, method="L-BFGS-B", options=opts)
            log(f"  D={D:2d} start={name:7s} C_train={res.fun:.3e} nit={res.nit} ({time.time() - t0:.0f}s)")
            if best is None or res.fun < best.fun:
                best, best_name = res, name
        params = best.x.reshape(D, na)
        c_val = local_infidelity(Layered.brickwork(cell, best.x).apply(vin, N), vtg, N)
        log(f"D={D:2d} best={best_name} C_train={best.fun:.3e} C_val={c_val:.3e}")
        results.append({"D": D, "params": params.tolist(), "C_train": float(best.fun),
                         "C_val": float(c_val), "start": best_name})
        results.sort(key=lambda r: r["D"])
        io.write_json(out, {**header, "results": results}, indent=None)
        prev = params
    return Path(out)
