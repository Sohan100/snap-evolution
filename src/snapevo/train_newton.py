"""Depth ladders trained by trust-region Newton, with no iteration cap, in both arms.

The open-chain arm (train_chain.train_chain_ladder) and the ring arm (train.train_ladder)
optimize each start with L-BFGS-B and a fixed iteration budget. Here every start is
optimized by trust-region Newton on the full Hessian (chain_snap.newton_minimize: scipy's
trust-exact, a fresh Hessian at every accepted point), until it has converged:

    the predicted Newton decrease g.H^+.g/2 and the last step's decrease are both below
    tol_rel x c (tol_rel = 1e-6, relative to the current cost), with no negative curvature
    beyond 1e-8 of the largest eigenvalue in the bulk block, "converged"; or trust-exact's
    model no longer predicts a decrease (steps stop lowering c, as at the MPS truncation
    floor), "noise floor".

No step cap; the only other exit is the deadline (or SIGTERM), which keeps each start's
checkpoint so a rerun continues it.

Ladder logic as the L-BFGS trainers: depths ascending; starts "trotter" (the order-1
Trotter brickwork), "warm" (the previous depth of this ladder padded with identity
sublayers) and "random" (trotter + 0.1 rad noise seeded by D); the best start is kept; the
ladder stops on the stall rule (the floor fell by less than stall[0] over stall[1]
sublayers, default (1.5, 8)) or once a floor is at or below ``target`` (default 1e-10);
depths already in the ladder are kept and a rerun continues past them.

train_chain_ladder_newton   The open chain of N = 100 sites, z = (bulk, edges), the cost
    mps_train.OpenChainCost on train_chain's training inputs (FarmCost on a ChainPool: the
    Hessian's 2 len(z) gradients are spread over groups x replicas workers, and the starts
    of a depth share the pool), the truncation cutoff_for picks from this ladder, the
    held-out c on the sweep's 8 inputs at the training cutoff (train_chain.heldout), the
    lock and runs/<tag>/t<T>/chain/ladder.json and ckpt/ of train_chain. It never builds a
    ring and never reads a ring result.
train_ring_ladder_newton    The 10-site ring, the production training set and validation
    set of train.py, jaxcost's exact Hessian (make_hessian, method "jax" by default), written
    to runs/<tag>/t<T>/ring2/ladder.json in cells.json's format (train.load_cell reads it), so
    the ring arm can be retrained with the identical optimizer; cells.json is not touched.
"""

from __future__ import annotations

import concurrent.futures as cf
import fcntl
import json
import time
from pathlib import Path

import numpy as np

from snapevo.cells import cell_for
from snapevo.chain_snap import ChainProblem, DeadlineReached, newton_minimize
from snapevo.hamiltonian import XXZ

__all__ = ["TOL_REL", "STALL", "TARGET", "train_chain_ladder_newton", "train_ring_ladder_newton"]

TOL_REL = 1e-6
"""Convergence: the predicted and the last decrease below this times the current c."""

STALL = (1.5, 8)
TARGET = 1e-10


def _lock(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / "lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise RuntimeError(f"another run holds {out_dir / 'lock'}") from None
    return fh


def _write(path: Path, doc: dict) -> None:
    import os
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)


def _start_record(z, info: dict, seconds: float) -> dict:
    return {"c": float(info["c_after"]), "c_start": float(info["c_before"]), "steps": int(info["steps"]),
            "hessians": int(info["hessians"]), "stop": info.get("stop"), "pred": info.get("pred"),
            "eig_min_bulk": info.get("eig_min_bulk_after"), "n_neg_bulk": info.get("n_neg_bulk_after"),
            "seconds": float(seconds)}


def _run_starts(z0: dict, run, parallel: bool) -> tuple[list, bool]:
    """Run the starts; returns ([(name, z, info, seconds)], stopped at the deadline)."""
    out, stopped = [], False

    def one(name):
        try:
            return run(name)
        except DeadlineReached:
            return name, None, None, 0.0

    if parallel and len(z0) > 1:
        with cf.ThreadPoolExecutor(len(z0)) as ex:
            res = list(ex.map(one, list(z0)))
    else:
        res = [one(n) for n in z0]
    for r in res:
        if r[1] is None:
            stopped = True
        else:
            out.append(r)
    return out, stopped


def train_chain_ladder_newton(model: XXZ, t: float, depths: list[int], runs: Path = Path("runs"), N: int = 100,
                              n_inputs: int = 12, groups: int | None = None, replicas: int = 1,
                              ref_cache: Path | None = None, starts=("trotter", "warm", "random"),
                              parallel_starts: bool = True, stall: tuple[float, int] | None = STALL,
                              target: float | None = TARGET, deadline: float | None = None, tol_rel: float = TOL_REL,
                              heldout_processes: int = 8, local: bool = False, log=print) -> Path:
    """Train every depth in ``depths`` (ascending) on the open chain by Newton, resuming what ``runs`` holds.

    One ChainProblem (groups x replicas workers, default one input per group) per depth,
    shared by its starts, which run at the same time with ``parallel_starts``. Records as
    train_chain's, with "optimizer": "newton" and per start: c, c_start, steps, Hessians,
    stop reason, final predicted decrease, the bulk Hessian's lowest eigenvalue, seconds.
    Records of the L-BFGS trainer already in the ladder are kept (warm starts use them).
    """
    from snapevo.mps_train import open_chain_problem
    from snapevo.train_chain import (EVAL_DT, EVAL_SET, TRAIN_SEED, _eval_targets, _stalled, cutoff_for, heldout,
                                     pad_z, trotter_z)

    cell = cell_for(model)
    out_dir = Path(runs) / model.tag / f"t{t:g}" / "chain"
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.json"
    header = {"model": model.to_dict(), "cell": cell.name, "t": float(t), "N": N, "bc": "open",
              "train_set": [TRAIN_SEED, n_inputs], "eval_set": list(EVAL_SET), "eval_dt": EVAL_DT,
              "optimizer": "newton", "tol_rel": tol_rel, "starts": list(starts)}
    lock = _lock(out_dir)
    try:
        results = []
        if out.exists():
            doc = json.loads(out.read_text())
            for k in ("model", "cell", "t", "N", "bc", "train_set", "eval_set", "eval_dt"):
                if doc.get(k) != header[k]:
                    raise ValueError(f"{out} has {k} = {doc.get(k)}, not {header[k]}")
            results = doc["results"]
        for c_ in {cutoff_for(t, None)[0], cutoff_for(t, 1e-12)[0]}:          # every held-out setting we may use
            _eval_targets(model, t, N, runs, c_, 1024)
        vecs, targets = open_chain_problem(model, t, N, n_inputs, TRAIN_SEED, ref_cache, processes=1, log=log)
        prev = None
        for D in sorted(depths):
            done = [r for r in results if int(r["D"]) == D]
            if done:
                prev = done[0]
                if target is not None and prev["c_train"] <= target:
                    break
                if stall is not None and _stalled(results, D, *stall):
                    log(f"D={D}: the ladder stopped here before (stall rule)")
                    break
                continue
            if deadline is not None and time.time() > deadline:
                log(f"deadline: stop before D = {D}")
                break
            lower = [r for r in results if int(r["D"]) < D]
            if prev is None and lower:
                prev = max(lower, key=lambda r: r["D"])
            cutoff, chi = cutoff_for(t, None if prev is None else prev["c_train"])
            z0 = {"trotter": trotter_z(model, cell, t, D)}
            if "warm" in starts and prev is not None:
                z0["warm"] = pad_z(cell, prev["z"], int(prev["D"]), D)
            if "random" in starts:
                z0["random"] = z0["trotter"] + 0.1 * np.random.default_rng(D).normal(size=z0["trotter"].size)
            z0 = {k: v for k, v in z0.items() if k in starts}
            nb = D * cell.n_angles
            t_depth = time.time()
            with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=chi, cutoff=cutoff, groups=groups,
                              replicas=replicas, local=local) as P:
                def run(name):
                    t0 = time.time()
                    z, info = newton_minimize(z0[name], P.objective, lambda zz, rows: P.hessian(zz, rows),
                                              lambda c: tol_rel * c, n_bulk=nb,
                                              log=lambda s: log(f"  D={D} {name}: {s}"),
                                              checkpoint=str(ck_dir / f"D{D}_{name}_newton_c{cutoff:.0e}.json"),
                                              deadline=deadline, label="newton")
                    return name, z, info, time.time() - t0

                runs_, stopped = _run_starts(z0, run, parallel_starts)
            if stopped:
                log(f"D={D}: deadline reached during training, checkpoints kept")
                break
            name, best_z, best, _ = min(runs_, key=lambda r: r[2]["c_after"])
            t0 = time.time()
            c_held = heldout(model, cell, t, D, best_z, N, runs, cutoff=cutoff, chi_max=1024,
                             processes=heldout_processes)
            rec = {"D": D, "z": [float(v) for v in best_z], "c_train": float(best["c_after"]), "c_heldout": c_held,
                   "start": name, "optimizer": "newton", "tol_rel": tol_rel, "train_cutoff": cutoff,
                   "train_chi": chi, "heldout_cutoff": cutoff, "heldout_seconds": time.time() - t0,
                   "seconds": time.time() - t_depth, "workers": [groups or n_inputs, replicas],
                   "starts": {n_: _start_record(z_, i_, s_) for n_, z_, i_, s_ in runs_}}
            log(f"D={D} best={name} c_train={rec['c_train']:.3e} c_heldout={c_held:.3e} "
                + ", ".join(f"{n_} {i_['c_after']:.3e} ({i_['steps']} steps, {i_['hessians']} Hessians, {i_['stop']})"
                            for n_, _z, i_, _s in runs_))
            results = [r for r in results if int(r["D"]) != D] + [rec]
            results.sort(key=lambda r: r["D"])
            _write(out, {**header, "results": results})
            prev = rec
            if target is not None and rec["c_train"] <= target:
                log(f"D={D}: floor {rec['c_train']:.2e} is at or below the target {target:.0e}, stopping the ladder")
                break
            if stall is not None and _stalled(results, D, *stall):
                log(f"D={D}: floor fell by less than {stall[0]:g}x over {stall[1]} sublayers, stopping the ladder")
                break
        return out
    finally:
        lock.close()


def train_ring_ladder_newton(model: XXZ, t: float, depths: list[int], runs: Path = Path("runs"),
                             out_sub: str = "ring2", starts=("trotter", "warm", "random"),
                             stall: tuple[float, int] | None = STALL, target: float | None = TARGET,
                             deadline: float | None = None, tol_rel: float = TOL_REL, hessian_method: str = "jax",
                             parallel_starts: bool = False, log=print) -> Path:
    """Train every depth in ``depths`` (ascending) on the 10-site ring by Newton, into runs/<tag>/t<T>/ring2/.

    train.train_ladder's problem (N = 10, circuits.TRAIN_SET, C_val on train.VAL_SET) and starts,
    jaxcost.make_cost and make_hessian (``hessian_method``: "jax", exact autodiff; "auto"
    switches to central differences of the jitted gradient above 96 angles), each start
    optimized by newton_minimize with tolerance tol_rel x c. ladder.json has cells.json's
    format (params as D x n_angles, C_train, C_val, start) plus the Newton statistics.
    """
    from snapevo.circuits import TRAIN_SET, Layered, local_infidelity, state_pairs
    from snapevo.jaxcost import make_cost, make_hessian
    from snapevo.train import L_TRAIN, VAL_SET, _stalled

    cell = cell_for(model)
    na = cell.n_angles
    N = L_TRAIN
    out_dir = Path(runs) / model.tag / f"t{t:g}" / out_sub
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.json"
    header = {"model": model.to_dict(), "cell": cell.name, "t": t, "N": N, "train_set": list(TRAIN_SET),
              "val_set": list(VAL_SET), "optimizer": "newton", "tol_rel": tol_rel, "hessian": hessian_method}
    lock = _lock(out_dir)
    try:
        results = json.loads(out.read_text())["results"] if out.exists() else []
        tin, tg = state_pairs(model, N, t, TRAIN_SET[1], TRAIN_SET[0])
        vin, vtg = state_pairs(model, N, t, VAL_SET[1], VAL_SET[0])
        prev = None
        for D in sorted(depths):
            done = [r for r in results if int(r["D"]) == D]
            if done:
                prev = np.asarray(done[0]["params"])
                if target is not None and done[0]["C_train"] <= target:
                    break
                if stall is not None and _stalled(results, D, *stall):
                    break
                continue
            if deadline is not None and time.time() > deadline:
                log(f"deadline: stop before D = {D}")
                break
            lower = [r for r in results if int(r["D"]) < D]
            if prev is None and lower:
                prev = np.asarray(max(lower, key=lambda r: r["D"])["params"])
            f = make_cost(cell, N, tin, tg, D)
            H = make_hessian(cell, N, tin, tg, D, method=hessian_method, grad=f)
            z0 = {"trotter": cell.trotter_angles(model, t, D)}
            if "warm" in starts and prev is not None and prev.shape[0] < D:
                z0["warm"] = np.vstack([prev, np.zeros((D - prev.shape[0], na))]).reshape(-1)
            if "random" in starts:
                z0["random"] = z0["trotter"] + 0.1 * np.random.default_rng(D).normal(size=na * D)
            z0 = {k: v for k, v in z0.items() if k in starts}
            t_depth = time.time()

            def run(name, f=f, H=H, D=D):
                t0 = time.time()
                z, info = newton_minimize(z0[name], f, lambda zz, rows: H(zz), lambda c: tol_rel * c, n_bulk=None,
                                          log=lambda s: log(f"  D={D:2d} {name}: {s}"),
                                          checkpoint=str(ck_dir / f"D{D}_{name}_newton.json"), deadline=deadline,
                                          label="newton")
                return name, z, info, time.time() - t0

            runs_, stopped = _run_starts(z0, run, parallel_starts)
            if stopped:
                log(f"D={D}: deadline reached during training, checkpoints kept")
                break
            name, best_z, best, _ = min(runs_, key=lambda r: r[2]["c_after"])
            c_val = local_infidelity(Layered.brickwork(cell, best_z).apply(vin, N), vtg, N)
            params = np.asarray(best_z).reshape(D, na)
            rec = {"D": D, "params": params.tolist(), "C_train": float(best["c_after"]), "C_val": float(c_val),
                   "start": name, "optimizer": "newton", "tol_rel": tol_rel, "seconds": time.time() - t_depth,
                   "starts": {n_: _start_record(z_, i_, s_) for n_, z_, i_, s_ in runs_}}
            log(f"D={D:2d} best={name} C_train={rec['C_train']:.3e} C_val={c_val:.3e} "
                + ", ".join(f"{n_} {i_['c_after']:.3e} ({i_['steps']} steps, {i_['stop']})"
                            for n_, _z, i_, _s in runs_))
            results = [r for r in results if int(r["D"]) != D] + [rec]
            results.sort(key=lambda r: r["D"])
            _write(out, {**header, "results": results})
            prev = params
            if target is not None and rec["C_train"] <= target:
                log(f"D={D:2d}: floor {rec['C_train']:.2e} is at or below the target {target:.0e}, "
                    f"stopping the ladder")
                break
            if stall is not None and _stalled(results, D, *stall):
                log(f"D={D:2d}: floor fell by less than {stall[0]:g}x over {stall[1]} sublayers, stopping the ladder")
                break
        return out
    finally:
        lock.close()
