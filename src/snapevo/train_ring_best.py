"""Ring depth ladder, each start optimized two ways with no iteration cap, the lower floor kept.

The ring arm's counterpart of train_chain_best (Sohan, 17 Oct 2026: no cap anywhere, the
method that gives the lowest T). Problem and ladder logic of train.train_ladder, the
production ring trainer that made cells.json: the 10-site ring, circuits.TRAIN_SET (12
product states, seed 0) for C_train and train.VAL_SET (32, seed 999) for C_val; starts
"trotter" (order-1 Trotter brickwork), "warm" (the previous depth of this ladder padded
with identity sublayers) and "random" (trotter + 0.1 rad noise seeded by D). Each start
is optimized by both

    newton   chain_snap.newton_minimize with the exact Hessian (forward-over-reverse autodiff,
             eight tangents at a time, as jaxcost.make_hessian) up to HESSIAN_EXACT_MAX angles and
             central differences of the jitted gradient above (the autodiff graph's XLA compile
             grows like D^2.8), a fresh Hessian at every accepted point,
             until the predicted Newton decrease and the last step's are below
             tol_rel x c with no negative curvature ("converged"), or steps stop lowering
             c ("noise floor");
    lbfgs    train_chain_best.lbfgs_minimize: L-BFGS-B with no iteration cap, stopped when
             c fell by less than tol_rel x c over its last WINDOW (25) iterations, on
             L-BFGS-B's own tests, or when its line search fails;

and per depth the lowest C_train over starts x optimizers is kept, with the start and the
optimizer that reached it and every run's statistics. The cost is jaxcost.make_cost's C
with ``precise=True`` (each state's infidelity from the component orthogonal to its target:
~1e-11 relative accuracy at C ~ 1e-9 instead of ~1e-7, the same C up to rounding), so the
convergence tests stay meaningful at the 1e-10 floors; the Hessian is that cost's.

The ladder stops on the stall rule (the floor fell by less than stall[0] over stall[1]
sublayers, (1.5, 8)) or once a floor is at or below ``target`` (1e-10). Output:
runs/<tag>/t<T>/ring2/ladder.json in cells.json's format (header model, cell, t, N,
train_set, val_set; results with D, params (D x n_angles), C_train, C_val, start), so
train.load_cell, train.cell_records and chain_snap.ring_point (cells="ring2/ladder.json")
read it; cells.json is never touched. Each run keeps a checkpoint in ring2/ckpt/ (a rerun
continues it; a finished one returns at once), the ladder holds ring2/lock while it runs,
and at ``deadline`` (or SIGTERM in a run) every run saves and the ladder stops, to be
continued by a rerun. The runs of one depth go to ``workers`` spawned processes at once
(each builds its own jitted cost; workers=1 runs them here, one after the other); above
HESSIAN_EXACT_MAX angles each Newton run spreads its difference Hessian over
``hessian_workers`` helper processes (XLA's own threads do not speed these small contractions up:
61.7 ms with 3 threads against 60.9 ms with one, D = 41).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from snapevo.cells import cell_for
from snapevo.hamiltonian import XXZ

__all__ = ["TOL_REL", "HESSIAN_EXACT_MAX", "STALL", "TARGET", "OPTIMIZERS", "STARTS", "train_ring_ladder_best",
           "ring_run", "make_precise_hessian", "make_fd_hessian", "make_parallel_fd_hessian", "depth_grid"]

TOL_REL = 1e-6
HESSIAN_EXACT_MAX = 300
"""Up to this many angles Newton uses the exact autodiff Hessian; above, central differences of the jitted
gradient (1e-4 rad), because XLA's compile of the unrolled forward-over-reverse graph grows like D^2.8 (100 s at
D = 25, 16 min at D = 57 with 285 angles, one core) and is repeated by every Newton run at every depth, while
one difference Hessian costs about what an autodiff one does there (2 n gradients of 0.1-0.2 s)."""
FD_STEP = 1e-4
STALL = (1.5, 8)
TARGET = 1e-10
OPTIMIZERS = ("newton", "lbfgs")
STARTS = ("trotter", "warm", "random")


def make_precise_hessian(cell, N: int, inputs, targets, D: int, chunk: int = 8):
    """jaxcost.make_hessian's "jax" Hessian (forward-over-reverse, ``chunk`` tangents at a time) of the precise cost."""
    from snapevo.jaxcost import jax, jnp, make_cost
    cost = make_cost(cell, N, inputs, targets, D, precise=True).cost
    gfun = jax.grad(cost)

    @jax.jit
    def rows(x, V):
        return jax.vmap(lambda v: jax.jvp(gfun, (x,), (v,))[1])(V)

    def hessian(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        n = x.shape[0]
        eye = np.eye(n)
        H = np.zeros((n, n))
        for lo in range(0, n, chunk):
            V = jnp.asarray(eye[lo: lo + chunk])
            if V.shape[0] < chunk:
                V = jnp.concatenate([V, jnp.zeros((chunk - V.shape[0], n))])
            H[lo: lo + chunk] = np.asarray(rows(x, V))[: min(chunk, n - lo)]
        return 0.5 * (H + H.T)

    return hessian


def make_fd_hessian(grad, h: float = FD_STEP):
    """Central differences of a jitted ``x -> (c, dc/dx)`` at step h, symmetrized (jaxcost.make_hessian's "fd")."""
    def hessian(x):
        x = np.asarray(x, dtype=np.float64)
        H = np.zeros((x.size, x.size))
        for k in range(x.size):
            e = np.zeros(x.size)
            e[k] = h
            H[k] = (grad(x + e)[1] - grad(x - e)[1]) / (2 * h)
        return 0.5 * (H + H.T)
    return hessian


_FD: dict = {}


def _fd_init(model: dict, t: float, D: int) -> None:
    """A difference-Hessian helper process: its own jitted precise gradient of (model, t, D)."""
    from snapevo.circuits import TRAIN_SET, state_pairs
    from snapevo.jaxcost import make_cost
    from snapevo.train import L_TRAIN
    m = XXZ(**model)
    tin, tg = state_pairs(m, L_TRAIN, t, TRAIN_SET[1], TRAIN_SET[0])
    _FD["grad"] = make_cost(cell_for(m), L_TRAIN, tin, tg, D, precise=True)


def _fd_rows(x, ks, h: float) -> np.ndarray:
    g = _FD["grad"]
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((len(ks), x.size))
    for a, k in enumerate(ks):
        e = np.zeros(x.size)
        e[k] = h
        out[a] = (g(x + e)[1] - g(x - e)[1]) / (2 * h)
    return out


def make_parallel_fd_hessian(model: XXZ, t: float, D: int, workers: int, h: float = FD_STEP):
    """make_fd_hessian with its 2 n gradients spread over ``workers`` spawned helpers; returns (hessian, executor)."""
    ex = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_fd_init,
                                initargs=(model.to_dict(), float(t), int(D)))

    def hessian(x):
        x = np.asarray(x, dtype=np.float64)
        parts = [list(c) for c in np.array_split(np.arange(x.size), workers) if len(c)]
        rows = list(ex.map(_fd_rows, [x] * len(parts), parts, [h] * len(parts)))
        H = np.vstack(rows)
        return 0.5 * (H + H.T)
    return hessian, ex


def ring_run(spec: dict) -> dict:
    """One start x optimizer at one depth (a spawned worker's job, or called directly).

    spec: model (dict), t, D, start (name), z0, optimizer, checkpoint, deadline, tol_rel.
    Returns spec's key with z, info and seconds, or status "deadline" / "failed" (with the error).
    """
    from snapevo.chain_snap import DeadlineReached, newton_minimize
    from snapevo.circuits import TRAIN_SET, state_pairs
    from snapevo.jaxcost import make_cost
    from snapevo.train import L_TRAIN
    from snapevo.train_chain_best import lbfgs_minimize

    model = XXZ(**spec["model"]) if isinstance(spec["model"], dict) else spec["model"]
    cell = cell_for(model)
    D, t, name, opt = int(spec["D"]), float(spec["t"]), spec["start"], spec["optimizer"]
    key = {"D": D, "start": name, "optimizer": opt}
    t0 = time.time()

    def say(m):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g}] D={D} {name}/{opt}: {m}", flush=True)
    try:
        tin, tg = state_pairs(model, L_TRAIN, t, TRAIN_SET[1], TRAIN_SET[0])
        f = make_cost(cell, L_TRAIN, tin, tg, D, precise=True)
        z0 = np.asarray(spec["z0"], dtype=float)
        if opt == "newton":
            exact = z0.size <= int(spec.get("hessian_exact_max", HESSIAN_EXACT_MAX))
            helpers = int(spec.get("hessian_workers", 1))
            pool = None
            if exact:
                H = make_precise_hessian(cell, L_TRAIN, tin, tg, D)
            elif helpers > 1:
                H, pool = make_parallel_fd_hessian(model, t, D, helpers)
            else:
                H = make_fd_hessian(f)
            try:
                z, info = newton_minimize(z0, f, lambda zz, rows: H(zz), lambda c: spec["tol_rel"] * c, n_bulk=None,
                                          log=say, checkpoint=spec["checkpoint"], deadline=spec["deadline"],
                                          label="newton")
            finally:
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
        elif opt == "lbfgs":
            z, info = lbfgs_minimize(z0, f, tol_rel=spec["tol_rel"], checkpoint=spec["checkpoint"],
                                     deadline=spec["deadline"], log=say, label="lbfgs")
        else:
            raise ValueError(f"unknown optimizer {opt!r}")
    except DeadlineReached:
        return {**key, "status": "deadline", "seconds": time.time() - t0}
    except Exception as exc:                                        # noqa: BLE001 (reported to the ladder)
        say(f"failed: {exc!r}")
        return {**key, "status": "failed", "error": repr(exc), "seconds": time.time() - t0}
    if opt == "newton":
        info["hessian"] = "jax" if exact else "fd"
    return {**key, "status": "done", "z": np.asarray(z, dtype=float).tolist(), "info": info,
            "seconds": time.time() - t0}


def _run_record(r: dict) -> dict:
    i = r["info"]
    return {"c": float(i["c_after"]), "c_start": float(i["c_before"]), "steps": int(i["steps"]),
            "hessians": int(i.get("hessians", 0)), "hessian": i.get("hessian"), "stop": i.get("stop"),
            "pred": i.get("pred"), "seconds": float(r["seconds"])}


def train_ring_ladder_best(model: XXZ, t: float, depths: list[int], runs: Path = Path("runs"),
                           out_sub: str = "ring2", starts=STARTS, optimizers=OPTIMIZERS,
                           stall: tuple[float, int] | None = STALL, target: float | None = TARGET,
                           deadline: float | None = None, tol_rel: float = TOL_REL, workers: int | None = None,
                           hessian_workers: int = 1, stop=None, log=print) -> Path:
    """Train every depth in ``depths`` (ascending) on the 10-site ring; per depth the lowest C over starts x optimizers.

    Resumes what runs/<tag>/t<T>/<out_sub>/ladder.json and ckpt/ hold. ``workers``: processes per
    depth (default one per run; 1 runs them in this process). hessian_workers: helper processes of each
    Newton run's difference Hessian (above HESSIAN_EXACT_MAX angles). ``stop`` (anything with is_set(), e.g.
    a threading.Event set by a SIGTERM handler) is checked before each depth. Returns the ladder's path.
    """
    from snapevo.circuits import TRAIN_SET, Layered, local_infidelity, state_pairs
    from snapevo.train import L_TRAIN, VAL_SET, _stalled
    from snapevo.train_newton import _lock, _write

    cell = cell_for(model)
    na = cell.n_angles
    N = L_TRAIN
    out_dir = Path(runs) / model.tag / f"t{t:g}" / out_sub
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.json"
    header = {"model": model.to_dict(), "cell": cell.name, "t": t, "N": N, "train_set": list(TRAIN_SET),
              "val_set": list(VAL_SET), "optimizer": "best of " + ", ".join(optimizers), "tol_rel": tol_rel,
              "starts": list(starts), "cost": "jaxcost precise",
              "hessian": f"jax (exact) up to {HESSIAN_EXACT_MAX} angles, central differences above"}
    lock = _lock(out_dir)
    try:
        results = []
        if out.exists():
            doc = json.loads(out.read_text())
            for k in ("model", "cell", "t", "N", "train_set", "val_set"):
                if doc.get(k) != header[k]:
                    raise ValueError(f"{out} has {k} = {doc.get(k)}, not {header[k]}")
            results = doc["results"]
        vin, vtg = state_pairs(model, N, t, VAL_SET[1], VAL_SET[0])
        prev = None
        for D in sorted(int(d) for d in depths):
            done = [r for r in results if int(r["D"]) == D]
            if done:
                prev = done[0]
                if target is not None and prev["C_train"] <= target:
                    log(f"D={D}: floor {prev['C_train']:.2e} at or below the target {target:.0e} (done before)")
                    break
                if stall is not None and _stalled(results, D, *stall):
                    log(f"D={D}: the ladder stopped here before (stall rule)")
                    break
                continue
            if (deadline is not None and time.time() > deadline) or (stop is not None and stop.is_set()):
                log(f"deadline or stop: stop before D = {D}")
                break
            lower = [r for r in results if int(r["D"]) < D]
            if prev is None and lower:
                prev = max(lower, key=lambda r: r["D"])
            z0 = {"trotter": cell.trotter_angles(model, t, D)}
            if "warm" in starts and prev is not None and int(prev["D"]) < D:
                p = np.asarray(prev["params"], dtype=float).reshape(-1, na)
                z0["warm"] = np.vstack([p, np.zeros((D - p.shape[0], na))]).reshape(-1)
            if "random" in starts:
                z0["random"] = z0["trotter"] + 0.1 * np.random.default_rng(D).normal(size=na * D)
            z0 = {k: v for k, v in z0.items() if k in starts}
            specs = [{"model": model.to_dict(), "t": t, "D": D, "start": s, "z0": z0[s].tolist(), "optimizer": o,
                      "checkpoint": str(ck_dir / f"D{D}_{s}_{o}.json"), "deadline": deadline, "tol_rel": tol_rel,
                      "hessian_workers": hessian_workers}
                     for s in z0 for o in optimizers]
            t_depth = time.time()
            n_work = len(specs) if workers is None else max(1, int(workers))
            if n_work == 1:
                res = [ring_run(sp) for sp in specs]
            else:
                with cf.ProcessPoolExecutor(min(n_work, len(specs)), mp_context=mp.get_context("spawn")) as ex:
                    res = list(ex.map(ring_run, specs))
            if any(r["status"] == "deadline" for r in res):
                log(f"D={D}: deadline reached during training, checkpoints kept in {ck_dir}")
                break
            failed = [r for r in res if r["status"] != "done"]
            if failed:
                log(f"D={D}: run(s) {[(r['start'], r['optimizer'], r.get('error')) for r in failed]} failed; "
                    f"stopping the ladder")
                break
            best = min(res, key=lambda r: r["info"]["c_after"])
            z = np.asarray(best["z"], dtype=float)
            c_val = local_infidelity(Layered.brickwork(cell, z).apply(vin, N), vtg, N)
            rec = {"D": D, "params": z.reshape(D, na).tolist(), "C_train": float(best["info"]["c_after"]),
                   "C_val": float(c_val), "start": best["start"], "optimizer": best["optimizer"],
                   "tol_rel": tol_rel, "seconds": time.time() - t_depth,
                   "runs": {f"{r['start']}/{r['optimizer']}": _run_record(r) for r in res}}
            log(f"D={D} best={best['start']}/{best['optimizer']} C_train={rec['C_train']:.3e} C_val={c_val:.3e} | "
                + ", ".join(f"{r['start']}/{r['optimizer']} {r['info']['c_after']:.3e} ({r['info']['steps']} it, "
                            f"{r['info'].get('stop')})" for r in res) + f" | {rec['seconds']:.0f}s")
            results = [r for r in results if int(r["D"]) != D] + [rec]
            results.sort(key=lambda r: r["D"])
            _write(out, {**header, "results": results})
            prev = rec
            if target is not None and rec["C_train"] <= target:
                log(f"D={D}: floor {rec['C_train']:.2e} is at or below the target {target:.0e}, stopping the ladder")
                break
            if stall is not None and _stalled(results, D, *stall):
                log(f"D={D}: floor fell by less than {stall[0]:g}x over {stall[1]} sublayers, stopping the ladder")
                break
        return out
    finally:
        lock.close()


def depth_grid(model: XXZ, t: float, runs: Path = Path("runs")) -> list[int]:
    """The production ladder's depths for (model, t): cluster/make_jobs.py's DEPTHS[t] and every depth in cells.json."""
    import importlib.util
    root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location("_make_jobs", root / "cluster" / "make_jobs.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    grid = {int(x) for x in mj.DEPTHS[int(t)].split(",")} if int(t) in mj.DEPTHS else set()
    cells = Path(runs) / model.tag / f"t{t:g}" / "cells.json"
    if cells.exists():
        grid |= {int(r["D"]) for r in json.loads(cells.read_text())["results"]}
    if not grid:
        raise ValueError(f"no depth grid for t = {t:g}")
    return sorted(grid)

