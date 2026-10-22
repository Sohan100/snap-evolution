"""Open-chain depth ladder, each start optimized two ways with no iteration cap, the lower floor kept.

Sohan, 17 Oct 2026: no cap anywhere, and use whichever method gives the lowest T counts. A
lower floor at a depth leaves more of a budget for snapping, so the ladder keeps, per depth,
the lowest c reached by any start and any optimizer, and records which one it was:

    newton   chain_snap.newton_minimize: trust-region Newton on the full finite-difference
             Hessian, a fresh Hessian at every accepted point, until the predicted Newton
             decrease and the last step's are below tol_rel x c with no negative bulk
             curvature ("converged"), or until steps stop lowering c ("noise floor").
    lbfgs    scipy L-BFGS-B with the same gradients and no iteration cap: it stops when c fell
             by less than tol_rel x c over the last WINDOW iterations ("converged"), on
             L-BFGS-B's own tests, or when its line search fails ("noise floor" / its message).

Everything else is train_chain's open-chain ladder (no ring is built or read): starts trotter
(order-1 Trotter brickwork with the cell's own end factor), warm (the previous depth of this
ladder padded with identity sublayers) and random (trotter + 0.1 rad noise seeded by D); the
truncation from cutoff_for; held-out c on the sweep's 8 inputs at the training cutoff; the
stall (1.5 over 8 sublayers) and target (1e-10) stop rules; runs/<tag>/t<T>/chain/ladder.json,
per-run checkpoints in chain/ckpt/ (a rerun continues every run), the lock, the deadline.
The runs of one depth share one ChainPool (groups x replicas workers) and run at once.
"""

from __future__ import annotations

import concurrent.futures as cf
import fcntl
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import scipy.optimize as sopt

from snapevo.cells import cell_for
from snapevo.chain_snap import ChainProblem, DeadlineReached, newton_minimize
from snapevo.hamiltonian import XXZ

__all__ = ["TOL_REL", "WINDOW", "OPTIMIZERS", "lbfgs_minimize", "train_chain_ladder_best", "precompute_chain_runs"]

TOL_REL = 1e-6
WINDOW = 25
OPTIMIZERS = ("newton", "lbfgs")
_LBFGS_OPTIONS = {"maxiter": 2 ** 62, "maxfun": 2 ** 62, "ftol": 0.0, "gtol": 1e-14, "maxcor": 10}


def lbfgs_minimize(z, objective, tol_rel: float = TOL_REL, window: int = WINDOW, checkpoint: str | None = None,
                   deadline: float | None = None, log=None, label: str = "lbfgs") -> tuple[np.ndarray, dict]:
    """L-BFGS-B with no iteration cap until c fell by less than tol_rel x c over ``window`` iterations.

    ``checkpoint`` (JSON) keeps z, the iteration count and the recent c every ``window``
    iterations and at the end; a rerun continues from it (L-BFGS-B's memory rebuilds). At the
    deadline or on SIGTERM it saves and raises DeadlineReached. Returns (z, info) in
    newton_minimize's format (c_before, c_after, steps, stop, seconds; hessians = 0).
    """
    say = log or (lambda _m: None)
    z_in = np.asarray(z, dtype=float).reshape(-1)
    start, done, hist, secs = z_in.copy(), 0, [], 0.0
    if checkpoint and os.path.exists(checkpoint):
        ck = json.loads(open(checkpoint).read())
        if not np.array_equal(np.asarray(ck["z_in"]), z_in):
            raise ValueError(f"{checkpoint} is for another start point")
        start, done, hist, secs = np.asarray(ck["z"], dtype=float), int(ck["steps"]), list(ck["hist"]), ck["seconds"]
        if ck.get("stop"):
            info = ck["info"]
            return start, info
        say(f"{label}: resumed {os.path.basename(checkpoint)} after {done} iterations")
    t0 = time.time() - secs
    term = {"flag": False}
    old = None
    try:
        old = signal.signal(signal.SIGTERM, lambda *_: term.__setitem__("flag", True))
    except ValueError:                                              # not the main thread
        pass
    state = {"z": start.copy(), "c": None, "n": done, "stop": None}
    c0 = {"v": None}

    def save(final_info=None):
        if not checkpoint:
            return
        tmp = f"{checkpoint}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"z_in": [float(v) for v in z_in], "z": [float(v) for v in state["z"]], "steps": state["n"],
                       "hist": hist[-(window + 1):], "seconds": time.time() - t0,
                       "stop": None if final_info is None else final_info["stop"], "info": final_info}, fh)
        os.replace(tmp, checkpoint)

    def f(x):
        c, g = objective(x)
        if c0["v"] is None:
            c0["v"] = float(c)
        return c, g

    def cb(xk):
        state["n"] += 1
        state["z"] = np.array(xk, dtype=float)
        c = float(objective_cache["c"]) if objective_cache["x"] is not None and np.array_equal(
            objective_cache["x"], xk) else float(f_only(xk))
        hist.append(c)
        if state["n"] % window == 0:
            save()
            say(f"{label}: iteration {state['n']}, c = {c:.6e}, {time.time() - t0:.0f}s")
        if len(hist) > window and hist[-window - 1] - c < tol_rel * c:
            state["stop"] = "converged"
            raise StopIteration
        if term["flag"] or (deadline is not None and time.time() > deadline):
            state["stop"] = "deadline"
            raise StopIteration

    objective_cache = {"x": None, "c": None}

    def fg(x):
        c, g = f(x)
        objective_cache["x"], objective_cache["c"] = np.array(x, dtype=float), c
        return c, g

    def f_only(x):
        return fg(x)[0]

    try:
        res = sopt.minimize(fg, start, jac=True, method="L-BFGS-B", callback=cb, options=_LBFGS_OPTIONS)
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)
    if state["stop"] == "deadline":
        save()
        raise DeadlineReached(label)
    z_out = np.asarray(res.x, dtype=float)
    c_out = float(res.fun)
    stop = state["stop"] or ("noise floor" if "ABNORMAL" in str(res.message) else f"lbfgsb: {res.message}")
    state["z"] = z_out
    info = {"c_before": float(c0["v"] if c0["v"] is not None else c_out), "c_after": c_out, "steps": state["n"],
            "hessians": 0, "stop": stop, "pred": None, "seconds": time.time() - t0}
    save(info)
    say(f"{label}: done, c = {c_out:.6e} after {state['n']} iterations ({stop}), {info['seconds']:.0f}s")
    return z_out, info


_DONE = ("converged", "noise floor")


def _mark_done(ck: str, z_in, z, info: dict) -> None:
    """Record a finished run next to its checkpoint (newton_minimize's checkpoint has no final stop reason)."""
    tmp = f"{ck}.done.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump({"z_in": [float(v) for v in np.asarray(z_in).reshape(-1)],
                   "z": [float(v) for v in np.asarray(z).reshape(-1)], "info": info}, fh, default=float)
    os.replace(tmp, ck + ".done")


def _finished(ck: str, z_in) -> tuple | None:
    """(z, info) of a run recorded as finished from this start point (its .done record, or an L-BFGS
    checkpoint with a stop reason), else None."""
    path = ck + ".done" if os.path.exists(ck + ".done") else ck
    if not os.path.exists(path):
        return None
    try:
        doc = json.loads(open(path).read())
    except (OSError, ValueError):
        return None
    if not np.array_equal(np.asarray(doc.get("z_in", [])), np.asarray(z_in)):
        return None
    info = doc.get("info") or {}
    stop = info.get("stop")
    if stop is None or not (stop in _DONE or str(stop).startswith("lbfgsb")) or "c_after" not in info:
        return None
    return np.asarray(doc["z"], dtype=float), info


def _run_lock(ck: str, block: bool, deadline: float | None):
    """Open and flock ck + '.lock'; with block, wait (checking the deadline) while another process holds it.

    Returns the open file (keep it until the run is done), or None if not blocking and held, or "deadline".
    """
    fh = open(ck + ".lock", "w")
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            if not block:
                fh.close()
                return None
            if deadline is not None and time.time() > deadline:
                fh.close()
                return "deadline"
            time.sleep(15)


def _newton(mode: str, z, P, tol, **kw):
    """A Newton run on ChainProblem P: newton_minimize ("full", the default), or newton_fast's "reuse" (the last
    full Hessian reused, refreshed on a poor step ratio, every 10 steps and to confirm convergence) or "cg"
    (truncated Newton-CG with Hessian-vector products of the adjoint gradient, preconditioned by the last full
    Hessian). Same stopping rule, checkpoint and record; runs may switch mode between jobs."""
    if mode == "full":
        return newton_minimize(z, P.objective, lambda zz, rows: P.hessian(zz, rows), tol, **kw)
    from snapevo.newton_fast import minimize_newton
    return minimize_newton(mode, z, P.objective, lambda zz, rows: P.hessian(zz, rows), tol,
                           grad_many=P.cost.value_and_grad_many, **kw)


def _start_record(info: dict, seconds: float) -> dict:
    return {"c": float(info["c_after"]), "c_start": float(info["c_before"]), "steps": int(info["steps"]),
            "hessians": int(info.get("hessians", 0)), "stop": info.get("stop"), "pred": info.get("pred"),
            "seconds": seconds}


def train_chain_ladder_best(model: XXZ, t: float, depths: list[int], runs: Path = Path("runs"), N: int = 100,
                            n_inputs: int = 12, groups: int | None = None, replicas: int = 1,
                            ref_cache: Path | None = None, starts=("trotter", "warm", "random"),
                            optimizers=OPTIMIZERS, stall: tuple[float, int] | None = (1.5, 8),
                            target: float | None = 1e-10, deadline: float | None = None, tol_rel: float = TOL_REL,
                            heldout_processes: int = 8, local: bool = False, log=print,
                            newton_mode: str = "full") -> Path:
    """Train every depth in ``depths`` (ascending) on the open chain; per depth the lowest c over starts x optimizers."""
    from snapevo.mps_train import open_chain_problem
    from snapevo.train_chain import (EVAL_DT, EVAL_SET, TRAIN_SEED, _eval_targets, _stalled, cutoff_for,
                                     eval_cutoff, heldout, pad_z, train_refs, trotter_z)
    from snapevo.train_newton import _lock, _write

    cell = cell_for(model)
    out_dir = Path(runs) / model.tag / f"t{t:g}" / "chain"
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.json"
    header = {"model": model.to_dict(), "cell": cell.name, "t": float(t), "N": N, "bc": "open",
              "train_set": [TRAIN_SEED, n_inputs], "eval_set": list(EVAL_SET), "eval_dt": EVAL_DT,
              "optimizer": "best of " + ", ".join(optimizers), "tol_rel": tol_rel, "starts": list(starts),
              "newton_mode": newton_mode}
    lock = _lock(out_dir)
    try:
        results = []
        if out.exists():
            doc = json.loads(out.read_text())
            for k in ("model", "cell", "t", "N", "bc", "train_set", "eval_set", "eval_dt"):
                if doc.get(k) != header[k]:
                    raise ValueError(f"{out} has {k} = {doc.get(k)}, not {header[k]}")
            results = doc["results"]
        for c_ in {eval_cutoff(t, cutoff_for(t, None)[0]), eval_cutoff(t, cutoff_for(t, 1e-12)[0])}:
            _eval_targets(model, t, N, runs, c_, 1024)
        vecs, targets = open_chain_problem(model, t, N, n_inputs, TRAIN_SEED, ref_cache, processes=n_inputs, log=log,
                                           **train_refs(t))
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
            jobs = [(s, o) for s in z0 for o in optimizers]
            nb = D * cell.n_angles
            t_depth = time.time()
            with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=chi, cutoff=cutoff, groups=groups,
                              replicas=replicas, local=local) as P:
                def run(job):
                    s, o = job
                    t0 = time.time()
                    ck = str(ck_dir / f"D{D}_{s}_{o}_c{cutoff:.0e}.json")
                    lg = lambda m: log(f"  D={D} {s}/{o}: {m}")             # noqa: E731
                    lk = _run_lock(ck, True, deadline)           # a precompute worker may be on this run
                    if lk == "deadline":
                        return job, None, None, time.time() - t0
                    try:
                        fin = _finished(ck, z0[s])
                        if fin is not None:
                            lg(f"finished earlier (precomputed or a previous job): c = {fin[1]['c_after']:.6e}")
                            return job, fin[0], fin[1], time.time() - t0
                        if o == "newton":
                            z, info = _newton(newton_mode, z0[s], P, lambda c: tol_rel * c, n_bulk=nb, log=lg,
                                              checkpoint=ck, deadline=deadline, label="newton")
                        else:
                            z, info = lbfgs_minimize(z0[s], P.objective, tol_rel=tol_rel, checkpoint=ck,
                                                     deadline=deadline, log=lg)
                        _mark_done(ck, z0[s], z, info)
                    except DeadlineReached:
                        return job, None, None, time.time() - t0
                    except Exception as exc:                            # noqa: BLE001
                        lg(f"failed: {exc!r}")
                        return job, "failed", None, time.time() - t0
                    finally:
                        lk.close()
                    return job, z, info, time.time() - t0

                with cf.ThreadPoolExecutor(len(jobs)) as ex:
                    res = list(ex.map(run, jobs))
            if any(r[1] is None for r in res):
                log(f"D={D}: deadline reached during training, checkpoints kept")
                break
            if any(isinstance(r[1], str) for r in res):
                log(f"D={D}: run(s) {[r[0] for r in res if isinstance(r[1], str)]} failed; stopping the ladder")
                break
            (s_best, o_best), z_best, i_best, _ = min(res, key=lambda r: r[2]["c_after"])
            t0 = time.time()
            c_held = heldout(model, cell, t, D, z_best, N, runs, cutoff=eval_cutoff(t, cutoff), chi_max=1024,
                             processes=heldout_processes)
            rec = {"D": D, "z": [float(v) for v in z_best], "c_train": float(i_best["c_after"]), "c_heldout": c_held,
                   "start": s_best, "optimizer": o_best, "tol_rel": tol_rel, "train_cutoff": cutoff, "train_chi": chi,
                   "heldout_cutoff": eval_cutoff(t, cutoff), "heldout_seconds": time.time() - t0, "seconds": time.time() - t_depth,
                   "workers": [groups or n_inputs, replicas],
                   "runs": {f"{s}/{o}": _start_record(i, sec) for (s, o), _z, i, sec in res}}
            log(f"D={D} best={s_best}/{o_best} c_train={rec['c_train']:.3e} c_heldout={c_held:.3e} | "
                + ", ".join(f"{s}/{o} {i['c_after']:.3e} ({i['steps']} it, {i['stop']})" for (s, o), _z, i, _ in res))
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


def precompute_chain_runs(model: XXZ, t: float, depths: list[int], runs: Path = Path("runs"), N: int = 100,
                          n_inputs: int = 12, groups: int | None = None, replicas: int = 1,
                          ref_cache: Path | None = None, starts=("trotter", "random"), optimizers=OPTIMIZERS,
                          tasks: int = 4, deadline: float | None = None, tol_rel: float = TOL_REL,
                          local: bool = False, log=print, newton_mode: str = "full") -> int:
    """Run ahead of the ladder: the starts that do not depend on the previous depth, at every depth not yet done.

    The trotter and random starts of a depth are the same start points the ladder builds, so their finished
    checkpoints (chain/ckpt/D<D>_<start>_<optimizer>_c<cutoff>.json) are what the ladder would compute; the
    ladder takes them over (it waits on a run's lock while a worker holds it). ``tasks`` depths at once, in
    order of depth, each with all its pending runs at once on one ChainProblem of groups x replicas workers. The truncation is cutoff_for at the
    deepest finished depth below; at t >= 4 the ladder may later pick the tighter cutoff, and then it recomputes
    (the file names carry the cutoff). Returns the number of runs finished here.
    """
    from snapevo.mps_train import open_chain_problem
    from snapevo.train_chain import cutoff_for, train_refs, trotter_z

    cell = cell_for(model)
    out_dir = Path(runs) / model.tag / f"t{t:g}" / "chain"
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    vecs, targets = open_chain_problem(model, t, N, n_inputs, 0, ref_cache, processes=n_inputs, log=log, **train_refs(t))

    def ladder():
        p = out_dir / "ladder.json"
        try:
            return {int(r["D"]): r for r in json.loads(p.read_text())["results"]}
        except (OSError, ValueError, KeyError):
            return {}

    def plan():
        have = ladder()
        todo = []
        for D in sorted(depths):
            if D in have:
                continue
            below = [d for d in have if d < D]
            cutoff, chi = cutoff_for(t, have[max(below)]["c_train"] if below else None)
            zt = trotter_z(model, cell, t, D)
            z0 = {"trotter": zt, "random": zt + 0.1 * np.random.default_rng(D).normal(size=zt.size)}
            for s in starts:
                for o in optimizers:
                    ck = str(ck_dir / f"D{D}_{s}_{o}_c{cutoff:.0e}.json")
                    if _finished(ck, z0[s]) is None:
                        todo.append((D, s, o, cutoff, chi, z0[s], ck))
        return todo

    n_done = [0]
    claimed: set = set()
    claim_lock = __import__("threading").Lock()

    def one(P, D, s, o, z, ck, lk):
        lg = lambda m: log(f"  precompute D={D} {s}/{o}: {m}")             # noqa: E731
        try:
            if o == "newton":
                zz, info = _newton(newton_mode, z, P, lambda c: tol_rel * c, n_bulk=D * cell.n_angles, log=lg,
                                   checkpoint=ck, deadline=deadline, label="newton")
            else:
                zz, info = lbfgs_minimize(z, P.objective, tol_rel=tol_rel, checkpoint=ck, deadline=deadline, log=lg)
            _mark_done(ck, z, zz, info)
            n_done[0] += 1
        except DeadlineReached:
            pass
        except Exception as exc:                                        # noqa: BLE001
            lg(f"failed: {exc!r}")
        finally:
            lk.close()

    def work(_w):
        """Take the shallowest depth with runs left; run them all at once on one shared pool."""
        while True:
            if deadline is not None and time.time() > deadline:
                return
            by_depth: dict = {}
            for D, s, o, cutoff, chi, z, ck in plan():
                by_depth.setdefault((D, cutoff, chi), []).append((s, o, z, ck))
            picked = None
            for (D, cutoff, chi), items in sorted(by_depth.items()):
                with claim_lock:
                    if (D, cutoff) in claimed:
                        continue
                    mine = []
                    for s, o, z, ck in items:
                        lk = _run_lock(ck, False, deadline)
                        if lk is None:
                            continue
                        if _finished(ck, z) is not None:
                            lk.close()
                            continue
                        mine.append((s, o, z, ck, lk))
                    if mine:
                        claimed.add((D, cutoff))
                        picked = (D, cutoff, chi, mine)
                        break
            if picked is None:
                return
            D, cutoff, chi, mine = picked
            try:
                with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=chi, cutoff=cutoff, groups=groups,
                                  replicas=replicas, local=local) as P:
                    with cf.ThreadPoolExecutor(len(mine)) as ex:
                        list(ex.map(lambda it: one(P, D, *it), mine))
            except Exception as exc:                                    # noqa: BLE001
                log(f"  precompute D={D}: failed: {exc!r}")
                for *_x, lk in mine:
                    lk.close()

    with cf.ThreadPoolExecutor(max(1, int(tasks))) as ex:
        list(ex.map(work, range(max(1, int(tasks)))))
    log(f"precompute: {n_done[0]} runs finished here")
    return n_done[0]
