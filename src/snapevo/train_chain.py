"""Train the brickwork depth ladder on the open chain only, by MPS: the open-chain arm of the protocol.

The ring arm (train.py) trains each depth on a 10-site ring with state vectors, and
edges.py fits the two end rotations afterwards. This module never builds a ring and
never reads a ring result: every start, every iteration and every number it reports
is on the open chain of N sites (N = 100), against MPS references of e^{-iHt} on that
chain. The bulk angles (one set per sublayer, shared by every bond of the sublayer)
and the edge angles (6 per parity-1 sublayer, the ZYZ angles of the gates on sites
N - 1 and 0, mps_train's layout) are trained together from the start.

Starts per depth, as in train.py, with the best of them after L-BFGS-B kept:

    trotter   the order-1 Trotter brickwork (cells.trotter_angles). On the open chain
              its end sites get the cell's single-qubit factor, which is what
              edges.start_angles gives.
    warm      the previous depth of this ladder, padded with identity sublayers
              (bulk and edge angles zero), as train.py pads.
    random    trotter plus 0.1 rad of noise on every angle, seeded by D.

Cost: c = 1 - mean_k F_k^{1/N} over product_vectors(N, n, seed 0), against order-4
references at dt 0.1, cutoff 1e-14, chi 2048 (mps_train.open_chain_problem; the
open_train cache already holds them for t <= 6), optimized by mps_train.train_open
(L-BFGS-B with train.py's tolerances). The circuit's truncation is (cutoff, chi_max)
per depth, chosen by ``cutoff_for`` from this ladder's own previous depth only.

Held-out: the sweep's 8 evaluation inputs (seed 2026) against its cached references
in runs/<tag>/t<T>/ref (dt 0.2), at the depth's training cutoff, i.e. the number
evaluate.py gives for the unrealized circuit at that cutoff.

The depth grid (cluster/chain_train.py) is the ring ladders' list of integers, so
the arms compare depth by depth; no ring angle is used. maxiter is far below the
ring ladders' 30,000 at t >= 4 (each chain iteration costs seconds), so deep floors
can be set by the iteration budget; each record stores its maxiter.

Output: runs/<tag>/t<T>/chain/ladder.json (one record per depth) and per-start
checkpoints in runs/<tag>/t<T>/chain/ckpt/, from which a killed run resumes.
"""

from __future__ import annotations

import concurrent.futures as cf
import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from snapevo.cells import Cell, cell_for
from snapevo.circuits import Layered, product_vectors
from snapevo.edges import start_angles
from snapevo.hamiltonian import XXZ

__all__ = ["trotter_z", "pad_z", "split_z", "cutoff_for", "heldout", "train_chain_ladder", "load_chain_cell",
           "TRAIN_SEED", "EVAL_SET", "EVAL_DT"]

TRAIN_SEED = 0
EVAL_SET = (2026, 8)
EVAL_DT = 0.2
STALL = (2.0, 4)


def split_z(cell: Cell, z, D: int) -> tuple[np.ndarray, np.ndarray]:
    """(bulk angles, flat, D * n_angles; edge angles, flat, 6 per parity-1 sublayer) of z."""
    z = np.asarray(z, dtype=float).reshape(-1)
    nb, ne = D * cell.n_angles, 6 * (D // 2)
    if z.size != nb + ne:
        raise ValueError(f"z has {z.size} entries, expected {nb + ne} for D = {D}")
    return z[:nb], z[nb:]


def trotter_z(model: XXZ, cell: Cell, t: float, D: int) -> np.ndarray:
    """The order-1 Trotter brickwork of depth D on the open chain, as z = (bulk, edges)."""
    x = cell.trotter_angles(model, t, D)
    return np.concatenate([x, start_angles(Layered.brickwork(cell, x))])


def pad_z(cell: Cell, z, D_old: int, D_new: int) -> np.ndarray:
    """z of depth D_old padded to D_new with identity sublayers (zero bulk angles, zero edge angles)."""
    if D_new < D_old:
        raise ValueError(f"cannot pad depth {D_old} down to {D_new}")
    x, e = split_z(cell, z, D_old)
    x = np.concatenate([x, np.zeros((D_new - D_old) * cell.n_angles)])
    e = np.concatenate([e, np.zeros(6 * (D_new // 2 - D_old // 2))])
    return np.concatenate([x, e])


def cutoff_for(t: float, prev_c: float | None) -> tuple[float, int]:
    """Training (cutoff, chi_max) of the circuit's MPS for the next depth.

    1e-12 and 1024, the sweep's default evaluation settings; 1e-14 and 1024 once this
    ladder's previous depth is below 3e-9 per site, where 1e-12 truncation would be a
    visible share of c (the sweep switches its evaluation the same way for its 3e-9 and
    1e-9 budgets). At t <= 2 the bond dimension is small, so 1e-14 throughout. At t >= 8, 1e-10
    until the previous depth is below 3e-7 per site, then 1e-12.
    """
    if t >= 8:                     # t = 8, 10: 1e-12 costs 5-6x per gradient (cluster/open_train.py DEEP_TRAIN)
        return (1e-10, 1024) if prev_c is None or prev_c > 3e-7 else (1e-12, 1024)
    if t <= 2 or (prev_c is not None and prev_c < 3e-9):
        return 1e-14, 1024
    return 1e-12, 1024


def eval_cutoff(t: float, train_cutoff: float) -> float:
    """Cutoff of the held-out number: the training cutoff, except 1e-12 at t >= 8 (the sweep's cached references)."""
    return 1e-12 if t >= 8 else train_cutoff


def train_refs(t: float) -> dict:
    """Step, cutoff and chi_max of the training references: dt 0.1, 1e-14, 2048 (t <= 6); dt 0.1, 1e-12, 1024 at
    t >= 8, where 1e-14 and 2048 cost most of a day per input and the training cutoff is 1e-10 anyway."""
    if t >= 8:
        return {"dt": 0.1, "cutoff": 1e-12, "chi_max": 1024}
    return {"dt": 0.1, "cutoff": 1e-14, "chi_max": 2048}


def _eval_targets(model: XXZ, t: float, N: int, runs: Path, cutoff: float, chi_max: int):
    from snapevo.evaluate import ref_path
    from snapevo.mps import MPS
    d = Path(runs) / model.tag / f"t{t:g}" / "ref"
    seed, n = EVAL_SET
    paths = [ref_path(d, N, seed, k, EVAL_DT, cutoff, chi_max) for k in range(n)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"evaluation references missing in {d}: {missing}")
    return product_vectors(N, n, seed), [MPS.load(p)[0] for p in paths]


def heldout(model: XXZ, cell: Cell, t: float, D: int, z, N: int = 100, runs: Path = Path("runs"),
            cutoff: float = 1e-12, chi_max: int = 1024, processes: int = 8) -> float:
    """c of the circuit z on the sweep's 8 evaluation inputs against its cached references."""
    from snapevo.mps_train import OpenChainCost
    vecs, targets = _eval_targets(model, t, N, runs, cutoff, chi_max)
    with OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=chi_max, cutoff=cutoff,
                       processes=processes) as cost:
        return float(cost.value(z))


def load_chain_cell(model: XXZ, t: float, D: int, runs: Path = Path("runs")) -> tuple[np.ndarray, dict]:
    """(z, record) of depth D from runs/<tag>/t<T>/chain/ladder.json."""
    path = Path(runs) / model.tag / f"t{t:g}" / "chain" / "ladder.json"
    for r in json.loads(path.read_text())["results"]:
        if int(r["D"]) == D:
            return np.asarray(r["z"], dtype=float), r
    raise KeyError(f"{path} has no D = {D}")


def _write(path: Path, doc: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)


def _stalled(results: list[dict], D: int, factor: float, window: int) -> bool:
    lower = [r for r in results if int(r["D"]) <= D - window]
    here = [r for r in results if int(r["D"]) == D]
    if not lower or not here:
        return False
    return here[0]["c_train"] > max(lower, key=lambda r: r["D"])["c_train"] / factor


def train_chain_ladder(model: XXZ, t: float, depths: list[int], runs: Path = Path("runs"), N: int = 100,
                       n_inputs: int = 12, maxiter: int = 3000, processes: int = 12, threads: int = 1,
                       ref_cache: Path | None = None, starts=("trotter", "warm", "random"),
                       parallel_starts: bool = True, stall: tuple[float, int] | None = STALL,
                       target: float | None = None, deadline: float | None = None, log=print) -> Path:
    """Train every depth in ``depths`` (ascending) on the open chain, resuming what ``runs`` already holds.

    Each start runs mps_train.train_open with its own OpenChainCost over ``processes``
    workers; with ``parallel_starts`` the starts of one depth run at the same time.
    The ladder stops as train.py's does (``stall``: the floor fell by less than 2x over
    4 sublayers; ``target``: a floor at or below it). ``deadline`` (epoch seconds) stops
    the starts at the next iteration after it, leaving their checkpoints to resume from.
    """
    from snapevo.mps_train import OpenChainCost, open_chain_problem, train_open

    cell = cell_for(model)
    out_dir = Path(runs) / model.tag / f"t{t:g}" / "chain"
    ck_dir = out_dir / "ckpt"
    ck_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ladder.json"
    header = {"model": model.to_dict(), "cell": cell.name, "t": float(t), "N": N, "bc": "open",
              "train_set": [TRAIN_SEED, n_inputs], "eval_set": list(EVAL_SET), "eval_dt": EVAL_DT,
              "maxiter": maxiter, "starts": list(starts)}
    lock = open(out_dir / "lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RuntimeError(f"another run holds {out_dir / 'lock'}") from None
    if out.exists():
        doc = json.loads(out.read_text())
        for k in ("model", "cell", "t", "N", "bc", "train_set", "eval_set", "eval_dt"):
            if doc.get(k) != header[k]:
                raise ValueError(f"{out} has {k} = {doc.get(k)}, not {header[k]}")
        results = doc["results"]
    else:
        results = []
    for c_ in {cutoff_for(t, None)[0], cutoff_for(t, 1e-12)[0]}:            # every held-out setting we may use
        _eval_targets(model, t, N, runs, c_, 1024)
    vecs, targets = open_chain_problem(model, t, N, n_inputs, TRAIN_SEED, ref_cache, processes=processes, log=log)

    def hook(nit, x, c, info):
        if deadline is not None and time.time() > deadline:
            return "deadline"
        return None

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

        def run(name):
            t0 = time.time()
            try:
                with OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=chi, cutoff=cutoff,
                                   processes=processes, threads=threads) as cost:
                    res, trace = train_open(cost, z0[name], maxiter, log=lambda s: log(f"  D={D} {name}: {s}"),
                                            checkpoint=ck_dir / f"D{D}_{name}_c{cutoff:.0e}.json",
                                            on_iteration=hook)
            except Exception as exc:                                     # noqa: BLE001
                log(f"  D={D} {name}: failed: {exc!r}")
                return name, None, None, time.time() - t0
            return name, res, trace, time.time() - t0

        if parallel_starts and len(z0) > 1:
            with cf.ThreadPoolExecutor(len(z0)) as ex:
                runs_ = list(ex.map(run, list(z0)))
        else:
            runs_ = [run(name) for name in z0]
        failed = [n_ for n_, r, _, _ in runs_ if r is None]
        runs_ = [x for x in runs_ if x[1] is not None]
        if any(r.stop == "hook: deadline" for _, r, _, _ in runs_):
            log(f"D={D}: deadline reached during training, checkpoints kept")
            break
        if failed:
            log(f"D={D}: start(s) {failed} failed; stopping the ladder so the depth is not recorded without them")
            break
        name, best, trace, _ = min(runs_, key=lambda r: r[1].fun)
        t0 = time.time()
        c_held = heldout(model, cell, t, D, best.x, N, runs, cutoff=cutoff, chi_max=1024, processes=processes)
        rec = {"D": D, "z": [float(v) for v in best.x], "c_train": float(best.fun), "c_heldout": c_held,
               "start": name, "train_cutoff": cutoff, "train_chi": chi, "maxiter": maxiter,
               "heldout_cutoff": cutoff, "heldout_seconds": time.time() - t0,
               "starts": {n_: {"c": float(r.fun), "nit": int(r.nit), "nfev": int(r.nfev), "stop": r.stop,
                               "seconds": s} for n_, r, _, s in runs_},
               "trace": trace}
        log(f"D={D} best={name} c_train={best.fun:.3e} c_heldout={c_held:.3e} "
            + ", ".join(f"{n_} {r.fun:.3e} ({r.nit} it, {r.stop})" for n_, r, _, _ in runs_))
        results = [r for r in results if int(r["D"]) != D] + [rec]
        results.sort(key=lambda r: r["D"])
        _write(out, {**header, "results": results})
        prev = rec
        if target is not None and best.fun <= target:
            log(f"D={D}: floor {best.fun:.2e} is at or below the target {target:.0e}, stopping the ladder")
            break
        if stall is not None and _stalled(results, D, *stall):
            log(f"D={D}: floor fell by less than {stall[0]:g}x over {stall[1]} sublayers, stopping the ladder")
            break
    return out
