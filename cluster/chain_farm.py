"""Open-chain arm on a farm of workers spread over many nodes (CPU or GPU): ladders, precompute, compiles.

    python cluster/chain_farm.py ladder     FIELD T --farm-file F --farm-expected W [--depths ..] [--deadline EPOCH]
    python cluster/chain_farm.py precompute FIELD T --farm-file F --farm-expected W [--tasks K] [--deadline EPOCH]
    python cluster/chain_farm.py compile    FIELD T BUDGET[,BUDGET..] --farm-file F --farm-expected W [--D D] ...
    python cluster/chain_farm.py refs       FIELD T            (no farm: the 12 training references, one node)

The same calls as cluster/chain_train.py (snapevo.train_chain_best: train_chain_ladder_best, precompute_chain_runs,
Newton only by default, no caps, stall 1.5 over 8 sublayers, target 1e-10, the same checkpoints and ladder file)
and cluster/chain_snap.py (snapevo.chain_snap.chain_point), but every ChainProblem leases its pool from a farm
(snapevo.chain_farm): W worker processes started by srun on the job's nodes (cluster/chain_farm.slurm), which
connect to this master through the address file F. A pool has the 12 training inputs as groups and
farm size // (12 x pools at once) replicas, so a Newton step's 2 len(z) gradients run over the whole farm at
once. CPU workers run the CPU code; GPU workers (srun ... --gpu local) run the same functions on the GPU kernels
of snapevo.mps_gpu. Results are those of the one-node runs up to rounding (bit for bit on CPU workers).

Stopping: at --deadline (or SIGTERM) every run saves its checkpoint and the master exits 3 after telling the
workers to stop; a rerun (cluster/chain_farm.slurm requeues itself) continues every run where it stopped.
"""

import argparse
import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INPUTS = 12


def _chain_train():
    spec = importlib.util.spec_from_file_location("_chain_train", REPO / "cluster" / "chain_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _bench(model, t, a, K: int, log) -> None:
    """Time one value_and_grad and one full finite-difference Hessian on the farm (a sizing measurement).

    Circuit: the production ring cell at depth --D with its own edges; targets: the open_train training references
    at step 0.2 (inputs 0-5, cutoff 1e-12, chi 1024; already built for t = 8 and 10); training cutoff 1e-10.
    """
    import numpy as np

    from snapevo.cells import cell_for
    from snapevo.chain_snap import ChainProblem
    from snapevo.circuits import product_vectors
    from snapevo.evaluate import ref_path
    from snapevo.mps import MPS
    from snapevo.mps_train import init_from_ring, ring_angles
    cell = cell_for(model)
    D = a.D or 25
    n = 6
    ref = _chain_train().REF_CACHE / model.tag / f"t{t:g}"
    targets = [MPS.load(ref_path(ref, 100, 0, k, 0.2, 1e-12, 1024))[0] for k in range(n)]
    vecs = product_vectors(100, n, 0)
    z = init_from_ring(cell, ring_angles(model, t, D, REPO / "runs"))
    with ChainProblem(model, cell, float(t), D, 100, vecs, targets, chi_max=1024, cutoff=1e-10, groups=n) as P:
        t0 = time.time()
        c, g = P.objective(z)
        t1 = time.time()
        log(f"bench: D = {D}, {n} inputs, pool {P.pool.G} x {P.pool.K}: value_and_grad {t1 - t0:.1f}s, c = {c:.6e}")
        rows = list(range(z.size)) if a.depths is None else [int(x) for x in a.depths.split(",")]
        H = P.hessian(z, rows)
        t2 = time.time()
        log(f"bench: Hessian over {len(rows)} rows ({2 * len(rows)} gradients of {n} inputs) {t2 - t1:.1f}s; "
            f"asymmetry {P.stats['hessian'][-1]['asym']}, eigenvalues {np.linalg.eigvalsh(H)[[0, -1]]}")


def _wait_for_ladder_lock(chain_dir: Path, wait: float, stop: dict, log) -> bool:
    """True once no other process holds chain_dir/lock (at once when wait is 0: the ladder then raises as before)."""
    import fcntl
    if wait <= 0:
        return True
    end, said = time.time() + wait, False
    chain_dir.mkdir(parents=True, exist_ok=True)
    while True:
        with open(chain_dir / "lock", "w") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True                                  # released on close; the ladder takes it next
            except OSError:
                pass
        if time.time() > end or stop["flag"]:
            return False
        if not said:
            log(f"waiting up to {wait:.0f}s for another ladder to release {chain_dir / 'lock'}")
            said = True
        time.sleep(15)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("ladder", "precompute", "compile", "bench", "refs"))
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("budgets", nargs="?", default=None, help="compile: comma-separated budgets")
    ap.add_argument("--farm-file", default=None, dest="farm_file")
    ap.add_argument("--farm-expected", type=int, default=0, dest="farm_expected", help="workers srun starts")
    ap.add_argument("--ref-threads", type=int, default=10, dest="ref_threads",
                    help="refs: BLAS threads per reference (12 at once)")
    ap.add_argument("--farm-minimum", type=int, default=None, dest="farm_minimum",
                    help="start with at least this many workers if not all connect (default: all)")
    ap.add_argument("--farm-wait", type=float, default=1800.0, dest="farm_wait")
    ap.add_argument("--depths", default=None, help="comma-separated depths (default: cluster/chain_train.py DEPTHS)")
    ap.add_argument("--optimizers", default="newton")
    ap.add_argument("--newton-mode", default="reuse", choices=("full", "reuse", "cg"), dest="newton_mode",
                    help="ladder/precompute Newton runs: full (a fresh finite-difference Hessian per step), reuse "
                         "(the last one reused), cg (truncated Newton-CG with Hessian-vector products); "
                         "snapevo.newton_fast")
    ap.add_argument("--tol-rel", type=float, default=1e-6, dest="tol_rel")
    ap.add_argument("--wait-lock", type=float, default=0.0, dest="wait_lock",
                    help="ladder: wait up to this many seconds for another ladder of this field and t (a login-node "
                         "one being handed over) to release runs/<tag>/t<T>/chain/lock; still held then, run "
                         "precompute (--tasks) instead")
    ap.add_argument("--heldout-processes", type=int, default=8, dest="heldout_processes",
                    help="ladder: local processes of each depth's held-out evaluation (fewer where memory is short)")
    ap.add_argument("--tasks", type=int, default=2, help="precompute: depths at once (each on its share of the farm)")
    ap.add_argument("--deadline", type=float, default=None)
    # compile options (cluster/chain_snap.py's)
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--reopt", default="newton", choices=("newton", "lbfgs-converged", "coordinate"))
    ap.add_argument("--chain-out", default=None, dest="chain_out")
    ap.add_argument("--edge-zero-t", action="store_true", dest="edge_zero_t")
    ap.add_argument("--all-depths", action="store_true", dest="all_depths")
    ap.add_argument("--env-layers", type=int, default=None, dest="env_layers")
    ap.add_argument("--steps", default="compile,realize,evaluate")
    a = ap.parse_args()

    from snapevo.chain_farm import SocketFarm, use_farm
    from snapevo.hamiltonian import XXZ

    ct = _chain_train()
    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, a.field)
    runs = REPO / "runs"
    stop = {"flag": False}

    def log(s):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g} farm] {s}", flush=True)

    def on_term(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_term)
    if a.mode == "refs":
        from snapevo.mps_train import open_chain_problem
        from snapevo.train_chain import TRAIN_SEED, train_refs
        t0 = time.time()
        open_chain_problem(model, float(t), 100, INPUTS, TRAIN_SEED, ct.REF_CACHE, processes=INPUTS,
                           threads=a.ref_threads, log=log, **train_refs(float(t)))
        log(f"refs: the {INPUTS} training references of t = {t:g} are in {ct.REF_CACHE} ({time.time() - t0:.0f}s)")
        return 0
    if not a.farm_file or a.farm_expected < 1:
        ap.error("--farm-file and --farm-expected are required for this mode")
    farm = SocketFarm(a.farm_file, a.farm_expected, minimum=a.farm_minimum, wait=a.farm_wait, log=log)
    t0 = time.time()
    try:
        pools = a.tasks if a.mode == "precompute" else 1
        G = 6 if a.mode == "bench" else INPUTS
        K = max(1, farm.size // (G * pools))
        use_farm(farm, replicas=K, groups=G, deadline=a.deadline)
        log(f"{a.mode}: farm of {farm.size} workers, pools of {G} x {K}, pid {os.getpid()}")
        depths = [int(x) for x in a.depths.split(",")] if a.depths else ct.DEPTHS[t]
        opts = tuple(o.strip() for o in a.optimizers.split(",") if o.strip())
        if a.mode == "bench":
            _bench(model, t, a, K, log)
        elif a.mode == "ladder" and not _wait_for_ladder_lock(runs / model.tag / f"t{t:g}" / "chain", a.wait_lock,
                                                               stop, log):
            from snapevo.train_chain_best import precompute_chain_runs
            log(f"ladder lock still held after {a.wait_lock:.0f}s: precompute instead ({a.tasks} depths at once)")
            precompute_chain_runs(model, float(t), depths, runs=runs, groups=INPUTS, replicas=K,
                                  ref_cache=ct.REF_CACHE, tasks=a.tasks, deadline=a.deadline, tol_rel=a.tol_rel,
                                  optimizers=opts, log=log, newton_mode=a.newton_mode)
        elif a.mode == "ladder":
            from snapevo.train_chain_best import train_chain_ladder_best
            train_chain_ladder_best(model, float(t), depths, runs=runs, groups=INPUTS, replicas=K,
                                    ref_cache=ct.REF_CACHE, optimizers=opts, stall=(1.5, 8), target=1e-10,
                                    deadline=a.deadline, tol_rel=a.tol_rel, log=log, newton_mode=a.newton_mode,
                                    heldout_processes=a.heldout_processes)
        elif a.mode == "precompute":
            from snapevo.train_chain_best import precompute_chain_runs
            precompute_chain_runs(model, float(t), depths, runs=runs, groups=INPUTS, replicas=K,
                                  ref_cache=ct.REF_CACHE, tasks=a.tasks, deadline=a.deadline, tol_rel=a.tol_rel,
                                  optimizers=opts, log=log, newton_mode=a.newton_mode)
        else:
            from snapevo.chain_snap import DeadlineReached, chain_point
            spec = importlib.util.spec_from_file_location("_chain_snap_runner", REPO / "cluster" / "chain_snap.py")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            kw = {"reopt": a.reopt, "max_round_trials": 48}
            if a.edge_zero_t:
                kw["edge_zero_t"] = True
            for b in [float(x) for x in (a.budgets or "").split(",") if x]:
                try:
                    chain_point(model, float(t), b, runs, D=a.D, ref_cache=ct.REF_CACHE, groups=INPUTS, replicas=K,
                                settings_kw=kw, eval_kw=runner.eval_settings(a.field, float(t), b),
                                steps=tuple(a.steps.split(",")), deadline=a.deadline, env_layers=a.env_layers,
                                out_sub=a.chain_out, all_depths=a.all_depths, log=log)
                except DeadlineReached as exc:
                    log(f"compile: {exc}")
                    break
    finally:
        farm.close()
    late = stop["flag"] or (a.deadline is not None and time.time() > a.deadline)
    log(f"{a.mode}: {'stopped at the deadline or SIGTERM' if late else 'finished'} after {time.time() - t0:.0f}s")
    return 3 if late else 0


if __name__ == "__main__":
    sys.exit(main())
