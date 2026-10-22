"""Open-chain arm: train one depth ladder on the 100-site open chain only, with no iteration cap.

    python cluster/chain_train.py FIELD T [--workers W] [--optimizer best|newton] [--depths D,D,..] [--deadline EPOCH]
    python cluster/chain_train.py FIELD T --precompute [--tasks K] [--workers W] [--deadline EPOCH]

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, J = 1, spin units, h_x = FIELD). No ring is built or read: the
starts are Trotter, the previous depth of this ladder padded with identity sublayers, and Trotter plus noise.
Sohan, 17 Oct 2026: no cap anywhere, and the method that gives the lowest T. With --optimizer best (default,
snapevo.train_chain_best) every start is optimized both by trust-region Newton on the finite-difference Hessian
and by L-BFGS-B, each until it has converged (c falls by less than 1e-6 x c) or reaches the noise floor, and the
depth keeps the lowest c, recording which start and optimizer won; --optimizer newton runs the Newton trainer
alone (snapevo.train_newton). The depth grid is the same list of integers the ring ladders use, so the arms
compare depth by depth; the stop rule is the extended ring ladders' (stall 1.5 over 8 sublayers, target 1e-10).
Writes runs/<tag>/t<T>/chain/ladder.json, per-run checkpoints in runs/<tag>/t<T>/chain/ckpt/ (a rerun resumes),
and reads or builds the training references in $SCRATCH/snapevo-scratch/open_train/refs (dt 0.1, cutoff 1e-14,
chi 2048). The depth's runs share one pool of W workers (12 input groups x W // 12 replicas); a Hessian spreads
its 2 len(z) gradients over all of them.

--precompute runs ahead of the ladder (train_chain_best.precompute_chain_runs): the trotter and random starts of
every depth not yet in the ladder, by both optimizers, K depths at once, each depth on one pool of W workers. These
are the ladder's own runs (same start points and checkpoints); the ladder takes the finished ones over and
waits on any a precompute worker holds, so extra nodes shorten a ladder without changing its result.
"""

import argparse
import os
import time
from pathlib import Path

DEPTHS = {
    1: [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29, 33],
    2: [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 29, 33, 37, 41, 45, 49, 57],
    4: [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 73, 81],
    6: [3, 5, 7, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 73, 81, 89, 97, 105],
    8: [5, 7, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145],
    10: [5, 7, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137,
         145, 153, 161],
}
REF_CACHE = Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch" / "open_train" / "refs"
INPUTS = 12


def pool_shape(workers: int) -> tuple[int, int]:
    """(groups, replicas) of a pool of about ``workers`` processes over the 12 training inputs."""
    if workers >= INPUTS:
        return INPUTS, max(1, workers // INPUTS)
    return max(1, workers), 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("--workers", type=int, default=24, help="worker pool of the ladder (default 24)")
    ap.add_argument("--optimizer", choices=("best", "newton"), default="best")
    ap.add_argument("--depths", default=None, help="comma-separated depths (default: the grid for this t)")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: checkpoint and stop after")
    ap.add_argument("--tol-rel", type=float, default=1e-6, dest="tol_rel")
    ap.add_argument("--optimizers", default="newton",
                    help="optimizers per start, comma-separated (default newton: on 18 Oct uncapped L-BFGS never beat "
                         "Newton's floor at t = 1, D = 2-9 and took up to 2.7 h per run against minutes)")
    ap.add_argument("--newton-mode", default="reuse", choices=("full", "reuse", "cg"), dest="newton_mode",
                    help="Newton runs (snapevo.newton_fast): reuse (default since 20 Oct: the last finite-difference "
                         "Hessian reused, refreshed on a poor step ratio, every 10 steps and to confirm convergence; "
                         "same floors as full at t = 2, D = 11 with half the Hessians), full (a fresh Hessian per "
                         "step), cg (truncated Newton-CG); checkpoints are shared, so runs switch mode on resume")
    ap.add_argument("--precompute", action="store_true", help="run the depth-independent starts ahead of the ladder")
    ap.add_argument("--tasks", type=int, default=4, help="precompute depths at once (each on one pool of --workers workers)")
    a = ap.parse_args()

    from snapevo.hamiltonian import XXZ
    from snapevo.train_chain_best import precompute_chain_runs, train_chain_ladder_best
    from snapevo.train_newton import train_chain_ladder_newton

    opts = tuple(o.strip() for o in a.optimizers.split(",") if o.strip())
    t = int(a.t) if float(a.t).is_integer() else a.t
    depths = [int(x) for x in a.depths.split(",")] if a.depths else DEPTHS[t]
    model = XXZ(0.25, a.field)
    groups, replicas = pool_shape(a.workers)
    t0 = time.time()
    log = lambda s: print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g}] {s}", flush=True)   # noqa: E731
    if a.precompute:
        log(f"precompute: depths {depths}, {a.tasks} runs at once, each on {groups} x {replicas} workers, pid {os.getpid()}")
        precompute_chain_runs(model, float(t), depths, runs=Path(__file__).resolve().parent.parent / "runs",
                              groups=groups, replicas=replicas, ref_cache=REF_CACHE, tasks=a.tasks,
                              deadline=a.deadline, tol_rel=a.tol_rel, optimizers=opts, log=log,
                              newton_mode=a.newton_mode)
        log(f"precompute: stopped after {time.time() - t0:.0f}s")
        return
    log(f"chain ladder ({a.optimizer}, Newton mode {a.newton_mode}): depths {depths}, pool {groups} x {replicas} workers, tolerance "
        f"{a.tol_rel:g} x c, no iteration cap, pid {os.getpid()}")
    kw = dict(runs=Path(__file__).resolve().parent.parent / "runs", groups=groups, replicas=replicas,
              ref_cache=REF_CACHE, stall=(1.5, 8), target=1e-10, deadline=a.deadline, tol_rel=a.tol_rel, log=log)
    if a.optimizer == "best":
        log(f"optimizers per start: {', '.join(opts)}")
        try:
            train_chain_ladder_best(model, float(t), depths, optimizers=opts, newton_mode=a.newton_mode, **kw)
        except RuntimeError as exc:
            if "another run holds" not in str(exc):
                raise
            log(f"{exc}: this job runs ahead of that ladder instead (precompute, {a.tasks} depths at once)")
            precompute_chain_runs(model, float(t), depths, runs=kw["runs"], groups=groups, replicas=replicas,
                                  ref_cache=REF_CACHE, tasks=a.tasks, deadline=a.deadline, tol_rel=a.tol_rel,
                                  optimizers=opts, log=log, newton_mode=a.newton_mode)
    else:
        train_chain_ladder_newton(model, float(t), depths, **kw)
    log(f"chain ladder: stopped after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
