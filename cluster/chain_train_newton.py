"""Open-chain arm, Newton trainer (on hold): one depth ladder on the 100-site open chain by trust-region Newton.

    python cluster/chain_train_newton.py FIELD T [--processes P | --workers W] [--depths D,D,..] [--deadline EPOCH]

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, J = 1, spin units, h_x = FIELD). No ring is built or read: the
starts are Trotter, the previous depth of this ladder padded with identity sublayers, and Trotter plus noise,
each optimized by trust-region Newton on the finite-difference Hessian of the open-chain cost
(snapevo.train_newton.train_chain_ladder_newton) until the predicted Newton decrease and the last step's are
below 1e-6 x c, or at the noise floor; there is no iteration cap. The depth grid is the same list of integers
the ring ladders use, so the two arms can be compared depth by depth; the stop rule is the extended ring
ladders' (stall 1.5 over 8 sublayers, target 1e-10). Writes runs/<tag>/t<T>/chain/ladder.json, checkpoints in
runs/<tag>/t<T>/chain/ckpt/ (a rerun resumes each start), and reads or builds the training references in
$SCRATCH/snapevo-scratch/open_train/refs (dt 0.1, cutoff 1e-14, chi 2048).

Workers: the three starts of a depth share one pool of W = 3 P workers (--processes P, cores per start as in
the L-BFGS runner) or --workers W, as 12 input groups (one training input each) x W // 12 replicas; a Hessian
spreads its 2 len(z) gradients over all of them.
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
    ap.add_argument("--processes", type=int, default=12, help="cores per start; the pool gets 3 x this (default 12)")
    ap.add_argument("--workers", type=int, default=None, help="workers of the depth's pool (overrides --processes)")
    ap.add_argument("--depths", default=None, help="comma-separated depths (default: the grid for this t)")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: checkpoint and stop after")
    ap.add_argument("--tol-rel", type=float, default=1e-6, dest="tol_rel")
    a = ap.parse_args()

    from snapevo.hamiltonian import XXZ
    from snapevo.train_newton import train_chain_ladder_newton

    t = int(a.t) if float(a.t).is_integer() else a.t
    depths = [int(x) for x in a.depths.split(",")] if a.depths else DEPTHS[t]
    model = XXZ(0.25, a.field)
    groups, replicas = pool_shape(a.workers or 3 * a.processes)
    t0 = time.time()
    log = lambda s: print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g}] {s}", flush=True)   # noqa: E731
    log(f"chain ladder (Newton): depths {depths}, pool {groups} x {replicas} workers, tolerance {a.tol_rel:g} x c, "
        f"pid {os.getpid()}")
    train_chain_ladder_newton(model, float(t), depths, runs=Path(__file__).resolve().parent.parent / "runs",
                              groups=groups, replicas=replicas, ref_cache=REF_CACHE, stall=(1.5, 8), target=1e-10,
                              deadline=a.deadline, tol_rel=a.tol_rel, log=log)
    log(f"chain ladder: stopped after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
