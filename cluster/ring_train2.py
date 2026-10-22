"""SUPERSEDED by cluster/ring_train.py (best of Newton and uncapped L-BFGS-B per start, Sohan 18 Oct); refuses to run,
since it would write the same runs/<tag>/t<T>/ring2/ladder.json. Kept for reference:

Ring arm, retrained with the protocol's optimizer: one depth ladder on the 10-site ring by trust-region Newton.

    python cluster/ring_train2.py FIELD T [--depths D,D,..] [--deadline EPOCH] [--hessian jax|auto] [--runs DIR]

train.train_ladder's problem (10-site ring, production training and validation sets) and starts, each optimized
by trust-region Newton on jaxcost's exact Hessian (snapevo.train_newton.train_ring_ladder_newton) until the
predicted Newton decrease and the last step's are below 1e-6 x c, or at the noise floor (no iteration cap);
stall 1.5 over 8 sublayers, target 1e-10. Writes runs/<tag>/t<T>/ring2/ladder.json in cells.json's format (the
sweep's cells.json is not touched); cluster/chain_snap.py --arm ring --ring-cells ring2/ladder.json compiles
from it. A rerun resumes each start from its checkpoint in ring2/ckpt/.
"""

import sys
import argparse
import os
import time
from pathlib import Path

DEPTHS = {t: [int(d) for d in v.split(",")] for t, v in {     # cluster/make_jobs.py's ring ladder grid
    1: "2,3,5,7,9,11,13,15,17,19,21,23,25,29,33",
    2: "2,3,5,7,9,11,13,15,17,19,21,25,29,33,37,41,45,49,57,65",
    4: "3,5,7,9,11,13,15,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97",
    6: "3,5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121",
    8: "5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121,129,137,145",
    10: "5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81,89,97,105,113,121,129,137,145,153,161",
}.items()}


def main():
    sys.exit("cluster/ring_train2.py is superseded by cluster/ring_train.py (it writes the same ring2/ladder.json)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("--depths", default=None, help="comma-separated depths (default: the grid for this t)")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: checkpoint and stop after")
    ap.add_argument("--hessian", default="jax", choices=("jax", "auto"),
                    help="jax: exact autodiff; auto: central differences of the jitted gradient above 96 angles")
    ap.add_argument("--tol-rel", type=float, default=1e-6, dest="tol_rel")
    ap.add_argument("--runs", default=str(Path(__file__).resolve().parent.parent / "runs"))
    a = ap.parse_args()

    from snapevo.hamiltonian import XXZ
    from snapevo.train_newton import train_ring_ladder_newton

    t = int(a.t) if float(a.t).is_integer() else a.t
    depths = [int(x) for x in a.depths.split(",")] if a.depths else DEPTHS[t]
    model = XXZ(0.25, a.field)
    t0 = time.time()
    log = lambda s: print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g}] {s}", flush=True)   # noqa: E731
    log(f"ring ladder (Newton, {a.hessian} Hessian): depths {depths}, pid {os.getpid()}")
    train_ring_ladder_newton(model, float(t), depths, runs=Path(a.runs), deadline=a.deadline, tol_rel=a.tol_rel,
                             hessian_method=a.hessian, log=log)
    log(f"ring ladder: stopped after {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
