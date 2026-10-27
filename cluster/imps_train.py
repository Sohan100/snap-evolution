"""Train one depth ladder on the infinite chain (2-site cell) by trust-region Newton (snapevo.imps_train).

    python cluster/imps_train.py FIELD T [--depths 2,3,5,...] [--workers 8] [--n-test 100] [--deadline EPOCH]
                                 [--runs DIR] [--serial-starts] [--optimizer newton|lbfgs]
                                 [--only-start trotter|warm|random] [--no-lock]

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, J = 1, spin units, h_x = FIELD); T is the evolution time. The depths
default to cluster/make_jobs.py's DEPTHS[T] (pass --depths for other times). Inputs: 4 training and --n-test test
random chi = 2 iMPS on the 2-site cell; references by iTEBD, cached in $SCRATCH/snapevo-scratch/imps_train/refs/.
Every start of every depth is minimized by trust-region Newton on the difference Hessian of the exact gradient,
with no step cap; the three starts of a depth run at once and share the --workers processes, which evaluate
one (input, point) each. Output runs/<tag>/t<T>/imps_train/ladder.json, checkpoints in imps_train/ckpt/; a rerun
resumes both. At --deadline every run saves its checkpoint and the ladder stops (exit 3). --optimizer lbfgs is the
comparison arm (L-BFGS-B steps, the Hessian kept for the start, the convergence test and the end), written to
imps_train*_lbfgs/. --only-start trains one start of the next depth and exits (cluster/imps_split.py runs the
three on three nodes); --no-lock is for runs under that driver, which holds the ladder lock.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy loads
    os.environ.setdefault(_v, "1")

import argparse                                                             # noqa: E402
import importlib.util                                                       # noqa: E402
import sys                                                                  # noqa: E402
import time                                                                 # noqa: E402
from pathlib import Path                                                    # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def depth_grid(t: float) -> list[int]:
    spec = importlib.util.spec_from_file_location("_make_jobs", REPO / "cluster" / "make_jobs.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    if int(t) != t or int(t) not in mj.DEPTHS:
        raise SystemExit(f"no depth grid for t = {t:g} in make_jobs.DEPTHS; pass --depths")
    return [int(d) for d in mj.DEPTHS[int(t)].split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("--depths", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-test", type=int, default=100, dest="n_test")
    ap.add_argument("--n-train", type=int, default=4, dest="n_train",
                    help="training states (4 writes imps_train/, others imps_train_n<N>/)")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: checkpoint and stop after")
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--serial-starts", action="store_true", dest="serial")
    ap.add_argument("--optimizer", choices=("newton", "lbfgs"), default="newton")
    ap.add_argument("--only-start", choices=("trotter", "warm", "random"), default=None, dest="only_start")
    ap.add_argument("--no-lock", action="store_true", dest="no_lock")
    a = ap.parse_args()

    from snapevo.hamiltonian import XXZ
    from snapevo.imps_train import train_imps_ladder

    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, a.field)
    depths = [int(d) for d in a.depths.split(",")] if a.depths else depth_grid(t)

    def log(s):
        who = f" {a.only_start}" if a.only_start else ""
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g}{who}] {s}", flush=True)

    log(f"imps ladder: pid {os.getpid()}, depths {depths}, workers {a.workers}, n_train {a.n_train}, n_test {a.n_test}, "
        f"optimizer {a.optimizer}, only start {a.only_start}, deadline {a.deadline}")
    t0 = time.time()
    out = train_imps_ladder(model, t, depths, runs=Path(a.runs), workers=a.workers, n_train=a.n_train, n_test=a.n_test,
                            deadline=a.deadline, parallel_starts=not a.serial, optimizer=a.optimizer,
                            only_start=a.only_start, hold_lock=not a.no_lock, log=log)
    stopped = a.deadline is not None and time.time() > a.deadline
    log(f"imps ladder: {'stopped at the deadline, rerun to continue' if stopped else 'finished'}: {out} "
        f"after {time.time() - t0:.0f}s")
    return 3 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
