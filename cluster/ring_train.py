"""Train one ring ladder, best of Newton and uncapped L-BFGS-B per start (snapevo.train_ring_best).

    python cluster/ring_train.py FIELD T [--depths 2,3,5,...] [--workers 6] [--runs DIR] [--deadline EPOCH]
                                 [--optimizers newton --out-sub ring_newton]

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, h_x = FIELD); T is the evolution time. The depths default to the
production grid (cluster/make_jobs.py's DEPTHS[T] and every depth in runs/<tag>/t<T>/cells.json); the ladder
stops on the stall rule (1.5 over 8 sublayers) or at a floor of 1e-10, whichever comes first. The ladder is
runs/<tag>/t<T>/ring2/ladder.json (cells.json's format; cells.json itself is not touched), with one checkpoint
per start x optimizer in ring2/ckpt/. Each depth runs its 3 starts x 2 optimizers in --workers spawned
processes at once (JAX single-threaded in each: set XLA_FLAGS as cluster/ring_train.slurm does).

Stopping: at --deadline every run saves its checkpoint at its next cost evaluation and the runner exits 3; on
SIGTERM the runner forwards it to its workers, which do the same. A rerun continues every run where it stopped,
and depths already in the ladder are kept. Exit 0: the ladder is finished (stall, target or the last depth).
"""

import argparse
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("--depths", default=None, help="comma-separated depths (default: the production grid)")
    ap.add_argument("--workers", type=int, default=6, help="processes per depth (6 = every start x optimizer)")
    ap.add_argument("--hessian-workers", type=int, default=4, dest="hessian_workers",
                    help="helper processes per Newton run for the difference Hessian above 300 angles")
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: checkpoint and stop after")
    ap.add_argument("--target", type=float, default=1e-10)
    ap.add_argument("--stall", default="1.5,8")
    ap.add_argument("--optimizers", default="newton,lbfgs",
                    help="comma-separated, from newton and lbfgs (newton alone: the protocol's trust-region Newton)")
    ap.add_argument("--out-sub", default="ring2", dest="out_sub", help="runs/<tag>/t<T>/<out-sub>/ladder.json")
    a = ap.parse_args()

    from snapevo.hamiltonian import XXZ
    from snapevo.train_ring_best import depth_grid, train_ring_ladder_best

    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, a.field)
    runs = Path(a.runs)
    depths = [int(d) for d in a.depths.split(",")] if a.depths else depth_grid(model, t, runs)
    fac, win = a.stall.split(",")
    term = threading.Event()

    def on_term(*_):
        term.set()
        for p in mp.active_children():
            try:
                os.kill(p.pid, signal.SIGTERM)
            except OSError:
                pass
    signal.signal(signal.SIGTERM, on_term)

    def log(s):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g}] {s}", flush=True)

    log(f"ring ladder: pid {os.getpid()}, depths {depths}, workers {a.workers}, runs {runs}, deadline {a.deadline}")
    t0 = time.time()
    out = train_ring_ladder_best(model, t, depths, runs=runs, out_sub=a.out_sub,
                                 optimizers=tuple(v for v in a.optimizers.split(",") if v),
                                 stall=(float(fac), int(win)), target=a.target,
                                 deadline=a.deadline, workers=a.workers, hessian_workers=a.hessian_workers,
                                 stop=term, log=log)
    stopped = term.is_set() or (a.deadline is not None and time.time() > a.deadline)
    log(f"ring ladder: {'stopped (deadline or SIGTERM), rerun to continue' if stopped else 'finished'}: {out} "
        f"after {time.time() - t0:.0f}s")
    return 3 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
