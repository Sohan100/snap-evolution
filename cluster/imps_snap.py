"""SNAP and the uniform-precision synthesis of the infinite-chain ladders (snapevo.imps_snap), all on the infinite chain.

    python cluster/imps_snap.py snap    FIELD T [--n-train 4] [--optimizer newton] [--budgets 1e-1,...] [--workers 30]
    python cluster/imps_snap.py uniform FIELD T [--n-train 4] [--optimizer newton] [--eps 1e-2,...] [--workers 30]
    python cluster/imps_snap.py reeval  FIELD T [--n-train 4] [--optimizer newton] [--workers 30]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = FIELD (1.6 or 0). The ladder is
runs/<tag>/t<T>/imps_train[_n<N>][_lbfgs]/ladder.json: --optimizer is the ladder's training optimizer (lbfgs: the
comparison arm), and the outputs go to the matching imps_snap[_n<N>][_lbfgs] and imps_uniform[_n<N>][_lbfgs]. The
compile's re-optimizer is Newton either way (the protocol).

snap: imps_snap.ti_point for every budget (default BUDGETS: 1e-1 to 1e-6, two per decade), reopt = "newton" (the
protocol); records in runs/<tag>/t<T>/imps_snap[_n<N>]/.

uniform: Gibbs and Cincio's procedure (arXiv:2506.02105, Sec. III.4) on the same ladders: every rotation of the
trained circuit synthesized at one precision eps (default EPS: 1e-2 to 1e-7 in half decades), with KN (the
synthesis of every arm here; they use Ross-Selinger), no snapping, then the cost recomputed on the infinite chain:
the ladder's 100 test inputs, imps_snap.evaluate_ti. Every ladder depth whose floor is at most the largest eps;
records runs/<tag>/t<T>/imps_uniform[_n<N>]/real_e<eps>_D<D>.json and eval_e<eps>_D<D>.json.

reeval: imps_snap.evaluate_ti again for every real_*.json of both arms (after a change to the evaluation).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy loads
    os.environ.setdefault(_v, "1")

import argparse                                                             # noqa: E402
import math                                                                 # noqa: E402
import sys                                                                  # noqa: E402
import time                                                                 # noqa: E402
from pathlib import Path                                                    # noqa: E402

import numpy as np                                                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BUDGETS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
EPS = tuple(10.0 ** (-k / 2) for k in range(4, 15))                          # 1e-2 ... 1e-7


def _synth(args):
    from snapevo.synthesis import synthesize
    theta, eps = args
    synthesize(float(theta), float(eps), unitaries=True)
    return theta, eps


def uniform(model, t, n_train: int, eps_list, workers: int, log, optimizer: str = "newton") -> None:
    from snapevo import io
    from snapevo import imps_train as it
    from snapevo.cells import cell_for
    from snapevo.imps_snap import TIProblem, evaluate_ti
    from snapevo.realize import realize
    from snapevo.imps_snap import ladder_sub
    cell = cell_for(model)
    sub = ladder_sub(n_train, optimizer)
    d = REPO / "runs" / model.tag / f"t{float(t):g}"
    lad = io.read_json(d / sub / "ladder.json")
    out = d / sub.replace("imps_train", "imps_uniform")
    out.mkdir(parents=True, exist_ok=True)
    for rec in sorted(lad["results"], key=lambda r: r["D"]):
        D = int(rec["D"])
        z = np.asarray(rec["params"], dtype=float).reshape(-1)
        todo = [e for e in eps_list if rec["C_train"] <= max(eps_list) and
                not (out / f"eval_e{e:.1e}_D{D}.json").exists()]
        if not todo:
            continue
        with TIProblem(model, t, D, n_train=n_train, n_test=int(lad["val_set"][1]), workers=workers, log=log) as P:
            mult = [cell.multiplicity[i % cell.n_angles] for i in range(z.size)]
            # every synthesis on the pool first (smallest eps first: the slowest), so realize() only looks them up
            need = sorted({(float(th), e) for e in todo for th in z}, key=lambda a: a[1])
            t0 = time.time()
            list(P.P.pool.map(_synth, need, chunksize=1))
            log(f"uniform D={D}: {len(need)} syntheses on the pool in {time.time() - t0:.0f}s")
            for e in todo:
                compiled = {"angles": z.tolist(), "snapped": [False] * z.size, "multiplicity": mult,
                            "settings": {"budget": math.inf}, "stragglers": {str(i): {"eps": e} for i in range(z.size)}}
                real = realize(compiled, model, cell, t, c_fn=P.c_fn, log=log, max_tries=1)
                real.update({"model": model.to_dict(), "cell": cell.name, "D": D, "eps": e, "bc": "infinite",
                             "arm": "uniform"})
                ev = evaluate_ti(real, P, log=log)
                ev.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D, "eps": e, "arm": "uniform",
                           "ladder_C_train": rec["C_train"], "ladder_C_val": rec.get("C_val"),
                           "train_set": lad["train_set"]})
                io.write_json(out / f"real_e{e:.1e}_D{D}.json", real)
                io.write_json(out / f"eval_e{e:.1e}_D{D}.json", ev)


def reeval(model, t, n_train: int, workers: int, log, optimizer: str = "newton") -> None:
    from snapevo import io
    from snapevo.imps_snap import TIProblem, evaluate_ti, ladder_sub
    sub = ladder_sub(n_train, optimizer)
    d = REPO / "runs" / model.tag / f"t{float(t):g}"
    lad = io.read_json(d / sub / "ladder.json")
    jobs: dict = {}
    for arm in ("imps_snap", "imps_uniform"):
        for p in sorted((d / sub.replace("imps_train", arm)).glob("real_*.json")):
            real = io.read_json(p)
            jobs.setdefault(int(real["D"]), []).append((arm, p, real))
    for D, items in sorted(jobs.items()):
        with TIProblem(model, t, D, n_train=n_train, n_test=int(lad["val_set"][1]), workers=workers, log=log) as P:
            for arm, p, real in items:
                pe = p.with_name("eval_" + p.name[len("real_"):])
                old = io.read_json(pe) if pe.exists() else {}
                F = None
                if arm == "imps_snap":
                    ps = p.with_name("snap_" + p.name[len("real_"):])
                    F = io.read_json(ps).get("settings", {}).get("F") if ps.exists() else None
                ev = evaluate_ti(real, P, F=F, log=log)
                keep = {k: v for k, v in old.items() if k not in ev and k not in ("c_total", "fallback_share")}
                io.write_json(pe, {**keep, **ev})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("snap", "uniform", "reeval"))
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("--n-train", type=int, default=4, dest="n_train")
    ap.add_argument("--optimizer", choices=("newton", "lbfgs"), default="newton",
                    help="the ladder's training optimizer (the compile re-optimizes by Newton either way)")
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--eps", default=None)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--deadline", type=float, default=None)
    a = ap.parse_args()

    from snapevo.hamiltonian import XXZ
    from snapevo.imps_snap import ti_point

    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, a.field)

    def log(s):
        opt = "" if a.optimizer == "newton" else " lbfgs"
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g} n{a.n_train}{opt} {a.cmd}] {s}", flush=True)

    if a.cmd == "snap":
        budgets = [float(b) for b in a.budgets.split(",")] if a.budgets else list(BUDGETS)
        if not a.budgets and float(t) >= 4:
            # 1e-2 ... 1e-6 first: at t = 4 the loose budgets cost the most (their curves must be exact wherever the
            # cost is below 3 x 0.1, nearly the whole period, at seconds per eigenvalue), up to a day per point
            budgets = [b for b in budgets if b <= 1e-2] + [b for b in budgets if b > 1e-2]
        for b in budgets:
            if a.deadline is not None and time.time() > a.deadline:
                log("deadline: stop")
                sys.exit(3)
            t0 = time.time()
            res = ti_point(model, t, b, REPO / "runs", n_train=a.n_train, workers=a.workers,
                           settings_kw={"reopt": "newton"}, deadline=a.deadline, optimizer=a.optimizer, log=log)
            log(f"budget {b:.0e}: {len(res)} points in {time.time() - t0:.0f}s: " + ", ".join(
                f"D={r['D']} c={r['c']:.2e} T/q={r['Tq']:.1f}" for r in res))
    elif a.cmd == "uniform":
        eps = [float(e) for e in a.eps.split(",")] if a.eps else list(EPS)
        uniform(model, t, a.n_train, eps, a.workers, log, a.optimizer)
    else:
        reeval(model, t, a.n_train, a.workers, log, a.optimizer)


if __name__ == "__main__":
    main()
