"""Train the translation-invariant cells on rings of different sizes and see how their error carries over.

    python cluster/size_transfer.py targets FIELD T          # exact ring targets for the evaluation
    python cluster/size_transfer.py task NAME                # one training and its evaluation (see tasks())
    python cluster/size_transfer.py list                     # every task name
    python cluster/size_transfer.py run [WORKERS]            # everything on one node (size_transfer.slurm)
    python cluster/size_transfer.py table                    # the results as markdown, with paired errors

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, J = 1, spin units, h_x = FIELD). Every training uses the
production cost (per-site infidelity over the 12 training product states of circuits.TRAIN_SET) and
train.py's L-BFGS-B options. Three kinds of task:

    warm     the production cell of depth D, trained on the 10-site ring, retrained for exactly WARM_ITERS
             iterations on a ring of N_TRAIN sites (8 to 16). The N = 10 row is the control: production
             plus the same retraining. Same start and same effort at every size, so the rows differ only in
             the ring they were trained on. This is the main measurement.
    scratch  from-scratch training on rings of 8, 10 and 12 sites, from three starts (the first-order
             Trotter angles, and those plus 0.1 rad of noise with seeds 1 and 2), to SCRATCH_ITERS
             iterations or convergence. Reported as best and spread over the starts.
    rev16    the 16-site warm cell retrained for WARM_ITERS at 10 sites, to see whether the larger ring
             moved the cell into another basin.

A low-floor warm point (h_x = 0, t = 2, D = 29, floor near 1e-9) runs at 10, 12 and 14 sites and is
evaluated at cutoff 1e-14. Training checkpoints every 15 min and resumes after a requeue, and the trained
angles are saved before the evaluation, so a preemption costs at most 15 min of training.

Each trained cell is then evaluated, unsnapped and unsynthesized:

    exactly on rings of 8 to 20 sites, by state vectors, against exact e^{-iHt} on the ring (scipy
        expm_multiply), with 64 held-out random product inputs per ring up to 16 sites and 16 at 18 and 20
        (seed 4242 + N, independent of the training, validation and production sets), per input;
    at N = 100 on the open chain by MPS against the sweep's cached order-4 references (the production 8
        inputs, cutoff 1e-12, chi 1024, with the cutoff/100 check), always with the edge rotations fitted
        as in production (edges.fit_edges at its default N_fit), as continuous angles.

Targets go to $SCRATCH/snapevo-scratch/size_transfer/, results to runs/size_transfer/<name>.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch" / "size_transfer"
OUT = Path(os.environ.get("ST_OUT", REPO / "runs" / "size_transfer"))     # ST_OUT and ST_NEVAL are for tests
FIELDS = {"1.6": "A", "0": "B"}            # snapevo's model keys: A is h_x = 1.6, B is h_x = 0
TIMES = (2, 4, 6)
DEPTHS = {                                 # per field and time, two production depths (their 10-site floors)
    ("1.6", 2): (11, 17), ("0", 2): (11, 17),      # 1.4e-6, 1.9e-7 | 9.2e-7, 1.6e-8
    ("1.6", 4): (21, 29), ("0", 4): (19, 29),      # 3.8e-6, 3.6e-7 | 1.1e-6, 7.0e-8
    ("1.6", 6): (25, 33), ("0", 6): (29, 37),      # 1.9e-5, 5.4e-6 | 9.0e-7, 2.3e-7
}
LOW_FLOOR = ("0", 2, 29, (10, 12, 14))     # floor near 1e-9, evaluated at cutoff 1e-14
N_WARM = (8, 10, 12, 14, 16)
N_SCRATCH = (8, 10, 12)
SCRATCH_STARTS = ("trotter", "noise1", "noise2")
WARM_ITERS = 3000                          # what the 16-site ring at D = 37 reaches in about 5 h on 2 cores
SCRATCH_ITERS = 30000
N_EVAL = tuple(int(v) for v in os.environ.get("ST_NEVAL", "8,10,12,14,16,18,20").split(","))
SAFETY_HOURS = 10.0                        # a training past this is stopped and marked invalid
CKPT_EVERY = float(os.environ.get("ST_CKPT", 900.0))   # seconds between training checkpoints (ST_CKPT for tests)
TRACE_EVERY = 250                          # iterations between (nit, C, seconds) trace points


def eval_set(N: int) -> tuple[int, int]:
    """(seed, n) of the held-out ring inputs at N sites."""
    return 4242 + N, (64 if N <= 16 else 16)


def tasks() -> list[str]:
    """Every task name, longest first (16-site rings, deep cells), the reverse checks last."""
    out = []
    for t in TIMES:
        for f in FIELDS:
            for D in DEPTHS[(f, t)]:
                out += [f"warm_{f}_{t}_{D}_{n}" for n in N_WARM]
                out += [f"scratch_{f}_{t}_{D}_{n}_{s}" for n in N_SCRATCH for s in SCRATCH_STARTS]
                out.append(f"rev16_{f}_{t}_{D}_10")
    f, t, D, ns = LOW_FLOOR
    out += [f"warm_{f}_{t}_{D}_{n}" for n in ns]

    def cost(name: str) -> float:
        kind, f, t, D, n, *_ = name.split("_")
        return 2.0 ** int(n) * int(D) * (WARM_ITERS if kind == "warm" else SCRATCH_ITERS / 3)
    return sorted(out, key=lambda k: (k.startswith("rev16"), -cost(k)))


def parse(name: str) -> dict:
    kind, f, t, D, n, *rest = name.split("_")
    return {"kind": kind, "field": f, "t": float(t), "D": int(D), "N_train": int(n),
            "start": rest[0] if rest else ("warm" if kind == "warm" else "rev16")}


def _model(field: str):
    from snapevo.hamiltonian import get_model
    return get_model(FIELDS[field])


def _target_path(field: str, t: float, N: int) -> Path:
    seed, n = eval_set(N)
    return SCRATCH / "targets" / f"hx{field}_t{t:g}_N{N}_s{seed}_n{n}.npz"


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as fh:                      # a file handle, so np.savez keeps the name
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def cmd_targets(field: str, t: str) -> None:
    from snapevo.circuits import state_pairs
    model, t = _model(field), float(t)
    for N in N_EVAL:
        p = _target_path(field, t, N)
        if p.exists():
            continue
        t0 = time.time()
        seed, n = eval_set(N)
        x, y = state_pairs(model, N, t, n, seed)
        _save_npz(p, inputs=x, targets=y)
        print(f"targets: h_x = {field}, t = {t:g}, N = {N}, {n} inputs, {time.time() - t0:.0f}s", flush=True)


def _wait(path: Path, hours: float = 3.0) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > hours * 3600:
            raise FileNotFoundError(f"{path} did not appear within {hours} h (did its targets job fail?)")
        time.sleep(30)


def _ring_eval(cell, x, field: str, t: float) -> dict:
    """Per-input per-site infidelity c_k = 1 - F_k^{1/N} on each held-out ring, and its mean and error."""
    from snapevo.circuits import Layered
    circ = Layered.brickwork(cell, x)
    out = {}
    for N in N_EVAL:
        p = _target_path(field, t, N)
        _wait(p)
        z = np.load(p)
        ov = np.sum(np.conj(z["targets"]) * circ.apply(z["inputs"], N), axis=1)
        c = 1.0 - (np.abs(ov) ** 2) ** (1.0 / N)
        out[str(N)] = {"mean": float(c.mean()), "se": float(c.std(ddof=1) / math.sqrt(c.size)), "c": c.tolist()}
    return out


def _check_refs(model, t: float, cutoff: float) -> Path:
    from snapevo import io
    from snapevo.evaluate import MPS_SET, ref_path
    d = io.run_dir(model.tag, t, create=False) / "ref"
    seed, n = MPS_SET
    need = [ref_path(d, 100, seed, k, 0.2, cutoff, 1024) for k in range(n)] + [ref_path(d, 100, seed, 0, 0.2, cutoff / 100, 1024)]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cached references missing in {d}: {missing}; this experiment does not build them")
    return d


def _train(name: str, f, x0: np.ndarray, maxiter: int, log) -> dict:
    """L-BFGS-B with train.py's options for maxiter iterations, checkpointed and resumable."""
    import scipy.optimize as sopt
    ck = (OUT / "ckpt" if "ST_OUT" in os.environ else SCRATCH / "ckpt") / f"{name}.json"
    st = json.loads(ck.read_text()) if ck.exists() else {"x": x0.tolist(), "nit": 0, "seconds": 0.0, "resumes": 0,
                                                          "trace": [], "nfev": 0, "invalid": False}
    if ck.exists() and not st.get("done"):
        st["resumes"] += 1
        log(f"train: resuming {name} at iteration {st['nit']} ({st['seconds']:.0f}s so far)")
    x = np.asarray(st["x"], dtype=float)
    t_start, last_ck = time.time(), time.time()
    base_s, base_nit = st["seconds"], st["nit"]
    it = [0]

    def save(xk, fun):
        st.update({"x": np.asarray(xk).tolist(), "fun": float(fun), "nit": base_nit + it[0],
                   "seconds": base_s + time.time() - t_start})
        _save_json(ck, st)

    def callback(intermediate_result):
        nonlocal last_ck
        it[0] += 1
        n = base_nit + it[0]
        if n % TRACE_EVERY == 0:
            st["trace"].append([n, float(intermediate_result.fun), base_s + time.time() - t_start])
        if time.time() - last_ck > CKPT_EVERY:
            save(intermediate_result.x, intermediate_result.fun)
            last_ck = time.time()
        if base_s + time.time() - t_start > SAFETY_HOURS * 3600:
            st["invalid"] = True
            raise StopIteration

    remaining = maxiter - base_nit
    if remaining > 0 and not st.get("done"):
        res = sopt.minimize(f, x, jac=True, method="L-BFGS-B", callback=callback,
                            options={"maxiter": remaining, "ftol": 1e-16, "gtol": 1e-14, "maxfun": 400000})
        st["nfev"] += int(res.nfev)
        st["message"] = str(res.message)
        save(res.x, res.fun)
        st["done"] = True
        _save_json(ck, st)
    return st


def cmd_task(name: str) -> None:
    from snapevo import io
    from snapevo.cells import cell_for
    from snapevo.circuits import TRAIN_SET, Layered, local_infidelity, state_pairs
    from snapevo.edges import fit_edges, pairs, worth_it
    from snapevo.evaluate import mps_infidelity
    from snapevo.jaxcost import make_cost
    from snapevo.train import VAL_SET, cell_records

    log = lambda s: print(s, flush=True)
    p = parse(name)
    out_path = OUT / f"{name}.json"
    if out_path.exists():
        log(f"skip: {out_path.name} exists")
        return
    model, t, D, N_train = _model(p["field"]), p["t"], p["D"], p["N_train"]
    cell = cell_for(model)
    low = (p["field"], int(t), D) == LOW_FLOOR[:3]
    cutoff = 1e-14 if low else 1e-12
    ref_dir = _check_refs(model, t, cutoff)
    d = io.run_dir(model.tag, t, create=False)
    prod = {int(r["D"]): r for r in cell_records(d / "cells.json")}[D]
    x_prod = np.asarray(prod["params"], dtype=float).reshape(-1)

    if p["kind"] == "warm":
        x0, maxiter = x_prod, WARM_ITERS
    elif p["kind"] == "rev16":
        src = OUT / f"warm_{p['field']}_{t:g}_{D}_16.json"
        _wait(src, hours=24.0)
        x0, maxiter = np.asarray(json.loads(src.read_text())["params"], dtype=float), WARM_ITERS
    else:
        x0 = cell.trotter_angles(model, t, D).reshape(-1)
        if p["start"] != "trotter":
            x0 = x0 + 0.1 * np.random.default_rng(int(p["start"][-1])).normal(size=x0.size)
        maxiter = SCRATCH_ITERS

    tin, tg = state_pairs(model, N_train, t, TRAIN_SET[1], TRAIN_SET[0])
    f = make_cost(cell, N_train, tin, tg, D)
    c_start = float(f(x0)[0])
    setup_ok = None
    if p["kind"] == "warm" and N_train == 10:             # the start must reproduce the production cost
        setup_ok = abs(c_start / prod["C_train"] - 1) < 1e-8
        log(f"setup check: C at the production cell {c_start:.10e} against production {prod['C_train']:.10e}: "
            f"{'ok' if setup_ok else 'MISMATCH'}")
        if not setup_ok:
            raise RuntimeError("the 10-site cost does not reproduce the production C_train")
    st = _train(name, f, x0, maxiter, log)
    x = np.asarray(st["x"], dtype=float)
    vin, vtg = state_pairs(model, N_train, t, VAL_SET[1], VAL_SET[0])
    c_train = float(f(x)[0])
    c_val = local_infidelity(Layered.brickwork(cell, x).apply(vin, N_train), vtg, N_train)
    log(f"train: {name}: C {c_start:.3e} -> {c_train:.3e} (val {c_val:.3e}), nit {st['nit']}, "
        f"{st['seconds']:.0f}s, resumes {st['resumes']}{' INVALID (safety stop)' if st['invalid'] else ''}")

    t1 = time.time()
    ring = _ring_eval(cell, x, p["field"], t)
    log("rings: " + ", ".join(f"N={k}: {v['mean']:.3e} +- {v['se'] / v['mean']:.0%}" for k, v in ring.items())
        + f" ({time.time() - t1:.0f}s)")

    circ = Layered.brickwork(cell, x)
    fit = fit_edges(circ, model, t, log=log)
    mps = mps_infidelity(circ, pairs(fit["angles"]), model, t, N=100, cutoff=cutoff, cache_dir=ref_dir, log=log)
    F = np.array([r["F"] for r in mps["rows"]])
    c100 = 1.0 - F ** (1.0 / 100)
    rec = {**p, "name": name, "model": model.to_dict(), "cell": cell.name,
           "prod_C_train": prod["C_train"], "prod_C_val": prod.get("C_val"), "setup_ok": setup_ok,
           "C_start": c_start, "C_train": c_train, "C_val": float(c_val),
           "train": {k: st[k] for k in ("nit", "nfev", "seconds", "resumes", "trace", "invalid", "message") if k in st},
           "maxiter": maxiter, "ring": ring,
           "edges": {**{k: fit[k] for k in ("N_fit", "infid_start", "infid_fit", "infid_ring", "nit")},
                     "worth_it": bool(worth_it(fit)), "angles": fit["angles"]},
           "mps100": mps, "c100": {"mean": float(mps["c_site"]), "se": float(c100.std(ddof=1) / math.sqrt(c100.size)),
                                   "c": c100.tolist()},
           "eval_seconds": time.time() - t1, "params": x.tolist()}
    _save_json(out_path, rec)
    log(f"done: {name}: c(N=100) = {mps['c_site']:.3e}")


def cmd_list() -> None:
    for n in tasks():
        print(n)


def cmd_run(workers: str = "60") -> None:
    """The six targets jobs on cores 120 to 125, the tasks on cores 0 to 119, 2 cores each, all pinned."""
    import queue
    import subprocess
    import threading
    from concurrent.futures import ThreadPoolExecutor
    logs = REPO / "cluster" / "logs" / "size_transfer"
    logs.mkdir(parents=True, exist_ok=True)
    for p in list(SCRATCH.rglob("*.tmp")) + list(OUT.glob("*.tmp")):   # leftovers of a killed write
        p.unlink(missing_ok=True)
    for f in FIELDS:                                                   # fail fast, before any training
        for t in TIMES:
            _check_refs(_model(f), float(t), 1e-12)
    _check_refs(_model(LOW_FLOOR[0]), float(LOW_FLOOR[1]), 1e-14)
    py = [str(REPO / ".venv" / "bin" / "python"), str(Path(__file__).resolve())]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"}

    def run(args: list[str], cpus: set[int]) -> int:
        with open(logs / ("_".join(args) + ".log"), "a") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] cpus {sorted(cpus)}: {' '.join(args)}\n")
            fh.flush()
            return subprocess.call(py + args, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                   preexec_fn=lambda: os.sched_setaffinity(0, cpus))

    targets = [(f, str(t)) for t in TIMES for f in FIELDS]
    tthreads = [threading.Thread(target=lambda a=a, i=i: print(f"targets {a}: rc {run(['targets', *a], {120 + i})}",
                                                               flush=True)) for i, a in enumerate(targets)]
    for th in tthreads:
        th.start()
    slots = queue.Queue()
    for i in range(int(workers)):
        slots.put({2 * i, 2 * i + 1})

    def job(name: str) -> int:
        cpus = slots.get()
        try:
            return run(["task", name], cpus)
        finally:
            slots.put(cpus)

    names = tasks()
    with ThreadPoolExecutor(int(workers)) as ex:
        rcs = list(ex.map(job, names))
    for th in tthreads:
        th.join()
    bad = [n for n, rc in zip(names, rcs) if rc != 0]
    print(f"tasks done, {len(names) - len(bad)} of {len(names)} succeeded" + (f"; failed: {bad}" if bad else ""), flush=True)


def _paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Relative paired difference mean(a - b) / mean(b) and its standard error."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    return float(d.mean() / b.mean()), float(d.std(ddof=1) / math.sqrt(d.size) / b.mean())


def cmd_table() -> None:
    rows = {r["name"]: r for r in (json.loads(p.read_text()) for p in sorted(OUT.glob("*.json")))}
    pct = lambda v, e: f"{v:+.1%} ± {e:.1%}"
    print("Warm: the production cell retrained for the same number of iterations on each ring. Paired differences "
          "against the 10-site row on the same inputs.\n")
    print("| h_x | t | D | ring | C_train | C_val | C ring(N_train) | c at N = 100 | c100 / C ring(N_train) - 1 | "
          "c100 vs 10-site | C ring 20 vs 10-site | train s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in sorted((n for n in rows if n.startswith("warm_")),
                       key=lambda n: (rows[n]["field"], rows[n]["t"], rows[n]["D"], rows[n]["N_train"])):
        r = rows[name]
        ref = rows.get(f"warm_{r['field']}_{r['t']:g}_{r['D']}_10")
        own = r["ring"][str(r["N_train"])]["mean"]
        dc = pct(*_paired(r["c100"]["c"], ref["c100"]["c"])) if ref and ref is not r else "-"
        dr = pct(*_paired(r["ring"]["20"]["c"], ref["ring"]["20"]["c"])) if ref and ref is not r and "20" in r["ring"] else "-"
        flag = " (invalid)" if r["train"].get("invalid") else ""
        print(f"| {r['field']} | {r['t']:g} | {r['D']} | {r['N_train']} | {r['C_train']:.2e} | {r['C_val']:.2e} | "
              f"{own:.2e} | {r['c100']['mean']:.2e} | {r['c100']['mean'] / own - 1:+.1%} | {dc} | {dr} | "
              f"{r['train']['seconds']:.0f}{flag} |")
    print("\nScratch: best and spread over the three starts.\n")
    print("| h_x | t | D | ring | best C_train | spread (max / min) | c at N = 100 (best) |")
    print("|---|---|---|---|---|---|---|")
    groups = {}
    for n, r in rows.items():
        if n.startswith("scratch_"):
            groups.setdefault((r["field"], r["t"], r["D"], r["N_train"]), []).append(r)
    for k in sorted(groups):
        g = sorted(groups[k], key=lambda r: r["C_train"])
        print(f"| {k[0]} | {k[1]:g} | {k[2]} | {k[3]} | {g[0]['C_train']:.2e} | {g[-1]['C_train'] / g[0]['C_train']:.2f} "
              f"({len(g)} starts) | {g[0]['c100']['mean']:.2e} |")
    print("\nReverse: the 16-site warm cell retrained at 10 sites.\n")
    for n, r in sorted(rows.items()):
        if n.startswith("rev16_"):
            base = n.replace("rev16_", "warm_")
            w16, w10 = rows.get(base.rsplit("_", 1)[0] + "_16"), rows.get(base)
            if w16 and w10:
                print(f"- h_x = {r['field']}, t = {r['t']:g}, D = {r['D']}: C_train {r['C_train']:.3e} "
                      f"(16-site cell at 10 sites {w16['ring']['10']['mean']:.3e}, 10-site control {w10['C_train']:.3e}), "
                      f"c100 {r['c100']['mean']:.3e}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["list"]
    {"targets": cmd_targets, "task": cmd_task, "list": cmd_list, "table": cmd_table, "run": cmd_run}[cmd](*rest)
