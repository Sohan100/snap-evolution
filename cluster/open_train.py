"""Fine-tune the ring-trained ladder cells on the 100-site open chain with MPS (snapevo.mps_train).

    python cluster/open_train.py freeze [--force] [--times T,..] [--only F:T:D,...]   # the task list
    python cluster/open_train.py list [--cores N] [--mem-gb M] [--times T,..]       # tasks, estimates, wall time
    python cluster/open_train.py refs FIELD T                          # the training references of one field and time
    python cluster/open_train.py task FIELD T D                        # one depth, in stages (below)
    python cluster/open_train.py run [--cores N] [--mem-gb M] [--times T,..] [--no-pin]   # frozen tasks, one node
    python cluster/open_train.py table                                 # the results as markdown

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, J = 1, spin units, h_x = FIELD). Times are the rows of SETTINGS:
t = 1, 2, 4, 6, 8 and 10. The two halves run as separate one-node jobs, `run --times 1,2,4,6` and
`run --times 8,10` (open_train.slurm passes $TIMES); each holds a lock per time, so they can run at once.

Tasks. One per (field, t, D) for every depth a SNAP point of the sweep uses (its eval_b*_D<D>.json files).
`freeze` writes them, with each depth's evaluation and training settings, to
$SCRATCH/snapevo-scratch/open_train/tasks.json. With --times it freezes those times only and keeps the
entries of the others. It refuses while pool.py still has points lines at those times that are not done
(--force overrides) and when a reference the evaluation needs is missing. list and table never freeze.

A task takes the production cell of depth D (runs/<model>/t<T>/cells.json, trained on the 10-site ring),
fits its edges as production does (edges.fit_edges at its default N_fit), and trains bulk and edges together
on the open chain of N = 100 sites: c = 1 - mean F^{1/N} over the training inputs (product_vectors seed 0,
independent of the evaluation's seed 2026 and the edge fit's seed 31415), L-BFGS-B with train.py's options.
Per t (SETTINGS):
    t <= 6   12 inputs, references at dt 0.1, cutoff 1e-14, chi 2048 (12 single-threaded builders),
             training cutoff 1e-12 and chi 1024, or the depth's evaluation cutoff and chi where that is
             tighter; workers single-threaded; 5000 iterations at t <= 2, 2000 at t = 4, 400 at t = 6
    t = 8    6 inputs, references at dt 0.2, cutoff 1e-12, chi 1024 (one builder with 16 threads), training
             cutoff 1e-10 and chi 1024, 2 BLAS threads per worker, 300 iterations
    t = 10   as t = 8, with 4 BLAS threads per worker and 100 iterations
dt 0.2 is accurate to <= 2.6e-11 per site at t >= 8, far below these cells' c. The held-out evaluation
keeps each depth's sweep settings at every t (below). Discarded-weight rule (t >= 8): the expected
discarded weight per input, about 34 x cutoff x D (measured at t = 10), must stay below 1e-2 of the
depth's 1 - F at N = 100, estimated at freeze as 1 - (1 - C)^N (C its ring C_train) times production's
edge-fit excess at this depth. A depth over the limit trains at 1e-12 and chi 1024 at t = 8 and is flagged
and skipped at t = 10, where 1e-12 costs at least 4x and up to weeks per task. The task records the same
ratio from its measured "before" 1 - F and from the discarded weight it trained with.

Evaluation, per depth: the tightest (cutoff, chi_max) among the "mps" blocks of that depth's eval_*.json files
(1e-12 and 1024 for most depths; 1e-14 with 1024 at t <= 4 or 2048 at t = 6 for the tightest points), on the
sweep's 8 inputs (seed 2026) against its cached dt 0.2 references in runs/<model>/t<T>/ref/, so "before"
compares directly with the sweep. Before and after training, input 0 is also run at cutoff/100 against the
cached cutoff/100 reference, as evaluate.mps_infidelity does. At t <= 6 the evaluation is repeated against
references of the same inputs at dt 0.1, cutoff 1e-14, chi 2048 (SETTINGS "fine", built in $SCRATCH): the
dt 0.2 references carry 2e-11 to 3e-11 per site of Trotter error at t = 4 and 6 at h_x = 1.6, about 2% of
c for the deepest points there (review of 14 Oct).

Stages, each saved under $SCRATCH/snapevo-scratch/open_train/ckpt/<model>/t<T>/D<D>/ as soon as it is done
and skipped when the task starts again:
    task.lock        held (flock) by the process running the task; a second runner fails at once (exit 73)
    task.json        the settings and cell the stages belong to; the task refuses to resume on a mismatch
    starts.jsonl     one line per start (the requeue count)
    fit.json         the edge fit
    before.json      held-out c of the ring cell with the fitted edges (8 inputs, the cutoff/100 check and,
                     at t <= 6, the dt 0.1 references), and its ring C_train, C_val and circuits.held_out at
                     10 and 12 sites
    before_own.json  when the fit is not worth it (edges.worth_it): the same evaluation with the cell's own
                     edges (init_from_ring(cell, x, None)), which is what the sweep then uses; the record and
                     the table give the change against it too
    train.json       train_open's checkpoint, every few minutes of training
    milestones.json  held-out c at 50, 100, 200, 400, 800, ... iterations, below maxiter, evaluated from
                     train_open's on-iteration hook: L-BFGS-B runs once to maxiter and does not restart there
    trained.json     the trained z and the c on the training inputs, written before the after-evaluation
Training stops at maxiter, when L-BFGS-B stops on its own, or when the job's end (SLURM_JOB_END_TIME)
leaves only the time the after-evaluation needs (1.5x what the before-evaluation took). Then the task evaluates
the last checkpoint, writes the result marked provisional and exits 75; the next run continues the training.
A resume restarts L-BFGS-B without its curvature pairs; train_open does not take a stop by its f-reduction
test in the first 2 iterations after a (re)start (mps_train.RESTART_GRACE), which near floors of 1e-10 would
otherwise end training at the first line search.

Results: runs/<model>/t<T>/open/D<D>.json (before, after, milestones, training record, diagnostics, edge
statistics, discarded-weight rule), the trained bulks in runs/<model>/t<T>/open/cells.json (train.py's header
and params layout, ring C_train and C_val computed as train.py does, plus circuits.held_out at 10 and 12
sites) and the trained edges in runs/<model>/t<T>/open/edges.json.

`run` schedules the frozen tasks of its times on one node much as pool.py does: one loop over a queue ordered
by estimated remaining time, longest first (then t and D descending); each task pinned with taskset to
max(n x threads, 8) + 1 cores of its own (n training inputs, threads per worker), started only when those cores
and its estimated memory fit the budgets and its minimum wall time fits before the job's end. It holds
$SCRATCH/run_t<T>.lock for each of its times and fails at once if another run holds one. A failed task is
retried once (not one whose stage directory is locked by another process). Exit 1 if a task failed, 2 if one
is unfinished, else 0; skipped depths are listed and do not count.

Environment for tests: OT_SCRATCH (stages, references, task list), OT_OUT (results instead of
runs/<model>/t<T>/open/), OT_LOGS (task logs), OT_MAXITER (iterations), OT_MAX_PROCS (worker processes per
MPS cost, for login nodes; the results do not depend on it), OT_MAX_THREADS (BLAS threads per worker and per
reference builder, for login nodes), OT_TRACE_EVERY (iterations between checkpoints), OT_DEADLINE (epoch
seconds, in place of the job's end).
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import glob
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(os.environ.get("OT_SCRATCH")
               or Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch" / "open_train")
RUNS = Path(os.environ.get("SNAPEVO_RUNS", REPO / "runs"))
OUT_ROOT = Path(os.environ["OT_OUT"]) if os.environ.get("OT_OUT") else None
LOGS = Path(os.environ.get("OT_LOGS") or REPO / "cluster" / "logs" / "open_train")
MAXITER_OVERRIDE = int(os.environ["OT_MAXITER"]) if os.environ.get("OT_MAXITER") else None
MAX_PROCS = int(os.environ["OT_MAX_PROCS"]) if os.environ.get("OT_MAX_PROCS") else None
MAX_THREADS = int(os.environ["OT_MAX_THREADS"]) if os.environ.get("OT_MAX_THREADS") else None
TRACE_OVERRIDE = int(os.environ["OT_TRACE_EVERY"]) if os.environ.get("OT_TRACE_EVERY") else None

FIELDS = {"1.6": "A", "0": "B"}              # snapevo's model keys: A is h_x = 1.6, B is h_x = 0
N = 100
TRAIN_SEED = 0
EVAL_SET = (2026, 8)                         # evaluate.MPS_SET: the sweep's evaluation inputs
EVAL_DT = 0.2
MILESTONES = (50, 100, 200, 400, 800, 1600, 3200)
EXIT_PROVISIONAL = 75
EXIT_LOCKED = 73
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
XLA_FLAGS = "--xla_cpu_multi_thread_eigen=false"
DISCARD_COEF = 34.0          # discarded weight per input and sublayer / cutoff, measured at t = 10 (25 at t = 4)
DISCARD_LIMIT = 1e-2         # of 1 - F (13 Oct: 1e-3 skipped t = 10, D = 53; measured 1e-10 vs 1e-12 there: 4e-4 relative in c)

# Per t: training inputs; training references (dt, cutoff, chi_max) and their builders (processes, None for
# one per input, and BLAS threads each); the training cutoff and chi_max ("follow_eval": a depth evaluated at a
# tighter cutoff trains at that one); BLAS threads per training worker; the iteration budget; "fine": the
# evaluation is repeated against references of the same 8 inputs at this (dt, cutoff, chi_max), built in
# $SCRATCH; "discard": the discarded-weight rule's fallback (cutoff, chi_max), or "skip" (module docstring).
# t <= 6: from the estimates below (`list` prints them, `_simulate` the schedule), t = 6 at 600 iterations
# would take about 47 h for its deepest likely depth (D = 105 at cutoff 1e-14), so it trains 400 (a milestone
# shared by every t). t >= 8 from the measurement of 14 Oct (snapevo-scratch/t10_measure/): 1e-10 changes c by
# at most 4.4e-4 relative against 1e-12 (the gradient by 2.6e-4 in norm at t = 8), and one value_and_grad at
# 1e-12 costs 4x (t = 8) to 10-15x (t = 10) one at 1e-10; 4 threads speed a t = 10 step up 2.64x, and memory,
# not cores, limits t = 10 tasks per node.
_LOW = {"n_train": 12, "ref": (0.1, 1e-14, 2048), "ref_build": (None, 1), "cutoff": 1e-12, "chi_max": 1024,
        "follow_eval": True, "threads": 1, "fine": (0.1, 1e-14, 2048), "discard": None}
_HIGH = {"n_train": 6, "ref": (0.2, 1e-12, 1024), "ref_build": (1, 16), "cutoff": 1e-10, "chi_max": 1024,
         "follow_eval": False, "fine": None}
SETTINGS = {
    1: {**_LOW, "maxiter": 5000},
    2: {**_LOW, "maxiter": 5000},
    4: {**_LOW, "maxiter": 2000},
    6: {**_LOW, "maxiter": 400},
    8: {**_HIGH, "threads": 2, "maxiter": 300, "discard": (1e-12, 1024)},
    10: {**_HIGH, "threads": 4, "maxiter": 100, "discard": "skip"},
}
TIMES = tuple(sorted(SETTINGS))

# Cost model at t <= 6, measured on login01 (AMD EPYC 7713, loaded), one thread, N = 100, one input,
# production cells, 14 Oct 2026 (snapevo-scratch/ot_test2/measure/): per sublayer, seconds of the forward pass,
# seconds of value_and_grad, MB of the forward states value_and_grad keeps; and MB of the final MPS.
#   t = 1: D = 13 at 1e-12 (0.10 + 0.15 s), D = 29 at 1e-14 (0.26 + 0.39 s)
#   t = 2: D = 29 at 1e-12 (0.31 + 0.59 s), D = 57 at 1e-14 (0.85 + 1.44 s)
#   t = 4: D = 53 at 1e-12 (1.47 + 5.50 s), D = 81 at 1e-14 (5.40 + 14.95 s; stored 0.28 GB)
#   t = 6: D = 53 at 1e-12, h_x = 0 (9.6 + 48.7 s; stored 0.35 GB; D = 81: stored 0.59 GB);
#          D = 105 at 1e-14, h_x = 1.6 (45.6 + 173.1 s; stored 1.69 GB, final 49 MB; peak RSS 2.8 GB with JAX);
#          D = 33 at 1e-14: forward 11.0 s (37.5 s in a run beside a 3-thread job: the login node is noisy)
LAYER_COST = {   # (t, cutoff): (forward s, value_and_grad s, stored MB, final MB) per sublayer
    (1, 1e-12): (0.0078, 0.0194, 0.10, 0.2),
    (1, 1e-14): (0.0089, 0.0223, 0.15, 0.3),
    (2, 1e-12): (0.0107, 0.0311, 0.25, 0.6),
    (2, 1e-14): (0.0149, 0.0401, 0.40, 1.0),
    (4, 1e-12): (0.0277, 0.1315, 2.0, 4.0),
    (4, 1e-14): (0.0667, 0.2513, 3.4, 8.1),
    (6, 1e-12): (0.181, 1.10, 7.3, 34.0),
    (6, 1e-14): (0.434, 2.09, 16.4, 68.0),
}
# Cost model at t >= 8, from snapevo-scratch/t10_measure/ (14 Oct): t = 10 on a compute node (h_x = 0, D = 53
# at 4 threads: value_and_grad 441 s, value 111.5 s at 1e-10 and 373 s at 1e-12, peak RSS 10.39 GiB of which
# 4.40 GiB stored states; h_x = 1.6, D = 29: 500 s, 51.4 s, 4.94 GiB of which 1.78 GiB), t = 8 on login01
# divided by 1.39 (the login/compute ratio measured at t = 10). Estimates built from them: 1.2 evaluations per
# iteration, x 1.15 for the slowest of 6 or 8 inputs, one thread 2.6x slower than 4 (measured at t = 10),
# 2 threads 1.33x slower than 4 and one thread 1.5x slower than 4 at t = 8 (guesses: the login node's load
# hid the thread speedup there).
DEEP_TRAIN = {   # (t, h_x, training cutoff): per sublayer: s per iteration, s of the training inputs' forward
                 # pass (both at the SETTINGS threads); peak GiB of one worker: fixed part, part per sublayer
    (8, "0", 1e-10): (2.27, 0.53, 1.11, 0.0171),
    (8, "1.6", 1e-10): (2.95, 0.51, 0.77, 0.0136),
    (8, "0", 1e-12): (12.5, 1.41, 1.62, 0.0349),
    (8, "1.6", 1e-12): (14.1, 1.24, 1.07, 0.0285),
    (10, "0", 1e-10): (11.49, 2.42, 5.99, 0.083),
    (10, "1.6", 1e-10): (23.8, 2.04, 3.16, 0.0614),
}
DEEP_EVAL = {    # (t, h_x): s per sublayer of a held-out forward pass at 1e-12 (one thread, slowest of 8); GiB of
                 # a (0.2, 1e-12, 1024) reference, of the final MPS at 1e-12, of the k = 0 reference at 1e-14
                 # (chi 1024) and of the final MPS at 1e-14
    (8, "0"): (1.6, 0.165, 0.23, 0.333, 0.51),
    (8, "1.6"): (1.4, 0.115, 0.20, 0.249, 0.40),
    (10, "0"): (21.1, 0.983, 1.49, 2.06, 2.2),
    (10, "1.6"): (12.6, 0.571, 1.16, 1.29, 2.1),
}
EVALS_PER_ITER = 1.2           # L-BFGS-B evaluations per iteration (1.15 to 1.2 measured)
CHECK_FACTOR = 2.5             # forward at cutoff/100 against at cutoff (chi up by about 1.35)
# Edge fit (edges.fit_edges, JAX, one thread): the slowest production fit per sublayer at each t (real_*.json,
# t = 6: 3320 s at D = 53, 2258 s at D = 33; t = 8: 3458 s at D = 45; t = 10: 3264 s at D = 37, 4901 s at
# D = 65), and its memory: 14.7 GB peak at t = 6, D = 105, N_fit = 18 (measured: 61 s per evaluation there),
# scaling with D 2^N_fit. Without XLA_FLAGS, on 3 pinned cores, one evaluation took 25 s and 5.1 GB; the flag
# is kept, as for pool.py, so the fit stays on one core.
FIT_S_PER_LAYER = {1: 2.0, 2: 2.0, 4: 45.0, 6: 70.0, 8: 80.0, 10: 90.0}
FIT_GB_PER_LAYER_18 = 0.14
# Training references, built once per field and time (estimates): t <= 6 the 12 on 12 single-threaded workers;
# t = 8 and 10 the 6 one after another at 16 threads (the sweep's own builds: 176-178 s each at 8 threads at
# t = 8, 1268-1307 s each at 16 threads at t = 10). FINE_REFS_S: the 8 "fine" references on 8 workers (one
# took 13-19 s at t = 4 and 110-147 s at t = 6, single-threaded).
REFS_S = {1: 30, 2: 60, 4: 200, 6: 600, 8: 800, 10: 7800}
FINE_REFS_S = {1: 15, 2: 30, 4: 40, 6: 200}


def _log(msg: str) -> None:
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def _model(field: str):
    from snapevo.hamiltonian import get_model
    return get_model(FIELDS[field])


def _tdir(model, t) -> Path:
    return RUNS / model.tag / f"t{float(t):g}"


def _out_dir(model, t) -> Path:
    return OUT_ROOT / model.tag / f"t{float(t):g}" if OUT_ROOT else _tdir(model, t) / "open"


def _out(model, t, D: int) -> Path:
    return _out_dir(model, t) / f"D{D}.json"


def _stage_dir(model, t, D: int) -> Path:
    return SCRATCH / "ckpt" / model.tag / f"t{float(t):g}" / f"D{D}"


def _refs_dir(model, t) -> Path:
    return SCRATCH / "refs" / model.tag / f"t{float(t):g}"


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serializable")


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, default=_default))
    os.replace(tmp, path)


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=_default))


@contextlib.contextmanager
def _flock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


class _Locked(Exception):
    """Another process holds a lock _hold wanted."""


_HELD: list = []                                 # lock files this process holds until it exits


def _hold(path: Path, what: str) -> None:
    """Take a non-blocking exclusive flock on path for the rest of this process, or raise _Locked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip() or "unknown"
        fh.close()
        raise _Locked(f"{what}: {path} is held by another process ({holder})") from None
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", ""),
                         "since": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    fh.flush()
    _HELD.append(fh)


@contextlib.contextmanager
def _single_threaded():
    """Processes started inside get one BLAS/OpenMP thread (as mps_train's workers do)."""
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    os.environ.update({k: "1" for k in THREAD_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _procs(n: int) -> int:
    return max(1, min(n, MAX_PROCS)) if MAX_PROCS else n


def _threads(n: int) -> int:
    return max(1, min(n, MAX_THREADS)) if MAX_THREADS else n


def _deadline() -> float:
    """OT_DEADLINE, else the end of this SLURM job minus 150 s (as pool.py), else infinity."""
    if os.environ.get("OT_DEADLINE"):
        return float(os.environ["OT_DEADLINE"])
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return math.inf
    end = os.environ.get("SLURM_JOB_END_TIME")
    if end and end.isdigit():
        return float(end) - 150
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True,
                             timeout=60).stdout.strip()
        return time.mktime(time.strptime(out, "%Y-%m-%dT%H:%M:%S")) - 150
    except Exception:                                              # noqa: BLE001
        return time.time() + 1800 - 150


def _when(ts: float) -> str:
    return "none" if ts == math.inf else time.strftime("%m-%d %H:%M:%S", time.localtime(ts))


# --------------------------------------------------------------------------- task list and settings

def _key(e: dict) -> tuple[str, int, int]:
    return e["field"], int(e["t"]), int(e["D"])


def _name(e: dict) -> str:
    return f"h_x = {e['field']}, t = {e['t']}, D = {e['D']}"


def _parse_times(text: str) -> tuple[int, ...]:
    """'1,2,4,6' (or separated by ':', '+' or spaces) -> (1, 2, 4, 6); every time must be a SETTINGS row."""
    times = tuple(sorted({int(float(x)) for x in re.split(r"[,:+\s]+", text.strip()) if x}))
    bad = [t for t in times if t not in SETTINGS]
    if bad or not times:
        raise SystemExit(f"--times {text!r}: each time must be one of {TIMES}")
    return times


def _eval_blocks(model, t) -> dict[int, list[dict]]:
    """The "mps" block of every eval_*.json at this model and time, by depth."""
    out: dict[int, list[dict]] = {}
    for p in sorted(_tdir(model, t).glob("eval_b*_D*.json")):
        m = re.search(r"_D(\d+)\.json$", p.name)
        if not m:
            continue
        b = json.loads(p.read_text()).get("mps") or {}
        out.setdefault(int(m.group(1)), []).append(
            {"cutoff": float(b["cutoff"]), "chi_max": int(b["chi_max"]), "dt": float(b["dt"]),
             "states": [int(v) for v in b["states"]], "file": p.name})
    return out


def _eval_settings(blocks: list[dict]) -> dict:
    """The tightest (cutoff, chi_max) of a depth's evaluations on the sweep's inputs and step."""
    ok = [b for b in blocks if b["dt"] == EVAL_DT and b["states"] == list(EVAL_SET)]
    if not ok:
        raise ValueError(f"no evaluation at dt {EVAL_DT} on inputs {EVAL_SET}: {blocks}")
    b = min(ok, key=lambda b: (b["cutoff"], -b["chi_max"]))
    return {"cutoff": b["cutoff"], "chi_max": b["chi_max"], "dt": EVAL_DT, "states": list(EVAL_SET),
            "files": sorted(x["file"] for x in ok)}


def _eval_paths(model, t, ev: dict) -> tuple[list[Path], Path]:
    """The sweep's cached references for the evaluation (8 inputs) and for its cutoff/100 check (input 0)."""
    from snapevo.evaluate import ref_path
    d = _tdir(model, t) / "ref"
    seed, n = ev["states"]
    main = [ref_path(d, N, seed, k, ev["dt"], ev["cutoff"], ev["chi_max"]) for k in range(n)]
    return main, ref_path(d, N, seed, 0, ev["dt"], ev["cutoff"] / 100, ev["chi_max"])


@functools.lru_cache(maxsize=None)
def _production_fit(field: str, t: int, D: int) -> dict | None:
    """Production's edge fit at this depth: the "edges" record of the smallest-budget real_b*_D<D>.json."""
    files = glob.glob(str(_tdir(_model(field), t) / f"real_b*_D{D}.json"))
    budget = lambda p: float(re.search(r"real_b([0-9.eE+-]+)_D", p).group(1))     # noqa: E731
    for p in sorted(files, key=budget):
        fit = json.loads(Path(p).read_text()).get("edges")
        if fit and all(k in fit for k in ("infid_start", "infid_fit", "infid_ring", "N_fit")):
            return {k: fit[k] for k in ("infid_start", "infid_fit", "infid_ring", "N_fit")} | {"file": Path(p).name}
    return None


@functools.lru_cache(maxsize=None)
def _ring_floor(field: str, t: int, D: int) -> float:
    from snapevo.train import cell_records
    rec = next(r for r in cell_records(_tdir(_model(field), t) / "cells.json") if int(r["D"]) == D)
    return float(rec["C_train"])


def _one_minus_F(field: str, t: int, D: int) -> tuple[float, str]:
    """1 - F at N = 100 of the ring cell with fitted edges, estimated: the bulk 1 - (1 - C)^N from the ring floor
    C, times what production's edge fit at this depth leaves above its ring (at N_fit, on the realized circuit)."""
    C = _ring_floor(field, t, D)
    fit = _production_fit(field, t, D)
    edge = max(0.0, fit["infid_fit"] - fit["infid_ring"]) if fit else 0.0
    est = 1.0 - (1.0 - min(C, 1.0)) ** N * (1.0 - min(edge, 1.0))
    how = f"1 - (1 - C)^{N} (C = {C:.3e})" + (f" x edge excess {edge:.2e} ({fit['file']})" if fit else "")
    return est, how


def _base_train(t: int, ev: dict) -> tuple[float, int]:
    """Training cutoff and chi_max before the discarded-weight rule."""
    s = SETTINGS[int(t)]
    if s["follow_eval"] and ev["cutoff"] < s["cutoff"]:
        return ev["cutoff"], ev["chi_max"]
    return s["cutoff"], s["chi_max"]


def _rule(e: dict) -> dict:
    """The discarded-weight rule at this depth (decision "keep", "fallback", "skip" or "not applied")."""
    t, D = int(e["t"]), int(e["D"])
    s = SETTINGS[t]
    cutoff, chi = _base_train(t, e["eval"])
    one_mF, how = _one_minus_F(e["field"], t, D)
    expected = DISCARD_COEF * cutoff * D
    out = {"coef": DISCARD_COEF, "limit": DISCARD_LIMIT, "cutoff": cutoff, "chi_max": chi,
           "expected_discarded": expected, "one_minus_F": one_mF, "estimate": how, "ratio": expected / one_mF}
    if s["discard"] is None:
        out["decision"] = "not applied"
    elif out["ratio"] <= DISCARD_LIMIT:
        out["decision"] = "keep"
    elif s["discard"] == "skip" or DISCARD_COEF * s["discard"][0] * D / one_mF > DISCARD_LIMIT:
        out["decision"] = "skip"
    else:
        out.update(decision="fallback", cutoff=s["discard"][0], chi_max=s["discard"][1],
                   ratio_fallback=DISCARD_COEF * s["discard"][0] * D / one_mF)
    return out


def _skipped(e: dict) -> bool:
    return (e.get("rule") or {}).get("decision") == "skip"


def _ref_settings(t: int) -> dict:
    s = SETTINGS[int(t)]
    return {"n": s["n_train"], "seed": TRAIN_SEED, "ref_dt": s["ref"][0], "ref_cutoff": s["ref"][1],
            "ref_chi": s["ref"][2]}


def _train_settings(e: dict) -> dict:
    """What the depth trains with: inputs, references, cutoff and chi_max (after the rule), worker threads."""
    t = int(e["t"])
    rule = e.get("rule") or _rule(e)
    cutoff, chi = (rule["cutoff"], rule["chi_max"]) if rule["decision"] == "fallback" else _base_train(t, e["eval"])
    return {**_ref_settings(t), "cutoff": cutoff, "chi_max": chi, "threads": SETTINGS[t]["threads"]}


def _maxiter(t: int) -> int:
    return MAXITER_OVERRIDE or SETTINGS[int(t)]["maxiter"]


def _stages(maxiter: int) -> list[int]:
    return [m for m in MILESTONES if m < maxiter] + [maxiter]


def _problem(e: dict) -> dict:
    """What the stage files depend on: a task refuses to resume when any of it changed (not the threads)."""
    from snapevo.mps_train import TRAIN_OPTIONS
    t = int(e["t"])
    tr = {k: v for k, v in _train_settings(e).items() if k != "threads"}
    return _jsonable({"N": N, "eval": {k: e["eval"][k] for k in ("cutoff", "chi_max", "dt", "states")},
                      "fine": SETTINGS[t].get("fine"), "train": tr, "milestones": list(MILESTONES),
                      "options": TRAIN_OPTIONS})


def _frozen() -> dict | None:
    return _read_json(SCRATCH / "tasks.json")


def _make_entry(field: str, t: int, D: int, blocks: list[dict]) -> dict:
    e = {"field": field, "t": int(t), "D": int(D), "eval": _eval_settings(blocks)}
    e["rule"] = _rule(e)
    fit = _production_fit(field, int(t), int(D))
    if fit:
        from snapevo.edges import worth_it
        e["prod_worth_it"] = bool(worth_it(fit))
    return e


def _scan(only=None, times=TIMES) -> tuple[list[dict], list[str]]:
    """Every (field, t, D) a SNAP point uses, with its settings, and the problems found."""
    from snapevo.train import cell_records
    entries, problems = [], []
    for t in times:
        for f in FIELDS:
            model = _model(f)
            blocks = _eval_blocks(model, t)
            ladder = {int(r["D"]) for r in cell_records(_tdir(model, t) / "cells.json")} if blocks else set()
            for D in sorted(blocks):
                if only and (f, t, D) not in only:
                    continue
                name = f"h_x = {f}, t = {t}, D = {D}"
                if D not in ladder:
                    problems.append(f"{name}: not in cells.json")
                    continue
                try:
                    e = _make_entry(f, t, D, blocks[D])
                    _estimate(e)
                except (ValueError, KeyError) as err:
                    problems.append(f"{name}: {err!r}")
                    continue
                main, chk = _eval_paths(model, t, e["eval"])
                missing = [p.name for p in main + [chk] if not p.exists()]
                if missing:
                    problems.append(f"{name}: references missing in {main[0].parent}: {missing}")
                entries.append(e)
    return entries, problems


def _entry(field: str, t: int, D: int) -> dict:
    """The frozen entry of a task, or (no list yet, for tests) the one the eval files give now."""
    fr = _frozen()
    for e in (fr or {}).get("tasks", []):
        if _key(e) == (field, int(t), int(D)):
            return e
    if fr:
        raise SystemExit(f"h_x = {field}, t = {t}, D = {D} is not in the frozen list {SCRATCH / 'tasks.json'}")
    blocks = _eval_blocks(_model(field), t).get(int(D))
    if not blocks:
        raise SystemExit(f"no eval_*_D{D}.json at h_x = {field}, t = {t}")
    return _make_entry(field, t, D, blocks)


def _pool_pending(times) -> list[str]:
    """points lines of pool.py at these times that are not done (read only)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pool
        lines = pool.load_tasks()
        st = pool.classify(lines)
    except Exception as err:                                       # noqa: BLE001
        return [f"(pool state unreadable: {err})"]
    return [f"{k.id} ({st[k.id]}): {k.cmd}" for k in lines
            if k.list == "points" and k.t in times and st[k.id] != "done"]


# --------------------------------------------------------------------------- estimates

def _layer_cost(t: int, cutoff: float) -> tuple[float, float, float, float]:
    return LAYER_COST[(int(t), 1e-14 if cutoff < 1e-13 else 1e-12)]


def _estimate(e: dict) -> dict:
    """Seconds and GB of one task from the cost models above (estimates, not measurements)."""
    from snapevo.edges import fit_size
    t, D, f = int(e["t"]), int(e["D"]), e["field"]
    s = SETTINGS[t]
    ev, tr = e["eval"], _train_settings(e)
    maxiter, n, th = _maxiter(t), tr["n"], tr["threads"]
    nfit = fit_size(t)
    fit_gb = FIT_GB_PER_LAYER_18 * D * 2.0 ** (nfit - 18)
    if t <= 6:
        fwd_e, _, _, fin_e = _layer_cost(t, ev["cutoff"])
        fwd_t, vg_t, mb_t, fin_t = _layer_cost(t, tr["cutoff"])
        held_s = fwd_e * D + 5
        check_s = CHECK_FACTOR * fwd_e * D + 10
        vg_s = vg_t * D
        iter_s = EVALS_PER_ITER * vg_s + 0.05
        train_value_s = fwd_t * D + 30
        fin_fine = _layer_cost(t, 1e-14)[3]
        gb = 1.2 * (1.0 + fit_gb + (n + 8) * max(fin_t, fin_e) / 1024
                    + n * (0.12 + 1.2 * mb_t * D / 1024 + 3 * fin_t / 1024) + 8 * (0.12 + 2 * fin_e / 1024) + 0.5
                    + (8 * (0.12 + 2 * fin_fine / 1024) if s["fine"] else 0.0))
    else:
        if not math.isclose(ev["cutoff"], 1e-12):
            raise KeyError(f"no cost model at t = {t} for the evaluation cutoff {ev['cutoff']:g}")
        it_l, fwdt_l, w0, w1 = DEEP_TRAIN[(t, f, tr["cutoff"])]
        fwde_l, tgt, fin_e, tgt_chk, fin_chk = DEEP_EVAL[(t, f)]
        held_s = fwde_l * D + 5
        check_s = CHECK_FACTOR * fwde_l * D + 10
        iter_s = it_l * D
        vg_s = iter_s / EVALS_PER_ITER
        train_value_s = fwdt_l * D + 30
        persistent = 1.0 + fit_gb + n * tgt + 8 * tgt + 8 * (0.12 + tgt)      # parent, idle evaluation workers
        training = n * 1.15 * (w0 + w1 * D)                                     # measured worker peaks (+15%)
        evaluating = 8 * fin_e + (0.12 + tgt_chk + fin_chk) + n * (0.12 + tgt)  # an evaluation; idle trainers
        gb = 1.1 * persistent + max(training, 1.1 * evaluating)
    full_s = max(held_s, check_s) + (held_s if s["fine"] else 0.0) + 30 + 0.5 * D   # check beside, fine, ring
    own_s = full_s if e.get("prod_worth_it") is False else 0.0
    stages = _stages(maxiter)
    train_s = maxiter * iter_s + (len(stages) - 1) * held_s + vg_s     # milestones (no restarts), the start
    fit_s = 60 + FIT_S_PER_LAYER[t] * D
    fine_refs_s = FINE_REFS_S.get(t, 0) if s["fine"] else 0
    after_s = full_s + train_value_s
    total = fit_s + full_s + own_s + fine_refs_s + REFS_S[t] + train_s + after_s
    return {"cores": max(n * th, 8) + 1, "gb": round(gb, 1), "fit_s": fit_s, "full_s": full_s, "own_s": own_s,
            "held_s": held_s, "iter_s": iter_s, "train_s": train_s, "refs_s": REFS_S[t] + fine_refs_s,
            "after_s": after_s, "total_s": total, "maxiter": maxiter, "threads": th}


def _trace_every(est: dict) -> int:
    """Iterations between checkpoints: about two minutes of training, 1 to 50."""
    return TRACE_OVERRIDE or int(min(50, max(1, round(120 / est["iter_s"]))))


_REFS_SEEN: set = set()                          # (field, t) whose training references exist (they stay)


def _refs_present(e: dict) -> bool:
    from snapevo.evaluate import ref_path
    if (e["field"], int(e["t"])) in _REFS_SEEN:
        return True
    tr = _ref_settings(e["t"])
    d = _refs_dir(_model(e["field"]), e["t"])
    ok = all(ref_path(d, N, tr["seed"], k, tr["ref_dt"], tr["ref_cutoff"], tr["ref_chi"]).exists()
             for k in range(tr["n"]))
    if ok:
        _REFS_SEEN.add((e["field"], int(e["t"])))
    return ok


def _min_wall(e: dict, est: dict) -> float:
    """Seconds a task needs to make progress worth starting it for, given its saved stages."""
    sd = _stage_dir(_model(e["field"]), e["t"], e["D"])
    need = 0.0 if (sd / "fit.json").exists() else est["fit_s"]
    need += 0.0 if (sd / "before.json").exists() else est["full_s"]
    need += 0.0 if (sd / "before_own.json").exists() else est["own_s"]
    if not (sd / "trained.json").exists():
        need += (0.0 if _refs_present(e) else est["refs_s"]) + min(est["train_s"], max(1800.0, 50 * est["iter_s"]))
    return 1.2 * (need + est["after_s"]) + 300


def _remaining_s(e: dict, est: dict) -> float:
    """Estimated seconds a task still needs, from its saved stages (all of est["total_s"] for a new one)."""
    model = _model(e["field"])
    r = _read_json(_out(model, e["t"], e["D"]))
    if r and not r.get("provisional"):
        return 0.0
    sd = _stage_dir(model, e["t"], e["D"])
    left = est["total_s"]
    left -= est["fit_s"] if (sd / "fit.json").exists() else 0.0
    left -= est["full_s"] if (sd / "before.json").exists() else 0.0
    left -= est["own_s"] if (sd / "before_own.json").exists() else 0.0
    ck = None if (sd / "trained.json").exists() else _read_json(sd / "train.json")
    if (sd / "trained.json").exists():
        left -= est["refs_s"] + est["train_s"]
    elif ck:
        left -= est["refs_s"] + min(int(ck["nit"]), est["maxiter"]) * est["iter_s"]
    return max(left, est["after_s"])


def _priority(e: dict, est: dict, fresh: bool = False) -> tuple:
    """Longest estimated remaining time first (fresh: the whole estimate), then t and D descending. Measured on
    today's lists: the same 26.3 h as t-then-D order at t <= 6, and 70 h instead of 97 h at t >= 8, where
    the longest tasks (t = 8 at the 1e-12 fallback) would otherwise start after every t = 10 task."""
    left = est["total_s"] if fresh else _remaining_s(e, est)
    return (-round(left), -int(e["t"]), -int(e["D"]), e["field"])


def _simulate(entries: list[dict], cores: int, mem_gb: float) -> dict:
    """List-schedule the tasks as `run` does (priority order, core and memory budgets), from the estimates."""
    entries = [e for e in entries if not _skipped(e)]
    est = {_key(e): _estimate(e) for e in entries}
    pending = sorted(entries, key=lambda e: _priority(e, est[_key(e)], fresh=True))
    running: list[tuple[float, int, float]] = []
    free_c, free_m, now, last_start = cores, mem_gb, 0.0, 0.0
    while pending or running:
        for e in list(pending):
            s = est[_key(e)]
            c, m = min(s["cores"], cores), min(s["gb"], mem_gb)
            if c <= free_c and m <= free_m:
                pending.remove(e)
                running.append((now + s["total_s"], c, m))
                free_c, free_m, last_start = free_c - c, free_m - m, now
        if not running:
            break
        running.sort()
        end, c, m = running.pop(0)
        now, free_c, free_m = end, free_c + c, free_m + m
    core_h = sum(min(s["cores"], cores) * s["total_s"] for s in est.values()) / 3600
    return {"makespan_h": now / 3600, "last_start_h": last_start / 3600, "core_h": core_h,
            "longest_h": max((s["total_s"] for s in est.values()), default=0.0) / 3600}


# --------------------------------------------------------------------------- pieces of a task

def _load_ref(path: Path, model, t, dt: float, cutoff: float | None = None):
    """A cached reference MPS, after checking it is for this model, time, size, step and truncation."""
    from snapevo.mps import MPS
    psi, meta = MPS.load(path)
    got = {"model": meta.get("model"), "t": float(meta.get("t")), "N": meta.get("N"), "dt": float(meta.get("dt"))}
    want = {"model": model.to_dict(), "t": float(t), "N": N, "dt": float(dt)}
    if got != want or (cutoff is not None and not math.isclose(float(meta.get("cutoff")), cutoff)):
        raise ValueError(f"{path} is for {got} at cutoff {meta.get('cutoff')}, not {want} at {cutoff}")
    return psi


def _c_of(F) -> list[float]:
    """Per-input c_k = 1 - F_k^{1/N}, as OpenChainCost averages it."""
    return [float(-math.expm1(math.log(max(float(f), 1e-300)) / N)) for f in F]


def _paired(cb, ca) -> dict:
    """Relative paired change mean(ca - cb) / mean(cb), its standard error, and how many inputs improved."""
    import numpy as np
    cb, ca = np.asarray(cb, dtype=float), np.asarray(ca, dtype=float)
    d = ca - cb
    return {"rel": float(d.mean() / cb.mean()),
            "se": float(d.std(ddof=1) / math.sqrt(d.size) / cb.mean()) if d.size > 1 else 0.0,
            "improved": int((d < 0).sum()), "n": int(d.size)}


def _check_k0(model_dict: dict, cell_name: str, t: float, D: int, chi_max: int, cutoff: float, path: str,
              z, vec) -> dict:
    """Input 0 at the check cutoff against its cached reference (runs in its own process)."""
    import numpy as np
    from snapevo.cells import get_cell
    from snapevo.hamiltonian import XXZ
    from snapevo.mps_train import OpenChainCost
    model = XXZ.from_dict(model_dict)
    target = _load_ref(Path(path), model, t, EVAL_DT, cutoff)
    t0 = time.time()
    with OpenChainCost(model, get_cell(cell_name), t, D, N, np.asarray(vec)[None], [target], chi_max=chi_max,
                       cutoff=cutoff, processes=1) as c:
        c.value(np.asarray(z, dtype=float))
        F = float(c.last["F"][0])
        return {"cutoff": cutoff, "chi_max": chi_max, "F": F, "c": _c_of([F])[0], "chi_peak": int(c.last["chi_u"][0]),
                "discarded": float(c.last["discarded_u"][0]), "seconds": time.time() - t0}


class EvalSet:
    """The sweep's evaluation inputs at one depth's settings, with their workers kept for the whole task.

    With fine = (dt, cutoff, chi_max), the same inputs are also evaluated against references at those
    settings (built in $SCRATCH/refs under the build lock), in full() only.
    """

    def __init__(self, model, cell, t, D: int, ev: dict, fine=None, log=_log):
        from snapevo.circuits import product_vectors
        from snapevo.mps_train import OpenChainCost, open_chain_problem
        main, self.check_path = _eval_paths(model, t, ev)
        seed, n = ev["states"]
        self.vecs = product_vectors(N, n, seed)
        targets = [_load_ref(p, model, t, ev["dt"], ev["cutoff"]) for p in main]
        self.cost = OpenChainCost(model, cell, float(t), D, N, self.vecs, targets, chi_max=ev["chi_max"],
                                  cutoff=ev["cutoff"], processes=_procs(n))
        self.check_args = (model.to_dict(), cell.name, float(t), D, ev["chi_max"], ev["cutoff"] / 100,
                           str(self.check_path))
        self.fine_cost = None
        if fine:
            folder = _refs_dir(model, t)
            with _flock(folder / "build.lock"):
                for p in folder.glob("*.tmp.npz"):                 # a build killed earlier (no one else builds now)
                    p.unlink(missing_ok=True)
                _, tf = open_chain_problem(model, float(t), N, n, seed, SCRATCH / "refs", processes=_procs(n),
                                           dt=fine[0], cutoff=fine[1], chi_max=fine[2], log=log)
            self.fine_cost = OpenChainCost(model, cell, float(t), D, N, self.vecs, tf, chi_max=ev["chi_max"],
                                           cutoff=ev["cutoff"], processes=_procs(n))

    @staticmethod
    def _value(cost, z) -> dict:
        t0 = time.time()
        c = cost.value(z)
        L = cost.last
        F = [float(v) for v in L["F"]]
        return {"c": float(c), "F": F, "c_k": _c_of(F), "chi_peak": int(L["chi_u"].max()),
                "discarded_max": float(L["discarded_u"].max()), "seconds": time.time() - t0}

    def held_out(self, z) -> dict:
        return self._value(self.cost, z)

    def full(self, z) -> dict:
        """held_out, with input 0 at cutoff/100 run beside it in one more process."""
        import concurrent.futures as cf
        import multiprocessing as mp
        t0 = time.time()
        with _single_threaded():
            ex = cf.ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))
            fut = ex.submit(_check_k0, *self.check_args, list(map(float, z)), self.vecs[0])
        try:
            ho = self.held_out(z)
            chk = fut.result()
        finally:
            ex.shutdown(wait=True, cancel_futures=True)
        F0 = ho["F"][0]
        chk["rel_change"] = abs((1.0 - chk["F"]) - (1.0 - F0)) / max(1.0 - chk["F"], 1e-300)
        out = {"held_out": ho, "check": chk, "diagnostics": self.cost.diagnostics()}
        if self.fine_cost is not None:
            out["held_out_fine"] = self._value(self.fine_cost, z)
        out["seconds"] = time.time() - t0
        return out

    def close(self) -> None:
        self.cost.close()
        if self.fine_cost is not None:
            self.fine_cost.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _ring(model, cell, t, D: int, x) -> dict:
    """Ring C of a bulk as train.py records it (C_train on TRAIN_SET with make_cost, C_val on VAL_SET at 10
    sites) and circuits.held_out at 10 and 12 sites."""
    import numpy as np
    from snapevo.circuits import TRAIN_SET, Layered, held_out, local_infidelity, state_pairs
    from snapevo.jaxcost import make_cost
    from snapevo.train import L_TRAIN, VAL_SET
    t0 = time.time()
    x = np.asarray(x, dtype=float).reshape(-1)
    tin, tg = state_pairs(model, L_TRAIN, float(t), TRAIN_SET[1], TRAIN_SET[0])
    vin, vtg = state_pairs(model, L_TRAIN, float(t), VAL_SET[1], VAL_SET[0])
    c_train = float(make_cost(cell, L_TRAIN, tin, tg, D)(x)[0])
    circ = Layered.brickwork(cell, x)
    c_val = float(local_infidelity(circ.apply(vin, L_TRAIN), vtg, L_TRAIN))
    return {"C_train": c_train, "C_val": c_val, "held_out": held_out(circ, model, float(t), sizes=(10, 12)),
            "seconds": time.time() - t0}


def _quarter(angles) -> dict:
    """Edge angles within 1e-2 of a multiple of pi/4 (the window edges.fit_edges snaps in), and exactly on one."""
    from snapevo.library import QUARTER
    off = [abs(math.remainder(float(a), QUARTER)) for a in angles]
    return {"n": len(off), "near": sum(o <= 1e-2 for o in off), "exact": sum(o <= 1e-12 for o in off)}


def _train_problem(model, t, tr: dict, log):
    """The training inputs and references; one task per (model, t) builds missing references, under a lock."""
    from snapevo.mps_train import open_chain_problem
    procs, threads = SETTINGS[int(t)]["ref_build"]
    folder = _refs_dir(model, t)
    with _flock(folder / "build.lock"):
        for p in folder.glob("*.tmp.npz"):                        # a build killed earlier (no one else builds now)
            p.unlink(missing_ok=True)
        return open_chain_problem(model, float(t), N, tr["n"], tr["seed"], SCRATCH / "refs",
                                  processes=_procs(tr["n"] if procs is None else procs), threads=_threads(threads),
                                  dt=tr["ref_dt"], cutoff=tr["ref_cutoff"], chi_max=tr["ref_chi"], log=log)


class _Stop(Exception):
    """Raised by _Deadlined to end training for the deadline."""


class _Deadlined:
    """The cost, refusing to start an evaluation that would not end before stop_at (train_open sees _Stop)."""

    def __init__(self, cost, stop_at: float):
        self._cost, self.stop_at, self.longest = cost, stop_at, 0.0

    def __getattr__(self, name):
        return getattr(self._cost, name)

    def value_and_grad(self, z):
        if time.time() + 1.2 * self.longest > self.stop_at:
            raise _Stop
        t0 = time.time()
        out = self._cost.value_and_grad(z)
        self.longest = max(self.longest, time.time() - t0)
        return out


def _staged_training(cost, z0, evs: EvalSet, sd: Path, stop_at: float, maxiter: int, held_s: float, log) -> dict:
    """train_open once to maxiter, the held-out c at each milestone below it from its on-iteration hook (no
    restart there); a dict with the trained point. held_s: seconds of one held-out evaluation so far."""
    import numpy as np
    from snapevo.mps_train import RESTART_GRACE, train_open
    ck, ms_path = sd / "train.json", sd / "milestones.json"
    ms = _read_json(ms_path) or []
    want = [m for m in _stages(maxiter) if m < maxiter]
    have = {r["iter"] for r in ms}
    held = [held_s] + [r["held_out"]["seconds"] for r in ms]
    proxy = _Deadlined(cost, stop_at)

    def milestone(m: int, x, c: float, nfev: int, seconds: float) -> None:
        ho = evs.held_out(np.asarray(x, dtype=float))
        ms.append({"iter": m, "nit": m, "nfev": int(nfev), "c_train": float(c), "train_seconds": float(seconds),
                   "held_out": ho, "diagnostics": cost.diagnostics() if cost.last else None})
        _save_json(ms_path, ms)
        have.add(m)
        held.append(ho["seconds"])
        log(f"milestone {m}: train c {c:.6e}, held-out c {ho['c']:.6e} ({ho['seconds']:.0f}s)")

    prior = _read_json(ck)
    if prior and prior["nit"] in want and prior["nit"] not in have:   # killed during that milestone's evaluation
        milestone(prior["nit"], prior["z"], prior["c"], prior["nfev"], prior["trace"][-1][2])
    for m in want:
        if m not in have and prior and prior["nit"] > m:
            log(f"milestone {m}: the checkpoint is already at iteration {prior['nit']}, not evaluated")
    clock = {"last": time.time(), "iter": 0.0}

    def on_iteration(nit: int, x, c: float, info: dict):
        now = time.time()
        clock["iter"] = max(clock["iter"], now - clock["last"])
        if nit in want and nit not in have:
            if now + 1.5 * max(held) > stop_at:
                return "deadline"
            milestone(nit, x, c, info["nfev"], info["seconds"])
        nxt = nit + 1 in want and nit + 1 not in have
        need = 1.2 * max(clock["iter"], EVALS_PER_ITER * proxy.longest) + (1.5 * max(held) if nxt else 0.0)
        clock["last"] = time.time()
        return "deadline" if clock["last"] + need > stop_at else None

    res = None
    try:
        res, _ = train_open(proxy, z0, maxiter, log=log, checkpoint=ck, on_iteration=on_iteration)
        stop = res.stop
    except _Stop:
        stop = "hook: deadline"
    diag = cost.diagnostics() if cost.last else None
    if stop == "hook: deadline":
        reason = "deadline"
    elif stop == "maxiter":
        reason = "maxiter"
    else:
        reason = f"stopped by L-BFGS-B before maxiter: {res.message}"
        if res.nit - res.restart_nit <= RESTART_GRACE:
            reason += f" (within {RESTART_GRACE} iterations of a restart at {res.restart_nit})"
    prior = _read_json(ck)
    if prior is None:                                              # stopped before the first checkpoint
        z, nit, nfev, c, trace = list(map(float, z0)), 0, 0, None, []
    else:
        z, nit, nfev, c, trace = prior["z"], int(prior["nit"]), int(prior["nfev"]), float(prior["c"]), prior["trace"]
    return {"provisional": reason == "deadline", "reason": reason, "z": z, "nit": nit, "nfev": nfev, "c_end": c,
            "trace": trace, "message": None if res is None else str(res.message),
            "status": None if res is None else int(res.status), "restart_nit": None if res is None else res.restart_nit,
            "train_seconds": float(trace[-1][2]) if trace else 0.0, "diagnostics": diag}


def _write_open_files(model, cell, t, D: int, x1, ring: dict, e_fit, e_new, rec: dict) -> None:
    """The trained bulk into open/cells.json (train.py's layout) and the trained edges into open/edges.json."""
    import numpy as np
    from snapevo.circuits import TRAIN_SET
    from snapevo.train import L_TRAIN, VAL_SET
    d = _out_dir(model, t)
    head = {"model": model.to_dict(), "cell": cell.name, "t": float(t)}
    with _flock(d / "cells.json.lock"):
        for name, extra, row in (
                ("cells.json", {"N": L_TRAIN, "train_set": list(TRAIN_SET), "val_set": list(VAL_SET)},
                 {"D": D, "params": np.asarray(x1, dtype=float).reshape(D, cell.n_angles).tolist(),
                  "C_train": ring["C_train"], "C_val": ring["C_val"], "start": "open_train", "held_out": ring["held_out"],
                  "provisional": rec["provisional"], "nit": rec["train"]["nit"], "c100_before": rec["c100_before"],
                  "c100_after": rec["c100_after"], "c100_before_own_edges": rec["c100_before_own_edges"]}),
                ("edges.json", {"N": N, "pairs": "edges.pairs order: 6 ZYZ angles (gate on site N-1, gate on site 0) "
                                                  "per parity-1 sublayer"},
                 {"D": D, "angles": list(map(float, e_new)), "angles_fit": list(map(float, e_fit)),
                  "worth_it_fit": rec["edges"]["worth_it"], "quarter_fit": rec["edges"]["quarter_fit"],
                  "quarter_trained": rec["edges"]["quarter_trained"], "provisional": rec["provisional"]})):
            p = d / name
            doc = _read_json(p) or {**head, **extra, "results": []}
            if {k: doc.get(k) for k in head} != head:
                raise ValueError(f"{p} belongs to another model, cell or time")
            doc["results"] = sorted([r for r in doc["results"] if int(r["D"]) != D] + [row], key=lambda r: r["D"])
            _save_json(p, doc)


# --------------------------------------------------------------------------- commands

def cmd_task(field: str, t: str, D: str) -> int:
    import numpy as np
    from snapevo import mps_train
    from snapevo.cells import cell_for
    from snapevo.circuits import Layered
    from snapevo.edges import fit_edges, worth_it
    from snapevo.mps_train import OpenChainCost, init_from_ring
    from snapevo.train import load_cell

    log = _log
    t, D = int(float(t)), int(D)
    if t not in SETTINGS:
        raise SystemExit(f"t = {t} has no SETTINGS row")
    model, e = _model(field), _entry(field, t, D)
    if _skipped(e):
        r = e["rule"]
        raise SystemExit(f"{_name(e)} is skipped by the discarded-weight rule: expected discarded weight "
                         f"{r['expected_discarded']:.1e} is {r['ratio']:.1e} of 1 - F = {r['one_minus_F']:.2e} "
                         f"(limit {r['limit']:g}) at cutoff {r['cutoff']:g}, and t = {t} has no fallback")
    cell = cell_for(model)
    sd = _stage_dir(model, t, D)
    try:
        _hold(sd / "task.lock", _name(e))
    except _Locked as err:
        log(f"LOCKED, not running: {err}")
        return EXIT_LOCKED
    out = _out(model, t, D)
    prev = _read_json(out)
    if prev and not prev.get("provisional"):
        log(f"skip: {out} is final")
        return 0
    for p in list(sd.glob("*.tmp")) + list(sd.glob("*.tmp.npz")):   # leftovers of writes killed earlier
        p.unlink(missing_ok=True)
    x_ring, header = load_cell(_tdir(model, t) / "cells.json", D)
    if header["cell"] != cell.name:
        raise SystemExit(f"cells.json was trained with {header['cell']}, not {cell.name}")
    problem = _problem(e)
    meta = _jsonable({"problem": problem, "cell": cell.name, "field": field, "t": t, "D": D,
                      "x_ring_sha256": hashlib.sha256(np.asarray(x_ring, dtype=np.float64).tobytes()).hexdigest()})
    old = _read_json(sd / "task.json")
    if old is None:
        _save_json(sd / "task.json", meta)
    elif old != meta:
        diff = {k: (old.get(k), meta[k]) for k in meta if old.get(k) != meta[k]}
        raise SystemExit(f"{sd} belongs to other settings or another cell: {diff}; delete it to start over")
    starts = sd / "starts.jsonl"
    with open(starts, "a") as fh:
        fh.write(json.dumps({"time": time.time(), "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID", ""),
                             "pid": os.getpid(), "stages": sorted(p.name for p in sd.glob("*.json"))}) + "\n")
    n_starts = len(starts.read_text().splitlines())
    deadline, est, maxiter = _deadline(), _estimate(e), _maxiter(t)
    tr, ev = _train_settings(e), problem["eval"]
    threads = _threads(tr["threads"])
    mps_train.TRACE_EVERY = _trace_every(est)
    rule = e.get("rule") or _rule(e)
    log(f"task: {_name(e)}, start {n_starts}: evaluation cutoff {ev['cutoff']:.0e} chi {ev['chi_max']}, training "
        f"cutoff {tr['cutoff']:.0e} chi {tr['chi_max']}, {tr['n']} inputs, {threads} thread(s) per worker, maxiter "
        f"{maxiter}, milestones {_stages(maxiter)[:-1]}, checkpoint every {mps_train.TRACE_EVERY}, deadline "
        f"{_when(deadline)}; discarded-weight rule: {rule['decision']} (estimated ratio {rule['ratio']:.1e}); "
        f"estimates: fit {est['fit_s'] / 3600:.2f} h, {est['iter_s']:.1f} s/iteration, {est['gb']} GB")

    def short(need: float, what: str) -> bool:
        if time.time() + need > deadline:
            log(f"stop: {what} needs about {need / 60:.0f} min, {(deadline - time.time()) / 60:.0f} min are left")
            return True
        return False

    # 1. the edge fit, as production does it
    st = _read_json(sd / "fit.json")
    if st is None:
        if short(est["fit_s"] + est["full_s"], "the edge fit"):
            return EXIT_PROVISIONAL
        fit = fit_edges(Layered.brickwork(cell, x_ring), model, float(t), log=log)
        _save_json(sd / "fit.json", {"fit": fit})
    else:
        fit = st["fit"]
        log(f"fit: from {sd / 'fit.json'}")
    z0 = init_from_ring(cell, x_ring, fit)
    keep = bool(worth_it(fit))
    n_bulk = D * cell.n_angles

    with EvalSet(model, cell, t, D, ev, problem["fine"], log) as evs:
        # 2. before: held-out c with the cutoff/100 check, ring C of the production bulk
        before = _read_json(sd / "before.json")
        if before is None:
            if short(est["full_s"], "the evaluation before training"):
                return EXIT_PROVISIONAL
            before = evs.full(z0)
            before["ring"] = _ring(model, cell, t, D, x_ring)
            prod = next(r for r in header["results"] if int(r["D"]) == D)
            before["ring"]["setup_rel"] = before["ring"]["C_train"] / prod["C_train"] - 1
            tol = max(1e-6, 1e-15 / prod["C_train"])                 # make_cost's rounding near low floors
            if abs(before["ring"]["setup_rel"]) > tol:
                raise RuntimeError(f"ring C_train {before['ring']['C_train']:.10e} does not reproduce cells.json's "
                                   f"{prod['C_train']:.10e} (tolerance {tol:.1e} relative)")
            _save_json(sd / "before.json", before)
            fine = (f" ({before['held_out_fine']['c']:.6e} against the dt {problem['fine'][0]:g} references)"
                    if "held_out_fine" in before else "")
            log(f"before: held-out c = {before['held_out']['c']:.6e}{fine}, cutoff/100 change of input 0 "
                f"{before['check']['rel_change']:.1e}, ring C_train {before['ring']['C_train']:.4e} "
                f"({before['seconds']:.0f}s + ring {before['ring']['seconds']:.0f}s)")
        else:
            log(f"before: from {sd / 'before.json'}, held-out c = {before['held_out']['c']:.6e}")
        # 2b. the fit is not worth it, so the sweep uses the cell's own edges: before with those as well
        own = None
        if not keep:
            own = _read_json(sd / "before_own.json")
            if own is None:
                if short(est["full_s"], "the evaluation with the cell's own edges"):
                    return EXIT_PROVISIONAL
                own = evs.full(init_from_ring(cell, x_ring, None))
                _save_json(sd / "before_own.json", own)
                log(f"before, the cell's own edges (fit not worth it): held-out c = {own['held_out']['c']:.6e} "
                    f"({own['seconds']:.0f}s)")
            else:
                log(f"before, own edges: from {sd / 'before_own.json'}, held-out c = {own['held_out']['c']:.6e}")
        reserve = 1.5 * (before["seconds"] + before["ring"]["seconds"] + before["held_out"]["seconds"]) + 120
        stop_at = deadline - reserve

        # 3. training: one L-BFGS-B run to maxiter, the milestones from its on-iteration hook
        trained = _read_json(sd / "trained.json")
        if trained is None or "train_inputs" not in trained:
            if trained is None and short(reserve + 60, "training"):
                return EXIT_PROVISIONAL
            vecs, targets = _train_problem(model, t, tr, log)
            with OpenChainCost(model, cell, float(t), D, N, vecs, targets, chi_max=tr["chi_max"], cutoff=tr["cutoff"],
                               processes=_procs(tr["n"]), threads=threads) as cost:
                workers = cost.worker_threads()
                log(f"training workers: {len(workers) or 'none (in this process)'}, BLAS threads "
                    f"{sorted({w['blas'] for w in workers}, key=str) if workers else 'of this process'}")
                if trained is None:
                    trained = _staged_training(cost, z0, evs, sd, stop_at, maxiter, before["held_out"]["seconds"], log)
                    trained["worker_threads"] = workers
                    if not trained["provisional"]:
                        _save_json(sd / "trained.json", trained)
                    log(f"training: {trained['reason']}, iteration {trained['nit']}, train c {trained['c_end']}, "
                        f"{trained['train_seconds'] / 3600:.2f} h of training")
                cval = cost.value(np.asarray(trained["z"], dtype=float))
                F = [float(v) for v in cost.last["F"]]
                trained["train_inputs"] = {"c": float(cval), "F": F, "c_k": _c_of(F), "cutoff": tr["cutoff"],
                                           "chi_max": tr["chi_max"], "diagnostics": cost.diagnostics()}
                if not trained["provisional"]:
                    _save_json(sd / "trained.json", trained)
        else:
            log(f"training: from {sd / 'trained.json'} ({trained['reason']}, iteration {trained['nit']})")

        # 4. after
        z1 = np.asarray(trained["z"], dtype=float)
        after = evs.full(z1)
    x1, e1 = z1[:n_bulk], z1[n_bulk:]
    after["ring"] = _ring(model, cell, t, D, x1)
    ms = _read_json(sd / "milestones.json") or []
    cb, ca = before["held_out"]["c_k"], after["held_out"]["c_k"]
    bk, ak = before["check"], after["check"]
    ck_main = ca[0] - cb[0]
    ck_fine = ak["c"] - bk["c"]
    trace = trained["trace"]
    ti = trained["train_inputs"]
    expected = DISCARD_COEF * tr["cutoff"] * D
    one_mF_before = 1.0 - float(np.mean(before["held_out"]["F"]))
    one_mF_train = 1.0 - float(np.mean(ti["F"]))
    discard = {**rule, "train_cutoff": tr["cutoff"], "expected_at_train_cutoff": expected,
               "one_minus_F_before": one_mF_before, "ratio_before": expected / max(one_mF_before, 1e-300),
               "discarded_train_inputs": ti["diagnostics"]["discarded_u"], "one_minus_F_train_inputs": one_mF_train,
               "ratio_measured": ti["diagnostics"]["discarded_u"] / max(one_mF_train, 1e-300)}
    if max(discard["ratio_before"], discard["ratio_measured"]) > DISCARD_LIMIT:
        log(f"WARNING: discarded weight over {DISCARD_LIMIT:g} of 1 - F at the training cutoff {tr['cutoff']:g}: "
            f"expected/before {discard['ratio_before']:.1e}, measured on the training inputs "
            f"{discard['ratio_measured']:.1e}")
    rec = {"field": float(field), "model": model.to_dict(), "cell": cell.name, "t": float(t), "D": D, "N": N,
           "provisional": bool(trained["provisional"]), "stopped": trained["reason"], "problem": problem,
           "maxiter": maxiter, "stages": _stages(maxiter),
           "edges": {"fit": {k: v for k, v in fit.items() if k != "angles"}, "worth_it": keep,
                     "fit_seconds": float(fit["seconds"]), "angles_fit": fit["angles"], "angles_trained": e1.tolist(),
                     "quarter_fit": _quarter(fit["angles"]), "quarter_trained": _quarter(e1)},
           "before": before, "before_own_edges": own, "after": after, "milestones": ms,
           "train": {"inputs": tr["n"], "seed": tr["seed"], "cutoff": tr["cutoff"], "chi_max": tr["chi_max"],
                     "threads": threads, "worker_threads": trained.get("worker_threads"),
                     "ref": [tr["ref_dt"], tr["ref_cutoff"], tr["ref_chi"]],
                     "c_start": float(trace[0][1]) if trace else None, "c_end": trained["c_end"],
                     "nit": trained["nit"], "nfev": trained["nfev"], "message": trained["message"],
                     "reason": trained["reason"], "restart_nit": trained.get("restart_nit"),
                     "seconds": trained["train_seconds"], "trace": trace,
                     "after_on_training_inputs": ti, "diagnostics": trained["diagnostics"]},
           "discard_rule": discard,
           "c100_before": before["held_out"]["c"], "c100_after": after["held_out"]["c"],
           "c100_before_own_edges": None if own is None else own["held_out"]["c"],
           "paired": _paired(cb, ca),
           "paired_own_edges": None if own is None else _paired(own["held_out"]["c_k"], ca),
           "paired_k0": {"at_cutoff": ck_main / cb[0], "at_cutoff_100": ck_fine / bk["c"],
                         "change": (ck_fine - ck_main) / cb[0]},
           "fine": None if "held_out_fine" not in before else {
               "settings": problem["fine"], "c_before": before["held_out_fine"]["c"],
               "c_after": after["held_out_fine"]["c"],
               "paired": _paired(before["held_out_fine"]["c_k"], after["held_out_fine"]["c_k"]),
               "c_before_own_edges": None if own is None else own["held_out_fine"]["c"],
               "paired_own_edges": None if own is None else _paired(own["held_out_fine"]["c_k"],
                                                                    after["held_out_fine"]["c_k"])},
           "ring_C_train": [before["ring"]["C_train"], after["ring"]["C_train"]],
           "starts": n_starts, "requeues": n_starts - 1, "z_before": z0.tolist(), "z_after": z1.tolist()}
    _save_json(out, rec)
    _write_open_files(model, cell, t, D, x1, after["ring"], fit["angles"], e1, rec)
    p = rec["paired"]
    po = rec["paired_own_edges"]
    vs_own = f", {po['rel']:+.2%} +- {po['se']:.2%} against the cell's own edges" if po else ""
    log(f"{'provisional' if rec['provisional'] else 'done'}: {_name(e)}: held-out c {rec['c100_before']:.4e} -> "
        f"{rec['c100_after']:.4e} ({p['rel']:+.2%} +- {p['se']:.2%}, {p['improved']}/{p['n']} better{vs_own}), "
        f"{trained['nit']} iterations, ring C_train {rec['ring_C_train'][0]:.3e} -> {rec['ring_C_train'][1]:.3e}")
    return EXIT_PROVISIONAL if rec["provisional"] else 0


def cmd_refs(field: str, t: str) -> int:
    t = int(float(t))
    _train_problem(_model(field), t, _ref_settings(t), _log)
    return 0


def _parse_only(text: str) -> set:
    out = set()
    for item in text.split(","):
        f, t, D = item.split(":")
        out.add((f, int(t), int(D)))
    return out


def cmd_freeze(*argv: str) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="open_train.py freeze")
    ap.add_argument("--force", action="store_true", help="freeze although points are pending, or rewrite entries")
    ap.add_argument("--times", default="", help="comma-separated times to freeze (default all); others are kept")
    ap.add_argument("--only", default="", help="comma-separated FIELD:T:D (tests)")
    a = ap.parse_args(argv)
    times = _parse_times(a.times) if a.times else TIMES
    path = SCRATCH / "tasks.json"
    old = _frozen()
    mine = [e for e in (old or {}).get("tasks", []) if int(e["t"]) in times]
    if mine and not a.force:
        raise SystemExit(f"{path} already has {len(mine)} tasks at t in {times} (frozen {old['frozen']}); "
                         f"--force rewrites them")
    pending = _pool_pending(times)
    if pending:
        print(f"{len(pending)} points lines at t in {times} are not done:\n  " + "\n  ".join(pending))
        if not a.force:
            raise SystemExit("not freezing while the sweep still adds depths (--force to freeze anyway)")
    entries, problems = _scan(_parse_only(a.only) if a.only else None, times)
    if problems:
        raise SystemExit("not freezing:\n  " + "\n  ".join(problems))
    if mine:
        was, now = {_key(e): e for e in mine}, {_key(e): e for e in entries}
        print(f"removed: {sorted(set(was) - set(now))}\nadded: {sorted(set(now) - set(was))}\n"
              f"changed: {sorted(k for k in set(was) & set(now) if was[k] != now[k])}")
    keep = [e for e in (old or {}).get("tasks", []) if int(e["t"]) not in times]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    frozen_times = {**(old or {}).get("frozen_times", {}), **{str(t): stamp for t in times}}
    doc = {"frozen": stamp, "frozen_times": frozen_times, "times": sorted({int(e["t"]) for e in keep} | set(times)),
           "tasks": sorted(keep + entries, key=_key)}
    _save_json(path, doc)
    skipped = [e for e in entries if _skipped(e)]
    fallback = [e for e in entries if e["rule"]["decision"] == "fallback"]
    print(f"froze {len(entries)} tasks at t in {times} in {path} ({len(keep)} kept at other times)")
    for e in fallback:
        print(f"  {_name(e)}: trains at {e['rule']['cutoff']:g} (discarded-weight ratio {e['rule']['ratio']:.1e} "
              f"at {SETTINGS[e['t']]['cutoff']:g})")
    for e in skipped:
        print(f"  {_name(e)}: SKIPPED by the discarded-weight rule (ratio {e['rule']['ratio']:.1e} at "
              f"{e['rule']['cutoff']:g}, 1 - F estimated {e['rule']['one_minus_F']:.2e})")
    return 0


def _status(e: dict) -> str:
    if _skipped(e):
        return "skipped (discarded-weight rule)"
    model = _model(e["field"])
    r = _read_json(_out(model, e["t"], e["D"]))
    if r:
        return f"provisional ({r['train']['nit']} it)" if r.get("provisional") else "done"
    sd = _stage_dir(model, e["t"], e["D"])
    ck = _read_json(sd / "train.json")
    if ck:
        return f"training ({ck['nit']} it)"
    for name in ("before_own", "before", "fit"):
        if (sd / f"{name}.json").exists():
            return f"{name} saved"
    return "new"


def _node_args(argv, prog: str) -> tuple[int, float, bool, tuple[int, ...]]:
    """Cores, memory (GB), pinning and times: the whole node in a job (all cores but 2, 90% of the memory);
    outside a job `run` needs --cores, and `list` simulates a 126-core 460 GB node."""
    import argparse
    ap = argparse.ArgumentParser(prog=f"open_train.py {prog}")
    ap.add_argument("--cores", type=int, default=None, help="physical cores (default in a job: all but 2)")
    ap.add_argument("--mem-gb", type=float, default=None, dest="mem_gb", help="default in a job: 90%% of the memory")
    ap.add_argument("--times", default="", help="comma-separated times, for example 1,2,4,6 or 8,10 (default all)")
    ap.add_argument("--no-pin", action="store_true", dest="no_pin")
    a = ap.parse_args(argv)
    job = bool(os.environ.get("SLURM_JOB_ID"))
    if prog == "run" and not job and not a.cores:
        raise SystemExit("outside a SLURM job, give --cores (and keep login-node runs small)")
    n = os.cpu_count() or 2
    cores = a.cores or (n // 2 - 2 if job else 126)
    mem = a.mem_gb or (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30 * 0.9 if job else 460.0)
    return cores, mem, not a.no_pin and job, (_parse_times(a.times) if a.times else TIMES)


def cmd_list(*argv: str) -> int:
    cores, mem, _, times = _node_args(argv, "list")
    fr = _frozen()
    entries = [e for e in (fr["tasks"] if fr else _scan(times=times)[0]) if int(e["t"]) in times]
    print(f"{'frozen ' + fr['frozen'] if fr else 'NOT FROZEN (a preview; `freeze` writes the list)'}: "
          f"{len(entries)} tasks at t in {times}")
    print(f"{'h_x':>4} {'t':>2} {'D':>4} {'eval':>12} {'train':>12} {'thr':>3} {'iters':>5} {'fit h':>6} {'s/it':>7} "
          f"{'task h':>7} {'GB':>6} {'disc/(1-F)':>10}  status")
    ests = {_key(e): _estimate(e) for e in entries}
    for e in sorted(entries, key=lambda e: _priority(e, ests[_key(e)])):
        s, tr = ests[_key(e)], _train_settings(e)
        rule = e.get("rule") or _rule(e)
        print(f"{e['field']:>4} {e['t']:>2} {e['D']:>4} {e['eval']['cutoff']:.0e}/{e['eval']['chi_max']:<4} "
              f"{tr['cutoff']:.0e}/{tr['chi_max']:<4} {tr['threads']:>3} {s['maxiter']:>5} {s['fit_s'] / 3600:6.2f} "
              f"{s['iter_s']:7.2f} {s['total_s'] / 3600:7.2f} {s['gb']:6.1f} {rule['ratio']:10.1e}  "
              f"{_status(e)}{'' if rule['decision'] in ('keep', 'not applied') else ' [' + rule['decision'] + ']'}")
    sim = _simulate(entries, cores, mem)
    print(f"estimate on {cores} cores and {mem:.0f} GB: {sim['core_h']:.0f} core-h, longest task "
          f"{sim['longest_h']:.1f} h, last start {sim['last_start_h']:.1f} h, all done at {sim['makespan_h']:.1f} h "
          f"(no preemption)")
    return 0


def cmd_run(*argv: str) -> int:
    cores, mem, pin, times = _node_args(argv, "run")
    fr = _frozen()
    if not fr:
        raise SystemExit(f"no task list in {SCRATCH}: run `freeze` first")
    tasks = [e for e in fr["tasks"] if int(e["t"]) in times]
    if not tasks:
        raise SystemExit(f"no frozen tasks at t in {times} in {SCRATCH / 'tasks.json'}: `freeze --times` them first")
    for t in times:                                                # one dispatcher per time
        try:
            _hold(SCRATCH / f"run_t{t}.lock", f"open_train run at t = {t}")
        except _Locked as err:
            raise SystemExit(f"{err}; not starting a second run on the same times") from None
    deadline = _deadline()
    LOGS.mkdir(parents=True, exist_ok=True)
    for root in sorted({_out_dir(_model(e["field"]), e["t"]) for e in tasks}):     # leftovers of killed writes
        if root.exists():                                          # (tasks clean their stage and reference dirs)
            for p in root.glob("*.tmp"):
                if time.time() - p.stat().st_mtime > 3600:
                    p.unlink(missing_ok=True)
    entries = {_key(e): e for e in tasks}
    skipped = sorted(k for k, e in entries.items() if _skipped(e))
    est = {k: _estimate(e) for k, e in entries.items() if k not in skipped}
    done0 = [k for k, e in entries.items() if k not in skipped and _status(e) == "done"]
    order = lambda e: _priority(e, est[_key(e)])                   # noqa: E731
    queue = sorted((e for k, e in entries.items() if k not in done0 and k not in skipped), key=order)
    py = [str(REPO / ".venv" / "bin" / "python"), str(Path(__file__).resolve())]
    who = f"{socket.gethostname()}:{os.environ.get('SLURM_JOB_ID', 'login')}"
    _log(f"open_train run on {who}: t in {times}, {cores} cores, {mem:.0f} GB, pinning {'on' if pin else 'off'}, "
         f"deadline {_when(deadline)}; {len(entries)} tasks, {len(done0)} done, {len(skipped)} skipped by the "
         f"discarded-weight rule, {len(queue)} to run")
    free = list(range(cores))
    running: dict = {}
    attempts: dict = {}
    not_before: dict = {}
    result: dict = {}
    stop = {"why": ""}

    def on_signal(signum, _frame):
        stop["why"] = f"signal {signal.Signals(signum).name}"

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(s, on_signal)

    def cpus(mine: list[int]) -> str:
        n = os.cpu_count() or 2 * cores
        sib = n // 2 if n >= 2 * cores else 0                      # hyperthread siblings: c and c + 128
        return ",".join(map(str, sorted(mine + ([c + sib for c in mine] if sib else []))))

    def launch(e: dict, c: int, m: float) -> None:
        k = _key(e)
        mine = free[:c]
        del free[:c]
        cmd = py + ["task", e["field"], str(e["t"]), str(e["D"])]
        if pin:
            cmd = ["taskset", "-c", cpus(mine)] + cmd
        env = {**os.environ, **{v: "1" for v in THREAD_VARS}, "XLA_FLAGS": XLA_FLAGS, "PYTHONUNBUFFERED": "1",
               "SNAPEVO_RUNS": str(RUNS)}
        if deadline < math.inf:
            env["OT_DEADLINE"] = repr(deadline)
        logp = LOGS / f"task_{e['field']}_t{e['t']}_D{e['D']}.log"
        fh = open(logp, "a")
        fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {who} cores {len(mine)}: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        running[k] = {"proc": proc, "cores": mine, "mem": m, "start": time.time(), "log": fh, "path": logp, "e": e}
        _log(f"start {_name(e)} on {len(mine)} cores, {m:.0f} GB (estimate {est[k]['total_s'] / 3600:.1f} h), "
             f"pid {proc.pid}, log {logp.name}")

    def finish(k, rc: int | None, why: str = "") -> None:
        r = running.pop(k)
        free.extend(r["cores"])
        free.sort()
        el = time.time() - r["start"]
        r["log"].write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exit {rc} after {el:.0f} s{' (' + why + ')' if why else ''}\n")
        r["log"].close()
        e = r["e"]
        if not why and stop["why"] and rc in (-signal.SIGTERM, -signal.SIGKILL, 128 + signal.SIGTERM):
            why = stop["why"]                                      # the job's signal reached the task first
        if why:
            result[k] = f"stopped ({why})"
        elif rc == 0 and _status(e) == "done":
            result[k] = "done"
        elif rc == EXIT_PROVISIONAL:
            result[k] = f"unfinished: {_status(e)}"
        elif rc == EXIT_LOCKED:
            result[k] = f"failed: its stage directory is locked by another process ({r['path']})"
        else:
            attempts[k] = attempts.get(k, 0) + 1
            if attempts[k] < 2:
                not_before[k] = time.time() + 60
                queue.append(e)
                queue.sort(key=order)
                _log(f"FAILED {_name(e)} rc {rc} after {el / 60:.1f} min; retrying once in 60 s ({r['path']})")
                return
            result[k] = f"failed twice, last rc {rc} ({r['path']})"
        _log(f"{result[k]}: {_name(e)} after {el / 60:.1f} min")

    def kill_all(why: str) -> None:
        for r in running.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(r["proc"].pid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < 20 and any(r["proc"].poll() is None for r in running.values()):
            time.sleep(0.5)
        for k in list(running):
            p = running[k]["proc"]
            if p.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            finish(k, p.returncode, why)

    while True:
        for k, r in list(running.items()):
            rc = r["proc"].poll()
            if rc is not None:
                finish(k, rc)
        now = time.time()
        if stop["why"] or now >= deadline:
            why = stop["why"] or "deadline"
            _log(f"stopping ({why}), {len(running)} tasks running")
            kill_all(why)
            break
        mem_used = sum(r["mem"] for r in running.values())
        startable = False
        for e in list(queue):
            k = _key(e)
            if not_before.get(k, 0) > now:
                startable = True
                continue
            if now + _min_wall(e, est[k]) > deadline:
                continue
            startable = True
            c, m = min(est[k]["cores"], cores), min(est[k]["gb"], mem)
            if c <= len(free) and mem_used + m <= mem:
                queue.remove(e)
                launch(e, c, m)
                mem_used += m
        if not running and (not queue or not startable):
            break
        time.sleep(5)

    for e in queue:
        why = f"stopped: {stop['why']}" if stop["why"] else "no time left before the deadline"
        result.setdefault(_key(e), f"not started: {_status(e)} ({why})")
    failed = {k: v for k, v in result.items() if v.startswith("failed")}
    open_ = {k: v for k, v in result.items() if v != "done" and k not in failed}
    _log(f"open_train done (t in {times}): {len(done0) + sum(v == 'done' for v in result.values())} of "
         f"{len(entries) - len(skipped)} tasks done, {len(skipped)} skipped")
    for k, v in sorted(failed.items()) + sorted(open_.items()):
        print(f"  h_x = {k[0]}, t = {k[1]}, D = {k[2]}: {v}", flush=True)
    for k in skipped:
        print(f"  h_x = {k[0]}, t = {k[1]}, D = {k[2]}: skipped by the discarded-weight rule", flush=True)
    return 1 if failed else (2 if open_ else 0)


def _control(field: str, t: int, D: int) -> dict | None:
    """The ring-retraining control of size_transfer.py at this depth (10 sites, WARM_ITERS iterations)."""
    return _read_json(RUNS / "size_transfer" / f"warm_{field}_{t}_{D}_10.json")


def cmd_table() -> int:
    fr = _frozen()
    skipped = []
    if fr:
        keys = sorted(_key(e) for e in fr["tasks"] if not _skipped(e))
        skipped = sorted((e for e in fr["tasks"] if _skipped(e)), key=_key)
    else:
        keys = []
        for f in FIELDS:
            for t in TIMES:
                for p in _out_dir(_model(f), t).glob("D*.json"):
                    keys.append((f, t, int(p.stem[1:])))
        keys.sort()
    pct = lambda p: f"{p['rel']:+.2%} ± {p['se']:.2%}"                 # noqa: E731
    print("Held-out c at N = 100 (8 inputs, seed 2026, the depth's tightest sweep settings); paired changes are "
          "input by input against before (the ring cell with fitted edges; \"vs own edges\": against the cell's "
          "own edges, which the sweep uses when the fit is not worth it). Control: size_transfer.py's ring "
          "retraining (10 sites, 3000 iterations) on the same inputs. disc/(1-F): the discarded weight on the "
          "training inputs at the training cutoff over their 1 - F (rule: below 1e-3).\n")
    print("| h_x | t | D | cutoff | train cutoff | c before | c after | change | better | change vs own edges | "
          "change at 100 it | control c100 | after vs control | change, dt 0.1 references | "
          "ring C_train before -> after | iterations | train h | starts | cutoff/100 change of the paired k=0 | "
          "disc/(1-F) |")
    print("|" + "---|" * 20)
    missing = 0
    for f, t, D in keys:
        r = _read_json(_out(_model(f), t, D))
        if not r:
            missing += 1
            continue
        cb, ca = r["before"]["held_out"]["c_k"], r["after"]["held_out"]["c_k"]
        m100 = next((m for m in r["milestones"] if m["iter"] == 100), None)
        if m100:
            at100 = pct(_paired(cb, m100["held_out"]["c_k"]))
        elif not r["provisional"] and r["train"]["nit"] <= 100:
            at100 = pct(r["paired"])                               # training ended at or before iteration 100
        else:
            at100 = "-"
        ctl = _control(f, t, D)
        if ctl:
            same = math.isclose(ctl["mps100"]["cutoff"], r["problem"]["eval"]["cutoff"])
            ctl_s = f"{ctl['c100']['mean']:.3e}" + ("" if same else f" (at {ctl['mps100']['cutoff']:.0e})")
            vs = pct(_paired(ctl["c100"]["c"], ca)) if same else "-"
        else:
            ctl_s = vs = "-"
        tr = r["train"]
        prov = " (provisional)" if r["provisional"] else ""
        own = r.get("paired_own_edges")
        fine = r.get("fine")
        fine_s = "-" if not fine else pct(fine["paired"]) + (
            f" (vs own edges {pct(fine['paired_own_edges'])})" if fine.get("paired_own_edges") else "")
        disc = (r.get("discard_rule") or {}).get("ratio_measured")
        print(f"| {f} | {t} | {D} | {r['problem']['eval']['cutoff']:.0e} | {tr['cutoff']:.0e} | "
              f"{r['c100_before']:.4e} | {r['c100_after']:.4e} | {pct(r['paired'])} | "
              f"{r['paired']['improved']}/{r['paired']['n']} | {pct(own) if own else '-'} | {at100} | {ctl_s} | {vs} | "
              f"{fine_s} | {r['ring_C_train'][0]:.3e} -> {r['ring_C_train'][1]:.3e} | {tr['nit']}{prov} | "
              f"{tr['seconds'] / 3600:.2f} | {r['starts']} | {r['paired_k0']['change']:+.1e} | "
              f"{'-' if disc is None else f'{disc:.1e}'} |")
    if missing:
        print(f"\n{missing} of {len(keys)} tasks have no result yet.")
    for e in skipped:
        ru = e["rule"]
        print(f"\nSkipped by the discarded-weight rule: {_name(e)} (expected discarded weight {ru['expected_discarded']:.1e}"
              f" at cutoff {ru['cutoff']:g} is {ru['ratio']:.1e} of the estimated 1 - F = {ru['one_minus_F']:.2e}).")
    return 0


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["list"]
    commands = {"freeze": cmd_freeze, "list": cmd_list, "refs": cmd_refs, "task": cmd_task, "run": cmd_run,
                "table": cmd_table}
    if cmd not in commands:
        raise SystemExit(f"unknown command {cmd!r}; one of {sorted(commands)}")
    sys.exit(commands[cmd](*rest) or 0)
