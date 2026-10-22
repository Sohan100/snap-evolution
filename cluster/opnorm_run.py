"""Operator-norm error against the per-site infidelity over the sweep's circuits (snapevo.opnorm).

    python cluster/opnorm_run.py list  [--out DIR] [--times 1,2,4] [--sources snap,baseline] [--freeze]
    python cluster/opnorm_run.py task  NAME [--out DIR] [--real] [--threads K]
    python cluster/opnorm_run.py run   [--out DIR] [--real] [--cores N | --cpus 60-63] [--only REGEX] [--times ..]
    python cluster/opnorm_run.py table [--out DIR] [--fit-N 14]
    python cluster/opnorm_run.py plot  [--out DIR] [--fig DIR] [--fit-N 14]

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0 (snapevo's model keys A and B).

What is measured. For every circuit, size N and boundary (ring, open chain with the circuit's fitted edges as
evaluated), against exact V = e^{-iHt} (opnorm.measure): the eigenphase arc width w of V^dag U by Lanczos,
eps_op = 2 sin(w/4), D_coh = 2 sin(w/2), and on the same unitary the per-site infidelity C(N) on 64 random
product inputs and the Haar values from 16 Haar states; at N <= 10 also the full diagonalization. The
paper's metric, c at N = 100 by MPS on the open chain, is read from the point's own files.

Sources (--sources; a task name starts with its source):
    snap      every realized SNAP point, runs/<model>/t<T>/real_<stem>.json (c at N = 100 from eval_<stem>.json)
    baseline  every Trotter + KN point, runs/<model>/t<T>/baseline_<stem>.json, rebuilt from its recorded
              success-branch angles (opnorm.baseline_circuit; no synthesis is rerun)
    trotter   the pure product formula (no synthesis) at each (order, r) of the baseline points
    open      the open-chain-trained cells, runs/<model>/t<T>/open/{cells,edges}.json (cluster/open_train.py),
              continuous angles; c at N = 100 is their c100_after
    lit       literature product formulas (snapevo.baseline_lit), open chain only. PROVISIONAL, until
              baseline_lit stores its points: points from --lit-points FILE (a JSON list of baseline_lit
              scan points: ordering, nesting, fold, order, steps, eps, and field and t), realized at each N
              with baseline_lit.realize and a KNCache in --kn-dir (seeded read-only from runs/kn_cache.json)
The default sources are snap and baseline, at t <= 4.

Sizes. Every circuit at N = 8, 10, 12, 14, ring and open ("base" tasks, name@8,10,12,14). N = 16 for about
30 representative circuits: per field and t in {1, 2, 4}, the SNAP points nearest c = 1e-3, 1e-5, 1e-7 and
the baseline points nearest 1e-4, 1e-6 (c at N = 100, else the 10-site ring C). N = 18 and 20 for 6, the
extrapolation check (and N = 16 for them too): the SNAP point nearest 1e-6 per field at t = 2 and 4, and the
baseline point nearest 1e-5 per field at t = 2. ``list --freeze`` (or the first ``run``) writes the task list
to <out>/opnorm_tasks.json and later runs keep it (``--refresh`` recomputes it).

Results: one JSON per circuit, size and boundary, <out>/<model>/t<T>/opnorm/<source>_<stem>_N<N>_<bc>.json,
written atomically; a task skips the ones that exist, so any run resumes. <out> is --out, else $OPNORM_OUT,
else runs/ (or $SNAPEVO_RUNS); writing under runs/ needs --real. Tests and demos use
$SCRATCH/snapevo-scratch/opnorm2/.

run. One node, as open_train.py does: tasks longest (estimated) first, each a subprocess pinned with taskset to
its own cores (and their hyperthread siblings), with that many threads: 2 for base tasks, 4 at N = 16, 8 at
N = 18 and 20. In a SLURM job it takes all cores but 2; outside one it needs --cores. For example
    sbatch -A m4980 -C cpu -N 1 --exclusive -q preempt --requeue -t 06:00:00 --wrap \\
        "module load python; source .venv/bin/activate; python cluster/opnorm_run.py run --real"
(not submitted by this script). A requeued job resumes from the files.

Cost model (``list`` prints the estimates): the seconds of one (N, bc) are
    matvecs(N) x [2 D s_layer(N) + K s_term(N)] + 10 x [D s_layer(N) layer8 + K s_term(N) term8] + overhead(N)
(the Lanczos run, then C and the Haar values on 80 states in blocks of 8), with D sublayers and K Chebyshev
terms (opnorm.chebyshev_terms of a t, a the half spectral span); the constants (COST) were measured on a
loaded login node on 15 Oct 2026, see the comment there. Compute nodes should be faster.
"""

from __future__ import annotations

import argparse
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
RUNS = Path(os.environ.get("SNAPEVO_RUNS", REPO / "runs"))
SCRATCH = Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch"
FIELDS = {"1.6": "A", "0": "B"}               # snapevo's model keys: A is h_x = 1.6, B is h_x = 0
TAGS = {"1.6": "xxz_d0.25_hx1.6", "0": "xxz_d0.25_hx0"}
TIMES = (1, 2, 4)
SOURCES = ("snap", "baseline", "trotter", "open", "lit")
DEFAULT_SOURCES = ("snap", "baseline")
BASE_SIZES = (8, 10, 12, 14)
REP_TARGETS = {"snap": (1e-3, 1e-5, 1e-7), "baseline": (1e-4, 1e-6)}
EXT_TARGETS = ((2, "snap", 1e-6), (4, "snap", 1e-6), (2, "baseline", 1e-5))
EXT_SIZES = (18, 20)
BCS = ("ring", "open")
VERSION = 1

# Measured on login08 (AMD EPYC 7713; load average about 110, at the end several thousand), 15 Oct 2026, over
# the demo's 242 results (snapevo-scratch/opnorm2/demo): s_layer(N), seconds of one sublayer on one vector
# (apply_ops, not threaded); s_term(N), seconds of one Chebyshev term on 2 vectors (2 threads at N <= 14,
# 4 at N = 16 and 18); matvecs(N), median G applications of one boundary's arc (open chains take more);
# overhead(N), seconds of the rest of one boundary (H, spectral bounds, the expm_multiply check at N <= 14,
# the full diagonalization at N <= 10). 8-vector blocks cost layer8 x one vector's sublayer and term8 x a
# 2-vector term. N = 20 is extrapolated (x 4 per 2 sites from N = 18): an estimate, not a measurement.
COST = {
    "s_layer": {6: 1.5e-5, 8: 3.1e-5, 10: 7.3e-5, 12: 2.2e-4, 14: 8.3e-4, 16: 3.3e-3, 18: 1.33e-2, 20: 5.3e-2},
    "s_term": {6: 2.0e-5, 8: 3.7e-5, 10: 9.1e-5, 12: 3.5e-4, 14: 1.5e-3, 16: 3.6e-3, 18: 1.05e-2, 20: 4.2e-2},
    "matvecs": {6: 40, 8: 50, 10: 62, 12: 70, 14: 82, 16: 96, 18: 116, 20: 135},
    "overhead": {6: 0.1, 8: 0.2, 10: 3.0, 12: 0.5, 14: 2.0, 16: 3.0, 18: 15.0, 20: 60.0},
    "layer8": 3.5, "term8": 2.6,
    "span_per_site": {"1.6": 1.63, "0": 0.65},      # (E_max - E_min)/N of H, measured (ring; open within 2%)
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _model(field: str):
    from snapevo.hamiltonian import get_model
    return get_model(FIELDS[field])


def _tdir(field: str, t: float, root: Path = RUNS) -> Path:
    return Path(root) / TAGS[field] / f"t{t:g}"


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, default=_default))
    os.replace(tmp, path)


def _default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- the circuits

def _snap_entries(field: str, t: int) -> list[dict]:
    d = _tdir(field, t)
    out = []
    for p in sorted(d.glob("real_b*_D*.json")):
        stem = p.stem[len("real_"):]
        ev = _read(d / f"eval_{stem}.json") or {}
        rec = _read(p)
        out.append({"source": "snap", "field": field, "t": t, "stem": stem, "D": int(rec["D"]),
                    "edges_used": bool((rec.get("edges") or {}).get("use")), "c100": ev.get("c_site"),
                    "C_ring10": ev.get("C_ring"), "Tq": ev.get("Tq"), "budget": rec.get("budget")})
    return out


def _baseline_entries(field: str, t: int) -> list[dict]:
    from snapevo.baseline import trotter_circuit
    d = _tdir(field, t)
    out = []
    for p in sorted(d.glob("baseline_o*_r*_e*.json")):
        rec = _read(p)
        if not rec or "synth" not in rec:
            continue
        D = trotter_circuit(_model(field), float(t), int(rec["order"]), int(rec["steps"])).depth
        out.append({"source": "baseline", "field": field, "t": t, "stem": p.stem[len("baseline_"):], "D": D,
                    "edges_used": False, "c100": rec.get("c_site"), "C_ring10": rec.get("C_ring"),
                    "Tq": rec.get("Tq"), "order": int(rec["order"]), "steps": int(rec["steps"]),
                    "eps": float(rec["eps"])})
    return out


def _trotter_entries(field: str, t: int) -> list[dict]:
    from snapevo.baseline import trotter_circuit
    seen = sorted({(b["order"], b["steps"]) for b in _baseline_entries(field, t)})
    return [{"source": "trotter", "field": field, "t": t, "stem": f"o{o}_r{r}", "D":
             trotter_circuit(_model(field), float(t), o, r).depth, "edges_used": False, "c100": None,
             "C_ring10": None, "Tq": None, "order": o, "steps": r} for o, r in seen]


def _open_entries(field: str, t: int) -> list[dict]:
    d = _tdir(field, t) / "open"
    doc = _read(d / "cells.json")
    if not doc:
        return []
    out = []
    for r in doc["results"]:
        out.append({"source": "open", "field": field, "t": t, "stem": f"D{int(r['D'])}", "D": int(r["D"]),
                    "edges_used": True, "c100": r.get("c100_after"), "C_ring10": r.get("C_val"), "Tq": None,
                    "provisional": bool(r.get("provisional", False))})
    return out


def _lit_points(path: str | None) -> list[dict]:
    if not path:
        return []
    pts = json.loads(Path(path).read_text())
    return pts if isinstance(pts, list) else pts.get("points", [])


def _lit_stem(p: dict) -> str:
    fam = f"{p['ordering']}-{p.get('nesting', 'B')}{'-fold' if p.get('fold') else ''}-o{p['order']}"
    return re.sub(r"[^A-Za-z0-9_.+-]", "", f"{fam}_r{p['steps']}_e{float(p['eps']):.0e}")


def _lit_entries(field: str, t: int, lit_points: str | None) -> list[dict]:
    out = []
    for p in _lit_points(lit_points):
        if str(p.get("field", "")) != field or float(p.get("t", -1)) != float(t):
            continue
        out.append({"source": "lit", "field": field, "t": t, "stem": _lit_stem(p), "D": p.get("n_gates"),
                    "edges_used": False, "c100": p.get("c_site"), "C_ring10": p.get("C"), "Tq": p.get("Tq"),
                    "point": p})
    return out


def entries(sources=DEFAULT_SOURCES, times=TIMES, lit_points: str | None = None) -> list[dict]:
    """Every circuit of the given sources and times, both fields."""
    out = []
    for field in FIELDS:
        for t in times:
            for s in sources:
                if s == "snap":
                    out += _snap_entries(field, t)
                elif s == "baseline":
                    out += _baseline_entries(field, t)
                elif s == "trotter":
                    out += _trotter_entries(field, t)
                elif s == "open":
                    out += _open_entries(field, t)
                elif s == "lit":
                    out += _lit_entries(field, t, lit_points)
                else:
                    raise SystemExit(f"unknown source {s!r}; choose from {SOURCES}")
    return out


def cname(e: dict) -> str:
    return f"{e['source']}:{e['field']}:{e['t']:g}:{e['stem']}"


def parse(name: str) -> tuple[str, str, int, str, tuple[int, ...]]:
    """'source:field:t:stem@8,10' -> (source, field, t, stem, sizes)."""
    head, _, sizes = name.partition("@")
    source, field, t, stem = head.split(":", 3)
    if source not in SOURCES or field not in FIELDS:
        raise SystemExit(f"bad task name {name!r}")
    return source, field, int(float(t)), stem, tuple(int(v) for v in sizes.split(",")) if sizes else BASE_SIZES


def load(source: str, field: str, t: int, stem: str, lit_points: str | None = None, kn_dir: str | None = None):
    """(model, opnorm.Circuit, entry) of one circuit."""
    from snapevo import opnorm as on
    model = _model(field)
    d = _tdir(field, t)
    name = f"{source}:{field}:{t:g}:{stem}"
    if source == "snap":
        rec = _read(d / f"real_{stem}.json")
        if rec is None:
            raise SystemExit(f"{d / f'real_{stem}.json'} missing")
        c = on.snap_circuit(rec, name)
        e = next(x for x in _snap_entries(field, t) if x["stem"] == stem)
    elif source == "baseline":
        rec = _read(d / f"baseline_{stem}.json")
        if rec is None:
            raise SystemExit(f"{d / f'baseline_{stem}.json'} missing")
        c = on.baseline_circuit(rec, model, name)
        e = next(x for x in _baseline_entries(field, t) if x["stem"] == stem)
    elif source == "trotter":
        o, r = (int(v[1:]) for v in stem.split("_"))
        c = on.trotter(model, float(t), o, r, name)
        e = {"source": "trotter", "field": field, "t": t, "stem": stem, "D": c.depth, "edges_used": False,
             "c100": None, "C_ring10": None, "Tq": None, "order": o, "steps": r}
    elif source == "open":
        D = int(stem[1:])
        c = on.open_trained_circuit(_read(d / "open" / "cells.json"), _read(d / "open" / "edges.json"), D, name)
        e = next(x for x in _open_entries(field, t) if x["stem"] == stem)
    else:                                                       # lit, provisional
        e = next((x for x in _lit_entries(field, t, lit_points) if x["stem"] == stem), None)
        if e is None:
            raise SystemExit(f"{name}: not among --lit-points")
        c = _lit_circuit(e["point"], model, float(t), kn_dir, name)
    return model, c, e


def _lit_circuit(p: dict, model, t: float, kn_dir: str | None, name: str):
    """A baseline_lit point realized at each N on the open chain (provisional, see the module docstring)."""
    from snapevo import baseline_lit as bl
    from snapevo import opnorm as on
    f = bl.Formula(model, t, int(p["order"]), int(p["steps"]), p["ordering"], p.get("nesting", "XB"),
                   bool(p.get("fold", False)))
    synth = bl.KNCache(Path(kn_dir or SCRATCH / "litbase" / "kn"), seed=RUNS / "kn_cache.json")
    return on.Circuit(gates=lambda N, bc: bl.realize(f, float(p["eps"]), N, synth)["gates"], bcs=("open",),
                      name=name, info={"family": f.family, "steps": f.r, "eps": float(p["eps"])})


# --------------------------------------------------------------------------- the task list

def _nearest(pool: list[dict], target: float, taken: set) -> dict | None:
    def key(e):
        c = e["c100"] if e["c100"] else e["C_ring10"]
        return abs(math.log(c / target)) if c and c > 0 else math.inf
    cand = sorted((e for e in pool if cname(e) not in taken), key=key)
    return cand[0] if cand and key(cand[0]) < math.inf else None


def representatives(es: list[dict]) -> tuple[list[str], list[str]]:
    """(circuits at N = 16, circuits at N = 18 and 20), by the rules of the module docstring."""
    rep, ext = [], []
    for field in FIELDS:
        for t in TIMES:
            for source, targets in REP_TARGETS.items():
                pool = [e for e in es if e["field"] == field and e["t"] == t and e["source"] == source]
                for c in targets:
                    e = _nearest(pool, c, set(rep))
                    if e:
                        rep.append(cname(e))
        for t, source, c in EXT_TARGETS:
            pool = [e for e in es if e["field"] == field and e["t"] == t and e["source"] == source]
            e = _nearest(pool, c, set(ext))
            if e:
                ext.append(cname(e))
    return rep, ext


def _span(field: str) -> float:
    return COST["span_per_site"][field]


def estimate(e: dict, N: int, bcs=BCS) -> float:
    """Estimated seconds of one circuit at one size, both boundaries (COST, module docstring)."""
    from snapevo.opnorm import chebyshev_terms
    D = e["D"] or 40
    K = chebyshev_terms(0.5 * _span(e["field"]) * N * e["t"])[0]
    sl, st = COST["s_layer"][N], COST["s_term"][N]
    one = (COST["matvecs"][N] * (2 * D * sl + K * st) + 10 * (D * sl * COST["layer8"] + K * st * COST["term8"])
           + COST["overhead"][N])
    return len(bcs) * one


def plan(sources=DEFAULT_SOURCES, times=TIMES, lit_points: str | None = None) -> list[dict]:
    es = entries(sources, times, lit_points)
    rep, ext = representatives([e for e in es if e["source"] in ("snap", "baseline")])
    tasks = []
    for e in es:
        bcs = ("open",) if e["source"] == "lit" else BCS
        n = cname(e)
        groups = [BASE_SIZES] + ([(16,)] if n in rep or n in ext else []) \
            + ([(N,) for N in EXT_SIZES] if n in ext else [])
        for sizes in groups:
            tasks.append({"name": f"{n}@{','.join(map(str, sizes))}", "circuit": n, "sizes": list(sizes),
                          "D": e["D"], "est_s": sum(estimate(e, N, bcs) for N in sizes),
                          "threads": 2 if max(sizes) <= 14 else (4 if max(sizes) == 16 else 8)})
    return tasks


# --------------------------------------------------------------------------- output

def _under_runs(path: Path) -> bool:
    path, runs = Path(path).resolve(), RUNS.resolve()
    return path == runs or runs in path.parents


def out_root(args, write: bool = True) -> Path:
    """The results root; writing under runs/ needs --real."""
    root = Path(args.out or os.environ.get("OPNORM_OUT") or RUNS).resolve()
    if write and _under_runs(root) and not getattr(args, "real", False):
        raise SystemExit(f"{root} is under {RUNS}: pass --real to write there, or --out DIR "
                         f"(tests and demos: {SCRATCH / 'opnorm2'})")
    return root


def result_path(root: Path, source: str, field: str, t: int, stem: str, N: int, bc: str) -> Path:
    return Path(root) / TAGS[field] / f"t{t:g}" / "opnorm" / f"{source}_{stem}_N{N}_{bc}.json"


def _strip(m: dict) -> dict:
    """measure()'s record without the per-attempt Lanczos history's bulk."""
    m = json.loads(json.dumps(m, default=_default))
    for h in m["arc"].get("history", []):
        h.pop("absz", None)
    return m


def cmd_task(args) -> int:
    from snapevo import opnorm as on
    root = out_root(args)
    source, field, t, stem, sizes = parse(args.name)
    if args.sizes:
        sizes = tuple(int(v) for v in args.sizes.split(","))
    model, c, e = load(source, field, t, stem, args.lit_points, args.kn_dir)
    threads = args.threads or on.default_threads()
    for N in sizes:
        for bc in c.bcs:
            p = result_path(root, source, field, t, stem, N, bc)
            if p.exists():
                _log(f"skip {p.name}")
                continue
            t0 = time.time()
            m = on.measure(c, model, float(t), N, bc, threads=threads)
            rec = {"version": VERSION, "circuit": {**{k: v for k, v in e.items() if k != "point"}, "name": cname(e),
                                                     "edges_on_open": bool(bc == "open" and c.edges is not None)},
                   "N": N, "bc": bc, "result": _strip(m), "host": socket.gethostname(),
                   "job": os.environ.get("SLURM_JOB_ID"), "threads": threads,
                   "when": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": time.time() - t0}
            _save_json(p, rec)
            a = m["arc"]
            sat = ", SATURATED" if m["saturated"] else ""
            _log(f"{cname(e)} N={N} {bc}: w/N {m['w'] / N:.4e} (eps_op {m['eps_op']:.3e}{sat}), "
                 f"C {m['C']:.3e}, kappa {m['kappa']:.3f}, {a['matvecs']} matvecs, K {a['cheb_terms']}, "
                 f"res {max(a['res']):.1e}" + (f", dense dw {m['dense_dw']:.1e}" if "dense_dw" in m else "")
                 + f", floor {'ok' if m['floor_ok'] else 'VIOLATED'}, {rec['seconds']:.1f}s "
                   f"(arc {a['seconds']:.1f}: circuit {a['seconds_circuit']:.1f}, prop {a['seconds_prop']:.1f}; "
                   f"fid {m['fid']['seconds']:.1f})")
    return 0


def _tasks(args) -> list[dict]:
    root = out_root(args, write=getattr(args, "freeze", False) or args.cmd == "run")
    frozen = _read(root / "opnorm_tasks.json")
    if frozen and not getattr(args, "refresh", False):
        return frozen["tasks"]
    tasks = plan(tuple(args.sources.split(",")), tuple(int(v) for v in args.times.split(",")), args.lit_points)
    if getattr(args, "freeze", False) or args.cmd == "run":
        _save_json(root / "opnorm_tasks.json", {"frozen": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": args.sources,
                                                "times": args.times, "lit_points": args.lit_points, "tasks": tasks})
    return tasks


def _done(root: Path, task: dict) -> bool:
    source, field, t, stem, sizes = parse(task["name"])
    bcs = ("open",) if source == "lit" else BCS
    return all(result_path(root, source, field, t, stem, N, bc).exists() for N in sizes for bc in bcs)


def cmd_list(args) -> int:
    root = out_root(args, write=args.freeze)
    tasks = _tasks(args)
    by = {}
    for tk in tasks:
        by.setdefault(tuple(tk["sizes"]), []).append(tk)
    print(f"{len(tasks)} tasks, {len({tk['circuit'] for tk in tasks})} circuits, results in {root}")
    for sizes, tks in sorted(by.items()):
        done = sum(_done(root, tk) for tk in tks)
        s = [tk["est_s"] for tk in tks]
        print(f"  N = {','.join(map(str, sizes)):>11}: {len(tks):4d} tasks ({done} done), {tks[0]['threads']} threads "
              f"each, estimate {min(s):.0f} to {max(s):.0f} s per task, {sum(s) * tks[0]['threads'] / 3600:.1f} core-h")
    if args.verbose:
        for tk in sorted(tasks, key=lambda k: -k["est_s"]):
            print(f"{tk['name']:60s} D {tk['D']!s:>4} est {tk['est_s']:8.0f} s  {'done' if _done(root, tk) else ''}")
    core_h = sum(tk["est_s"] * tk["threads"] for tk in tasks) / 3600
    longest = max(tk["est_s"] for tk in tasks) / 3600
    print(f"estimate: {core_h:.1f} core-h in all (login-node speeds); on 126 cores at least "
          f"{max(core_h / 126, longest):.1f} h wall (longest task {longest:.1f} h)")
    return 0


def _cpu_list(text: str) -> list[int]:
    """'60-63,70' -> [60, 61, 62, 63, 70]."""
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def cmd_run(args) -> int:
    root = out_root(args)
    job = bool(os.environ.get("SLURM_JOB_ID"))
    if not job and not (args.cores or args.cpus):
        raise SystemExit("outside a SLURM job, give --cores or --cpus (and keep login-node runs small)")
    cpu_list = _cpu_list(args.cpus) if args.cpus else None
    cores = len(cpu_list) if cpu_list else (args.cores or (os.cpu_count() or 4) // 2 - 2)
    tasks = [tk for tk in _tasks(args) if not _done(root, tk)]
    if args.only:
        rx = re.compile(args.only)
        tasks = [tk for tk in tasks if rx.search(tk["name"])]
    tasks.sort(key=lambda tk: -tk["est_s"])
    py = [str(REPO / ".venv" / "bin" / "python"), str(Path(__file__).resolve())]
    logs = root / "opnorm_logs"
    logs.mkdir(parents=True, exist_ok=True)
    n_cpu = os.cpu_count() or 2 * cores
    sib = n_cpu // 2 if n_cpu >= 2 * cores and job and not cpu_list else 0
    free = list(cpu_list or range(cores))
    running: dict = {}
    failed: list[str] = []
    stop = {"why": ""}

    def on_signal(signum, _frame):
        stop["why"] = signal.Signals(signum).name

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(s, on_signal)
    _log(f"opnorm run on {socket.gethostname()}: {len(tasks)} tasks to run, {cores} cores, results in {root}")
    queue = list(tasks)
    while queue or running:
        for k, r in list(running.items()):
            rc = r["proc"].poll()
            if rc is not None:
                free.extend(r["cores"])
                free.sort()
                r["fh"].close()
                el = time.time() - r["start"]
                if rc != 0:
                    failed.append(k)
                _log(f"{'done' if rc == 0 else f'FAILED rc {rc}'}: {k} after {el:.0f} s (estimate {r['est']:.0f} s)")
                del running[k]
        if stop["why"]:
            _log(f"stopping ({stop['why']}), {len(running)} tasks running")
            for r in running.values():
                try:
                    os.killpg(r["proc"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for r in running.values():
                r["proc"].wait()
            return 2
        for tk in list(queue):
            c = min(tk["threads"], cores)
            if c <= len(free):
                mine, free[:] = free[:c], free[c:]
                cpus = ",".join(map(str, sorted(mine + ([x + sib for x in mine] if sib else []))))
                cmd = py + ["task", tk["name"], "--out", str(root), "--threads", str(c)] \
                    + (["--real"] if args.real else []) \
                    + (["--lit-points", args.lit_points] if args.lit_points else []) \
                    + (["--kn-dir", args.kn_dir] if args.kn_dir else [])
                cmd = ["taskset", "-c", cpus] + cmd
                env = {**os.environ, "OMP_NUM_THREADS": str(c), "OPENBLAS_NUM_THREADS": str(c),
                       "MKL_NUM_THREADS": str(c), "PYTHONUNBUFFERED": "1"}
                fh = open(logs / (re.sub(r"[^A-Za-z0-9_.@,-]", "_", tk["name"]) + ".log"), "a")
                fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] cpus {cpus}: {' '.join(cmd)}\n")
                fh.flush()
                proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
                running[tk["name"]] = {"proc": proc, "cores": mine, "start": time.time(), "fh": fh, "est": tk["est_s"]}
                queue.remove(tk)
        time.sleep(2)
    _log(f"opnorm run done: {len(tasks) - len(failed)} of {len(tasks)} tasks succeeded"
         + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


# --------------------------------------------------------------------------- analysis

def collect(root: Path) -> dict:
    """{circuit name: {"entry": ..., "rows": {(N, bc): result}}} from every result file under root."""
    out: dict = {}
    for p in sorted(Path(root).glob("*/t*/opnorm/*_N*_*.json")):
        rec = _read(p)
        if not rec or rec.get("version") != VERSION:
            continue
        c = out.setdefault(rec["circuit"]["name"], {"entry": rec["circuit"], "rows": {}})
        c["rows"][(int(rec["N"]), rec["bc"])] = rec["result"]
    return out


def _fit(xs, ys) -> dict:
    import numpy as np
    x, y = np.log10(np.asarray(xs, dtype=float)), np.log10(np.asarray(ys, dtype=float))
    if x.size < 3:
        return {"n": int(x.size), "slope": math.nan, "se": math.nan, "pref": math.nan, "scatter": math.nan}
    p, cov = np.polyfit(x, y, 1, cov=True)
    r = y - np.polyval(p, x)
    return {"n": int(x.size), "slope": float(p[0]), "se": float(math.sqrt(cov[0, 0])), "pref": float(10 ** p[1]),
            "scatter": float(np.max(np.abs(r)))}


def _usable(r: dict) -> bool:
    return r is not None and not r["saturated"] and r["C"] > 0 and r["w"] > 0


def analysis(data: dict, fit_N: int = 14) -> dict:
    """Fits of w/N against C at fit_N, kappa statistics, the open-chain extrapolation per circuit."""
    import numpy as np
    from snapevo import opnorm as on
    rows = []
    for name, c in data.items():
        e, R = c["entry"], c["rows"]
        row = {"name": name, **{k: e.get(k) for k in ("source", "field", "t", "stem", "D", "c100", "C_ring10",
                                                      "edges_used")}}
        for bc in BCS:
            r = R.get((fit_N, bc))
            if r is not None:
                row[f"C_{bc}"], row[f"w_{bc}"], row[f"eps_{bc}"] = r["C"], r["w"], r["eps_op"]
                row[f"kappa_{bc}"] = r["kappa"] if _usable(r) else math.nan
                row[f"sat_{bc}"] = r["saturated"]
                row[f"c_haar_{bc}"] = r.get("c_haar")
        opn = {N: r for (N, bc), r in R.items() if bc == "open"}
        fitN = [N for N in (10, 12, 14, 16) if N in opn and _usable(opn[N])]
        if len(fitN) >= 2:
            checks = {N: opn[N]["w"] for N in EXT_SIZES if N in opn and _usable(opn[N])}
            row["ext"] = on.extrapolate(fitN, [opn[N]["w"] for N in fitN], targets=(50, 100),
                                        c=e.get("c100") or (row.get("C_open") or None), checks=checks or None)
            if e.get("c100"):
                row["kappa100"] = row["ext"]["alpha"] / math.sqrt(e["c100"])
        rows.append(row)
    fits = {}
    for bc in BCS:
        ok = [r for r in rows if r.get(f"sat_{bc}") is False and r.get(f"C_{bc}", 0) > 0]
        groups = {"all": ok}
        for s in ("snap", "baseline", "trotter", "open", "lit"):
            groups[s] = [r for r in ok if r["source"] == s]
        for f in FIELDS:
            groups[f"h_x = {f}"] = [r for r in ok if r["field"] == f]
        for t in TIMES:
            groups[f"t = {t}"] = [r for r in ok if r["t"] == t]
        for g, sel in groups.items():
            if len(sel) >= 3:
                fw = _fit([r[f"C_{bc}"] for r in sel], [r[f"w_{bc}"] / fit_N for r in sel])
                fe = _fit([r[f"C_{bc}"] for r in sel], [r[f"eps_{bc}"] / fit_N for r in sel])
                k = np.array([r[f"kappa_{bc}"] for r in sel])
                fits[(bc, g)] = {"w": fw, "eps": fe, "kappa_median": float(np.median(k)), "kappa_min": float(k.min()),
                                 "kappa_max": float(k.max()), "n": len(sel)}
    sel = [r for r in rows if r.get("kappa100") and r["ext"]["alpha"] > 0]
    if len(sel) >= 3:
        k = np.array([r["kappa100"] for r in sel])
        ns = [r for r in sel if not r["ext"]["targets"]["100"]["saturated"]]
        fits[("open", "alpha_O vs c(100)")] = {
            "w": _fit([r["c100"] for r in sel], [r["ext"]["alpha"] for r in sel]),
            "eps": _fit([r["c100"] for r in ns], [r["ext"]["targets"]["100"]["eps_op"] / 100 for r in ns]),
            "kappa_median": float(np.median(k)), "kappa_min": float(k.min()), "kappa_max": float(k.max()),
            "n": len(sel)}
    kN = {}
    for name, c in data.items():
        for (N, bc), r in c["rows"].items():
            if _usable(r):
                kN.setdefault((bc, N), []).append(r["kappa"])
    k_by_N = {k: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}
              for k, v in sorted(kN.items())}
    return {"rows": rows, "fits": fits, "kappa_by_N": k_by_N, "fit_N": fit_N}


def _validation(data: dict) -> dict:
    v = {"results": 0, "saturated": 0, "floor_violations": 0, "max_dense_dw": 0.0, "max_prop_check": 0.0,
         "max_rel_res": 0.0, "recentered": 0, "back_checks": 0, "back_fail": 0, "matvecs": {}, "seconds": {}}
    for c in data.values():
        for (N, bc), r in c["rows"].items():
            v["results"] += 1
            a = r["arc"]
            v["saturated"] += bool(r["saturated"])
            v["floor_violations"] += not r["floor_ok"]
            if "dense_dw" in r:
                v["max_dense_dw"] = max(v["max_dense_dw"], abs(r["dense_dw"]))
            if "check" in r["prop"]:
                v["max_prop_check"] = max(v["max_prop_check"], r["prop"]["check"])
            if not r["saturated"]:
                lam = max(abs(x) for x in a["lambda"])
                v["max_rel_res"] = max(v["max_rel_res"], max(a["res"]) / max(lam, 1e-300))
            v["recentered"] += a["recentered"] > 0
            if a.get("back_check"):
                v["back_checks"] += 1
                v["back_fail"] += not a["back_check"]["ok"]
            v["matvecs"].setdefault(N, []).append(a["matvecs"])
            v["seconds"].setdefault(N, []).append(r["seconds"])
    return v


def _g(x, fmt=".2e"):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def cmd_table(args) -> int:
    import numpy as np
    root = out_root(args, write=False)
    data = collect(root)
    if not data:
        raise SystemExit(f"no results under {root}")
    A = analysis(data, args.fit_N)
    N0 = args.fit_N
    print(f"# Operator norm against per-site infidelity ({len(data)} circuits, results in {root})\n")
    print(f"Fits of log10(w/N) and log10(eps_op/N) against log10 C at N = {N0} (same unitary, same N; saturated "
          f"arcs left out). kappa = (w/N)/sqrt(C).\n")
    print("| boundary | group | n | slope (w/N) | prefactor | max scatter (dex) | slope (eps_op/N) | kappa median "
          "| kappa range |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (bc, g), f in A["fits"].items():
        print(f"| {bc} | {g} | {f['n']} | {f['w']['slope']:.3f} ± {f['w']['se']:.3f} | {f['w']['pref']:.2f} | "
              f"{f['w']['scatter']:.3f} | {f['eps']['slope']:.3f} ± {f['eps']['se']:.3f} | {f['kappa_median']:.2f} | "
              f"{f['kappa_min']:.2f} to {f['kappa_max']:.2f} |")
    print("\nkappa against N (all usable results):\n")
    print("| boundary | N | n | median | range |")
    print("|---|---|---|---|---|")
    for (bc, N), k in A["kappa_by_N"].items():
        print(f"| {bc} | {N} | {k['n']} | {k['median']:.3f} | {k['min']:.3f} to {k['max']:.3f} |")
    print(f"\nPer circuit, N = {N0}; open-chain extrapolation w = alpha N + beta from N = 10 to 16, "
          "eps_op at N = 50 and 100 (estimates; 'sat' where w >= pi), floor sqrt(1 - (1 - c)^100) with c the MPS "
          "value at N = 100.\n")
    print("| h_x | t | source | point | D | c(100) | C ring | C open | w/N ring | w/N open | kappa ring | kappa open | "
          "alpha_O/sqrt(c100) | beta/alpha | eps_op(50) | eps_op(100) | floor(100) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(A["rows"], key=lambda r: (r["field"], r["t"], r["source"], -(r.get("C_open") or 0))):
        ext = r.get("ext") or {}
        tg = ext.get("targets", {})

        def eps(nt):
            x = tg.get(nt)
            return "-" if not x else ("sat" if x["saturated"] else f"{x['eps_op']:.3g}")
        wr = r["w_ring"] / N0 if "w_ring" in r else None
        wo = r["w_open"] / N0 if "w_open" in r else None
        print(f"| {r['field']} | {r['t']} | {r['source']} | {r['stem']} | {r['D']} | {_g(r['c100'])} | "
              f"{_g(r.get('C_ring'))} | {_g(r.get('C_open'))} | {_g(wr)}{' sat' if r.get('sat_ring') else ''} | "
              f"{_g(wo)}{' sat' if r.get('sat_open') else ''} | {_g(r.get('kappa_ring'), '.2f')} | "
              f"{_g(r.get('kappa_open'), '.2f')} | {_g(r.get('kappa100'), '.2f')} | "
              f"{_g(ext.get('beta_sites'), '.2f')} | "
              f"{eps('50')} | {eps('100')} | "
              f"{_g(tg.get('100', {}).get('floor'), '.3g')} |")
    checks = [(r["name"], N, ch) for r in A["rows"] for N, ch in (r.get("ext") or {}).get("checks", {}).items()]
    if checks:
        print("\nExtrapolation check (open chain): w predicted from N = 10 to 16 against w measured.\n")
        print("| circuit | N | w measured | w predicted | relative error |")
        print("|---|---|---|---|---|")
        for name, N, ch in checks:
            print(f"| {name} | {N} | {ch['w']:.4e} | {ch['w_pred']:.4e} | {ch['rel_err']:+.2%} |")
    v = _validation(data)
    print(f"\nValidation over {v['results']} results: {v['saturated']} saturated, {v['floor_violations']} floor "
          f"violations, max |w_Lanczos - w_dense| {v['max_dense_dw']:.1e} (N <= 10), max propagator error "
          f"{v['max_prop_check']:.1e} (N <= 14), max relative residual {v['max_rel_res']:.1e}, {v['recentered']} "
          f"recentered, {v['back_checks']} back-half checks ({v['back_fail']} failed).")
    print("Matvecs and seconds per result, median (range):",
          "; ".join(f"N = {N}: {np.median(m):.0f} ({min(m)} to {max(m)}), {np.median(v['seconds'][N]):.1f} s "
                    f"({min(v['seconds'][N]):.1f} to {max(v['seconds'][N]):.1f})"
                    for N, m in sorted(v["matvecs"].items())))
    return 0


_SNAP_COLOR, _BASE_COLOR, _INK, _MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"      # cli.py cmd_plot
_COLORS = {"snap": _SNAP_COLOR, "baseline": _BASE_COLOR, "trotter": "#8a8986", "open": "#1f9e6e", "lit": "#9a5fc4"}
_LABELS = {"snap": "SNAP", "baseline": "Trotter + KN", "trotter": "Trotter (no synthesis)",
           "open": "SNAP, open-trained", "lit": "literature formulas + KN"}
_MARKERS = {"1.6": "o", "0": "s"}
_FIELD = {"1.6": r"$h_x = 1.6$", "0": r"$h_x = 0$"}
_HAM = (r"XXZ chain, $H = J\sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta\, S^z_i S^z_{i+1})"
        r" + h_x \sum_i S^x_i$, $J = 1$, $\Delta = 0.25$, spin units")


def _style(ax, title=None):
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#c9c8c2")
    if title:
        ax.set_title(title, fontsize=9, color=_INK)


def cmd_plot(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root = out_root(args, write=False)
    data = collect(root)
    if not data:
        raise SystemExit(f"no results under {root}")
    A = analysis(data, args.fit_N)
    N0 = args.fit_N
    fig_dir = Path(args.fig or (root / "figures"))
    if _under_runs(fig_dir) and not args.real:
        raise SystemExit(f"{fig_dir} is under {RUNS}: pass --real to write the figures there, or --fig DIR")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = A["rows"]
    names = []

    # (a) w/N and eps_op/N against C at N0, open chain and ring
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharex=True)
    for ax, (key, ylabel) in zip(axes, (("w", r"arc width per site $w/N$"), ("eps", r"$\epsilon_{\rm op}/N$"))):
        for bc, face in (("open", None), ("ring", "white")):
            for src in _COLORS:
                for f in FIELDS:
                    sel = [r for r in rows if r["source"] == src and r["field"] == f and r.get(f"sat_{bc}") is False]
                    if not sel:
                        continue
                    x = [r[f"C_{bc}"] for r in sel]
                    y = [r[f"{key}_{bc}"] / N0 for r in sel]
                    ax.plot(x, y, ls="none", marker=_MARKERS[f], ms=3.8, color=_COLORS[src],
                            mfc=face or _COLORS[src], mew=0.9, alpha=0.85 if bc == "open" else 0.6,
                            label=f"{_LABELS[src]}, {_FIELD[f]}" if bc == "open" else None)
        fit = A["fits"].get(("open", "all"))
        if fit:
            xs = np.array([r["C_open"] for r in rows if r.get("sat_open") is False])
            xx = np.logspace(math.log10(xs.min()), math.log10(xs.max()), 50)
            ff = fit[key]
            ax.plot(xx, ff["pref"] * xx ** ff["slope"], color=_INK, lw=1.0,
                    label=rf"fit, open: slope ${ff['slope']:.3f} \pm {ff['se']:.3f}$")
            k = fit["kappa_median"] / (1 if key == "w" else 2)
            ax.plot(xx, k * np.sqrt(xx), color=_MUTED, lw=0.8, ls="--",
                    label=(rf"$\kappa\sqrt{{c}}$, $\kappa = {fit['kappa_median']:.2f}$" if key == "w"
                           else r"$(\kappa/2)\sqrt{c}$"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, rf"(a) $N = {N0}$: filled open chain, hollow ring" if key == "w" else None)
        ax.set_ylabel(ylabel, fontsize=8, color=_MUTED)
        ax.set_xlabel(rf"per-site infidelity $C$ (same unitary, $N = {N0}$)", fontsize=8, color=_MUTED)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.suptitle(_HAM, fontsize=7.5, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    names.append(_save(fig, fig_dir / "opnorm_a_vs_c"))

    # (b) kappa against N for the representative circuits (those measured at N >= 16)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    reps = [n for n, c in data.items() if any(N >= 16 for N, _ in c["rows"])]
    for ax, bc in zip(axes, BCS):
        for n in reps:
            c = data[n]
            e = c["entry"]
            pts = sorted((N, r["kappa"]) for (N, b), r in c["rows"].items() if b == bc and _usable(r))
            if len(pts) >= 2:
                ax.plot(*zip(*pts), color=_COLORS[e["source"]], marker=_MARKERS[e["field"]], ms=3, lw=0.8,
                        alpha=0.8, mfc=_COLORS[e["source"]] if e["field"] == "1.6" else "white")
        _style(ax, rf"(b) {'open chain' if bc == 'open' else 'ring'}: $\kappa = (w/N)/\sqrt{{C(N)}}$")
        ax.set_xlabel(r"$N$", fontsize=8, color=_MUTED)
    axes[0].set_ylabel(r"$\kappa$", fontsize=8, color=_MUTED)
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=_COLORS[s], marker="o", lw=0.8, ms=3, label=_LABELS[s])
            for s in _COLORS if any(data[n]["entry"]["source"] == s for n in reps)]
    hand += [Line2D([], [], color=_MUTED, marker=_MARKERS[f], lw=0, ms=3, mfc=_MUTED if f == "1.6" else "white",
                    label=_FIELD[f]) for f in FIELDS]
    axes[1].legend(handles=hand, fontsize=6, frameon=False)
    fig.suptitle(_HAM, fontsize=7.5, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    names.append(_save(fig, fig_dir / "opnorm_b_kappa_vs_N"))

    # (c) eps_op at N = 50 and 100 (extrapolated) against the MPS c at N = 100
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    kfit = A["fits"].get(("open", "alpha_O vs c(100)")) or A["fits"].get(("open", "all")) or {}
    kap = kfit.get("kappa_median", 1.9)
    for ax, Nt in zip(axes, (50, 100)):
        cs = []
        for src in _COLORS:
            for f in FIELDS:
                sel = [r for r in rows if r["source"] == src and r["field"] == f and r.get("ext") and r.get("c100")]
                if not sel:
                    continue
                x = [r["c100"] for r in sel]
                y = [r["ext"]["targets"][str(Nt)]["eps_op"] for r in sel]
                sat = [r["ext"]["targets"][str(Nt)]["saturated"] for r in sel]
                cs += x
                ax.plot([a for a, s in zip(x, sat) if not s], [b for b, s in zip(y, sat) if not s], ls="none",
                        marker=_MARKERS[f], ms=3.8, color=_COLORS[src], mfc=_COLORS[src],
                        label=f"{_LABELS[src]}, {_FIELD[f]}")
                if any(sat):
                    ax.plot([a for a, s in zip(x, sat) if s], [math.sqrt(2)] * sum(sat), ls="none",
                            marker=_MARKERS[f], ms=3.8, color=_COLORS[src], mfc="white")
        if cs:
            from snapevo.opnorm import floor_eps
            xx = np.logspace(math.log10(min(cs)) - 0.3, math.log10(max(cs)) + 0.3, 200)
            ax.plot(xx, [floor_eps(c, Nt) for c in xx], color=_INK, lw=1.0, ls=":",
                    label=r"floor $\sqrt{1-(1-c)^N}$ (rigorous)")
            ax.plot(xx, [2 * math.sin(min(kap * Nt * math.sqrt(c), math.pi) / 4) for c in xx], color=_MUTED,
                    lw=0.8, ls="--",
                    label=rf"$2\sin(\kappa N\sqrt{{c}}/4)$, $\kappa = \alpha_O/\sqrt{{c}}$ median ${kap:.2f}$")
            ax.axhline(math.sqrt(2), color="#c43c39", lw=0.8)
            ax.axvline((math.pi / (kap * Nt)) ** 2, color="#c43c39", lw=0.8, ls="--")
            ax.text(min(cs), math.sqrt(2) * 1.08, r"$w = \pi$: $\epsilon_{\rm op} = \sqrt{2}$, saturated",
                    fontsize=6, color="#c43c39")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax, rf"(c) $N = {Nt}$, open chain, $w = \alpha N + \beta$ (estimate)")
        ax.set_xlabel(r"per-site infidelity $c$ at $N = 100$ (MPS, success branch)", fontsize=8, color=_MUTED)
    axes[0].set_ylabel(r"$\epsilon_{\rm op} = \min_\phi \|U - e^{i\phi}V\|$", fontsize=8, color=_MUTED)
    axes[0].legend(fontsize=5.5, frameon=False, loc="lower right")
    fig.suptitle(_HAM, fontsize=7.5, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    names.append(_save(fig, fig_dir / "opnorm_c_eps_at_N"))
    for n in names:
        print(f"wrote {n}.png and .pdf")
    return 0


def _save(fig, stem: Path) -> Path:
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=160)
    plt.close(fig)
    return stem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="opnorm_run.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default=None, help="results root (default $OPNORM_OUT, else runs/ with --real)")
        p.add_argument("--real", action="store_true", help="allow writing under runs/")
        p.add_argument("--lit-points", default=None, dest="lit_points")
        p.add_argument("--kn-dir", default=None, dest="kn_dir")

    def selection(p):
        p.add_argument("--times", default=",".join(map(str, TIMES)))
        p.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
        p.add_argument("--refresh", action="store_true", help="recompute the task list instead of the frozen one")

    p = sub.add_parser("list")
    common(p)
    selection(p)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("task")
    common(p)
    p.add_argument("name")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--sizes", default=None, help="override the sizes of the name")
    p = sub.add_parser("run")
    common(p)
    selection(p)
    p.add_argument("--cores", type=int, default=None)
    p.add_argument("--cpus", default=None, help="explicit CPU list, e.g. 60-63 (login nodes)")
    p.add_argument("--only", default=None, help="regex on task names")
    for cmd in ("table", "plot"):
        p = sub.add_parser(cmd)
        common(p)
        p.add_argument("--fit-N", type=int, default=14, dest="fit_N")
        if cmd == "plot":
            p.add_argument("--fig", default=None, help="figure directory (default <out>/figures)")
    args = ap.parse_args(argv)
    return {"list": cmd_list, "task": cmd_task, "run": cmd_run, "table": cmd_table, "plot": cmd_plot}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
