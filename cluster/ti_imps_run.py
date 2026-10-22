"""Infinite-chain (translation-invariant) evaluation of every sweep SNAP point and Trotter + KN baseline.

    python cluster/ti_imps_run.py plan  [--times 1,2,4,6,8,10]
    python cluster/ti_imps_run.py run   [--times ..] [--workers 16] [--shard 0/1]
    python cluster/ti_imps_run.py table [--times ..] [--json FILE]
    python cluster/ti_imps_run.py pack  [--times ..]        # runs/ti/imps_Lu8_n32/<tag>_t<t>.json, for the repo

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0.

What. c_inf = 1 - |lambda0|^{2/L_u} per point and input, from the mixed transfer matrix of the circuit's and
the reference's infinite MPS (snapevo.imps; Gibbs and Cincio, arXiv:2506.02105). The inputs are L_u = 8 site
unit cells of random product states, circuits.product_vectors(8, 32, 6161 + 8), repeated along the infinite
chain; the reference e^{-iHt} is mps.reference_circuit (order-4 Suzuki, step 0.1) run by iTEBD at cutoff 1e-14,
bond dimension at most 2048, the same for every t. Points are the ones of ti_ring_eval.py, methods snap (the
sweep's ring-trained cells, one training recipe for all t) and baseline; the open-chain arm is left out.

Work. A unit is one (field, t, input k): its reference first (cached in <out>/imps_ref/ with the name
ti_ring_eval.py imps uses, so the two share it), then every point on that input. Each (point, k) is one JSON,
<out>/imps_pts/<tag>/t<t>/<method>_<stem>/k<k>.json, written atomically, so runs resume and --shard i/n splits
the units across nodes (round robin over the units in order of decreasing t, so every shard gets its share of
t = 10). Workers are single-threaded processes; a reference's points are queued as soon as it is done.

table: per point the mean c_inf over the inputs done, its standard error, and the 100-site open-chain c for
comparison.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy loads
    os.environ.setdefault(_v, "1")

import argparse                                                             # noqa: E402
import math                                                                 # noqa: E402
import socket                                                               # noqa: E402
import sys                                                                  # noqa: E402
import time                                                                 # noqa: E402
from pathlib import Path                                                    # noqa: E402

import numpy as np                                                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ti_ring_eval as tr                                                   # noqa: E402

L_U = 8
N_IN = 32
SEED = tr.IMPS_SEED0 + L_U
DT = 0.1
CUTOFF = 1e-14
CHI_MAX = 2048
METHODS = ("snap", "baseline")
VERSION = 1


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ref_path(root: Path, field: str, t, k: int) -> Path:
    return (Path(root) / "imps_ref" /
            f"{tr.TAGS[field]}_t{float(t):g}_Lu{L_U}_s{SEED}_k{k}_dt{DT:g}_c{CUTOFF:.0e}_x{CHI_MAX}.npz")


def row_path(root: Path, p: dict, k: int) -> Path:
    return (Path(root) / "imps_pts" / tr.TAGS[p["field"]] / f"t{p['t']:g}" / f"{p['method']}_{p['stem']}" /
            f"k{k}.json")


def units(times) -> list[tuple[str, int, int]]:
    return [(f, t, k) for t in sorted(times, reverse=True) for f in tr.FIELDS for k in range(N_IN)]


def _inputs():
    from snapevo import imps as im
    return im.unit_cell_inputs(L_U, N_IN, SEED)


def _ref_task(args) -> dict:
    field, t, k, root = args
    from snapevo import imps as im
    path = ref_path(root, field, t, k)
    t0 = time.time()
    ref, meta = im.reference_state(tr._model(field), float(t), _inputs()[k], DT, CHI_MAX, CUTOFF, path)
    return {"field": field, "t": t, "k": k, "seconds": time.time() - t0, "chi": ref.chi_peak}


def _point_task(args) -> dict:
    p, k, root = args
    from snapevo import imps as im
    path = row_path(root, p, k)
    if path.exists():
        return {"name": tr.pname(p), "k": k, "seconds": 0.0, "skipped": True}
    ref, _ = im.IMPS.load(ref_path(root, p["field"], p["t"], k))
    t0 = time.time()
    r = im.c_inf(tr.circuit(p), tr._model(p["field"]), float(p["t"]), _inputs()[k], DT, CHI_MAX, CUTOFF, ref=ref)
    row = {"version": VERSION, "point": p, "k": k, "L_u": L_U, "seed": SEED, "n_inputs": N_IN, "dt": DT,
           "cutoff": CUTOFF, "chi_max": CHI_MAX, "c_inf": r["c_inf"], "residual": r["residual"], "gap": r["gap"],
           "dim": r["dim"], "method": r["method"], "chi_u": r["chi_u"], "chi_v": r["chi_v"],
           "discarded_u": r["discarded_u"], "discarded_v": r["discarded_v"],
           "lambda_aa": abs(r["lambda_aa"]), "lambda_bb": abs(r["lambda_bb"]),
           "seconds_circuit": r["seconds_circuit"], "seconds_fid": r["seconds"], "seconds": time.time() - t0,
           "host": socket.gethostname(), "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    tr._save_json(path, row)
    return {"name": tr.pname(p), "k": k, "seconds": row["seconds"], "c_inf": row["c_inf"], "skipped": False}


def _points(times) -> dict:
    by = {}
    for p in tr.points(tuple(times), METHODS):
        by.setdefault((p["field"], p["t"]), []).append(p)
    for v in by.values():                                   # deepest first: the long ones start early
        v.sort(key=lambda p: -p["D"])
    return by


def _shard(arg: str) -> tuple[int, int]:
    i, n = (int(v) for v in arg.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad shard {arg!r}")
    return i, n


def cmd_plan(args) -> int:
    times = [int(v) for v in args.times.split(",")]
    by = _points(times)
    root = tr.out_root(args.out)
    for t in times:
        for f in tr.FIELDS:
            pts = by.get((f, t), [])
            todo = sum(not row_path(root, p, k).exists() for p in pts for k in range(N_IN))
            refs = sum(ref_path(root, f, t, k).exists() for k in range(N_IN))
            print(f"h_x = {f:>3}, t = {t:2d}: {len(pts):3d} points (D up to {max((p['D'] for p in pts), default=0)}),"
                  f" {refs}/{N_IN} references, {todo} of {len(pts) * N_IN} point-inputs to do")
    return 0


def cmd_run(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    import multiprocessing as mp
    root = tr.out_root(args.out)
    times = [int(v) for v in args.times.split(",")]
    i, n = _shard(args.shard)
    mine = [u for j, u in enumerate(units(times)) if j % n == i]
    by = _points(times)
    _log(f"ti_imps_run on {socket.gethostname()}, shard {i}/{n}: {len(mine)} units (field, t, input) at t = {times},"
         f" L_u = {L_U}, {N_IN} inputs, dt {DT}, cutoff {CUTOFF:.0e}, chi_max {CHI_MAX}, {args.workers} workers")
    pend, done, t0 = set(), 0, time.time()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        def queue_points(field, t, k):
            for p in by.get((field, t), []):
                if not row_path(root, p, k).exists():
                    fu = ex.submit(_point_task, (p, k, root))
                    fu.kind = "point"
                    pend.add(fu)
        for field, t, k in mine:
            if ref_path(root, field, t, k).exists():
                queue_points(field, t, k)
            else:
                fu = ex.submit(_ref_task, (field, t, k, root))
                fu.kind = "ref"
                pend.add(fu)
        while pend:
            fin, _ = wait(pend, return_when=FIRST_COMPLETED)
            for fu in fin:
                pend.discard(fu)
                try:
                    r = fu.result()
                except Exception as e:                       # noqa: BLE001 (logged, the rest goes on)
                    _log(f"FAILED {fu.kind}: {e!r}")
                    continue
                if fu.kind == "ref":
                    _log(f"reference h_x = {r['field']}, t = {r['t']}, k = {r['k']}: chi {r['chi']}, "
                         f"{r['seconds']:.0f} s")
                    queue_points(r["field"], r["t"], r["k"])
                else:
                    done += 1
                    if not r["skipped"] and (done % args.every == 0 or r["seconds"] > 600):
                        _log(f"{done} done ({len(pend)} queued, {time.time() - t0:.0f} s): {r['name']} k {r['k']} "
                             f"c_inf {r['c_inf']:.3e}, {r['seconds']:.0f} s")
    _log(f"shard {i}/{n} finished: {done} point-inputs in {time.time() - t0:.0f} s")
    return 0


def collect(root: Path, times) -> list[dict]:
    out = []
    by = _points(times)
    for (f, t), pts in sorted(by.items(), key=lambda kv: (-float(kv[0][0]), kv[0][1])):
        for p in pts:
            rows = [tr._read(row_path(root, p, k)) for k in range(N_IN)]
            rows = [r for r in rows if r]
            if not rows:
                continue
            c = np.array([r["c_inf"] for r in rows])
            o = tr._open100(tr._read(Path(p["eval_file"]))) if p.get("eval_file") else None
            out.append({"name": tr.pname(p), "field": f, "t": t, "method": p["method"], "stem": p["stem"],
                        "D": p["D"], "n": len(rows), "c_inf": float(c.mean()),
                        "c_inf_se": float(c.std(ddof=1) / math.sqrt(len(c))) if len(c) > 1 else None,
                        "c_inf_k": c.tolist(), "c100": o["c_site"] if o else None,
                        "chi_peak": max(max(r["chi_u"], r["chi_v"]) for r in rows),
                        "discarded_max": max(max(r["discarded_u"], r["discarded_v"]) for r in rows)})
    return out


def cmd_table(args) -> int:
    root = tr.out_root(args.out)
    times = [int(v) for v in args.times.split(",")]
    R = collect(root, times)
    print("| h_x | t | method | point | D | inputs | c_inf | se | c(100, open) | c100/c_inf - 1 | chi peak |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(R, key=lambda r: (-float(r["field"]), r["t"], r["method"], -r["c_inf"])):
        se = f"{r['c_inf_se'] / r['c_inf']:.1%}" if r["c_inf_se"] else "-"
        c100 = f"{r['c100']:.2e}" if r["c100"] else "-"
        rel = f"{r['c100'] / r['c_inf'] - 1:+.1%}" if r["c100"] else "-"
        print(f"| {r['field']} | {r['t']} | {r['method']} | {r['stem']} | {r['D']} | {r['n']} | {r['c_inf']:.3e} | "
              f"{se} | {c100} | {rel} | {r['chi_peak']} |")
    if args.json:
        tr._save_json(Path(args.json), {"L_u": L_U, "n_inputs": N_IN, "seed": SEED, "dt": DT, "cutoff": CUTOFF,
                                        "chi_max": CHI_MAX, "rows": R})
        print(f"\nwrote {args.json}")
    return 0


def cmd_pack(args) -> int:
    """One file per (field, t) under runs/ti/ for the repo: every point's per-input c_inf and bookkeeping."""
    root = tr.out_root(args.out)
    times = [int(v) for v in args.times.split(",")]
    dest = tr.RUNS / "ti" / f"imps_Lu{L_U}_n{N_IN}"
    for t in times:
        for f in tr.FIELDS:
            R = [r for r in collect(root, [t]) if r["field"] == f]
            if not R:
                continue
            path = dest / f"{tr.TAGS[f]}_t{t}.json"
            tr._save_json(path, {"version": VERSION, "field": f, "t": t, "L_u": L_U, "n_inputs": N_IN, "seed": SEED,
                                 "inputs": f"circuits.product_vectors({L_U}, {N_IN}, {SEED})", "dt": DT,
                                 "cutoff": CUTOFF, "chi_max": CHI_MAX, "complete": all(r["n"] == N_IN for r in R),
                                 "points": R})
            print(f"wrote {path.relative_to(tr.REPO)}: {len(R)} points, "
                  f"{sum(r['n'] == N_IN for r in R)} with all {N_IN} inputs")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "table", "pack"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=None, help="results root (default $TI_OUT, else $SCRATCH/snapevo-scratch/ti)")
        p.add_argument("--times", default=",".join(map(str, tr.TIMES)))
        if name == "run":
            p.add_argument("--workers", type=int, default=16)
            p.add_argument("--shard", default="0/1", help="i/n: this process's share of the (field, t, input) units")
            p.add_argument("--every", type=int, default=50, help="log every this many point-inputs")
        if name == "table":
            p.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "table": cmd_table, "pack": cmd_pack}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
