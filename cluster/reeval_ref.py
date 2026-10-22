"""Re-evaluate the tight points against finer N = 100 references (step 0.1 instead of 0.2).

The reference e^{-iHt}|phi_k> is the order-4 brickwork Trotter formula at step dt (mps.reference_circuit),
the same family as the Trotter + KN baseline. At h_x = 1.6 its own per-site error (2.6e-11 at t = 2, 5.1e-11
at t = 4) lines up with a baseline circuit's error and partly cancels it: against step-0.05 references the
baseline's c rises 31% at 9e-10 and 80% at 2.4e-10 (t = 4), 5-8% near 1e-8, while SNAP moves -0.2% to +2.7%
(independent check, 17 Oct 2026, snapevo-scratch/litbase/verify_2026-10-17/REPORTS.md). Step 0.1 makes the
reference error 2^8 = 256x smaller in infidelity. Sohan, 17 Oct: re-evaluate.

    python cluster/reeval_ref.py list [--cmax 1e-6] [--fields 1.6]   points to redo and the references they need
    python cluster/reeval_ref.py refs  [--cmax ..] [--fields ..]      one `snapevo reference` command per missing ref
    python cluster/reeval_ref.py files [--cmax ..] [--fields ..]     the files still to redo, one per line
    python cluster/reeval_ref.py point FILE                           re-evaluate one eval_/baseline_ file (+ draws)
All listing commands take --times T [T ...] to restrict the evolution times.

`point` keeps the old file as cluster/logs/pool/backup/reeval_dt0.2/<tag>/t<T>/<name> (once), then writes the
new evaluation in place at step 0.1, with the same inputs, cutoff and chi_max, and redraws the KN fallbacks
when the old record had them (same number of draws). Records carry mps.dt = 0.1, so a rerun skips them.
"""

import argparse
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
BACKUP = ROOT / "cluster" / "logs" / "pool" / "backup" / "reeval_dt0.2"
DT_NEW = 0.1
LETTER = {1.6: "A", 0.0: "B"}


def _points(cmax: float, fields, times=None) -> list[dict]:
    out = []
    for f in fields:
        tag = f"xxz_d0.25_hx{f:g}"
        for p in sorted(RUNS.glob(f"{tag}/t*/eval_b*.json")) + sorted(RUNS.glob(f"{tag}/t*/baseline_o*.json")):
            d = json.loads(p.read_text())
            if "mps" not in d or d.get("c_site") is None or d["c_site"] > cmax:
                continue
            if times and float(d["t"]) not in times:
                continue
            m = d["mps"]
            out.append({"file": p, "field": f, "t": float(d["t"]), "kind": "snap" if p.name.startswith("eval_") else
                        "baseline", "c_site": d["c_site"], "cutoff": m["cutoff"], "chi_max": m["chi_max"],
                        "states": tuple(m["states"]), "dt": m["dt"], "sampled": "sampled" in d,
                        "samples": (d.get("sampled") or {}).get("samples")})
    return out


def _refs_needed(points) -> list[tuple]:
    """(field, t, seed, k, cutoff, chi_max) for every reference the re-evaluations read at step 0.1."""
    from snapevo.evaluate import ref_path
    need = set()
    for p in points:
        if p["dt"] == DT_NEW:
            continue
        seed, n = p["states"]
        for k in range(n):
            need.add((p["field"], p["t"], seed, k, p["cutoff"], p["chi_max"]))
        need.add((p["field"], p["t"], seed, 0, p["cutoff"] / 100, p["chi_max"]))     # the cutoff/100 check
    missing = []
    for f, t, seed, k, c, x in sorted(need):
        d = RUNS / f"xxz_d0.25_hx{f:g}" / f"t{t:g}" / "ref"
        if not ref_path(d, 100, seed, k, DT_NEW, c, x).exists():
            missing.append((f, t, seed, k, c, x))
    return missing


def cmd_list(a):
    pts = _points(a.cmax, a.fields, a.times)
    todo = [p for p in pts if p["dt"] != DT_NEW]
    for p in todo:
        print(f"{p['file'].relative_to(RUNS)}  c {p['c_site']:.2e}  cutoff {p['cutoff']:.0e} chi {p['chi_max']}"
              + ("  +draws" if p["sampled"] else ""))
    miss = _refs_needed(pts)
    print(f"{len(todo)} points to re-evaluate ({len(pts) - len(todo)} already at dt {DT_NEW}); "
          f"{len(miss)} references missing")


def cmd_refs(a):
    for f, t, seed, k, c, x in _refs_needed(_points(a.cmax, a.fields, a.times)):
        print(f"python -m snapevo reference --model {LETTER[f]} --t {t:g} --dt {DT_NEW} --cutoff {c:.0e} "
              f"--chi-max {x} --seed {seed} --only {k}")


def cmd_files(a):
    for p in _points(a.cmax, a.fields, a.times):
        if p["dt"] != DT_NEW:
            print(p["file"])


def _args(model_letter, t, d, **kw):
    base = dict(model=model_letter, t=t, cell=None, N=100, states=d["mps"]["states"][1], seed=d["mps"]["states"][0],
                chi_max=d["mps"]["chi_max"], cutoff=d["mps"]["cutoff"], dt=DT_NEW, no_cache=False, no_check=False,
                f_tol=0.01, fresh=True)
    base.update(kw)
    return Namespace(**base)


def _sample_snap_file(p: Path, letter: str, t: float, samples: int) -> None:
    """cli.cmd_sample's SNAP branch for one eval_ file, at step 0.1, with its own inputs, cutoff and chi_max."""
    from snapevo import cli, io
    from snapevo.fallbacks import sample_snap, snap_total
    model, cell, folder = cli._ctx(Namespace(model=letter, t=t, cell=None))
    e = io.read_json(p)
    m = e["mps"]
    kw = dict(N=100, states=tuple(m["states"]), samples=samples, dt=DT_NEW, chi_max=m["chi_max"], cutoff=m["cutoff"],
              cache_dir=folder / "ref", log=cli._log)
    real = io.read_json(folder / p.name.replace("eval_", "real_"))
    edges = real.get("edges") or {}
    n_kn = len(real["kn"]) + (len(edges.get("kn", {})) if edges.get("use") else 0)
    if e["infid"] > 0.999 or n_kn == 0:
        sm = {"infid": e["infid"], "drawn": False, "n_kn": n_kn}
    else:
        sm = {**sample_snap(real, cell, model, t, **kw), "drawn": True, "n_kn": n_kn}
    e["sampled"] = sm
    e["infid_sampled_kn"] = sm["infid"]
    e["infid_sampled"] = snap_total(real, str(e["F"]), sm["infid"], e["N"])
    io.write_json(p, e)


def cmd_point(a):
    from snapevo import cli, io
    p = Path(a.file).resolve()
    d = json.loads(p.read_text())
    if d.get("mps", {}).get("dt") == DT_NEW:
        print(f"{p.name}: already at dt {DT_NEW}")
        return
    field = float(d["model"]["hx"])
    letter, t = LETTER[field], float(d["t"])
    bk = BACKUP / p.parent.parent.name / p.parent.name / p.name
    if not bk.exists():
        bk.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, bk)
    old_c, samples = d["c_site"], (d.get("sampled") or {}).get("samples")
    if p.name.startswith("eval_"):
        args = _args(letter, t, d, budget=float(d["budget"]), D=int(d["D"]))
        with io.claim(p) as mine:                     # one process per file (cmd_sample's lock)
            if not mine:
                print(f"{p.name}: another process holds it, skipping")
                return
            cli.cmd_evaluate(args, int(d["D"]))
            if samples:                               # this file only: cmd_sample would redraw every depth of
                _sample_snap_file(p, letter, t, samples)   # the budget and could write a stale record back
    else:
        model, _cell, folder = cli._ctx(Namespace(model=letter, t=t, cell=None))
        args = _args(letter, t, d)
        cli._baseline_point(args, model, folder, int(d["order"]), int(d["steps"]), float(d["eps"]))
        if samples:
            from snapevo.baseline import realize_baseline, trotter_circuit
            from snapevo.fallbacks import sample_baseline
            b = io.read_json(p)
            circ = trotter_circuit(model, t, b["order"], b["steps"])
            rb = realize_baseline(circ, b["eps"])
            sm = sample_baseline(circ, rb["circuit"], rb["synth"], b["eps"], model, t, N=100,
                                 states=tuple(b["mps"]["states"]), samples=samples, dt=DT_NEW,
                                 chi_max=b["mps"]["chi_max"], cutoff=b["mps"]["cutoff"], cache_dir=folder / "ref",
                                 log=cli._log)
            b["sampled"], b["infid_sampled"] = sm, sm["infid"]
            io.write_json(p, b)
    new = json.loads(p.read_text())
    print(f"{p.relative_to(RUNS)}: c {old_c:.4e} (dt 0.2) -> {new['c_site']:.4e} (dt {DT_NEW}), "
          f"change {new['c_site'] / old_c - 1:+.1%}" + (f", drawn {new.get('infid_sampled', float('nan')):.4e}"
                                                         if samples else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "refs", "files"):
        s = sub.add_parser(name)
        s.add_argument("--cmax", type=float, default=1e-6)
        s.add_argument("--fields", type=float, nargs="+", default=[1.6])
        s.add_argument("--times", type=float, nargs="+", default=None)
    s = sub.add_parser("point")
    s.add_argument("file")
    a = ap.parse_args()
    {"list": cmd_list, "refs": cmd_refs, "files": cmd_files, "point": cmd_point}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
