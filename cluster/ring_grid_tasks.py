"""Task lists for cluster/ring_grid.slurm: the production ring grid from the ring2 ladders, and its references.

    python cluster/ring_grid_tasks.py points [--fields 1.6,0] [--times 1,2,4,6,8,10] [--out FILE]
    python cluster/ring_grid_tasks.py refs   [--fields 1.6,0] [--times 1,2,4,6,8,10] [--out FILE]
    python cluster/ring_grid_tasks.py size   [--fields ..] [--times ..]      the point count and a rough cost
    python cluster/ring_grid_tasks.py chain-points [--fields ..] [--times ..] [--chain-out newton] [--per-node 10]

chain-points: the open-chain arm's grid for cluster/chain_snap.slurm, from runs/<tag>/t<T>/chain/ladder.json:
"FIELD:T:BUDGET:D" for every budget and every ladder depth whose floor max(c_train, c_heldout) is at most
BUDGET/1.5 without an eval_ record in chain/<chain-out>/, longest first, --per-node entries per output line
(one line = the POINTS of one chain_snap.slurm job).

points: one line "point FIELD T BUDGET D" for every budget of the production grid (cluster/make_jobs.py's 15
BUDGETS and the EXTRA_BUDGETS 3e-9 and 1e-9) and every depth of runs/<tag>/t<T>/ring2/ladder.json whose floor
max(C_train, C_val) is at most BUDGET/1.5 (chain_snap.ring_point's all_depths rule, one depth per task so they
run in parallel), skipping those whose evaluation record exists in runs/<tag>/t<T>/ring2/protocol/. Lines are
sorted by a rough cost (D x the budget's decades), longest first, so a pool packs them well.

refs: one line "ref FIELD T DT CUTOFF CHI K" for every N = 100 reference the evaluations of those budgets read
(cluster/chain_snap.py's eval_settings: cutoff 1e-12/chi 1024 for all inputs and input 0 at 1e-14 for the
truncation check; for the budgets 3e-9 and 1e-9, cutoff 1e-14 with chi 1024 below t = 6 or chi 2048 from t = 6
and for make_jobs' REEVAL points, and input 0 at 1e-16; step 0.1 at h_x = 1.6, 0.2 at h_x = 0) that is not in
runs/<tag>/t<T>/ref yet. Run the refs list before (or while) the ladders train; the points need the ladders.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
OUT_SUB = "ring2/protocol"
LADDER = "ring2/ladder.json"
STATES = (2026, 8)                       # evaluate.MPS_SET


def _jobs():
    spec = importlib.util.spec_from_file_location("_make_jobs", REPO / "cluster" / "make_jobs.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    return mj


def _snap_runner():
    spec = importlib.util.spec_from_file_location("_chain_snap_runner", REPO / "cluster" / "chain_snap.py")
    cs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cs)
    return cs


def budgets() -> list[str]:
    mj = _jobs()
    return list(mj.BUDGETS) + list(mj.EXTRA_BUDGETS)


def tag_of(field: float) -> str:
    return f"xxz_d0.25_hx{field:g}"


def point_tasks(fields, times) -> list[tuple]:
    out = []
    for f in fields:
        for t in times:
            lad = RUNS / tag_of(f) / f"t{t:g}" / LADDER
            if not lad.exists():
                print(f"# no ladder {lad}", file=sys.stderr)
                continue
            recs = json.loads(lad.read_text())["results"]
            for b in budgets():
                bv = float(b)
                for r in recs:
                    floor = max(r["C_train"], r.get("C_val", 0.0))
                    if floor * 1.5 > bv:
                        continue
                    ev = RUNS / tag_of(f) / f"t{t:g}" / OUT_SUB / f"eval_b{bv:.1e}_D{int(r['D'])}.json"
                    if ev.exists():
                        continue
                    cost = int(r["D"]) * max(1.0, -math.log10(bv))
                    out.append((cost, f"point {f:g} {t:g} {b} {int(r['D'])}"))
    out.sort(key=lambda x: -x[0])
    return out


def chain_point_entries(fields, times, chain_out: str) -> list[str]:
    out = []
    for f in fields:
        for t in times:
            lad = RUNS / tag_of(f) / f"t{t:g}" / "chain" / "ladder.json"
            if not lad.exists():
                print(f"# no ladder {lad}", file=sys.stderr)
                continue
            recs = json.loads(lad.read_text())["results"]
            for b in budgets():
                bv = float(b)
                for r in recs:
                    floor = max(float(r["c_train"]), float(r.get("c_heldout") or 0.0))
                    if floor * 1.5 > bv:
                        continue
                    ev = RUNS / tag_of(f) / f"t{t:g}" / "chain" / chain_out / f"eval_b{bv:.1e}_D{int(r['D'])}.json"
                    if ev.exists():
                        continue
                    out.append((int(r["D"]) * max(1.0, -math.log10(bv)), f"{f:g}:{t:g}:{b}:{int(r['D'])}"))
    out.sort(key=lambda x: -x[0])
    return [e for _c, e in out]


def ref_tasks(fields, times) -> list[str]:
    from snapevo.evaluate import ref_path
    cs = _snap_runner()
    need = []
    for f in fields:
        for t in times:
            sets = {}
            for b in budgets():
                ev = cs.eval_settings(float(f), float(t), float(b))
                sets[(ev["dt"], ev["cutoff"], ev["chi_max"])] = True
            d = RUNS / tag_of(f) / f"t{t:g}" / "ref"
            for dt, c, chi in sets:
                wanted = [(c, k) for k in range(STATES[1])] + [(c / 100, 0)]
                for cc, k in wanted:
                    p = ref_path(d, 100, STATES[0], k, dt, cc, chi)
                    if not Path(p).exists():
                        need.append(f"ref {f:g} {t:g} {dt:g} {cc:.0e} {chi} {k}")
    # longest first: later times, larger chi, smaller cutoffs
    return sorted(set(need), key=lambda s: (-float(s.split()[2]), -int(s.split()[5]), float(s.split()[4]), s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("points", "refs", "size", "chain-points"))
    ap.add_argument("--chain-out", default="newton", dest="chain_out")
    ap.add_argument("--per-node", type=int, default=10, dest="per_node")
    ap.add_argument("--fields", default="1.6,0")
    ap.add_argument("--times", default="1,2,4,6,8,10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fields = [float(x) for x in a.fields.split(",")]
    times = [float(x) for x in a.times.split(",")]
    if a.what == "refs":
        lines = ref_tasks(fields, times)
    elif a.what == "chain-points":
        ents = chain_point_entries(fields, times, a.chain_out)
        lines = [" ".join(ents[i: i + a.per_node]) for i in range(0, len(ents), a.per_node)]
    else:
        pts = point_tasks(fields, times)
        if a.what == "size":
            by = {}
            for _c, line in pts:
                _p, f, t, _b, _D = line.split()
                by.setdefault((f, t), 0)
                by[(f, t)] += 1
            for (f, t), n in sorted(by.items(), key=lambda kv: (float(kv[0][1]), kv[0][0])):
                print(f"h_x = {f}, t = {t}: {n} compiles")
            print(f"total {len(pts)} compiles")
            return 0
        lines = [line for _c, line in pts]
    text = "\n".join(lines) + ("\n" if lines else "")
    if a.out:
        Path(a.out).write_text(text)
        print(f"{len(lines)} tasks -> {a.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
