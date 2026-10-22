"""Compile, realize and evaluate one SNAP point with compile_chain's protocol, on the open chain or the ring.

    python cluster/chain_snap.py FIELD T BUDGET --reopt newton|lbfgs-converged|coordinate [--arm chain|ring]
                                 [--D D | --margins 3,30] [--no-polish] [--replicas 1] [--runs DIR]
                                 [--steps compile,realize,evaluate] [--fresh] [--deadline EPOCH]

--reopt is required (there is no default re-optimizer; snapevo.chain_snap.ChainSettings): "newton", trust-region
Newton with the round's Hessian; "lbfgs-converged", L-BFGS-B to a budget-relative convergence test; "coordinate",
sweeps setting each free angle to the global minimum of its exact single-angle curve. None has an iteration
cap and none falls back to another.

--arm ring recompiles the ring arm with the same protocol code (snapevo.chain_snap.ring_point): the cell from
runs/<tag>/t<T>/cells.json, compile_chain on the 10-site ring (RingProblem), production realize, the edges
fitted by --edge-fit (required: "lbfgs" is production's edges.fit_edges, "lbfgs-uncapped" the same without
its iteration cap, "newton" edges_newton.fit_edges_newton, "auto" newton up to t = 2 and lbfgs above),
the same evaluation, records in runs/<tag>/t<T>/<--ring-out>/ (the sweep's files are not touched). The rest of
this note is about the default, --arm chain.

FIELD is 1.6 or 0 (XXZ chain, Delta = 0.25, spin units, h_x = FIELD). No ring is built or read at any step:
the cell is runs/<tag>/t<T>/chain/ladder.json (cluster/chain_train.py, N = 100), the compile's costs are on the
100-site open chain over the ladder's 12 training inputs (references from $SCRATCH/snapevo-scratch/open_train/
refs), the straggler check is on that chain, the edge angles are the compiled ones, simplified under the budget
and synthesized with KN, and the evaluation is the sweep's 8 inputs at N = 100 against runs/<tag>/t<T>/ref.
Without --D, each floor margin picks the shallowest depth whose ladder floor max(c_train, c_heldout) is at most
BUDGET/margin (cli point --margins). Before compiling, the depth's z is polished with trust-region Newton
steps on the finite-difference Hessian until the predicted decrease and the last step's are below 1e-6 x
BUDGET with a positive bulk Hessian, or the noise floor (cached in runs/<tag>/t<T>/chain/polish_D<D>.json and
reused by every budget it is tight enough for; --no-polish skips it). Writes snap_/real_/eval_b<budget>_D<D>.json
in runs/<tag>/t<T>/chain/; a killed compile resumes exactly from its checkpoint there (snap_...ckpt.pkl).

Stopping: at --deadline, or on SIGTERM, the polish or the compile stops at its next cost evaluation, writes its
checkpoint and the runner exits with code 3 (requeue it; cluster/chain_snap.slurm does).

Workers: 12 input groups (one training input each) x --replicas single-process workers. Re-optimizations, more
than 90% of a compile, are sequential and use only the 12 groups, so --replicas 1 and about 10 compiles per
128-core node give the most compiles per node-hour; replicas only shorten the Hessian and curve stages.
--trial-workers runs that many trials of a round at once (default 1: in the pilots nearly every round
accepted its first trial). --env-layers N builds the curve caches N sublayers at a time (same curves): at
t = 6 all sublayers of a D = 25 cell need about 3 GB per worker, so use e.g. --env-layers 4 there.
Evaluation settings follow cluster/make_jobs.py: cutoff 1e-12 and chi 1024,
cutoff 1e-14 for the budgets 3e-9 and 1e-9 (and chi 2048 for those at t >= 6 and for the REEVAL points
there); --cutoff/--chi-max override. The reference step is 0.1 at h_x = 1.6 and 0.2 at h_x = 0 (EVAL_DT;
--eval-dt overrides); the references must be in runs/<tag>/t<T>/ref (cluster/ring_grid_tasks.py refs lists
the missing ones), or the evaluation builds them itself.

Production ring arm (Sohan, 18 Oct): train runs/<tag>/t<T>/ring2/ladder.json with cluster/ring_train.slurm,
then compile every depth of that ladder whose floor is at most BUDGET/1.5, e.g.
    python cluster/chain_snap.py 1.6 1 1e-4 --arm ring --reopt newton --ring-cells ring2/ladder.json \
        --ring-out ring2/protocol --edge-fit lbfgs-uncapped --all-depths
(cluster/ring_grid.slurm runs the whole grid, one depth per task, from cluster/ring_grid_tasks.py's list).
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF_CACHE = Path(os.environ.get("SCRATCH", "/pscratch/sd/s/sohan100")) / "snapevo-scratch" / "open_train" / "refs"
EXTRA_BUDGETS = (3e-9, 1e-9)                     # make_jobs.EXTRA_BUDGETS
REEVAL = {(1.6, 6, 1e-8), (1.6, 8, 3e-8), (1.6, 8, 1e-8), (1.6, 10, 3e-8), (1.6, 10, 1e-8),
          (0.0, 8, 3e-8), (0.0, 8, 1e-8), (0.0, 10, 1e-8)}          # make_jobs.REEVAL, by field


EVAL_DT = {1.6: 0.1}
"""Reference step of new evaluations by field (default 0.2): 0.1 at h_x = 1.6, where the step-0.2 order-4
reference has a per-site error of ~1.4e-11 that biases Trotter-like circuits (cluster/reeval_ref.py); at
h_x = 0 that error is ~2e-13 (open chain, N = 12, t = 1 to 4), so 0.2 stays (--eval-dt overrides)."""


def eval_settings(field: float, t: float, budget: float) -> dict:
    """cutoff, chi_max and reference step of this point's evaluation (cluster/make_jobs.py's rules, EVAL_DT)."""
    def near(a, b):
        return abs(a - b) <= 1e-6 * b
    extra = any(near(budget, b) for b in EXTRA_BUDGETS)
    reeval = any(f == field and tt == t and near(budget, b) for f, tt, b in REEVAL)
    dt = EVAL_DT.get(field, 0.2)
    if (extra and t >= 6) or reeval:
        return {"cutoff": 1e-14, "chi_max": 2048, "dt": dt}
    if extra:
        return {"cutoff": 1e-14, "chi_max": 1024, "dt": dt}
    return {"cutoff": 1e-12, "chi_max": 1024, "dt": dt}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", type=float)
    ap.add_argument("t", type=float)
    ap.add_argument("budget", type=float)
    ap.add_argument("--arm", default="chain", choices=("chain", "ring"))
    ap.add_argument("--reopt", required=True, choices=("newton", "lbfgs-converged", "coordinate"),
                    help="re-optimizer (required; no default)")
    ap.add_argument("--top-k", type=int, default=6, dest="top_k")
    ap.add_argument("--relax-ladder", default="coarse", choices=("coarse", "gentle"), dest="relax_ladder")
    ap.add_argument("--ring-out", default="ring2", dest="ring_out",
                    help="ring: records go to runs/<tag>/t<T>/<this>/ (one folder per variant)")
    ap.add_argument("--edge-fit", default=None, choices=("newton", "lbfgs", "lbfgs-uncapped", "auto"),
                    dest="edge_fit",
                    help="ring (required there): edges_newton.fit_edges_newton, edges.fit_edges (production's), "
                         "edges_newton.fit_edges_lbfgs (production's without its iteration cap; the lowest T), or "
                         "auto (newton up to N_fit = 12, i.e. t <= 2; lbfgs above)")
    ap.add_argument("--all-depths", action="store_true", dest="all_depths",
                    help="compile every ladder depth whose floor is at most BUDGET/1.5 (both arms)")
    ap.add_argument("--ring-cells", default="cells.json", dest="ring_cells",
                    help="ring: the ladder to compile from, relative to runs/<tag>/t<T>/ (e.g. ring2/ladder.json)")
    ap.add_argument("--chain-out", default=None, dest="chain_out",
                    help="chain: records in runs/<tag>/t<T>/chain/<this>/ (e.g. one folder per re-optimizer; "
                         "default chain/ itself)")
    ap.add_argument("--edge-zero-t", action="store_true", dest="edge_zero_t",
                    help="chain: let the zero-T stage set edge angles to cost-0 points (exact c, no re-optimization)")
    ap.add_argument("--no-edge-stragglers", action="store_true", dest="no_edge_stragglers",
                    help="chain: leave the edges out of the end game (edge weight 0)")
    ap.add_argument("--D", type=int, default=None, help="depth (default: picked by --margins from the ladder)")
    ap.add_argument("--margins", default="3,30")
    ap.add_argument("--no-polish", action="store_true", dest="no_polish", help="chain: compile the ladder z as it is")
    ap.add_argument("--replicas", type=int, default=1, help="workers per input group (12 groups)")
    ap.add_argument("--groups", type=int, default=None, help="input groups (default: one per training input)")
    ap.add_argument("--trial-workers", type=int, default=1, dest="trial_workers",
                    help="trials of one round at once (default 1; at most --replicas is useful)")
    ap.add_argument("--env-blocks", type=int, default=None, dest="env_blocks",
                    help="sublayer blocks of the environments (default: --replicas)")
    ap.add_argument("--env-layers", type=int, default=None, dest="env_layers",
                    help="at most this many sublayers cached per environment (bounds worker memory at t >= 6)")
    ap.add_argument("--hessian-chunk", type=int, default=None, dest="hessian_chunk",
                    help="Hessian rows per step between checkpoints (default 8 x --replicas)")
    ap.add_argument("--runs", default=str(REPO / "runs"), help="runs directory (scratch while developing)")
    ap.add_argument("--ref-cache", default=str(REF_CACHE), dest="ref_cache")
    ap.add_argument("--steps", default="compile,realize,evaluate")
    ap.add_argument("--max-round-trials", type=int, default=48, dest="max_round_trials")
    ap.add_argument("--F", type=float, default=22.0)
    ap.add_argument("--cutoff", type=float, default=None, help="evaluation cutoff (default: make_jobs' rule)")
    ap.add_argument("--chi-max", type=int, default=None, dest="chi_max")
    ap.add_argument("--eval-dt", type=float, default=None, dest="eval_dt",
                    help="reference step of the evaluation (default 0.1 at h_x = 1.6, 0.2 at h_x = 0)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--deadline", type=float, default=None, help="epoch seconds: checkpoint and stop after")
    a = ap.parse_args()
    if a.arm == "ring" and a.edge_fit is None:
        ap.error("--arm ring needs --edge-fit (lbfgs: production's edges.fit_edges; newton: fit_edges_newton)")
    if a.reopt == "coordinate" and a.trial_workers > 1:
        ap.error("--reopt coordinate runs one trial at a time (--trial-workers 1)")

    from snapevo.chain_snap import DeadlineReached, chain_point, ring_point
    from snapevo.hamiltonian import XXZ

    t = int(a.t) if float(a.t).is_integer() else a.t
    model = XXZ(0.25, a.field)
    ev = eval_settings(a.field, float(t), a.budget)
    if a.cutoff is not None:
        ev["cutoff"] = a.cutoff
    if a.chi_max is not None:
        ev["chi_max"] = a.chi_max
    if a.eval_dt is not None:
        ev["dt"] = a.eval_dt
    t0 = time.time()

    def log(s):
        print(f"{time.strftime('%m-%d %H:%M:%S')} [{model.tag} t={t:g} b={a.budget:.0e}] {s}", flush=True)

    kw = {"max_round_trials": a.max_round_trials, "F": a.F, "reopt": a.reopt, "top_k": a.top_k,
          "relax_ladder": a.relax_ladder}
    log(f"{a.arm} point: pid {os.getpid()}, runs {a.runs}, settings {kw}, replicas {a.replicas}, "
        f"trial workers {a.trial_workers}, polish {not a.no_polish}, evaluation {ev}, deadline {a.deadline}")
    margins = [float(m) for m in a.margins.split(",")]
    try:
        if a.arm == "ring":
            out = ring_point(model, float(t), a.budget, Path(a.runs), out_sub=a.ring_out, D=a.D, margins=margins,
                             settings_kw=kw, eval_kw=ev, steps=tuple(a.steps.split(",")), fresh=a.fresh,
                             deadline=a.deadline, cells=a.ring_cells, edge_fit=a.edge_fit,
                             all_depths=a.all_depths, log=log)
        else:
            if a.edge_zero_t:
                kw["edge_zero_t"] = True
            if a.no_edge_stragglers:
                kw["edge_weight"] = 0.0
            out = chain_point(model, float(t), a.budget, Path(a.runs), D=a.D, margins=margins,
                              ref_cache=Path(a.ref_cache), groups=a.groups, replicas=a.replicas,
                              trial_workers=a.trial_workers, env_blocks=a.env_blocks, settings_kw=kw,
                              eval_kw=ev, steps=tuple(a.steps.split(",")), fresh=a.fresh, deadline=a.deadline,
                              polish=not a.no_polish, hessian_chunk=a.hessian_chunk, env_layers=a.env_layers,
                              out_sub=a.chain_out, all_depths=a.all_depths, log=log)
    except DeadlineReached as exc:
        log(f"{a.arm} point: {exc} after {time.time() - t0:.0f}s")
        return 3
    for e in out:
        log(f"{a.arm} point: D = {e['D']}: 1 - F = {e['infid']:.3e}, c = {e['c_site']:.3e}, T/q {e['Tq']:.2f}, "
            f"with fallbacks {e['infid_total']:.3e}")
    log(f"{a.arm} point: done after {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
