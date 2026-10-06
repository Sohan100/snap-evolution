"""Command line: ``python -m snapevo <command> ...``. Run with ``-h`` for options.

    train            train the depth ladder on the ring (train.py)
    compile          Hessian SNAP at one budget (snap.py)
    realize          price every rotation, KN for stragglers, fit and synthesize edges (realize.py, edges.py)
    evaluate         MPS state infidelity at N = 100 on an open chain, fallbacks, F (evaluate.py)
    point            compile, realize and evaluate in one go
    baseline         one product-formula + KN point, evaluated the same way (baseline.py)
    baseline-scan    product formulas + KN screened on the ring, then the Pareto front by MPS
    reference        build and cache the MPS references e^{-iHt}|phi_k> (mps.py)
    crosscheck       small-ring metrics of a realized point: op norm, diamond, process fidelity (crosscheck.py)
    report           table of every evaluated point of a model and time
    verify-library   independent check of the corrected gadget file (library_check.py)

Files go to runs/<model tag>/t<t>/ (io.py); cached MPS references to its ref/ folder.
"""

from __future__ import annotations

import argparse
import sys
import time

from snapevo import io
from snapevo.cells import cell_for, get_cell
from snapevo.hamiltonian import get_model

N_COMPILE = 10


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _ctx(args):
    model = get_model(args.model)
    cell = get_cell(args.cell) if getattr(args, "cell", None) else cell_for(model)
    d = io.run_dir(model.tag, args.t)
    return model, cell, d


def _pick_depth(records, budget: float, margin: float) -> int:
    """Shallowest depth whose floor (max of train and val C) times margin is under the budget."""
    for r in sorted(records, key=lambda r: r["D"]):
        if max(r["C_train"], r.get("C_val", 0.0)) * margin <= budget:
            return int(r["D"])
    return int(max(records, key=lambda r: r["D"])["D"])


def _stem(budget: float, D: int) -> str:
    return f"{io.budget_tag(budget)}_D{D}"


def cmd_train(args) -> None:
    from snapevo.train import train_ladder
    model, cell, d = _ctx(args)
    _log(f"train: {model.describe()}, cell {cell.name}, t = {args.t:g}, depths {args.depths}")
    train_ladder(model, cell, args.t, [int(v) for v in args.depths.split(",")], d / "cells.json",
                 maxiter=args.maxiter, log=_log)


def cmd_compile(args) -> tuple[int, dict]:
    from snapevo.circuits import TRAIN_SET, Layered, local_infidelity, state_pairs
    from snapevo.jaxcost import make_cost, make_hessian
    from snapevo.library import Library
    from snapevo.snap import SnapSettings, compile_snap
    from snapevo.train import cell_records, load_cell

    model, cell, d = _ctx(args)
    records = cell_records(d / "cells.json")
    D = args.D or _pick_depth(records, args.budget, args.floor_margin)
    x_star, header = load_cell(d / "cells.json", D)
    if header["cell"] != cell.name:
        raise SystemExit(f"cells.json was trained with {header['cell']}, not {cell.name}")
    out = d / f"snap_{_stem(args.budget, D)}.json"
    if out.exists() and not args.fresh:
        _log(f"compile: {out.name} exists (use --fresh to redo)")
        return D, io.read_json(out)
    tin, tg = state_pairs(model, N_COMPILE, args.t, TRAIN_SET[1], TRAIN_SET[0])
    objective = make_cost(cell, N_COMPILE, tin, tg, D)
    hessian = make_hessian(cell, N_COMPILE, tin, tg, D, method=args.hessian, grad=objective)

    def c_fn(x):
        return local_infidelity(Layered.brickwork(cell, x).apply(tin, N_COMPILE), tg, N_COMPILE)

    settings = SnapSettings(budget=args.budget, F=args.F, top_k=args.top_k, reserve_t=args.reserve_t,
                            reserve_frac=args.reserve_frac, max_round_trials=args.max_round_trials,
                            share_cap=args.share_cap, relax_ladder=args.relax_ladder,
                            library_stragglers=not args.no_library_stragglers, skip_failed=args.skip_failed,
                            checkpoint=str(out.with_suffix(".ckpt.pkl")))
    _log(f"compile: {model.describe()}, cell {cell.name}, t = {args.t:g}, D = {D}, budget {args.budget:.1e}")
    res = compile_snap(x_star, cell, settings, objective, c_fn, hessian, tin, tg, N_COMPILE,
                       Library(args.F), log=_log)
    res.update({"model": model.to_dict(), "cell": cell.name, "t": args.t, "D": D, "N": N_COMPILE})
    io.write_json(out, res)
    ck = out.with_suffix(".ckpt.pkl")
    if ck.exists():
        ck.unlink()
    _log(f"compile: wrote {out}")
    return D, res


def _resolve_D(args, d) -> int:
    if args.D:
        return args.D
    hits = sorted(d.glob(f"snap_{io.budget_tag(args.budget)}_D*.json"))
    if len(hits) != 1:
        raise SystemExit(f"give --D: found {[h.name for h in hits]}")
    return int(hits[0].stem.split("_D")[-1])


def _add_edges(rec: dict, model, cell, t: float, N_fit: int | None) -> dict:
    """Fit the open-chain edge rotations of a realized brickwork and synthesize them."""
    from snapevo.circuits import Layered
    from snapevo.edges import edge_eps, fit_edges, realize_edges
    fit = fit_edges(Layered.brickwork(cell, rec["angles_real"]), model, t, N_fit=N_fit, log=_log)
    rec["edges"] = E = realize_edges(fit, edge_eps(rec["budget"]))
    if E["use"]:
        _log(f"edges: kept, {E['n_kn']} KN rotations at eps {E['eps']:.0e}, {E['T_total']:.0f} T in total")
    else:
        _log("edges: the fit gains less than one bulk site's worth; keeping the cell's own factor")
    return rec


def cmd_realize(args, D: int | None = None) -> dict:
    from snapevo.realize import realize
    model, cell, d = _ctx(args)
    D = D or _resolve_D(args, d)
    compiled = io.read_json(d / f"snap_{_stem(args.budget, D)}.json")
    rec = realize(compiled, model, cell, args.t, log=_log)
    rec.update({"model": model.to_dict(), "cell": cell.name, "D": D})
    _add_edges(rec, model, cell, args.t, args.edge_N)
    io.write_json(d / f"real_{_stem(args.budget, D)}.json", rec)
    return rec


def _mps_kw(args, d) -> dict:
    return {"states": (args.seed, args.states), "dt": args.dt, "chi_max": args.chi_max, "cutoff": args.cutoff,
            "cache_dir": None if args.no_cache else d / "ref", "check": not getattr(args, "no_check", False)}


def cmd_evaluate(args, D: int | None = None) -> dict:
    from snapevo.circuits import Layered
    from snapevo.evaluate import evaluate_circuit
    model, cell, d = _ctx(args)
    D = D or _resolve_D(args, d)
    path = d / f"real_{_stem(args.budget, D)}.json"
    rec = io.read_json(path)
    if "edges" not in rec:
        _log(f"evaluate: {path.name} has no edge rotations yet; fitting them now")
        io.write_json(path, _add_edges(rec, model, cell, args.t, None))
    ev = evaluate_circuit(Layered.brickwork(cell, rec["angles_real"]), rec["edges"], rec["per_F"],
                          rec["rotations_per_qubit"], model, args.t, N=args.N, ledger_frac=args.ledger_frac,
                          log=_log, **_mps_kw(args, d))
    ev.update({"model": model.to_dict(), "cell": cell.name, "t": args.t, "D": D, "budget": args.budget,
               "n_free": rec["n_free"], "n_t": rec["n_t"], "n_rus": rec["n_rus"], "n_kn": rec["n_kn"]})
    io.write_json(d / f"eval_{_stem(args.budget, D)}.json", ev)
    return ev


def cmd_point(args) -> None:
    D, _ = cmd_compile(args)
    cmd_realize(args, D)
    cmd_evaluate(args, D)


def cmd_crosscheck(args) -> dict:
    from snapevo.crosscheck import crosscheck
    model, cell, d = _ctx(args)
    D = _resolve_D(args, d)
    rec = io.read_json(d / f"real_{_stem(args.budget, D)}.json")
    sizes = tuple(int(v) for v in args.sizes.split(","))
    ev = crosscheck(rec, model, cell, args.t, sizes=sizes, N_target=args.N_target,
                    ledger_frac=args.ledger_frac, log=_log)
    ev.update({"model": model.to_dict(), "cell": cell.name, "t": args.t, "D": D, "budget": args.budget})
    io.write_json(d / f"xcheck_{_stem(args.budget, D)}.json", ev)
    return ev


def _baseline_point(args, model, d, order: int, steps: int, eps: float, crosscheck: bool = False) -> dict:
    from snapevo.baseline import realize_baseline, trotter_circuit
    from snapevo.evaluate import evaluate_circuit
    circ = trotter_circuit(model, args.t, order, steps)
    rb = realize_baseline(circ, eps)
    real = rb.pop("circuit")
    pricing = {"KN": {k: rb[k] for k in ("Tq", "ledger_per_qubit", "ledger2_per_qubit")}}
    _log(f"baseline: order {order}, r = {steps}, KN eps {eps:.0e}")
    ev = evaluate_circuit(real, None, pricing, rb["rotations_per_qubit"], model, args.t, N=args.N,
                          ledger_frac=args.ledger_frac, log=_log, **_mps_kw(args, d))
    out = {"model": model.to_dict(), "t": args.t, "order": order, "steps": steps, **rb, **ev}
    if crosscheck:
        from snapevo.crosscheck import measure
        out["crosscheck"] = {"trotter": measure(circ, model, args.t, log=_log),
                             "realized": measure(real, model, args.t, log=_log)}
    io.write_json(d / f"baseline_o{order}_r{steps}_e{eps:.0e}.json", out)
    return out


def cmd_baseline(args) -> None:
    model, _cell, d = _ctx(args)
    _baseline_point(args, model, d, args.order, args.steps, args.eps, args.crosscheck)


def cmd_baseline_scan(args) -> None:
    from snapevo.baseline import scan
    model, _cell, d = _ctx(args)
    path = d / "baseline_scan.json"
    res = io.read_json(path) if path.exists() and not args.fresh else scan(model, args.t, log=_log)
    io.write_json(path, res)
    lo, hi = (float(v) for v in args.c_range.split(","))
    todo = [p for p in res["front"] if lo <= p["C"] <= hi]
    _log(f"baseline-scan: {len(res['points'])} points, front of {len(res['front'])}, "
         f"{len(todo)} with C in [{lo:.0e}, {hi:.0e}]")
    for p in todo:
        _log(f"  front: order {p['order']}, r = {p['steps']}, eps {p['eps']:.0e}: C {p['C']:.3e}, T/q {p['Tq']:.1f}")
    if args.evaluate:
        for p in todo:
            if (d / f"baseline_o{p['order']}_r{p['steps']}_e{p['eps']:.0e}.json").exists() and not args.fresh:
                continue
            _baseline_point(args, model, d, p["order"], p["steps"], p["eps"])


def cmd_reference(args) -> None:
    from snapevo.circuits import product_vectors
    from snapevo.evaluate import ref_path
    from snapevo.mps import reference_state
    model, _cell, d = _ctx(args)
    vecs = product_vectors(args.N, args.states, args.seed)
    ks = range(args.states) if args.only is None else [int(k) for k in args.only.split(",")]
    for k in ks:
        reference_state(model, args.t, vecs[k], args.dt, args.chi_max, args.cutoff,
                        ref_path(d / "ref", args.N, args.seed, k, args.dt, args.cutoff, args.chi_max), log=_log)


_HEAD = (f"{'point':>10} {'D':>3} {'C ring':>9} {'1-F mps':>9} {'stat':>6} {'bar':>6} {'conv':>7} "
         f"{'1-F tot':>9} {'per site':>9} {'F':>3} {'T/q':>7} {'T/rot':>6} {'T/q/t':>7} {'chi':>5}")


def _row(label: str, D, e: dict, F: str) -> str:
    m = e["mps"]
    conv = f"{m['trunc_check']['rel_change']:7.0e}" if "trunc_check" in m else f"{'-':>7}"
    return (f"{label:>10} {D!s:>3} {e['C_ring']:9.2e} {e['infid']:9.3e} {m['stderr'] / e['infid']:6.1%} "
            f"{m['trunc_rel']:6.1%} {conv} {e['infid_total']:9.3e} {e['c_site']:9.2e} {F:>3} {e['Tq']:7.1f} "
            f"{e['T_per_rotation']:6.2f} {e['T_per_qubit_per_time']:7.1f} {m['chi_peak']:5d}")


def cmd_report(args) -> None:
    model, _cell, d = _ctx(args)
    print(f"{model.describe()}, t = {args.t:g}, open chain. State infidelity on random product inputs by MPS.")
    print("C ring = held-out C at N = 10 on the ring. 1-F mps = success branch at N. stat = statistical error,")
    print("bar = pessimistic truncation bar, conv = change of input 0 at a 100x smaller cutoff, all relative.")
    print("1-F tot includes the fallbacks.")
    print(_HEAD)
    for p in sorted(d.glob("eval_*.json"), key=lambda p: -io.read_json(p)["budget"]):
        e = io.read_json(p)
        print(_row(f"{e['budget']:.1e}", e["D"], e, str(e["F"])) + f"   N = {e['N']}")
    for p in sorted(d.glob("baseline_*.json")):
        b = io.read_json(p)
        if "mps" not in b:
            continue
        print(_row(f"o{b['order']}r{b['steps']}", "", b, "") + f"   N = {b['N']}, KN eps {b['eps']:.0e}")


def cmd_verify_library(args) -> None:
    from snapevo.library_check import main as check
    sys.exit(check())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="snapevo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, budget=False):
        p.add_argument("--model", required=True, help="A (h_x = 1.6) or B (h_x = 0)")
        p.add_argument("--t", type=float, required=True, help="evolution time, spin units")
        p.add_argument("--cell", default=None, help="override the model's default cell")
        if budget:
            p.add_argument("--budget", type=float, required=True, help="training-C budget")
            p.add_argument("--D", type=int, default=None, help="depth (default: picked from the ladder)")

    p = sub.add_parser("train", help="train the depth ladder")
    common(p)
    p.add_argument("--depths", required=True, help="comma-separated depths")
    p.add_argument("--maxiter", type=int, default=6000)
    p.set_defaults(func=cmd_train)

    snap_opts = argparse.ArgumentParser(add_help=False)
    snap_opts.add_argument("--F", type=float, default=22.0, help="library fallback cost while compiling")
    snap_opts.add_argument("--floor-margin", type=float, default=3.0, dest="floor_margin")
    snap_opts.add_argument("--top-k", type=int, default=6, dest="top_k")
    snap_opts.add_argument("--reserve-t", type=float, default=18.0, dest="reserve_t")
    snap_opts.add_argument("--reserve-frac", type=float, default=0.1, dest="reserve_frac")
    snap_opts.add_argument("--max-round-trials", type=int, default=48, dest="max_round_trials")
    snap_opts.add_argument("--share-cap", type=float, default=3.0, dest="share_cap")
    snap_opts.add_argument("--relax-ladder", default="coarse", choices=("coarse", "gentle"), dest="relax_ladder")
    snap_opts.add_argument("--no-library-stragglers", action="store_true", dest="no_library_stragglers")
    snap_opts.add_argument("--skip-failed", action="store_true", dest="skip_failed")
    snap_opts.add_argument("--hessian", default="auto", choices=("auto", "jax", "fd"))
    snap_opts.add_argument("--fresh", action="store_true", help="recompile even if the result exists")

    real_opts = argparse.ArgumentParser(add_help=False)
    real_opts.add_argument("--edge-N", type=int, default=None, dest="edge_N",
                           help="open-chain size for the edge fit (default grows with t, 12 to 18)")

    eval_opts = argparse.ArgumentParser(add_help=False)
    eval_opts.add_argument("--N", type=int, default=100, help="open-chain length of the MPS run")
    eval_opts.add_argument("--states", type=int, default=8, help="number of random product inputs")
    eval_opts.add_argument("--seed", type=int, default=2026, help="seed of the inputs")
    eval_opts.add_argument("--chi-max", type=int, default=1024, dest="chi_max")
    eval_opts.add_argument("--cutoff", type=float, default=1e-12, help="discarded weight allowed per SVD")
    eval_opts.add_argument("--dt", type=float, default=0.2, help="step of the order-4 reference")
    eval_opts.add_argument("--no-cache", action="store_true", dest="no_cache", help="do not cache references")
    eval_opts.add_argument("--no-check", action="store_true", dest="no_check",
                           help="skip redoing input 0 at a 100x smaller cutoff")
    eval_opts.add_argument("--ledger-frac", type=float, default=0.1, dest="ledger_frac")

    xc_opts = argparse.ArgumentParser(add_help=False)
    xc_opts.add_argument("--sizes", default="6,8,10,12,14", help="ring sizes for the exact op norm and process fidelity")
    xc_opts.add_argument("--N-target", type=int, default=100, dest="N_target")
    xc_opts.add_argument("--ledger-frac", type=float, default=0.1, dest="ledger_frac")

    p = sub.add_parser("compile", parents=[snap_opts], help="Hessian SNAP at one budget")
    common(p, budget=True)
    p.set_defaults(func=cmd_compile)
    p = sub.add_parser("realize", parents=[real_opts], help="price rotations, synthesize stragglers, fit edges")
    common(p, budget=True)
    p.set_defaults(func=cmd_realize)
    p = sub.add_parser("evaluate", parents=[eval_opts], help="MPS infidelity of a realized circuit at N = 100")
    common(p, budget=True)
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("point", parents=[snap_opts, real_opts, eval_opts], help="compile + realize + evaluate")
    common(p, budget=True)
    p.set_defaults(func=cmd_point)
    p = sub.add_parser("baseline", parents=[eval_opts], help="product formula + KN, evaluated by MPS")
    common(p)
    p.add_argument("--order", type=int, default=2, choices=(1, 2, 4))
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eps", type=float, required=True, help="KN diamond budget per rotation")
    p.add_argument("--crosscheck", action="store_true", help="also run the small-ring metrics")
    p.set_defaults(func=cmd_baseline)
    p = sub.add_parser("baseline-scan", parents=[eval_opts], help="screen product formulas, MPS on the front")
    common(p)
    p.add_argument("--c-range", default="1e-7,1e-3", dest="c_range", help="ring C range of front points to run")
    p.add_argument("--evaluate", action="store_true", help="run the MPS evaluation of the front points")
    p.add_argument("--fresh", action="store_true", help="redo the scan and the evaluations")
    p.set_defaults(func=cmd_baseline_scan)
    p = sub.add_parser("reference", parents=[eval_opts], help="build and cache the MPS references")
    common(p)
    p.add_argument("--only", default=None, help="comma-separated input indices (default: all)")
    p.set_defaults(func=cmd_reference)
    p = sub.add_parser("crosscheck", parents=[xc_opts], help="small-ring metrics of a realized point")
    common(p, budget=True)
    p.set_defaults(func=cmd_crosscheck)
    p = sub.add_parser("report", help="table of evaluated points")
    common(p)
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("verify-library", help="independent check of the corrected gadget file")
    p.set_defaults(func=cmd_verify_library)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
