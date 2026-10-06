"""Command line: ``python -m snapevo <command> ...``. Run with ``-h`` for options.

    train            train the depth ladder (train.py)
    compile          Hessian SNAP at one budget (snap.py)
    realize          price every rotation, synthesize stragglers with KN (realize.py)
    evaluate         held-out C, exact op norm vs N, extrapolation, ledger, F (metrics.py)
    point            compile, realize and evaluate in one go
    baseline         one product-formula + KN point, measured the same way (baseline.py)
    report           table of every evaluated point of a model and time
    verify-library   independent check of the corrected gadget file (library_check.py)

Files go to runs/<model tag>/t<t>/ (io.py).
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


def cmd_realize(args, D: int | None = None) -> dict:
    from snapevo.realize import realize
    model, cell, d = _ctx(args)
    D = D or _resolve_D(args, d)
    compiled = io.read_json(d / f"snap_{_stem(args.budget, D)}.json")
    rec = realize(compiled, model, cell, args.t, log=_log)
    rec.update({"model": model.to_dict(), "cell": cell.name, "D": D})
    io.write_json(d / f"real_{_stem(args.budget, D)}.json", rec)
    return rec


def cmd_evaluate(args, D: int | None = None) -> dict:
    from snapevo.metrics import evaluate
    model, cell, d = _ctx(args)
    D = D or _resolve_D(args, d)
    rec = io.read_json(d / f"real_{_stem(args.budget, D)}.json")
    sizes = tuple(int(v) for v in args.sizes.split(","))
    ev = evaluate(rec, model, cell, args.t, sizes=sizes, N_target=args.N_target,
                  ledger_frac=args.ledger_frac, log=_log)
    ev.update({"model": model.to_dict(), "cell": cell.name, "t": args.t, "D": D, "budget": args.budget})
    io.write_json(d / f"eval_{_stem(args.budget, D)}.json", ev)
    return ev


def cmd_point(args) -> None:
    D, _ = cmd_compile(args)
    cmd_realize(args, D)
    cmd_evaluate(args, D)


def cmd_baseline(args) -> None:
    from snapevo.baseline import realize_baseline, trotter_circuit
    from snapevo.metrics import measure, totals
    model, _cell, d = _ctx(args)
    circ = trotter_circuit(model, args.t, args.order, args.steps)
    sizes = tuple(int(v) for v in args.sizes.split(","))
    rb = realize_baseline(circ, args.eps)
    real = rb.pop("circuit")
    out = {"model": model.to_dict(), "t": args.t, "order": args.order, "steps": args.steps, **rb,
           "trotter": measure(circ, model, args.t, sizes, args.N_target, log=_log),
           "realized": measure(real, model, args.t, sizes, args.N_target, log=_log)}
    out.update(totals(out["realized"], rb["ledger_per_qubit"], rb["ledger2_per_qubit"]))
    out["T_per_rotation"] = out["Tq"] / out["rotations_per_qubit"]
    out["T_per_qubit_per_time"] = out["Tq"] / args.t
    io.write_json(d / f"baseline_o{args.order}_r{args.steps}_e{args.eps:.0e}.json", out)
    _log(f"baseline: T/q {out['Tq']:.1f}, T/rot {out['T_per_rotation']:.2f}; at N = {args.N_target}: "
         f"D_total {out['D_total']:.3e}, state infid {out['state_infid_total']:.3e}, "
         f"process infid {out['process_infid_total']:.3e} (ledger {out['ledger']:.1e})")


_HEAD = (f"{'point':>9} {'D':>3} {'C held':>9} {'alpha':>9} {'D_total':>9} {'1-F st':>9} {'1-F pr':>9} "
         f"{'F':>3} {'T/q':>7} {'T/rot':>6} {'T/q/t':>7}")


def _row(label, D, e, F, Tq, Trot, Tqt):
    return (f"{label:>9} {D:>3} {e['held_out']['worst']:9.2e} {e['alpha']:9.2e} {e['D_total']:9.2e} "
            f"{e['state_infid_total']:9.2e} {e['process_infid_total']:9.2e} {F:>3} {Tq:7.1f} {Trot:6.2f} "
            f"{Tqt:7.1f}")


def cmd_report(args) -> None:
    model, _cell, d = _ctx(args)
    print(f"{model.describe()}, t = {args.t:g}. Totals at N = {args.N_target} (fallbacks included).")
    print("budget/order, depth, held-out C, alpha = w/N, D_total, state and process infidelity,")
    print("fallback cost F, T per qubit, T per rotation, T per qubit per unit time")
    print(_HEAD)
    for p in sorted(d.glob("eval_*.json"), key=lambda p: -io.read_json(p)["budget"]):
        e = io.read_json(p)
        print(_row(f"{e['budget']:.1e}", e["D"], e, f"{e['F']:.0f}", e["Tq"], e["T_per_rotation"],
                   e["T_per_qubit_per_time"]))
    for p in sorted(d.glob("baseline_*.json")):
        b = io.read_json(p)
        e = {**b["realized"], **{k: b[k] for k in ("D_total", "state_infid_total", "process_infid_total")}}
        print(_row(f"o{b['order']}r{b['steps']}", "", e, "", b["Tq"], b["T_per_rotation"],
                   b["T_per_qubit_per_time"]) + f"   (KN eps {b['eps']:.0e})")


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

    eval_opts = argparse.ArgumentParser(add_help=False)
    eval_opts.add_argument("--sizes", default="6,8,10,12,14", help="ring sizes for the exact op norm and process fidelity")
    eval_opts.add_argument("--N-target", type=int, default=100, dest="N_target")
    eval_opts.add_argument("--ledger-frac", type=float, default=0.1, dest="ledger_frac")

    p = sub.add_parser("compile", parents=[snap_opts], help="Hessian SNAP at one budget")
    common(p, budget=True)
    p.set_defaults(func=cmd_compile)
    p = sub.add_parser("realize", help="price rotations, synthesize stragglers")
    common(p, budget=True)
    p.set_defaults(func=cmd_realize)
    p = sub.add_parser("evaluate", parents=[eval_opts], help="error metrics of a realized circuit")
    common(p, budget=True)
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("point", parents=[snap_opts, eval_opts], help="compile + realize + evaluate")
    common(p, budget=True)
    p.set_defaults(func=cmd_point)
    p = sub.add_parser("baseline", parents=[eval_opts], help="product formula + KN, measured")
    common(p)
    p.add_argument("--order", type=int, default=2, choices=(1, 2, 4))
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eps", type=float, required=True, help="KN diamond budget per rotation")
    p.set_defaults(func=cmd_baseline)
    p = sub.add_parser("report", help="table of evaluated points")
    common(p)
    p.add_argument("--N-target", type=int, default=100, dest="N_target")
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
