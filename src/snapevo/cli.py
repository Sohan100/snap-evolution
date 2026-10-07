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
    summary          one CSV row per evaluated point, every model and time (runs/summary.csv)
    plot             T per qubit and T per rotation against the per-site infidelity at N = 100 (runs/figures/)
    compare          SNAP against the baseline at equal per-site infidelity, as a markdown table
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
    rec_D = next(r for r in records if int(r["D"]) == D)
    floor = max(rec_D["C_train"], rec_D.get("C_val", 0.0))
    if floor * 1.5 > args.budget:
        _log(f"compile: skip, the D = {D} floor {floor:.2e} leaves less than a third of the budget "
             f"{args.budget:.1e} for snapping (train deeper)")
        return D, None
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
                          rec["rotations_per_qubit"], model, args.t, N=args.N, f_tol=args.f_tol,
                          log=_log, **_mps_kw(args, d))
    ev.update({"model": model.to_dict(), "cell": cell.name, "t": args.t, "D": D, "budget": args.budget,
               "n_free": rec["n_free"], "n_t": rec["n_t"], "n_rus": rec["n_rus"], "n_kn": rec["n_kn"]})
    io.write_json(d / f"eval_{_stem(args.budget, D)}.json", ev)
    return ev


def cmd_point(args) -> None:
    _model, _cell, d = _ctx(args)
    done = sorted(d.glob(f"eval_{io.budget_tag(args.budget)}_D*.json"))
    if done and not args.fresh and not args.D:
        _log(f"point: {done[0].name} exists (use --fresh to redo)")
        return
    D, compiled = cmd_compile(args)
    if compiled is None:
        return
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
                          f_tol=args.f_tol, log=_log, **_mps_kw(args, d))
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
    from snapevo.baseline import SCAN_VERSION, scan
    model, _cell, d = _ctx(args)
    path = d / "baseline_scan.json"
    res = io.read_json(path) if path.exists() and not args.fresh else None
    if res is not None and res.get("version") != SCAN_VERSION:
        _log("baseline-scan: the cached scan is from an older version, scanning again")
        res = None
    if res is None:
        res = scan(model, args.t, log=_log)
        io.write_json(path, res)
    lo, hi = (float(v) for v in args.c_range.split(","))
    fronts = {"C": [p for p in res["front"] if lo <= p["C"] <= hi],                  # success branch
              "C_bound": [p for p in res["front_bound"] if lo <= p["C_bound"] <= hi]}  # with fallbacks
    todo = []
    if args.targets:
        import math
        for c in (float(v) for v in args.targets.split(",")):
            for key, front in fronts.items():
                if front:
                    best = min(front, key=lambda p: abs(math.log(max(p[key], 1e-300) / c)))
                    if best not in todo:
                        todo.append(best)
    else:
        for front in fronts.values():
            todo += [p for p in front if p not in todo]
    _log(f"baseline-scan: {len(res['points'])} points, fronts of {len(res['front'])} (success branch) and "
         f"{len(res['front_bound'])} (with fallbacks), {len(todo)} to evaluate")
    for p in todo:
        _log(f"  order {p['order']}, r = {p['steps']}, eps {p['eps']:.0e}: C {p['C']:.3e}, "
             f"with fallbacks {p['C_bound']:.3e}, T/q {p['Tq']:.1f}")
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


_SUMMARY_KEYS = ("infid", "infid_total", "fallback_share", "c_site", "C_ring", "infid_ring_pred", "Tq",
                 "T_edges_per_qubit", "T_per_rotation", "T_per_qubit_per_time", "rotations_per_qubit")


def cmd_summary(args) -> None:
    import csv
    import os
    from pathlib import Path
    base = Path(os.environ.get("SNAPEVO_RUNS", io.repo_root() / "runs"))
    rows = []
    for p in sorted(base.glob("*/t*/eval_*.json")) + sorted(base.glob("*/t*/baseline_o*.json")):
        e = io.read_json(p)
        if "mps" not in e:
            continue
        m = e["mps"]
        snap = p.name.startswith("eval_")
        c_total = 1.0 - (1.0 - min(e["infid_total"], 1.0)) ** (1.0 / e["N"])
        rows.append({"model": p.parent.parent.name, "t": e["t"], "method": "snap" if snap else "baseline",
                     "budget": e.get("budget", ""), "D": e.get("D", ""), "order": e.get("order", ""),
                     "steps": e.get("steps", ""), "eps": e.get("eps", ""), "F": e["F"], "N": e["N"],
                     "states": m["states"][1], "stderr": m["stderr"], "trunc_rel": m["trunc_rel"],
                     "trunc_check": m.get("trunc_check", {}).get("rel_change", ""), "chi_peak": m["chi_peak"],
                     **{k: e.get(k, "") for k in _SUMMARY_KEYS}, "c_total": c_total,
                     **{k: e.get(k, "") for k in ("n_free", "n_t", "n_rus", "n_kn")}})
    out = base / "summary.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} points -> {out}")


def _envelope(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower envelope of (infidelity, T) points: each kept point beats every cheaper one in infidelity."""
    env, best = [], float("inf")
    for x, y in sorted(pts, key=lambda p: (p[1], p[0])):
        if x < best:
            env.append((x, y))
            best = x
    return sorted(env)


def _interp(env: list[tuple[float, float]], x: float) -> float | None:
    """T on the envelope at infidelity x, log-log between neighbours; None outside its range."""
    import math
    if not env or x < env[0][0] or x > env[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(env, env[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return min(y0, y1)
            f = math.log(x / x0) / math.log(x1 / x0)
            return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
    return env[0][1]


def cmd_compare(args) -> None:
    """SNAP against the baseline at equal per-site infidelity, success branch and with the fallback bound.

    The baseline's T is interpolated (log-log) along the lower envelope of its
    evaluated points, and left blank where a SNAP point falls outside that range.
    """
    import csv
    import os
    from pathlib import Path
    cmd_summary(args)
    base = Path(os.environ.get("SNAPEVO_RUNS", io.repo_root() / "runs"))
    rows = list(csv.DictReader((base / "summary.csv").open()))
    names = {"xxz_d0.25_hx1.6": "A", "xxz_d0.25_hx0": "B"}
    fmt = lambda v, f: "" if v is None else format(v, f)
    print("| model | t | budget | D | c | SNAP T/q | base T/q | ratio | c with fallbacks | base T/q | ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ("xxz_d0.25_hx1.6", "xxz_d0.25_hx0"):
        for t in sorted({float(r["t"]) for r in rows if r["model"] == m}):
            sel = [r for r in rows if r["model"] == m and float(r["t"]) == t]
            base_pts = [r for r in sel if r["method"] == "baseline" and float(r["Tq"]) > 0]
            env_s = _envelope([(float(r["c_site"]), float(r["Tq"])) for r in base_pts])
            env_b = _envelope([(float(r["c_total"]), float(r["Tq"])) for r in base_pts
                               if float(r["infid_total"]) < 1.0])       # a bound at 1 says nothing
            for r in sorted((r for r in sel if r["method"] == "snap"), key=lambda r: -float(r["budget"])):
                tq, cs, ct = float(r["Tq"]), float(r["c_site"]), float(r["c_total"])
                bs, bb = _interp(env_s, cs), _interp(env_b, ct)
                print(f"| {names.get(m, m)} | {t:g} | {float(r['budget']):.0e} | {r['D']} | {cs:.2e} | {tq:.1f} | "
                      f"{fmt(bs, '.1f')} | {fmt(bs / tq if bs and tq > 0 else None, '.2f')} | {ct:.2e} | "
                      f"{fmt(bb, '.1f')} | {fmt(bb / tq if bb and tq > 0 else None, '.2f')} |")


_SNAP_COLOR, _BASE_COLOR, _INK, _MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def cmd_plot(args) -> None:
    """One panel per model and time: SNAP and the KN baseline, T against the per-site infidelity.

    The x axis is c = 1 - mean F^(1/N) from the 100-site MPS runs, so that the
    loose budgets, where 1 - F at N = 100 is close to 1, stay apart.
    1 - F = 1 - (1 - c)^100 is about 100 c below 1e-3. The *_bound figures use
    the infidelity with the fallback bound instead of the success branch. The
    baseline curve is the lower envelope of its evaluated points, and points
    with no T at all (a Clifford circuit) are left out of the log axes, and so
    are points whose fallback bound reaches 1 from the *_bound figures.
    """
    import csv
    import os
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmd_summary(args)
    base = Path(os.environ.get("SNAPEVO_RUNS", io.repo_root() / "runs"))
    rows = list(csv.DictReader((base / "summary.csv").open()))
    names = {"xxz_d0.25_hx1.6": "A", "xxz_d0.25_hx0": "B"}
    models = [m for m in ("xxz_d0.25_hx1.6", "xxz_d0.25_hx0") if any(r["model"] == m for r in rows)]
    times = sorted({float(r["t"]) for r in rows})
    out = base / "figures"
    out.mkdir(exist_ok=True)
    xs = {"": ("c_site", r"per-site infidelity $c$ at $N = 100$, success branch"),
          "_bound": ("c_total", r"per-site infidelity at $N = 100$ with the fallback bound")}
    for suffix, (xkey, xlabel) in xs.items():
        for key, label, fname in (("Tq", "T per qubit", "sweep_T_per_qubit"),
                                  ("T_per_rotation", "T per rotation", "sweep_T_per_rotation")):
            fig, axes = plt.subplots(len(models), len(times), figsize=(3.2 * len(times), 2.9 * len(models)),
                                     squeeze=False, sharex=True, sharey=True)
            for i, m in enumerate(models):
                for j, t in enumerate(times):
                    ax = axes[i][j]
                    for method, color, marker, face in (("snap", _SNAP_COLOR, "o", _SNAP_COLOR),
                                                        ("baseline", _BASE_COLOR, "s", "white")):
                        pts = sorted(((float(r[xkey]), float(r[key])) for r in rows
                                      if r["model"] == m and float(r["t"]) == t and r["method"] == method
                                      and float(r[key]) > 0 and (xkey != "c_total" or float(r["infid_total"]) < 1.0)),
                                     key=lambda p: p[0])
                        if method == "baseline":             # lower envelope of the baseline points
                            pts = _envelope(pts)
                        if pts:
                            ax.plot(*zip(*pts), color=color, lw=1.5, marker=marker, ms=4.5, mfc=face, mew=1.2,
                                    label="SNAP" if method == "snap" else "Trotter + KN")
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
                    ax.tick_params(colors=_MUTED, labelsize=8)
                    for sp in ax.spines.values():
                        sp.set_color("#c9c8c2")
                    ax.set_title(f"model {names[m]}, $t = {t:g}$", fontsize=9, color=_INK)
                    if i == len(models) - 1:
                        ax.set_xlabel(xlabel, fontsize=8, color=_MUTED)
                    if j == 0:
                        ax.set_ylabel(label, fontsize=8, color=_MUTED)
            axes[0][0].legend(fontsize=8, frameon=False)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out / f"{fname}{suffix}.{ext}", dpi=160)
            plt.close(fig)
            print(f"wrote {out / (fname + suffix)}.png and .pdf")


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
    eval_opts.add_argument("--f-tol", type=float, default=0.01, dest="f_tol",
                           help="F is the smallest whose total is within this fraction of the total at F = 30")

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
    p.add_argument("--c-range", default="1e-12,1", dest="c_range",
                   help="per-site range of front points to run (ring C, or with fallbacks for the second front)")
    p.add_argument("--evaluate", action="store_true", help="run the MPS evaluation of the front points")
    p.add_argument("--targets", default=None,
                   help="comma-separated per-site values; evaluate only the point of each front nearest each")
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
    p = sub.add_parser("summary", help="CSV of every evaluated point")
    p.set_defaults(func=cmd_summary)
    p = sub.add_parser("plot", help="T against the per-site infidelity at N = 100 (needs matplotlib)")
    p.set_defaults(func=cmd_plot)
    p = sub.add_parser("compare", help="SNAP against the baseline at equal infidelity, as a markdown table")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("verify-library", help="independent check of the corrected gadget file")
    p.set_defaults(func=cmd_verify_library)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
