"""Plug in the Clifford+T circuits: turn a compiled circuit into a priced, realized one.

Every rotation of the compiled brickwork ends up as one of

* an exact lattice angle (Clifford, T, or an RUS gadget). Its success branch is
  exactly Rz(theta), so the realized angle is the compiled angle. A gadget with
  a 2+T correction runs an approximate KN correction with probability q per use;
* a straggler, synthesized with KN non-mixed at the precision the compiler
  allotted. Its success branch is exactly Rz(theta_s), so the realized angle is
  theta_s, and its fallback branch runs with probability 1 - p.

The realized circuit (success branch everywhere) is what evaluate.py measures
directly. The heralded branches go in the ledger: per qubit per circuit,

    ledger  = (1/2) sum_i m_i q_i delta_i,      ledger2 = (1/2) sum_i m_i q_i delta_i^2,

with m_i the slot multiplicity and delta_i the diamond distance of the fallback
branch from the success branch. At N sites the ledger contributes N x ledger
to the diamond error, and both enter the fidelity bound (docs/error_metric.md).

The library part depends on the fallback cost F only through cost and q, so it
is priced on a grid of F and evaluate.py picks F.
"""

from __future__ import annotations

import numpy as np

from snapevo.cells import Cell
from snapevo.circuits import Layered, TRAIN_SET, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ
from snapevo.library import Library
from snapevo.synthesis import synthesize

__all__ = ["F_GRID", "realize"]

F_GRID = (18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0)


def realize(compiled: dict, model: XXZ, cell: Cell, t: float, N: int = 10,
            tighten: float = 0.5, max_tries: int = 6, log=print) -> dict:
    """Price every rotation and build the realized (success-branch) angle vector.

    KN stragglers are synthesized at the compiler's allotted eps. If the
    realized training C then exceeds the budget, every straggler eps is
    multiplied by ``tighten`` and the stragglers are synthesized again.
    """
    x = np.asarray(compiled["angles"], dtype=float)
    snapped = np.asarray(compiled["snapped"], dtype=bool)
    mult = np.asarray(compiled["multiplicity"], dtype=float)
    budget = float(compiled["settings"]["budget"])
    libs = {F: Library(F) for F in F_GRID}

    lib_rows = {}
    for i in np.flatnonzero(snapped):
        priced = {}
        for F, lib in libs.items():
            hit = lib.lookup(x[i])
            if hit is None:
                raise ValueError(f"snapped angle {i} = {x[i]!r} is not a lattice point")
            priced[F] = hit
        lib_rows[int(i)] = {"kind": Library.kind(x[i], priced[F_GRID[0]][0]),
                            "cost": {f"{F:g}": c for F, (c, q) in priced.items()},
                            "q": {f"{F:g}": q for F, (c, q) in priced.items()}}

    tin, tg = state_pairs(model, N, t, TRAIN_SET[1], TRAIN_SET[0])
    eps_plan = {int(k): float(v["eps"]) for k, v in compiled["stragglers"].items()}
    scale = 1.0
    for attempt in range(max_tries):
        kn_rows = {i: synthesize(x[i], e * scale) for i, e in eps_plan.items()}
        x_real = x.copy()
        for i, r in kn_rows.items():
            x_real[i] = r["theta_s"]
        c_real = local_infidelity(Layered.brickwork(cell, x_real).apply(tin, N), tg, N)
        log(f"realize: {len(lib_rows)} lattice angles, {len(kn_rows)} KN stragglers (eps x {scale:g}), "
            f"C_train {c_real:.3e} vs budget {budget:.3e}")
        if c_real <= budget * (1 + 1e-9) or not kn_rows:
            break
        scale *= tighten
    else:
        log("realize: realized C still above budget after tightening; keeping the last attempt")

    per_F = {}
    for F, lib in libs.items():
        cost, q, delta = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
        for i, row in lib_rows.items():
            cost[i], q[i], delta[i] = row["cost"][f"{F:g}"], row["q"][f"{F:g}"], lib.delta_fb
        for i, r in kn_rows.items():
            cost[i], q[i], delta[i] = r["expected_t"], 1.0 - r["p"], r["d_branch"]
        per_F[f"{F:g}"] = {"Tq": 0.5 * float(mult @ cost),
                           "ledger_per_qubit": 0.5 * float(mult @ (q * delta)),
                           "ledger2_per_qubit": 0.5 * float(mult @ (q * delta ** 2)),
                           "delta_fb": lib.delta_fb}
    rot_per_q = cell.rotations_per_sublayer_per_qubit * (len(x) // cell.n_angles)
    return {
        "budget": budget,
        "angles_compiled": x.tolist(),
        "angles_real": x_real.tolist(),
        "multiplicity": mult.tolist(),
        "C_train_real": c_real,
        "straggler_eps_scale": scale,
        "lattice": {str(i): r for i, r in lib_rows.items()},
        "kn": {str(i): r for i, r in kn_rows.items()},
        "per_F": per_F,
        "rotations_per_qubit": rot_per_q,
        "n_free": int(sum(1 for r in lib_rows.values() if r["kind"] == "clifford")),
        "n_t": int(sum(1 for r in lib_rows.values() if r["kind"] == "t")),
        "n_rus": int(sum(1 for r in lib_rows.values() if r["kind"] == "rus")),
        "n_kn": len(kn_rows),
        "max_kn_offset": max((abs(r["theta_s"] - x[i]) for i, r in kn_rows.items()), default=0.0),
        "t": t,
    }
