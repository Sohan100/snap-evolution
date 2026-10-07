"""Baseline: product formulas with every rotation synthesized by KN non-mixed.

H = H_E + H_O + H_X: even bonds, odd bonds, field. Each part is exact as a layer
of commuting rotations, written as an ``xxz+x`` sublayer (bond angles 2 c tau,
field angle h_x tau). Formulas, merged where adjacent layers are the same part:

    order 1: [X(dt) E(dt) O(dt)]^r
    order 2: [X(dt/2) E(dt/2) O(dt) E(dt/2) X(dt/2)]^r
    order 4: Suzuki, five order-2 steps with p = 1/(4 - 4^{1/3})

Errors are measured, never bounded (meeting of 2026-10-05): every distinct angle
is synthesized once with KN at diamond budget eps, every copy uses the same
circuit (deterministic, no mixing), the realized success-branch circuit goes
through the same metrics as SNAP, and the fallback branches go in the ledger.
Because every copy of an angle carries the same synthesis error, these errors
add coherently over the ring and over time steps, which the direct measurement
captures.
"""

from __future__ import annotations

import math

import numpy as np

from snapevo.cells import CELLS
from snapevo.circuits import HELD_OUT_SETS, Layered, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ
from snapevo.library import QUARTER
from snapevo.synthesis import kn_cost, synthesize

__all__ = ["trotter_circuit", "rotation_tally", "realize_baseline", "plan_cost", "EPS_GRID", "ORDERS", "SCAN_VERSION",
           "bound_per_site", "scan"]

EPS_GRID = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)
"""KN budgets the scan chooses from."""

ORDERS = (1, 2, 4)
"""Product-formula orders the scan tries."""

SCAN_VERSION = 3
"""Bumped when the scan's choice of points changes, so cached scans are redone."""

_CELL = CELLS["xxz+x"]


def _suzuki4() -> list[float]:
    p = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    return [p, p, 1.0 - 4.0 * p, p, p]


def _sequence(order: int, dt: float, r: int, field: bool) -> list[tuple[str, float]]:
    def s2(h):
        seq = [("X", h / 2)] if field else []
        seq += [("E", h / 2), ("O", h), ("E", h / 2)]
        return seq + ([("X", h / 2)] if field else [])

    raw: list[tuple[str, float]] = []
    for _ in range(r):
        if order == 1:
            raw += ([("X", dt)] if field else []) + [("E", dt), ("O", dt)]
        elif order == 2:
            raw += s2(dt)
        elif order == 4:
            for c in _suzuki4():
                raw += s2(c * dt)
        else:
            raise ValueError("order must be 1, 2 or 4")
    merged: list[list] = []
    for kind, tau in raw:
        if merged and merged[-1][0] == kind:
            merged[-1][1] += tau
        else:
            merged.append([kind, tau])
    return [(k, tau) for k, tau in merged]


def trotter_circuit(model: XXZ, t: float, order: int, r: int) -> Layered:
    """The order-``order`` product formula with r steps, as a layered circuit."""
    cxx, cyy, czz = model.bond_pauli
    layers = []
    for kind, tau in _sequence(order, t / r, r, field=bool(model.hx)):
        if kind == "X":
            layers.append((0, np.array([0.0, 0.0, 0.0, model.hx * tau, model.hx * tau])))
        else:
            layers.append((0 if kind == "E" else 1,
                           np.array([2 * cxx * tau, 2 * cyy * tau, 2 * czz * tau, 0.0, 0.0])))
    return Layered(_CELL, layers)


def _is_exact(theta: float) -> bool:
    k = theta / QUARTER
    return abs(k - round(k)) < 1e-12


def rotation_tally(circuit: Layered) -> dict[float, float]:
    """Distinct non-Clifford+T angles and how many rotations per qubit use each.

    A bond slot runs on N/2 bonds (1/2 per qubit), a field slot on N/2 bonds
    with one rotation per bond (also 1/2 per qubit).
    """
    tally: dict[float, float] = {}
    for _parity, angles in circuit.layers:
        for theta in angles:
            if not _is_exact(theta):
                key = float(round(theta, 15))
                tally[key] = tally.get(key, 0.0) + 0.5
    return tally


def plan_cost(circuit: Layered, eps: float) -> float:
    """Expected T per qubit with the calibrated KN law (for scans before synthesis)."""
    return sum(n for n in rotation_tally(circuit).values()) * kn_cost(eps)


def realize_baseline(circuit: Layered, eps: float) -> dict:
    """Synthesize every distinct angle at eps, return the realized circuit and its prices.

    ``ledger_per_qubit`` is (1/2) sum over slots of (1 - p) D(U1, U0), the
    heralded-branch term per qubit per circuit, as for SNAP.
    """
    syn = {}
    for theta in rotation_tally(circuit):
        syn[theta] = synthesize(theta, eps)
    layers = []
    Tq = ledger = ledger2 = 0.0
    for parity, angles in circuit.layers:
        real = angles.copy()
        for s, theta in enumerate(angles):
            if _is_exact(theta):
                Tq += 0.5 * (0.0 if round(theta / QUARTER) % 2 == 0 else 1.0)
                continue
            r = syn[float(round(theta, 15))]
            real[s] = r["theta_s"]
            Tq += 0.5 * r["expected_t"]
            ledger += 0.5 * (1.0 - r["p"]) * r["d_branch"]
            ledger2 += 0.5 * (1.0 - r["p"]) * r["d_branch"] ** 2
        layers.append((parity, real))
    rot = 0.5 * sum(1 for _p, a in circuit.layers for th in a if abs(np.remainder(th + np.pi, 2 * np.pi) - np.pi) > 1e-12)
    return {"circuit": Layered(circuit.cell, layers), "Tq": Tq, "ledger_per_qubit": ledger,
            "ledger2_per_qubit": ledger2, "rotations_per_qubit": rot, "n_distinct": len(syn), "eps": eps,
            "synth": {f"{k:.15f}": v for k, v in syn.items()}}


def bound_per_site(C: float, ledger: float, ledger2: float, N: int = 100) -> float:
    """Per-site infidelity with the fallback bound at N sites, predicted from the ring C and the ledgers."""
    from snapevo.evaluate import fidelity_bound
    total = fidelity_bound(1.0 - (1.0 - C) ** N, N * ledger, N * ledger2 / 2)
    return 1.0 - (1.0 - total) ** (1.0 / N)


def _front(points: list[dict], key: str) -> list[dict]:
    front, best = [], math.inf
    for p in sorted(points, key=lambda p: (p["Tq"], p[key])):
        if p[key] < best:
            front.append(p)
            best = p[key]
    return front


def scan(model: XXZ, t: float, orders=ORDERS, c_floor: float = 1e-8, r_max: int = 1024, N_eval: int = 100,
         log=print) -> dict:
    """Ring C at N = 10 and T per qubit of product formulas plus KN, and their Pareto front.

    For each order, r grows geometrically until the Trotter-only C is below
    ``c_floor``. Order 1 stops once its Trotter C is a tenth of one order-2
    step's, since past that point order 2 reaches the same C with fewer
    rotations. The KN budget is chosen around eps*, where KN's coherent error,
    about (R x 0.45 eps)^2 per site for R rotations per qubit, is a third of
    the Trotter C. The estimate is pessimistic (KN's errors partly cancel), so
    each (order, r) is synthesized at the first grid budget at or below eps*
    and the two above it, and at r = 1 at every looser budget too, which gives
    the front cheap points for loose targets. C is measured on one held-out
    set (64 states).

    There are two fronts. ``front`` ranks by the success-branch C. At loose
    budgets KN's fallback branches carry most of the error, and at N = 100
    nearly every run takes dozens of them, so ``front_bound`` ranks by the
    per-site infidelity with the fallback bound at N_eval sites instead
    (``bound_per_site``). evaluate.py runs points of both at N = 100.
    """
    seed, n = HELD_OUT_SETS[1]
    tin, tg = state_pairs(model, 10, t, n, seed)
    c_step2 = local_infidelity(trotter_circuit(model, t, 2, 1).apply(tin, 10), tg, 10)
    points = []
    for order in orders:
        stop = max(c_floor, c_step2 / 10) if order == 1 else c_floor
        r = 1
        while True:
            circ = trotter_circuit(model, t, order, r)
            c_tr = local_infidelity(circ.apply(tin, 10), tg, 10)
            R = sum(rotation_tally(circ).values())
            eps_star = math.sqrt(c_tr / 3) / (0.45 * R) if R else EPS_GRID[0]
            looser = [e for e in EPS_GRID if e > eps_star]
            tighter = [e for e in EPS_GRID if e <= eps_star]
            for eps in (looser if r == 1 else looser[-2:]) + tighter[:1]:
                rb = realize_baseline(circ, eps)
                c = local_infidelity(rb["circuit"].apply(tin, 10), tg, 10)
                cb = bound_per_site(c, rb["ledger_per_qubit"], rb["ledger2_per_qubit"], N_eval)
                points.append({"order": order, "steps": r, "eps": eps, "C": c, "C_trotter": c_tr, "C_bound": cb,
                               "Tq": rb["Tq"], "rotations_per_qubit": rb["rotations_per_qubit"],
                               "ledger_per_qubit": rb["ledger_per_qubit"],
                               "ledger2_per_qubit": rb["ledger2_per_qubit"]})
                log(f"scan: order {order}, r = {r:3d}, eps {eps:.0e}: C {c:.3e} (Trotter {c_tr:.3e}), "
                    f"with fallbacks {cb:.3e}, T/q {rb['Tq']:.1f}")
            if c_tr < stop or r >= r_max:
                break
            r = r + 1 if r < 4 else int(round(r * 1.5))
    return {"model": model.to_dict(), "t": t, "N": 10, "states": [seed, n], "version": SCAN_VERSION,
            "orders": list(orders), "eps_grid": list(EPS_GRID), "N_eval": N_eval, "points": points,
            "front": _front(points, "C"), "front_bound": _front(points, "C_bound")}
