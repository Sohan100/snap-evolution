"""The reported error: state infidelity on random product inputs at N = 100, by MPS.

For each input |phi_k> (a random state on every site, MPS_SET draws) we run the
realized circuit (every rotation on its success branch, edge rotations included)
and the reference e^{-iHt} on an open chain of N sites as MPSs (mps.py), and take

    F_k = |<v_k|u_k>|^2 .

We report the mean infidelity 1 - mean_k F_k and the per-site value
c = 1 - mean_k F_k^{1/N}, which is defined like the training C.

Truncation. Each MPS carries its summed discarded weight eps. The usual TEBD
estimate of its distance from the untruncated state is sqrt(eps). If those
errors lined up with the circuit's own error they would move sqrt(1 - F) by
sqrt(eps_u) + sqrt(eps_v); that pessimistic bar is ``trunc_rel``. The test that
counts is ``trunc_check``: input 0 is redone at a cutoff 100 times smaller and the
relative change of its infidelity is reported. Either above 5% flags the point.

Fallbacks. The heralded branches enter through fidelity_bound, with
L = N l + L_edges and Q2 = N l2 / 2 + Q2_edges (realize.py, edges.py). A larger
library fallback cost F only shrinks the library's share of L, for a few tenths
of a T, so F is the smallest grid value whose total is within ``f_tol`` (1%) of
the total at the largest F. What is left (mostly KN stragglers) is reported as
``fallback_share`` of the total and flagged above 10%.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from snapevo.circuits import Layered, held_out, product_vectors
from snapevo.hamiltonian import XXZ
from snapevo.mps import MPS, apply_layered, reference_state

__all__ = ["MPS_SET", "ref_path", "fidelity_bound", "mps_infidelity", "evaluate_circuit"]

MPS_SET: tuple[int, int] = (2026, 8)
"""(seed, n_states) of the product inputs of the N = 100 runs."""


def ref_path(cache_dir: Path, N: int, seed: int, k: int, dt: float, cutoff: float, chi_max: int) -> Path:
    """Cache file of the reference MPS for input k of the set (seed, .)."""
    return Path(cache_dir) / f"ref_N{N}_s{seed}_k{k}_dt{dt:g}_c{cutoff:.0e}_x{chi_max}.npz"


def fidelity_bound(infid_success: float, L: float, Q2: float) -> float:
    """Infidelity of the realized channel given the success-branch infidelity.

    L = sum_i q_i delta_i is the ledger and Q2 = sum_i q_i delta_i^2 / 2. Each
    fallback moves the output by at most delta_i/2 in trace distance, so for
    every branch pattern sqrt(1 - F_b) <= sqrt(1 - F_s) + X_b with X_b the sum of
    its delta_i/2, and averaging (1 - F) <= E(sqrt(1 - F_s) + X)^2 gives this.
    Var X <= Q2 needs q_i <= 1, which holds since every library gadget has p >= 0.9.
    """
    return min(1.0, (math.sqrt(max(infid_success, 0.0)) + L / 2) ** 2 + Q2)


def _fidelity(circuit, edges, model, t, vec, dt, chi_max, cutoff, cache, log):
    v, _ = reference_state(model, t, vec, dt, chi_max, cutoff, cache, log)
    t0 = time.time()
    u = apply_layered(MPS.product(vec, chi_max=chi_max, cutoff=cutoff), circuit, edges)
    return abs(v.overlap(u)) ** 2 / abs(v.overlap(v) * u.overlap(u)), u, v, time.time() - t0


def mps_infidelity(circuit: Layered, edges, model: XXZ, t: float, N: int = 100,
                   states: tuple[int, int] = MPS_SET, dt: float = 0.2, chi_max: int = 1024,
                   cutoff: float = 1e-12, cache_dir: Path | None = None, check: bool = True,
                   log=print) -> dict:
    """Success-branch state infidelity of an open-chain circuit at N sites, with its error bars."""
    seed, n = states
    vecs = product_vectors(N, n, seed)

    def cache(k, c):
        return None if cache_dir is None else ref_path(cache_dir, N, seed, k, dt, c, chi_max)

    rows = []
    for k in range(n):
        F, u, v, sec = _fidelity(circuit, edges, model, t, vecs[k], dt, chi_max, cutoff, cache(k, cutoff), log)
        T = math.sqrt(max(1.0 - F, 0.0))
        bar = math.sqrt(u.discarded) + math.sqrt(v.discarded)
        rows.append({"k": k, "F": F, "infid": 1.0 - F, "infid_lo": max(T - bar, 0.0) ** 2,
                     "infid_hi": (T + bar) ** 2, "discarded_u": u.discarded, "discarded_v": v.discarded,
                     "chi_u": u.chi_peak, "chi_v": v.chi_peak, "seconds": sec})
    F = np.array([r["F"] for r in rows])
    infid = float(1.0 - F.mean())
    lo = float(np.mean([r["infid_lo"] for r in rows]))
    hi = float(np.mean([r["infid_hi"] for r in rows]))
    out = {"N": N, "states": [seed, n], "dt": dt, "chi_max": chi_max, "cutoff": cutoff, "rows": rows,
           "infid": infid, "stderr": float(np.std(1.0 - F, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
           "c_site": float(1.0 - np.mean(F ** (1.0 / N))), "infid_lo": lo, "infid_hi": hi,
           "trunc_rel": float((hi - infid) / infid) if infid > 0 else 0.0,
           "chi_peak": int(max(max(r["chi_u"], r["chi_v"]) for r in rows))}
    if check:
        c2 = cutoff / 100
        F2, u2, v2, _ = _fidelity(circuit, edges, model, t, vecs[0], dt, chi_max, c2, cache(0, c2), log)
        out["trunc_check"] = {"cutoff": c2, "infid": 1.0 - F2, "chi": max(u2.chi_peak, v2.chi_peak),
                              "rel_change": abs((1.0 - F2) - rows[0]["infid"]) / max(1.0 - F2, 1e-300)}
    rel = out["trunc_check"]["rel_change"] if check else 0.0
    out["trunc_flag"] = out["trunc_rel"] > 0.05 or rel > 0.05
    log(f"mps: N = {N}, {n} inputs: 1 - F = {infid:.4e} +- {out['stderr']:.1e} (stat), "
        f"truncation bar {out['trunc_rel']:.1%}" + (f", change at cutoff/100 {rel:.1e}" if check else "")
        + f", per site {out['c_site']:.3e}, chi peak {out['chi_peak']}")
    return out


def _order(label: str) -> float:
    try:
        return float(label)
    except ValueError:
        return math.inf


def evaluate_circuit(circuit: Layered, edges: dict | None, pricing: dict, rotations_per_qubit: float,
                     model: XXZ, t: float, N: int = 100, f_tol: float = 0.01, log=print, **mps_kw) -> dict:
    """MPS infidelity of a realized circuit, its totals with fallbacks, and its T counts.

    ``edges`` is the record of edges.realize_edges (None for circuits that need
    none, like the product formulas, and ignored when its fit was not kept). ``pricing`` maps a fallback-cost label to
    Tq, ledger_per_qubit and ledger2_per_qubit (realize.py's per_F, or a single
    entry for the baseline).
    """
    pairs_ = None
    E = {"T_total": 0.0, "ledger_total": 0.0, "ledger2_total": 0.0, "n_rotations": 0}
    if edges is not None and edges["use"]:
        from snapevo.edges import pairs
        pairs_ = pairs(edges["angles_real"])
        E = edges
    m = mps_infidelity(circuit, pairs_, model, t, N=N, log=log, **mps_kw)
    C = held_out(circuit, model, t, sizes=(10,))["worst"]
    labels = sorted(pricing, key=_order)
    per = {}
    for label in labels:
        pf = pricing[label]
        L = N * pf["ledger_per_qubit"] + E["ledger_total"]
        Q2 = N * pf["ledger2_per_qubit"] / 2 + E["ledger2_total"] / 2
        total = fidelity_bound(m["infid"], L, Q2)
        per[label] = {"Tq": pf["Tq"] + E["T_total"] / N, "ledger": L, "Q2": Q2, "infid_total": total}
    floor = per[labels[-1]]["infid_total"]
    choice = next(lab for lab in labels if per[lab]["infid_total"] <= (1.0 + f_tol) * floor)
    best = per[choice]
    share = max(0.0, (best["infid_total"] - m["infid"]) / best["infid_total"]) if best["infid_total"] > 0 else 0.0
    rot = rotations_per_qubit + E["n_rotations"] / N
    out = {"N": N, "mps": m, "C_ring": C, "infid_ring_pred": 1.0 - (1.0 - C) ** N, "per_F": per, "F": choice,
           "infid": m["infid"], "infid_total": best["infid_total"], "fallback_share": share,
           "fallback_flag": share > 0.1, "c_site": m["c_site"], "ledger": best["ledger"],
           "Tq": best["Tq"], "T_edges_per_qubit": E["T_total"] / N, "rotations_per_qubit": rot,
           "T_per_rotation": best["Tq"] / rot, "T_per_qubit_per_time": best["Tq"] / t}
    log(f"evaluate: N = {N}, F = {choice}: 1 - F = {m['infid']:.3e} (ring C predicts "
        f"{out['infid_ring_pred']:.3e}), with fallbacks {best['infid_total']:.3e} (share {share:.1%}), "
        f"T/q {best['Tq']:.1f} (edges {out['T_edges_per_qubit']:.2f}), T/rot {out['T_per_rotation']:.2f}")
    return out
