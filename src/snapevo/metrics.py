"""Error metrics of a realized circuit (docs/error_metric.md has the definitions).

For the success-branch unitary U at size N and the exact V = e^{-iHt}, let the
eigenphases of W = V^dag U span an arc of width w (after removing the global
phase). Then

    operator norm  eps_op(N) = min_phi ||U - e^{i phi} V|| = 2 sin(w/4)
    diamond        D_coh(N)  = || U.U^dag - V.V^dag ||_<> = 2 sin(w/2)

computed exactly by Lanczos on G = Im(e^{-i phi0} W), whose extreme eigenvalues
are sin of the arc ends. w is extensive (local errors add over the ring), so it
is extrapolated in N with an affine and a power-law fit, and the larger of the
two predictions is reported. The heralded fallbacks add N x ledger (realize.py)
to the diamond error, rigorously.
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse.linalg as spla

from snapevo.cells import Cell
from snapevo.circuits import HELD_OUT_SETS, Layered, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ

__all__ = ["held_out", "arc_width", "arc_width_dense", "extrapolate", "evaluate"]


def held_out(circuit: Layered, model: XXZ, t: float, sizes=(10, 12)) -> dict:
    """C on the four held-out sets at each size, and the worst value."""
    out = {}
    for N in sizes:
        vals = []
        for seed, n in HELD_OUT_SETS:
            tin, tg = state_pairs(model, N, t, n, seed)
            vals.append(local_infidelity(circuit.apply(tin, N), tg, N))
        out[str(N)] = vals
    out["worst"] = max(max(v) for v in out.values())
    return out


def arc_width_dense(circuit: Layered, model: XXZ, t: float, N: int) -> float:
    """w from a full diagonalization (N <= 12; used to test the Lanczos path)."""
    import scipy.linalg as sla
    U = circuit.apply(np.eye(2 ** N, dtype=complex), N).T            # columns = U|e_k>
    V = sla.expm(-1j * t * model.sparse(N).toarray())
    ph = np.angle(np.linalg.eigvals(V.conj().T @ U))
    z = np.exp(1j * ph)
    phi0 = np.angle(np.mean(z))
    d = np.angle(z * np.exp(-1j * phi0))
    return float(d.max() - d.min())


def arc_width(circuit: Layered, model: XXZ, t: float, N: int, tol: float = 1e-10,
              seed: int = 0) -> dict:
    """w at size N by Lanczos, matrix free. Returns w, phi0, matvecs, seconds."""
    t0 = time.time()
    H = model.sparse(N)
    dim = 2 ** N

    def W(v):
        return spla.expm_multiply(1j * t * H, circuit.apply(v.reshape(1, -1), N).ravel(), traceA=0.0)

    def Wdag(v):
        return circuit.apply(spla.expm_multiply(-1j * t * H, v, traceA=0.0).reshape(1, -1), N, dagger=True).ravel()

    rng = np.random.default_rng(seed)
    acc = 0j
    for _ in range(4):
        v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        v /= np.linalg.norm(v)
        acc += np.vdot(v, W(v))
    phi0 = float(np.angle(acc))
    e = np.exp(-1j * phi0)
    count = [0]

    def G(v):
        count[0] += 1
        v = np.asarray(v).ravel()
        return (e * W(v) - np.conj(e) * Wdag(v)) / 2j

    op = spla.LinearOperator((dim, dim), matvec=G, dtype=complex)
    hi = spla.eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False, maxiter=5000)[0]
    lo = spla.eigsh(op, k=1, which="SA", tol=tol, return_eigenvectors=False, maxiter=5000)[0]
    w = math.asin(min(1.0, float(hi))) - math.asin(max(-1.0, float(lo)))
    return {"N": N, "w": w, "eps_op": 2 * math.sin(w / 4), "D_coh": 2 * math.sin(min(w, math.pi) / 2),
            "phi0": phi0, "matvecs": count[0], "seconds": time.time() - t0}


def extrapolate(Ns, ws, N_target: int, n_fit: int = 3) -> dict:
    """Extrapolate the arc width to N_target from the largest n_fit sizes.

    Affine w = alpha N + beta (the large-N form of a sum of local errors) and a
    power law w = c N^a. The reported prediction is the larger of the two.
    """
    Ns = np.asarray(Ns, dtype=float)
    ws = np.asarray(ws, dtype=float)
    idx = np.argsort(Ns)[-n_fit:]
    x, y = Ns[idx], ws[idx]
    alpha, beta = np.polyfit(x, y, 1)
    a, logc = np.polyfit(np.log(x), np.log(y), 1)
    w_aff = float(alpha * N_target + beta)
    w_pow = float(math.exp(logc) * N_target ** a)
    w = max(w_aff, w_pow)
    return {"N_target": N_target, "fit_sizes": x.tolist(), "affine": [float(alpha), float(beta)],
            "power": [float(math.exp(logc)), float(a)], "w_affine": w_aff, "w_power": w_pow, "w": w,
            "eps_op": 2 * math.sin(min(w, 2 * math.pi) / 4), "D_coh": 2 * math.sin(min(w, math.pi) / 2)}


def evaluate(realized: dict, model: XXZ, cell: Cell, t: float, sizes=(6, 8, 10, 12, 14),
             N_target: int = 100, ledger_frac: float = 0.1, held_sizes=(10, 12), log=print) -> dict:
    """Every metric of a realized circuit, and the fallback cost F it should be priced at.

    F is the smallest grid value whose ledger at N_target is at most
    ``ledger_frac`` of the total diamond bound there.
    """
    circuit = Layered.brickwork(cell, realized["angles_real"])
    ho = held_out(circuit, model, t, held_sizes)
    log(f"evaluate: held-out C worst {ho['worst']:.3e}")
    arcs = []
    for N in sizes:
        r = arc_width(circuit, model, t, N)
        arcs.append(r)
        log(f"evaluate: N = {N:2d}  w = {r['w']:.4e}  eps_op = {r['eps_op']:.3e}  ({r['seconds']:.0f}s)")
    ext = extrapolate([r["N"] for r in arcs], [r["w"] for r in arcs], N_target)
    choice = None
    rows = {}
    for F, pf in realized["per_F"].items():
        led = N_target * pf["ledger_per_qubit"]
        total = ext["D_coh"] + led
        rows[F] = {"Tq": pf["Tq"], "ledger": led, "D_total": total}
        if choice is None and led <= ledger_frac * total:
            choice = F
    if choice is None:
        choice = max(realized["per_F"], key=float)
    best = rows[choice]
    rot = realized["rotations_per_qubit"]
    big = max(arcs, key=lambda r: r["N"])
    out = {
        "held_out": ho,
        "arcs": arcs,
        "alpha": big["w"] / big["N"],
        "extrapolation": ext,
        "per_F": rows,
        "F": float(choice),
        "N_target": N_target,
        "D_total": best["D_total"],
        "D_coh": ext["D_coh"],
        "ledger": best["ledger"],
        "Tq": best["Tq"],
        "T_per_rotation": best["Tq"] / rot,
        "T_per_qubit_per_time": best["Tq"] / t,
    }
    log(f"evaluate: N = {N_target}: D_coh {ext['D_coh']:.3e} + ledger {best['ledger']:.2e} "
        f"= {best['D_total']:.3e} at F = {choice}; T/q {best['Tq']:.1f}, T/rot {out['T_per_rotation']:.2f}")
    return out
