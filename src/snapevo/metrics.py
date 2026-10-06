"""Error metrics of a realized circuit (docs/error_metric.md has the definitions).

Two kinds of error are measured on the success-branch unitary U at ring size N,
against V = e^{-iHt}.

Worst case. If the eigenphases of W = V^dag U span an arc of width w (after
removing the global phase), then

    operator norm  eps_op(N) = min_phi ||U - e^{i phi} V|| = 2 sin(w/4)
    diamond        D_coh(N)  = || U.U^dag - V.V^dag ||_<> = 2 sin(w/2).

w comes from Lanczos on G = Im(e^{-i phi0} W), whose extreme eigenvalues are the
sines of the arc ends. w is extensive, so it is extrapolated in N with an affine
and a power-law fit, and the larger prediction is kept.

Average case (fidelity). The state infidelity on random product inputs is
1 - (1 - C)^N, with C the held-out per-site infidelity, and the process
(entanglement) infidelity is 1 - |tr(W)/d|^2, measured at each size (exactly up
to N = 10, by 16 Haar-random states above) and extrapolated through its per-site
value. Both fidelities are multiplicative over sites, so their per-site values
do not depend on N once N exceeds the light cone.

The heralded fallbacks enter every metric through the ledger of realize.py.
They add N l to the diamond error. For fidelities they enter the bound
1 - F <= (sqrt(1 - F_s) + L/2)^2 + Q2 (fidelity_bound).
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse.linalg as spla

from snapevo.cells import Cell
from snapevo.circuits import HELD_OUT_SETS, Layered, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ, evolve

__all__ = ["held_out", "arc_width", "arc_width_dense", "process_fidelity", "extrapolate",
           "fidelity_bound", "measure", "totals", "evaluate"]


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


def process_fidelity(circuit: Layered, model: XXZ, t: float, N: int, exact_max: int = 10,
                     n_haar: int = 16, seed: int = 7) -> dict:
    """Entanglement fidelity F_e = |tr(V^dag U)/d|^2 at size N, and its per-site infidelity.

    Exact for N <= exact_max (the circuit and e^{-iHt} applied to every basis
    state). Above that, the average state fidelity over n_haar Haar-random
    states estimates F_avg, and F_e = ((d + 1) F_avg - 1)/d.
    """
    t0 = time.time()
    d = 2 ** N
    H = model.sparse(N)
    if N <= exact_max:
        eye = np.eye(d, dtype=complex)
        tr = np.sum(np.conj(evolve(H, eye, t)) * circuit.apply(eye, N))
        Fe, stderr, method = float(abs(tr / d) ** 2), 0.0, "exact"
    else:
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n_haar, d)) + 1j * rng.normal(size=(n_haar, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = np.abs(np.sum(np.conj(evolve(H, v, t)) * circuit.apply(v, N), axis=1)) ** 2
        Fe = float(((d + 1) * f.mean() - 1) / d)
        stderr = float((d + 1) / d * f.std(ddof=1) / math.sqrt(n_haar))
        method = f"haar{n_haar}"
    return {"N": N, "F_e": Fe, "infid": 1.0 - Fe, "c_e": 1.0 - max(Fe, 1e-300) ** (1.0 / N),
            "stderr": stderr, "method": method, "seconds": time.time() - t0}


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


def fidelity_bound(infid_success: float, L: float, Q2: float) -> float:
    """Infidelity of the realized channel given the success-branch infidelity.

    L = sum_i q_i delta_i is the ledger and Q2 = sum_i q_i delta_i^2 / 2. Each
    fallback moves the output by at most delta_i/2 in trace distance, so for
    every branch pattern sqrt(1 - F_b) <= sqrt(1 - F_s) + X_b with X_b the sum of
    its delta_i/2, and averaging (1 - F) <= E(sqrt(1 - F_s) + X)^2 gives this.
    Var X <= Q2 needs q_i <= 1, which holds since every library gadget has p >= 0.9.
    """
    return min(1.0, (math.sqrt(max(infid_success, 0.0)) + L / 2) ** 2 + Q2)


def measure(circuit: Layered, model: XXZ, t: float, sizes=(6, 8, 10, 12, 14), N_target: int = 100,
            held_sizes=(10, 12), log=print) -> dict:
    """Success-branch metrics of one circuit: worst case, fidelities, and their N = N_target values."""
    ho = held_out(circuit, model, t, held_sizes)
    log(f"measure: held-out C worst {ho['worst']:.3e}")
    arcs, proc = [], []
    for N in sizes:
        r = arc_width(circuit, model, t, N)
        p = process_fidelity(circuit, model, t, N)
        arcs.append(r)
        proc.append(p)
        log(f"measure: N = {N:2d}  w = {r['w']:.4e}  eps_op = {r['eps_op']:.3e}  "
            f"process infid = {p['infid']:.3e} ({p['method']})  ({r['seconds'] + p['seconds']:.0f}s)")
    ext = extrapolate([r["N"] for r in arcs], [r["w"] for r in arcs], N_target)
    big = max(arcs, key=lambda r: r["N"])
    c_e = max(proc, key=lambda p: p["N"])["c_e"]
    C = ho["worst"]
    fid = {"C": C, "c_e": c_e,
           "state_infid_10": 1 - (1 - C) ** 10, "state_infid": 1 - (1 - C) ** N_target,
           "process_infid": 1 - (1 - c_e) ** N_target}
    return {"held_out": ho, "arcs": arcs, "process": proc, "alpha": big["w"] / big["N"],
            "extrapolation": ext, "fidelity": fid, "N_target": N_target}


def totals(meas: dict, ledger_per_qubit: float, ledger2_per_qubit: float) -> dict:
    """Bounds on the realized channel at N_target, heralded fallbacks included."""
    N = meas["N_target"]
    L = N * ledger_per_qubit
    Q2 = N * ledger2_per_qubit / 2
    f = meas["fidelity"]
    return {"ledger": L, "Q2": Q2,
            "D_total": min(2.0, meas["extrapolation"]["D_coh"] + L),
            "state_infid_total": fidelity_bound(f["state_infid"], L, Q2),
            "process_infid_total": fidelity_bound(f["process_infid"], L, Q2)}


def evaluate(realized: dict, model: XXZ, cell: Cell, t: float, sizes=(6, 8, 10, 12, 14),
             N_target: int = 100, ledger_frac: float = 0.1, held_sizes=(10, 12), log=print) -> dict:
    """Every metric of a realized circuit, and the fallback cost F it should be priced at.

    F is the smallest grid value at which the fallbacks are at most
    ``ledger_frac`` of the total, both in diamond norm and in state infidelity.
    """
    meas = measure(Layered.brickwork(cell, realized["angles_real"]), model, t, sizes, N_target,
                   held_sizes, log)
    s_inf = meas["fidelity"]["state_infid"]
    rows, choice = {}, None
    for F, pf in realized["per_F"].items():
        tot = totals(meas, pf["ledger_per_qubit"], pf["ledger2_per_qubit"])
        rows[F] = {"Tq": pf["Tq"], **tot}
        ok = (tot["ledger"] <= ledger_frac * tot["D_total"]
              and tot["state_infid_total"] - s_inf <= ledger_frac * tot["state_infid_total"])
        if choice is None and ok:
            choice = F
    if choice is None:
        choice = max(realized["per_F"], key=float)
    best = rows[choice]
    rot = realized["rotations_per_qubit"]
    out = {**meas, "per_F": rows, "F": float(choice), **{k: best[k] for k in best},
           "D_coh": meas["extrapolation"]["D_coh"],
           "T_per_rotation": best["Tq"] / rot, "T_per_qubit_per_time": best["Tq"] / t}
    log(f"evaluate: N = {N_target}, F = {choice}: D_total {best['D_total']:.3e} (ledger {best['ledger']:.1e}), "
        f"state infid {best['state_infid_total']:.3e}, process infid {best['process_infid_total']:.3e}; "
        f"T/q {best['Tq']:.1f}, T/rot {out['T_per_rotation']:.2f}")
    return out
