"""Sampled fallbacks: the average state infidelity over the heralded branches, not a bound.

Every KN rotation succeeds with probability p and applies U0, which is Rz(theta_s)
up to a phase, or falls back and applies U1 (synthesis.py). evaluate.py measures
the circuit with every rotation on its success branch, and fidelity_bound charges
the fallbacks as if all of them lined up with that error. Here we draw the
branches instead. Each KN rotation in the circuit falls back on its own with
probability 1 - p, the circuit runs with U1 in those places, and the infidelity
is averaged over draws and inputs. Fidelity is linear in the state, so the mean
is an unbiased estimate of the realized channel's average infidelity.

Every slot of a cell is an Rz between Cliffords (cells.py), applied in a fixed
order, single-qubit slots first:

    Rx(e)  = H Rz(e) H                      xxz+x, and Rz(d) x Rz(-d) for xxz+stag
    Rzz(c) = CNOT (1 x Rz(c)) CNOT
    Ryy(b) = (SH x SH) Rzz(b) (HS^dag x HS^dag)
    Rxx(a) = (H x H) Rzz(a) (H x H)         last

and the edge rotations are Rz(a) Ry(b) Rz(c) with Ry(b) = SH Rz(b) HS^dag. A
fallback puts U1 in place of that Rz, or its complex conjugate for the Rz(-d) of
the staggered slot, which runs the conjugate circuit. With every rotation on its
success branch the drawn circuit is the realized circuit of evaluate.py
(tests/test_fallbacks.py).

What is drawn: every KN rotation, so all of the baseline, and the KN stragglers
and edge rotations of SNAP. The library gadgets of SNAP fall back to a KN
correction whose circuit the library file does not carry, so their part stays a
bound, applied on top of the drawn infidelity (snap_total). It is small, since
those corrections are synthesized to about 1e-5.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from snapevo.circuits import Layered, product_vectors
from snapevo.evaluate import MPS_SET, fidelity_bound, ref_path
from snapevo.hamiltonian import XXZ
from snapevo.mps import MPS, reference_state
from snapevo.synthesis import branch_unitaries, synthesize

__all__ = ["slot_gates", "sample_circuit", "sample_baseline", "sample_snap", "snap_total"]

_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_S = np.diag([1, 1j])
_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_V = _S @ _H                                     # V Z V^dag = Y
_VD = _V.conj().T
_I2 = np.eye(2, dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])


def slot_gates(U: np.ndarray) -> dict[str, np.ndarray]:
    """The gate of each slot with the 2x2 U in place of its Rz: 'zz', 'yy', 'xx' (4x4) and 'x', 'y' (2x2)."""
    zz = _CNOT @ np.kron(_I2, U) @ _CNOT
    yy = np.kron(_V, _V) @ zz @ np.kron(_VD, _VD)
    xx = np.kron(_H, _H) @ zz @ np.kron(_H, _H)
    return {"zz": zz, "yy": yy, "xx": xx, "x": _H @ U @ _H, "y": _V @ U @ _VD}


class _Rot:
    """One Rz of the circuit: its success branch and, if it can fail, (q, U1)."""

    __slots__ = ("ok", "q", "fb")

    def __init__(self, theta_s: float, rec: dict | None = None, conj: bool = False):
        self.ok = _rz(theta_s)
        self.q, self.fb = 0.0, None
        if rec is not None and rec.get("U1") is not None and rec["p"] < 1.0:
            _u0, u1 = branch_unitaries(rec)
            self.q, self.fb = 1.0 - rec["p"], (u1.conj() if conj else u1)

    def draw(self, rng: np.random.Generator) -> np.ndarray | None:
        """The fallback unitary if this use falls back, else None (one random number if it can fail)."""
        if self.fb is not None and rng.random() < self.q:
            return self.fb
        return None


def _singles(cell: str, r: dict, U: dict) -> tuple[np.ndarray, np.ndarray]:
    """Single-qubit factor of a bond, with U[name] in place of slot name's Rz where given."""
    if cell == "xxz+x":
        return (_H @ U.get("e", r["e"].ok) @ _H, _H @ U.get("f", r["f"].ok) @ _H)
    return U.get("d+", r["d+"].ok), U.get("d-", r["d-"].ok)


def _gate(cell: str, r: dict, U: dict) -> np.ndarray:
    g1, g2 = _singles(cell, r, U)
    two = (slot_gates(U.get("a", r["a"].ok))["xx"] @ slot_gates(U.get("b", r["b"].ok))["yy"]
           @ slot_gates(U.get("c", r["c"].ok))["zz"])
    return two @ np.kron(g1, g2)


_ORDER = {"xxz+x": ("a", "b", "c", "e", "f"), "xxz+stag": ("a", "b", "c", "d+", "d-")}


def _layer_rots(cell: str, ang: np.ndarray, rec_for: Callable[[int], dict | None]) -> dict:
    """The rotations of one sublayer; rec_for(slot) is the KN record of a slot or None."""
    r = {"a": _Rot(ang[0], rec_for(0)), "b": _Rot(ang[1], rec_for(1)), "c": _Rot(ang[2], rec_for(2))}
    if cell == "xxz+x":
        r["e"], r["f"] = _Rot(ang[3], rec_for(3)), _Rot(ang[4], rec_for(4))
    else:
        rec = rec_for(3)
        r["d+"], r["d-"] = _Rot(ang[3], rec), _Rot(-ang[3], rec, conj=True)
    return r


def _edge_rots(edges: dict | None) -> list[list[_Rot]] | None:
    """Per parity-1 sublayer, the six edge rotations (three on site N-1, three on site 0), or None."""
    if not edges or not edges.get("use"):
        return None
    x = np.asarray(edges["angles_real"], dtype=float)
    rots = [_Rot(a, edges["kn"].get(str(i))) for i, a in enumerate(x)]
    return [rots[6 * j: 6 * j + 6] for j in range(x.size // 6)]


def _zyz(rots: list[_Rot], drawn: list) -> np.ndarray:
    u = [d if d is not None else r.ok for r, d in zip(rots, drawn)]
    return u[0] @ slot_gates(u[1])["y"] @ u[2]


def _run(psi: MPS, real: Layered, plan: list[dict], edge_plan, rng: np.random.Generator) -> int:
    """Apply one draw of the branches to psi in place; returns the number of fallbacks."""
    N, cell = psi.N, real.cell.name
    order = _ORDER[cell]
    n_fb = 0
    j_odd = 0
    for layer in plan:
        r, par = layer["rots"], layer["parity"]
        if par == 1:                                       # the lost wrap bond
            if edge_plan is None:                          # the cell's own factor, with its KN slots
                U = {}
                for name in order[3:]:
                    d = r[name].draw(rng)
                    if d is not None:
                        U[name] = d
                        n_fb += 1
                g_last, g_first = _singles(cell, r, U)
            else:
                six = edge_plan[j_odd]
                drawn = [e.draw(rng) for e in six]
                n_fb += sum(d is not None for d in drawn)
                g_last, g_first = _zyz(six[:3], drawn[:3]), _zyz(six[3:], drawn[3:])
            j_odd += 1
            psi.apply_1q(g_last, N - 1)
            psi.apply_1q(g_first, 0)
        bonds = list(range(par, N - 1, 2))
        if layer["field"]:                                 # no coupling: single-qubit layer
            for i in bonds:
                U = {}
                for name in order[3:]:
                    d = r[name].draw(rng)
                    if d is not None:
                        U[name] = d
                        n_fb += 1
                g1, g2 = _singles(cell, r, U)
                psi.apply_1q(g1, i)
                psi.apply_1q(g2, i + 1)
            continue
        left = psi.center > N // 2
        for i in (reversed(bonds) if left else bonds):
            U = {}
            for name in order:
                d = r[name].draw(rng)
                if d is not None:
                    U[name] = d
            n_fb += len(U)
            psi.apply_2q(_gate(cell, r, U) if U else layer["gate"], i, move="left" if left else "right")
    return n_fb


def sample_circuit(real: Layered, rec_for: Callable[[int, int], dict | None], edges: dict | None, model: XXZ,
                   t: float, N: int = 100, states: tuple[int, int] = MPS_SET, samples: int = 4, seed: int = 2718,
                   dt: float = 0.2, chi_max: int = 1024, cutoff: float = 1e-12, cache_dir=None,
                   success_only: bool = False, log=print) -> dict:
    """Average infidelity of a realized circuit over drawn KN fallbacks, at N sites by MPS.

    ``real`` is the success-branch circuit, rec_for(layer, slot) the KN record
    (with branch unitaries) of a slot or None for an exact angle, and ``edges``
    the edge record of edges.realize_edges or None. Draw j of input k uses the
    generator seeded with (seed, k, j), so a rerun gives the same draws.
    ``success_only`` keeps every rotation on its success branch (a check).
    """
    cell = real.cell.name
    if cell not in _ORDER:
        raise ValueError(f"no slot decomposition for cell {cell}")
    nofail = (lambda _s: None)
    plan = []
    for l, (par, ang) in enumerate(real.layers):
        rots = _layer_rots(cell, np.asarray(ang, dtype=float), nofail if success_only else (lambda s: rec_for(l, s)))
        field = not np.any(np.asarray(ang[:3], dtype=float))
        plan.append({"parity": par, "field": field, "rots": rots, "gate": None if field else _gate(cell, rots, {})})
    edge_plan = _edge_rots({**edges, "kn": {}} if success_only and edges else edges)
    s0, n = states
    vecs = product_vectors(N, n, s0)
    t0 = time.time()
    F = np.zeros((n, samples))
    n_fb = np.zeros((n, samples), dtype=int)
    discarded = 0.0
    for k in range(n):
        ref_cache = None if cache_dir is None else ref_path(cache_dir, N, s0, k, dt, cutoff, chi_max)
        v, _ = reference_state(model, t, vecs[k], dt, chi_max, cutoff, ref_cache, log)
        for j in range(samples):
            u = MPS.product(vecs[k], chi_max=chi_max, cutoff=cutoff)
            n_fb[k, j] = _run(u, real, plan, edge_plan, np.random.default_rng((seed, k, j)))
            F[k, j] = abs(v.overlap(u)) ** 2 / abs(v.overlap(v) * u.overlap(u))
            discarded = max(discarded, u.discarded)
    infid = 1.0 - F
    per_input = F.mean(axis=1)
    out = {"N": N, "states": [s0, n], "samples": samples, "seed": seed, "infid": float(infid.mean()),
           "stderr": float(infid.std(ddof=1) / math.sqrt(infid.size)) if infid.size > 1 else 0.0,
           "c_site": float(1.0 - np.mean(per_input ** (1.0 / N))),
           "fallbacks_mean": float(n_fb.mean()), "discarded_max": discarded,
           "infid_draws": infid.tolist(), "seconds": time.time() - t0}
    log(f"sampled: N = {N}, {n} inputs x {samples} draws: 1 - F = {out['infid']:.4e} +- {out['stderr']:.1e}, "
        f"per site {out['c_site']:.3e}, {out['fallbacks_mean']:.1f} fallbacks per run, {out['seconds']:.0f}s")
    return out


def sample_baseline(circ: Layered, real: Layered, synth: dict, eps: float, model: XXZ, t: float,
                    **kw) -> dict:
    """Draw the fallbacks of a realized product formula (baseline.realize_baseline).

    ``circ`` is the product formula and ``synth`` its KN records by angle.
    Records without branch unitaries are synthesized again at ``eps`` to get them.
    """
    synth = {k: (v if "U0" in v else synthesize(float(k), eps, unitaries=True)) for k, v in synth.items()}

    def rec_for(l, s):
        return synth.get(f"{float(round(circ.layers[l][1][s], 15)):.15f}")

    return sample_circuit(real, rec_for, None, model, t, **kw)


def sample_snap(rec: dict, cell, model: XXZ, t: float, **kw) -> dict:
    """Draw the KN fallbacks (stragglers and edge rotations) of a realized SNAP circuit (real_*.json)."""
    x = np.asarray(rec["angles_compiled"], dtype=float)
    kn = {int(i): (r if "U0" in r else synthesize(float(x[int(i)]), r["eps"], unitaries=True))
          for i, r in rec["kn"].items()}
    edges = rec.get("edges")
    if edges and edges.get("use"):
        xf = np.asarray(edges["angles"], dtype=float)
        edges = {**edges, "kn": {i: (r if "U0" in r else synthesize(float(xf[int(i)]), edges["eps"], unitaries=True))
                                 for i, r in edges["kn"].items()}}
    na = cell.n_angles

    def rec_for(l, s):
        return kn.get(l * na + s)

    return sample_circuit(Layered.brickwork(cell, rec["angles_real"]), rec_for, edges, model, t, **kw)


def snap_total(rec: dict, F: str, infid_drawn: float, N: int) -> float:
    """SNAP's infidelity with fallbacks: the drawn KN part plus the bound for the library gadgets at F."""
    mult = np.asarray(rec["multiplicity"], dtype=float)
    delta = rec["per_F"][F]["delta_fb"]
    L = L2 = 0.0
    for i, row in rec["lattice"].items():
        q = row["q"][F]
        L += 0.5 * mult[int(i)] * q * delta
        L2 += 0.5 * mult[int(i)] * q * delta ** 2
    return fidelity_bound(infid_drawn, N * L, N * L2 / 2)
