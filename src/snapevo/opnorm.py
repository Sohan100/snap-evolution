"""Operator-norm error of a circuit against exact e^{-iHt}, and the per-site infidelity of the same unitary.

The paper reports the per-site state infidelity c = 1 - mean_k F_k^{1/N} on random product inputs
(evaluate.py). The literature reports the operator norm. This module measures both on the same
success-branch unitary U at the same size N, against the exact V = e^{-iHt}, on a ring or on an open
chain, so the two can be related directly (cluster/opnorm_run.py runs it over the sweep).

Worst case. Let W = V^dag U have eigenphases theta_k, spanning a shortest arc of width w. Then

    eps_op(N) = min_phi ||U - e^{i phi} V|| = 2 sin(w/4),        D_coh(N) = 2 sin(min(w, pi)/2),

the second being the diamond distance of the two unitary channels. w comes from Lanczos on
G = Im(e^{-i phi0} W) = (e^{-i phi0} W - e^{i phi0} W^dag)/2i, whose extreme eigenvalues are the sines of
the arc ends measured from phi0: w = asin(lambda_max) - asin(lambda_min). This is crosscheck.arc_width,
with its conventions (phi0 the phase of the mean of <v|W|v> over 4 random vectors, seed 0; residual
tolerance 1e-10 relative), generalized to open chains and to any gate list. The sine map is right only
while every eigenphase lies within pi/2 of phi0. The Ritz vectors give the true phases of the two ends
(the phase of <x|W|x>); if an end lies more than 1.2 rad from phi0, phi0 moves to the middle of the arc and
the Lanczos run is repeated. A measured arc of pi - 0.05 or more, or an end still beyond pi/2 - 0.05, is
flagged ``saturated``: there the sines cannot resolve the arc, and eps_op >= 2 sin(pi/4) is all that is
known. For arcs above 1 rad a second Lanczos run on Re(e^{-i phi_c} W) (phi_c the arc's middle) gives the
largest distance of any eigenphase from phi_c, acos(lambda_min), which must equal w/2: phases hidden in
the back half of the circle would show up there (``back_check``).

Exact V. e^{-iHt} acts on blocks of vectors by a Chebyshev expansion in H (sparse, row blocks on
``threads`` threads), truncated where the remaining Bessel coefficients sum below 1e-15, and checked
against scipy's expm_multiply on a random vector (``prop_check``). It is exact to rounding. At N = 16,
t = 4 it costs 0.43 s per vector on one thread (0.13 s on four) against 2.7 s for one expm_multiply call,
which dominated crosscheck.arc_width. One Lanczos run converges both arc ends, with about half of
crosscheck's ARPACK matvecs (measured 15 Oct 2026 on login08).

Geometry. On the ring every parity-1 sublayer of a Layered circuit acts on the wrap bond (N-1, 0). On
the open chain that gate is replaced by the edge gates, the circuit's fitted edge pairs or the cell's own
single-qubit factor, exactly as Layered.apply(bc="open", edges=...) and mps.apply_layered do, and H is
the open-chain Hamiltonian. A generic gate list (pairs (sites, unitary), or objects with ``sites`` and
``matrix()`` such as baseline_lit.Gate) is applied as given, at the size it was built for or through a
builder (N, bc) -> gates.

Average case, same unitary and N. C(N) = 1 - mean_k |<V phi_k|U phi_k>|^{2/N} over 64 random product
inputs (circuits.HELD_OUT_SETS[1], seed 777: circuits.local_infidelity's formula and draws), with its
standard error. The Haar values from 16 Haar-random states (seed 7, crosscheck.process_fidelity's draws):
c_haar = 1 - mean f^{1/N}, and c_e = 1 - F_e^{1/N} with the entanglement fidelity
F_e = ((d + 1) mean f - 1)/d. Rigorous link: for every state ||(U - e^{i phi} V) psi||^2 >= 1 - F(psi), so
eps_op >= sqrt(max_k (1 - F_k)) >= sqrt(1 - (1 - C)^N), checked at every size (``floor_ok``). Nothing
bounds eps_op from above by fidelities: the relation eps_op ~ (kappa/2) N sqrt(C) is empirical.

Exact path. dense_metrics diagonalizes W itself for N <= 10 (all 2^N eigenphases, the shortest arc from
the largest gap, F_e = |tr W/d|^2); the tests check the Lanczos path against it.

Extrapolation. w is extensive, w(N) = alpha N + beta on the open chain (beta is the boundary). extrapolate
fits it on N = 10, 12, 14 (and 16 when present) and predicts w, eps_op and D_coh at N = 50 and 100,
flagging w >= pi, with the rigorous floor sqrt(1 - (1 - c)^N) <= eps_op.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.special import jv

from snapevo.circuits import HELD_OUT_SETS, Layered, product_vectors
from snapevo.hamiltonian import XXZ

__all__ = ["PRODUCT_SET", "HAAR_SET", "Circuit", "as_circuit", "layered_ops", "gate_ops", "apply_ops",
           "Propagator", "chebyshev_terms", "eps_of_w", "dcoh_of_w", "floor_eps", "kappa", "arc",
           "dense_unitaries", "dense_metrics", "infidelities", "measure", "extrapolate", "snap_circuit",
           "baseline_circuit", "trotter", "open_trained_circuit"]

PRODUCT_SET: tuple[int, int] = HELD_OUT_SETS[1]
"""(seed, n) of the product inputs of C(N): circuits.HELD_OUT_SETS[1], the set baseline.scan uses."""

HAAR_SET: tuple[int, int] = (7, 16)
"""(seed, n) of the Haar-random states, crosscheck.process_fidelity's draws."""

RECENTER_AT = 1.2          # rad: an arc end further than this from phi0 moves phi0 to the arc's middle
SATURATION_MARGIN = 0.05   # rad: arcs within this of pi (or ends within it of pi/2) are flagged saturated
BACK_CHECK_ABOVE = 1.0     # rad: arcs wider than this get the back-half check

Op = tuple[tuple[int, ...], np.ndarray]


# --------------------------------------------------------------------------- circuits as gate lists

def layered_ops(circuit: Layered, N: int, bc: str = "ring", edges=None) -> list[Op]:
    """The gates of a Layered circuit at N sites in time order, as (sites, unitary).

    Ring: parity-1 sublayers act on the wrap bond (N-1, 0), first qubit N-1. Open chain: that gate is
    replaced by the edge pair (gate on N-1, gate on 0) of circuit.edge_pairs(edges), as in
    Layered.apply(bc="open", edges=edges).
    """
    if N % 2:
        raise ValueError("the brickwork needs an even number of sites")
    if bc not in ("ring", "open"):
        raise ValueError(f"bc must be 'ring' or 'open', not {bc!r}")
    wrap = iter(circuit.edge_pairs(edges)) if bc == "open" else None
    out: list[Op] = []
    for parity, angles in circuit.layers:
        U = circuit.cell.gate(angles)
        g_last = g_first = None
        if wrap is not None and parity == 1:
            g_last, g_first = next(wrap)
        for i in range(parity, N, 2):
            if i == N - 1 and bc == "open":
                out += [((N - 1,), np.asarray(g_last, dtype=complex)), ((0,), np.asarray(g_first, dtype=complex))]
            else:
                out.append(((i, (i + 1) % N), U))
    return out


def gate_ops(gates, N: int) -> list[Op]:
    """A generic gate list as (sites, unitary): pairs (sites, U) or objects with .sites and .matrix()."""
    out: list[Op] = []
    for g in gates:
        if hasattr(g, "sites") and hasattr(g, "matrix"):
            sites, U = tuple(int(s) for s in g.sites), np.asarray(g.matrix(), dtype=complex)
        else:
            sites, U = tuple(int(s) for s in g[0]), np.asarray(g[1], dtype=complex)
        if len(sites) not in (1, 2) or U.shape != (2 ** len(sites),) * 2:
            raise ValueError(f"gate on {sites} has shape {U.shape}")
        if any(s < 0 or s >= N for s in sites) or len(set(sites)) != len(sites):
            raise ValueError(f"gate on sites {sites} does not fit {N} sites")
        out.append((sites, U))
    return out


@dataclass
class Circuit:
    """A circuit at any even N: a Layered brickwork (with open-chain edge pairs), or a gate list.

    ``gates`` is a fixed gate list (valid at one size, ``N_fixed``) or a builder (N, bc) -> gate list.
    ``bcs`` lists the boundaries it is defined on.
    """

    layered: Layered | None = None
    edges: list | None = None
    gates: Callable | list | None = None
    N_fixed: int | None = None
    bcs: tuple[str, ...] = ("ring", "open")
    name: str = ""
    info: dict = field(default_factory=dict)

    def ops(self, N: int, bc: str) -> list[Op]:
        if bc not in self.bcs:
            raise ValueError(f"{self.name or 'this circuit'} is defined on {self.bcs}, not {bc!r}")
        if self.layered is not None:
            return layered_ops(self.layered, N, bc, self.edges if bc == "open" else None)
        if callable(self.gates):
            return gate_ops(self.gates(N, bc), N)
        if self.N_fixed is not None and N != self.N_fixed:
            raise ValueError(f"this gate list was built for N = {self.N_fixed}, not {N}")
        return gate_ops(self.gates, N)

    @property
    def depth(self) -> int | None:
        return self.layered.depth if self.layered is not None else None


def as_circuit(circuit, edges=None) -> Circuit:
    """Circuit from a Circuit, a Layered (with open-chain edges), or a gate list or builder."""
    if isinstance(circuit, Circuit):
        return circuit
    if isinstance(circuit, Layered):
        return Circuit(layered=circuit, edges=edges)
    return Circuit(gates=circuit)


def apply_ops(ops: list[Op], X: np.ndarray, N: int, dagger: bool = False) -> np.ndarray:
    """Apply gates to a block of states stored as columns, shape (2^N, n), qubit 0 most significant.

    Same convention as circuits.apply_2q: a two-qubit gate on (i, j) has its first qubit on site i.
    (Splitting the gates over threads gained nothing on login01: they are memory bound.)
    """
    X = np.asarray(X)
    dim = 2 ** N
    if X.shape[0] != dim:
        raise ValueError(f"states have {X.shape[0]} rows, not 2^{N}")
    n = X.shape[1]
    psi = np.array(X, dtype=complex, order="C", copy=True)
    buf = np.empty_like(psi)
    for sites, U in (reversed(ops) if dagger else ops):
        if dagger:
            U = U.conj().T
        if len(sites) == 1:
            i, = sites
            np.matmul(U, psi.reshape(2 ** i, 2, -1), out=buf.reshape(2 ** i, 2, -1))
        elif sites[1] == sites[0] + 1:
            i = sites[0]
            np.matmul(U, psi.reshape(2 ** i, 4, -1), out=buf.reshape(2 ** i, 4, -1))
        elif sites == (N - 1, 0):                         # the ring's wrap bond, first qubit on N-1
            P = psi.reshape(2, 2 ** (N - 2), 2, n)          # (q0, middle, q_{N-1}, column)
            O = buf.reshape(2, 2 ** (N - 2), 2, n)
            U4 = U.reshape(2, 2, 2, 2)                       # [a', b', a, b]: a on N-1, b on 0
            for ap in (0, 1):
                for bp in (0, 1):
                    O[bp, :, ap, :] = (U4[ap, bp, 0, 0] * P[0, :, 0, :] + U4[ap, bp, 0, 1] * P[1, :, 0, :]
                                       + U4[ap, bp, 1, 0] * P[0, :, 1, :] + U4[ap, bp, 1, 1] * P[1, :, 1, :])
        else:                                              # any other pair
            i, j = sites
            t = psi.reshape((2,) * N + (n,))
            r = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [i, j]))
            buf[...] = np.moveaxis(r, [0, 1], [i, j]).reshape(dim, n)
        psi, buf = buf, psi
    return psi


# --------------------------------------------------------------------------- exact e^{-iHt}

_POOLS: dict[int, ThreadPoolExecutor] = {}


def _pool(threads: int) -> ThreadPoolExecutor:
    if threads not in _POOLS:
        _POOLS[threads] = ThreadPoolExecutor(threads)
    return _POOLS[threads]


def default_threads() -> int:
    """Threads for the sparse products: OMP_NUM_THREADS, else the CPUs this process may use (at most 8)."""
    v = os.environ.get("OMP_NUM_THREADS")
    if v and v.isdigit():
        return max(1, int(v))
    try:
        return max(1, min(8, len(os.sched_getaffinity(0))))
    except AttributeError:                                  # pragma: no cover (not Linux)
        return 1


def chebyshev_terms(x: float, tol: float = 1e-15) -> tuple[int, np.ndarray]:
    """(K, J_0..J_K(x)): the last order kept in the Chebyshev series of e^{-i x Ht}, where the rest sums below tol."""
    kmax = int(x + 80 + 6 * x ** (1 / 3))
    J = jv(np.arange(kmax + 1), x)
    w = np.abs(J) * 2.0
    w[0] = abs(J[0])
    tail = np.cumsum(w[::-1])[::-1]                           # tail[k] = sum of the terms from k on
    if tail[-1] > 1e-30:
        raise RuntimeError(f"Chebyshev series not resolved at a t = {x:.1f}")
    K = int(np.flatnonzero(tail < tol)[0]) - 1                # keep terms 0..K
    return K, J[: K + 1]


class Propagator:
    """e^{-iHt} and e^{+iHt} on blocks of column vectors, by a Chebyshev expansion in H.

    H = a Ht + b with the spectrum of Ht inside [-1, 1] (bounds from Lanczos, padded); then
    e^{s i H t} = e^{s i b t} sum_k eps_k (s i)^k J_k(a t) T_k(Ht), eps_0 = 1, eps_k = 2, truncated where
    the remaining terms sum below ``tol``. Columns may carry different signs s.
    """

    def __init__(self, model: XXZ, N: int, bc: str, t: float, tol: float = 1e-15, threads: int | None = None,
                 H=None):
        t0 = time.time()
        self.N, self.bc, self.t, self.dim = N, bc, float(t), 2 ** N
        self.H = (model.sparse(N, bc) if H is None else H).tocsr()
        self.H.sort_indices()
        Hb = self.H.real if not np.any(self.H.imag.data) else self.H
        if self.dim <= 64:
            ev = np.linalg.eigvalsh(Hb.toarray())
            lo, hi = float(ev[0]), float(ev[-1])
        else:
            lo = float(spla.eigsh(Hb, k=1, which="SA", tol=1e-8, return_eigenvectors=False)[0])
            hi = float(spla.eigsh(Hb, k=1, which="LA", tol=1e-8, return_eigenvectors=False)[0])
        span = hi - lo
        pad = 1e-6 * span + 1e-9 * max(abs(hi), abs(lo), 1.0)
        self.bounds = (lo, hi)
        self.a = 0.5 * span + pad
        self.b = 0.5 * (hi + lo)
        self.K, J = chebyshev_terms(self.a * self.t, tol)
        self.J = J
        self.tol = tol
        self.threads = threads or default_threads()
        n_blk = self.threads if self.dim >= 2 ** 13 else 1
        edges_ = np.linspace(0, self.dim, n_blk + 1).astype(int)
        self.blocks = [(int(r0), int(r1), self.H[r0:r1]) for r0, r1 in zip(edges_[:-1], edges_[1:])]
        self.applications = 0                                 # vectors propagated
        self.setup_seconds = time.time() - t0

    def apply(self, X: np.ndarray, signs) -> np.ndarray:
        """Columns of X propagated by e^{s i H t}, s = signs[col] in {-1, +1} (-1 is e^{-iHt})."""
        X = np.ascontiguousarray(X, dtype=complex)
        if X.ndim != 2 or X.shape[0] != self.dim:
            raise ValueError(f"X must be (2^{self.N}, m), not {X.shape}")
        m = X.shape[1]
        s = np.broadcast_to(np.asarray(signs, dtype=float), (m,))
        self.applications += m
        ph = np.exp(1j * s * self.b * self.t)
        a, b = self.a, self.b
        Y = X * (self.J[0] * ph)
        if self.K == 0:
            return Y
        T0 = X
        T1 = np.empty_like(X)
        T2 = np.empty_like(X)
        i_pow = (1.0, 1j, -1.0, -1j)

        def unit_pow(k):                                     # (s i)^k, exactly
            return (s if k % 2 else 1.0) * i_pow[k % 4]

        def first(blk):
            r0, r1, Hb = blk
            v = Hb @ T0
            v -= b * T0[r0:r1]
            v *= 1.0 / a
            T1[r0:r1] = v
            Y[r0:r1] += coef * v

        def step(blk):
            r0, r1, Hb = blk
            v = Hb @ T1
            v -= b * T1[r0:r1]
            v *= 2.0 / a
            v -= T0[r0:r1]
            T2[r0:r1] = v
            Y[r0:r1] += coef * v

        run = (lambda f: [f(blk) for blk in self.blocks]) if len(self.blocks) == 1 else \
            (lambda f: list(_pool(self.threads).map(f, self.blocks)))
        coef = 2.0 * self.J[1] * unit_pow(1) * ph
        run(first)
        for k in range(2, self.K + 1):
            coef = 2.0 * self.J[k] * unit_pow(k) * ph
            run(step)
            T0, T1, T2 = T1, T2, (T0 if T0 is not X else np.empty_like(X))
        return Y

    def check(self, seed: int = 11) -> float:
        """Largest difference from scipy's expm_multiply on one random unit vector, both signs."""
        rng = np.random.default_rng(seed)
        v = rng.normal(size=self.dim) + 1j * rng.normal(size=self.dim)
        v /= np.linalg.norm(v)
        Y = self.apply(np.stack([v, v], axis=1), [-1.0, 1.0])
        ref_m = spla.expm_multiply(-1j * self.t * self.H, v, traceA=0.0)
        ref_p = spla.expm_multiply(1j * self.t * self.H, v, traceA=0.0)
        return float(max(np.linalg.norm(Y[:, 0] - ref_m), np.linalg.norm(Y[:, 1] - ref_p)))


# --------------------------------------------------------------------------- formulas

def eps_of_w(w: float) -> float:
    """min_phi ||U - e^{i phi} V|| for an eigenphase arc of width w (any w in [0, 2 pi])."""
    return 2.0 * math.sin(min(max(w, 0.0), 2 * math.pi) / 4.0)


def dcoh_of_w(w: float) -> float:
    """Diamond distance of the two unitary channels, 2 sin(w/2) for w < pi and 2 above."""
    return 2.0 * math.sin(min(max(w, 0.0), math.pi) / 2.0)


def floor_eps(c: float, N: int) -> float:
    """sqrt(1 - (1 - c)^N), a rigorous lower bound on eps_op from the per-site infidelity c at N sites."""
    return math.sqrt(max(0.0, -math.expm1(N * math.log1p(-min(max(c, 0.0), 1.0))))) if c < 1 else 1.0


def kappa(w: float, N: int, C: float) -> float:
    """(w/N)/sqrt(C): the constant of w/N = kappa sqrt(C)."""
    return (w / N) / math.sqrt(C) if C > 0 else math.nan


# --------------------------------------------------------------------------- Lanczos

class _W:
    """W = V^dag U and W^dag on blocks of columns, counting G applications."""

    def __init__(self, ops: list[Op], prop: Propagator, N: int):
        self.ops, self.prop, self.N = ops, prop, N
        self.matvecs = 0
        self.seconds_circuit = 0.0
        self.seconds_prop = 0.0

    def both(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = X.shape[1]
        t0 = time.time()
        A = apply_ops(self.ops, X, self.N)                           # U X
        t1 = time.time()
        B = self.prop.apply(np.concatenate([A, X], axis=1), [1.0] * m + [-1.0] * m)
        t2 = time.time()
        WdX = apply_ops(self.ops, B[:, m:], self.N, dagger=True)     # U^dag e^{-iHt} X
        t3 = time.time()
        self.seconds_circuit += (t1 - t0) + (t3 - t2)
        self.seconds_prop += t2 - t1
        return B[:, :m], WdX                                         # e^{iHt} U X, W^dag X

    def W(self, X: np.ndarray) -> np.ndarray:
        t0 = time.time()
        A = apply_ops(self.ops, X, self.N)
        t1 = time.time()
        out = self.prop.apply(A, 1.0)
        self.seconds_circuit += t1 - t0
        self.seconds_prop += time.time() - t1
        return out

    def G(self, X: np.ndarray, e: complex) -> np.ndarray:
        """Im(e W) X = (e W X - conj(e) W^dag X) / 2i."""
        self.matvecs += X.shape[1]
        WX, WdX = self.both(X)
        return (e * WX - np.conj(e) * WdX) / 2j

    def R(self, X: np.ndarray, e: complex) -> np.ndarray:
        """Re(e W) X = (e W X + conj(e) W^dag X) / 2."""
        self.matvecs += X.shape[1]
        WX, WdX = self.both(X)
        return (e * WX + np.conj(e) * WdX) / 2


def _tridiagonal_ends(d, e) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest eigenpairs of a real symmetric tridiagonal matrix: ([lo, hi], columns [s_lo, s_hi])."""
    m = len(d)
    if m == 1:
        return np.array([d[0], d[0]]), np.ones((1, 2))
    d, e = np.asarray(d, dtype=float), np.asarray(e, dtype=float)
    lo, slo = sla.eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
    hi, shi = sla.eigh_tridiagonal(d, e, select="i", select_range=(m - 1, m - 1))
    return np.array([lo[0], hi[0]]), np.concatenate([slo, shi], axis=1)


def _lanczos_ends(matvec, dim: int, seed: int, tol: float, atol: float, m_max: int,
                  check_every: int = 4, min_steps: int = 8, need: tuple[bool, bool] = (True, True)) -> dict:
    """Both extreme eigenpairs of a Hermitian operator from one Lanczos run, full reorthogonalization.

    Converged when each needed end's (``need`` = (top, bottom)) Ritz residual |beta_m s_m| is at most
    max(tol |theta|, atol).
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    q /= np.linalg.norm(q)
    m_max = max(2, min(m_max, dim))
    Q = np.empty((m_max, dim), dtype=complex)
    al: list[float] = []
    be: list[float] = []
    th = S = None
    conv = False
    for j in range(m_max):
        Q[j] = q
        w = matvec(q.reshape(-1, 1)).ravel()
        a = float(np.vdot(q, w).real)
        w -= a * q
        if j:
            w -= be[-1] * Q[j - 1]
        for _ in range(2):                                   # h_i = <Q_i|w>, without copying Q
            h = np.conj(Q[: j + 1] @ np.conj(w))
            w -= h @ Q[: j + 1]
        b = float(np.linalg.norm(w))
        al.append(a)
        be.append(b)
        exhausted = j + 1 == dim or b <= 1e-15 * max(1.0, max(abs(x) for x in al))
        if exhausted or j + 1 == m_max or (j + 1 >= min_steps and (j + 1) % check_every == 0):
            th, S = _tridiagonal_ends(al, be[:-1])
            r_hi, r_lo = abs(b * S[-1, -1]), abs(b * S[-1, 0])
            conv = exhausted or ((not need[0] or r_hi <= max(tol * abs(th[-1]), atol))
                                 and (not need[1] or r_lo <= max(tol * abs(th[0]), atol)))
            if conv or j + 1 == m_max:
                break
        q = w / b
    m = len(al)
    X = (S[:, [-1, 0]].T @ Q[:m]).T                                   # columns x_hi, x_lo
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    return {"hi": float(th[-1]), "lo": float(th[0]), "X": X, "steps": m, "converged": bool(conv),
            "ritz_res": [float(abs(be[-1] * S[-1, -1])), float(abs(be[-1] * S[-1, 0]))]}


def _ends(Wop: _W, e: complex, dim: int, method: str, tol: float, atol: float, m_max: int, seed: int) -> dict:
    """Extreme eigenpairs of G = Im(e W): own Lanczos (falls back on ARPACK), or ARPACK as crosscheck does."""
    G = lambda X: Wop.G(X, e)                                          # noqa: E731
    out = None
    if method == "lanczos":
        out = _lanczos_ends(G, dim, seed, tol, atol, m_max)
        out["method"] = "lanczos"
        if not out["converged"]:
            out = {**_arpack(G, dim, tol, out["X"]), "method": "lanczos+arpack", "lanczos_steps": out["steps"]}
    elif method == "arpack":
        out = {**_arpack(G, dim, tol, None), "method": "arpack"}
    else:
        raise ValueError(f"method must be 'lanczos' or 'arpack', not {method!r}")
    return out


def _arpack(G, dim: int, tol: float, X0) -> dict:
    op = spla.LinearOperator((dim, dim), matvec=lambda v: G(np.asarray(v).reshape(-1, 1)).ravel(), dtype=complex)
    res = {}
    for col, which in ((0, "LA"), (1, "SA")):
        v0 = None if X0 is None else X0[:, col]
        val, vec = spla.eigsh(op, k=1, which=which, tol=tol, maxiter=5000, v0=v0)
        res[which] = (float(val[0]), vec[:, 0] / np.linalg.norm(vec[:, 0]))
    X = np.stack([res["LA"][1], res["SA"][1]], axis=1)
    return {"hi": res["LA"][0], "lo": res["SA"][0], "X": X, "converged": True, "ritz_res": [math.nan, math.nan]}


def _crosscheck_phi0(Wop: _W, dim: int, seed: int = 0) -> float:
    """crosscheck.arc_width's phi0: the phase of sum_v <v|W|v> over 4 random vectors (same draws)."""
    rng = np.random.default_rng(seed)
    V = np.empty((dim, 4), dtype=complex)
    for k in range(4):
        v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        V[:, k] = v / np.linalg.norm(v)
    WV = Wop.W(V)
    return float(np.angle(np.sum(np.conj(V) * WV)))


def arc(circuit, model: XXZ, t: float, N: int, bc: str = "ring", edges=None, prop: Propagator | None = None,
        tol: float = 1e-10, atol: float = 1e-14, method: str = "lanczos", mem_gb: float = 4.0, seed: int = 1,
        threads: int | None = None) -> dict:
    """Arc width w of the eigenphases of V^dag U at N sites, matrix free, with its diagnostics.

    Returns w, eps_op, D_coh, phi0 (final), the two arc ends (the phases of the Ritz vectors from phi0),
    Ritz and explicit residuals, matvecs (G applications), recentering count, the back-half check, the
    saturation flag, and the time split between circuit and propagator.
    """
    t0 = time.time()
    c = as_circuit(circuit, edges)
    ops = c.ops(N, bc)
    prop = prop or Propagator(model, N, bc, t, threads=threads)
    dim = 2 ** N
    Wop = _W(ops, prop, N)
    phi0 = _crosscheck_phi0(Wop, dim)
    m_max = int(min(600, max(60, mem_gb * 2 ** 30 / (16 * dim))))
    history = []
    for attempt in range(4):
        e = np.exp(-1j * phi0)
        r = _ends(Wop, e, dim, method, tol, atol, m_max, seed)
        X = r["X"]
        WX, WdX = Wop.both(X)
        Wop.matvecs += X.shape[1]
        GX = (e * WX - np.conj(e) * WdX) / 2j
        lam = np.array([r["hi"], r["lo"]])
        res = np.linalg.norm(GX - X * lam, axis=0)
        z = e * np.sum(np.conj(X) * WX, axis=0)                     # <x|e W|x> = e^{i (theta - phi0)}
        d = np.angle(z)
        w_asin = math.asin(min(1.0, r["hi"])) - math.asin(max(-1.0, r["lo"]))
        history.append({"phi0": phi0, "hi": r["hi"], "lo": r["lo"], "d_hi": float(d[0]), "d_lo": float(d[1]),
                        "w": w_asin, "method": r["method"], "steps": r.get("steps"),
                        "ritz_res": r["ritz_res"], "res": res.tolist(), "absz": np.abs(z).tolist()})
        mid = 0.5 * (d[0] + d[1])
        if (max(abs(d[0]), abs(d[1])) <= RECENTER_AT or abs(mid) < 1e-3 or attempt == 3
                or w_asin >= math.pi - SATURATION_MARGIN):         # no phi0 resolves an arc of pi
            break
        phi0 = float(np.angle(np.exp(1j * (phi0 + mid))))
    last = history[-1]
    w = last["w"]
    saturated = bool(w >= math.pi - SATURATION_MARGIN
                     or max(abs(last["d_hi"]), abs(last["d_lo"])) >= math.pi / 2 - SATURATION_MARGIN)
    back = None
    if w > BACK_CHECK_ABOVE and not saturated:
        phic = phi0 + 0.5 * (last["d_hi"] + last["d_lo"])
        ec = np.exp(-1j * phic)
        rb = _lanczos_ends(lambda Y: Wop.R(Y, ec), dim, seed + 1, 1e-8, 1e-12, m_max, need=(False, True))
        dmax = math.acos(max(-1.0, min(1.0, rb["lo"])))
        back = {"phi_c": float(phic), "lambda_min": rb["lo"], "d_max": dmax, "steps": rb["steps"],
                "converged": rb["converged"], "ok": bool(abs(dmax - w / 2) <= 1e-3 + 1e-3 * w)}
        if not back["ok"]:
            saturated = True
    return {"N": N, "bc": bc, "w": w, "eps_op": eps_of_w(w), "D_coh": dcoh_of_w(w), "saturated": saturated,
            "phi0": last["phi0"], "ends": [last["d_hi"], last["d_lo"]], "lambda": [last["hi"], last["lo"]],
            "ritz_res": last["ritz_res"], "res": last["res"], "absz": last["absz"],
            "w_phase": float(last["d_hi"] - last["d_lo"]), "recentered": len(history) - 1, "history": history,
            "back_check": back, "method": last["method"], "matvecs": Wop.matvecs, "tol": tol,
            "seconds": time.time() - t0, "seconds_circuit": Wop.seconds_circuit,
            "seconds_prop": Wop.seconds_prop, "cheb_terms": prop.K, "threads": prop.threads}


# --------------------------------------------------------------------------- exact path

def dense_unitaries(circuit, model: XXZ, t: float, N: int, bc: str = "ring",
                    edges=None) -> tuple[np.ndarray, np.ndarray]:
    """(U, V) as dense matrices (N <= 10): U from the gate list, V = e^{-iHt} from eigh of H."""
    if N > 12:
        raise ValueError("dense matrices only up to N = 12")
    c = as_circuit(circuit, edges)
    d = 2 ** N
    U = apply_ops(c.ops(N, bc), np.eye(d, dtype=complex), N)          # columns U|e_k>
    E, Q = np.linalg.eigh(model.sparse(N, bc).toarray())
    V = (Q * np.exp(-1j * t * E)) @ Q.conj().T
    return U, V


def dense_metrics(circuit, model: XXZ, t: float, N: int, bc: str = "ring", edges=None) -> dict:
    """Every eigenphase of V^dag U by full diagonalization (N <= 10): the shortest arc, eps_op, D_coh, F_e."""
    t0 = time.time()
    U, V = dense_unitaries(circuit, model, t, N, bc, edges)
    W = V.conj().T @ U
    d = W.shape[0]
    theta = np.sort(np.angle(np.linalg.eigvals(W)))
    gaps = np.diff(np.append(theta, theta[0] + 2 * math.pi))
    k = int(np.argmax(gaps))
    w = float(2 * math.pi - gaps[k])
    center = float(np.angle(np.exp(1j * (theta[(k + 1) % d] + w / 2))))
    Fe = float(abs(np.trace(W) / d) ** 2)
    return {"N": N, "bc": bc, "w": w, "eps_op": eps_of_w(w), "D_coh": dcoh_of_w(w), "center": center,
            "F_e": Fe, "c_e": 1.0 - max(Fe, 1e-300) ** (1.0 / N),
            "unitarity": float(np.linalg.norm(W @ W.conj().T - np.eye(d), 2)), "seconds": time.time() - t0}


# --------------------------------------------------------------------------- fidelities

def infidelities(circuit, model: XXZ, t: float, N: int, bc: str = "ring", edges=None,
                 prop: Propagator | None = None, product: tuple[int, int] = PRODUCT_SET,
                 haar: tuple[int, int] | None = HAAR_SET, batch: int = 8, threads: int | None = None) -> dict:
    """C(N) on random product inputs and the Haar values, on the same unitary and N as arc().

    C = 1 - mean_k |<V phi_k|U phi_k>|^{2/N} (circuits.local_infidelity, draws of
    circuits.random_product_states(N, n, seed)); c_haar = 1 - mean f^{1/N} and c_e = 1 - F_e^{1/N},
    F_e = ((d + 1) mean f - 1)/d, over Haar states drawn as crosscheck.process_fidelity draws them.
    """
    t0 = time.time()
    c = as_circuit(circuit, edges)
    ops = c.ops(N, bc)
    prop = prop or Propagator(model, N, bc, t, threads=threads)
    d = 2 ** N

    def overlaps(X):
        Y = apply_ops(ops, X, N)
        T = prop.apply(X, -1.0)
        return np.abs(np.sum(np.conj(T) * Y, axis=0)) ** 2

    seed, n = product
    vecs = product_vectors(N, n, seed)
    F = []
    for k0 in range(0, n, batch):
        X = np.empty((d, min(batch, n - k0)), dtype=complex)
        for j, sites in enumerate(vecs[k0: k0 + batch]):
            psi = np.array([1.0 + 0j])
            for v in sites:
                psi = np.kron(psi, v)
            X[:, j] = psi
        F.append(overlaps(X))
    F = np.concatenate(F)
    ck = 1.0 - F ** (1.0 / N)
    out = {"N": N, "bc": bc, "product_set": [seed, n], "C": float(ck.mean()),
           "C_se": float(ck.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
           "max_infid": float((1.0 - F).max()), "mean_infid": float((1.0 - F).mean())}
    if haar:
        hs, hn = haar
        rng = np.random.default_rng(hs)
        v = rng.normal(size=(hn, d)) + 1j * rng.normal(size=(hn, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        f = np.concatenate([overlaps(np.ascontiguousarray(v[k0: k0 + batch].T)) for k0 in range(0, hn, batch)])
        Fe = float(((d + 1) * f.mean() - 1) / d)
        out.update({"haar_set": [hs, hn], "c_haar": float(1.0 - np.mean(f ** (1.0 / N))),
                    "F_e": Fe, "c_e": 1.0 - max(Fe, 1e-300) ** (1.0 / N),
                    "F_e_se": float((d + 1) / d * f.std(ddof=1) / math.sqrt(hn)) if hn > 1 else 0.0,
                    "haar_max_infid": float((1.0 - f).max())})
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- one size and boundary

def measure(circuit, model: XXZ, t: float, N: int, bc: str = "ring", edges=None, dense: bool | None = None,
            prop_check: bool | None = None, threads: int | None = None, product: tuple[int, int] = PRODUCT_SET,
            haar: tuple[int, int] | None = HAAR_SET, **arc_kw) -> dict:
    """Every number at one (N, bc): the arc and its diagnostics, C, the Haar values, kappa, the floor check.

    ``dense`` (default N <= 10) adds the full diagonalization and its difference from Lanczos;
    ``prop_check`` (default N <= 14) compares the propagator with expm_multiply.
    """
    t0 = time.time()
    c = as_circuit(circuit, edges)
    prop = Propagator(model, N, bc, t, threads=threads)
    out = {"N": N, "bc": bc, "t": float(t), "model": model.to_dict(), "dim": 2 ** N,
           "prop": {"K": prop.K, "bounds": list(prop.bounds), "setup_seconds": prop.setup_seconds,
                    "threads": prop.threads}}
    if prop_check if prop_check is not None else N <= 14:
        t1 = time.time()
        out["prop"]["check"] = prop.check()
        out["prop"]["check_seconds"] = time.time() - t1
    a = arc(c, model, t, N, bc, prop=prop, **arc_kw)
    f = infidelities(c, model, t, N, bc, prop=prop, product=product, haar=haar)
    out["arc"] = a
    out["fid"] = f
    out.update({"w": a["w"], "eps_op": a["eps_op"], "D_coh": a["D_coh"], "saturated": a["saturated"],
                "C": f["C"], "C_se": f["C_se"], "c_haar": f.get("c_haar"), "c_e": f.get("c_e"),
                "kappa": kappa(a["w"], N, f["C"]), "matvecs": a["matvecs"]})
    floor = floor_eps(f["C"], N)
    out["floor"] = floor
    out["floor_max"] = math.sqrt(max(f["max_infid"], 0.0))
    out["floor_ok"] = bool(a["saturated"] or out["floor_max"] <= a["eps_op"] * (1 + 1e-8) + 1e-12)
    if dense if dense is not None else N <= 10:
        dm = dense_metrics(c, model, t, N, bc)
        out["dense"] = dm
        out["dense_dw"] = float(a["w"] - dm["w"])
    out["seconds"] = time.time() - t0
    return out


# --------------------------------------------------------------------------- extrapolation

def extrapolate(Ns, ws, targets=(50, 100), fit_sizes=(10, 12, 14, 16), c: float | None = None,
                checks: dict | None = None) -> dict:
    """Affine fit w(N) = alpha N + beta on the fit sizes present, and w, eps_op, D_coh at the targets.

    ``c`` (the per-site infidelity at the target sizes, e.g. the MPS c at N = 100) gives the rigorous
    floor sqrt(1 - (1 - c)^N) <= eps_op. ``checks`` maps sizes outside the fit (e.g. 18, 20) to their
    measured w, reported against the prediction.
    """
    pts = sorted((int(N), float(w)) for N, w in zip(Ns, ws) if int(N) in fit_sizes and np.isfinite(w))
    if len(pts) < 2:
        raise ValueError(f"need two fit sizes among {fit_sizes}, have {[p[0] for p in pts]}")
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts])
    alpha, beta = (float(v) for v in np.polyfit(x, y, 1))
    resid = y - (alpha * x + beta)
    out = {"fit_sizes": x.astype(int).tolist(), "alpha": alpha, "beta": beta,
           "beta_sites": beta / alpha if alpha else math.nan,
           "max_rel_resid": float(np.max(np.abs(resid) / np.abs(y))) if np.all(y) else math.nan, "targets": {}}
    for Nt in targets:
        wt = alpha * Nt + beta
        row = {"w": wt, "eps_op": eps_of_w(min(wt, math.pi)), "D_coh": dcoh_of_w(wt), "saturated": bool(wt >= math.pi)}
        if c is not None:
            row["floor"] = floor_eps(c, Nt)
            row["ratio_to_floor"] = row["eps_op"] / row["floor"] if row["floor"] > 0 else math.nan
        out["targets"][str(Nt)] = row
    if checks:
        out["checks"] = {str(N): {"w": float(w), "w_pred": alpha * N + beta,
                                  "rel_err": float((alpha * N + beta - w) / w)} for N, w in checks.items()}
    return out


# --------------------------------------------------------------------------- circuits from the sweep's files

def snap_circuit(rec: dict, name: str = "") -> Circuit:
    """The realized SNAP circuit of a real_*.json record: angles_real, and its fitted edges when kept
    (edges.use), else the cell's own single-qubit factor, as evaluate.evaluate_circuit runs it."""
    from snapevo.cells import get_cell
    from snapevo.edges import pairs
    e = rec.get("edges") or {}
    use = bool(e.get("use"))
    return Circuit(layered=Layered.brickwork(get_cell(rec["cell"]), rec["angles_real"]),
                   edges=pairs(e["angles_real"]) if use else None, name=name,
                   info={"D": int(rec["D"]), "edges_used": use, "cell": rec["cell"]})


def baseline_circuit(rec: dict, model: XXZ, name: str = "") -> Circuit:
    """The realized Trotter + KN circuit of a baseline_*.json record.

    Rebuilt from the product formula (baseline.trotter_circuit) with every non-Clifford+T angle replaced
    by its recorded success-branch angle synth[...]["theta_s"], keyed as baseline.realize_baseline keys
    them; no synthesis is rerun. Open chain: the cell's own factor at the ends, as evaluated.
    """
    from snapevo.baseline import _is_exact, trotter_circuit
    circ0 = trotter_circuit(model, float(rec["t"]), int(rec["order"]), int(rec["steps"]))
    layers = []
    for parity, angles in circ0.layers:
        real = np.array(angles, dtype=float)
        for s, th in enumerate(angles):
            if not _is_exact(th):
                real[s] = rec["synth"][f"{float(round(th, 15)):.15f}"]["theta_s"]
        layers.append((parity, real))
    return Circuit(layered=Layered(circ0.cell, layers), name=name,
                   info={"D": circ0.depth, "order": int(rec["order"]), "steps": int(rec["steps"]),
                         "eps": float(rec["eps"])})


def trotter(model: XXZ, t: float, order: int, r: int, name: str = "") -> Circuit:
    """The pure product formula of baseline.trotter_circuit, no synthesis."""
    from snapevo.baseline import trotter_circuit
    c = trotter_circuit(model, t, order, r)
    return Circuit(layered=c, name=name, info={"D": c.depth, "order": order, "steps": r})


def open_trained_circuit(cells_doc: dict, edges_doc: dict | None, D: int, name: str = "") -> Circuit:
    """An open-chain-trained cell (cluster/open_train.py: open/cells.json and open/edges.json), continuous
    angles, with its trained edge rotations (6 ZYZ angles per parity-1 sublayer, edges.pairs order)."""
    from snapevo.cells import get_cell
    from snapevo.edges import pairs
    row = next(r for r in cells_doc["results"] if int(r["D"]) == D)
    x = np.asarray(row["params"], dtype=float).reshape(-1)
    edges = None
    if edges_doc is not None:
        er = next((r for r in edges_doc["results"] if int(r["D"]) == D), None)
        edges = pairs(er["angles"]) if er is not None else None
    return Circuit(layered=Layered.brickwork(get_cell(cells_doc["cell"]), x), edges=edges, name=name,
                   info={"D": D, "edges_used": edges is not None, "cell": cells_doc["cell"],
                         "provisional": bool(row.get("provisional", False))})
