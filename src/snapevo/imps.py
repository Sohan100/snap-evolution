"""Infinite MPS, iTEBD for the translation-invariant brickworks, and the per-site fidelity of two iMPS.

The translation-invariant (TI) error convention (Gibbs and Cincio, arXiv:2506.02105) reports the infidelity
per site in the thermodynamic limit. For two infinite MPS |a> and |b> with the same L-site unit cell the
overlap of N = n L sites goes as lambda0^n, lambda0 the leading eigenvalue of the mixed transfer matrix of one
unit cell, T = prod_j sum_s A_j^s (x) conj(B_j^s). So the per-site fidelity is f = |lambda0|^{1/L} (normalized by
the two states' own transfer matrices), and the per-site infidelity in the convention of circuits.py and
evaluate.py, c = 1 - F^{1/N} with F = |<a|b>|^2, is

    c_inf = 1 - |lambda0|^{2/L}.

State. An infinite MPS with an L-site unit cell, L even (L = 2 is the brickwork's own period; a product input
with an L_u-site pattern needs L = L_u), in Vidal form Gamma_j, Lambda_j, j = 0..L-1, Lambda_j on the bond left of
site j (Lambda_0 between site L-1 of one cell and site 0 of the next). It is stored as Hastings' B_j =
Gamma_j Lambda_{j+1} (right-canonical) and S_j = Lambda_j, which lets a two-site update avoid dividing by small
singular values (Hastings, J. Math. Phys. 50, 095207 (2009)): for the bond (i, i+1),

    C = U . (B_i B_{i+1}),  theta = S_i C = X Y Z (SVD, truncated),  B_{i+1} <- Z,  B_i <- C Z^dag / |Y|,
    S_{i+1} <- Y / |Y|.

Truncation as mps.MPS: keep the fewest singular values whose discarded weight (relative) is at most
``cutoff``, at most ``chi_max`` of them; the discarded weights are summed in ``discarded``.

Circuits. ``apply_layered`` runs a snapevo Layered circuit sublayer by sublayer in the gate convention of
Layered.apply(bc="ring"): the cell gate on every bond of the sublayer's parity, a gate on (i, i+1) with its
first qubit on i; the parity-1 bond (L-1, L) of the infinite chain is the ring's wrap bond (N-1, 0), first
qubit N-1. A sublayer with no coupling (the product formulas' field layers) is applied as single-qubit gates,
the cell's one_qubit_factor, as mps.apply_layered does. The reference e^{-iHt} is mps.reference_circuit (order-4
Suzuki, steps of at most dt) applied by the same code (``reference_state``).

Fidelity. ``fidelity`` finds the leading eigenvalue of the mixed transfer matrix (dense at small bond
dimension, else ARPACK on the chi_a x chi_b matrix-free map X -> sum_s A^s X B^s^dag), normalizes by the two
self transfer matrices (1 for exact right-canonical states, off by the truncation otherwise) and returns c_inf.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from snapevo.circuits import Layered
from snapevo.hamiltonian import XXZ

__all__ = ["IMPS", "apply_layered", "reference_state", "leading_eigenvalue", "fidelity", "c_inf",
           "ring_state", "unit_cell_inputs"]

DENSE_MAX = 400          # mixed transfer matrices up to this dimension are diagonalized densely


def _svd(M: np.ndarray):
    try:
        return np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:                # gesdd occasionally fails to converge
        return sla.svd(M, full_matrices=False, lapack_driver="gesvd")


class IMPS:
    """Infinite MPS with an L-site unit cell: B[j] (chi_j, 2, chi_{j+1}) = Gamma_j Lambda_{j+1}, S[j] = Lambda_j."""

    def __init__(self, B, S, chi_max: int = 1024, cutoff: float = 1e-12):
        self.B = [np.asarray(b, dtype=complex) for b in B]
        self.S = [np.asarray(s, dtype=float) for s in S]
        self.L = len(self.B)
        if self.L < 2 or self.L % 2 or len(self.S) != self.L:
            raise ValueError(f"need an even unit cell with one S per site, have {self.L} tensors, {len(self.S)} S")
        for j in range(self.L):
            jr = (j + 1) % self.L
            if self.B[j].shape[0] != self.S[j].size or self.B[j].shape[2] != self.S[jr].size or self.B[j].shape[1] != 2:
                raise ValueError(f"site {j}: B {self.B[j].shape} does not match S {self.S[j].size}, {self.S[jr].size}")
        self.chi_max = int(chi_max)
        self.cutoff = float(cutoff)
        self.discarded = 0.0                     # sum of discarded weights over every truncation
        self.truncations = 0                     # two-site updates that discarded anything
        self.updates = 0                         # two-site updates
        self.chi_peak = max(s.size for s in self.S)

    @classmethod
    def product(cls, vectors, **kw) -> IMPS:
        """Product state with the unit cell's single-site vectors, shape (L, 2)."""
        V = np.asarray(vectors, dtype=complex)
        if V.ndim != 2 or V.shape[1] != 2:
            raise ValueError(f"vectors must be (L, 2), not {V.shape}")
        return cls([(v / np.linalg.norm(v)).reshape(1, 2, 1) for v in V], [np.ones(1) for _ in V], **kw)

    def copy(self) -> IMPS:
        out = IMPS([b.copy() for b in self.B], [s.copy() for s in self.S], self.chi_max, self.cutoff)
        out.discarded, out.truncations, out.updates, out.chi_peak = (self.discarded, self.truncations, self.updates,
                                                                     self.chi_peak)
        return out

    @property
    def chi(self) -> list[int]:
        """Bond dimensions, chi[j] on the bond left of site j."""
        return [s.size for s in self.S]

    @property
    def Gamma(self) -> list[np.ndarray]:
        """Vidal's Gamma_j = B_j Lambda_{j+1}^{-1}."""
        return [self.B[j] / self.S[(j + 1) % self.L][None, None, :] for j in range(self.L)]

    @property
    def Lambda(self) -> list[np.ndarray]:
        return [s.copy() for s in self.S]

    # ----------------------------------------------------------------------- gates

    def apply_1q(self, g: np.ndarray, j: int) -> None:
        """A 2x2 unitary on site j of every unit cell (keeps the canonical form)."""
        j %= self.L
        self.B[j] = np.einsum("st,atb->asb", np.asarray(g, dtype=complex), self.B[j])

    def apply_2q(self, U: np.ndarray, i: int) -> None:
        """A 4x4 gate on sites (i, i+1) of every unit cell, first qubit on i, then truncate."""
        L = self.L
        i %= L
        j = (i + 1) % L
        Bi, Bj = self.B[i], self.B[j]
        l, r = Bi.shape[0], Bj.shape[2]
        C = np.tensordot(Bi, Bj, axes=(2, 0))                                       # (l, 2, 2, r)
        C = np.tensordot(np.asarray(U, dtype=complex).reshape(2, 2, 2, 2), C, axes=([2, 3], [1, 2]))  # (s, t, l, r)
        C = C.transpose(2, 0, 1, 3)                                                 # (l, s, t, r)
        theta = self.S[i][:, None, None, None] * C
        u, y, z = _svd(theta.reshape(l * 2, 2 * r))
        w = y * y
        tot = float(w.sum())
        tail = np.cumsum(w[::-1])[::-1] / tot       # tail[k] = weight discarded if k values are kept
        ok = np.flatnonzero(tail <= self.cutoff)
        k = max(1, min(int(ok[0]) if ok.size else y.size, self.chi_max))
        self.updates += 1
        if k < y.size:
            self.discarded += float(tail[k])
            self.truncations += 1
        yk = y[:k]
        nrm = float(np.linalg.norm(yk))
        zk = z[:k]
        self.B[j] = zk.reshape(k, 2, r)
        self.B[i] = np.tensordot(C, zk.conj().reshape(k, 2, r), axes=([2, 3], [1, 2])) / nrm   # (l, 2, k)
        self.S[j] = yk / nrm
        self.chi_peak = max(self.chi_peak, k)

    def apply_layer(self, U: np.ndarray, parity: int) -> None:
        """U on every bond (i, i+1) with i = parity mod 2 (bonds of one parity commute)."""
        for i in range(parity % 2, self.L, 2):
            self.apply_2q(U, i)

    def apply_1q_layer(self, g_first: np.ndarray, g_second: np.ndarray, parity: int) -> None:
        """g_first on sites i = parity mod 2 and g_second on sites i + 1: a sublayer of product gates."""
        for i in range(parity % 2, self.L, 2):
            self.apply_1q(g_first, i)
            self.apply_1q(g_second, i + 1)

    # ----------------------------------------------------------------------- measurements

    def transfer_right(self, other: IMPS, X: np.ndarray) -> np.ndarray:
        """One unit cell of the mixed transfer matrix on X (chi_self[0], chi_other[0]): sum_s A^s X B^s^dag."""
        for j in reversed(range(self.L)):
            X = np.tensordot(self.B[j], X, axes=(2, 0))                             # (a', s, b)
            X = np.tensordot(X, other.B[j].conj(), axes=([1, 2], [1, 2]))           # (a', b')
        return X

    def rdm(self, j: int, m: int) -> np.ndarray:
        """Reduced density matrix of sites j..j+m-1, (2^m, 2^m), first site most significant (exact in canonical form)."""
        L = self.L
        s = self.S[j % L]
        T = s[:, None, None] * self.B[j % L]                                        # (l, 2, r)
        for k in range(1, m):
            T = np.tensordot(T, self.B[(j + k) % L], axes=(-1, 0))
        l, r = T.shape[0], T.shape[-1]
        T = T.reshape(l, 2 ** m, r)
        return np.einsum("asb,atb->st", T, T.conj())

    def norm_eigenvalue(self) -> complex:
        """Leading eigenvalue of the state's own transfer matrix (1 in exact canonical form)."""
        return leading_eigenvalue(self, self)["lambda"]

    # ----------------------------------------------------------------------- files

    def save(self, path: Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        info = {**(meta or {}), "L": self.L, "chi_max": self.chi_max, "cutoff": self.cutoff,
                "discarded": self.discarded, "truncations": self.truncations, "updates": self.updates,
                "chi_peak": self.chi_peak}
        np.savez(tmp, meta=np.array(json.dumps(info)), **{f"B{j}": b for j, b in enumerate(self.B)},
                 **{f"S{j}": s for j, s in enumerate(self.S)})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> tuple[IMPS, dict]:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            L = int(meta["L"])
            psi = cls([z[f"B{j}"] for j in range(L)], [z[f"S{j}"] for j in range(L)], meta["chi_max"],
                      meta["cutoff"])
        psi.discarded, psi.truncations, psi.updates, psi.chi_peak = (meta["discarded"], meta["truncations"],
                                                                     meta["updates"], meta["chi_peak"])
        return psi, meta


# --------------------------------------------------------------------------- circuits

def apply_layered(psi: IMPS, circuit: Layered) -> IMPS:
    """Apply a Layered circuit to an iMPS in place, sublayer by sublayer (module docstring)."""
    for parity, angles in circuit.layers:
        if not np.any(np.asarray(angles[:3], dtype=float)):      # no coupling: a single-qubit sublayer
            g1, g2 = circuit.cell.one_qubit_factor(angles)
            psi.apply_1q_layer(g1, g2, parity)
        else:
            psi.apply_layer(circuit.cell.gate(angles), parity)
    return psi


def reference_state(model: XXZ, t: float, vectors, dt: float = 0.1, chi_max: int = 1024, cutoff: float = 1e-12,
                    cache: Path | None = None, log=None) -> tuple[IMPS, dict]:
    """e^{-iHt} on the unit-cell product input ``vectors`` (L, 2): mps.reference_circuit by iTEBD, cached on disk."""
    from snapevo.mps import reference_circuit
    if cache is not None and Path(cache).exists():
        return IMPS.load(cache)
    t0 = time.time()
    circ = reference_circuit(model, t, dt)
    psi = apply_layered(IMPS.product(vectors, chi_max=chi_max, cutoff=cutoff), circ)
    meta = {"model": model.to_dict(), "t": float(t), "L": psi.L, "dt": dt, "order": 4, "sublayers": circ.depth,
            "seconds": time.time() - t0, "vectors_re": np.real(vectors).tolist(), "vectors_im": np.imag(vectors).tolist()}
    if cache is not None:
        psi.save(cache, meta)
    if log:
        log(f"imps reference: t = {t:g}, L = {psi.L}, chi peak {psi.chi_peak}, discarded {psi.discarded:.1e}, "
            f"{meta['seconds']:.1f} s")
    return psi, {**meta, "discarded": psi.discarded, "chi_peak": psi.chi_peak}


# --------------------------------------------------------------------------- the transfer matrix

def _dense_transfer(a: IMPS, b: IMPS) -> np.ndarray:
    """The mixed transfer matrix of one unit cell as a dense (chi_a chi_b, chi_a chi_b) matrix."""
    T = None
    for j in range(a.L):
        E = np.einsum("asb,csd->acbd", a.B[j], b.B[j].conj())
        E = E.reshape(a.B[j].shape[0] * b.B[j].shape[0], a.B[j].shape[2] * b.B[j].shape[2])
        T = E if T is None else T @ E
    return T


def leading_eigenvalue(a: IMPS, b: IMPS, tol: float = 0.0, dense_max: int = DENSE_MAX, seed: int = 0) -> dict:
    """The eigenvalue of largest modulus of the mixed transfer matrix of a and b, with its residual.

    Dense diagonalization up to ``dense_max``, else ARPACK (k = 1, which = "LM", tol = 0: machine precision)
    on the matrix-free map. ``gap`` is |lambda_1/lambda_0| when it is known (dense path, or ARPACK's second Ritz
    value), the rate at which a ring of n unit cells approaches lambda_0^n.
    """
    if a.L != b.L:
        raise ValueError(f"unit cells differ: {a.L} and {b.L}")
    ca, cb = a.B[0].shape[0], b.B[0].shape[0]
    dim = ca * cb
    t0 = time.time()
    if dim <= dense_max:
        T = _dense_transfer(a, b)
        ev, V = np.linalg.eig(T)
        o = np.argsort(-np.abs(ev))
        lam = complex(ev[o[0]])
        v = V[:, o[0]]
        gap = float(abs(ev[o[1]]) / abs(lam)) if dim > 1 and abs(lam) > 0 else 0.0
        res = float(np.linalg.norm(T @ v - lam * v) / np.linalg.norm(v))
        method = "dense"
    else:
        def mv(x):
            return a.transfer_right(b, np.asarray(x).reshape(ca, cb)).reshape(-1)
        op = spla.LinearOperator((dim, dim), matvec=mv, dtype=complex)
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        k = 2 if dim > 8 else 1
        ev, V = spla.eigs(op, k=k, which="LM", tol=tol, v0=v0, maxiter=10000, ncv=min(dim - 1, 24))
        o = np.argsort(-np.abs(ev))
        lam = complex(ev[o[0]])
        v = V[:, o[0]]
        gap = float(abs(ev[o[1]]) / abs(lam)) if k > 1 else math.nan
        res = float(np.linalg.norm(mv(v) - lam * v) / np.linalg.norm(v))
        method = "arpack"
    return {"lambda": lam, "abs": abs(lam), "residual": res, "gap": gap, "dim": dim, "method": method,
            "seconds": time.time() - t0}


def fidelity(a: IMPS, b: IMPS, **kw) -> dict:
    """Per-site fidelity of two iMPS with the same unit cell, and c_inf = 1 - |lambda0|^{2/L} (normalized)."""
    ab = leading_eigenvalue(a, b, **kw)
    aa = leading_eigenvalue(a, a, **kw)
    bb = leading_eigenvalue(b, b, **kw)
    ratio = ab["abs"] / math.sqrt(aa["abs"] * bb["abs"])
    L = a.L
    log_f = math.log(ratio) / L                                    # log of the per-site amplitude fidelity
    return {"c_inf": float(-math.expm1(2 * log_f)), "f_site": math.exp(log_f), "lambda": ab["lambda"],
            "lambda_aa": aa["lambda"], "lambda_bb": bb["lambda"], "residual": max(ab["residual"], aa["residual"],
                                                                                  bb["residual"]),
            "gap": ab["gap"], "dim": ab["dim"], "method": ab["method"], "L": L,
            "seconds": ab["seconds"] + aa["seconds"] + bb["seconds"]}


def c_inf(circuit: Layered, model: XXZ, t: float, vectors, dt: float = 0.1, chi_max: int = 1024,
          cutoff: float = 1e-12, ref: IMPS | None = None, ref_cache: Path | None = None, log=None) -> dict:
    """c_inf of a circuit against e^{-iHt} on one unit-cell product input, with the truncation bookkeeping."""
    if ref is None:
        ref, _ = reference_state(model, t, vectors, dt, chi_max, cutoff, ref_cache, log)
    t0 = time.time()
    u = apply_layered(IMPS.product(vectors, chi_max=chi_max, cutoff=cutoff), circuit)
    sec = time.time() - t0
    f = fidelity(u, ref)
    return {**f, "chi_u": u.chi_peak, "chi_v": ref.chi_peak, "chi_u_end": max(u.chi), "chi_v_end": max(ref.chi),
            "discarded_u": u.discarded, "discarded_v": ref.discarded, "seconds_circuit": sec, "L": u.L}


# --------------------------------------------------------------------------- inputs, and the ring for checks

def unit_cell_inputs(L: int, n: int, seed: int) -> np.ndarray:
    """n random unit-cell product inputs of L sites, (n, L, 2): circuits.product_vectors(L, n, seed)."""
    from snapevo.circuits import product_vectors
    return product_vectors(L, n, seed)


def ring_state(vectors, N: int) -> np.ndarray:
    """The unit cell's product state placed periodically on an N-site ring, dense (2^N,), site 0 most significant."""
    V = np.asarray(vectors, dtype=complex)
    L = V.shape[0]
    if N % L:
        raise ValueError(f"N = {N} is not a multiple of the unit cell {L}")
    psi = np.array([1.0 + 0j])
    for i in range(N):
        psi = np.kron(psi, V[i % L] / np.linalg.norm(V[i % L]))
    return psi
