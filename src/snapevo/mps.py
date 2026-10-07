"""Matrix product states on an open chain, for the 100-site evaluation.

A state is a list of tensors A[i] of shape (chi_left, 2, chi_right), site 0 first,
kept in mixed canonical form around an orthogonality center. A two-qubit gate on
neighbouring sites is followed by an SVD that keeps the fewest singular values
whose discarded weight is at most ``cutoff``, at most ``chi_max`` of them, and
renormalizes. The discarded weights add up in ``discarded``. For two states
truncated this way, the trace distance moves by at most sqrt(discarded) each,
which is how evaluate.py bounds the truncation error of a fidelity.

The reference e^{-iHt}|phi> runs through the same code as a fine order-4 product
formula (reference_circuit). Its step is calibrated against exact evolution in
docs/validation.md. A reference depends only on the model, the time and the
input, so evaluate.py caches it on disk and every circuit at that time reuses it.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from snapevo.circuits import Layered
from snapevo.hamiltonian import XXZ

__all__ = ["MPS", "apply_layered", "reference_circuit", "reference_state"]


def _svd(M: np.ndarray):
    try:
        return np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:                # gesdd occasionally fails to converge
        return sla.svd(M, full_matrices=False, lapack_driver="gesvd")


class MPS:
    """Open-chain qubit MPS with an orthogonality center and truncation bookkeeping."""

    def __init__(self, tensors, chi_max: int = 1024, cutoff: float = 1e-12):
        self.A = [np.asarray(a, dtype=complex) for a in tensors]
        self.N = len(self.A)
        self.center = 0
        self.chi_max = int(chi_max)
        self.cutoff = float(cutoff)
        self.discarded = 0.0                     # sum of discarded weights
        self.chi_peak = max(a.shape[2] for a in self.A)

    @classmethod
    def product(cls, vectors, **kw) -> MPS:
        """Product state from single-site vectors, shape (N, 2). Canonical about any site."""
        return cls([np.asarray(v, dtype=complex).reshape(1, 2, 1) / np.linalg.norm(v) for v in vectors], **kw)

    @property
    def bond_dims(self) -> list[int]:
        return [a.shape[2] for a in self.A[:-1]]

    def _move_to(self, i: int) -> None:
        while self.center < i:
            c = self.center
            l, d, r = self.A[c].shape
            Q, R = np.linalg.qr(self.A[c].reshape(l * d, r))
            self.A[c] = Q.reshape(l, d, Q.shape[1])
            self.A[c + 1] = np.tensordot(R, self.A[c + 1], axes=(1, 0))
            self.center += 1
        while self.center > i:
            c = self.center
            l, d, r = self.A[c].shape
            Q, R = np.linalg.qr(self.A[c].reshape(l, d * r).T)
            self.A[c] = Q.T.reshape(Q.shape[1], d, r)
            self.A[c - 1] = np.tensordot(self.A[c - 1], R.T, axes=(2, 0))
            self.center -= 1

    def apply_1q(self, g: np.ndarray, i: int) -> None:
        """Apply a 2x2 unitary to site i (keeps the canonical form)."""
        self.A[i] = np.tensordot(g, self.A[i], axes=(1, 1)).transpose(1, 0, 2)

    def apply_2q(self, U: np.ndarray, i: int, move: str = "right") -> None:
        """Apply a 4x4 gate to sites (i, i+1), site i being the first qubit, then truncate.

        ``move`` says where the center ends up, i + 1 ("right") or i ("left").
        """
        if self.center < i:
            self._move_to(i)
        elif self.center > i + 1:
            self._move_to(i + 1)
        theta = np.tensordot(self.A[i], self.A[i + 1], axes=(2, 0))             # (l, 2, 2, r)
        theta = np.tensordot(U.reshape(2, 2, 2, 2), theta, axes=([2, 3], [1, 2]))
        l, r = theta.shape[2], theta.shape[3]
        u, s, vh = _svd(theta.transpose(2, 0, 1, 3).reshape(l * 2, 2 * r))
        w = s * s
        tail = np.cumsum(w[::-1])[::-1] / w.sum()  # tail[k] = weight discarded if k values are kept
        ok = np.flatnonzero(tail <= self.cutoff)
        k = max(1, min(int(ok[0]) if ok.size else len(s), self.chi_max))
        if k < len(s):
            self.discarded += float(tail[k])
        s = s[:k] / math.sqrt(float(w[:k].sum()))
        if move == "right":
            self.A[i] = u[:, :k].reshape(l, 2, k)
            self.A[i + 1] = (s[:, None] * vh[:k]).reshape(k, 2, r)
            self.center = i + 1
        else:
            self.A[i] = (u[:, :k] * s).reshape(l, 2, k)
            self.A[i + 1] = vh[:k].reshape(k, 2, r)
            self.center = i
        self.chi_peak = max(self.chi_peak, k)

    def overlap(self, other: MPS) -> complex:
        """<self|other>."""
        E = np.ones((1, 1), dtype=complex)
        for a, b in zip(self.A, other.A):
            E = np.tensordot(np.tensordot(E, a.conj(), axes=(0, 0)), b, axes=([0, 1], [0, 1]))
        return complex(E[0, 0])

    def to_dense(self) -> np.ndarray:
        """The full state vector (small N only, for tests)."""
        v = self.A[0]
        for a in self.A[1:]:
            v = np.tensordot(v, a, axes=(-1, 0))
        return v.reshape(-1)

    def save(self, path: Path, meta: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        np.savez(tmp, meta=np.array(json.dumps({**meta, "discarded": self.discarded, "chi_peak": self.chi_peak,
                                                "chi_max": self.chi_max, "cutoff": self.cutoff,
                                                "center": self.center})),
                 **{f"A{i}": a for i, a in enumerate(self.A)})
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple[MPS, dict]:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            n = sum(1 for k in z.files if k.startswith("A"))
            psi = cls([z[f"A{i}"] for i in range(n)], chi_max=meta["chi_max"], cutoff=meta["cutoff"])
        psi.center, psi.discarded, psi.chi_peak = int(meta["center"]), meta["discarded"], meta["chi_peak"]
        return psi, meta


def apply_layered(psi: MPS, circuit: Layered, edges=None) -> MPS:
    """Apply a layered circuit to an open-chain MPS in place (edges as in Layered.apply)."""
    N = psi.N
    if N % 2:
        raise ValueError("the brickwork needs an even number of sites")
    wrap = iter(circuit.edge_pairs(edges))
    for parity, angles in circuit.layers:
        bonds = list(range(parity, N - 1, 2))
        if parity == 1:
            g_last, g_first = next(wrap)
            psi.apply_1q(g_last, N - 1)
            psi.apply_1q(g_first, 0)
        if not np.any(np.asarray(angles[:3], dtype=float)):    # no coupling: single-qubit layer
            g1, g2 = circuit.cell.one_qubit_factor(angles)
            for i in bonds:
                psi.apply_1q(g1, i)
                psi.apply_1q(g2, i + 1)
            continue
        U = circuit.cell.gate(angles)
        if psi.center > N // 2:
            for i in reversed(bonds):
                psi.apply_2q(U, i, move="left")
        else:
            for i in bonds:
                psi.apply_2q(U, i, move="right")
    return psi


def reference_circuit(model: XXZ, t: float, dt: float) -> Layered:
    """Order-4 Suzuki product formula with steps of at most dt: the stand-in for e^{-iHt}."""
    from snapevo.baseline import trotter_circuit
    return trotter_circuit(model, t, 4, max(1, math.ceil(t / dt - 1e-9)))


def reference_state(model: XXZ, t: float, vectors, dt: float, chi_max: int, cutoff: float,
                    cache: Path | None = None, log=None) -> tuple[MPS, dict]:
    """e^{-iHt} on the product state ``vectors`` (N, 2), from the cache if it is there."""
    if cache is not None and Path(cache).exists():
        return MPS.load(cache)
    t0 = time.time()
    psi = apply_layered(MPS.product(vectors, chi_max=chi_max, cutoff=cutoff), reference_circuit(model, t, dt))
    meta = {"model": model.to_dict(), "t": t, "N": psi.N, "dt": dt, "order": 4, "seconds": time.time() - t0}
    if cache is not None:
        psi.save(cache, meta)
    if log:
        log(f"reference: t = {t:g}, N = {psi.N}, chi peak {psi.chi_peak}, discarded {psi.discarded:.1e}, "
            f"{meta['seconds']:.0f}s")
    return psi, {**meta, "discarded": psi.discarded, "chi_peak": psi.chi_peak}
