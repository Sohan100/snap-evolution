"""The open-chain MPS kernels of the chain cost on a GPU (CuPy): mps.py's MPS and apply_layered, and the per-input
forward/backward/environment code of mps_train (_run_input) and chain_snap (curve environments).

Same gates, order, sweep directions, truncation rule (fewest values whose discarded weight is at most ``cutoff``,
at most ``chi_max``) and renormalization as the CPU code; the numbers agree up to rounding (tests/test_mps_gpu.py).
One difference in method, not in result: a two-qubit gate's truncation does not call an SVD (cuSOLVER's gesvd
takes 0.85 s on a 2048 x 2048 complex matrix on an A100, against 0.075 s for an eigh). The theta matrix M
(2 chi_l x 2 chi_r) is split through the Gram matrix of its smaller side: eigh of M^H M (or M M^H) gives the
squared singular values, from which the kept number k and the discarded weight follow exactly as from an SVD,
and the top-k right (left) singular subspace; the other factor is M projected on it, re-orthonormalized by a QR
where the sweep needs an isometry there. So every kept factor is an exact isometry or exactly the projection of
M, as with the SVD; squared singular values come with an absolute error of ~1e-16 of the largest, which moves
the truncation point only at cutoffs near 1e-16 (the training cutoffs are 1e-10 to 1e-14).

Device placement: one process uses one GPU, cupy's current device (set CUDA_VISIBLE_DEVICES, or call
cupy.cuda.Device(i).use(), before the first array). The CPU code paths are untouched; chain_farm swaps these
functions into a GPU worker process's copy of chain_snap.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from snapevo.circuits import Layered

__all__ = ["GMPS", "to_gpu", "apply_layered_gpu", "apply_dagger_gpu", "sublayer_environments_gpu",
           "run_input_gpu", "snapshot_gpu", "sandwich_gpu", "env_states_gpu"]


def _cp():
    import cupy as cp
    return cp


_GATES: dict = {}


def _dev(g):
    """A gate (numpy) on the device, cached by its bytes (a sublayer's gate is applied to ~50 bonds)."""
    cp = _cp()
    g = np.asarray(g, dtype=complex)
    key = (g.shape, g.tobytes())
    out = _GATES.get(key)
    if out is None:
        if len(_GATES) > 4096:
            _GATES.clear()
        out = _GATES[key] = cp.asarray(g)
    return out


class GMPS:
    """mps.MPS with its tensors on the GPU (same attributes and methods; overlap returns a Python complex)."""

    def __init__(self, tensors, chi_max: int = 1024, cutoff: float = 1e-12):
        cp = _cp()
        self.A = [cp.asarray(a, dtype=complex) for a in tensors]
        self.N = len(self.A)
        self.center = 0
        self.chi_max = int(chi_max)
        self.cutoff = float(cutoff)
        self.discarded = 0.0
        self.chi_peak = max(a.shape[2] for a in self.A)

    @classmethod
    def product(cls, vectors, **kw) -> GMPS:
        return cls([np.asarray(v, dtype=complex).reshape(1, 2, 1) / np.linalg.norm(v) for v in vectors], **kw)

    @classmethod
    def from_cpu(cls, psi) -> GMPS:
        g = cls(psi.A, chi_max=psi.chi_max, cutoff=psi.cutoff)
        g.center, g.discarded, g.chi_peak = psi.center, psi.discarded, psi.chi_peak
        return g

    def to_cpu(self):
        from snapevo.mps import MPS
        m = MPS([a.get() for a in self.A], chi_max=self.chi_max, cutoff=self.cutoff)
        m.center, m.discarded, m.chi_peak = self.center, self.discarded, self.chi_peak
        return m

    @property
    def bond_dims(self) -> list[int]:
        return [a.shape[2] for a in self.A[:-1]]

    def _move_to(self, i: int) -> None:
        cp = _cp()
        while self.center < i:
            c = self.center
            l, d, r = self.A[c].shape
            Q, R = cp.linalg.qr(self.A[c].reshape(l * d, r))
            self.A[c] = Q.reshape(l, d, Q.shape[1])
            self.A[c + 1] = cp.tensordot(R, self.A[c + 1], axes=(1, 0))
            self.center += 1
        while self.center > i:
            c = self.center
            l, d, r = self.A[c].shape
            Q, R = cp.linalg.qr(self.A[c].reshape(l, d * r).T)
            self.A[c] = Q.T.reshape(Q.shape[1], d, r)
            self.A[c - 1] = cp.tensordot(self.A[c - 1], R.T, axes=(2, 0))
            self.center -= 1

    def apply_1q(self, g, i: int) -> None:
        cp = _cp()
        self.A[i] = cp.tensordot(_dev(g), self.A[i], axes=(1, 1)).transpose(1, 0, 2)

    def apply_2q(self, U, i: int, move: str = "right") -> None:
        """mps.MPS.apply_2q, the truncation through the smaller Gram matrix (module docstring)."""
        cp = _cp()
        if self.center < i:
            self._move_to(i)
        elif self.center > i + 1:
            self._move_to(i + 1)
        theta = cp.tensordot(self.A[i], self.A[i + 1], axes=(2, 0))                  # (l, 2, 2, r)
        theta = cp.tensordot(_dev(U).reshape(2, 2, 2, 2), theta, axes=([2, 3], [1, 2]))
        l, r = theta.shape[2], theta.shape[3]
        M = theta.transpose(2, 0, 1, 3).reshape(l * 2, 2 * r)
        # the Gram matrix of the side the center moves away from gives the isometry directly (no QR); take the
        # other side only when it is clearly smaller (an eigh costs ~ n^3)
        right_side = (r < 0.9 * l) if move == "right" else (r <= l / 0.9)
        if right_side:
            lam, V = cp.linalg.eigh(M.conj().T @ M)            # ascending
        else:
            lam, V = cp.linalg.eigh(M @ M.conj().T)
        lam = lam[::-1]
        V = V[:, ::-1]
        w = cp.clip(lam, 0.0, None)
        w_h = w.get()
        total = float(w_h.sum())
        tail = np.cumsum(w_h[::-1])[::-1] / total
        ok = np.flatnonzero(tail <= self.cutoff) if self.cutoff > 0 else np.array([], dtype=int)
        k = max(1, min(int(ok[0]) if ok.size else len(w_h), self.chi_max))
        if k < len(w_h):
            self.discarded += float(tail[k])
        Vk = V[:, :k]
        if right_side:                                         # Vk: right singular vectors (2r x k)
            P = M @ Vk                                         # U S, (2l x k)
            if move == "right":
                Q, R = cp.linalg.qr(P)
                B = R @ Vk.conj().T
                B /= cp.linalg.norm(B)
                self.A[i] = Q.reshape(l, 2, k)
                self.A[i + 1] = B.reshape(k, 2, r)
            else:
                P /= cp.linalg.norm(P)
                self.A[i] = P.reshape(l, 2, k)
                self.A[i + 1] = Vk.conj().T.reshape(k, 2, r)
        else:                                                  # Vk: left singular vectors (2l x k)
            P = Vk.conj().T @ M                                # S V^H, (k x 2r)
            if move == "right":
                P /= cp.linalg.norm(P)
                self.A[i] = Vk.reshape(l, 2, k)
                self.A[i + 1] = P.reshape(k, 2, r)
            else:
                Q, R = cp.linalg.qr(P.conj().T)                # (2r x k), (k x k)
                B = Vk @ R.conj().T
                B /= cp.linalg.norm(B)
                self.A[i] = B.reshape(l, 2, k)
                self.A[i + 1] = Q.conj().T.reshape(k, 2, r)
        self.center = i + 1 if move == "right" else i
        self.chi_peak = max(self.chi_peak, k)

    def overlap(self, other) -> complex:
        cp = _cp()
        E = cp.ones((1, 1), dtype=complex)
        for a, b in zip(self.A, other.A):
            E = cp.tensordot(cp.tensordot(E, a.conj(), axes=(0, 0)), b, axes=([0, 1], [0, 1]))
        return complex(E[0, 0].get())


def to_gpu(psi) -> GMPS:
    return psi if isinstance(psi, GMPS) else GMPS.from_cpu(psi)


def snapshot_gpu(psi: GMPS) -> GMPS:
    out = copy.copy(psi)
    out.A = list(psi.A)
    return out


def apply_layered_gpu(psi: GMPS, circuit: Layered, edges=None) -> GMPS:
    """mps.apply_layered on a GMPS."""
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
        if not np.any(np.asarray(angles[:3], dtype=float)):
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


def apply_dagger_gpu(psi: GMPS, cell, parity: int, angles, pair) -> None:
    """mps_train._apply_dagger on a GMPS."""
    N = psi.N
    bonds = list(range(parity, N - 1, 2))
    if parity == 1:
        g_last, g_first = pair
        psi.apply_1q(np.asarray(g_last).conj().T, N - 1)
        psi.apply_1q(np.asarray(g_first).conj().T, 0)
    if not np.any(np.asarray(angles[:3], dtype=float)):
        g1, g2 = cell.one_qubit_factor(angles)
        for i in bonds:
            psi.apply_1q(g1.conj().T, i)
            psi.apply_1q(g2.conj().T, i + 1)
        return
    Ud = cell.gate(angles).conj().T
    if psi.center > N // 2:
        for i in reversed(bonds):
            psi.apply_2q(Ud, i, move="left")
    else:
        for i in bonds:
            psi.apply_2q(Ud, i, move="right")


def _blocks(N: int, parity: int) -> list[tuple[int, int]]:
    if parity == 0:
        return [(i, 2) for i in range(0, N, 2)]
    return [(0, 1)] + [(i, 2) for i in range(1, N - 2, 2)] + [(N - 1, 1)]


def sublayer_environments_gpu(a: GMPS, b: GMPS, parity: int, G, pair=None):
    """mps_train.sublayer_environments on GMPSs; returns numpy (E_bulk, E_last, E_first) and the amplitude."""
    cp = _cp()
    N = a.N
    blocks = _blocks(N, parity)
    Gd = _dev(G)
    if parity == 1:
        g_last, g_first = (_dev(g) for g in pair)
    gates, kets, bras = [], [], []
    for i, w in blocks:
        if w == 2:
            th = cp.tensordot(a.A[i], a.A[i + 1], axes=(2, 0))
            br = cp.tensordot(b.A[i], b.A[i + 1], axes=(2, 0)).conj()
            kets.append(th.reshape(th.shape[0], 4, th.shape[3]))
            bras.append(br.reshape(br.shape[0], 4, br.shape[3]))
            gates.append(Gd)
        else:
            kets.append(a.A[i])
            bras.append(b.A[i].conj())
            gates.append(g_first if i == 0 else g_last)
    R = [None] * len(blocks)
    r = cp.ones((1, 1), dtype=complex)
    for j in range(len(blocks) - 1, -1, -1):
        R[j] = r
        X = cp.tensordot(kets[j], r, axes=(2, 1))
        X = cp.tensordot(gates[j], X, axes=(1, 1))
        r = cp.tensordot(bras[j], X, axes=([1, 2], [0, 2]))
    E_bulk = cp.zeros((4, 4), dtype=complex)
    E_last = E_first = None
    left = cp.ones((1, 1), dtype=complex)
    for j, (i, w) in enumerate(blocks):
        Y = cp.tensordot(left, kets[j], axes=(1, 0))
        E = cp.tensordot(bras[j], cp.tensordot(Y, R[j], axes=(2, 1)), axes=([0, 2], [0, 2]))
        if w == 2:
            E_bulk += E
        elif i == 0:
            E_first = E
        else:
            E_last = E
        W = cp.tensordot(gates[j], Y, axes=(1, 1))
        left = cp.tensordot(bras[j], W, axes=([0, 1], [1, 0]))
    return (E_bulk.get(), None if E_last is None else E_last.get(), None if E_first is None else E_first.get(),
            complex(left[0, 0].get()))


def run_input_gpu(cell, x, e, vec, target, chi_max: int, cutoff: float, grad: bool) -> dict:
    """mps_train._run_input on the GPU: the same record (numpy and Python numbers); ``target`` an MPS or GMPS."""
    from snapevo.edges import pairs
    cp = _cp()
    t0 = time.perf_counter()
    target = to_gpu(target)
    circuit = Layered.brickwork(cell, x)
    edge = iter(pairs(e))
    layers = [(p, a, next(edge) if p == 1 else None) for p, a in circuit.layers]
    psi = GMPS.product(vec, chi_max=chi_max, cutoff=cutoff)
    store = []
    for parity, angles, pair in layers:
        if grad:
            store.append(snapshot_gpu(psi))
        apply_layered_gpu(psi, Layered(cell, [(parity, angles)]), None if pair is None else [pair])
    A = target.overlap(psi)
    vv = target.overlap(target)
    norm = abs(vv * psi.overlap(psi))
    out = {"A": A, "norm": norm, "F": abs(A) ** 2 / norm, "discarded_u": psi.discarded, "chi_u": psi.chi_peak,
           "seconds_forward": time.perf_counter() - t0}
    if not grad:
        return out
    del psi
    t1 = time.perf_counter()
    D, M = len(layers), sum(1 for p, _, _ in layers if p == 1)
    E_bulk = np.zeros((D, 4, 4), dtype=complex)
    E_last = np.zeros((M, 2, 2), dtype=complex)
    E_first = np.zeros((M, 2, 2), dtype=complex)
    nv = math.sqrt(abs(vv))
    b = snapshot_gpu(target)
    b.A[b.center] = b.A[b.center] / nv
    b.chi_max, b.cutoff, b.discarded = int(chi_max), float(cutoff), 0.0
    b.chi_peak = max(a.shape[2] for a in b.A)
    m, dev = M, 0.0
    for l in range(D - 1, -1, -1):
        parity, angles, pair = layers[l]
        Eb, El, Ef, amp = sublayer_environments_gpu(store[l], b, parity, cell.gate(angles), pair)
        store[l] = None
        E_bulk[l] = Eb
        if parity == 1:
            m -= 1
            E_last[m], E_first[m] = El, Ef
        dev = max(dev, abs(amp * nv - A))
        if l:
            apply_dagger_gpu(b, cell, parity, angles, pair)
    E_bulk *= nv
    E_last *= nv
    E_first *= nv
    del store
    cp.get_default_memory_pool().free_all_blocks()
    out.update(E_bulk=E_bulk, E_last=E_last, E_first=E_first, discarded_b=b.discarded, chi_b=b.chi_peak,
               amp_dev=dev / max(abs(A), 1e-300), seconds_backward=time.perf_counter() - t1)
    return out


# ----------------------------------------------------------------------------- curve environments (chain_snap)

def env_states_gpu(cell, layers, vec, target, chi_max: int, cutoff: float, want: set):
    """chain_snap._env_states on the GPU: a_l and b_l (GMPS) for l in want."""
    D = len(layers)
    lo, hi = min(want), max(want)
    a, b = {}, {}
    psi = GMPS.product(vec, chi_max=chi_max, cutoff=cutoff)
    for l in range(hi + 1):
        if l in want:
            a[l] = snapshot_gpu(psi)
        if l < hi:
            parity, angles, pair = layers[l]
            apply_layered_gpu(psi, Layered(cell, [(parity, angles)]), None if pair is None else [pair])
    target = to_gpu(target)
    nv = math.sqrt(abs(target.overlap(target)))
    bra = snapshot_gpu(target)
    bra.A[bra.center] = bra.A[bra.center] / nv
    bra.chi_max, bra.cutoff, bra.discarded = int(chi_max), float(cutoff), 0.0
    bra.chi_peak = max(t.shape[2] for t in bra.A)
    for l in range(D - 1, lo - 1, -1):
        if l in want:
            b[l] = snapshot_gpu(bra)
        if l > lo:
            parity, angles, pair = layers[l]
            apply_dagger_gpu(bra, cell, parity, angles, pair)
    return a, b


def sandwich_gpu(a: GMPS, b: GMPS, parity: int, G, pair) -> np.ndarray:
    """chain_snap._sandwich on the GPU (G: (Q, 4, 4) numpy; pair entries (2, 2) or (Q, 2, 2)); numpy (Q,)."""
    cp = _cp()
    Gd = cp.asarray(np.asarray(G, dtype=complex))
    Q = Gd.shape[0]
    L = cp.ones((Q, 1, 1), dtype=complex)
    pr = None if pair is None else tuple(cp.asarray(np.asarray(g, dtype=complex)) for g in pair)
    for i, w in _blocks(a.N, parity):
        if w == 2:
            ket = cp.tensordot(a.A[i], a.A[i + 1], axes=(2, 0))
            bra = cp.tensordot(b.A[i], b.A[i + 1], axes=(2, 0)).conj()
            l, r, lb, rb = ket.shape[0], ket.shape[3], bra.shape[0], bra.shape[3]
            Y = (L.reshape(Q * lb, l) @ ket.reshape(l, 4 * r)).reshape(Q, lb, 4, r)
            Y = cp.matmul(Gd, Y.transpose(0, 2, 1, 3).reshape(Q, 4, lb * r))
            Y = Y.reshape(Q, 4, lb, r).transpose(2, 1, 0, 3).reshape(lb * 4, Q * r)
            L = (bra.reshape(lb * 4, rb).T @ Y).reshape(rb, Q, r).transpose(1, 0, 2)
        else:
            g = pr[1] if i == 0 else pr[0]
            bra = b.A[i].conj()
            if g.ndim == 2:
                ket = cp.tensordot(g, a.A[i], axes=(1, 1)).transpose(1, 0, 2)
                l, r, lb, rb = ket.shape[0], ket.shape[2], bra.shape[0], bra.shape[2]
                Y = (L.reshape(Q * lb, l) @ ket.reshape(l, 2 * r)).reshape(Q, lb, 2, r)
            else:
                ket = a.A[i]
                l, r, lb, rb = ket.shape[0], ket.shape[2], bra.shape[0], bra.shape[2]
                Y = (L.reshape(Q * lb, l) @ ket.reshape(l, 2 * r)).reshape(Q, lb, 2, r)
                Y = cp.matmul(g, Y.transpose(0, 2, 1, 3).reshape(Q, 2, lb * r))
                Y = Y.reshape(Q, 2, lb, r).transpose(0, 2, 1, 3)
            Y = Y.transpose(1, 2, 0, 3).reshape(lb * 2, Q * r)
            L = (bra.reshape(lb * 2, rb).T @ Y).reshape(rb, Q, r).transpose(1, 0, 2)
    return L[:, 0, 0].get()
