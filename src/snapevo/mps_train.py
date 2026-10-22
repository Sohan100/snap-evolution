"""Train the brickwork on the open chain it is evaluated on, by MPS, with adjoint gradients.

train.py trains a cell on a 10-site ring with state vectors and edges.py fits
the end rotations afterwards. Here the bulk angles and the edge angles are
trained together, on an open chain of N sites (N = 50 is the intended size),
against MPS references, as Gibbs and Cincio do with MPO environments (Quantum
2025) and as Lukasz Cincio recommended for us.

Parameters. z = concat(x, e). The bulk angles x, D n_angles of them, are
translation invariant: one set per sublayer, shared by every bond of that
sublayer (Layered.brickwork). The edge angles e, 6 per parity-1 sublayer, are
the ZYZ angles of the two end gates that replace the lost wrap bond, in the
order of edges.pairs: (gate on site N-1, gate on site 0).

Cost. For product inputs |phi_k> and targets |v_k> ~ e^{-iHt}|phi_k>,

    c = 1 - (1/n) sum_k F_k^{1/N},   F_k = |A_k|^2 / |<v_k|v_k><u_k|u_k>|,   A_k = <v_k|u_k>,

with |u_k> = U|phi_k> run by mps.apply_layered one sublayer at a time, so the
gates, their order, the sweep direction and every truncation are those of
apply_layered on the whole circuit, and F_k is the number evaluate.py would get.

Gradient (adjoint / environment method). Write U = L_{D-1} ... L_0, with L_l one
sublayer. For every l,

    A = <b_l| L_l |a_l>,   |a_l> = L_{l-1} ... L_0 |phi>,   |b_l> = L_{l+1}^dag ... L_{D-1}^dag |v>.

The forward pass stores every |a_l>; the backward pass applies L_l^dag to the
target, one sublayer at a time. The gates of one sublayer act on disjoint sites,
so for each gate G of L_l (4x4 on a bond, 2x2 on an end site) one sweep of
left and one of right environments over <b_l| L_l |a_l> gives its environment

    E[p, q] = dA / dG[p, q]     (G's entries taken as independent complex numbers;
                                 p the output and q the input index, qubits
                                 ordered (i, i+1) as in kron(g_i, g_{i+1})),

so that dA = sum_{p,q} E[p, q] dG[p, q] = Tr(E^T dG). Then dF = 2 Re(conj(A) dA)
/ (<v|v><u|u>) (the norms do not depend on z: every gate is unitary, and the MPS
renormalizes after each truncation), and

    dc/dz = Re sum_gates sum_{p,q} W[p, q] dG[p, q]/dz,
    W = -(1/n) sum_k (2/N) F_k^{1/N-1} conj(A_k) E_k / (<v_k|v_k><u_k|u_k>).

The backward pass runs the bra as v/|v| (every apply_2q renormalizes an MPS to
norm 1, so a bra of another norm would change scale at its first two-qubit
sublayer) and multiplies the environments by |v| at the end, so targets of any
norm and phase are fine. They must be in canonical form about MPS.center, as
reference_state, MPS.load and mps_from_dense give them; OpenChainCost checks
this (a hand-built MPS() has center 0 whatever its tensors are).

The bulk gates of a sublayer share its angles, so their W are summed over the
bonds of the sublayer. dG/dz goes through one JAX vjp of cell.jax_gate, the gate
function jaxcost.py trains with (vmapped over the sublayers), and of jax_zyz,
the JAX mirror of edges.u_zyz. The workers that run the MPSs never import JAX.

Single-qubit layers (coupling angles exactly 0, as in the baselines' field
layers). The forward pass applies them through cell.one_qubit_factor, exactly
as apply_layered does. Their gradient uses the bond environments with the full
4x4 cell gate, which is then exactly kron(g1, g2), so the derivatives along the
coupling angles, which leave the single-qubit manifold, are included too.

Truncation. Without truncation (cutoff 0, chi_max large) the gradient is exact.
With truncation, the forward states |a_l> and the bra states |b_l> are each
truncated as apply_layered truncates (the bra with the cost's chi_max and
cutoff), and the gradient is that of the sandwiches <b_l| L_l |a_l> at fixed
truncated states, not a derivative through the SVDs. Its error is of the order
of the discarded weights; diagnostics() reports both and how far the sandwich
amplitudes stray from A. Measured at N = 50, t = 2, D = 17, 3 inputs, near a
trained point (gradient norm 1.5e-5 at h_x = 0, 5.8e-5 at h_x = 1.6): against
cutoff 1e-16, the gradient at cutoff 1e-12 is off by 3.9e-4 (h_x = 0) and
7.2e-5 (h_x = 1.6) in norm, 1.2e-2 and 1.3e-3 in the worst component; at
cutoff 1e-14 by 1.5e-5 and 5.6e-6 in norm, for 1.6x the time (4 threads).
Near convergence, cutoff 1e-14 is the safer choice.

Memory. Per input, the D forward states are kept until the backward pass
reaches them, so the peak is about the sum of D MPS sizes (16 bytes per
complex entry). Inputs are independent: with processes > 1 they are split
round-robin over that many worker processes (spawned, one concurrent.futures
executor each), each of which holds only its own inputs and targets. Each
worker gets ``threads`` BLAS/OpenMP threads (default 1, single-threaded).

Restarts. train_open resumes from its checkpoint without L-BFGS-B's curvature
pairs, so its first iteration after a (re)start is a steepest-descent step.
Near floors of 1e-10 such a step can gain less than train.py's ftol of 1e-16
(train.py's docstring), and L-BFGS-B's own f-reduction test would then end the
run at its first line search. train_open therefore applies that test itself,
from the RESTART_GRACE + 1-th iteration after each start or resume, and gives
L-BFGS-B ftol = 0. The iterates are those of train.py's options; only a stop
in the first RESTART_GRACE iterations of a start is not taken. Its
on_iteration hook lets a caller evaluate or stop at any iteration without
restarting the optimizer.
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import copy
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import scipy.optimize as sopt

from snapevo.cells import Cell, get_cell
from snapevo.circuits import Layered, product_vectors
from snapevo.edges import pairs, start_angles
from snapevo.evaluate import ref_path
from snapevo.hamiltonian import XXZ
from snapevo.mps import MPS, apply_layered, reference_state

__all__ = ["OpenChainCost", "open_chain_problem", "train_open", "init_from_ring", "ring_angles",
           "mps_from_dense", "sublayer_environments", "jax_zyz", "REF_DT", "REF_CUTOFF", "REF_CHI",
           "TRAIN_OPTIONS", "TRACE_EVERY", "RESTART_GRACE"]

REF_DT = 0.1
REF_CUTOFF = 1e-14
REF_CHI = 2048
"""Step, cutoff and bond cap of the order-4 references the targets are built from."""

TRAIN_OPTIONS = {"ftol": 1e-16, "gtol": 1e-14, "maxfun": 400000}
"""L-BFGS-B options of train.py (maxiter is train_open's argument)."""

TRACE_EVERY = 25
"""train_open records (iteration, c, seconds) every this many iterations."""

RESTART_GRACE = 2
"""Iterations after a start or resume in which train_open does not apply the f-reduction (ftol) test."""

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


# ----------------------------------------------------------------------------- MPS pieces

def _snapshot(psi: MPS) -> MPS:
    """A copy of psi that later gates on psi cannot change.

    MPS methods never write into a tensor, they replace list entries, so copying
    the list is enough.
    """
    out = copy.copy(psi)
    out.A = list(psi.A)
    return out


def _single_qubit(angles) -> bool:
    """The test apply_layered uses: no coupling, so the sublayer is single-qubit gates only."""
    return not np.any(np.asarray(angles[:3], dtype=float))


def _canonical_mismatch(psi: MPS) -> float:
    """| |A[center]|^2 - <psi|psi> | / <psi|psi>, inf for psi = 0.

    Rounding-small in canonical form about psi.center (at most 4e-14 on 162
    N = 50 and 100 references); a hand-built MPS() has center 0 whatever its
    tensors are, so it may fail.
    """
    vv = abs(psi.overlap(psi))
    c = psi.A[psi.center]
    return abs(float(np.vdot(c, c).real) - vv) / vv if vv > 0 else math.inf


def mps_from_dense(vector, chi_max: int = 10 ** 6, cutoff: float = 0.0) -> MPS:
    """Exact MPS of a dense state vector (site 0 the most significant qubit), by successive SVDs.

    No singular value is dropped; chi_max and cutoff only set what later gates
    on the returned MPS truncate with. Small N only (tests).
    """
    v = np.asarray(vector, dtype=complex).reshape(-1)
    N = int(round(math.log2(v.size)))
    if 2 ** N != v.size:
        raise ValueError(f"length {v.size} is not a power of 2")
    tensors, rest, left = [], v.reshape(1, -1), 1
    for _ in range(N - 1):
        u, s, vh = np.linalg.svd(rest.reshape(left * 2, -1), full_matrices=False)
        tensors.append(u.reshape(left, 2, s.size))
        rest, left = s[:, None] * vh, s.size
    tensors.append(rest.reshape(left, 2, 1))
    psi = MPS(tensors, chi_max=chi_max, cutoff=cutoff)
    psi.center = N - 1                     # every tensor but the last is a left isometry
    return psi


def _blocks(N: int, parity: int) -> list[tuple[int, int]]:
    """(first site, width) of the gates of one sublayer, left to right; the widths cover the chain."""
    if parity == 0:
        return [(i, 2) for i in range(0, N, 2)]
    return [(0, 1)] + [(i, 2) for i in range(1, N - 2, 2)] + [(N - 1, 1)]


def sublayer_environments(a: MPS, b: MPS, parity: int, G: np.ndarray, pair=None):
    """Environments of every gate of one sublayer in <b| L |a>, summed where the gates are shared.

    L applies the 4x4 gate G to every bond (i, i+1) of the given parity and, for
    parity 1, pair = (g_last, g_first) to sites N-1 and 0. Returns (E_bulk, E_last,
    E_first, amp): E_bulk is the 4x4 environment summed over the bonds, E_last and
    E_first are the 2x2 environments of the end gates (None for parity 0), and amp
    is <b| L |a>. Convention: E[p, q] = d<b|L|a> / dG[p, q], so a change dG of
    every bulk gate changes the amplitude by Tr(E_bulk^T dG).
    """
    N = a.N
    blocks = _blocks(N, parity)
    if parity == 1:
        g_last, g_first = (np.asarray(g, dtype=complex) for g in pair)
    gates, kets, bras = [], [], []
    for i, w in blocks:
        if w == 2:
            th = np.tensordot(a.A[i], a.A[i + 1], axes=(2, 0))
            br = np.tensordot(b.A[i], b.A[i + 1], axes=(2, 0)).conj()
            kets.append(th.reshape(th.shape[0], 4, th.shape[3]))
            bras.append(br.reshape(br.shape[0], 4, br.shape[3]))
            gates.append(G)
        else:
            kets.append(a.A[i])
            bras.append(b.A[i].conj())
            gates.append(g_first if i == 0 else g_last)
    # right environments R[j]: everything right of block j, shape (chi_b, chi_a)
    R = [None] * len(blocks)
    r = np.ones((1, 1), dtype=complex)
    for j in range(len(blocks) - 1, -1, -1):
        R[j] = r
        X = np.tensordot(kets[j], r, axes=(2, 1))                  # (l, s, rb)
        X = np.tensordot(gates[j], X, axes=(1, 1))                 # (p, l, rb)
        r = np.tensordot(bras[j], X, axes=([1, 2], [0, 2]))        # (lb, l)
    E_bulk = np.zeros((4, 4), dtype=complex)
    E_last = E_first = None
    left = np.ones((1, 1), dtype=complex)
    for j, (i, w) in enumerate(blocks):
        Y = np.tensordot(left, kets[j], axes=(1, 0))               # (lb, s, r)
        E = np.tensordot(bras[j], np.tensordot(Y, R[j], axes=(2, 1)), axes=([0, 2], [0, 2]))   # (p, s)
        if w == 2:
            E_bulk += E
        elif i == 0:
            E_first = E
        else:
            E_last = E
        W = np.tensordot(gates[j], Y, axes=(1, 1))                 # (p, lb, r)
        left = np.tensordot(bras[j], W, axes=([0, 1], [1, 0]))     # (rb, r)
    return E_bulk, E_last, E_first, complex(left[0, 0])


def _apply_dagger(psi: MPS, cell: Cell, parity: int, angles, pair) -> None:
    """Apply one sublayer's inverse to psi in place, mirroring apply_layered."""
    N = psi.N
    bonds = list(range(parity, N - 1, 2))
    if parity == 1:
        g_last, g_first = pair
        psi.apply_1q(np.asarray(g_last).conj().T, N - 1)
        psi.apply_1q(np.asarray(g_first).conj().T, 0)
    if _single_qubit(angles):
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


def _run_input(cell: Cell, x: np.ndarray, e: np.ndarray, vec, target: MPS, chi_max: int, cutoff: float,
               grad: bool) -> dict:
    """Forward (and, with grad, backward and environments) for one input."""
    t0 = time.perf_counter()
    circuit = Layered.brickwork(cell, x)
    edge = iter(pairs(e))
    layers = [(p, a, next(edge) if p == 1 else None) for p, a in circuit.layers]
    psi = MPS.product(vec, chi_max=chi_max, cutoff=cutoff)
    store = []
    for parity, angles, pair in layers:
        if grad:
            store.append(_snapshot(psi))
        apply_layered(psi, Layered(cell, [(parity, angles)]), None if pair is None else [pair])
    A = target.overlap(psi)
    vv = target.overlap(target)
    norm = abs(vv * psi.overlap(psi))
    out = {"A": A, "norm": norm, "F": abs(A) ** 2 / norm,           # evaluate.py's F, operation for operation
           "discarded_u": psi.discarded, "chi_u": psi.chi_peak, "seconds_forward": time.perf_counter() - t0}
    if not grad:
        return out
    del psi
    t1 = time.perf_counter()
    D, M = len(layers), sum(1 for p, _, _ in layers if p == 1)
    E_bulk = np.zeros((D, 4, 4), dtype=complex)
    E_last = np.zeros((M, 2, 2), dtype=complex)
    E_first = np.zeros((M, 2, 2), dtype=complex)
    # the bra runs as v/|v| (apply_2q renormalizes to 1), scaled at its center so it stays canonical;
    # the environments are scaled back to those of v below (see the module docstring)
    nv = math.sqrt(abs(vv))
    b = _snapshot(target)
    b.A[b.center] = b.A[b.center] / nv
    b.chi_max, b.cutoff, b.discarded = int(chi_max), float(cutoff), 0.0
    b.chi_peak = max(a.shape[2] for a in b.A)
    m, dev = M, 0.0
    for l in range(D - 1, -1, -1):
        parity, angles, pair = layers[l]
        Eb, El, Ef, amp = sublayer_environments(store[l], b, parity, circuit.cell.gate(angles), pair)
        store[l] = None
        E_bulk[l] = Eb
        if parity == 1:
            m -= 1
            E_last[m], E_first[m] = El, Ef
        dev = max(dev, abs(amp * nv - A))
        if l:
            _apply_dagger(b, cell, parity, angles, pair)
    E_bulk *= nv
    E_last *= nv
    E_first *= nv
    out.update(E_bulk=E_bulk, E_last=E_last, E_first=E_first, discarded_b=b.discarded, chi_b=b.chi_peak,
               amp_dev=dev / max(abs(A), 1e-300), seconds_backward=time.perf_counter() - t1)
    return out


# ----------------------------------------------------------------------------- workers

_WORKER: dict = {}


def _worker_init(cell_name: str, inputs, targets, chi_max: int, cutoff: float) -> None:
    _WORKER.update(cell=get_cell(cell_name), inputs=inputs, targets=targets, chi_max=chi_max, cutoff=cutoff)


def _worker_ping() -> int:
    return os.getpid()


def _blas_threads() -> int | None:
    """Threads of numpy's OpenBLAS in this process (scipy-openblas builds), None if it cannot be asked."""
    import ctypes
    import re
    try:
        with open("/proc/self/maps") as fh:
            libs = sorted({m.group(1) for line in fh if (m := re.search(r"(/\S*numpy\S*openblas\S*\.so\S*)", line))})
        for path in libs:
            lib = ctypes.CDLL(path)
            for name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
                f = getattr(lib, name, None)
                if f is not None:
                    f.restype = ctypes.c_int
                    return int(f())
    except OSError:
        pass
    return None


def _worker_threads() -> dict:
    """What a worker runs with: its OPENBLAS_NUM_THREADS and numpy's OpenBLAS thread count."""
    return {"pid": os.getpid(), "env": os.environ.get("OPENBLAS_NUM_THREADS"), "blas": _blas_threads()}


def _worker_run(x, e, grad: bool) -> list[dict]:
    w = _WORKER
    return [_run_input(w["cell"], x, e, v, tg, w["chi_max"], w["cutoff"], grad)
            for v, tg in zip(w["inputs"], w["targets"])]


@contextlib.contextmanager
def _threads_env(threads: int = 1):
    """Processes started inside this block get ``threads`` BLAS/OpenMP threads each."""
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: str(int(threads)) for k in _THREAD_VARS})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _single_threaded_env():
    """Processes started inside this block get one BLAS/OpenMP thread each."""
    return _threads_env(1)


def _spawn_executors(n_workers: int, initargs_of, threads: int = 1) -> list[cf.ProcessPoolExecutor]:
    """n single-process executors (spawn) with ``threads`` BLAS threads each, started with initargs_of(w), now."""
    ctx = mp.get_context("spawn")
    with _threads_env(threads):
        pools = [cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_worker_init, initargs=initargs_of(w))
                 for w in range(n_workers)]
        for f in [p.submit(_worker_ping) for p in pools]:       # the process starts on first submit
            f.result()
    return pools


# ----------------------------------------------------------------------------- JAX gate maps

def jax_zyz(a, b, c):
    """edges.u_zyz in JAX: Rz(a) Ry(b) Rz(c)."""
    import jax.numpy as jnp
    cb, sb = jnp.cos(b / 2), jnp.sin(b / 2)
    m, p = jnp.exp(-0.5j * (a + c)), jnp.exp(-0.5j * (a - c))
    return jnp.stack([jnp.stack([m * cb, -p * sb]), jnp.stack([jnp.conj(p) * sb, jnp.conj(m) * cb])])


def _jax_maps(cell: Cell, D: int, M: int):
    """Jitted z -> gates and (z, cotangents) -> dc/dz, the vjp of the gate map."""
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    na = cell.n_angles
    nx = D * na

    def gates(z):
        G = jax.vmap(cell.jax_gate)(z[:nx].reshape(D, na))
        if M == 0:
            return (G,)
        e = z[nx:].reshape(M, 6)
        return (G, jax.vmap(lambda r: jax_zyz(r[0], r[1], r[2]))(e),       # site N-1
                jax.vmap(lambda r: jax_zyz(r[3], r[4], r[5]))(e))           # site 0

    def parts(z):
        return tuple(f(g) for g in gates(z) for f in (jnp.real, jnp.imag))

    @jax.jit
    def pull(z, Ws):
        # Re sum W * G = sum ReW ReG - ImW ImG: the cotangents of (Re G, Im G) are (Re W, -Im W)
        _, vjp = jax.vjp(parts, z)
        return vjp(tuple(c for W in Ws for c in (jnp.real(W), -jnp.imag(W))))[0]

    return jax.jit(gates), pull


# ----------------------------------------------------------------------------- the cost

class OpenChainCost:
    """c and dc/dz of the open-chain brickwork on MPS inputs and targets (see the module docstring).

    Args:
        model: The Hamiltonian the targets come from (kept for the record).
        cell: The brickwork cell.
        t: The time (kept for the record).
        D: Depth, the number of sublayers.
        N: Sites of the open chain (even).
        inputs: (n, N, 2) single-site input vectors, as circuits.product_vectors gives.
        targets: n target MPSs, mps.reference_state outputs for those inputs (any norm and
            phase; canonical about their center, else ValueError).
        chi_max, cutoff: Truncation of the circuit's MPS and of the bra in the backward pass.
        processes: Worker processes over the inputs; None or 1 runs in this process
            (with threads 1).
        threads: BLAS/OpenMP threads of each worker process (default 1, as before this
            option existed). With threads > 1 the inputs always run in spawned workers,
            even with processes 1, since this process's BLAS is already set up.

    Use as a context manager, or call close(), to stop the workers. The workers are
    spawned, so with processes > 1 the calling script must keep its top-level code
    under ``if __name__ == "__main__":`` (open_chain_problem's workers too).
    """

    def __init__(self, model: XXZ, cell: Cell, t: float, D: int, N: int, inputs, targets,
                 chi_max: int = 1024, cutoff: float = 1e-12, processes: int | None = None, threads: int = 1):
        inputs = np.asarray(inputs, dtype=complex)
        if inputs.ndim != 3 or inputs.shape[1:] != (N, 2):
            raise ValueError(f"inputs must have shape (n, {N}, 2), not {inputs.shape}")
        if len(targets) != inputs.shape[0]:
            raise ValueError(f"{len(targets)} targets for {inputs.shape[0]} inputs")
        if any(tg.N != N for tg in targets):
            raise ValueError(f"every target must have {N} sites")
        if N % 2 or N < 4:
            raise ValueError("the brickwork needs an even number of sites, at least 4")
        for k, tg in enumerate(targets):
            if not _canonical_mismatch(tg) <= 1e-10:
                raise ValueError(f"target {k} is zero or not in canonical form about its center {tg.center}")
        self.model, self.cell, self.t, self.D, self.N = model, cell, float(t), int(D), int(N)
        self.inputs, self.targets = inputs, list(targets)
        self.n = inputs.shape[0]
        self.chi_max, self.cutoff = int(chi_max), float(cutoff)
        self.n_bulk = self.D * cell.n_angles
        self.n_odd = self.D // 2
        self.n_params = self.n_bulk + 6 * self.n_odd
        self.processes = max(1, min(int(processes or 1), self.n))
        if int(threads) < 1:
            raise ValueError(f"threads must be at least 1, not {threads}")
        self.threads = int(threads)
        self._groups = [list(range(w, self.n, self.processes)) for w in range(self.processes)]
        self._pools: list[cf.ProcessPoolExecutor] | None = None
        self._gates = self._pull = None
        self.ncalls = 0
        self.last: dict = {}

    # -- parameters
    def split(self, z) -> tuple[np.ndarray, np.ndarray]:
        """(bulk angles (D, n_angles), edge angles (D // 2, 6)) of a flat z."""
        z = np.asarray(z, dtype=float).reshape(-1)
        if z.size != self.n_params:
            raise ValueError(f"z has {z.size} entries, expected {self.n_params}")
        return z[: self.n_bulk].reshape(self.D, -1), z[self.n_bulk:].reshape(self.n_odd, 6)

    def circuit(self, z) -> tuple[Layered, list]:
        """(Layered, edge pairs) of z, as evaluate.mps_infidelity and Layered.apply(bc="open") take them."""
        x, e = self.split(z)
        return Layered.brickwork(self.cell, x.reshape(-1)), pairs(e)

    # -- evaluation
    def _results(self, z, grad: bool) -> list[dict]:
        x, e = self.split(z)
        x = x.reshape(-1)
        if self.processes == 1 and self.threads == 1:
            return [_run_input(self.cell, x, e, v, tg, self.chi_max, self.cutoff, grad)
                    for v, tg in zip(self.inputs, self.targets)]
        futures = [p.submit(_worker_run, x, e, grad) for p in self._workers()]
        out = [None] * self.n
        for keys, f in zip(self._groups, futures):
            for k, r in zip(keys, f.result()):
                out[k] = r
        return out

    def _fidelities(self, rows: list[dict]) -> tuple[float, np.ndarray, np.ndarray]:
        A = np.array([r["A"] for r in rows])
        F = np.maximum(np.array([r["F"] for r in rows]), 1e-300)
        c = float(-np.mean(np.expm1(np.log(F) / self.N)))
        return c, F, A

    def _record(self, rows, F, seconds, grad):
        keys = ["discarded_u", "chi_u", "seconds_forward"] + (
            ["discarded_b", "chi_b", "amp_dev", "seconds_backward"] if grad else [])
        self.last = {"F": F, "seconds": seconds, **{k: np.array([r[k] for r in rows]) for k in keys}}

    def value(self, z) -> float:
        """c alone (forward pass only)."""
        t0 = time.perf_counter()
        rows = self._results(z, grad=False)
        c, F, _ = self._fidelities(rows)
        self._record(rows, F, time.perf_counter() - t0, grad=False)
        return c

    def value_and_grad(self, z) -> tuple[float, np.ndarray]:
        """(c, dc/dz) at the flat parameter vector z."""
        t0 = time.perf_counter()
        z = np.asarray(z, dtype=float).reshape(-1)
        rows = self._results(z, grad=True)
        c, F, A = self._fidelities(rows)
        nrm = np.array([r["norm"] for r in rows])
        w = -(2.0 / (self.n * self.N)) * np.exp((1.0 / self.N - 1.0) * np.log(F)) * np.conj(A) / nrm
        Ws = [np.einsum("k,kdpq->dpq", w, np.array([r[key] for r in rows]))
              for key in (("E_bulk",) if self.n_odd == 0 else ("E_bulk", "E_last", "E_first"))]
        if self._pull is None:
            self._gates, self._pull = _jax_maps(self.cell, self.D, self.n_odd)
        g = np.asarray(self._pull(z, tuple(Ws)), dtype=np.float64)
        self.ncalls += 1
        self._record(rows, F, time.perf_counter() - t0, grad=True)
        return c, g

    __call__ = value_and_grad

    def jax_gates(self, z):
        """The gates the vjp differentiates, as numpy arrays: (D, 4, 4) bulk, (M, 2, 2) site N-1, site 0."""
        if self._gates is None:
            self._gates, self._pull = _jax_maps(self.cell, self.D, self.n_odd)
        return tuple(np.asarray(g) for g in self._gates(np.asarray(z, dtype=float)))

    def diagnostics(self) -> dict:
        """Summary of the last evaluation: worst discarded weights, peak chi, sandwich drift, seconds."""
        L = self.last
        out = {"seconds": L.get("seconds"), "chi_u": int(L["chi_u"].max()),
               "discarded_u": float(L["discarded_u"].max()), "F_min": float(L["F"].min())}
        if "chi_b" in L:
            out.update(chi_b=int(L["chi_b"].max()), discarded_b=float(L["discarded_b"].max()),
                       amp_dev=float(L["amp_dev"].max()))
        return out

    # -- workers
    def _workers(self) -> list[cf.ProcessPoolExecutor]:
        if self._pools is None:
            self._pools = _spawn_executors(self.processes, lambda w: (
                self.cell.name, self.inputs[self._groups[w]], [self.targets[k] for k in self._groups[w]],
                self.chi_max, self.cutoff), threads=self.threads)
        return self._pools

    def worker_threads(self) -> list[dict]:
        """Per worker: pid, its OPENBLAS_NUM_THREADS and numpy's OpenBLAS thread count (starts the workers;
        [] when the cost runs in this process)."""
        if self.processes == 1 and self.threads == 1:
            return []
        return [f.result() for f in [p.submit(_worker_threads) for p in self._workers()]]

    def close(self) -> None:
        for p in self._pools or []:
            p.shutdown(wait=True, cancel_futures=True)
        self._pools = None

    def __enter__(self) -> OpenChainCost:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:                                          # noqa: BLE001 (interpreter shutdown)
            pass


# ----------------------------------------------------------------------------- inputs and targets

def _build_reference(model_dict: dict, t: float, vec, dt: float, chi_max: int, cutoff: float, path):
    model = XXZ.from_dict(model_dict)
    psi, meta = reference_state(model, t, vec, dt, chi_max, cutoff, cache=path)
    return None if path is not None else psi, meta


def open_chain_problem(model: XXZ, t: float, N: int, n: int, seed: int, cache_dir: Path | None,
                       processes: int | None = None, dt: float = REF_DT, cutoff: float = REF_CUTOFF,
                       chi_max: int = REF_CHI, log=None, threads: int = 1) -> tuple[np.ndarray, list[MPS]]:
    """Inputs product_vectors(N, n, seed) and their order-4 references e^{-iHt}|phi_k>.

    References are mps.reference_state at step dt, cutoff and chi_max, cached as
    cache_dir/<model.tag>/t<t>/ref_N.._s.._k.._dt.._c.._x...npz (evaluate.ref_path
    names), and read back from there when present, after checking that the file
    is for this model, time, size and step. cache_dir None caches nothing. Input
    k does not depend on n, so a larger n reuses the files of a smaller one.
    Missing references are built over ``processes`` workers with ``threads``
    BLAS threads each (default single-threaded); with threads > 1 they are built
    in spawned workers even with processes 1.
    """
    vecs = product_vectors(N, n, seed)
    folder = None if cache_dir is None else Path(cache_dir) / model.tag / f"t{t:g}"
    paths = [None if folder is None else ref_path(folder, N, seed, k, dt, cutoff, chi_max) for k in range(n)]
    targets: list[MPS | None] = [None] * n
    todo = []
    for k, p in enumerate(paths):
        if p is not None and p.exists():
            psi, meta = MPS.load(p)
            want = {"model": model.to_dict(), "t": float(t), "N": N, "dt": dt}
            got_t = meta.get("t")
            with contextlib.suppress(TypeError, ValueError):        # a missing or odd "t" is refused below
                got_t = float(got_t)
            got = {"model": meta.get("model"), "t": got_t, "N": meta.get("N"), "dt": meta.get("dt")}
            if got != want:
                raise ValueError(f"{p} is for {got}, not {want}")
            targets[k] = psi
        else:
            todo.append(k)
    t0 = time.time()
    procs = max(1, min(int(processes or 1), len(todo) or 1))
    if todo and procs == 1 and int(threads) == 1:
        built = [_build_reference(model.to_dict(), t, vecs[k], dt, chi_max, cutoff, paths[k]) for k in todo]
    elif todo:
        with _threads_env(threads):
            with cf.ProcessPoolExecutor(procs, mp_context=mp.get_context("spawn")) as ex:
                built = list(ex.map(_build_reference, *zip(*[(model.to_dict(), t, vecs[k], dt, chi_max, cutoff,
                                                                paths[k]) for k in todo])))
    for k, (psi, meta) in zip(todo, built if todo else []):
        targets[k] = psi if psi is not None else MPS.load(paths[k])[0]
        if log:
            log(f"reference k = {k}: chi peak {meta.get('chi_peak')}, discarded {meta.get('discarded', 0.0):.1e}, "
                f"{meta['seconds']:.1f}s")
    if log and todo:
        log(f"references: {len(todo)} built for {model.tag}, t = {t:g}, N = {N} in {time.time() - t0:.0f}s "
            f"on {procs} process(es) with {int(threads)} thread(s) each, {n - len(todo)} from the cache")
    return vecs, targets


# ----------------------------------------------------------------------------- training

def _reduction_stop(f_old: float, f_new: float, ftol: float) -> bool:
    """L-BFGS-B's f-reduction test: (f_old - f_new) <= factr * epsmch * max(|f_old|, |f_new|, 1)."""
    eps = np.finfo(float).eps
    return (f_old - f_new) <= (ftol / eps) * eps * max(abs(f_old), abs(f_new), 1.0)


def train_open(cost: OpenChainCost, z0, maxiter: int, log=print, checkpoint: Path | None = None,
               on_iteration=None):
    """L-BFGS-B on the open-chain cost with train.py's options.

    Returns (scipy result, trace), the trace a list of (iteration, c, seconds
    since the start) at iteration 0, every TRACE_EVERY iterations and at the end.

    checkpoint: a JSON file, replaced atomically every TRACE_EVERY iterations and
    at the end with the problem, z, c, iterations, evaluations and trace. If it
    exists when train_open starts, the run resumes from it (after checking it is
    for this model, cell, t, D, N and number of inputs): z0 is ignored, the
    iterations already done count against maxiter, and the trace and its clock
    continue. L-BFGS-B's curvature pairs are not saved, so its memory rebuilds
    over the first iterations after a resume. res.nit and res.nfev then count the
    whole run; if maxiter was already reached, nothing is evaluated.

    The f-reduction test (TRAIN_OPTIONS["ftol"]) is applied here, not by L-BFGS-B,
    and only from the RESTART_GRACE + 1-th iteration of this call (module
    docstring, Restarts); the result is then as with the test inside L-BFGS-B.
    L-BFGS-B still stops by itself on the projected gradient (gtol), on maxfun, on
    a failed line search and on an iteration that does not lower c at all.
    res.stop says why the run ended: "maxiter" (also when the hook asked to stop at
    that iteration), "ftol", "hook: <reason>" or "lbfgsb: <its message>";
    res.restart_nit is the iteration this call started at.

    on_iteration(nit, x, c, info): called after every iteration (and its
    checkpoint, if one is due), with the whole run's iteration count and info =
    {"nfev": evaluations so far, "seconds": the trace's clock}; the time it takes
    is left out of that clock. A non-empty string returned stops the run there:
    the checkpoint is saved at that iteration and res.stop is "hook: <string>".
    """
    t0 = time.time()
    trace: list[tuple[int, float, float]] = []
    problem = {"model": cost.model.to_dict(), "cell": cost.cell.name, "t": cost.t, "D": cost.D, "N": cost.N,
               "n": cost.n, "n_params": cost.n_params}
    z0 = np.asarray(z0, dtype=float)
    prior = {"nit": 0, "nfev": 0}
    if checkpoint is not None and Path(checkpoint).exists():
        ck = json.loads(Path(checkpoint).read_text())
        if ck.get("problem") != problem:
            raise ValueError(f"{checkpoint} is for {ck.get('problem')}, not {problem}")
        z0 = np.asarray(ck["z"], dtype=float)
        trace = [tuple(r) for r in ck["trace"]]
        prior = {"nit": int(ck["nit"]), "nfev": int(ck["nfev"])}
        t0 -= trace[-1][2]
        log(f"open chain: resumed {checkpoint} at iteration {prior['nit']}, c = {ck['c']:.6e}")
        if prior["nit"] >= maxiter:
            return sopt.OptimizeResult(x=z0, fun=float(ck["c"]), nit=prior["nit"], nfev=prior["nfev"], success=False,
                                       status=1, message="maxiter already reached in the checkpoint", stop="maxiter",
                                       restart_nit=prior["nit"]), trace
    state = {"nit": prior["nit"], "nfev": prior["nfev"], "f_prev": None, "stop": None}
    ftol = float(TRAIN_OPTIONS["ftol"])

    def save(z, c) -> None:
        if checkpoint is None:
            return
        p = Path(checkpoint)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"problem": problem, "nit": state["nit"], "nfev": state["nfev"], "c": float(c),
                                   "z": [float(v) for v in z], "trace": trace}))
        tmp.replace(p)

    def f(z):
        c, g = cost.value_and_grad(z)
        state["nfev"] += 1
        if state["f_prev"] is None:                              # L-BFGS-B evaluates z0 first
            state["f_prev"] = float(c)
        if not trace:
            trace.append((0, c, time.time() - t0))
            log(f"open chain: iteration 0, c = {c:.6e}, {cost.diagnostics()}")
        return c, g

    def callback(intermediate_result):
        nonlocal t0
        state["nit"] += 1
        x, c = intermediate_result.x, float(intermediate_result.fun)
        if state["nit"] % TRACE_EVERY == 0:
            trace.append((state["nit"], c, time.time() - t0))
            save(x, c)
            log(f"open chain: iteration {state['nit']}, c = {c:.6e}, {trace[-1][2]:.0f}s, {cost.diagnostics()}")
        f_old, state["f_prev"] = state["f_prev"], c
        # scipy's driver ends at maxiter before L-BFGS-B tests the reduction, so the test skips that iteration
        if (state["nit"] - prior["nit"] > RESTART_GRACE and state["nit"] < maxiter
                and _reduction_stop(f_old, c, ftol)):
            state["stop"] = "ftol"
            raise StopIteration
        if on_iteration is not None:
            h0 = time.time()
            why = on_iteration(state["nit"], np.array(x, dtype=float), c,
                               {"nfev": state["nfev"], "seconds": h0 - t0})
            t0 += time.time() - h0                               # the hook's time is not training time
            if why:
                state["stop"] = f"hook: {why}"
                raise StopIteration

    res = sopt.minimize(f, z0, jac=True, method="L-BFGS-B", callback=callback,
                        options={"maxiter": int(maxiter) - prior["nit"], **TRAIN_OPTIONS, "ftol": 0.0})
    res.nit, res.nfev = prior["nit"] + int(res.nit), prior["nfev"] + int(res.nfev)
    state["nit"], state["nfev"] = res.nit, res.nfev
    if res.nit >= maxiter:                                       # reached, whatever the hook said at that iteration
        res.stop = "maxiter"
    elif state["stop"] == "ftol":                                # as L-BFGS-B reports its own f-reduction stop
        res.status, res.success, res.stop = 0, True, "ftol"
        res.message = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH (applied by train_open)"
    elif state["stop"]:
        res.message, res.stop = f"STOP: {state['stop']}", state["stop"]
    else:
        res.stop = f"lbfgsb: {res.message}"
    res.restart_nit = prior["nit"]
    if trace[-1][0] != res.nit:
        trace.append((int(res.nit), float(res.fun), time.time() - t0))
    save(res.x, res.fun)
    log(f"open chain: done, c = {res.fun:.6e} after {res.nit} iterations and {res.nfev} evaluations, "
        f"{time.time() - t0:.0f}s ({res.message})")
    return res, trace


def ring_angles(model: XXZ, t: float, D: int, runs_dir: Path = Path("runs")) -> np.ndarray:
    """Flat ring-trained angles at depth D from runs_dir/<model.tag>/t<t>/cells.json."""
    from snapevo.train import load_cell
    return load_cell(Path(runs_dir) / model.tag / f"t{t:g}" / "cells.json", D)[0]


def init_from_ring(cell: Cell, x_ring, fit=None) -> np.ndarray:
    """z0 from ring-trained bulk angles and edge angles.

    ``fit`` is an edges.fit_edges record (its "angles"), a flat array of 6 per
    parity-1 sublayer, or None for edges.start_angles: the cell's own
    single-qubit factor, which is what apply_layered does with edges=None.
    """
    x = np.asarray(x_ring, dtype=float).reshape(-1)
    circuit = Layered.brickwork(cell, x)
    if fit is None:
        e = start_angles(circuit)
    else:
        e = np.asarray(fit["angles"] if isinstance(fit, dict) else fit, dtype=float).reshape(-1)
    want = 6 * sum(1 for p, _ in circuit.layers if p == 1)
    if e.size != want:
        raise ValueError(f"{e.size} edge angles for {want // 6} parity-1 sublayers")
    return np.concatenate([x, e])
