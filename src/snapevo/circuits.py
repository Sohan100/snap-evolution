"""State-vector simulation of layered circuits, and the local infidelity C.

A :class:`Layered` circuit is a list of sublayers ``(parity, angles)``: the cell
gate at those angles on every bond of that parity. The trained brickwork is the
case parity = l mod 2; the Trotter baselines (baseline.py) use the same type with
their own parities, so every metric works on both.

Geometry. On a ring (training, compiling) parity-1 sublayers include the wrap
bond (N-1, 0). On an open chain (the evaluation) that bond does not exist, and
the wrap gate is replaced by single-qubit edge gates on sites N-1 and 0: by
default the cell's own single-qubit factor, which makes an open-chain Trotter
circuit exact at the ends, or fitted edge rotations (edges.py).

States are dense (n, 2^N) arrays, one state per row, so this module is good to
about N = 20. mps.py runs the same circuits at N = 100.

Local infidelity. For input states |phi_k> and exact outputs |psi_k> = e^{-iHt}|phi_k>,

    C = 1 - (1/n) sum_k |<psi_k| U |phi_k>|^{2/N},

the per-site infidelity on random product states. It is what training and the
SNAP compiler minimize (docs/error_metric.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snapevo.cells import Cell
from snapevo.hamiltonian import XXZ, evolve

__all__ = [
    "Layered",
    "apply_2q",
    "product_vectors",
    "random_product_states",
    "state_pairs",
    "local_infidelity",
    "held_out",
    "TRAIN_SET",
    "HELD_OUT_SETS",
]

TRAIN_SET: tuple[int, int] = (0, 12)
"""(seed, n_states) of the training set."""

HELD_OUT_SETS: tuple[tuple[int, int], ...] = ((54321, 16), (777, 64), (2468, 64), (13579, 64))
"""(seed, n_states) of the four held-out sets, disjoint from training and from each other."""


def apply_2q(states: np.ndarray, U: np.ndarray, i: int, j: int, N: int) -> np.ndarray:
    """Apply a 4x4 gate to qubits (i, j) of a batch of states (qubit 0 is the most significant)."""
    n = states.shape[0]
    t = states.reshape((n,) + (2,) * N)
    t = np.tensordot(t, U.reshape(2, 2, 2, 2), axes=([i + 1, j + 1], [2, 3]))
    t = np.moveaxis(t, [-2, -1], [i + 1, j + 1])
    return t.reshape(n, 2 ** N)


@dataclass
class Layered:
    """A circuit of cell sublayers, each ``(parity, angles)``."""

    cell: Cell
    layers: list[tuple[int, np.ndarray]] = field(default_factory=list)

    @classmethod
    def brickwork(cls, cell: Cell, x) -> Layered:
        """The translation-invariant brickwork from a flat angle vector."""
        x = np.asarray(x, dtype=float)
        na = cell.n_angles
        if x.size % na:
            raise ValueError(f"{x.size} angles is not a multiple of {na}")
        return cls(cell, [(l % 2, x[na * l: na * l + na]) for l in range(x.size // na)])

    @property
    def depth(self) -> int:
        return len(self.layers)

    def edge_pairs(self, edges=None) -> list[tuple[np.ndarray, np.ndarray]]:
        """(gate on site N-1, gate on site 0) for each parity-1 sublayer of an open chain.

        ``edges`` is a list of such pairs of 2x2 unitaries, one per parity-1
        sublayer, in order. None means the cell's single-qubit factor, which is
        the wrap gate with its coupling removed.
        """
        odd = [angles for parity, angles in self.layers if parity == 1]
        if edges is None:
            return [self.cell.one_qubit_factor(a) for a in odd]
        if len(edges) != len(odd):
            raise ValueError(f"{len(edges)} edge pairs for {len(odd)} parity-1 sublayers")
        return [(np.asarray(g1), np.asarray(g0)) for g1, g0 in edges]

    def apply(self, states: np.ndarray, N: int, dagger: bool = False, bc: str = "ring",
              edges=None) -> np.ndarray:
        """Apply the circuit (or its inverse) to a batch of states on N sites.

        On a ring every parity-1 sublayer acts on the wrap bond (N-1, 0). On an
        open chain that gate is replaced by the edge gates (see edge_pairs).
        """
        if N % 2:
            raise ValueError("the brickwork needs an even number of sites")
        if bc not in ("ring", "open"):
            raise ValueError(f"bc must be 'ring' or 'open', not {bc!r}")
        wrap = iter([np.kron(g1, g0) for g1, g0 in self.edge_pairs(edges)]) if bc == "open" else None
        ops = []                                      # (gate, wrap gate or None) per sublayer
        for parity, angles in self.layers:
            ops.append((self.cell.gate(angles), next(wrap) if wrap and parity == 1 else None))
        s = states
        for (parity, _a), (U, W) in reversed(list(zip(self.layers, ops))) if dagger else zip(self.layers, ops):
            if dagger:
                U = U.conj().T
                W = None if W is None else W.conj().T
            for i in range(parity, N, 2):            # bonds of one parity commute
                if i == N - 1 and bc == "open":
                    s = apply_2q(s, W, N - 1, 0, N)
                else:
                    s = apply_2q(s, U, i, (i + 1) % N, N)
        return s


def product_vectors(N: int, n: int, seed: int) -> np.ndarray:
    """Single-site vectors of n Haar-random product states, shape (n, N, 2).

    Site 0 is the most significant qubit. Same draws as the Sep 2026 code, so the
    MPS runs and the state-vector runs see the same inputs.
    """
    rng = np.random.default_rng(seed)
    v = np.empty((n, N, 2), dtype=complex)
    for k in range(n):
        for i in range(N):
            w = rng.normal(size=2) + 1j * rng.normal(size=2)
            v[k, i] = w / np.linalg.norm(w)
    return v


def random_product_states(N: int, n: int, seed: int) -> np.ndarray:
    """n Haar-random product states on N qubits, as dense rows (n, 2^N)."""
    out = np.empty((n, 2 ** N), dtype=complex)
    for k, sites in enumerate(product_vectors(N, n, seed)):
        psi = np.array([1.0 + 0j])
        for v in sites:
            psi = np.kron(psi, v)
        out[k] = psi
    return out


def state_pairs(model: XXZ, N: int, t: float, n: int, seed: int,
                bc: str = "ring") -> tuple[np.ndarray, np.ndarray]:
    """(inputs, exact outputs) for n random product states."""
    inputs = random_product_states(N, n, seed)
    return inputs, evolve(model.sparse(N, bc), inputs, t)


def local_infidelity(out: np.ndarray, target: np.ndarray, N: int) -> float:
    """C = 1 - mean_k |<target_k|out_k>|^{2/N}."""
    ov = np.sum(np.conj(target) * out, axis=1)
    return 1.0 - float(np.mean((np.abs(ov) ** 2) ** (1.0 / N)))


def held_out(circuit: Layered, model: XXZ, t: float, sizes=(10, 12)) -> dict:
    """C of a circuit on the ring on the four held-out sets at each size, and the worst value."""
    out = {}
    for N in sizes:
        vals = []
        for seed, n in HELD_OUT_SETS:
            tin, tg = state_pairs(model, N, t, n, seed)
            vals.append(local_infidelity(circuit.apply(tin, N), tg, N))
        out[str(N)] = vals
    out["worst"] = max(max(v) for v in out.values())
    return out
