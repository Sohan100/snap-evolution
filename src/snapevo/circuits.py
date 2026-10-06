"""State-vector simulation of layered circuits, and the local infidelity C.

A :class:`Layered` circuit is a list of sublayers ``(parity, angles)``: the cell
gate at those angles on every bond of that parity. The trained brickwork is the
case parity = l mod 2; the Trotter baselines (baseline.py) use the same type with
their own parities, so every metric works on both.

States are dense (n, 2^N) arrays, one state per row, so this module is good to
about N = 20.

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
    "random_product_states",
    "state_pairs",
    "local_infidelity",
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

    def apply(self, states: np.ndarray, N: int, dagger: bool = False) -> np.ndarray:
        """Apply the circuit (or its inverse) to a batch of states on an N-site ring."""
        if N % 2:
            raise ValueError("the brickwork needs an even number of sites")
        seq = list(reversed(self.layers)) if dagger else self.layers
        s = states
        for parity, angles in seq:
            U = self.cell.gate(angles)
            if dagger:
                U = U.conj().T
            for i in range(parity, N, 2):            # bonds of one parity commute
                s = apply_2q(s, U, i, (i + 1) % N, N)
        return s


def random_product_states(N: int, n: int, seed: int) -> np.ndarray:
    """n Haar-random product states on N qubits, as rows. Same draws as the Sep 2026 code."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2 ** N), dtype=complex)
    for k in range(n):
        psi = np.array([1.0 + 0j])
        for _ in range(N):
            v = rng.normal(size=2) + 1j * rng.normal(size=2)
            psi = np.kron(psi, v / np.linalg.norm(v))
        out[k] = psi
    return out


def state_pairs(model: XXZ, N: int, t: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(inputs, exact outputs) for n random product states."""
    inputs = random_product_states(N, n, seed)
    return inputs, evolve(model.sparse(N), inputs, t)


def local_infidelity(out: np.ndarray, target: np.ndarray, N: int) -> float:
    """C = 1 - mean_k |<target_k|out_k>|^{2/N}."""
    ov = np.sum(np.conj(target) * out, axis=1)
    return 1.0 - float(np.mean((np.abs(ov) ** 2) ** (1.0 / N)))
