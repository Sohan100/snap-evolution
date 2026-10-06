"""Two-qubit cells of the translation-invariant brickwork.

A depth-D circuit applies one cell gate per sublayer to every bond of one
parity, even bonds (0,1), (2,3), ... in sublayers 0, 2, 4, ... and odd bonds
(1,2), ..., (N-1,0) in sublayers 1, 3, .... Every sublayer has its own angles,
and the same angles on every bond of that sublayer, so the circuit is
invariant under translation by two sites and works at any even N.

Two cells ship, one per model (docs/conventions.md):

``xxz+stag`` (model B, h_x = 0)
    exp[-i (a XX + b YY + c ZZ)/2] (Rz(d) x Rz(-d)). The staggered field is
    what makes the zero-field cell trainable at depth (Sep 2026 runs).

``xxz+x`` (model A, h_x != 0)
    exp[-i (a XX + b YY + c ZZ)/2] (Rx(e) x Rx(f)). The single-qubit
    rotations act first. At a = b = J tau/2, c = J Delta tau/2 and
    e = f = h_x tau/2 two sublayers are one first-order Trotter step of
    length tau.

Each angle enters its gate as cos(theta/2), sin(theta/2) (harmonic 1) or as
e^{+-i theta} (the staggered angle, harmonic 2). environment.py relies on that.
Every slot costs Rz rotations plus Cliffords: Rzz(a) = CNOT (1 x Rz(a)) CNOT,
Rx(a) = H Rz(a) H, and XX, YY are Rzz conjugated by single-qubit Cliffords.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snapevo.hamiltonian import XXZ

__all__ = ["Cell", "CELLS", "get_cell", "cell_for"]

_I2 = np.eye(2, dtype=complex)
_I4 = np.eye(4, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_XX, _YY, _ZZ = np.kron(_X, _X), np.kron(_Y, _Y), np.kron(_Z, _Z)
_STAG = np.array([0.0, 1.0, -1.0, 0.0])        # (Z x 1 - 1 x Z)/2 on |00>,|01>,|10>,|11>


def _rot(a: float, P: np.ndarray, Id: np.ndarray) -> np.ndarray:
    return np.cos(a / 2) * Id - 1j * np.sin(a / 2) * P


@dataclass(frozen=True)
class Cell:
    """One brickwork cell.

    Attributes:
        name: ``"xxz+stag"`` or ``"xxz+x"``.
        slots: Angle names in flat-vector order within a sublayer.
        multiplicity: Rz rotations each slot costs per bond.
        harmonic: Highest harmonic of theta/2 one bond contributes per slot.
    """

    name: str
    slots: tuple[str, ...]
    multiplicity: tuple[int, ...]
    harmonic: tuple[int, ...]

    @property
    def n_angles(self) -> int:
        return len(self.slots)

    @property
    def rotations_per_sublayer_per_qubit(self) -> float:
        """Rz rotations per qubit per sublayer (each angle runs on N/2 bonds)."""
        return 0.5 * sum(self.multiplicity)

    def gate(self, angles) -> np.ndarray:
        a, b, c, *rest = (float(v) for v in angles)
        U = _rot(a, _XX, _I4) @ _rot(b, _YY, _I4) @ _rot(c, _ZZ, _I4)
        if self.name == "xxz+stag":
            return U @ np.diag(np.exp(-1j * rest[0] * _STAG))
        return U @ np.kron(_rot(rest[0], _X, _I2), _rot(rest[1], _X, _I2))

    def jax_gate(self, angles):
        """The same gate as a traced jax array (angles: a length-n_angles jax vector)."""
        import jax.numpy as jnp

        I2 = jnp.eye(2, dtype=jnp.complex128)
        I4 = jnp.eye(4, dtype=jnp.complex128)

        def rot(t, P, Id):
            return jnp.cos(t / 2) * Id - 1j * jnp.sin(t / 2) * jnp.asarray(P)

        U = rot(angles[0], _XX, I4) @ rot(angles[1], _YY, I4) @ rot(angles[2], _ZZ, I4)
        if self.name == "xxz+stag":
            d = angles[3]
            return U @ jnp.kron(rot(d, _Z, I2), rot(-d, _Z, I2))
        return U @ jnp.kron(rot(angles[3], _X, I2), rot(angles[4], _X, I2))

    def trotter_sublayer(self, model: XXZ, tau: float) -> np.ndarray:
        """Angles of one sublayer of the first-order brickwork Trotter step.

        Bonds of the sublayer's parity evolve for ``tau`` and every site gets
        half the field step, ``tau/2``, so two sublayers advance time by ``tau``.
        """
        cxx, cyy, czz = model.bond_pauli
        bond = [2 * cxx * tau, 2 * cyy * tau, 2 * czz * tau]
        if self.name == "xxz+stag":
            if model.hx:
                raise ValueError("the xxz+stag cell has no x field; use xxz+x")
            return np.array(bond + [0.0])
        e = 2 * model.field_pauli * tau / 2
        return np.array(bond + [e, e])

    def trotter_angles(self, model: XXZ, t: float, D: int) -> np.ndarray:
        """Flat angles of the depth-D first-order brickwork Trotter circuit for time t."""
        return np.tile(self.trotter_sublayer(model, 2.0 * t / D), D)


CELLS: dict[str, Cell] = {
    "xxz+stag": Cell("xxz+stag", ("xx", "yy", "zz", "stag"), (1, 1, 1, 2), (1, 1, 1, 2)),
    "xxz+x": Cell("xxz+x", ("xx", "yy", "zz", "x1", "x2"), (1, 1, 1, 1, 1), (1, 1, 1, 1, 1)),
}


def get_cell(name: str) -> Cell:
    try:
        return CELLS[name]
    except KeyError:
        raise KeyError(f"unknown cell {name!r}; choose from {sorted(CELLS)}") from None


def cell_for(model: XXZ) -> Cell:
    """The default cell of a model: xxz+stag without a field, xxz+x with one."""
    return CELLS["xxz+x" if model.hx else "xxz+stag"]
