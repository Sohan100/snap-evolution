"""The target Hamiltonians, in spin units, and exact time evolution.

Both models are the spin-1/2 XXZ ring with an optional uniform field along x,

    H = J sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + Delta S^z_i S^z_{i+1}) + h_x sum_i S^x_i,

with S = sigma / 2 and periodic boundary conditions. In Pauli operators this is

    H = (J/4) sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Delta Z_i Z_{i+1}) + (h_x/2) sum_i X_i,

so at h_x = 0 the spin-unit time t is four times the Pauli-unit time of the
Sep 2026 runs (H = sum XX + YY + Delta ZZ). See docs/conventions.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

__all__ = ["XXZ", "MODELS", "get_model", "evolve"]

_X = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
_Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
_Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
_I = sp.identity(2, format="csr", dtype=complex)


@dataclass(frozen=True)
class XXZ:
    """Spin-1/2 XXZ ring with a uniform x field, in spin units.

    Attributes:
        delta: Anisotropy of the ZZ term.
        hx: Field along x, coupling to S^x = X/2.
        J: Exchange coupling.
    """

    delta: float = 0.25
    hx: float = 0.0
    J: float = 1.0

    @property
    def tag(self) -> str:
        """Filesystem label, e.g. ``xxz_d0.25_hx1.6``."""
        tag = f"xxz_d{self.delta:g}_hx{self.hx:g}"
        return tag if self.J == 1.0 else f"{tag}_J{self.J:g}"

    @property
    def bond_pauli(self) -> tuple[float, float, float]:
        """Coefficients of XX, YY, ZZ on one bond: (J/4, J/4, J Delta/4)."""
        return (self.J / 4, self.J / 4, self.J * self.delta / 4)

    @property
    def field_pauli(self) -> float:
        """Coefficient of X on one site: h_x / 2."""
        return self.hx / 2

    def describe(self) -> str:
        return f"XXZ ring, Delta = {self.delta:g}, h_x = {self.hx:g}, J = {self.J:g} (spin units)"

    def to_dict(self) -> dict:
        return {"delta": self.delta, "hx": self.hx, "J": self.J}

    @classmethod
    def from_dict(cls, d: dict) -> XXZ:
        return cls(float(d["delta"]), float(d["hx"]), float(d.get("J", 1.0)))

    def sparse(self, N: int) -> sp.csr_matrix:
        """The Hamiltonian on an N-site ring as a sparse (2^N, 2^N) matrix."""
        def embed(ops: dict[int, sp.csr_matrix]) -> sp.csr_matrix:
            out = None
            for site in range(N):
                m = ops.get(site, _I)
                out = m if out is None else sp.kron(out, m, format="csr")
            return out

        cxx, cyy, czz = self.bond_pauli
        H = sp.csr_matrix((2 ** N, 2 ** N), dtype=complex)
        for i in range(N):
            j = (i + 1) % N
            H = H + cxx * embed({i: _X, j: _X}) + cyy * embed({i: _Y, j: _Y})
            if czz:
                H = H + czz * embed({i: _Z, j: _Z})
        if self.hx:
            for i in range(N):
                H = H + self.field_pauli * embed({i: _X})
        return H.tocsr()


MODELS: dict[str, XXZ] = {
    "A": XXZ(delta=0.25, hx=1.6),
    "B": XXZ(delta=0.25, hx=0.0),
}
"""The two production models (meeting of 2026-10-05). A has the field, B does not."""


def get_model(name: str) -> XXZ:
    """Look up a model by its key in :data:`MODELS`."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None


def evolve(H: sp.spmatrix, states: np.ndarray, t: float) -> np.ndarray:
    """Apply e^{-iHt} to a batch of states stored as rows, shape (n, 2^N)."""
    return expm_multiply(-1j * t * H, states.T).T
