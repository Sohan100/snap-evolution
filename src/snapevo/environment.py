"""Exact C as a function of one angle, from cached environments.

A sublayer applies the same gate to N/2 bonds, and each angle enters the gate
through cos(theta/2), sin(theta/2) (or e^{+-i theta} for the staggered slot). So
the overlap <psi_k| U |phi_k> is a trigonometric polynomial in theta/2 of degree
h N/2, with h the slot's harmonic, and h N + 1 probes over the 4 pi period fix it
exactly. After a DFT, C(theta) = 1 - mean_k |O_k(theta)|^{2/N} is a closed form
that can be evaluated at every library angle at once. This is the plug-in
infidelity the SNAP compiler screens candidates with.

Caches: the state before each sublayer (forward) and the target pulled back past
each sublayer (backward). Each probe re-applies one sublayer, so fitting all
angles of a depth-D circuit costs about 2 + n_angles (h N + 1) / D circuits.
Ported unchanged in substance from tcount_tfim.environment (Sep 2026).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snapevo.cells import Cell
from snapevo.circuits import apply_2q

__all__ = ["AngleCurve", "Environments"]


@dataclass
class AngleCurve:
    """C of the circuit as a function of one flat angle, all others fixed."""

    index: int
    coeffs: np.ndarray        # (n_modes, n_states)
    modes: np.ndarray         # harmonics of theta/2, fftfreq order
    N: int
    theta0: float
    residual: float

    CHUNK = 1 << 17

    def overlap_at(self, theta) -> np.ndarray:
        scalar = np.ndim(theta) == 0
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        out = np.empty((th.size, self.coeffs.shape[1]), dtype=complex)
        for lo in range(0, th.size, self.CHUNK):
            block = th[lo: lo + self.CHUNK]
            out[lo: lo + self.CHUNK] = np.exp(0.5j * block[:, None] * self.modes[None, :]) @ self.coeffs
        return out[0] if scalar else out

    def c_at(self, theta):
        """Exact training C with this angle set to theta (scalar or array)."""
        f = np.clip(np.abs(self.overlap_at(theta)) ** 2, 0.0, None)
        return 1.0 - np.mean(f ** (1.0 / self.N), axis=-1)

    def curvature(self) -> float:
        """d^2 C / d theta^2 at the current value, floored at 1e-12."""
        d = 1e-4
        c = self.c_at(np.array([self.theta0 - d, self.theta0, self.theta0 + d]))
        return max(float((c[0] + c[2] - 2 * c[1]) / d ** 2), 1e-12)


class Environments:
    """Forward and backward caches of the brickwork at one angle vector."""

    def __init__(self, x: np.ndarray, cell: Cell, inputs: np.ndarray, targets: np.ndarray, N: int):
        self.x = np.asarray(x, dtype=float).copy()
        self.cell = cell
        self.N = N
        na = cell.n_angles
        self.D = self.x.size // na
        self._bonds = [[(i, (i + 1) % N) for i in range(l % 2, N, 2)] for l in range(self.D)]
        self._n_probe = N * max(cell.harmonic) + 1
        gates = [cell.gate(self.x[na * l: na * l + na]) for l in range(self.D)]
        pre, s = [None] * self.D, inputs
        for l in range(self.D):
            pre[l] = s
            s = self._apply(s, gates[l], l)
        self._pre, self._out = pre, s
        post, b = [None] * self.D, targets
        for l in range(self.D - 1, -1, -1):
            post[l] = b
            if l > 0:
                b = self._apply(b, gates[l].conj().T, l)
        self._post = post

    def _apply(self, s: np.ndarray, U: np.ndarray, layer: int) -> np.ndarray:
        for i, j in self._bonds[layer]:
            s = apply_2q(s, U, i, j, self.N)
        return s

    def c_now(self) -> float:
        ov = np.sum(np.conj(self._post[self.D - 1]) * self._out, axis=1)
        return 1.0 - float(np.mean((np.abs(ov) ** 2) ** (1.0 / self.N)))

    def curve(self, index: int) -> AngleCurve:
        na = self.cell.n_angles
        if not 0 <= index < na * self.D:
            raise IndexError(f"angle index {index} outside 0..{na * self.D - 1}")
        layer, slot = divmod(index, na)
        M = self._n_probe
        probes = 4.0 * np.pi * np.arange(M) / M
        samples = np.empty((M, self._pre[layer].shape[0]), dtype=complex)
        base = self.x[na * layer: na * layer + na].copy()
        for q, theta in enumerate(probes):
            ang = base.copy()
            ang[slot] = theta
            s = self._apply(self._pre[layer], self.cell.gate(ang), layer)
            samples[q] = np.sum(np.conj(self._post[layer]) * s, axis=1)
        coeffs = np.fft.fft(samples, axis=0) / M
        modes = np.fft.fftfreq(M, d=1.0 / M)
        curve = AngleCurve(index, coeffs, modes, self.N, float(self.x[index]), 0.0)
        scale = max(float(np.max(np.abs(samples))), 1e-300)
        curve.residual = float(np.max(np.abs(curve.overlap_at(probes) - samples))) / scale
        return curve

    def curves(self, indices) -> dict[int, AngleCurve]:
        return {int(i): self.curve(int(i)) for i in indices}
