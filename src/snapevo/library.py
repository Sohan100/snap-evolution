"""The exact-angle library: Isaac's corrected RUS gadgets (2026-10-05), no catalysis.

Source: third_party/rz_library_corrected_2026-10-05/rz_rus_gadgets.csv, two rows
per gadget (cheapest same-angle and cheapest odd-shift dressing), t_exact = T
count of one attempt. docs/cost_model.md explains the pricing and what was
verified about the file.

Pricing at fallback cost F (the expected T of the KN synthesis that corrects a
"2+T" failure), cheaper of the two strategies:

    repeat:      t/p + r (1 - p)/p      r = 0 (Clifford), 1 (T), F (2+T)
    single-shot: t + (1 - p) F

Each priced gadget also carries q, the probability per use that an approximate
KN correction runs: (1 - p)/p for repeat with a 2+T correction, 1 - p for
single-shot, 0 otherwise. Each such event has diamond error kn_eps(F).

Dressing: +-angle + k pi/4 for k = 0..7, odd k costs one more T, plus the eight
k pi/4 anchors at 0 or 1 T. At F = 18 the lattice equals the shipped
rz_library_cat0_kn18.csv (tests/test_library.py).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np

from snapevo import io
from snapevo.synthesis import kn_eps

__all__ = ["LIB_DIR", "Gadget", "load_gadgets", "gadget_cost", "Library", "QUARTER", "TWO_PI"]

LIB_DIR = "rz_library_corrected_2026-10-05"
TWO_PI = 2.0 * math.pi
QUARTER = math.pi / 4.0


@dataclass(frozen=True)
class Gadget:
    angle: float
    t: float              # T count of one attempt
    p: float
    correction: str       # "Clifford", "T" or "2+T"
    source: str


def load_gadgets() -> list[Gadget]:
    path = io.third_party(LIB_DIR) / "rz_rus_gadgets.csv"
    with path.open(newline="") as fh:
        return [Gadget(float(r["angle_rad"]), float(r["t_exact"]), float(r["p"]),
                       r["correction"], r["source"]) for r in csv.DictReader(fh)]


def gadget_cost(t: float, p: float, correction: str, F: float) -> tuple[float, float]:
    """(expected T, q) of one use; q = probability that the approximate fallback runs."""
    reset = {"Clifford": 0.0, "T": 1.0, "2+T": F}[correction]
    repeat = t / p + reset * (1 - p) / p
    single = t + (1 - p) * F
    if repeat <= single:
        return repeat, ((1 - p) / p if correction == "2+T" else 0.0)
    return single, 1 - p


@dataclass
class Library:
    """Dressed lattice at fallback cost F: sorted angles S in [0, 2 pi], costs C, fallback rates Q."""

    F: float = 22.0
    S: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)
    Q: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        lat: dict[float, tuple[float, float]] = {}

        def add(a: float, c: float, q: float) -> None:
            a = round(a % TWO_PI, 12)
            cur = lat.get(a)
            if cur is None or c < cur[0] or (c == cur[0] and q < cur[1]):
                lat[a] = (c, q)

        for g in load_gadgets():
            c, q = gadget_cost(g.t, g.p, g.correction, self.F)
            for s in (1, -1):
                for k in range(8):
                    add(s * g.angle + k * QUARTER, c + (0.0 if k % 2 == 0 else 1.0), q)
        for k in range(8):
            add(k * QUARTER, 0.0 if k % 2 == 0 else 1.0, 0.0)
        self.S = np.array(sorted(lat))
        cq = np.array([lat[a] for a in self.S])
        self.C, self.Q = cq[:, 0], cq[:, 1]

    @property
    def delta_fb(self) -> float:
        """Diamond error of one fallback correction at this F (calibrated KN law)."""
        return kn_eps(self.F)

    def __len__(self) -> int:
        return len(self.S)

    def lookup(self, theta: float, tol: float = 1e-9) -> tuple[float, float] | None:
        """(cost, q) of the cheapest lattice point within tol of theta, or None."""
        x = theta % TWO_PI
        lo, hi = np.searchsorted(self.S, x - tol), np.searchsorted(self.S, x + tol)
        idx = list(range(lo, hi))
        if x - tol < 0:
            idx += list(range(np.searchsorted(self.S, x - tol + TWO_PI), len(self.S)))
        if x + tol > TWO_PI:
            idx += list(range(0, np.searchsorted(self.S, x + tol - TWO_PI)))
        if not idx:
            return None
        j = min(idx, key=lambda k: (self.C[k], self.Q[k]))
        return float(self.C[j]), float(self.Q[j])

    @staticmethod
    def kind(theta: float, cost: float) -> str:
        """'clifford', 't' or 'rus' for a lattice point."""
        k = theta / QUARTER
        if abs(k - round(k)) < 1e-9:
            return "clifford" if round(k) % 2 == 0 else "t"
        return "rus"
