"""Literature product formulas on the open chain, priced and measured like baseline.py.

baseline.py fixes one family: Suzuki orders 1, 2, 4 over three commuting groups
(even bonds E, odd bonds O, field X) nested as X(h/2) E(h/2) O(h) E(h/2) X(h/2).
This module builds the formulas a referee could call "best vanilla Trotter",
without changing baseline.py:

* Suzuki order 6 by the same recursion, mu_6 = 1/(4 - 4^{1/5}) (Liu et al. Eq. 10).
* The bond orderings of Liu, Wei, Foss-Feig, Gorshkov, Childs, arXiv:2610.03672,
  named by their consecutive-label pattern c (Def. 1, Eq. 3; c_k = 1 when bond k
  comes before bond k+1): brickwall BW10 (c = 1010..., the paper's odd-even, which
  with 0-indexed bonds is baseline.py's E before O) and BW01 (0101...), sequential
  SEQ1 (11...1, forward staircase) and SEQ0 (00...0), and the depth-d wave
  WAVE<d> (0^{d-1}1^{d-1}..., Sec. V.4) and WAVE<d>R (1^{d-1}0^{d-1}...). Each
  first-order stage lists the bonds by Lemma 8's layers, S^(2) is the permuted
  formula of Eq. 11 and orders 4 and 6 follow the recursion of Eq. 12.
* At h_x != 0, either the field as its own group X with any nesting of the
  groups (Lane & Browne, arXiv:2505.04240, ref. [18] of Liu et al.), or the
  field folded into the bond terms (``fold``): with Table 1's convention, the
  field of each site goes to the bond on its right, and the last bond also
  carries the field of the last site, so the terms add up to the model's H.
  (Table 1's models, H = sum_{j=1}^{N-1} h_j, simply have no field on site N;
  ours has one, so it rides on the last bond. Only that one term differs from
  the bulk term, so bulk densities are unaffected.)

Nesting grammar. A sweep such as ``XEO`` is the permuted S^(2) of Eq. 11 over
its items, X(tau/2) E(tau/2) O(tau/2) O(tau/2) E(tau/2) X(tau/2), which after
merging is baseline.py's formula. ``B`` is the bond sweep in the ordering's
Lemma 8 order (for BW10 it is E then O, so ``XB`` = ``XEO``). ``[L,R]`` is the
nested split S2[L](tau/2) S2[R](tau) S2[L](tau/2), e.g. ``[XE,O]`` =
X/4 E/2 X/4 O X/4 E/2 X/4. Order 1 is the first half of a flat sweep at the
full step.

Circuits. A formula becomes a list of :class:`Gate` in time order, after
merging every gate into the previous gate of the same term when everything in
between commutes with it (disjoint supports). That is how brickwall layers
merge (Sec. VII of Liu et al.); ``stages`` counts the paper's way, merging only
literally adjacent equal generators (401 for the n = 9 sixth-order sequential
formula, Sec. VI.1). Bond gates exp(-i d (cxx XX + cyy YY + czz ZZ)) cost three
Pauli rotations (the cell's xx, yy, zz slots), field gates Rx(h_x d) one. A
folded term is a generic two-qubit unitary: it goes through the KAK
decomposition, the single-qubit factors between two-qubit gates on a site are
multiplied and written as Rz Rx Rz, so it costs three rotations for the
coupling and three per merged single-qubit gate (about nine per bond gate).

Pricing is baseline.py's: one KN non-mixed synthesis per distinct angle at
budget eps (baseline.py keys angles by numpy's round(theta, 15); angles that come out
of a numerical decomposition by round(theta, 11)), multiples of pi/4 cost 0
or 1 T, every copy of an angle uses the same circuit. Angles from the
decomposition within diamond distance eps of a multiple of pi/4 are set to it,
as edges.py does for its fitted angles. Folded circuits also have Euler angles
of 1e-7 to 1e-4, on which KN's search stalls; one that is farther than eps but
within SMALL_OFFSET of a multiple of pi/4 is written as two generic KN rotations
(Rz(off + A) Rz(-A), A = SPLIT_ANGLE), which costs one extra rotation. Per-qubit numbers are
bulk densities, the count per added site of the translation-invariant
interior, measured by building the circuit at two lengths that differ by a
whole number of pattern periods. For a brickwall formula that is exactly
baseline.py's 1/2 per slot per sublayer. ``Tq_chain`` is the total on the
measured chain divided by its length, boundary gates included.

Errors are measured: the realized circuit (every rotation at its success
branch angle) runs on product inputs, exactly by state vector up to N ~ 14
(apply_dense) or by MPS at N = 100 (apply_mps, mps_infidelity) with the
reference states of evaluate.py, read from runs/<tag>/t<t>/ref/ and never
written there.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from snapevo.baseline import EPS_GRID as _BASE_EPS_GRID
from snapevo.baseline import bound_per_site
from snapevo.cells import CELLS
from snapevo.circuits import HELD_OUT_SETS, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ
from snapevo.library import QUARTER

__all__ = ["ORDERS", "ORDERINGS", "EPS_GRID", "SCAN_VERSION", "BW_NESTINGS", "c_pattern", "lemma8_layers",
           "sweep_order", "pattern_period", "stage_coeffs", "parse_nesting", "Term", "Chain", "Gate",
           "product_sequence", "merge_sequence", "count_stages", "Formula", "kak", "zxz_angles", "compile_gates",
           "fuse", "apply_dense", "unitary", "apply_mps", "SNAP_TOL", "SMALL_OFFSET", "SPLIT_ANGLE", "bulk_tally",
           "KNCache", "realize", "plan_cost", "mps_infidelity", "scan_family", "refine_family", "default_families",
           "scan", "pareto"]

ORDERS = (1, 2, 4, 6)
"""Suzuki orders built here (baseline.py has 1, 2, 4)."""

EPS_GRID = tuple(_BASE_EPS_GRID) + (3e-8, 1e-8)
"""KN budgets: baseline.py's grid and two tighter ones, for targets near 1e-8 per site."""

SCAN_VERSION = 1
"""Bumped when the scan's choice of points changes."""

ORDERINGS = {"BW10": "10", "BW01": "01", "SEQ1": "1", "SEQ0": "0"}
"""Named periods of the consecutive-label pattern c. WAVE<d> and WAVE<d>R are built on demand."""

BW_NESTINGS = ("XEO", "XOE", "EXO", "EOX", "OXE", "OEX",
               "[XE,O]", "[EX,O]", "[XO,E]", "[OX,E]", "[EO,X]", "[OE,X]")
"""The twelve second-order splittings of three groups. Lane & Browne (arXiv:2505.04240, Sec. 2.1,
Eqs. 6-9) count m! orderings times 2^(m-1) shallow/wide steps, 24 for m = 3; the second step only
swaps the last two groups, so there are 12 distinct formulas: six flat sweeps (shallow first step,
their Eq. 8) and six ``[ab,c]`` (wide first step, their Eq. 9)."""

_CELL = CELLS["xxz+x"]
_I2 = np.eye(2, dtype=complex)
_PX = np.array([[0, 1], [1, 0]], dtype=complex)
_PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------- orderings (Liu et al.)

def _period(ordering: str) -> str:
    name = ordering.upper()
    if name in ORDERINGS:
        return ORDERINGS[name]
    if name.startswith("C:"):
        bits = ordering[2:]
        if not bits or set(bits) - {"0", "1"}:
            raise ValueError(f"bad pattern {ordering!r}")
        return bits
    if name.startswith("WAVE"):
        rest = name[4:]
        rev = rest.endswith("R")
        d = int(rest[:-1] if rev else rest)
        if d < 2:
            raise ValueError("a wave needs depth d >= 2")
        bits = "0" * (d - 1) + "1" * (d - 1)
        return bits[::-1] if rev else bits
    raise ValueError(f"unknown ordering {ordering!r}; use {sorted(ORDERINGS)}, WAVE<d>, WAVE<d>R or C:<bits>")


def c_pattern(ordering: str, n_bonds: int) -> list[int]:
    """The consecutive-label pattern c (length n_bonds - 1) of an ordering, its period repeated.

    Bonds are 0-indexed, bond k on sites (k, k+1); c[k] = 1 when bond k comes
    before bond k + 1 (the paper's c_{k+1}, its terms being 1-indexed).
    """
    per = _period(ordering)
    return [int(per[k % len(per)]) for k in range(max(n_bonds - 1, 0))]


def lemma8_layers(c: list[int]) -> list[int]:
    """Layer of each bond: 1 + the longest directed path ending at it (Lemma 8)."""
    n = len(c) + 1
    left, right = [0] * n, [0] * n
    for k in range(1, n):
        left[k] = left[k - 1] + 1 if c[k - 1] == 1 else 0
    for k in range(n - 2, -1, -1):
        right[k] = right[k + 1] + 1 if c[k] == 0 else 0
    return [1 + max(a, b) for a, b in zip(left, right)]


def sweep_order(ordering: str, n_bonds: int) -> list[int]:
    """Bonds of one first-order stage in time order: by Lemma 8 layer, then by position."""
    lay = lemma8_layers(c_pattern(ordering, n_bonds))
    return sorted(range(n_bonds), key=lambda k: (lay[k], k))


def pattern_period(ordering: str) -> int:
    return len(_period(ordering))


def stage_coeffs(order: int) -> list[float]:
    """Step fractions of the S^(2) stages of the Suzuki formula of that order (Eq. 10)."""
    if order == 2:
        return [1.0]
    if order < 2 or order % 2:
        raise ValueError("Suzuki orders are 2, 4, 6, ... (order 1 is a single sweep)")
    mu = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    inner = stage_coeffs(order - 2)
    return [w * x for w in (mu, mu, 1.0 - 4.0 * mu, mu, mu) for x in inner]


def parse_nesting(spec: str):
    """Parse a nesting: ('sweep', items) or ('pair', left, right). Items are X, E, O, B."""
    s = spec.replace(" ", "")
    pos = 0

    def node():
        nonlocal pos
        if pos < len(s) and s[pos] == "[":
            pos += 1
            left = node()
            if pos >= len(s) or s[pos] != ",":
                raise ValueError(f"expected ',' in nesting {spec!r}")
            pos += 1
            right = node()
            if pos >= len(s) or s[pos] != "]":
                raise ValueError(f"expected ']' in nesting {spec!r}")
            pos += 1
            return ("pair", left, right)
        items = []
        while pos < len(s) and s[pos] in "XEOB":
            items.append(s[pos])
            pos += 1
        if not items:
            raise ValueError(f"empty sweep in nesting {spec!r}")
        return ("sweep", tuple(items))

    tree = node()
    if pos != len(s):
        raise ValueError(f"trailing characters in nesting {spec!r}")
    return tree


def _items(tree) -> set[str]:
    if tree[0] == "sweep":
        return set(tree[1])
    return _items(tree[1]) | _items(tree[2])


# ---------------------------------------------------------------------------- terms and chains

@dataclass(frozen=True, eq=False)
class Term:
    """One local generator. kind "xxz": coef = (cxx, cyy, czz) on a bond; "x": coef = (c,) of X on a
    site; "h": a generic Hermitian ``mat`` on its sites."""

    sites: tuple
    kind: str
    coef: tuple = ()
    mat: np.ndarray | None = None

    def generator(self) -> np.ndarray:
        if self.kind == "xxz":
            cxx, cyy, czz = self.coef
            return cxx * np.kron(_PX, _PX) + cyy * np.kron(_PY, _PY) + czz * np.kron(_PZ, _PZ)
        if self.kind == "x":
            return self.coef[0] * _PX
        return self.mat

    def gate(self, d: float) -> Gate:
        """exp(-i d h) as a gate. The angle expressions are baseline.py's, so angles agree bitwise."""
        if self.kind == "xxz":
            cxx, cyy, czz = self.coef
            return Gate("xxz", self.sites, (2 * cxx * d, 2 * cyy * d, 2 * czz * d))
        if self.kind == "x":
            return Gate("rx", self.sites, (2 * self.coef[0] * d,))
        return Gate("u2" if len(self.sites) == 2 else "u1", self.sites, mat=_expm_cached(self, d))


_EXPM: dict = {}


def _expm_cached(term: Term, d: float) -> np.ndarray:
    h = term.generator()
    key = (h.shape[0], h.tobytes(), d)
    U = _EXPM.get(key)
    if U is None:
        if len(_EXPM) > 200000:
            _EXPM.clear()
        U = expm(-1j * d * h)
        _EXPM[key] = U
    return U


class Chain:
    """Local terms of an open chain of N sites: bonds ("b", k) on (k, k+1) and fields ("x", i)."""

    def __init__(self, N: int, bonds: list[Term], fields: list[Term]):
        self.N = N
        self.terms: dict = {("b", k): t for k, t in enumerate(bonds)}
        self.terms.update({("x", i): t for i, t in enumerate(fields)})
        self.n_bonds = len(bonds)
        self.has_field = bool(fields)
        self.support = {lab: t.sites for lab, t in self.terms.items()}

    @classmethod
    def xxz(cls, model: XXZ, N: int, fold: bool = False) -> Chain:
        """The model on an open chain: bonds and a separate field, or the field folded into the bonds.

        Folded (Liu et al. Table 1, fields on the left site of each bond): bond k
        carries (h_x/2) X_k, and the last bond also (h_x/2) X_{N-1}.
        """
        cxx, cyy, czz = model.bond_pauli
        hf = model.field_pauli
        if fold and model.hx:
            bonds = []
            for k in range(N - 1):
                h = Term((k, k + 1), "xxz", (cxx, cyy, czz)).generator() + hf * np.kron(_PX, _I2)
                if k == N - 2:
                    h = h + hf * np.kron(_I2, _PX)
                bonds.append(Term((k, k + 1), "h", mat=h))
            return cls(N, bonds, [])
        bonds = [Term((k, k + 1), "xxz", (cxx, cyy, czz)) for k in range(N - 1)]
        fields = [Term((i,), "x", (hf,)) for i in range(N)] if model.hx else []
        return cls(N, bonds, fields)

    @classmethod
    def generic(cls, h2: np.ndarray, N: int) -> Chain:
        """A translation-invariant chain with the same two-qubit term h2 on every bond (Def. 4)."""
        h2 = np.asarray(h2, dtype=complex)
        return cls(N, [Term((k, k + 1), "h", mat=h2) for k in range(N - 1)], [])

    def groups(self, ordering: str) -> dict[str, list]:
        b = self.n_bonds
        return {"E": [("b", k) for k in range(0, b, 2)], "O": [("b", k) for k in range(1, b, 2)],
                "X": [("x", i) for i in range(self.N)] if self.has_field else [],
                "B": [("b", k) for k in sweep_order(ordering, b)]}

    def hamiltonian(self) -> np.ndarray:
        """Dense H (small N only)."""
        N = self.N
        H = np.zeros((2 ** N, 2 ** N), dtype=complex)
        for t in self.terms.values():
            i = t.sites[0]
            H += np.kron(np.kron(np.eye(2 ** i), t.generator()), np.eye(2 ** (N - i - len(t.sites))))
        return H


# ---------------------------------------------------------------------------- formulas

def _emit(tree, tau: float, groups: dict, out: list, order1: bool = False) -> None:
    if tree[0] == "sweep":
        items = tree[1]
        if order1:
            for it in items:
                out.extend((lab, tau) for lab in groups[it])
            return
        h = tau / 2
        for it in items:
            out.extend((lab, h) for lab in groups[it])
        for it in reversed(items):
            out.extend((lab, h) for lab in reversed(groups[it]))
        return
    if order1:
        raise ValueError("order 1 needs a flat sweep, not a nested splitting")
    _emit(tree[1], tau / 2, groups, out)
    _emit(tree[2], tau, groups, out)
    _emit(tree[1], tau / 2, groups, out)


def product_sequence(chain: Chain, nesting: str, ordering: str, order: int, t: float, r: int) -> list:
    """The unmerged (term, duration) list of r steps of the order-``order`` formula, in time order."""
    tree = parse_nesting(nesting)
    groups = chain.groups(ordering)
    if "X" in _items(tree) and not chain.has_field:
        raise ValueError(f"nesting {nesting!r} has a field group but the chain has no separate field")
    if chain.has_field and "X" not in _items(tree):
        raise ValueError(f"nesting {nesting!r} leaves out the field")
    if ("B" in _items(tree)) and (_items(tree) & {"E", "O"}):
        raise ValueError("use either B or E and O in a nesting, not both")
    dt = t / r
    raw: list = []
    for _ in range(r):
        if order == 1:
            _emit(tree, dt, groups, raw, order1=True)
        else:
            for w in stage_coeffs(order):
                _emit(tree, w * dt, groups, raw)
    return raw


def merge_sequence(raw: list, support: dict) -> list:
    """Merge each gate into the previous gate of the same term when everything in between commutes.

    Commuting means disjoint supports. Durations add in time order, as in baseline.py.
    """
    out: list = []
    last: dict = {}
    for lab, d in raw:
        sup = support[lab]
        k = max(last.get(s, -1) for s in sup)
        if k >= 0 and out[k][0] == lab:
            out[k][1] += d
        else:
            out.append([lab, d])
            for s in sup:
                last[s] = len(out) - 1
    return [(lab, d) for lab, d in out]


def count_stages(raw: list) -> int:
    """Exponentials after merging literally adjacent equal generators (how Liu et al. count stages)."""
    n, prev = 0, None
    for lab, _d in raw:
        if lab != prev:
            n += 1
        prev = lab
    return n


@dataclass(frozen=True)
class Formula:
    """A product formula for e^{-iHt} on the open chain of an XXZ model.

    ``nesting`` is used only with a separate field (h_x != 0 and not ``fold``);
    otherwise the formula is the bond sweep ``B``.
    """

    model: XXZ
    t: float
    order: int
    r: int
    ordering: str = "BW10"
    nesting: str = "XB"
    fold: bool = False

    @property
    def spec(self) -> str:
        return self.nesting if (self.model.hx and not self.fold) else "B"

    @property
    def family(self) -> str:
        field = "" if not self.model.hx else ("/fold" if self.fold else f"/{self.nesting}")
        return f"{self.ordering}{field}/o{self.order}"

    @property
    def label(self) -> str:
        return f"{self.family}/r{self.r}"

    def chain(self, N: int) -> Chain:
        return _chain(self.model, N, self.fold)

    def raw(self, N: int) -> list:
        return product_sequence(self.chain(N), self.spec, self.ordering, self.order, self.t, self.r)

    def sequence(self, N: int) -> list:
        ch = self.chain(N)
        return merge_sequence(self.raw(N), ch.support)

    def gates(self, N: int) -> list[Gate]:
        """Term-level gates (bond, field or folded two-qubit unitaries) in time order."""
        return _gates(self, N)

    def compiled(self, N: int) -> list[Gate]:
        """Gates whose every angle is one priced rotation (folded terms through KAK)."""
        return _compiled(self, N)

    def bulk_sizes(self, N_min: int = 12) -> tuple[int, int]:
        """Two chain lengths with the same boundaries, a whole number of pattern periods apart."""
        P = math.lcm(2, pattern_period(self.ordering))
        N1 = P * max(3, math.ceil((N_min + 2) / P)) if P > 2 else max(N_min, 12)
        return N1, N1 + 2 * P


@lru_cache(maxsize=64)
def _chain(model: XXZ, N: int, fold: bool) -> Chain:
    return Chain.xxz(model, N, fold)


@lru_cache(maxsize=256)
def _gates(f: Formula, N: int) -> list:
    ch = f.chain(N)
    return [ch.terms[lab].gate(d) for lab, d in f.sequence(N)]


@lru_cache(maxsize=256)
def _compiled(f: Formula, N: int) -> list:
    return compile_gates(f.gates(N))


# ---------------------------------------------------------------------------- gates

def _rx(theta: float) -> np.ndarray:
    return math.cos(theta / 2) * _I2 - 1j * math.sin(theta / 2) * _PX


def _rz(theta: float) -> np.ndarray:
    return math.cos(theta / 2) * _I2 - 1j * math.sin(theta / 2) * _PZ


@dataclass(frozen=True, eq=False)
class Gate:
    """One gate. Priced kinds: "xxz" (angles of the XX, YY, ZZ rotations, exp[-i(a XX + b YY + c ZZ)/2]),
    "rx", "rz", "zxz" (Rz(a) Rx(b) Rz(c), Rz(c) first). "u1"/"u2" carry a matrix and are compiled
    before pricing. ``derived`` marks angles from a numerical decomposition."""

    kind: str
    sites: tuple
    angles: tuple = ()
    mat: np.ndarray | None = None
    derived: bool = False

    def matrix(self) -> np.ndarray:
        k = self.kind
        if k == "xxz":
            return _CELL.two_qubit_part(self.angles)
        if k == "rx":
            return _rx(self.angles[0])
        if k == "rz":
            return _rz(self.angles[0])
        if k == "zxz":
            a, b, c = self.angles
            return _rz(a) @ _rx(b) @ _rz(c)
        return self.mat

    def with_angles(self, angles) -> Gate:
        return Gate(self.kind, self.sites, tuple(angles), self.mat, self.derived)


# KAK in the magic basis: B^dag (A x B) B is in SO(4) for A, B in SU(2), and
# B^dag exp(i(a XX + b YY + c ZZ)) B is diagonal.
_MAGIC = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]], dtype=complex) / math.sqrt(2)
_MAGIC_D = _MAGIC.conj().T
_PAULI_DIAG = np.array([np.real(np.diag(_MAGIC_D @ np.kron(P, P) @ _MAGIC)) for P in (_PX, _PY, _PZ)])
_KAK_SOLVE = np.linalg.inv(np.column_stack([np.ones(4), _PAULI_DIAG.T]))


def _kron_factor(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = K.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = np.linalg.svd(R)
    A = math.sqrt(s[0]) * u[:, 0].reshape(2, 2)
    B = math.sqrt(s[0]) * vh[0].reshape(2, 2)
    return A, B


def kak(U: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple, np.ndarray, np.ndarray]:
    """(A1, B1, (a, b, c), A2, B2) with U = phase (A1 x B1) exp[-i(a XX + b YY + c ZZ)/2] (A2 x B2)."""
    U = np.asarray(U, dtype=complex)
    V = U / np.linalg.det(U) ** 0.25
    Um = _MAGIC_D @ V @ _MAGIC
    M = Um.T @ Um
    for alpha in (0.6180339887, 1.4142135623, 2.7182818284, 0.3183098861):
        _w, P = np.linalg.eigh(M.real + alpha * M.imag)
        D = P.T @ M @ P
        if np.abs(D - np.diag(np.diag(D))).max() < 1e-10:
            break
    else:
        raise ArithmeticError("KAK: could not diagonalize Um^T Um")
    if np.linalg.det(P) < 0:
        P[:, 0] = -P[:, 0]
    s = np.sqrt(np.diag(D))
    K1 = Um @ P @ np.diag(1.0 / s)
    if np.linalg.det(K1.real) < 0:
        s[0] = -s[0]
        K1[:, 0] = -K1[:, 0]
    A1, B1 = _kron_factor(_MAGIC @ K1.real @ _MAGIC_D)
    A2, B2 = _kron_factor(_MAGIC @ P.T @ _MAGIC_D)
    th = np.angle(s)
    phi, a, b, c = _KAK_SOLVE @ th             # s_k = e^{i(phi + a x_k + b y_k + c z_k)}
    return A1, B1, (-2.0 * a, -2.0 * b, -2.0 * c), A2, B2


def zxz_angles(V: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) with V = phase Rz(a) Rx(b) Rz(c). A diagonal V is (a, 0, 0), an antidiagonal one
    (a, pi, 0), so that each costs one rotation."""
    W = V / np.sqrt(np.linalg.det(V))
    x, y = W[0, 0], W[1, 0]
    b = 2.0 * math.atan2(abs(y), abs(x))
    s = -2.0 * np.angle(x) if abs(x) > 1e-14 else 0.0             # a + c
    d = 2.0 * np.angle(1j * y) if abs(y) > 1e-14 else 0.0          # a - c
    if abs(y) <= 1e-14:
        return float(s), 0.0, 0.0
    if abs(x) <= 1e-14:
        return float(d), math.pi, 0.0
    return float((s + d) / 2), float(b), float((s - d) / 2)


SMALL_OFFSET = 1e-3
"""A decomposition angle closer than this to a multiple of pi/4 (and farther than eps) is split as
Rz(k pi/4) Rz(off + SPLIT_ANGLE) Rz(-SPLIT_ANGLE): KN takes ~1 s at offsets >= 1e-3, 19 s at 3e-4 and
over 60 s at 1e-4 (eps = 1e-7, measured 15 Oct). Costs one more rotation, an upper bound."""

SPLIT_ANGLE = 0.5

SNAP_TOL = 1e-10
"""Angles from a numerical decomposition within this of a multiple of pi/4 are set to it (as edges.py
does): numerical noise around 0 or pi/2 would otherwise be priced as a KN rotation."""


def _snap(theta: float) -> float:
    k = round(theta / QUARTER)
    return k * QUARTER if abs(theta - k * QUARTER) < SNAP_TOL else theta


def compile_gates(gates: list[Gate]) -> list[Gate]:
    """Every angle one rotation: u2 by KAK, then the single-qubit gates between two-qubit gates on a
    site multiplied and written as Rz Rx Rz. Bond and field gates pass unchanged."""
    if not any(g.kind in ("u1", "u2") for g in gates):
        return list(gates)
    memo: dict = {}
    out: list[Gate] = []
    pend: dict = {}

    def flush(site):
        V = pend.pop(site, None)
        if V is not None:
            out.append(Gate("zxz", (site,), tuple(_snap(a) for a in zxz_angles(V)), derived=True))

    for g in gates:
        if g.kind == "u1":
            s, = g.sites
            pend[s] = g.mat @ pend[s] if s in pend else g.mat
        elif g.kind == "u2":
            key = g.mat.tobytes()
            dec = memo.get(key)
            if dec is None:
                dec = memo[key] = kak(g.mat)
            A1, B1, abc, A2, B2 = dec
            i, j = g.sites
            pend[i] = A2 @ pend[i] if i in pend else A2
            pend[j] = B2 @ pend[j] if j in pend else B2
            flush(i)
            flush(j)
            out.append(Gate("xxz", (i, j), tuple(_snap(a) for a in abc), derived=True))
            pend[i], pend[j] = A1, B1
        else:
            for s in g.sites:
                flush(s)
            out.append(g)
    for s in sorted(pend):
        flush(s)
    return out


def fuse(gates) -> list[tuple[tuple, np.ndarray]]:
    """(sites, unitary) ops: single-qubit gates folded into the next two-qubit gate on their site,
    consecutive gates on the same bond multiplied."""
    out: list = []
    last: dict = {}
    pend: dict = {}
    for g in gates:
        sites, U = (g.sites, g.matrix()) if isinstance(g, Gate) else (tuple(g[0]), np.asarray(g[1]))
        if len(sites) == 1:
            s, = sites
            pend[s] = U @ pend[s] if s in pend else U
            continue
        i, j = sites
        if j != i + 1:
            raise ValueError(f"two-qubit gates must act on neighbours (i, i+1), not {sites}")
        A, B = pend.pop(i, None), pend.pop(j, None)
        if A is not None or B is not None:
            U = U @ np.kron(_I2 if A is None else A, _I2 if B is None else B)
        k = last.get(i, -1)
        if k >= 0 and last.get(j, -1) == k and out[k][0] == (i, j):
            out[k][1] = U @ out[k][1]
        else:
            out.append([(i, j), U])
            last[i] = last[j] = len(out) - 1
    for s in sorted(pend):
        out.append([(s,), pend[s]])
    return [(s, U) for s, U in out]


def apply_dense(gates, states: np.ndarray, N: int) -> np.ndarray:
    """Apply a gate list (or fused ops) to a batch of states, rows (n, 2^N), qubit 0 most significant."""
    ops = fuse(gates)
    n = states.shape[0]
    psi = np.ascontiguousarray(np.asarray(states, dtype=complex).T)
    for sites, U in ops:
        psi = np.matmul(U, psi.reshape(2 ** sites[0], U.shape[0], -1)).reshape(2 ** N, n)
    return np.ascontiguousarray(psi.T)


def unitary(gates, N: int) -> np.ndarray:
    """The circuit's 2^N x 2^N unitary (small N only)."""
    return apply_dense(gates, np.eye(2 ** N, dtype=complex), N).T


def apply_mps(psi, gates):
    """Apply a gate list to an open-chain snapevo.mps.MPS in place.

    Consecutive ops on disjoint sites form a layer, applied in the direction
    that keeps the orthogonality center moving, as apply_layered does.
    """
    ops = fuse(gates)
    run: list = []
    used: set = set()

    def flush():
        two = [(s, U) for s, U in run if len(s) == 2]
        for s, U in run:
            if len(s) == 1:
                psi.apply_1q(U, s[0])
        if two:
            lo, hi = min(s[0] for s, _ in two), max(s[0] for s, _ in two)
            if psi.center <= (lo + hi + 1) / 2:
                for s, U in sorted(two, key=lambda x: x[0][0]):
                    psi.apply_2q(U, s[0], move="right")
            else:
                for s, U in sorted(two, key=lambda x: -x[0][0]):
                    psi.apply_2q(U, s[0], move="left")

    for sites, U in ops:
        if used & set(sites):
            flush()
            run, used = [], set()
        run.append((sites, U))
        used |= set(sites)
    if run:
        flush()
    return psi


# ---------------------------------------------------------------------------- pricing

def _is_exact(theta: float) -> bool:
    k = theta / QUARTER
    return abs(k - round(k)) < 1e-12


def _is_identity(theta: float) -> bool:
    return abs(np.remainder(theta + np.pi, 2 * np.pi) - np.pi) <= 1e-12


def _key(theta: float, derived: bool) -> float:
    """The angle's synthesis key. numpy's rounding, as baseline.py's ``round`` on its np.float64 angles:
    Python's float round differs from it at exact decimal ties (0.2947489932760005 -> ...001 vs ...000,
    h_x = 1.6, order 4, t = 4, r = 9), which would synthesize such an angle under a different key."""
    return float(np.round(np.float64(theta), 11 if derived else 15))


def _tally(compiled: list[Gate], derived: set | None = None) -> tuple[dict, int]:
    """{angle key: number of rotations} over non-Clifford+T angles, and the number of non-identity angles.

    Keys of angles from a numerical decomposition are added to ``derived``.
    """
    cnt: dict = {}
    rot = 0
    for g in compiled:
        for th in g.angles:
            if not _is_identity(th):
                rot += 1
            if _is_exact(th):
                continue
            k = _key(th, g.derived)
            cnt[k] = cnt.get(k, 0) + 1
            if g.derived and derived is not None:
                derived.add(k)
    return cnt, rot


def _snap_within(theta: float, eps: float):
    """The multiple of pi/4 within diamond distance eps of Rz(theta), on theta's branch, or None (edges.py)."""
    rem = math.remainder(theta, 2 * math.pi)
    k = round(rem / QUARTER)
    if 2.0 * abs(math.sin((rem - k * QUARTER) / 2.0)) <= eps:
        return theta - rem + k * QUARTER, k
    return None


def _exact_T(compiled: list[Gate]) -> int:
    return sum(0 if round(th / QUARTER) % 2 == 0 else 1 for g in compiled for th in g.angles if _is_exact(th))


def bulk_tally(f: Formula, N_min: int = 12) -> dict:
    """Bulk densities per qubit: {angle: rotations} (non-Clifford+T), all rotations, exact-angle T."""
    N1, N2 = f.bulk_sizes(N_min)
    c1, r1 = _tally(f.compiled(N1))
    c2, r2 = _tally(f.compiled(N2))
    dN = N2 - N1
    dens = {k: (c2.get(k, 0) - c1.get(k, 0)) / dN for k in set(c1) | set(c2)}
    dens = {k: v for k, v in dens.items() if abs(v) > 1e-12}
    if any(v < 0 for v in dens.values()):
        raise ArithmeticError(f"{f.label}: negative bulk density, boundaries differ between N = {N1} and {N2}")
    return {"density": dens, "rotations": (r2 - r1) / dN,
            "exact_T": (_exact_T(f.compiled(N2)) - _exact_T(f.compiled(N1))) / dN, "sizes": (N1, N2)}


def plan_cost(f: Formula, eps: float) -> float:
    """Expected T per qubit with the calibrated KN law, before synthesis (baseline.plan_cost)."""
    from snapevo.synthesis import kn_cost
    b = bulk_tally(f)
    return sum(b["density"].values()) * kn_cost(eps) + b["exact_T"]


class KNCache:
    """KN non-mixed synthesis records, kept off runs/kn_cache.json.

    Reads ``seed`` (a copy of runs/kn_cache.json) once, plus every ``*.jsonl``
    in ``directory``; new syntheses (synthesis.synthesize without its cache)
    are appended to this process's own JSONL file there.
    """

    def __init__(self, directory: Path, seed: Path | None = None):
        from snapevo import synthesis
        self._syn = synthesis
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.recs: dict = {}
        if seed is not None and Path(seed).exists():
            self.recs.update(json.loads(Path(seed).read_text()))
        self.reload()
        self.path = self.dir / f"kn_{os.uname().nodename}_{os.getpid()}.jsonl"
        self.n_new = 0

    def reload(self) -> None:
        for p in sorted(glob.glob(str(self.dir / "*.jsonl"))):
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            k, rec = json.loads(line)
                        except ValueError:
                            continue
                        self.recs[k] = rec

    def __call__(self, theta: float, eps: float) -> dict:
        k = self._syn._key(theta, eps)
        rec = self.recs.get(k)
        if rec is None:
            rec = dict(self._syn.synthesize(theta, eps, use_cache=False))
            rec.pop("U0", None)
            rec.pop("U1", None)
            rec["theta_s"] = math.remainder(rec["theta_s"], 2 * math.pi)
            self.recs[k] = rec
            with open(self.path, "a") as fh:
                fh.write(json.dumps([k, rec]) + "\n")
            self.n_new += 1
        return self._syn._unwrap(rec, theta)


def _default_synth():
    from snapevo.synthesis import synthesize
    if "SNAPEVO_RUNS" not in os.environ:
        raise RuntimeError("baseline_lit never writes runs/kn_cache.json: pass synth= (e.g. a KNCache) "
                           "or set SNAPEVO_RUNS to a directory with a writable copy")
    return synthesize


def realize(f: Formula, eps: float, N: int, synth=None, N_min: int = 12) -> dict:
    """Synthesize every distinct angle at eps; the realized circuit at N and its bulk prices.

    ``Tq``, ``ledger_per_qubit`` and ``ledger2_per_qubit`` are bulk densities
    (baseline.realize_baseline's quantities); ``Tq_chain`` is the total on the
    N-site chain over N.
    """
    synth = synth or _default_synth()
    comp = f.compiled(N)
    bulk = bulk_tally(f, N_min)
    derived: set = set()
    chain_cnt, rot_chain = _tally(comp, derived)
    _tally(f.compiled(f.bulk_sizes(N_min)[1]), derived)
    syn: dict = {}
    n_snapped = n_split = 0
    for k in sorted(set(chain_cnt) | set(bulk["density"])):
        rec = None
        if k in derived:
            snap = _snap_within(k, eps)
            if snap is not None:                 # as edges.py: cheaper than KN and within the budget
                rec = {"expected_t": float(snap[1] % 2), "theta_s": snap[0], "l1": 0.0, "l2": 0.0, "how": "snap"}
                n_snapped += 1
            else:
                rem = math.remainder(k, 2 * math.pi)
                kk = round(rem / QUARTER)
                off = rem - kk * QUARTER
                if abs(off) < SMALL_OFFSET:      # KN stalls here: Rz(k pi/4) Rz(off + A) Rz(-A)
                    a, b = synth(off + SPLIT_ANGLE, eps), synth(-SPLIT_ANGLE, eps)
                    rec = {"expected_t": float(kk % 2) + a["expected_t"] + b["expected_t"],
                           "theta_s": k - off + (a["theta_s"] - SPLIT_ANGLE) + (b["theta_s"] + SPLIT_ANGLE),
                           "l1": sum((1 - x["p"]) * x["d_branch"] for x in (a, b)),
                           "l2": sum((1 - x["p"]) * x["d_branch"] ** 2 for x in (a, b)), "how": "split"}
                    n_split += 1
        if rec is None:
            r = synth(k, eps)
            rec = {"expected_t": r["expected_t"], "theta_s": r["theta_s"], "l1": (1 - r["p"]) * r["d_branch"],
                   "l2": (1 - r["p"]) * r["d_branch"] ** 2, "how": "kn"}
        syn[k] = rec
    real = []
    for g in comp:
        new = []
        for th in g.angles:
            new.append(th if _is_exact(th) else syn[_key(th, g.derived)]["theta_s"])
        real.append(g.with_angles(new) if g.angles else g)
    Tq = bulk["exact_T"] + sum(n * syn[k]["expected_t"] for k, n in bulk["density"].items())
    led = sum(n * syn[k]["l1"] for k, n in bulk["density"].items())
    led2 = sum(n * syn[k]["l2"] for k, n in bulk["density"].items())
    T_chain = _exact_T(comp) + sum(n * syn[k]["expected_t"] for k, n in chain_cnt.items())
    return {"gates": real, "Tq": Tq, "ledger_per_qubit": led, "ledger2_per_qubit": led2,
            "rotations_per_qubit": bulk["rotations"], "Tq_chain": T_chain / N, "rotations_chain": rot_chain / N,
            "n_distinct": len(bulk["density"]), "n_distinct_chain": len(chain_cnt), "n_snapped": n_snapped,
            "n_split": n_split, "split_per_qubit": sum(n for k, n in bulk["density"].items() if syn[k]["how"] == "split"),
            "eps": eps, "synth": {f"{k:.15f}": syn[k] for k in syn}}


# ---------------------------------------------------------------------------- N = 100 by MPS

def mps_infidelity(gates, model: XXZ, t: float, N: int = 100, states=None, dt: float = 0.2,
                   chi_max: int = 1024, cutoff: float = 1e-12, ref_dirs=(), ref_out: Path | None = None,
                   check: bool = True, log=print) -> dict:
    """Success-branch state infidelity of a gate list at N sites by MPS, like evaluate.mps_infidelity.

    References are read from the first of ``ref_dirs`` that has them (e.g.
    runs/<tag>/t<t>/ref, read only) and otherwise computed, and saved only to
    ``ref_out`` when given.
    """
    from snapevo.circuits import product_vectors
    from snapevo.evaluate import MPS_SET, ref_path
    from snapevo.mps import MPS, reference_state

    seed, n = states or MPS_SET
    vecs = product_vectors(N, n, seed)

    def ref(k, c):
        for d in ref_dirs:
            p = ref_path(Path(d), N, seed, k, dt, c, chi_max)
            if p.exists():
                return MPS.load(p)[0]
        cache = None if ref_out is None else ref_path(Path(ref_out), N, seed, k, dt, c, chi_max)
        return reference_state(model, t, vecs[k], dt, chi_max, c, cache, log)[0]

    def fid(k, c):
        v = ref(k, c)
        t0 = time.time()
        u = apply_mps(MPS.product(vecs[k], chi_max=chi_max, cutoff=c), gates)
        return abs(v.overlap(u)) ** 2 / abs(v.overlap(v) * u.overlap(u)), u, v, time.time() - t0

    rows = []
    for k in range(n):
        F, u, v, sec = fid(k, cutoff)
        T = math.sqrt(max(1.0 - F, 0.0))
        bar = math.sqrt(u.discarded) + math.sqrt(v.discarded)
        rows.append({"k": k, "F": F, "infid": 1.0 - F, "infid_lo": max(T - bar, 0.0) ** 2,
                     "infid_hi": (T + bar) ** 2, "discarded_u": u.discarded, "discarded_v": v.discarded,
                     "chi_u": u.chi_peak, "chi_v": v.chi_peak, "seconds": sec})
    F = np.array([r["F"] for r in rows])
    infid = float(1.0 - F.mean())
    hi = float(np.mean([r["infid_hi"] for r in rows]))
    out = {"N": N, "states": [seed, n], "dt": dt, "chi_max": chi_max, "cutoff": cutoff, "rows": rows,
           "infid": infid, "stderr": float(np.std(1.0 - F, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
           "c_site": float(1.0 - np.mean(F ** (1.0 / N))), "infid_hi": hi,
           "trunc_rel": float((hi - infid) / infid) if infid > 0 else 0.0,
           "chi_peak": int(max(max(r["chi_u"], r["chi_v"]) for r in rows))}
    if check:
        c2 = cutoff / 100
        F2, u2, v2, _ = fid(0, c2)
        out["trunc_check"] = {"cutoff": c2, "infid": 1.0 - F2, "chi": max(u2.chi_peak, v2.chi_peak),
                              "rel_change": abs((1.0 - F2) - rows[0]["infid"]) / max(1.0 - F2, 1e-300)}
    rel = out["trunc_check"]["rel_change"] if check else 0.0
    out["trunc_flag"] = out["trunc_rel"] > 0.05 or rel > 0.05
    log(f"mps: N = {N}, {n} inputs: 1 - F = {infid:.4e}, per site {out['c_site']:.3e}, chi peak "
        f"{out['chi_peak']}, truncation bar {out['trunc_rel']:.1%}" + (f", change at cutoff/100 {rel:.1e}"
                                                                         if check else ""))
    return out


# ---------------------------------------------------------------------------- screening

def pareto(points: list[dict], key: str = "C") -> list[dict]:
    """Points no other point beats in both Tq and ``key``."""
    front, best = [], math.inf
    for p in sorted(points, key=lambda p: (p["Tq"], p[key])):
        if p[key] < best:
            front.append(p)
            best = p[key]
    return front


def scan_family(model: XXZ, t: float, ordering: str, order: int, nesting: str = "XB", fold: bool = False,
                N: int = 12, states: tuple = HELD_OUT_SETS[1], c_floor: float = 1e-9, r_max: int = 512,
                r_dense: int = 10, r_growth: float = 1.15, eps_grid=EPS_GRID, N_eval: int = 100, synth=None,
                pairs=None, c_step2: float | None = None, r1_all_looser: bool = True, done=None, on_step=None,
                log=print) -> list[dict]:
    """Screen one family on the open chain of N sites by exact state vectors, baseline.scan's way.

    r runs 1, 2, ..., ``r_dense``, then grows by ``r_growth`` (baseline.scan:
    1 to 4, then 1.5x; here 1 to 10, then 1.15x, so that the T per qubit at a
    given C is resolved to about 15%) until the Trotter-only per-site C is below
    ``c_floor`` (order 1: below a tenth of one order-2 step's, ``c_step2``).
    Around eps* = sqrt(C_tr/3) / (0.45 R) each r is synthesized at the first
    grid budget at or below eps* and the two above it (every looser one at
    r = 1 when ``r1_all_looser``, as baseline.scan does; those points matter
    only above the r = 1 Trotter C). C is the per-site infidelity on
    ``states`` (default: the 64 inputs of held-out set 1).

    ``done`` (points of an interrupted run of the same family) is reused r by
    r instead of recomputed, and ``on_step(points)`` is called after each r, so
    a long scan can be checkpointed.
    """
    tin, tg = pairs if pairs is not None else state_pairs(model, N, t, states[1], states[0], bc="open")
    stop = c_floor
    if order == 1:
        if c_step2 is None:
            f2 = Formula(model, t, 2, 1, ordering, nesting, fold)
            c_step2 = local_infidelity(apply_dense(f2.gates(N), tin, N), tg, N)
        stop = max(c_floor, c_step2 / 10)
    pts = []
    prev: dict = {}
    for p in done or ():
        prev.setdefault(p["steps"], []).append(p)
    r = 1
    while True:
        f = Formula(model, t, order, r, ordering, nesting, fold)
        if r in prev:
            pts += prev[r]
            if prev[r][0]["C_trotter"] < stop or r >= r_max:
                break
            r = r + 1 if r < r_dense else max(r + 1, int(round(r * r_growth)))
            continue
        t0 = time.time()
        c_tr = local_infidelity(apply_dense(f.gates(N), tin, N), tg, N)
        bulk = bulk_tally(f)
        R = sum(bulk["density"].values())
        eps_star = math.sqrt(max(c_tr, 1e-300) / 3) / (0.45 * R) if R else eps_grid[0]
        looser = [e for e in eps_grid if e > eps_star]
        tighter = [e for e in eps_grid if e <= eps_star]
        for eps in (looser if (r == 1 and r1_all_looser) else looser[-2:]) + tighter[:1]:
            rb = realize(f, eps, N, synth)
            c = local_infidelity(apply_dense(rb["gates"], tin, N), tg, N)
            cb = bound_per_site(c, rb["ledger_per_qubit"], rb["ledger2_per_qubit"], N_eval)
            pts.append({"family": f.family, "ordering": ordering, "nesting": f.spec, "fold": fold, "order": order,
                        "steps": r, "eps": eps, "C": c, "C_trotter": c_tr, "C_bound": cb, "Tq": rb["Tq"],
                        "Tq_chain": rb["Tq_chain"], "rotations_per_qubit": rb["rotations_per_qubit"],
                        "ledger_per_qubit": rb["ledger_per_qubit"], "ledger2_per_qubit": rb["ledger2_per_qubit"],
                        "n_distinct": rb["n_distinct"], "n_gates": len(f.gates(N))})
        log(f"scan {f.family}: r = {r:3d}: Trotter C {c_tr:.3e}, R/q {R:.1f}, "
            + ", ".join(f"eps {p['eps']:.0e} C {p['C']:.2e} T/q {p['Tq']:.0f}" for p in pts if p["steps"] == r)
            + f" ({time.time() - t0:.1f}s)")
        if on_step is not None:
            on_step(pts)
        if c_tr < stop or r >= r_max:
            break
        r = r + 1 if r < r_dense else max(r + 1, int(round(r * r_growth)))
    return pts


def refine_family(model: XXZ, t: float, ordering: str, order: int, points: list[dict], targets,
                  nesting: str = "XB", fold: bool = False, N: int = 12, states: tuple = HELD_OUT_SETS[1],
                  pairs=None, synth=None, eps_grid=EPS_GRID, N_eval: int = 100, n_r: int = 3,
                  on_point=None, log=print) -> list[dict]:
    """Extra points so that each family's cheapest point at each target C is resolved on the eps grid.

    The scan picks eps near eps*(C_tr), where KN adds a third of the Trotter C.
    For a target well above C_tr the cheapest point uses a looser budget, and
    for a target just above C_tr a tighter one. For each target and each of
    the ``n_r`` smallest r whose Trotter C is below it, this walks the grid
    from the scanned budgets: looser until C exceeds the target, or tighter
    until C meets it. Points get ``"refine": True``. Refinement points already
    in ``points`` are reused; ``on_point(new_points)`` is called after each one.
    """
    tin, tg = pairs if pairs is not None else state_pairs(model, N, t, states[1], states[0], bc="open")
    by_r: dict = {}
    for p in points:
        by_r.setdefault(p["steps"], []).append(p)
    new: list[dict] = []
    grid = sorted(eps_grid, reverse=True)

    def at(r, eps):
        for p in by_r[r]:
            if p["eps"] == eps:
                return p
        f = Formula(model, t, order, r, ordering, nesting, fold)
        rb = realize(f, eps, N, synth)
        c = local_infidelity(apply_dense(rb["gates"], tin, N), tg, N)
        p = {**{k: by_r[r][0][k] for k in ("family", "ordering", "nesting", "fold", "order", "steps", "C_trotter",
                                            "n_gates")},
             "eps": eps, "C": c, "C_bound": bound_per_site(c, rb["ledger_per_qubit"], rb["ledger2_per_qubit"], N_eval),
             "Tq": rb["Tq"], "Tq_chain": rb["Tq_chain"], "rotations_per_qubit": rb["rotations_per_qubit"],
             "ledger_per_qubit": rb["ledger_per_qubit"], "ledger2_per_qubit": rb["ledger2_per_qubit"],
             "n_distinct": rb["n_distinct"], "refine": True}
        by_r[r].append(p)
        new.append(p)
        if on_point is not None:
            on_point(new)
        log(f"refine {p['family']}: r = {r}, eps {eps:.0e}: C {c:.3e}, T/q {p['Tq']:.0f}")
        return p

    for target in targets:
        for r in sorted(r for r in by_r if by_r[r][0]["C_trotter"] < target)[:n_r]:
            passing = [p["eps"] for p in by_r[r] if p["C"] <= target]
            if passing:
                for eps in [e for e in grid if e > max(passing)][::-1]:
                    if at(r, eps)["C"] > target:
                        break
            else:
                for eps in [e for e in grid if e < min(p["eps"] for p in by_r[r])]:
                    if at(r, eps)["C"] <= target:
                        break
    return new


def default_families(model: XXZ) -> list[dict]:
    """The families of the literature screen (kwargs of scan_family).

    h_x = 0: brickwall BW10 at orders 1, 2, 4, 6, sequential SEQ1 and wave WAVE3 at 2, 4, 6, and the
    mirror images BW01 and WAVE3R at 4 (SEQ0 is SEQ1's mirror image, so it has the same C at h_x = 0).
    h_x != 0: baseline.py's BW10/XEO at order 1; BW10 with each of the 12 BW_NESTINGS at 2, 4, 6;
    SEQ1 and WAVE3 with the field before (XB), between (BX) or outside ([B,X]) the bond sweep at
    2, 4, 6; and the folded field (Table 1 convention) with BW10, SEQ1, SEQ0, WAVE3 at 2, 4, 6.
    """
    if not model.hx:
        fam = [dict(ordering="BW10", order=o) for o in (1, 2, 4, 6)]
        fam += [dict(ordering=c, order=o) for c in ("SEQ1", "WAVE3") for o in (2, 4, 6)]
        return fam + [dict(ordering="BW01", order=4), dict(ordering="WAVE3R", order=4)]
    fam = [dict(ordering="BW10", nesting="XEO", order=1)]
    fam += [dict(ordering="BW10", nesting=n, order=o) for o in (2, 4, 6) for n in BW_NESTINGS]
    fam += [dict(ordering=c, nesting=n, order=o) for c in ("SEQ1", "WAVE3") for o in (2, 4, 6)
            for n in ("XB", "BX", "[B,X]")]
    return fam + [dict(ordering=c, nesting="B", fold=True, order=o) for c in ("BW10", "SEQ1", "SEQ0", "WAVE3")
                  for o in (2, 4, 6)]


def scan(model: XXZ, t: float, families: list[dict] | None = None, N: int = 12, states: tuple = HELD_OUT_SETS[1],
         synth=None, log=print, **kw) -> dict:
    """scan_family over ``families`` (dicts of ordering, order, nesting, fold; default_families(model) when
    None), with both fronts: ``front`` ranks by the success-branch C, ``front_bound`` by C with KN's
    fallback bound at N_eval sites, as baseline.scan."""
    families = default_families(model) if families is None else families
    tin, tg = state_pairs(model, N, t, states[1], states[0], bc="open")
    points = []
    for fam in families:
        points += scan_family(model, t, N=N, states=states, synth=synth, pairs=(tin, tg), log=log, **fam, **kw)
    return {"model": model.to_dict(), "t": t, "N": N, "bc": "open", "states": list(states),
            "version": SCAN_VERSION, "eps_grid": list(EPS_GRID), "points": points,
            "front": pareto(points, "C"), "front_bound": pareto(points, "C_bound")}
