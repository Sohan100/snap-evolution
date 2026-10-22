"""Rigorous product-formula route: Liu et al.'s ordering-aware Trotter bound, and the T count built on it.

Reference: Z. Liu, Z.-Y. Wei, M. Foss-Feig, A. V. Gorshkov, A. M. Childs, "Ordering-Aware Theory of
Trotter Error", arXiv:2610.03672 (v1). Section, equation and definition numbers below are the paper's.

The single-step bound (Sec. IV)
-------------------------------
An open chain carries n two-site terms H_1 ... H_n on qubits 1 ... n+1 (Sec. II.2). A product formula
is a list of gates e^{-i alpha_l tau g_l} in matrix-product order (Eq. 8, forward product Eq. 7), each
with a term label j_l and a generator g_l acting inside the qubits of H_{j_l}. S_k keeps the gates of
label <= k (Eq. 28), U_k = e^{-i H_[1,k] tau} (Eq. 27). Adding q terms at a time (Sec. IV.4) and slicing
each increment to its last L = p + q + v terms (Eq. 70),

    ||U_n - S_n|| <= ||U_k0 - S_k0|| + sum_{k = k0+q, k0+2q, ..., n} (||C^[L]_kq - F^[L]_kq|| + beta_F + beta_C)

(Eq. 68; q = 1 is Eq. 57), with k0 in {L-q+1, ..., L}, q | n - k0, and the first term computed exactly.
For each k:

* ||C^[L] - F^[L]|| is the sliced error increment on the L+1 qubits of H_{k-L+1} ... H_k, with
  F^[L] = e^{i H_[k-L+1,k-q] tau} e^{-i H_[k-L+1,k] tau} (Eq. 61) and C^[L] = S_{k-q}^dag S_k with every
  gate of label <= k-L deleted;
* beta_F >= int_0^tau ds (tau - s) ||[H_{k-L}, e^{is A} B e^{-is A}]||, A = H_[k-L+1,k-q], B = H_[k-q+1,k]
  (Eq. 63). The paper evaluates the integral by quadrature. Here the default is a rigorous upper bound
  (``_beta_F_taylor``): on each of 16 panels the integrand is majorized by its Taylor polynomial about
  the panel's left end, with norms of the nested commutators [H_{k-L}, ad_A^m X(s_i)] and the remainder
  ||[H_{k-L}, .]|| <= spread(H_{k-L}) ||ad_A^M B||, integrated exactly. It exceeds the converged
  quadrature by 1-30% (beta_F is a median 0.1% of the bound at order 4, 2% at order 6), and
  ``beta_f="quad"`` gives the paper's 8-node Gauss-Legendre value, converged to 1e-6;
* beta_C = sum over gates of label k-L of ||[W^[L]_{l-1}, G_l]||, with the recursion of Eq. 66
  (labels in (k-q, k] multiply on the right, labels in (k-L, k-q] conjugate) (Eq. 67).

Every local quantity is a spectral norm of a dense matrix on at most L+2 qubits. A norm whose cheap upper
bound min(||M||_F, sqrt(||M||_1 ||M||_inf)) is below a small share of the window's total takes that bound
instead, which is still an upper bound. Near-identity unitaries are carried as deviations E = U - I
(e^{-ix} - 1 = -2 sin^2(x/2) - i sin x), so the norms stay accurate at 1e-14 and below. The norms are
floating point, not interval arithmetic. With q = 2, v = 0 this is the paper's "q = 2 bound", and
v = 1 at order 6 is its refinement of Sec. VI.1 (Fig. 10). Every (q, v) gives a valid bound, so the
smallest over the v evaluated is used (Sec. IV.5).

Brickwall, sequential and wave orderings are the bit patterns c of Def. 1 (c_k = 1 when term k comes
before term k+1 in the first-order sweep): odd-even brickwall c = 1010... (baseline.py's E(h/2) O(h) E(h/2),
whose outer layer holds the bonds (0,1), (2,3), ... = labels 1, 3, ...), even-odd brickwall 0101...,
forward and backward sequential 11...1 and 00...0, and the depth-3 wave 0011 0011... and its
translations (Sec. V.4). Suzuki formulas follow Def. 5/6 with mu_p = 1/(4 - 4^{1/(p-1)}).

The field (h_x = 1.6)
---------------------
The terms follow the paper's convention (Def. 4, Table 1: on-site fields on the left site of each bond):
H_j = J/4 (XX + YY + Delta ZZ)_{j,j+1} + h_x/2 X_j, and the last term also carries the last site's
field, so sum_j H_j is our open-chain Hamiltonian (hamiltonian.XXZ.sparse(N, "open")). Two formulas:

* ``field="layer"`` is baseline.py's formula: separate field layers X(h/2) [bond sweep] X(h/2) in every
  second-order stage. The field gate on site j gets label j (site n+1's gets label n). The bound holds
  for it rigorously. The derivation of Eqs. 32-68 uses only that each gate of label j acts inside the
  qubits of H_j, that the generators of label <= k sum to H_[1,k], and that S_k is the formula restricted
  to labels <= k (S_k is then the same separate-layer formula for H_[1,k]). Gates of label < k-L act on
  qubits <= k-L and commute with W^[L]; the field gate of label k-L acts on qubit k-L only, so its
  commutator in Eq. 67 is zero and only the bond gate of label k-L enters, as in the paper. The tests
  check each inequality of the chain (Eqs. 32, 62, 63, 67) against full-chain matrices for this formula.
* ``field="fold"`` is the paper's own formula: each gate exponentiates the whole term H_j. Its gates
  are generic two-qubit unitaries (X_j does not commute with Y_j Y_{j+1}), not products of Pauli
  rotations, so it is bounded here but not priced in T.

The rigorous route therefore uses baseline.py's separate-layer formula at both fields.

Synthesis and the fidelity relation
-----------------------------------
For a total operator-norm budget delta = delta_T + delta_S, the Trotter part needs r steps with
r b(t/r) <= delta_T (single-step bound times r, as the paper does for its step counts, Sec. VI.1), and the
R non-Clifford rotations of the whole N-qubit circuit each get the KN diamond budget eps = delta_S / R,
priced with synthesis.kn_cost (calibrated mean expected T, independent of the angle; each distinct angle
is synthesized once and reused, but every copy costs its T gates). T falls as delta_S grows, so for a
given r the cheapest split puts delta_T at r b(t/r), and optimizing the split is a search over r.

Why 1 - F <= delta^2 for every input: for a unitary V with ||U - V|| <= delta (up to a global phase),
Re <U phi|V phi> >= 1 - delta^2/2, so 1 - F <= delta^2. A per-site target c at N sites then needs
delta = sqrt(1 - (1 - c)^N), about sqrt(N c) (``delta_for_c``), since F >= 1 - delta^2 for every input
gives 1 - mean F^(1/N) <= 1 - (1 - delta^2)^(1/N) = c.

A rotation at diamond distance D from its target is within op(D) = 2 sin(arcsin(D/2)/2) ~ D/2 of it in
phase-optimized operator norm, so using the diamond eps as the operator budget is conservative by
about 2x per rotation (about 1.1 T per rotation). KN's protocol is heralded with two branches
(synthesis.py): success U0 with D(U0, Rz) <= 0.9 eps, fallback U1 with (1 - p) D(U1, Rz) <= 0.1 eps.
* Success branch of every rotation: a unitary circuit within delta_T + 0.45 delta_S < delta of
  e^{-iHt}. This holds unconditionally.
* Fallbacks drawn: on a branch pattern b the circuit is a unitary within delta_T + sum_i e_i(b), so the
  branch-averaged 1 - F <= E[(delta_T + sum_i e_i)^2] = (delta_T + mu)^2 + sigma^2. KN's guarantee
  gives mu <= p op(0.9 eps) + (1 - p) op(D1) <= 0.521 eps per rotation, so mu <= 0.521 delta_S, and the
  result is <= delta^2 whenever sigma^2 = sum_i p_i (1 - p_i)(op(D1_i) - op(D0_i))^2 <= 0.728 delta_S^2
  (condition C). C does not follow from the guarantee, which bounds the averaged channel only, but it
  holds unless a rare fallback is very wrong: over the 15,929 entries of runs/kn_cache.json (16 Oct
  2026), Var e_i / eps^2 has median 0.098, 99.9th percentile 204 and maximum 8,127, so C holds for any
  circuit with R >= 1.2e4 rotations even if every rotation drew the worst entry (``condition_c``
  checks given synthesis records).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sl

from snapevo.hamiltonian import XXZ
from snapevo.library import QUARTER
from snapevo.synthesis import kn_cost

__all__ = ["Chain", "Formula", "StepBound", "ORDERINGS", "FAMILIES", "xxz_chain", "ti_chain", "dense_hamiltonian",
           "ordering_bits", "sweep", "suzuki_coefficients", "build_formula", "formula_unitary", "step_bound",
           "exact_step_error", "trotter_bound", "steps_needed", "rotation_count", "delta_for_c", "c_for_delta",
           "condition_c", "t_per_qubit", "best_split"]

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

ORDERINGS: dict[str, str] = {
    "bw": "10",        # odd-even brickwall, c = 1010...: baseline.py's E(h/2) O(h) E(h/2)
    "bw_eo": "01",     # even-odd brickwall, c = 0101...
    "seq": "1",        # forward sequential, c = 11...1
    "seq_b": "0",      # backward sequential, c = 00...0
    "wave": "0011",    # depth-3 wave (Sec. V.4, Fig. 6) and its translations
    "wave_1": "0110",
    "wave_2": "1100",
    "wave_3": "1001",
}
"""Ordering name -> the repeating unit of the bit pattern c (Def. 1), truncated to n - 1 bits."""

FAMILIES: dict[str, tuple[str, ...]] = {"brickwall": ("bw", "bw_eo"), "sequential": ("seq", "seq_b"),
                                        "wave": ("wave", "wave_1", "wave_2", "wave_3")}


# ---------------------------------------------------------------------------------------------- chains

@dataclass
class Chain:
    """An open chain of n two-site terms H_1 ... H_n on qubits 1 ... n+1 (labels as in the paper).

    ``pieces[j-1]`` maps generator names to 4x4 Hermitian matrices on qubits (j, j+1): "term" is H_j
    itself; an XXZ chain with a field also has "bond", "fL" (h_x/2 X on qubit j) and, on the last
    term, "fR" (h_x/2 X on qubit n+1). ``classes[j-1]`` names the pieces of label j, so that windows
    with the same content are computed once.
    """

    pieces: list[dict[str, np.ndarray]]
    classes: list[str]
    name: str = ""

    @property
    def n(self) -> int:
        return len(self.pieces)

    @property
    def N(self) -> int:
        return len(self.pieces) + 1

    def term(self, j: int) -> np.ndarray:
        return self.pieces[j - 1]["term"]

    def has_field(self) -> bool:
        return "fL" in self.pieces[0]

    def fingerprint(self) -> str:
        """A digest of the pieces of every class (part of the window cache key)."""
        import hashlib
        h = hashlib.sha1()
        seen = set()
        for cls, p in zip(self.classes, self.pieces):
            if cls in seen:
                continue
            seen.add(cls)
            for name in sorted(p):
                h.update(f"{cls}/{name}".encode())
                h.update(np.ascontiguousarray(p[name], dtype=complex).tobytes())
        return h.hexdigest()

    @staticmethod
    def support(j: int, name: str) -> tuple[int, ...]:
        if name == "fL":
            return (j,)
        if name == "fR":
            return (j + 1,)
        return (j, j + 1)


def xxz_chain(model: XXZ, N: int) -> Chain:
    """Our XXZ model on an open chain of N qubits, fields on the left site of each term (Def. 4, Table 1)."""
    if N < 3:
        raise ValueError("need at least 3 qubits")
    cxx, cyy, czz = model.bond_pauli
    bond = cxx * np.kron(_X, _X) + cyy * np.kron(_Y, _Y) + czz * np.kron(_Z, _Z)
    pieces, classes = [], []
    for j in range(1, N):
        p = {"bond": bond, "term": bond}
        cls = "bulk"
        if model.hx:
            fL = model.field_pauli * np.kron(_X, _I2)
            p = {"bond": bond, "fL": fL, "term": bond + fL}
            if j == N - 1:
                fR = model.field_pauli * np.kron(_I2, _X)
                p["fR"] = fR
                p["term"] = bond + fL + fR
                cls = "last"
        pieces.append(p)
        classes.append(cls)
    return Chain(pieces, classes, name=f"{model.tag} N={N}")


def ti_chain(h: np.ndarray, N: int, name: str = "ti") -> Chain:
    """A translation-invariant chain H = sum_j h_j of the paper's tests (Def. 4, Table 1, Eq. 80)."""
    h = np.asarray(h, dtype=complex)
    return Chain([{"term": h} for _ in range(N - 1)], ["bulk"] * (N - 1), name=f"{name} N={N}")


def dense_hamiltonian(chain: Chain, upto: int | None = None) -> np.ndarray:
    """H_[1,upto] as a dense matrix on qubits 1 ... upto+1 (all terms by default)."""
    k = chain.n if upto is None else upto
    return _slice_h(chain, 1, k, 1, k + 1)


# -------------------------------------------------------------------------------------------- formulas

def ordering_bits(name: str, n: int) -> tuple[int, ...]:
    """The n - 1 bits c of an ordering (Def. 1)."""
    unit = ORDERINGS[name]
    return tuple(int(unit[i % len(unit)]) for i in range(n - 1))


def sweep(c: tuple[int, ...]) -> list[int]:
    """A permutation pi of 1 ... n with pattern c (any one will do, Lemma 2), smallest label first among ties."""
    import heapq
    n = len(c) + 1
    indeg = {j: 0 for j in range(1, n + 1)}
    succ: dict[int, list[int]] = {j: [] for j in range(1, n + 1)}
    for k, bit in enumerate(c, start=1):
        a, b = (k, k + 1) if bit else (k + 1, k)
        succ[a].append(b)
        indeg[b] += 1
    ready = [j for j in indeg if indeg[j] == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        j = heapq.heappop(ready)
        out.append(j)
        for b in succ[j]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(ready, b)
    return out


def suzuki_coefficients(order: int) -> list[float]:
    """Durations (in units of tau) of the second-order stages of the order-p Suzuki formula (Def. 5)."""
    if order == 2:
        return [1.0]
    if order < 2 or order % 2:
        raise ValueError("order must be 2, 4, 6, ...")
    mu = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    inner = suzuki_coefficients(order - 2)
    return [x * a for a in (mu, mu, 1.0 - 4.0 * mu, mu, mu) for x in inner]


@dataclass(frozen=True)
class Formula:
    """A product formula: gates (label, generator name, alpha) in matrix-product order, delta = alpha tau."""

    gates: tuple[tuple[int, str, float], ...]
    order: int
    ordering: str
    field: str
    steps: int = 1


def _merge(raw: list[tuple[int, str, float]]) -> list[tuple[int, str, float]]:
    """Merge each gate into the latest earlier gate on its qubits when that gate has the same generator.

    Every gate in between acts on other qubits, so it commutes and the merge is exact (for every S_k too).
    """
    out: list[list] = []
    last: dict[int, int] = {}
    for label, name, a in raw:
        qs = Chain.support(label, name)
        idx = max((last.get(x, -1) for x in qs), default=-1)
        if idx >= 0 and out[idx][0] == label and out[idx][1] == name:
            out[idx][2] += a
            continue
        out.append([label, name, a])
        for x in qs:
            last[x] = len(out) - 1
    return [(g[0], g[1], float(g[2])) for g in out if g[2] != 0.0]


def build_formula(chain: Chain, order: int, ordering: str = "bw", field: str = "layer", steps: int = 1) -> Formula:
    """The order-p Suzuki formula with term ordering ``ordering``, ``steps`` steps of unit length, merged.

    ``field``: "layer" puts the field in its own layers at both ends of every second-order stage
    (baseline.py), "fold" exponentiates whole terms (the paper's convention). Without a field both give
    the bond formula.
    """
    if field not in ("layer", "fold"):
        raise ValueError("field must be 'layer' or 'fold'")
    pi = sweep(ordering_bits(ordering, chain.n))
    layer = field == "layer" and chain.has_field()
    bond = "bond" if layer else "term"
    fields = [(j, "fL") for j in range(1, chain.n + 1)] + [(chain.n, "fR")] if layer else []

    def s2(h):
        seq = [(j, name, h / 2) for j, name in fields]
        seq += [(j, bond, h / 2) for j in pi] + [(j, bond, h / 2) for j in reversed(pi)]
        return seq + [(j, name, h / 2) for j, name in fields]

    raw = []
    for _ in range(steps):
        for a in suzuki_coefficients(order):
            raw += s2(a)
    return Formula(tuple(_merge(raw)), order, ordering, field, steps)


# ---------------------------------------------------------------------------------- dense local algebra

def _em1(x: np.ndarray) -> np.ndarray:
    """e^{ix} - 1 without cancellation."""
    return -2.0 * np.sin(x / 2) ** 2 + 1j * np.sin(x)


def _expm1_herm(H: np.ndarray, t: float) -> np.ndarray:
    """e^{-i t H} - I for Hermitian H."""
    w, V = np.linalg.eigh(H)
    return (V * _em1(-t * w)) @ V.conj().T


def _embed(op4: np.ndarray, i: int, m: int) -> np.ndarray:
    """A 4x4 operator on qubits (i, i+1) of m qubits (qubit 0 most significant)."""
    return np.kron(np.kron(np.eye(2 ** i), op4), np.eye(2 ** (m - i - 2)))


def _right(M: np.ndarray, g4: np.ndarray, i: int, m: int) -> np.ndarray:
    """M (I x g4 x I) with g4 on qubits (i, i+1)."""
    D = M.shape[0]
    a, r = 2 ** i, 2 ** (m - i - 2)
    T = np.tensordot(M.reshape(D, a, 4, r), g4, axes=([2], [0]))        # (D, a, r, 4)
    return np.ascontiguousarray(np.moveaxis(T, 3, 2)).reshape(D, D)


def _left(g4: np.ndarray, M: np.ndarray, i: int, m: int) -> np.ndarray:
    """(I x g4 x I) M with g4 on qubits (i, i+1)."""
    D = M.shape[1]
    a, r = 2 ** i, 2 ** (m - i - 2)
    T = np.tensordot(g4, M.reshape(a, 4, r, D), axes=([1], [1]))         # (4, a, r, D)
    return np.ascontiguousarray(np.moveaxis(T, 0, 1)).reshape(a * 4 * r, D)


def _top_commutator_blocks(op4: np.ndarray, M: np.ndarray) -> np.ndarray:
    """[op4 (x) I, I_1 (x) M] on 1 + m qubits, op4 on qubits (0, 1), M on qubits 1 ... m (block form)."""
    D = M.shape[0]
    h = D // 2
    blocks = op4.reshape(2, 2, 2, 2)                                    # <a x| op |b y>
    out = np.empty((2 * D, 2 * D), dtype=complex)
    Mr = M.reshape(D, 2, h)
    Ml = M.reshape(2, h, D)
    for a in range(2):
        for b in range(2):
            e = blocks[a, :, b, :]                                      # 2x2 on qubit 1 = M's qubit 0
            left = np.tensordot(e, Ml, axes=([1], [0])).reshape(D, D)   # (e x I) M
            right = np.tensordot(Mr, e, axes=([1], [0]))                # (D, h, 2)
            right = np.ascontiguousarray(np.moveaxis(right, 2, 1)).reshape(D, D)   # M (e x I)
            out[a * D:(a + 1) * D, b * D:(b + 1) * D] = left - right
    return out


def _opnorm(M: np.ndarray) -> float:
    """Spectral norm of a dense matrix (svdvals; top eigenvalue of M^dag M above 1024 dimensions)."""
    if M.shape[0] <= 1024:
        return float(sl.svdvals(M, check_finite=False)[0])
    K = M.conj().T @ M
    lam = sl.eigvalsh(K, subset_by_index=[K.shape[0] - 1, K.shape[0] - 1], check_finite=False)[0]
    return float(math.sqrt(max(lam, 0.0)))


def _cheap_norm(M: np.ndarray) -> float:
    """An upper bound on ||M||: min(||M||_F, sqrt(||M||_1 ||M||_inf))."""
    a = np.abs(M)
    return min(math.sqrt(float(np.sum(a * a))), math.sqrt(float(a.sum(axis=0).max() * a.sum(axis=1).max())))


def _opnorm_ub(M: np.ndarray, budget: float) -> float:
    """An upper bound on ||M||: the cheap bound when it is at most ``budget``, else ||M||."""
    cheap = _cheap_norm(M)
    return cheap if cheap <= budget else _opnorm(M)


def _slice_h(chain: Chain, a: int, b: int, q0: int, q1: int) -> np.ndarray:
    """sum_{j=a}^{b} H_j on qubits q0 ... q1."""
    m = q1 - q0 + 1
    H = np.zeros((2 ** m, 2 ** m), dtype=complex)
    for j in range(a, b + 1):
        H += _embed(chain.term(j), j - q0, m)
    return H


class _Gates:
    """Gate unitaries and deviations, computed once per (class, generator, delta)."""

    def __init__(self, chain: Chain):
        self.chain = chain
        self.cache: dict = {}

    def get(self, label: int, name: str, delta: float) -> tuple[np.ndarray, np.ndarray]:
        key = (self.chain.classes[label - 1], name, delta)
        if key not in self.cache:
            E = _expm1_herm(self.chain.pieces[label - 1][name], delta)
            self.cache[key] = (np.eye(4) + E, E)
        return self.cache[key]


def _product_deviation(gates: _Gates, seq, tau: float, q0: int, m: int) -> np.ndarray:
    """Pi_l G_l - I over the gates in ``seq`` (matrix-product order) on qubits q0 ... q0+m-1."""
    E = np.zeros((2 ** m, 2 ** m), dtype=complex)
    for label, name, a in seq:
        G, Eg = gates.get(label, name, a * tau)
        i = label - q0
        E = _right(E, G, i, m) + _embed(Eg, i, m)                       # (I + E) G - I
    return E


def formula_unitary(chain: Chain, formula: Formula, tau: float) -> np.ndarray:
    """The formula's unitary on all N qubits (dense; small chains only)."""
    return np.eye(2 ** chain.N) + _product_deviation(_Gates(chain), formula.gates, tau, 1, chain.N)


# ------------------------------------------------------------------------------------------- beta_F

def _gl_nodes(tau: float, panels: int, nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(nodes)
    s, ws = [], []
    h = tau / panels
    for k in range(panels):
        s.append(k * h + (x + 1) * h / 2)
        ws.append(w * h / 2)
    return np.concatenate(s), np.concatenate(ws)


def _beta_F_quad(Hb: np.ndarray, A: np.ndarray, B: np.ndarray, tau: float, panels: int = 1,
                 nodes: int = 8) -> float:
    """Eq. 63 by composite Gauss-Legendre quadrature (the paper's evaluation; not an upper bound)."""
    wA, VA = np.linalg.eigh(A)
    Bt = VA.conj().T @ B @ VA
    s, w = _gl_nodes(tau, panels, nodes)
    tot = 0.0
    for si, wi in zip(s, w):
        ph = np.exp(1j * si * wA)
        Xs = VA @ (Bt * np.outer(ph, ph.conj())) @ VA.conj().T
        tot += wi * (tau - si) * _opnorm(_top_commutator_blocks(Hb, Xs))
    return float(tot)


def _beta_F_taylor(Hb: np.ndarray, A: np.ndarray, B: np.ndarray, tau: float, M: int, panels: int = 16,
                   budget: float = 0.0) -> float:
    """A rigorous upper bound on int_0^tau (tau - s) ||[Hb, X(s)]|| ds, X(s) = e^{isA} B e^{-isA} (Eq. 63).

    On a panel [s_i, s_i + h], X(s_i + u) = sum_{m<M} (iu)^m/m! ad_A^m X(s_i) + R_M(u) with
    ||[Hb, R_M]|| <= u^M/M! spread(Hb) ||ad_A^M B|| (unitary invariance, and ||[Hb, Z]|| <= spread(Hb) ||Z||).
    So ||[Hb, X(s_i + u)]|| <= sum_{m<M} u^m/m! a_im + u^M/M! K_M with a_im = ||[Hb, ad_A^m X(s_i)]||, and
    the weight (tau - s) integrates against each power exactly. a_i0 and a_i1 are spectral norms unless
    their cheap bound adds at most ``budget``; higher m take the cheap bound (small terms near a node).
    The integrand vanishes like s^(L-q) at 0, so M = L - q + 2 resolves the first panel too.
    """
    wA, VA = np.linalg.eigh(A)
    Bt = VA.conj().T @ B @ VA
    Dl = wA[:, None] - wA[None, :]                     # ad_A in A's eigenbasis
    eb = np.linalg.eigvalsh(Hb)
    K_M = float(eb[-1] - eb[0]) * _opnorm(Bt * Dl ** M)
    h = tau / panels
    fact = [math.factorial(m) for m in range(M + 3)]

    def weight(c, m):                                  # int_0^h (c - u) u^m / m! du
        return c * h ** (m + 1) / fact[m + 1] - (m + 1) * h ** (m + 2) / fact[m + 2]

    total = 0.0
    for i in range(panels):
        si = i * h
        c = tau - si
        ph = np.exp(1j * si * wA)
        Bs = Bt * np.outer(ph, ph.conj())
        Dm = np.ones_like(Dl)
        for m in range(M):
            J = weight(c, m)
            Y = VA @ (Bs * Dm) @ VA.conj().T           # ad_A^m X(s_i) (up to a phase i^m)
            Cm = _top_commutator_blocks(Hb, Y)
            a = _opnorm_ub(Cm, budget / J if J > 0 else math.inf) if m < 2 else _cheap_norm(Cm)
            total += a * J
            Dm = Dm * Dl
        total += K_M * weight(c, M)
    return float(total)


# ------------------------------------------------------------------------------------------- the bound

@dataclass
class StepBound:
    """The single-step bound of Eq. 68 and its parts."""

    total: float
    exact_head: float                      # ||U_k0 - S_k0||
    k0: int
    L: int
    q: int
    windows: list[dict] = field(default_factory=list)   # k, local, beta_F, beta_C (+ beta_F_quad with check)

    @property
    def local(self) -> float:
        return sum(w["local"] for w in self.windows)

    @property
    def beta_F(self) -> float:
        return sum(w["beta_F"] for w in self.windows)

    @property
    def beta_C(self) -> float:
        return sum(w["beta_C"] for w in self.windows)


_WINDOW_CACHE: dict = {}

_SCREEN = 1e-5
"""A beta_C commutator whose cheap bound adds less than this share of the window's local term takes it."""

_SCREEN_F = 1e-4
"""Share of the window's local + beta_C that the cheap bounds in beta_F may add, spread over its terms."""


def _window(chain: Chain, gates: _Gates, sub, k: int, L: int, q: int, tau: float, beta_f: str,
            check: bool) -> dict:
    """||C^[L]_kq - F^[L]_kq||, beta_F (Eq. 63) and beta_C (Eq. 67) for the increment ending at term k."""
    lo = k - L                       # the boundary term H_{k-L}
    q0, ms = lo + 1, L + 1           # the slice: qubits k-L+1 ... k+1
    key = (tuple((j - lo, nm, a) for j, nm, a in sub), tuple(chain.classes[j - 1] for j in range(lo, k + 1)),
           chain.fingerprint(), L, q, tau, beta_f, check)
    if key in _WINDOW_CACHE:
        return dict(_WINDOW_CACHE[key], k=k)
    A = _slice_h(chain, lo + 1, k - q, q0, k + 1)
    B = _slice_h(chain, k - q + 1, k, q0, k + 1)
    # F^[L] (Eq. 61) and C^[L] as deviations from the identity
    EA = _expm1_herm(A, -tau)
    EAB = _expm1_herm(A + B, tau)
    EF = EA + EAB + EA @ EAB
    in_A = [g for g in sub if lo < g[0] <= k - q]
    in_AB = [g for g in sub if lo < g[0] <= k]
    PA = _product_deviation(gates, in_A, tau, q0, ms)
    PAB = _product_deviation(gates, in_AB, tau, q0, ms)
    EC = PA.conj().T + PAB + PA.conj().T @ PAB
    local = _opnorm(EC - EF)
    # beta_C: the recursion of Eq. 66 on the slice, commutators with the boundary gates (Eq. 67)
    W = np.zeros((2 ** ms, 2 ** ms), dtype=complex)        # W^[L] - I
    beta_C = 0.0
    n_comm = 0
    for label, name, a in sub:
        G, Eg = gates.get(label, name, a * tau)
        if label > k - q:
            W = _right(W, G, label - q0, ms) + _embed(Eg, label - q0, ms)
        elif label > lo:
            W = _left(G.conj().T, _right(W, G, label - q0, ms), label - q0, ms)
        elif label == lo and chain.support(label, name) != (lo,):     # a field on qubit k-L commutes
            beta_C += _opnorm_ub(_top_commutator_blocks(Eg, W), _SCREEN * local)
            n_comm += 1
    # beta_F: Eq. 63, as a rigorous upper bound or by quadrature
    Hb = chain.term(lo)
    M = L - q + 2
    terms = 16 * 2
    rig = _beta_F_taylor(Hb, A, B, tau, M, budget=_SCREEN_F * (local + beta_C) / terms)
    out = {"k": k, "local": local, "beta_C": beta_C, "n_commutators": n_comm, "beta_F_taylor": rig}
    if beta_f == "quad" or check:
        out["beta_F_quad"] = _beta_F_quad(Hb, A, B, tau)
    if check:
        out["beta_F_quad_fine"] = _beta_F_quad(Hb, A, B, tau, panels=4)
    out["beta_F"] = rig if beta_f == "taylor" else out["beta_F_quad"]
    _WINDOW_CACHE[key] = out
    return dict(out)


def step_bound(chain: Chain, formula: Formula, tau: float, q: int = 2, v: int = 0, beta_f: str = "taylor",
               check: bool = False) -> StepBound:
    """The (q, v) bound of Eq. 68 (Eq. 70: L = p + q + v) on ||e^{-iH tau} - S(tau)|| for one step.

    ``formula`` must be a single step (steps = 1). ``beta_f`` is "taylor" (rigorous, default) or
    "quad" (the paper's quadrature). ``check`` also stores both quadratures in each window.
    """
    if formula.steps != 1:
        raise ValueError("step_bound takes a one-step formula")
    if beta_f not in ("taylor", "quad"):
        raise ValueError("beta_f must be 'taylor' or 'quad'")
    n = chain.n
    L = formula.order + q + v
    gates = _Gates(chain)
    if n <= L:
        k0 = n
    else:
        k0 = next(k for k in range(L - q + 1, L + 1) if (n - k) % q == 0)
    head_seq = [g for g in formula.gates if g[0] <= k0]
    EU = _expm1_herm(dense_hamiltonian(chain, k0), tau)
    ES = _product_deviation(gates, head_seq, tau, 1, k0 + 1)
    head = _opnorm(EU - ES)
    out = StepBound(head, head, k0, L, q)
    for k in range(k0 + q, n + 1, q):
        sub = [g for g in formula.gates if k - L <= g[0] <= k]
        w = _window(chain, gates, sub, k, L, q, tau, beta_f, check)
        out.windows.append(w)
        out.total += w["local"] + w["beta_F"] + w["beta_C"]
    return out


def exact_step_error(chain: Chain, formula: Formula, tau: float) -> float:
    """||e^{-iH tau} - S(tau)|| by dense matrices (the paper's ground truth, Sec. VI.1)."""
    gates = _Gates(chain)
    EU = _expm1_herm(dense_hamiltonian(chain), tau)
    ES = _product_deviation(gates, formula.gates, tau, 1, chain.N)
    return _opnorm(EU - ES)


def trotter_bound(chain: Chain, formula: Formula, t: float, r: int, **kw) -> float:
    """r times the single-step bound at tau = t / r: ||U^r - S^r|| <= r ||U - S|| (Sec. VI.1)."""
    return r * step_bound(chain, formula, t / r, **kw).total


def steps_needed(bound_of_r, delta_T: float, r_max: int = 10 ** 8) -> int:
    """r with bound_of_r(r) <= delta_T < bound_of_r(r - 1), by doubling and bisection (r = 1 if one step is enough).

    bound_of_r should fall with r; the bracket holds at the returned r either way.
    """
    if bound_of_r(1) <= delta_T:
        return 1
    lo, hi = 1, 2
    while bound_of_r(hi) > delta_T:
        lo, hi = hi, 2 * hi
        if hi > r_max:
            raise ValueError("no r below r_max meets the target")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if bound_of_r(mid) <= delta_T:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------------------------------ circuit and cost

def _gate_rotations(chain: Chain, name: str, delta: float, model: XXZ) -> list[float]:
    """Pauli-rotation angles of one gate (XX, YY, ZZ for a bond, X for a field), as in the xxz+x cell."""
    if name in ("fL", "fR"):
        return [model.hx * delta]
    if name == "term" and chain.has_field():
        raise ValueError("a folded term with a field is a generic two-qubit gate, not Pauli rotations")
    cxx, cyy, czz = model.bond_pauli
    return [2 * cxx * delta, 2 * cyy * delta, 2 * czz * delta]


def rotation_count(model: XXZ, N: int, order: int, ordering: str, t: float, r: int,
                   field: str = "layer") -> dict:
    """Non-Clifford rotations R of the whole r-step circuit for time t on N qubits, after merging.

    Counted exactly for 1, 2 and 3 steps (adjacent steps merge their outer layers), checked to be
    linear, and extended linearly in r. Angles that are multiples of pi/4 are exact Clifford+T and are
    counted apart.
    """
    chain = xxz_chain(model, N)
    tau = t / r

    def count(steps):
        f = build_formula(chain, order, ordering, field, steps)
        rot, exact, angles = 0, 0, set()
        for _label, name, a in f.gates:
            for th in _gate_rotations(chain, name, a * tau, model):
                k = th / QUARTER
                if abs(k - round(k)) < 1e-12:
                    exact += 1
                else:
                    rot += 1
                    angles.add(round(abs(th), 13))
        return rot, exact, angles, len(f.gates)

    c1, c2, c3 = count(1), count(2), count(3)
    if c3[0] - c2[0] != c2[0] - c1[0]:
        raise AssertionError("rotation count is not linear in the number of steps")
    per = c2[0] - c1[0]
    R = c1[0] + (r - 1) * per
    exact = c1[1] + (r - 1) * (c2[1] - c1[1])
    return {"R": R, "per_step": per, "first_step": c1[0], "exact": exact,
            "rotations_per_qubit": R / N, "distinct_angles": len(c3[2]), "gates_per_step": c2[3] - c1[3]}


def delta_for_c(c: float, N: int) -> float:
    """Operator-norm budget delta for a per-site infidelity c at N sites: delta^2 = 1 - (1 - c)^N."""
    return math.sqrt(-math.expm1(N * math.log1p(-c)))


def c_for_delta(delta: float, N: int) -> float:
    """Inverse of delta_for_c: the per-site infidelity c guaranteed by an operator-norm error delta."""
    return -math.expm1(math.log1p(-delta * delta) / N)


def _op_of_diamond(D: float) -> float:
    """Phase-optimized operator distance of single-qubit unitaries at diamond distance D."""
    return 2.0 * math.sin(math.asin(min(D, 2.0) / 2.0) / 2.0)


def condition_c(records, delta_S: float, counts=None) -> dict:
    """Check sigma^2 = sum_i p_i (1 - p_i) (op(D1_i) - op(D0_i))^2 <= 0.728 delta_S^2 for synthesized rotations.

    ``records`` are synthesis.synthesize results (one per distinct angle), ``counts`` how many rotations
    use each (default 1). Also returns mu = sum_i (p_i op(D0_i) + (1 - p_i) op(D1_i)), which KN's
    guarantee keeps below 0.521 delta_S.
    """
    counts = counts or [1] * len(records)
    sig2 = sum(n * r["p"] * (1 - r["p"]) * (_op_of_diamond(r["d1"]) - _op_of_diamond(r["d0"])) ** 2
               for r, n in zip(records, counts))
    mu = sum(n * (r["p"] * _op_of_diamond(r["d0"]) + (1 - r["p"]) * _op_of_diamond(r["d1"]))
             for r, n in zip(records, counts))
    ok = sig2 <= 0.728 * delta_S ** 2 and mu <= 0.521 * delta_S
    return {"sigma2": sig2, "mu": mu, "limit": 0.728 * delta_S ** 2, "ok": ok}


def t_per_qubit(R: int, delta_S: float, N: int) -> float:
    """Expected T per qubit with every rotation at the KN diamond budget eps = delta_S / R."""
    return R * kn_cost(delta_S / R) / N if R else 0.0


def best_split(delta: float, N: int, candidates) -> dict | None:
    """Cheapest (r, delta_T = r b(t/r), delta_S = delta - delta_T) among candidate steps.

    ``candidates`` is an iterable of (r, r b(t/r), R(r)). For a fixed r the cheapest split puts
    delta_T at its smallest allowed value, so the search over the split is a search over r.
    """
    best = None
    for r, dT, R in candidates:
        if dT >= delta:
            continue
        Tq = t_per_qubit(R, delta - dT, N)
        if best is None or Tq < best["Tq"]:
            best = {"r": r, "delta_T": dT, "delta_S": delta - dT, "R": R, "eps": (delta - dT) / R if R else None,
                    "Tq": Tq}
    return best


# ------------------------------------------------------------------------- production runs (litbase)
#
#   python -m snapevo.rigorous grid --orders 2 4 --budget 540     single-step bounds on a tau grid
#   python -m snapevo.rigorous estimate                           T per qubit against c, best formula
#   python -m snapevo.rigorous verify --budget 540                exact bounds at the chosen r
#   python -m snapevo.rigorous table                              the comparison table (markdown)
#
# Outputs go to --out (default /pscratch/sd/s/sohan100/snapevo-scratch/litbase/rigorous). Run with one BLAS
# thread per worker (OPENBLAS_NUM_THREADS=1) under taskset; grid and verify append to bounds.jsonl and
# skip what is there, so they resume across commands. The curves interpolate b(tau) between grid points
# (an estimate); every table entry is an exact bound at its own r.

FIELDS = (1.6, 0.0)
TIMES = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
SIZES = (50, 100)
C_GRID = tuple(10.0 ** (-e / 4) for e in range(8, 45))          # 1e-2 ... 1e-11, four per decade
TAU_GRIDS = {2: (2e-4, 0.6, 10), 4: (0.02, 1.6, 10), 6: (0.08, 1.6, 8)}    # (min, max, points per decade)
DEFAULT_OUT = "/pscratch/sd/s/sohan100/snapevo-scratch/litbase/rigorous"


def tau_grid(order: int) -> list[float]:
    lo, hi, per = TAU_GRIDS[order]
    k0, k1 = math.floor(per * math.log10(lo) + 1e-9), math.ceil(per * math.log10(hi) - 1e-9)
    return [float(f"{10.0 ** (k / per):.12g}") for k in range(k0, k1 + 1)]


def _key(rec: dict) -> tuple:
    return (float(rec["hx"]), int(rec["order"]), rec["ordering"], int(rec["v"]), int(rec["N"]),
            rec.get("field", "layer"), float(f"{rec['tau']:.12g}"))


def _bound_job(job):
    import time
    hx, order, ordering, v, Ns, tau, fld = job[:7]
    source = job[7] if len(job) > 7 else "grid"
    model = XXZ(0.25, hx)
    out = []
    for N in Ns:
        ch = xxz_chain(model, N)
        f = build_formula(ch, order, ordering, fld)
        t0 = time.time()
        b = step_bound(ch, f, tau, v=v)
        out.append({"hx": hx, "order": order, "ordering": ordering, "v": v, "N": N, "field": fld, "tau": tau,
                    "total": b.total, "head": b.exact_head, "local": b.local, "beta_F": b.beta_F,
                    "beta_C": b.beta_C, "k0": b.k0, "L": b.L, "windows": len(b.windows), "beta_f": "taylor",
                    "seconds": time.time() - t0, "source": source})
    return out


def read_bounds(path) -> dict:
    import json
    from pathlib import Path
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[_key(rec)] = rec
    return out


def run_jobs(jobs, path, workers: int = 4, budget: float | None = None, log=print) -> int:
    """Run (hx, order, ordering, v, Ns, tau, field[, source]) jobs not yet in ``path``, appending as they finish."""
    import json
    import multiprocessing as mp
    import time
    have = read_bounds(path)
    todo = [j for j in jobs if any((float(j[0]), j[1], j[2], j[3], N, j[6], float(f"{j[5]:.12g}")) not in have
                                   for N in j[4])]
    log(f"jobs: {len(todo)} of {len(jobs)} to run")
    if not todo:
        return 0
    t0, done = time.time(), 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool, open(path, "a") as fh:
        for recs in pool.imap_unordered(_bound_job, todo):
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            r = recs[-1]
            log(f"  h_x {r['hx']:g} p {r['order']} {r['ordering']:6s} v {r['v']} tau {r['tau']:.4g}: "
                f"b(N={r['N']}) {r['total']:.3e}  [{r['seconds']:.0f} s]  ({done}/{len(todo)})")
            if budget is not None and time.time() - t0 > budget:
                log("jobs: time budget reached, stopping")
                pool.terminate()
                break
    return done


def _interp_bound(samples: list[tuple[float, float]]):
    """b(tau) from samples, monotone cubic in log-log (PCHIP); nan outside the sampled range."""
    from scipy.interpolate import PchipInterpolator
    s = sorted(samples)
    x = np.log([a for a, _ in s])
    y = np.log([b for _, b in s])
    f = PchipInterpolator(x, y, extrapolate=False)

    def b(tau):
        return np.exp(f(np.log(np.asarray(tau, dtype=float))))
    return b, s[0][0], s[-1][0]


_ROT_CACHE: dict = {}


def _rotations(hx: float, N: int, order: int, ordering: str) -> tuple[int, int]:
    """(R of one step, R per extra step) at a generic tau."""
    key = (hx, N, order, ordering)
    if key not in _ROT_CACHE:
        rc = rotation_count(XXZ(0.25, hx), N, order, ordering, t=0.1 * math.sqrt(2.0), r=1)
        _ROT_CACHE[key] = (rc["first_step"], rc["per_step"])
    return _ROT_CACHE[key]


def optimize(b, tau_lo: float, tau_hi: float, rot: tuple[int, int], t: float, delta: float, N: int) -> dict | None:
    """Cheapest r (and so split) with interpolated bounds, r over every integer with t/r in the grid."""
    r_lo, r_hi = max(1, math.ceil(t / tau_hi - 1e-9)), math.floor(t / tau_lo + 1e-9)
    if r_hi < r_lo:
        return None
    r = np.arange(r_lo, r_hi + 1, dtype=float)
    dT = r * b(t / r)
    R = rot[0] + (r - 1) * rot[1]
    ok = np.isfinite(dT) & (dT < delta)
    if not ok.any():
        return None
    from snapevo.synthesis import kn_law
    a_, b_ = kn_law()
    dS = np.where(ok, delta - dT, np.nan)
    Tq = np.where(ok, R * (a_ * np.log2(R / dS) + b_) / N, np.inf)
    i = int(np.argmin(Tq))
    edge = bool(i == 0 and r_lo > 1 and t / r_lo > 0.98 * tau_hi) or bool(i == len(r) - 1)
    return {"r": int(r[i]), "delta_T": float(dT[i]), "delta_S": float(dS[i]), "R": int(R[i]),
            "eps": float(dS[i] / R[i]), "Tq": float(Tq[i]), "at_grid_edge": edge}


_INTERP: dict = {}


def estimate(path_bounds, field: str = "layer", cs=C_GRID, times=TIMES, sizes=SIZES) -> dict:
    """T per qubit against c for every formula on the grid (interpolated bounds), and the best formula at each c."""
    recs = read_bounds(path_bounds)
    series: dict = {}
    for (hx, order, ordering, v, N, fld, tau), rec in recs.items():
        if fld == field and rec.get("source", "grid") == "grid" and N in sizes:
            series.setdefault((hx, order, ordering, v, N), []).append((tau, rec["total"]))
    merged: dict = {}                    # min over v at common taus (each v is a valid bound, Sec. IV.5)
    for (hx, order, ordering, v, N), s in series.items():
        d = merged.setdefault((hx, order, ordering, N), {})
        for tau, tot in s:
            d[tau] = min(d.get(tau, math.inf), tot)
    curves = []
    for (hx, order, ordering, N), d in sorted(merged.items()):
        if len(d) < 3:
            continue
        b, lo, hi = _interp_bound(list(d.items()))
        _INTERP[(hx, order, ordering, N)] = (b, lo, hi)
        rot = _rotations(hx, N, order, ordering)
        for t in times:
            pts = []
            for c in cs:
                opt = optimize(b, lo, hi, rot, t, delta_for_c(c, N), N)
                pts.append({"c": c, **opt} if opt else {"c": c})
            curves.append({"hx": hx, "order": order, "ordering": ordering, "N": N, "t": t, "points": pts,
                           "R_first": rot[0], "R_step": rot[1]})
    best = {}
    for cv in curves:
        for p in cv["points"]:
            if "Tq" not in p:
                continue
            k = (cv["hx"], cv["N"], cv["t"], p["c"])
            if k not in best or p["Tq"] < best[k]["Tq"]:
                best[k] = {"order": cv["order"], "ordering": cv["ordering"], **p}
    return {"curves": curves, "best": [{"hx": k[0], "N": k[1], "t": k[2], **v} for k, v in sorted(best.items())]}


def measured_baseline(summary_csv, hx: float, t: float, c: float) -> dict:
    """Measured Trotter + KN T per qubit at per-site c, N = 100, along the lower envelope (sweep_tables.py).

    "drawn" uses c_sampled (fallbacks drawn), as sweep_tables.drawn_c does for the baseline; "success"
    uses c_site (success branch). Log-log interpolation between envelope points (cli._interp); None
    outside the envelope's range.
    """
    import csv
    from snapevo.cli import _envelope, _interp
    tag = XXZ(0.25, hx).tag
    with open(summary_csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r["model"] == tag and float(r["t"]) == t
                and r["method"] == "baseline" and float(r["Tq"]) > 0]
    out = {}
    for name, col in (("drawn", "c_sampled"), ("success", "c_site")):
        env = _envelope([(float(r[col]), float(r["Tq"])) for r in rows if r.get(col)])
        out[name] = _interp(env, c)
        out[name + "_range"] = (env[0][0], env[-1][0]) if env else None
    out["N"] = sorted({int(r["N"]) for r in rows})
    return out


TABLE_T = (2.0, 4.0, 10.0)
TABLE_C = (1e-4, 1e-6, 1e-8)
V_USED = {2: (0, 1, 2), 4: (0, 1), 6: (0, 1)}
"""Slice enlargements v evaluated per order at the table points; the smallest bound is used (Sec. IV.5)."""


def _best_by_order(est: dict, hx: float, N: int, t: float, c: float) -> dict:
    out = {}
    for cv in est["curves"]:
        if (cv["hx"], cv["N"], cv["t"]) != (hx, N, t):
            continue
        p = next((p for p in cv["points"] if math.isclose(p["c"], c, rel_tol=1e-9)), None)
        if p is None:
            b, lo, hi = _INTERP[(cv["hx"], cv["order"], cv["ordering"], cv["N"])]
            p = {"c": c, **(optimize(b, lo, hi, (cv["R_first"], cv["R_step"]), t, delta_for_c(c, N), N) or {})}
        if "Tq" in p and (cv["order"] not in out or p["Tq"] < out[cv["order"]]["Tq"]):
            out[cv["order"]] = {"ordering": cv["ordering"], **p}
    return out


def verify_jobs(est: dict, Ns=SIZES, times=TABLE_T, cs=TABLE_C, fields=FIELDS, orders=(2, 4, 6),
                vs: dict | None = None) -> list:
    """Exact single-step bounds at r - 1, r, r + 1 around the interpolated optimum of each order."""
    vs = vs or V_USED
    jobs = set()
    for hx in fields:
        for N in Ns:
            for t in times:
                for c in cs:
                    for order, p in _best_by_order(est, hx, N, t, c).items():
                        if order not in orders:
                            continue
                        for r in (p["r"] - 1, p["r"], p["r"] + 1):
                            if r < 1:
                                continue
                            for v in vs[order]:
                                jobs.add((hx, order, p["ordering"], v, (N,), float(f"{t / r:.12g}"), "layer",
                                          "verify"))
    return sorted(jobs, key=lambda j: (j[1], j[3], -j[5]))


def exact_point(bounds: dict, hx: float, N: int, order: int, ordering: str, t: float, r: int, c: float,
                vs=(0, 1, 2)) -> dict | None:
    """Rigorous T per qubit at r from exact single-step bounds (smallest over the v evaluated)."""
    tau = float(f"{t / r:.12g}")
    vals = {v: bounds[k]["total"] for v in vs if (k := (hx, order, ordering, v, N, "layer", tau)) in bounds}
    if not vals:
        return None
    v_best = min(vals, key=vals.get)
    b = vals[v_best]
    delta = delta_for_c(c, N)
    R1, dR = _rotations(hx, N, order, ordering)
    R = R1 + (r - 1) * dR
    dT = r * b
    if dT >= delta:
        return {"r": r, "feasible": False, "delta_T": dT, "b_by_v": vals}
    return {"r": r, "feasible": True, "v": v_best, "b_step": b, "delta": delta, "delta_T": dT,
            "delta_S": delta - dT, "R": R, "eps": (delta - dT) / R, "Tq": t_per_qubit(R, delta - dT, N),
            "condition_C_worst_case_safe": R >= 1.2e4, "b_by_v": vals}


def table(est: dict, bounds: dict, summary_csv, Ns=SIZES, times=TABLE_T, cs=TABLE_C, fields=FIELDS) -> list[dict]:
    rows = []
    for hx in fields:
        for t in times:
            for c in cs:
                meas = measured_baseline(summary_csv, hx, t, c)
                for N in Ns:
                    per = {}
                    for order, p in _best_by_order(est, hx, N, t, c).items():
                        cands = [exact_point(bounds, hx, N, order, p["ordering"], t, r, c) for r in
                                 (p["r"] - 1, p["r"], p["r"] + 1) if r >= 1]
                        cands = [x for x in cands if x and x.get("feasible")]
                        if not cands:
                            per[order] = {"ordering": p["ordering"], "interp": p, "exact": None}
                            continue
                        ex = min(cands, key=lambda x: x["Tq"])
                        bi = _INTERP[(hx, order, p["ordering"], N)][0](t / ex["r"])
                        per[order] = {"ordering": p["ordering"], "interp": p, "exact": ex,
                                      "interp_rel_err": float(bi / ex["b_by_v"].get(0, ex["b_step"]) - 1.0)}
                    ok = {o: x for o, x in per.items() if x["exact"]}
                    best = min(ok, key=lambda o: ok[o]["exact"]["Tq"]) if ok else None
                    rows.append({"hx": hx, "t": t, "c": c, "N": N, "by_order": per, "best_order": best,
                                 "best": ok[best] if best else None, "measured": meas})
    return rows


def table_markdown(rows: list[dict]) -> str:
    def f(x, fmt="{:.0f}"):
        return "-" if x is None else fmt.format(x)
    lines = ["| $h_x$ | $t$ | $c$ | N | rigorous T/q, order 2 | order 4 | order 6 | best (order, ordering, r, v, eps) | "
             "measured T/q (N = 100), drawn | success | rigorous / measured (drawn) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bo = r["by_order"]
        cell = lambda o: f(bo[o]["exact"]["Tq"]) if o in bo and bo[o]["exact"] else "-"    # noqa: E731
        b = r["best"]
        bdesc = "-" if not b else (f"{f(b['exact']['Tq'])} ({r['best_order']}, {b['ordering']}, "
                                   f"r = {b['exact']['r']}, v = {b['exact']['v']}, eps = {b['exact']['eps']:.1e})")
        m = r["measured"]
        ratio = (b["exact"]["Tq"] / m["drawn"]) if b and m["drawn"] else None
        lines.append(f"| {r['hx']:g} | {r['t']:g} | {r['c']:.0e} | {r['N']} | {cell(2)} | {cell(4)} | {cell(6)} | "
                     f"{bdesc} | {f(m['drawn'])} | {f(m['success'])} | {f(ratio, '{:.1f}x')} |")
    return "\n".join(lines)


def plot(est: dict, summary_csv, path, N_lines=SIZES) -> None:
    """T per qubit against c: each order's best ordering at N = 50 (dashed) and 100 (solid), and the measured envelope."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from snapevo.cli import _envelope
    with open(summary_csv) as fh:
        rows = list(csv.DictReader(fh))
    fig, axes = plt.subplots(2, len(TIMES), figsize=(3.1 * len(TIMES), 6.4), sharey="row")
    colors = {2: "#4c78a8", 4: "#e45756", 6: "#54a24b"}
    for i, hx in enumerate(FIELDS):
        for j, t in enumerate(TIMES):
            ax = axes[i, j]
            for order in (2, 4, 6):
                for N, ls in zip(N_lines, ("--", "-")):
                    pts = {}
                    for cv in est["curves"]:
                        if (cv["hx"], cv["N"], cv["t"], cv["order"]) == (hx, N, t, order):
                            for p in cv["points"]:
                                if "Tq" in p and (p["c"] not in pts or p["Tq"] < pts[p["c"]]):
                                    pts[p["c"]] = p["Tq"]
                    if pts:
                        x = sorted(pts)
                        ax.plot(x, [pts[k] for k in x], ls=ls, color=colors[order], lw=1.3,
                                label=f"rigorous, order {order}, N = {N}" if (i, j) == (0, 0) else None)
            tag = XXZ(0.25, hx).tag
            mp = [(float(r["c_sampled"]), float(r["Tq"])) for r in rows if r["model"] == tag and float(r["t"]) == t
                  and r["method"] == "baseline" and float(r["Tq"]) > 0 and r.get("c_sampled")]
            env = _envelope(mp)
            if env:
                ax.plot(*zip(*env), color="k", marker="o", ms=3, lw=1.2,
                        label="measured Trotter + KN (N = 100, drawn)" if (i, j) == (0, 0) else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-11, 1e-2)
            ax.grid(True, which="major", color="#e4e3df", lw=0.6)
            ax.set_title(f"h_x = {hx:g}, t = {t:g}", fontsize=9)
            if i == 1:
                ax.set_xlabel("per-site infidelity c")
            if j == 0:
                ax.set_ylabel("T per qubit")
    fig.legend(loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    fig.savefig(str(path).rsplit(".", 1)[0] + ".pdf")


def _main(argv=None) -> None:
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser(prog="python -m snapevo.rigorous")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6])
    g.add_argument("--orderings", nargs="+", default=list(ORDERINGS))
    g.add_argument("--fields", type=float, nargs="+", default=list(FIELDS))
    g.add_argument("--v", type=int, nargs="+", default=[0])
    g.add_argument("--taus", type=float, nargs="*")
    g.add_argument("--field-mode", default="layer")
    vv = sub.add_parser("verify")
    vv.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6])
    vv.add_argument("--v6", type=int, nargs="+", default=list(V_USED[6]))
    for p in (g, vv):
        p.add_argument("--out", default=DEFAULT_OUT)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--budget", type=float, default=540.0)
    e = sub.add_parser("estimate")
    e.add_argument("--out", default=DEFAULT_OUT)
    tb = sub.add_parser("table")
    tb.add_argument("--out", default=DEFAULT_OUT)
    tb.add_argument("--summary", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "grid":
        jobs = []
        for hx in a.fields:
            for order in a.orders:
                for ordering in a.orderings:
                    for v in a.v:
                        for tau in (a.taus or tau_grid(order)):
                            jobs.append((hx, order, ordering, v, list(SIZES), tau, a.field_mode))
        jobs.sort(key=lambda j: (-j[5], j[1]))
        run_jobs(jobs, out / "bounds.jsonl", a.workers, a.budget)
    elif a.cmd == "estimate":
        res = estimate(out / "bounds.jsonl")
        (out / "estimate.json").write_text(json.dumps(res, indent=1))
        cols = ["hx", "N", "t", "order", "ordering", "c", "r", "delta_T", "delta_S", "R", "eps", "Tq", "at_grid_edge"]
        with open(out / "curves.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for cv in res["curves"]:
                for pt in cv["points"]:
                    if "Tq" in pt:
                        row = {**{k: cv[k] for k in ("hx", "N", "t", "order", "ordering")}, **pt}
                        fh.write(",".join(str(row[k]) for k in cols) + "\n")
        with open(out / "best.csv", "w") as fh:
            fh.write(",".join(cols) + "\n")
            for pt in res["best"]:
                fh.write(",".join(str(pt[k]) for k in cols) + "\n")
        print(f"estimate: {len(res['curves'])} curves, {len(res['best'])} best points -> {out}")
    elif a.cmd == "verify":
        est = estimate(out / "bounds.jsonl")
        vs = {**V_USED, 6: tuple(a.v6)}
        run_jobs(verify_jobs(est, orders=tuple(a.orders), vs=vs), out / "bounds.jsonl", a.workers, a.budget)
    elif a.cmd == "table":
        from snapevo import io
        summary = a.summary or (io.repo_root() / "runs" / "summary.csv")
        est = estimate(out / "bounds.jsonl")
        rows = table(est, read_bounds(out / "bounds.jsonl"), summary)
        (out / "table.json").write_text(json.dumps(rows, indent=1, default=float))
        md = table_markdown(rows)
        (out / "table.md").write_text(md + "\n")
        print(md)
        try:
            plot(est, summary, out / "rigorous_T_per_qubit.png")
        except ImportError as exc:
            print(f"no plot: {exc}")


if __name__ == "__main__":
    _main()
