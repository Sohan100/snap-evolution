"""Independent check of the corrected gadget file (``python -m snapevo verify-library``).

Shares no code with pygridsynth or Isaac's tcount_vetted. For every row of
rz_rus_gadgets.csv it rebuilds the dressed ancilla unitary

    V' = [[z, -w^n y*], [y, w^n z*]] / sqrt2^L,   w = e^{i pi/4},

with n = 0 on same-angle rows and n = 1 on odd-shift rows (the file does not
store n), simulates CNOT(s->a) V' CNOT(s->a) and an ancilla measurement, and checks

* success probability p, and that the success branch is Rz(-angle_rad)
  (the file reports the angle with the opposite sign of the literal circuit);
* failure branch diagonal, theta_f = pi - (failure angle), correction class;
* T count = least sqrt2-denominator exponent of the SO(3) matrix of V', a
  formula this module first confirms against a 0-1 BFS over all Clifford+T
  elements with T count <= 8;
* t_exact is the cheaper of the two dressings in its class (V', T V' T^dag);
* same-angle rows cost 2k - 2 and odd-shift rows 2k - 3 or 2k - 1 (k = lowest
  denominator exponent), for every gadget with p < 1;
* E_T_kn18 and E_T_rs50 recompute, and the dressed library at F = 18 equals
  the shipped rz_library_cat0_kn18.csv.

Results of the 2026-10-06 run are in docs/validation.md.
"""

from __future__ import annotations

import cmath
import csv
import math
from collections import Counter, deque

import numpy as np

from snapevo import io
from snapevo.library import LIB_DIR, Library, gadget_cost

W8 = cmath.exp(1j * math.pi / 4)
ZERO, ONE, I_ = (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0)


# ---------------------------------------------------------------- Z[w] arithmetic
# x = (x0, x1, x2, x3) = x0 + x1 w + x2 w^2 + x3 w^3; the CSV's (a, b, c, d) is a w^3 + b w^2 + c w + d.
def _add(x, y):
    return tuple(i + j for i, j in zip(x, y))


def _neg(x):
    return tuple(-i for i in x)


def _mul(x, y):
    out = [0, 0, 0, 0]
    for i in range(4):
        if x[i]:
            for j in range(4):
                k = i + j
                if k < 4:
                    out[k] += x[i] * y[j]
                else:
                    out[k - 4] -= x[i] * y[j]
    return tuple(out)


def _conj(x):
    return (x[0], -x[3], -x[2], -x[1])


def _wpow(m):
    m %= 8
    e = [0, 0, 0, 0]
    if m < 4:
        e[m] = 1
    else:
        e[m - 4] = -1
    return tuple(e)


def _c(x):
    return x[0] + x[1] * W8 + x[2] * W8 ** 2 + x[3] * W8 ** 3


def _mm(A, B):
    return [[_add(_mul(A[i][0], B[0][j]), _mul(A[i][1], B[1][j])) for j in range(2)] for i in range(2)]


def _dag(A):
    return [[_conj(A[0][0]), _conj(A[1][0])], [_conj(A[0][1]), _conj(A[1][1])]]


_PAULI = ([[ZERO, ONE], [ONE, ZERO]], [[ZERO, _neg(I_)], [I_, ZERO]], [[ONE, ZERO], [ZERO, _neg(ONE)]])


def _so3_lde(M, L):
    """(least exponent e, numerators) of the SO(3) matrix of U = M / sqrt2^L."""
    Md = _dag(M)
    N = []
    for si in _PAULI:
        row = []
        for sj in _PAULI:
            P = _mm(_mm(_mm(si, M), sj), Md)
            x = _add(P[0][0], P[1][1])           # real: x2 = 0, x3 = -x1, value x0 + x1 sqrt2
            assert x[2] == 0 and x[3] == -x[1]
            row.append([x[0], x[1]])
        N.append(row)
    e = 2 * L + 2
    while e > 0 and all(v[0] % 2 == 0 for r in N for v in r):
        for r in N:
            for v in r:
                v[0], v[1] = v[1], v[0] // 2
        e -= 1
    return e, N


def tcount(M, L) -> int:
    return _so3_lde(M, L)[0]


def _div_sqrt2(x):
    y = _mul(x, (0, 1, 0, -1))
    return None if any(v % 2 for v in y) else tuple(v // 2 for v in y)


def lowest_k(M, L) -> int:
    k, cur = L, [row[:] for row in M]
    while k > 0:
        nxt = [[_div_sqrt2(v) for v in row] for row in cur]
        if any(v is None for row in nxt for v in row):
            break
        cur, k = nxt, k - 1
    return k


def bfs_check(nmax: int = 8) -> tuple[int, int]:
    """(elements reached, mismatches between BFS T count and the SO(3) exponent)."""
    H = ([[ONE, ONE], [ONE, _neg(ONE)]], 1)
    S = ([[ONE, ZERO], [ZERO, I_]], 0)
    T = ([[ONE, ZERO], [ZERO, _wpow(1)]], 0)

    def key(M, L):
        e, N = _so3_lde(M, L)
        return e, tuple(tuple((v[0], v[1]) for v in r) for r in N)

    def reduce(M, L):
        k = lowest_k(M, L)
        while L > k:
            M = [[_div_sqrt2(v) for v in row] for row in M]
            L -= 1
        return M, L

    start = ([[ONE, ZERO], [ZERO, ONE]], 0)
    dist = {key(*start): 0}
    dq, mism = deque([(start, 0)]), 0
    while dq:
        (M, L), d = dq.popleft()
        k = key(M, L)
        if dist.get(k, 10 ** 9) < d:
            continue
        mism += k[0] != d
        for (G, Lg), c in ((H, 0), (S, 0), (T, 1)):
            if d + c > nmax:
                continue
            M2, L2 = reduce(_mm(G, M), L + Lg)
            k2 = key(M2, L2)
            if k2 not in dist or d + c < dist[k2]:
                dist[k2] = d + c
                (dq.appendleft if c == 0 else dq.append)(((M2, L2), d + c))
    return len(dist), mism


# ---------------------------------------------------------------- gadget rows
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)


def _sim(M, L):
    V = np.array([[_c(x) for x in row] for row in M]) / math.sqrt(2) ** L
    U = _CX @ np.kron(np.eye(2), V) @ _CX
    K0, K1 = U[np.ix_([0, 2], [0, 2])], U[np.ix_([1, 3], [0, 2])]
    p = float(np.real(np.trace(K0.conj().T @ K0)) / 2)
    diag_ok = abs(K0[0, 1]) + abs(K0[1, 0]) + abs(K1[0, 1]) + abs(K1[1, 0]) < 1e-10
    th = (cmath.phase(K0[1, 1]) - cmath.phase(K0[0, 0])) % (2 * math.pi)
    phf = None if p > 1 - 1e-14 else (cmath.phase(K1[1, 1]) - cmath.phase(K1[0, 0])) % (2 * math.pi)
    return p, th, phf, diag_ok


def _close(a, b, tol=1e-9):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) < tol


def check_rows() -> Counter:
    rows = list(csv.DictReader((io.third_party(LIB_DIR) / "rz_rus_gadgets.csv").open(newline="")))
    bad = Counter()
    Tg = [[ONE, ZERO], [ZERO, _wpow(1)]]
    Tdg = [[ONE, ZERO], [ZERO, _wpow(7)]]
    t_same = {}
    for idx, r in enumerate(rows):
        odd = r["source"].endswith("_oddshift")
        n = 1 if odd else 0
        z = tuple(int(r["z_" + c]) for c in "dcba")
        y = tuple(int(r["y_" + c]) for c in "dcba")
        L = int(r["L"])
        M = [[z, _neg(_mul(_wpow(n), _conj(y)))], [y, _mul(_wpow(n), _conj(z))]]
        p, th, phf, diag_ok = _sim(M, L)
        bad["not diagonal"] += not diag_ok
        bad["p"] += abs(p - float(r["p"])) > 1e-12
        bad["angle"] += not _close(-th, float(r["angle_rad"]))
        if phf is None:
            cls = "Clifford"
        else:
            j = phf / (math.pi / 4)
            cls = ("Clifford" if round(j) % 2 == 0 else "T") if abs(j - round(j)) < 1e-9 else "2+T"
            bad["theta_f"] += not _close(math.pi - phf, float(r["theta_f_pi"]) * math.pi)
        bad["correction class"] += cls != r["correction"]
        t = tcount(M, L)
        bad["t_exact vs SO(3) T count"] += t != int(r["t_exact"])
        bad["T parity vs det"] += t % 2 != n
        bad["not the cheaper dressing"] += t != min(t, tcount(_mm(_mm(Tg, M), Tdg), L))
        if p < 1 - 1e-14:
            k = lowest_k(M, L)
            if odd:
                bad["odd row not 2k-3 or 2k-1"] += t not in (2 * k - 3, 2 * k - 1)
                bad["odd row not same row +- 1"] += abs(t - t_same[idx - 1]) != 1
            else:
                bad["same row not 2k-2"] += t != 2 * k - 2
        if not odd:
            t_same[idx] = t
        for F, col in ((18.0, "E_T_kn18"), (50.0, "E_T_rs50")):
            bad[col] += abs(gadget_cost(float(r["t_exact"]), float(r["p"]), r["correction"], F)[0]
                            - float(r[col])) > 5e-6
    bad["rows checked"] = len(rows)
    return bad


def check_shipped_library() -> float:
    """Largest difference between the F = 18 lattice and the shipped cat0 file (cheapest within 1e-9)."""
    lib = Library(18.0)
    path = io.third_party(LIB_DIR) / "rz_library_cat0_kn18.csv"
    worst = 0.0
    shipped = [(float(r["angle_rad"]), float(r["expected_T"])) for r in csv.DictReader(path.open(newline=""))]
    for a, c in shipped:
        hit = lib.lookup(a)
        worst = max(worst, abs(hit[0] - c) if hit else math.inf)
    S = np.array([a for a, _ in shipped])
    C = np.array([c for _, c in shipped])
    for a, c in zip(lib.S, lib.C):
        lo, hi = np.searchsorted(S, a - 1e-9), np.searchsorted(S, a + 1e-9)
        if hi > lo:
            worst = max(worst, abs(C[lo:hi].min() - c))
        elif 1e-9 < a < 2 * math.pi - 1e-9:
            worst = math.inf
    return worst


def main() -> int:
    n, mism = bfs_check(8)
    print(f"SO(3) exponent vs BFS T count, all {n} Clifford+T elements with T <= 8: {mism} mismatches")
    bad = check_rows()
    rows = bad.pop("rows checked")
    fails = {k: v for k, v in bad.items() if v}
    print(f"{rows} gadget rows checked: " + ("all checks pass" if not fails else f"FAILURES {fails}"))
    worst = check_shipped_library()
    print(f"F = 18 lattice vs shipped rz_library_cat0_kn18.csv: max cost difference {worst:.1e}")
    return int(bool(fails) or mism > 0 or worst > 1e-6)


if __name__ == "__main__":
    raise SystemExit(main())
