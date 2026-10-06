"""Kliuchnikov et al. (arXiv:2203.10064) non-mixed FALLBACK protocol as
explicit, verified circuits -- the baseline that was only a formula
(1.03 log2(1/eps) + 5.75) until 2026-09-29.

Protocol for target Rz(phi) = diag(e^{-i phi/2}, e^{i phi/2}) (paper Fig. 1,
Problem 3.8, Props 3.9-3.10):
  projective rotation: ancilla |0>, CNOT(s->a), V on a, CNOT(s->a), measure a
    (tcount_vetted.rus_circuit). Success, prob p = |u|^2: a diagonal U0 close
    to Rz(phi). Failure: a diagonal F.
  fallback, on failure only: B ~ Rz(phi) F^dag by Ross-Selinger at eps2.
Budget: D(U0, Rz(phi)) <= eps1 = (1-s) eps, D(B F, Rz(phi)) <= eps2 =
  s eps/(1-p); total <= p D0 + (1-p) D1 <= eps (convexity of the diamond norm).
Cost: expected T = T(V) + (1-p) T(B), single shot.
V is searched with pygridsynth's grid solver over FallbackRegion (a convex
subset of the paper's Fig. 6 annular sector: disk, half-plane at distance
sqrt(q), angular wedge; the discarded sliver is ~sqrt(q) delta^2/2 thick), in
both copies u ~ +-e^{-i phi/2}. For every candidate at the first feasible k and
one level beyond, all eight phases of the failure entry are tried and the
minimum T(V) kept; the success branch is unchanged by that phase. Both
parities of V are searched (n = 0 and the odd-T-count family n = 1).
Every T is counted in a circuit verified by tcount_vetted's simulator.
Run `--selftest`, then `--calibrate` (reproduces the paper's Table 1 row).
"""
import argparse
import cmath
import json
import math
import os
import sys

import mpmath
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pygridsynth as pg                                      # noqa: E402
from pygridsynth import UnitDisk, DOmegaUnitary              # noqa: E402
from pygridsynth.region import ConvexSet, Ellipse           # noqa: E402
from pygridsynth.ring import DOmega, DRootTwo, ZOmega        # noqa: E402
from pygridsynth.mymath import solve_quadratic               # noqa: E402
from pygridsynth.to_upright import to_upright_set_pair       # noqa: E402
from pygridsynth.tdgp import solve_TDGP                      # noqa: E402
from pygridsynth.diophantine import diophantine_dyadic, Result  # noqa: E402

import tcount_vetted as tv                                    # noqa: E402

mpmath.mp.dps = 50


class FallbackRegion(ConvexSet):
    """{u : |u| <= 1, Re(u conj(c)) >= sqrt(q), |Arg u - alpha| <= delta},
    c = e^{i alpha}. Convex: disk, one half-plane, two half-planes (wedge)."""

    def __init__(self, alpha, delta, q):
        self.a = mpmath.mpf(alpha)
        self.d = mpmath.mpf(delta)
        self.sq = mpmath.sqrt(mpmath.mpf(q))
        self.cx, self.cy = mpmath.cos(self.a), mpmath.sin(self.a)
        self.n1 = (-mpmath.sin(self.a - self.d), mpmath.cos(self.a - self.d))
        self.n2 = (mpmath.sin(self.a + self.d), -mpmath.cos(self.a + self.d))
        hr = (1 - self.sq) / 2 + mpmath.mpf("1e-40")
        ht = mpmath.sin(self.d) + (1 - mpmath.cos(self.d))
        ar, at = hr * mpmath.sqrt(2) * mpmath.mpf("1.05"), ht * mpmath.sqrt(2) * mpmath.mpf("1.05")
        Rm = mpmath.matrix([[self.cx, -self.cy], [self.cy, self.cx]])
        Dd = mpmath.matrix([[1 / ar ** 2, 0], [0, 1 / at ** 2]])
        centre = (1 + self.sq) / 2
        super().__init__(Ellipse(Rm @ Dd @ Rm.T,
                                 mpmath.matrix([centre * self.cx, centre * self.cy])))

    @staticmethod
    def _lin(n, u):
        return n[0] * u.real + n[1] * u.imag

    def inside(self, u):
        return (DRootTwo.fromDOmega(u.conj * u) <= 1
                and self.cx * u.real + self.cy * u.imag >= self.sq
                and self._lin(self.n1, u) >= 0 and self._lin(self.n2, u) >= 0)

    def intersect(self, u0, v):
        a = v.conj * v
        b = 2 * v.conj * u0
        c = u0.conj * u0 - 1
        t = solve_quadratic(a.real, b.real, c.real)
        if t is None:
            return None
        lo, hi = t
        for nx, ny, rhs in ((self.cx, self.cy, self.sq), (*self.n1, 0), (*self.n2, 0)):
            num = rhs - (nx * u0.real + ny * u0.imag)
            den = nx * v.real + ny * v.imag
            if den > 0:
                lo = max(lo, num / den)
            elif den < 0:
                hi = min(hi, num / den)
            elif num > 0:
                return None
        return (lo, hi) if lo <= hi else None


def rz(phi):
    return np.diag([cmath.exp(-1j * phi / 2), cmath.exp(1j * phi / 2)])


def diamond_rz(U, phi):
    """Diamond distance between the unitary channels of U and Rz(phi):
    2 sin(Delta/2), Delta = eigenphase spread of Rz(phi)^dag U (<= pi)."""
    th = np.sort(np.angle(np.linalg.eigvals(rz(phi).conj().T @ U)))
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * math.pi]]))
    spread = 2 * math.pi - gaps.max()
    return 2.0 if spread >= math.pi else 2 * math.sin(spread / 2)


def projective_search(phi, eps1, q, extra_k=1, kmax=80):
    """Min-T(V) V whose success branch is within diamond eps1 of Rz(phi),
    with |u|^2 >= q."""
    delta = math.asin(min(eps1 / 2, 1.0))
    disk = UnitDisk()
    # V = DOmegaUnitary(z, w, n) = [[z, -w* w^n], [w, z* w^n]] has det w^n and
    # success branch diag(z, z* w^n) ~ Rz(n pi/4 - 2 arg z), so the region for
    # z is centred at -phi/2 + n pi/8. Even n add no new candidates (the lattice
    # Z[omega] is invariant under omega); odd n is the odd-T-count family, a
    # genuinely new lattice copy that the paper's SU(2) formulation includes.
    prepared = []
    for n in (0, 1):
        for alpha in (-phi / 2 + n * math.pi / 8, -phi / 2 + n * math.pi / 8 + math.pi):
            reg = FallbackRegion(alpha, delta, q)
            prepared.append((n, (reg, disk, *to_upright_set_pair(reg, disk))))
    best, k0 = None, None
    for k in range(kmax):
        if k0 is not None and k > k0 + extra_k:
            break
        for n, sets in prepared:
            for z in solve_TDGP(*sets, k):
                xi = 1 - DRootTwo.fromDOmega(z.conj * z)
                if xi == 0:
                    w = DOmega(ZOmega(0, 0, 0, 0), 0)
                else:
                    if (z * z.conj).residue == 0:
                        continue
                    w = diophantine_dyadic(xi)
                    if isinstance(w, Result):
                        continue
                k0 = k if k0 is None else k0
                wj = w
                for _j in range(8):
                    V = DOmegaUnitary(z, wj, n).reduce_denomexp()
                    t = tv.synth_1q_exact(V)["t_count"]
                    if best is None or t < best[0]:
                        best = (t, V)
                    wj = wj.mul_by_omega()
    if best is None:
        raise RuntimeError(f"no projective rotation found for phi={phi} eps1={eps1} q={q}")
    return best[1]


def _gridsynth_word(phi, eps):
    g = str(pg.gridsynth_gates(mpmath.mpf(repr(float(phi % (2 * math.pi)))),
                               mpmath.mpf(repr(float(eps)))))
    return [c for c in g[::-1] if c in "HSTXW"]           # time order


def fallback_protocol(phi, eps, q=0.99, split=0.1):
    """Build, verify and price one fallback protocol. Raises if any branch
    misses its budget."""
    eps1 = (1 - split) * eps
    V = projective_search(phi, eps1, q)
    s = tv.synth_1q_exact(V)
    K0, K1 = tv.simulate_rus(tv.rus_circuit(s["word"]))
    p = float(np.real(np.trace(K0.conj().T @ K0)) / 2)
    qf = float(np.real(np.trace(K1.conj().T @ K1)) / 2)
    assert abs(p + qf - 1) < 1e-12 and p >= q - 1e-12, (p, qf, q)
    U0 = K0 / math.sqrt(p)
    d0 = diamond_rz(U0, phi)
    assert d0 <= eps1 * (1 + 1e-9), f"success branch {d0} > eps1 {eps1}"
    branches = [(p, U0)]
    t_b, d1, eps2 = 0, 0.0, 0.0
    if qf > 1e-14:
        F = K1 / math.sqrt(qf)
        eps2 = min(split * eps / qf, 1.0)
        phiB = tv._diag_angle(rz(phi) @ F.conj().T)     # target of the fallback
        wordB = _gridsynth_word(phiB, eps2)
        U1 = tv.word_matrix(wordB) @ F
        d1 = diamond_rz(U1, phi)
        assert d1 <= eps2 * (1 + 1e-6), f"fallback branch {d1} > eps2 {eps2}"
        t_b = tv.count_t(wordB)
        branches.append((qf, U1))
    bound = p * d0 + qf * d1
    assert bound <= eps * (1 + 1e-9), (bound, eps)
    return dict(phi=phi, eps=eps, q=q, split=split, k=s["k"], n=V.n, t_v=s["t_count"],
                t_b=t_b, p=p, expected_t=s["t_count"] + qf * t_b, d0=d0, d1=d1,
                eps2=eps2, bound=bound, branches=branches)


def best_protocol(phi, eps, q_options=(0.99, 0.95, 0.9), split=0.1):
    return min((fallback_protocol(phi, eps, q, split) for q in q_options),
               key=lambda r: r["expected_t"])


def selftest():
    fails = []
    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        if not cond: fails.append(name)
    # (a) the enclosing ellipse contains the region (else TDGP misses candidates)
    for alpha, eps1, q in ((0.3, 1e-2, 0.99), (2.0, 1e-5, 0.95), (-1.1, 3e-2, 0.9)):
        R = FallbackRegion(alpha, math.asin(eps1 / 2), q)
        E, p0 = R.ellipse.D, R.ellipse.p
        pts = []
        d = math.asin(eps1 / 2)
        for ang in (alpha - d, alpha + d):
            pts.append((math.cos(ang), math.sin(ang)))                        # outer corners
            r_in = math.sqrt(q) / math.cos(d)                                  # chord meets wedge edge
            pts.append((r_in * math.cos(ang), r_in * math.sin(ang)))           # inner corners
        pts.append((math.cos(alpha), math.sin(alpha)))                         # outer arc midpoint
        def qform(x, y):
            dx, dy = x - float(p0[0]), y - float(p0[1])
            return (dx * (float(E[0, 0]) * dx + float(E[0, 1]) * dy)
                    + dy * (float(E[1, 0]) * dx + float(E[1, 1]) * dy))
        inside_all = all(qform(x, y) <= 1 + 1e-9 for x, y in pts)
        check(f"ellipse encloses region (alpha={alpha}, eps1={eps1}, q={q})", inside_all)
    # (b) exact Clifford targets: p = 1, zero T
    for phi in (0.0, math.pi / 2, math.pi):
        r = fallback_protocol(phi, 1e-6)
        check(f"Clifford target phi={phi:.3f}: p=1, 0 T", r["p"] > 1 - 1e-12 and r["expected_t"] == 0,
              f"p={r['p']:.12f} T={r['expected_t']}")
    # (c) generic targets: every branch within budget, measured; T(V) even (det V = 1)
    for phi in (0.1, 1.3, 2.9, 5.5):
        for eps in (3.16e-2, 1e-3, 1e-5):
            r = fallback_protocol(phi, eps)
            ok = (r["d0"] <= (1 - r["split"]) * eps * (1 + 1e-9) and r["bound"] <= eps * (1 + 1e-9)
                  and r["t_v"] % 2 == r["n"] % 2 and r["p"] >= r["q"] - 1e-12)
            check(f"phi={phi} eps={eps:.0e}: branches within budget, T(V) parity = det parity", ok,
                  f"n={r['n']} d0={r['d0']:.2e} d1={r['d1']:.2e} bound={r['bound']:.2e} T_V={r['t_v']} T_B={r['t_b']} p={r['p']:.4f}")
    # (d) best_protocol never worse than the q=0.99 protocol
    r99, rb = fallback_protocol(0.7, 1e-4), best_protocol(0.7, 1e-4)
    check("best_protocol <= q=0.99 protocol", rb["expected_t"] <= r99["expected_t"] + 1e-12,
          f"{rb['expected_t']:.2f} vs {r99['expected_t']:.2f} (q={rb['q']})")
    return fails


def calibrate(n_angles=40, seed=3):
    rng = np.random.default_rng(seed)
    angles = [float(x) for x in rng.uniform(0, 2 * math.pi, n_angles)]
    rows, pts = [], []
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        et = [fallback_protocol(phi, eps, q=0.99)["expected_t"] for phi in angles]
        pm, px = 1.03 * math.log2(1 / eps) + 5.75, 1.05 * math.log2(1 / eps) + 11.83
        rows.append([eps, float(np.mean(et)), float(min(et)), float(max(et)), pm, px])
        pts += [(eps, e) for e in et]
        print(f"  eps={eps:.0e}  ours mean {np.mean(et):6.2f} ({min(et):.1f}..{max(et):.1f})"
              f"   paper mean {pm:6.2f}  max {px:6.2f}   gap {np.mean(et) - pm:+.2f}", flush=True)
    P = np.array(pts)
    slope, icpt = np.polyfit(np.log2(1 / P[:, 0]), P[:, 1], 1)
    gaps = [r[1] - r[4] for r in rows]
    ok = 0.93 <= slope <= 1.13 and all(-1.5 <= g <= 1.0 for g in gaps)
    print(f"  fit: {slope:.3f} log2(1/eps) + {icpt:.2f}   (paper 1.03, 5.75)   "
          f"{'PASS' if ok else 'FAIL'}")
    json.dump(dict(rows=rows, fit=[float(slope), float(icpt)], passed=bool(ok),
                   gap_T=float(np.mean(gaps))),
              open(os.path.join(HERE, "kliuchnikov_calibration.json"), "w"), indent=1)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("kliuchnikov_fallback self-test")
        f = selftest()
        print(f"\n{'ALL PASSED' if not f else f'{len(f)} FAILED: {f}'}")
        sys.exit(1 if f else 0)
    if a.calibrate:
        print("calibration vs arXiv:2203.10064 Table 1 (Clifford+T, Fallback, mean), q=0.99")
        sys.exit(0 if calibrate() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
