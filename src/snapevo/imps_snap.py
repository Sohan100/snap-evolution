"""SNAP on the infinite chain: chain_snap.compile_chain's protocol against the iMPS cost of the ladders.

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 (cell xxz+x) and h_x = 0 (cell xxz+stag).

Every stage is translation invariant (Sohan, 24 Oct 2026: "Everything should be TI systems, even the main sims after
training"): the trained circuit is a ladder of snapevo.imps_train (2-site cell, chi = 2 random iMPS inputs), the
compile snaps and re-optimizes it against that ladder's training cost, and the evaluation is the per-site cost on its
100 test inputs. There are no edges.

TIProblem gives compile_chain its objective, c_fn, hessian(z, rows, cols) and envs_at, through imps_train.Problem's
process pool, with the training inputs of the ladder it compiles. z is the bulk angles.

Curves. compile_chain needs each angle's cost with that angle moved and all others fixed. For angle i in sublayer l
and training input k, the per-cell overlap lambda_k(theta) is the leading eigenvalue of the mixed transfer matrix
with sublayer l's gate at that angle, between the forward environment of input k (sublayers < l applied to phi_k)
and the backward one (sublayers > l undone from psi_k), as imps_train.input_eval builds them; the cost is
c(theta) = 1 - mean_k |lambda_k(theta)|, period 2 pi in theta. Unlike the open chain's overlap, lambda_k is not a
trigonometric polynomial: it is analytic but has singularities near the real axis (eigenvalues meeting at complex
theta), so a Fourier series over the whole period converges slowly (64 samples miss by ~2e-4, 256 by ~6e-10 at
t = 0.5). The compile only needs the cost where it is small, so a TICurve is
  - exact (piecewise Chebyshev interpolants of each lambda_k, CHEB_NODES nodes per piece, a piece halved until its
    last coefficients are below CURVE_TOL of the budget, down to MIN_WIDTH) on windows: the one around the current
    angle, of half-width
    1.5 sqrt(2 (cap - c0) / c'') from the curvature c'' (lambda at theta0 +- H_CURV), and one around any other point
    of the coarse grid (COARSE samples over the period) whose cost is below the cap;
  - outside them, at least the cap (the coarse samples, linearly interpolated, floored at the cap), cap =
    CAP_FACTOR x the budget.
A snap threshold can exceed the budget (compile_chain admits a candidate when eps0 + rfrac (c - eps0) is under the
ceiling, and the response fraction rfrac can be small), so admissible(c_max, S) first extends the curve when c_max
reaches its cap: exact windows over every part of the period whose coarse samples are below EXTEND_MARGIN x c_max,
the rest floored at that level. Then it is c_at(S) <= c_max (chain_snap._admissible_mask defers to it).

Pricing follows Gibbs and Cincio (arXiv:2506.02105, Sec. III.4) in what is counted: T gates per qubit of the
translation-invariant circuit in Clifford+T, and the cost of the synthesized circuit recomputed on the infinite
chain. realize.realize prices the snapped angles with the library and synthesizes the rest with KN, checking the
realized training cost (c_fn) against the budget; evaluate_ti then gives the per-site cost of the realized
(success-branch) circuit on the 100 test inputs and its standard error, with the T counts at the compile's fallback
cost F (evaluate_ti's docstring: why the infinite chain has no fallback draw).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
import time
from pathlib import Path

import numpy as np

from snapevo import imps_train as it
from snapevo.cells import cell_for
from snapevo.circuits import Layered

__all__ = ["TICurve", "TIProblem", "evaluate_ti", "ti_point", "curve_samples"]

COARSE = 32
CHEB_NODES = 32
MIN_WIDTH = 1e-6
CURVE_TOL = 1e-4
CAP_FACTOR = 3.0
CHUNK_MIN = 16
EXTEND_MARGIN = 1.5
H_CURV = 1e-3


# --------------------------------------------------------------------------- lambda samples (in the pool workers)

def curve_samples(cell, layers, phi, psi, want: dict, chi_max: int = it.CHI_MAX, cutoff: float = it.CUTOFF) -> dict:
    """{(layer, slot): lambda at each theta} for ``want`` {layer: {slot: thetas}}, every other angle as in layers."""
    from snapevo import imps as im
    D = len(layers)
    gates = [cell.gate(a) for _, a in layers]
    lo, hi = min(want), max(want)
    back = [None] * D
    cur = it._fresh(psi, chi_max, cutoff)
    back[D - 1] = cur.copy()
    for l in range(D - 1, lo, -1):
        cur.apply_layer(gates[l].conj().T, layers[l][0])
        back[l - 1] = cur.copy()
    fwd = it._fresh(phi, chi_max, cutoff)
    out = {}
    for l in range(hi + 1):
        if l in want:
            for s, thetas in want[l].items():
                lam = np.empty(len(thetas), dtype=complex)
                v0r = v0l = None
                for g, th in enumerate(thetas):
                    a = np.array(layers[l][1], dtype=float)
                    a[s] = th
                    e = im.GateTransfer(fwd, back[l], cell.gate(a), layers[l][0]).eig(v0r=v0r, v0l=v0l)
                    lam[g], v0r, v0l = e["lambda"], e["r"], e["l"]
                out[(l, s)] = lam
            back[l] = None
        if l < hi:
            fwd.apply_layer(gates[l], layers[l][0])
    return out


def _curve_task(args):
    kind, k, x, want = args[:4]
    phi, psi = it._get(kind, k)
    cell = it._W["cell"]
    s = it._W["spec"]
    lay = Layered.brickwork(cell, x).layers
    out = curve_samples(cell, lay, phi, psi, want, s["chi_max"], s["cutoff"])
    return (k, out) if len(args) == 4 else (k, out, args[4])


# --------------------------------------------------------------------------- the curves

def _cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    k = np.arange(n)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (k + 0.5) / n)


def _cheb_coeffs(values: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients (n, K) of the interpolant through values (n, K) at _cheb_nodes (first kind)."""
    n = values.shape[0]
    k = np.arange(n)
    T = np.cos(np.pi * np.outer(k, k + 0.5) / n)               # T_j(x_k), rows j
    c = (2.0 / n) * (T @ values)
    c[0] *= 0.5
    return c


def _cost(lam: np.ndarray) -> np.ndarray:
    """1 - mean_k |lambda_k| along the last axis, without the cancellation (as ChainCurve with N = 2)."""
    F = np.maximum(np.abs(lam) ** 2, 1e-300)
    return -np.mean(np.expm1(np.log(F) / 2.0), axis=-1)


class TICurve:
    """The cost of the circuit as a function of one angle on the infinite chain (module docstring, Curves)."""

    N = 2

    def __init__(self, index: int, theta0: float, c0: float, cap: float, coarse_t: np.ndarray, coarse_c: np.ndarray,
                 problem=None, z=None, curv: float | None = None):
        self.index, self.theta0, self.c0, self.cap = int(index), float(theta0), float(c0), float(cap)
        self.curv = curv
        d = (np.asarray(coarse_t) - self.theta0 + np.pi) % (2.0 * np.pi) - np.pi
        o = np.argsort(d)
        self.coarse_d, self.coarse_c = d[o], np.asarray(coarse_c)[o]
        self.windows: list[tuple[float, float, np.ndarray]] = []     # (a, b, Chebyshev coefficients (n, K))
        self.residual = 0.0
        self._problem, self._z = problem, z                          # for extend(); not pickled

    def __getstate__(self):
        st = dict(self.__dict__)
        st["_problem"] = None
        return st

    def _reduce(self, theta) -> np.ndarray:
        d = np.asarray(theta, dtype=float) - self.theta0
        return self.theta0 + (d + np.pi) % (2.0 * np.pi) - np.pi

    def c_at(self, theta):
        scalar = np.ndim(theta) == 0
        th = np.atleast_1d(self._reduce(theta))
        out = np.maximum(np.interp(th - self.theta0, self.coarse_d, self.coarse_c, period=2.0 * np.pi), self.cap)
        for a, b, coef in self.windows:
            m = (th >= a) & (th <= b)
            if m.any():
                x = (2.0 * th[m] - a - b) / (b - a)
                lam = np.polynomial.chebyshev.chebval(x, coef)              # (K, n_m)
                out[m] = _cost(lam.T)
        return float(out[0]) if scalar else out

    def curvature(self) -> float:
        """d^2 c / d theta^2 at theta0, floored at 1e-12: the exact lambdas at theta0 +- H_CURV (central difference),
        not the interpolant, whose error (the tolerance) a 1e-4 difference would amplify by 1e8."""
        if self.curv is not None:
            return max(float(self.curv), 1e-12)
        d = 1e-4
        c = self.c_at(np.array([self.theta0 - d, self.theta0, self.theta0 + d]))
        return max(float((c[0] + c[2] - 2 * c[1]) / d ** 2), 1e-12)

    def extend(self, level: float) -> None:
        """Make the curve exact wherever the coarse samples leave room for a cost up to ``level`` (every coarse point
        below EXTEND_MARGIN x level, with its neighbours' intervals), and floor the rest at level."""
        if level <= self.cap:
            return
        if self._problem is None:
            raise ValueError(f"cost level {level:.3e} above the curve's cap {self.cap:.3e} and no problem to extend it")
        new = self._uncovered(level)
        if new:
            self._add(*self._problem._refine(self._z, {self.index: new})[self.index])
        self.cap = float(level)

    def _add(self, pieces, res: float) -> None:
        self.windows = sorted(self.windows + pieces, key=lambda w: w[0])
        self.residual = max(self.residual, res)

    @staticmethod
    def prefetch(curves: dict, c_max: dict) -> None:
        """What admissible(c_max[i], .) (c_max[i] a number) or extend(inf) (None) would do to each curve, as one
        batched refinement per (problem, z) instead of one per curve: the same intervals, so the same curves, but
        the pool works on every curve's pieces at once (one curve at a time left most workers idle)."""
        groups: dict = {}
        for i, cv in curves.items():
            cm = c_max[i]
            level = math.inf if cm is None else (EXTEND_MARGIN * cm if cm >= cv.cap else None)
            if level is None or level <= cv.cap or cv._problem is None:
                continue
            groups.setdefault((id(cv._problem), cv._z.tobytes()), []).append((cv, level))
        for items in groups.values():
            need = {cv.index: cv._uncovered(level) for cv, level in items}
            need = {i: w for i, w in need.items() if w}
            got = items[0][0]._problem._refine(items[0][0]._z, need) if need else {}
            for cv, level in items:
                if cv.index in got:
                    cv._add(*got[cv.index])
                cv.cap = float(level)

    def _uncovered(self, level: float) -> list:
        """The intervals extend(level) refines: around every coarse point below EXTEND_MARGIN x level, minus the
        windows the curve has."""
        h = 2.0 * np.pi / len(self.coarse_d)
        low = np.flatnonzero(self.coarse_c <= EXTEND_MARGIN * level)
        spans = sorted((max(self.coarse_d[g] - h, -np.pi), min(self.coarse_d[g] + h, np.pi)) for g in low)
        merged: list[list[float]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        have = sorted((a - self.theta0, b - self.theta0) for a, b, _ in self.windows)
        new = []
        for a, b in merged:                                  # the parts not covered yet
            cur = a
            for wa, wb in have:
                if wb <= cur or wa >= b:
                    continue
                if wa > cur:
                    new.append((cur, wa))
                cur = max(cur, wb)
            if cur < b:
                new.append((cur, b))
        return [(self.theta0 + a, self.theta0 + b) for a, b in new if b - a > MIN_WIDTH]

    def admissible(self, c_max: float, S: np.ndarray) -> np.ndarray:
        if c_max >= self.cap:
            self.extend(EXTEND_MARGIN * c_max)
        return np.asarray(self.c_at(S)) <= c_max * (1.0 + 1e-9) + 1e-15


# --------------------------------------------------------------------------- the problem

class TIProblem:
    """compile_chain's costs on the infinite chain, for one ladder depth (module docstring)."""

    def __init__(self, model, t: float, D: int, n_train: int = it.N_TRAIN, n_test: int = it.N_TEST,
                 workers: int = 8, budget: float | None = None, log=print):
        self.model, self.t, self.D = model, float(t), int(D)
        self.cell = cell_for(model)
        self.cap = min(0.5, CAP_FACTOR * budget) if budget else 0.5
        self.tol = CURVE_TOL * (budget if budget else 1e-2)
        self.workers = int(workers)
        self.P = it.Problem(model, t, workers=workers, n_train=n_train, n_test=n_test, log=log)
        self.P.build_references(test=n_test > 0)
        self._H = (None, None)
        self.stats = {"n_grad": 0, "n_value": 0, "n_hessian": 0, "n_curves": 0, "curve_seconds": 0.0,
                      "max_residual": 0.0}

    def close(self) -> None:
        self.P.close()

    def __enter__(self) -> TIProblem:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def objective(self, z):
        self.stats["n_grad"] += 1
        return self.P.cost_grad(np.asarray(z, dtype=float))

    def c_fn(self, z) -> float:
        self.stats["n_value"] += 1
        return float(self.P.value(np.asarray(z, dtype=float), "train")["c"])

    def c_test(self, z) -> tuple[float, float]:
        """Per-site cost on the test inputs and its standard error over them."""
        v = self.P.value(np.asarray(z, dtype=float), "test")
        c = 1.0 - np.asarray(v["abs"])
        return float(c.mean()), float(c.std(ddof=1) / math.sqrt(c.size)) if c.size > 1 else 0.0

    def hessian(self, z, rows, cols=None) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        key = z.tobytes()
        if self._H[0] != key:
            self._H = (key, self.P.hessian(z))
            self.stats["n_hessian"] += 1
        H = self._H[1]
        rows = list(rows)
        return H[np.ix_(rows, list(cols) if cols is not None else rows)]

    def _lambdas(self, z, want: dict) -> dict:
        """{(layer, slot): (len(thetas), K) lambdas} over the training inputs for want {layer: {slot: thetas}}.

        One task per (input, angle, chunk of thetas), about four per worker: a task per (input,
        layer) left most of a compute node's workers idle (4 inputs x D layers tasks) and its slowest layer set the
        time. A chunk pays for its own environments (D layer applications) and restarts the eigensolver's warm
        start, so chunks keep at least CHUNK_MIN thetas."""
        K = self.P.n_train
        items = [(l, sl, np.asarray(th, dtype=float)) for l in want for sl, th in want[l].items()]
        total = sum(len(th) for _, _, th in items)
        size = max(CHUNK_MIN, math.ceil(total * K / (4 * self.workers)))
        tasks = [("train", k, z, {l: {sl: th[j:j + size]}}, (l, sl, j))
                 for l, sl, th in items for j in range(0, len(th), size) for k in range(K)]
        got: dict = {(l, sl): np.empty((len(th), K), dtype=complex) for l, sl, th in items}
        for k, out, (l, sl, j) in self.P.pool.map(_curve_task, tasks):
            lam = out[(l, sl)]
            got[(l, sl)][j:j + len(lam), k] = lam
        return got

    def curves(self, z, indices) -> dict:
        """{index: TICurve} for these angle indices at z (module docstring, Curves)."""
        z = np.asarray(z, dtype=float)
        na = self.cell.n_angles
        idx = sorted({int(i) for i in indices})
        t0 = time.time()
        coarse = {i: z[i] + 2.0 * np.pi * np.arange(COARSE) / COARSE for i in idx}
        probe = {i: np.concatenate([coarse[i], [z[i] - H_CURV, z[i] + H_CURV]]) for i in idx}
        lam = self._lambdas(z, self._want(probe))
        curves, win = {}, {}
        for i in idx:
            L = lam[(i // na, i % na)]
            c = _cost(L)
            c0, cm, cp = c[0], c[COARSE], c[COARSE + 1]
            curv = max((cm + cp - 2.0 * c0) / H_CURV ** 2, 1e-12)
            cv = TICurve(i, z[i], c0, self.cap, coarse[i], c[:COARSE], curv=curv)
            w = min(max(1.5 * math.sqrt(2.0 * max(self.cap - c0, 0.0) / curv), 4.0 * H_CURV), math.pi)
            win[i] = [(z[i] - w, z[i] + w)]
            h = 2.0 * math.pi / COARSE
            for g in range(1, COARSE):                       # other low points of the coarse grid
                d = 2.0 * math.pi * g / COARSE
                d = d - 2.0 * math.pi if d > math.pi else d
                if c[g] < self.cap and abs(d) > w:
                    win[i].append((z[i] + d - h, z[i] + d + h))
            curves[i] = cv
        got = self._refine(z, win)
        for i in idx:
            curves[i].windows, curves[i].residual = got[i]
            curves[i]._problem, curves[i]._z = self, z
        self.stats["n_curves"] += len(curves)
        self.stats["curve_seconds"] += time.time() - t0
        return curves

    def _refine(self, z, intervals: dict) -> dict:
        """{i: (pieces [(a, b, coef)], residual)}: piecewise Chebyshev interpolants of the lambdas of angle i over
        its intervals, each piece halved until its last coefficients are below the tolerance (or MIN_WIDTH)."""
        na = self.cell.n_angles
        n = CHEB_NODES
        pending = {int(i): list(ws) for i, ws in intervals.items()}
        done: dict = {i: [] for i in pending}
        res = {i: 0.0 for i in pending}
        while pending:
            thetas = {i: np.concatenate([_cheb_nodes(a, b, n) for a, b in ws]) for i, ws in pending.items()}
            lam = self._lambdas(z, self._want(thetas))
            again: dict = {}
            for i, ws in pending.items():
                L = lam[(i // na, i % na)]
                scale = max(float(np.max(np.abs(L))), 1e-300)
                for j, (a, b) in enumerate(ws):
                    coef = _cheb_coeffs(L[j * n:(j + 1) * n])
                    tail = float(np.max(np.abs(coef[-4:]))) / scale
                    if tail <= self.tol or b - a <= MIN_WIDTH:
                        done[i].append((a, b, coef))
                        res[i] = max(res[i], tail)
                    else:                                    # not converged: the two halves, piecewise
                        m = 0.5 * (a + b)
                        again.setdefault(i, []).extend([(a, m), (m, b)])
            pending = again
        for i in done:
            self.stats["max_residual"] = max(self.stats["max_residual"], res[i])
            self.stats["max_pieces"] = max(self.stats.get("max_pieces", 0), len(done[i]))
        return {i: (sorted(done[i], key=lambda w: w[0]), res[i]) for i in done}

    def _want(self, thetas: dict) -> dict:
        na = self.cell.n_angles
        want: dict = {}
        for i, th in thetas.items():
            want.setdefault(i // na, {})[i % na] = np.asarray(th, dtype=float)
        return want

    def envs_at(self, z, layers=None, sweep: bool = False):
        if sweep:
            raise NotImplementedError("the coordinate re-optimizer has no infinite-chain environments; use "
                                      "reopt = 'newton' (the protocol)")
        return _TIEnv(self, np.asarray(z, dtype=float).copy())


class _TIEnv:
    def __init__(self, problem: TIProblem, z: np.ndarray):
        self.problem, self.z = problem, z

    def curves(self, indices):
        return self.problem.curves(self.z, indices)

    def curve(self, index: int):
        return self.curves([index])[int(index)]

    def release(self) -> None:
        pass


# --------------------------------------------------------------------------- fallbacks on the infinite chain

def fallback_layers(cell_name: str, layers, rec_for, lib_for=None) -> list:
    """Per sublayer (parity, success gate, [(kind, q, [defect gates])]) for fallback_terms.

    rec_for(layer, slot) is the KN record of a slot (with branch unitaries) or None; its fallback replaces that Rz by
    U1 (fallbacks.py's slot decomposition, the conjugate for the staggered slot's Rz(-d)). lib_for(layer, slot) is
    (q, delta) for a library gadget or None: its correction is a KN rotation the library does not carry, so the
    defect is the worst of six error rotations of diamond distance delta (+-x, +-y, +-z) on that Rz.
    """
    from snapevo import fallbacks as fb
    slot_of = {"a": 0, "b": 1, "c": 2, "e": 3, "f": 4, "d+": 3, "d-": 3}
    out = []
    for l, (par, ang) in enumerate(layers):
        r = fb._layer_rots(cell_name, np.asarray(ang, dtype=float), lambda s, l=l: rec_for(l, s))
        inst = []
        for name in fb._ORDER[cell_name]:
            rot = r[name]
            if rot.fb is not None and rot.q > 0:
                inst.append(("kn", float(rot.q), [fb._gate(cell_name, r, {name: rot.fb})]))
            elif lib_for is not None:
                lib = lib_for(l, slot_of[name])
                if lib is not None and lib[0] > 0 and lib[1] > 0:
                    q, delta = lib
                    phi = 2.0 * math.asin(min(0.5 * delta, 1.0))
                    errs = [np.cos(0.5 * phi) * np.eye(2) - 1j * sg * np.sin(0.5 * phi) * P
                            for P in (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1]))
                            for sg in (1.0, -1.0)]
                    inst.append(("lib", float(q), [fb._gate(cell_name, r, {name: E @ rot.ok}) for E in errs]))
        out.append((int(par), fb._gate(cell_name, r, {}), inst))
    return out


def fallback_terms(flayers, phi, psi, chi_max: int = it.CHI_MAX, cutoff: float = it.CUTOFF) -> dict:
    """log|lambda| per cell on the success branch and the expected change of it from independent fallbacks.

    Each instance falls back on its own with probability q. One fallback at one bond is a local defect: it multiplies
    the overlap of the infinite chain by rho = (l^dag E' r) / (l^dag E r) = sum U' G / sum U G, with l, r the fixed
    points of that sublayer's transfer matrix and G = GateTransfer.env(l, r) (the gradient's tensor). For dilute
    independent defects the per-cell log-amplitude of a random draw is log|lambda| + sum_instances q log|rho|; pairs of
    defects within the correlation length change it at second order in the (small) defects.
    """
    from snapevo import imps as im
    D = len(flayers)
    gates = [U for _, U, _ in flayers]
    out = {"log_abs": 0.0, "kn": 0.0, "lib": 0.0, "n_kn": 0.0, "n_lib": 0.0, "worst_rho": 1.0}
    fwd = it._fresh(phi, chi_max, cutoff)
    if not any(inst for _, _, inst in flayers):                 # nothing can fall back: the overlap alone
        for l in range(D - 1):
            fwd.apply_layer(gates[l], flayers[l][0])
        e = im.GateTransfer(fwd, it._fresh(psi, chi_max, cutoff), gates[D - 1], flayers[D - 1][0]).eig()
        out["log_abs"] = math.log(abs(e["lambda"]))
        return out
    # backward environments back[l] (psi with sublayers > l undone), kept only at the end of each segment of
    # ceil(sqrt(D)) sublayers and remade one segment at a time: memory ~ 2 sqrt(D) iMPS instead of D (deep formulas)
    step = max(1, math.ceil(math.sqrt(D)))
    ends = sorted({D - 1} | {l for l in range(D) if (l + 1) % step == 0})
    ckpt = {}
    cur = it._fresh(psi, chi_max, cutoff)
    for l in range(D - 1, -1, -1):
        if l in ends:
            ckpt[l] = cur.copy()
        if l > 0:
            cur.apply_layer(gates[l].conj().T, flayers[l][0])
    del cur
    lo = 0
    for l1 in ends:
        seg = {l1: ckpt.pop(l1)}
        cur = seg[l1].copy()
        for l in range(l1, lo, -1):
            cur.apply_layer(gates[l].conj().T, flayers[l][0])
            seg[l - 1] = cur.copy()
        del cur
        for l in range(lo, l1 + 1):
            par, U, inst = flayers[l]
            T = im.GateTransfer(fwd, seg.pop(l), U, par)
            e = T.eig()
            if inst:
                G = T.env(e["l"], e["r"])
                den = complex(np.sum(U.reshape(2, 2, 2, 2) * G))
                for kind, q, variants in inst:
                    rho = min(abs(complex(np.sum(V.reshape(2, 2, 2, 2) * G)) / den) for V in variants)
                    out[kind] += q * math.log(max(rho, 1e-300))
                    out["n_" + kind] += q
                    out["worst_rho"] = min(out["worst_rho"], rho)
            if l == D - 1:
                out["log_abs"] = math.log(abs(e["lambda"]))
            else:
                fwd.apply_layer(U, par)
        lo = l1 + 1
    return out


def _fb_task(args):
    kind, k, flayers = args
    phi, psi = it._get(kind, k)
    s = it._W["spec"]
    return k, fallback_terms(flayers, phi, psi, s["chi_max"], s["cutoff"])


def real_fallback_layers(real: dict, cell, F: str) -> list:
    """fallback_layers of a realized circuit (real_*.json): KN stragglers (all of the uniform arm) and library gadgets."""
    from snapevo.synthesis import synthesize
    x = np.asarray(real["angles_compiled"], dtype=float)
    kn = {int(i): (r if "U0" in r else synthesize(float(x[int(i)]), r["eps"], unitaries=True))
          for i, r in real["kn"].items()}
    lat = {int(i): r for i, r in real["lattice"].items()}
    delta = float(real["per_F"][F]["delta_fb"])
    na = cell.n_angles

    def lib_for(l, s):
        row = lat.get(l * na + s)
        return None if row is None else (float(row["q"][F]), delta)

    return fallback_layers(cell.name, Layered.brickwork(cell, real["angles_real"]).layers,
                           lambda l, s: kn.get(l * na + s), lib_for)


def expected_with_fallbacks(problem: TIProblem, flayers, kind: str = "test") -> dict:
    """Per-site cost with the expected fallbacks over an input set (test: the ladder's test inputs)."""
    n = problem.P.n_test if kind == "test" else problem.P.n_train
    res = dict(problem.P.pool.map(_fb_task, [(kind, k, flayers) for k in range(n)]))
    la = np.array([res[k]["log_abs"] for k in range(n)])
    kn = np.array([res[k]["kn"] for k in range(n)])
    lib = np.array([res[k]["lib"] for k in range(n)])
    c_s = -np.expm1(la)
    c_f = -np.expm1(la + kn + lib)
    se = float(c_f.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {"c_success_tm": float(c_s.mean()), "c_fallbacks": float(c_f.mean()), "c_fallbacks_se": se,
            "c_kn_only": float((-np.expm1(la + kn)).mean()),
            "fallbacks_per_cell": float(res[0]["n_kn"] + res[0]["n_lib"]),
            "worst_rho": float(min(res[k]["worst_rho"] for k in range(n)))}


# --------------------------------------------------------------------------- realize and evaluate

def evaluate_ti(real: dict, problem: TIProblem, F: float | None = None, log=print) -> dict:
    """Per-site cost of the realized circuit on the test inputs, with its fallbacks and on the success branch, and its
    T counts.

    A fallback drawn at random on one bond breaks translation invariance, so the infinite chain takes the expectation
    instead: every synthesized rotation falls back on its own with its probability, each fallback a local defect
    (fallback_terms; checked against explicit draws on a ring, tests/test_imps_snap.py). This is the reported error,
    c_fallbacks; c is the success branch. The T counts are realize's at fallback cost F, the compile's own (its settings.F, default 22); the
    circuits with no library angle (uniform, product formulas) cost the same at every F. "c_plus_ledger" adds the
    fallback ledger per qubit to c, a crude first-order figure kept for reference, not a bound.
    """
    from snapevo.evaluate import _order
    x = np.asarray(real["angles_real"], dtype=float)
    c, se = problem.c_test(x)
    labels = sorted(real["per_F"], key=_order)
    choice = f"{F:g}" if F is not None and f"{F:g}" in real["per_F"] else ("22" if "22" in real["per_F"] else labels[0])
    pf = real["per_F"][choice]
    rot = float(real["rotations_per_qubit"])
    fbx = expected_with_fallbacks(problem, real_fallback_layers(real, problem.cell, choice))
    out = {"metric": "per site on the 100 random chi = 2 iMPS test inputs (imps_train): c_fallbacks with every "
                     "synthesized rotation falling back independently (fallback_terms), c on the success branch",
           "c_fallbacks": fbx["c_fallbacks"], "c_fallbacks_se": fbx["c_fallbacks_se"], "c_kn_only": fbx["c_kn_only"],
           "fallbacks_per_cell": fbx["fallbacks_per_cell"], "worst_rho": fbx["worst_rho"],
           "c": c, "c_se": se, "F": choice, "Tq": pf["Tq"], "T_edges_per_qubit": 0.0, "rotations_per_qubit": rot,
           "T_per_rotation": pf["Tq"] / rot, "T_per_qubit_per_time": pf["Tq"] / problem.t,
           "ledger_per_qubit": pf["ledger_per_qubit"], "c_plus_ledger": c + pf["ledger_per_qubit"],
           "per_F": {lab: {"Tq": v["Tq"], "ledger_per_qubit": v["ledger_per_qubit"]} for lab, v in real["per_F"].items()},
           "C_train_real": real["C_train_real"],
           "over_budget": bool(real["C_train_real"] > float(real.get("budget", math.inf)) * (1.0 + 1e-9))}
    if out["over_budget"]:
        log(f"evaluate: the realized training cost {real['C_train_real']:.3e} is over the budget {real['budget']:.3e}")
    log(f"evaluate (infinite chain): c = {fbx['c_fallbacks']:.3e} with fallbacks ({fbx['fallbacks_per_cell']:.3f} per "
        f"cell), {c:.3e} +- {se:.1e} on the success branch, F = {choice}, T/q {pf['Tq']:.1f}, "
        f"T/rot {out['T_per_rotation']:.2f}")
    return out


def ladder_sub(n_train: int, optimizer: str = "newton") -> str:
    """The ladder folder: imps_train[_n<N>][_lbfgs] (imps_train.train_imps_ladder's names)."""
    if optimizer not in ("newton", "lbfgs"):
        raise ValueError(f"optimizer must be newton or lbfgs, not {optimizer!r}")
    return ("imps_train" if n_train == it.N_TRAIN else f"imps_train_n{n_train}") + \
        ("_lbfgs" if optimizer == "lbfgs" else "")


@contextmanager
def _point_lock(path: Path):
    """Non-blocking exclusive lock on one (budget, depth) point: yields True if this process holds it. Any number of
    processes (login chains, compute jobs) can then work through the same ladder's budgets, each compiling the next
    point no one else holds; without it two compiles of one point collided on its checkpoint (25 Oct 2026)."""
    import fcntl
    fh = open(path, "a")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def ti_point(model, t: float, budget: float, runs, n_train: int = it.N_TRAIN, D: int | None = None,
             margins=(3.0, 30.0), workers: int = 8, settings_kw: dict | None = None,
             steps=("compile", "realize", "evaluate"), fresh: bool = False, deadline: float | None = None,
             all_depths: bool = False, hessian_chunk: int | None = None, optimizer: str = "newton",
             log=print) -> list[dict]:
    """Compile, realize and evaluate one budget on the infinite chain, at one ladder depth per floor margin.

    Reads runs/<tag>/t<T>/imps_train[_n<N>][_lbfgs]/ladder.json (``optimizer``: the ladder's training optimizer; the
    compile's re-optimizer is settings_kw's either way) and writes snap_/real_/eval_b<budget>_D<D>.json in
    runs/<tag>/t<T>/imps_snap[_n<N>][_lbfgs]/. Depths as the ring arm (chain_snap._ring_pick_depth): each margin's shallowest
    depth with max(C_train, C_val) <= budget / margin, a floor above 2 budget / 3 skipped; all_depths takes every
    depth whose floor is at most budget / 1.5. The compile is compile_chain on TIProblem with ChainSettings updated by
    settings_kw, which must name the re-optimizer (the protocol: reopt = "newton"); a killed compile resumes from its
    checkpoint. Existing records are reused unless ``fresh``.
    """
    from snapevo import io
    from snapevo.chain_snap import ChainSettings, Library, _ring_pick_depth, _same_reopt, compile_chain
    from snapevo.realize import realize

    cell = cell_for(model)
    sub = ladder_sub(n_train, optimizer)
    d = Path(runs) / model.tag / f"t{float(t):g}"
    lad = io.read_json(d / sub / "ladder.json")
    if lad.get("cell") != cell.name:
        raise ValueError(f"the ladder was trained with {lad.get('cell')}, not {cell.name}")
    recs = lad["results"]
    out_dir = d / sub.replace("imps_train", "imps_snap")
    if D:
        depths = [int(D)]
    elif all_depths:
        depths = sorted(int(r["D"]) for r in recs if max(r["C_train"], r.get("C_val") or 0.0) * 1.5 <= budget)
    else:
        depths = sorted({_ring_pick_depth(recs, budget, float(m)) for m in margins})
    results = []
    for D_ in depths:
        stem = f"{io.budget_tag(budget)}_D{D_}"
        p_snap, p_real, p_eval = (out_dir / f"{k}_{stem}.json" for k in ("snap", "real", "eval"))
        if p_eval.exists() and not fresh and "evaluate" in steps:
            log(f"ti point: {p_eval.name} exists (fresh=True to redo)")
            results.append(io.read_json(p_eval))
            continue
        rec = next(r for r in recs if int(r["D"]) == D_)
        floor = max(rec["C_train"], rec.get("C_val") or 0.0)
        if floor * 1.5 > budget:
            log(f"ti point: skip, the D = {D_} floor {floor:.2e} leaves less than a third of the budget "
                f"{budget:.1e} for snapping")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with _point_lock(out_dir / f"{stem}.busy") as mine:
            if not mine:                                   # another process compiles this point: take the next
                log(f"ti point: {stem} is being compiled by another process; skipping it")
                continue
            if p_eval.exists() and not fresh and "evaluate" in steps:     # finished while we waited for the lock
                results.append(io.read_json(p_eval))
                continue
            compiled = io.read_json(p_snap) if p_snap.exists() and not fresh else None
            _same_reopt(compiled, settings_kw, p_snap)
            real = io.read_json(p_real) if p_real.exists() and not fresh else None
            with TIProblem(model, t, D_, n_train=n_train, n_test=int(lad["val_set"][1]), workers=workers,
                           budget=budget, log=log) as P:
                if compiled is None and "compile" in steps:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    ck = p_snap.with_suffix(".ckpt.pkl")
                    if fresh:
                        for f in (ck, Path(str(ck) + ".curves")):
                            if f.exists():
                                f.unlink()
                    settings = ChainSettings(budget=budget, **{"checkpoint": str(ck), **(settings_kw or {})})
                    z0 = np.asarray(rec["params"], dtype=float).reshape(-1)
                    log(f"ti compile: {model.describe()}, cell {cell.name}, t = {t:g}, D = {D_}, budget {budget:.1e}, "
                        f"{n_train} training inputs, ladder floor {rec['C_train']:.3e} (test {rec.get('C_val')})")
                    compiled = compile_chain(z0, cell, D_, settings, P.objective, P.c_fn, P.hessian, P.envs_at,
                                             Library(settings.F), log=log, deadline=deadline, hessian_chunk=hessian_chunk)
                    compiled.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D_, "N": "inf",
                                     "bc": "infinite", "unit_cell": 2, "arm": "ti", "ladder": f"{sub}/ladder.json",
                                     "train_set": lad["train_set"], "ladder_C_train": rec["C_train"],
                                     "ladder_C_val": rec.get("C_val"), "ladder_start": rec.get("start"),
                                     "curve_cap": P.cap, "curve_tol": P.tol,
                                     "curve_max_residual": P.stats["max_residual"],
                                     "problem_stats": dict(P.stats)})
                    io.write_json(p_snap, compiled)
                    for f in (ck, Path(str(ck) + ".curves")):      # the point is done: its checkpoint files go
                        if f.exists():
                            f.unlink()
                    log(f"ti compile: wrote {p_snap}")
                if real is None and compiled is not None and "realize" in steps:
                    real = realize(compiled, model, cell, t, c_fn=P.c_fn, log=log)
                    real.update({"model": model.to_dict(), "cell": cell.name, "D": D_, "bc": "infinite"})
                    io.write_json(p_real, real)
                if real is not None and "evaluate" in steps:
                    F = (compiled or {}).get("settings", {}).get("F") if compiled else None
                    ev = evaluate_ti(real, P, F=F, log=log)
                    ev.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D_, "budget": budget,
                               "n_free": real["n_free"], "n_t": real["n_t"], "n_rus": real["n_rus"],
                               "n_kn": real["n_kn"], "arm": "ti", "train_set": lad["train_set"]})
                    io.write_json(p_eval, ev)
                    results.append(ev)
    return results
