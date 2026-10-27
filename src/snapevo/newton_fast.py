"""Cheaper trust-region Newton steps for the ladders and the polish: Hessian reuse, and truncated Newton-CG.

chain_snap.newton_minimize takes a fresh finite-difference Hessian (2 len(z) gradients) at every accepted
point. Both functions here take the same arguments, keep the same checkpoint format (a run can switch between
them and newton_minimize, in either direction) and the same stopping rule, and return the same record:

    converged    the predicted Newton decrease g.H^+.g/2 (H PSD-projected) and the last accepted step's
                 decrease are below ``tol`` (a number, or a function of the current c: the ladders use 1e-6 c),
                 and the bulk block of a FRESH full Hessian (the first n_bulk parameters) has no negative
                 curvature beyond 1e-8 of the largest eigenvalue; every "converged" is confirmed this way;
    noise floor  the trust radius has collapsed below NEWTON_DELTA_MIN with a fresh Hessian (steps no longer
                 lower c, as at the MPS truncation floor).

No iteration cap. The result is kept only if it lowers c.

newton_reuse_minimize  The last full Hessian is kept and reused for trust-region steps (the exact trust-region
    subproblem on the eigendecomposition, indefinite H allowed) with a fresh gradient each step. It is
    recomputed when a step's ratio actual/predicted decrease falls below ``rho_refresh``, after
    ``refresh_every`` accepted steps, and when the cheap convergence test (with the stale Hessian) passes, to
    confirm it.
newton_cg_minimize     Truncated Newton: Steihaug's conjugate-gradient solution of the trust-region subproblem,
    each Hessian-vector product a central difference of the adjoint gradient along the direction (2
    gradients at x +- h d/|d|, h = FD_STEP, submitted together through ``grad_many``, so a pool runs them at
    once). CG stops at the trust-region boundary, at negative curvature, when the residual is below
    min(0.5, sqrt|g|) |g| (the usual forcing term), or when an iteration adds less than 1e-6 of the model
    decrease already gained; it has no fixed iteration count. The full Hessian is computed only to confirm
    convergence (and its step is then used if the confirmation fails).

Records add "mode", "gradients" (gradient evaluations: objective calls, 2 per Hessian-vector product, 2 len(z)
per full Hessian), "hvps" and "cg_iterations" (newton-cg).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from snapevo.chain_snap import (NEWTON_DELTA_MAX, NEWTON_DELTA_MIN, DeadlineReached, _eig_stats, _Stop,
                                _Stopper, newton_minimize)

__all__ = ["newton_reuse_minimize", "newton_cg_minimize", "lbfgs_hessian_minimize", "minimize_newton", "NEWTON_MODES",
           "FD_STEP"]

NEWTON_MODES = ("full", "reuse", "cg")
FD_STEP = 1e-4


def _pred_newton(H, g) -> float:
    """g.H^+.g/2 with H PSD-projected (newton_minimize's predicted decrease)."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    b = V.T @ g
    keep = w > 1e-12 * max(float(np.abs(w).max()), 1e-300)
    return 0.5 * float(np.sum(b[keep] ** 2 / w[keep]))


def _tr_exact(H, g, delta):
    """Exact minimizer of g.p + p.H.p/2 over |p| <= delta (H symmetric, possibly indefinite), and its decrease."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    b = V.T @ g
    wmin = float(w[0])

    def p_of(lam):
        return -(V @ (b / (w + lam)))
    if wmin > 0:
        p = p_of(0.0)
        if np.linalg.norm(p) <= delta:
            return p, -float(g @ p + 0.5 * p @ (H @ p))
    lo = max(0.0, -wmin) * (1 + 1e-12) + 1e-300
    hi = lo + float(np.linalg.norm(g)) / delta + float(np.abs(w).max())
    if np.linalg.norm(p_of(lo)) <= delta and wmin <= 0:          # the hard case: add the lowest eigenvector
        p = p_of(lo)
        u = V[:, 0]
        a, bb, cc = u @ u, 2 * p @ u, p @ p - delta * delta
        tau = (-bb + math.sqrt(max(bb * bb - 4 * a * cc, 0.0))) / (2 * a)
        p = p + tau * u
        return p, -float(g @ p + 0.5 * p @ (H @ p))
    for _ in range(300):
        mid = 0.5 * (lo + hi) if hi / max(lo, 1e-300) < 4 else math.sqrt(lo * hi)
        if np.linalg.norm(p_of(mid)) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * hi:
            break
    p = p_of(hi)
    return p, -float(g @ p + 0.5 * p @ (H @ p))


class _Run:
    """Shared bookkeeping: checkpoint (newton_minimize's format), counters, the stop test and the record."""

    def __init__(self, z, objective, hessian, tol, n_bulk, log, checkpoint, deadline, stop, label, mode):
        self.say = log or (lambda _m: None)
        self.z0 = np.asarray(z, dtype=float).reshape(-1)
        self.nb = self.z0.size if n_bulk is None else int(n_bulk)
        self.rows = list(range(self.z0.size))
        self.tol_at = tol if callable(tol) else (lambda _c, v=float(tol): v)
        self.objective, self.hessian = objective, hessian
        self.checkpoint, self.label = checkpoint, label
        self.start, self.done = self.z0.copy(), 0
        self.info = {"tol": None if callable(tol) else float(tol), "hessians": 0}
        if checkpoint and os.path.exists(checkpoint):
            ck = json.loads(open(checkpoint).read())
            if np.asarray(ck["z_in"]).size != self.z0.size or not np.array_equal(np.asarray(ck["z_in"]), self.z0):
                raise ValueError(f"{checkpoint} is for another start point")
            self.start, self.done, self.info = np.asarray(ck["z"], dtype=float), int(ck["steps"]), ck["info"]
            self.info["tol"] = None if callable(tol) else float(tol)
            self.say(f"{label}: resumed {os.path.basename(checkpoint)} after {self.done} steps")
        self.info["mode"] = mode
        for k in ("gradients", "hvps", "cg_iterations"):
            self.info.setdefault(k, 0)
        self.t0 = time.time() - self.info.get("seconds", 0.0)
        self.stopping = _Stopper(deadline, stop)

    def f(self, x):
        if self.stopping():
            raise _Stop
        self.info["gradients"] += 1
        c, g = self.objective(x)
        return float(c), np.asarray(g, dtype=float)

    def H(self, x):
        if self.stopping():
            raise _Stop
        self.info["hessians"] += 1
        self.info["gradients"] += 2 * len(self.rows)
        return np.asarray(self.hessian(x, self.rows), dtype=float)

    def step_log(self, c, delta) -> None:
        if self.done % 5 == 0:
            self.say(f"{self.label}: step {self.done}, c = {c:.6e}, radius {delta:.2e}, {self.info['hessians']} "
                     f"Hessians, {self.info['hvps']} products, {self.info['gradients']} gradients, "
                     f"{time.time() - self.t0:.0f}s")

    def save(self, x, c) -> None:
        if self.checkpoint:
            tmp = f"{self.checkpoint}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"z_in": self.z0.tolist(), "z": np.asarray(x, dtype=float).tolist(), "steps": self.done,
                           "c": c, "info": {**self.info, "seconds": time.time() - self.t0}}, fh)
            os.replace(tmp, self.checkpoint)

    def converged_with(self, H, g, c, last_dec) -> bool:
        """newton_minimize's test with a fresh Hessian H at the current point."""
        st = _eig_stats(H, self.nb)
        self.info["pred"] = _pred_newton(H, g)
        tl = self.tol_at(c)
        self.info["tol_last"] = tl
        flat_ok = st["eig_min_bulk"] > -1e-8 * max(abs(st["eig_max"]), 1e-300)
        return flat_ok and self.info["pred"] < tl and (last_dec is None or last_dec < tl)

    def begin(self, x, c, H):
        """Record c and the curvature at the start (a run's first call, x = z0, as newton_minimize does)."""
        if "c_before" not in self.info:
            self.info["c_before"] = c
            self.info.update({f"{k}_before": v for k, v in _eig_stats(H, self.nb).items()})
            self.say(f"{self.label}: c = {c:.6e}, lowest eigenvalue {self.info['eig_min_before']:.2e} "
                     f"({self.info['n_neg_before']} negative; bulk {self.info['eig_min_bulk_before']:.2e}), "
                     f"tolerance {self.tol_at(c):.1e}")
            self.save(x, c)

    def finish(self, x, c, H_fresh):
        better = c < self.info["c_before"]
        out = x if better else self.z0
        H = H_fresh if better else self.H(self.z0)
        self.info.update({f"{k}_after": v for k, v in _eig_stats(H, self.nb).items()})
        self.info.update(c_after=c if better else self.info["c_before"], kept=bool(better), steps=self.done,
                         message=None, seconds=time.time() - self.t0)
        self.save(out, self.info["c_after"])
        self.say(f"{self.label}: {self.done} steps ({self.info['stop']}), c {self.info['c_before']:.6e} -> "
                 f"{self.info['c_after']:.6e}{'' if better else ' (kept the start)'}, {self.info['hessians']} full "
                 f"Hessians, {self.info['hvps']} Hessian-vector products, {self.info['gradients']} gradients, "
                 f"{self.info['seconds']:.0f}s")
        return out, self.info


def newton_reuse_minimize(z, objective, hessian, tol, n_bulk: int | None = None, log=None,
                          checkpoint: str | None = None, deadline: float | None = None, stop=None,
                          label: str = "newton-reuse", refresh_every: int = 10, rho_refresh: float = 0.25,
                          delta0: float = 1.0) -> tuple[np.ndarray, dict]:
    """newton_minimize with the last full Hessian reused across steps (module docstring)."""
    R = _Run(z, objective, hessian, tol, n_bulk, log, checkpoint, deadline, stop, label, "reuse")
    with R.stopping:
        x, c = R.start.copy(), None
        try:
            c, g = R.f(x)
            H = R.H(x)
            fresh, since = True, 0
            R.begin(x, c, H)
            delta, last_dec = float(R.info.get("delta", delta0)), None
            while True:
                tl = R.tol_at(c)
                if _pred_newton(H, g) < tl and (last_dec is None or last_dec < tl):
                    if not fresh:                                  # confirm with a fresh Hessian
                        H, fresh, since = R.H(x), True, 0
                        R.info["refresh_confirm"] = R.info.get("refresh_confirm", 0) + 1
                    if R.converged_with(H, g, c, last_dec):
                        R.info["stop"] = "converged"
                        break
                if delta < NEWTON_DELTA_MIN:
                    if fresh:
                        R.info["stop"] = "noise floor"
                        break
                    H, fresh, since, delta = R.H(x), True, 0, max(delta, NEWTON_DELTA_MIN * 4)
                    continue
                p, pred = _tr_exact(H, g, delta)
                if not pred > 0:
                    if fresh:
                        delta *= 0.25
                        continue
                    H, fresh, since = R.H(x), True, 0
                    continue
                ct, gt = R.f(x + p)
                rho = (c - ct) / pred
                if ct < c:
                    last_dec = c - ct
                    x, c, g = x + p, ct, gt
                    R.done += 1
                    since += 1
                    fresh = False
                    R.info["delta"] = delta
                    R.save(x, c)
                    R.step_log(c, delta)
                if rho < 0.25:
                    delta *= 0.25
                elif rho > 0.75 and np.linalg.norm(p) > 0.99 * delta:
                    delta = min(2.0 * delta, NEWTON_DELTA_MAX)
                if (rho < rho_refresh or since >= refresh_every) and not fresh:
                    why = "refresh_ratio" if rho < rho_refresh else "refresh_every"
                    R.info[why] = R.info.get(why, 0) + 1
                    H, fresh, since = R.H(x), True, 0
        except _Stop:
            R.save(x, c)
            raise DeadlineReached(f"{label} stopped ({R.stopping.why()}) after {R.done} steps; "
                                  f"checkpoint {checkpoint}") from None
        return R.finish(x, c, H if fresh else R.H(x))


def newton_cg_minimize(z, objective, hessian, tol, n_bulk: int | None = None, log=None,
                       checkpoint: str | None = None, deadline: float | None = None, stop=None,
                       label: str = "newton-cg", grad_many=None, fd_step: float = FD_STEP,
                       delta0: float = 1.0, refresh_hvps: float = 1.0, cg_rtol: float = 1e-2) -> tuple[np.ndarray, dict]:
    """Truncated trust-region Newton-CG (Steihaug-Toint) with central-difference Hessian-vector products.

    The CG runs on the true Hessian's action (each product: gradients at x +- h d/|d|, h = fd_step, through
    ``grad_many(points) -> [(c, g)]``, FarmCost.value_and_grad_many, so a pool evaluates the pair at once), and
    is preconditioned by the last full Hessian M (PSD-projected, eigenvalues floored at 1e-6 of the largest),
    with the trust region in M's norm: with M close to the current Hessian it converges in one or two
    iterations. CG stops at the boundary, at negative curvature, when its M^-1-residual has fallen by
    ``cg_rtol``, when an iteration adds less than 1e-6 of the model decrease gained, or after len(z) iterations
    (where it has spanned the space). M is recomputed (a full Hessian, 2 len(z) gradients) once the CG
    iterations since the last one reach ``refresh_hvps`` x len(z) (they have then cost as much as it does),
    and to confirm convergence (module docstring); its own exact step is then used if the confirmation fails.
    """
    R = _Run(z, objective, hessian, tol, n_bulk, log, checkpoint, deadline, stop, label, "cg")
    many = grad_many or (lambda pts: [objective(p) for p in pts])

    def hvp(x, d):
        nd = float(np.linalg.norm(d))
        if nd == 0.0:
            return np.zeros_like(d)
        if R.stopping():
            raise _Stop
        u = d / nd
        (_c1, g1), (_c2, g2) = many([x + fd_step * u, x - fd_step * u])
        R.info["hvps"] += 1
        R.info["gradients"] += 2
        return nd * (np.asarray(g1, dtype=float) - np.asarray(g2, dtype=float)) / (2 * fd_step)

    class Precond:
        def __init__(self, H):
            w, V = np.linalg.eigh(0.5 * (H + H.T))
            top = max(float(np.abs(w).max()), 1e-300)
            self.w, self.V = np.maximum(np.abs(w), 1e-6 * top), V

        def solve(self, r):
            return self.V @ ((self.V.T @ r) / self.w)

        def norm(self, p):
            q = self.V.T @ p
            return math.sqrt(float(np.sum(self.w * q * q)))

        def boundary(self, p, d, delta):
            qp, qd = self.V.T @ p, self.V.T @ d
            a, b, c_ = float(np.sum(self.w * qd * qd)), 2 * float(np.sum(self.w * qp * qd)), \
                float(np.sum(self.w * qp * qp)) - delta * delta
            return (-b + math.sqrt(max(b * b - 4 * a * c_, 0.0))) / (2 * a)

    def steihaug(x, g, delta, M):
        """(step, model decrease, interior?, CG iterations) in M's trust region of radius delta."""
        n = g.size
        p, Hp = np.zeros(n), np.zeros(n)
        r = g.copy()
        y = M.solve(r)
        d = -y
        ry0 = ry = float(r @ y)
        gained, it = 0.0, 0
        while True:
            Hd = hvp(x, d)
            it += 1
            R.info["cg_iterations"] += 1
            dHd = float(d @ Hd)
            if dHd <= 0:
                tau = M.boundary(p, d, delta)
                p, Hp = p + tau * d, Hp + tau * Hd
                return p, -float(g @ p + 0.5 * p @ Hp), False, it
            alpha = ry / dHd
            pn, Hpn = p + alpha * d, Hp + alpha * Hd
            if M.norm(pn) >= delta:
                tau = M.boundary(p, d, delta)
                p, Hp = p + tau * d, Hp + tau * Hd
                return p, -float(g @ p + 0.5 * p @ Hp), False, it
            rn = r + alpha * Hd
            yn = M.solve(rn)
            ryn = float(rn @ yn)
            model = -float(g @ pn + 0.5 * pn @ Hpn)
            step_gain = model - gained
            p, Hp, gained = pn, Hpn, model
            if ryn <= (cg_rtol ** 2) * ry0 or (gained > 0 and step_gain < 1e-6 * gained) or it >= n:
                return p, model, True, it
            beta = ryn / ry
            d = -yn + beta * d
            r, ry = rn, ryn

    with R.stopping:
        x, c = R.start.copy(), None
        try:
            c, g = R.f(x)
            H_full = R.H(x)                                      # the start's curvature and the first M
            R.begin(x, c, H_full)
            M = Precond(H_full)
            delta, last_dec = float(R.info.get("delta", delta0)), None
            its_since = 0
            while True:
                if H_full is not None:                           # a full Hessian here: its exact step is free
                    p, pred = _tr_exact(H_full, g, delta)
                    interior, its = bool(np.linalg.norm(p) < delta * (1 - 1e-9)), 0
                else:
                    p, pred, interior, its = steihaug(x, g, delta, M)
                    its_since += its
                tl = R.tol_at(c)
                cheap_ok = interior and pred < tl and (last_dec is None or last_dec < tl)
                if cheap_ok or delta < NEWTON_DELTA_MIN:
                    if H_full is None:
                        H_full = R.H(x)
                        M, its_since = Precond(H_full), 0
                    if R.converged_with(H_full, g, c, last_dec):
                        R.info["stop"] = "converged"
                        break
                    if delta < NEWTON_DELTA_MIN:
                        R.info["stop"] = "noise floor"
                        break
                    p, pred = _tr_exact(H_full, g, delta)
                if not pred > 0:
                    delta *= 0.25
                    continue
                ct, gt = R.f(x + p)
                rho = (c - ct) / pred
                pn = np.linalg.norm(p) if H_full is not None else M.norm(p)
                if ct < c:
                    last_dec = c - ct
                    x, c, g = x + p, ct, gt
                    R.done += 1
                    R.info["delta"] = delta
                    R.save(x, c)
                    R.step_log(c, delta)
                    H_full = None
                if rho < 0.25:
                    delta *= 0.25
                elif rho > 0.75 and pn > 0.99 * delta:
                    delta = min(2.0 * delta, NEWTON_DELTA_MAX)
                if its_since >= refresh_hvps * x.size and H_full is None:
                    H_full = R.H(x)
                    M, its_since = Precond(H_full), 0
                    R.info["refresh_hvps"] = R.info.get("refresh_hvps", 0) + 1
        except _Stop:
            R.save(x, c)
            raise DeadlineReached(f"{label} stopped ({R.stopping.why()}) after {R.done} steps; "
                                  f"checkpoint {checkpoint}") from None
        return R.finish(x, c, H_full if H_full is not None else R.H(x))


def lbfgs_hessian_minimize(z, objective, hessian, tol, n_bulk: int | None = None, log=None,
                           checkpoint: str | None = None, deadline: float | None = None, stop=None,
                           label: str = "lbfgs", memory: int = 10, check_gap: int = 50,
                           stall_window: int = 1000, stall_rel: float = 1e-5) -> tuple[np.ndarray, dict]:
    """L-BFGS steps with the Hessian kept for the checks: a comparison arm for the ladders, not the protocol.

    The steps are scipy's L-BFGS-B (``memory`` correction pairs, no bounds, ftol = gtol = 0, no iteration cap) on
    the exact gradient. The Hessian never makes a step. A fresh one is taken at the start, to confirm convergence
    with the Newton runs' test (_Run.converged_with: predicted Newton decrease and last decrease below ``tol``, no
    negative bulk curvature), and at the end, so the record has a Newton run's curvature fields (eig_min_after,
    n_neg_after, ...). The test runs when an iteration lowers c by less than ``tol``, at most once every
    ``check_gap`` iterations. Stops: "converged" (confirmed by the Hessian), "noise floor" (L-BFGS-B's line search
    finds no decrease), "stalled" (``stall_window`` iterations together lowered c by less than ``tol`` or
    ``stall_rel`` x c, whichever is larger: near a minimum L-BFGS gains ~1e-6 c per 1000 iterations for days while
    the Hessian test keeps failing, at t = 4 more than 13 hours at one depth; 1e-5 c moves no T count). Same
    checkpoint format as the Newton runs; a resumed run starts a fresh L-BFGS memory. Records add "checks".
    """
    from collections import OrderedDict, deque

    import scipy.optimize as sopt

    R = _Run(z, objective, hessian, tol, n_bulk, log, checkpoint, deadline, stop, label, "lbfgs")
    R.info.setdefault("checks", 0)
    evals: OrderedDict = OrderedDict()
    held = {"x": None, "H": None}

    def fg(x):
        key = np.asarray(x, dtype=float).tobytes()
        if key not in evals:
            evals[key] = R.f(np.array(x, dtype=float))
            while len(evals) > 8:
                evals.popitem(last=False)
        return evals[key]

    def H_at(x):
        if held["x"] is None or not np.array_equal(held["x"], x):
            held.update(x=np.array(x, dtype=float), H=R.H(x))
        return held["H"]

    with R.stopping:
        st = {"x": R.start.copy(), "c": None, "last_check": -check_gap}
        hist: deque = deque(maxlen=stall_window + 1)
        try:
            st["c"] = fg(st["x"])[0]
            if "c_before" not in R.info:
                R.begin(st["x"], st["c"], H_at(st["x"]))
            hist.append(st["c"])
            R.info["stop"] = None

            def cb(intermediate_result):
                x, c = np.array(intermediate_result.x, dtype=float), float(intermediate_result.fun)
                dec = st["c"] - c
                st.update(x=x, c=c)
                R.done += 1
                R.save(x, c)
                hist.append(c)
                if R.done % 25 == 0:
                    R.say(f"{label}: step {R.done}, c = {c:.6e}, {R.info['hessians']} Hessians, {R.info['checks']} "
                          f"checks, {R.info['gradients']} gradients, {time.time() - R.t0:.0f}s")
                tl = R.tol_at(c)
                if dec < tl and R.done - st["last_check"] >= check_gap:
                    st["last_check"] = R.done
                    R.info["checks"] += 1
                    if R.converged_with(H_at(x), fg(x)[1], c, dec):
                        R.info["stop"] = "converged"
                        raise StopIteration
                if len(hist) > stall_window and hist[0] - c < max(tl, stall_rel * c):
                    R.info["stop"] = "stalled"
                    raise StopIteration

            res = sopt.minimize(fg, st["x"], jac=True, method="L-BFGS-B", callback=cb,
                                options={"maxiter": 10 ** 9, "maxfun": 10 ** 9, "maxcor": int(memory), "ftol": 0.0,
                                         "gtol": 0.0, "maxls": 50})
            if R.info["stop"] is None:
                msg = str(res.message)
                R.info["stop"] = "noise floor" if ("ABNORMAL" in msg or "REDUCTION" in msg) else msg
            x, c = np.asarray(res.x, dtype=float), float(res.fun)
            if c > st["c"]:                                    # keep the last accepted point
                x, c = st["x"], st["c"]
            return R.finish(x, c, H_at(x))
        except _Stop:
            R.save(st["x"], st["c"])
            raise DeadlineReached(f"{label} stopped ({R.stopping.why()}) after {R.done} steps; "
                                  f"checkpoint {checkpoint}") from None


def minimize_newton(mode: str, z, objective, hessian, tol, grad_many=None, **kw) -> tuple[np.ndarray, dict]:
    """Dispatch by mode: "full" (chain_snap.newton_minimize, unchanged), "reuse", "cg"."""
    if mode == "full":
        return newton_minimize(z, objective, hessian, tol, **kw)
    if mode == "reuse":
        return newton_reuse_minimize(z, objective, hessian, tol, **kw)
    if mode == "cg":
        return newton_cg_minimize(z, objective, hessian, tol, grad_many=grad_many, **kw)
    raise ValueError(f"newton mode must be one of {NEWTON_MODES}, not {mode!r}")
