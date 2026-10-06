"""Hessian-ranked SNAP: compile a trained brickwork onto the exact-angle library.

The protocol of Sec. III of the paper, ported from tcount_tfim.escalation_hessian
(Sep 2026) without catalysis. One compile, at a training budget B on C:

1. Re-optimize the free angles; their C is eps0.
2. Zero-T stage. Snap any free angle whose best cost-0 lattice point fits by
   plug-in C alone (environment.py), lowest plug-in first, re-optimizing after
   each, until none fits.
3. Hessian stage. With H the PSD-projected Hessian of C over the free angles,
   the response of the others to moving angle p is x_f = -H_ff^+ H_fp and the
   surviving fraction of the plug-in rise is r_p = v^T H v / H_pp, v = (x_f, 1).
   A candidate's predicted C after compensation is eps0 + r_p (C_plug - eps0).
4. Cost-first search. Candidates are the lowest-plug-in lattice point of each
   cost level for each free angle, kept if the prediction fits. Levels go up in
   cost, ties by (prediction, max|x|, |x|, plug-in). Up to top_k trials per
   level and max_round_trials per round: pin the angle, start the others at the
   predicted response, re-optimize, accept if C is under the ceiling. Accepting
   returns to step 2.
5. Two guards. A snap must leave enough budget for every still-free angle to be
   synthesized at reserve_t expected T (capped at reserve_frac of the budget),
   and must not eat more than share_cap times its fair share of the remaining
   room. The share cap relaxes 1, 4, 16, inf (coarse) when a round accepts nothing.
6. End game. Every prefix of the accepted sequence is a valid compile once the
   angles still free are given the budget left over, either a library angle
   that fits (library stragglers) or KN synthesis at a precision set by
   curvature. The cheapest prefix wins. Straggler costs here use the calibrated
   KN law; realize.py replaces them with real KN circuits and measures.

The accept test is always a true re-optimized C, so the heuristics can only cost
time, never correctness. Checkpoints after every accepted snap let a long run
resume where it stopped.
"""

from __future__ import annotations

import math
import os
import pickle
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.optimize as sopt

from snapevo.cells import Cell
from snapevo.environment import Environments
from snapevo.library import TWO_PI, Library
from snapevo.synthesis import kn_law

__all__ = ["SnapSettings", "compile_snap"]

_REOPT = {"jac": True, "method": "L-BFGS-B", "options": {"maxiter": 250, "ftol": 1e-16, "gtol": 1e-14}}
_GRID = 1 << 15


@dataclass
class SnapSettings:
    """Everything that sets a compile's trajectory. Stored in the result file."""

    budget: float
    F: float = 22.0
    top_k: int = 6
    reserve_t: float = 18.0
    reserve_frac: float = 0.1
    max_round_trials: int = 48
    share_cap: float = 3.0
    relax_ladder: str = "coarse"
    library_stragglers: bool = True
    skip_failed: bool = False
    prefilter: bool = True
    checkpoint: str | None = None
    checkpoint_every: float = 120.0


@dataclass
class _State:
    x: np.ndarray
    snapped: np.ndarray
    costs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_zero: int = 0
    n_hess: int = 0
    n_trials: int = 0
    n_rounds: int = 0
    relax: float = 1.0
    failed: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    history: list = field(default_factory=list)
    cur_failed_relax: list = field(default_factory=list)
    cur_relax: float = 0.0
    cur_trials_done: int = 0
    loop_done: bool = False
    eg_teq: dict = field(default_factory=dict)


def _unwrap(theta: float, target: float) -> float:
    """target on the same 2 pi branch as theta."""
    return theta - ((theta - target + math.pi) % TWO_PI - math.pi)


def _psd(H: np.ndarray) -> np.ndarray:
    H = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H)
    return (V * np.clip(w, 0.0, None)) @ V.T


def _admissible_mask(curve, c_max: float, S: np.ndarray) -> np.ndarray | None:
    """Superset of the lattice angles whose plug-in C can be <= c_max (rigorous grid bound).

    C <= E forces every |O_k|^{2/N} >= 1 - n E, hence g = mean |O_k|^2 >= 1 - N n E.
    g is a trigonometric polynomial, so g on a grid plus derivative and curvature
    bounds covers every cell. Returns None when the superset is too large to help.
    """
    if not np.isfinite(c_max):
        return None
    c = np.asarray(curve.coeffs)
    m = 0.5 * np.asarray(curve.modes, dtype=float)
    n = c.shape[1]
    thr = 1.0 - curve.N * n * max(c_max, 0.0) * (1.0 + 1e-9) - 1e-12
    if thr <= 0.5:
        return None
    h = 2.0 * math.pi / _GRID
    th = np.arange(_GRID + 1) * h
    basis = np.exp(1j * th[:, None] * m[None, :])
    O = basis @ c
    dO = basis @ (1j * m[:, None] * c)
    g = np.mean(np.abs(O) ** 2, axis=1)
    dg = np.mean(2.0 * np.real(np.conj(O) * dO), axis=1)
    a0 = np.sum(np.abs(c), axis=0)
    a1 = np.sum(np.abs(m)[:, None] * np.abs(c), axis=0)
    a2 = np.sum((m ** 2)[:, None] * np.abs(c), axis=0)
    B2 = float(np.mean(2.0 * a1 ** 2 + 2.0 * a0 * a2))
    keep = g + np.abs(dg) * h / 2.0 + B2 * h * h / 8.0 >= thr
    if keep.mean() > 0.3:
        return None
    cell_idx = np.clip(np.rint(S / h).astype(np.int64), 0, _GRID)
    return keep[cell_idx]


def compile_snap(x_star: np.ndarray, cell: Cell, settings: SnapSettings, objective,
                 c_fn: Callable[[np.ndarray], float], hessian: Callable[[np.ndarray], np.ndarray],
                 inputs: np.ndarray, targets: np.ndarray, N: int, library: Library,
                 log: Callable[[str], None] | None = None) -> dict:
    """Run the protocol and return the compiled circuit as a JSON-ready dict.

    Args:
        x_star: Trained flat angles.
        objective: ``x -> (C, grad)`` for the re-optimization (jaxcost.make_cost).
        c_fn: ``x -> C`` on the training set.
        hessian: ``x -> (n, n)`` Hessian of C (jaxcost.make_hessian).
        inputs, targets: The training states, for the environments.
        library: The lattice to snap onto.
    """
    say = log or (lambda _m: None)
    n = len(x_star)
    na = cell.n_angles
    weights = np.array([cell.multiplicity[i % na] for i in range(n)], dtype=float)
    A_GEN, B_GEN = kn_law()
    S, C = library.S, library.C
    S0 = S[C == 0.0]
    say(f"snap: {len(S)} library angles (F = {library.F:g}), {len(S0)} free, "
        f"{len(np.unique(C))} cost levels, budget {settings.budget:.3e}")

    st = _State(x=np.asarray(x_star, dtype=float).copy(), snapped=np.zeros(n, dtype=bool),
                costs=np.zeros(n), lower=np.full(n, -np.inf), upper=np.full(n, np.inf))
    eps = settings.budget
    resumed = None
    if settings.checkpoint and os.path.exists(settings.checkpoint):
        with open(settings.checkpoint, "rb") as fh:
            resumed = pickle.load(fh)
        st = resumed["state"]
        say(f"snap: resumed {os.path.basename(settings.checkpoint)} ({st.n_zero} zero-T, "
            f"{st.n_hess} Hessian, {int((~st.snapped).sum())} free, {resumed['elapsed']:.0f}s spent)")
    last_ckpt = [time.time()]

    def checkpoint(elapsed: float, floor: float) -> None:
        if not settings.checkpoint:
            return
        tmp = settings.checkpoint + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"state": st, "elapsed": elapsed, "floor": floor}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, settings.checkpoint)
        last_ckpt[0] = time.time()

    def checkpoint_if_due() -> None:
        if time.time() - last_ckpt[0] >= settings.checkpoint_every:
            checkpoint(time.time() - t0, floor)

    def reopt() -> float:
        if st.snapped.all():
            return c_fn(st.x)
        res = sopt.minimize(objective, st.x, bounds=list(zip(st.lower, st.upper)), **_REOPT)
        st.x[:] = np.asarray(res.x)
        return float(res.fun)

    def pin(i: int, angle: float) -> None:
        st.x[i] = angle
        st.lower[i] = st.upper[i] = angle
        st.snapped[i] = True

    def unpin(i: int, saved: np.ndarray) -> None:
        st.x[:] = saved
        st.lower[i], st.upper[i] = -np.inf, np.inf
        st.snapped[i] = False

    def free_indices() -> list[int]:
        return [int(i) for i in np.flatnonzero(~st.snapped)]

    def envs_at(x):
        return Environments(x, cell, inputs, targets, N)

    e_res2 = (2.0 ** (-(settings.reserve_t - B_GEN) / A_GEN)) ** 2 if settings.reserve_t > 0 else 0.0

    def share_for(eps0: float, n_free: int, relax: float = 1.0) -> float:
        if settings.share_cap <= 0 or not np.isfinite(relax):
            return float("inf")
        return relax * settings.share_cap * max(eps - eps0, 0.0) / max(n_free, 1)

    def reserve_for(curv: dict, exclude: int) -> float:
        if e_res2 == 0.0:
            return 0.0
        need = sum(cv * e_res2 for i, cv in curv.items() if i != exclude)
        return min(need, settings.reserve_frac * eps * max(len(curv) - 1, 0) / n)

    # ------------------------------------------------------------- zero-T stage
    def zero_t_stage() -> int:
        total = 0
        while True:
            free = free_indices()
            if not free:
                return total
            envs = envs_at(st.x)
            curves = {i: envs.curve(i) for i in free}
            curv = {i: c.curvature() for i, c in curves.items()}
            best = None
            for i in free:
                ec = np.asarray(curves[i].c_at(S0))
                j = int(np.argmin(ec))
                if ec[j] <= eps - reserve_for(curv, i) and (best is None or ec[j] < best[0]):
                    best = (float(ec[j]), i, float(S0[j]))
            if best is None:
                return total
            ec, i, ang = best
            saved = st.x.copy()
            c_before = c_fn(st.x)
            pin(i, _unwrap(st.x[i], ang))
            c_after = reopt()
            if c_after > eps - reserve_for(curv, i) or c_after - c_before > share_for(c_before, len(free)):
                unpin(i, saved)
                return total
            st.costs[i] = 0.0
            st.n_zero += 1
            st.trace.append({"i": i, "stage": "zero", "cost": 0.0, "c_pre": ec, "c_post": c_after})
            st.history.append((st.x.copy(), st.snapped.copy(), st.costs.copy(), c_after))
            total += 1
            checkpoint_if_due()

    # --------------------------------------------------------- Hessian stage
    order = np.argsort(C, kind="stable")
    C_sorted = C[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(C_sorted)) + 1])
    seg = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(C))))
    level_cost = C_sorted[starts]
    n_lev = len(starts)
    pos_in_order = np.empty(len(C), dtype=np.int64)
    pos_in_order[order] = np.arange(len(C))

    def best_per_level(ec: np.ndarray) -> np.ndarray:
        lex = np.lexsort((ec[order], seg))
        return order[lex[starts]]

    def best_per_level_subset(cand: np.ndarray, ec_c: np.ndarray):
        J = order[starts].copy()
        E = np.full(n_lev, np.inf)
        if len(cand):
            pos = pos_in_order[cand]
            sg = seg[pos]
            lex = np.lexsort((pos, ec_c, sg))
            first = np.ones(len(lex), dtype=bool)
            first[1:] = sg[lex][1:] != sg[lex][:-1]
            pick = lex[first]
            J[sg[pick]] = cand[pick]
            E[sg[pick]] = ec_c[pick]
        return J, E

    def hessian_round(relax: float = 1.0) -> bool:
        free = free_indices()
        M = len(free)
        if M == 0:
            return False
        st.n_rounds += 1
        eps0 = reopt()
        H = _psd(hessian(st.x)[np.ix_(free, free)])
        resp = np.zeros((M, M - 1))
        rfrac = np.zeros(M)
        for a in range(M):
            mask = np.ones(M, dtype=bool)
            mask[a] = False
            xf = -np.linalg.pinv(H[np.ix_(mask, mask)], rcond=1e-10) @ H[mask, a] if M > 1 else np.zeros(0)
            v = np.zeros(M)
            v[a] = 1.0
            v[mask] = xf
            rfrac[a] = min(max(float(v @ H @ v) / max(float(H[a, a]), 1e-300), 0.0), 1.0)
            resp[a] = xf
        max_xf = np.max(np.abs(resp), axis=1) if M > 1 else np.zeros(M)
        norm_xf = np.linalg.norm(resp, axis=1)

        envs = envs_at(st.x)
        curves = {i: envs.curve(i) for i in free}
        curv = {i: c.curvature() for i, c in curves.items()}
        ceiling = np.minimum(np.array([eps - reserve_for(curv, i) for i in free]),
                             eps0 + share_for(eps0, M, relax))
        free_arr = np.asarray(free)
        x_free = st.x[free_arr]
        pos_free = {int(i): a for a, i in enumerate(free)}

        def candidates(J, EC):
            with np.errstate(invalid="ignore"):
                EP = eps0 + rfrac[None, :] * (EC - eps0)
            adm = EP <= ceiling[None, :]
            if settings.skip_failed and st.failed:
                for (fi, fj), frelax in st.failed.items():
                    if relax <= frelax and fi in pos_free:
                        aa = pos_free[fi]
                        adm[np.flatnonzero(J[:, aa] == fj), aa] = False
            lev_idx, a_idx = np.nonzero(adm)
            j_sel = J[lev_idx, a_idx]
            d_sel = (S[j_sel] - x_free[a_idx] + math.pi) % TWO_PI - math.pi
            eff = level_cost[lev_idx] * weights[free_arr[a_idx]]
            ep_sel = EP[lev_idx, a_idx]
            ec_sel = EC[lev_idx, a_idx]
            rank = np.lexsort((ec_sel, np.abs(d_sel) * norm_xf[a_idx], np.abs(d_sel) * max_xf[a_idx], ep_sel, eff))
            return lev_idx, a_idx, j_sel, d_sel, eff, ep_sel, ec_sel, rank

        def trial_plan(eff, rank, cutoff=np.inf):
            tried: dict[float, int] = {}
            plan = []
            for k in rank:
                level = float(eff[k])
                if tried.get(level, 0) >= settings.top_k:
                    continue
                if level >= cutoff:
                    return plan, False
                tried[level] = tried.get(level, 0) + 1
                plan.append(int(k))
                if len(plan) >= settings.max_round_trials:
                    return plan, True
            return plan, not np.isfinite(cutoff)

        # Best lattice point per cost level for every free angle. Prefiltered
        # angles evaluate only admissible points; angles that admit almost
        # everything are evaluated cheapest level first, only as far as the
        # trials of this round can reach. Both leave the trials unchanged.
        J = np.tile(order[starts][:, None], (1, M))
        EC = np.full((n_lev, M), np.inf)
        lazy = []
        for a, i in enumerate(free):
            mask = None
            if settings.prefilter and rfrac[a] > 1e-12:
                mask = _admissible_mask(curves[i], eps0 + (ceiling[a] - eps0) / rfrac[a], S)
            if mask is not None:
                cand = np.flatnonzero(mask)
                J[:, a], EC[:, a] = best_per_level_subset(cand, np.asarray(curves[i].c_at(S[cand])))
            elif settings.prefilter:
                lazy.append(a)
            else:
                ec = np.asarray(curves[i].c_at(S))
                j = best_per_level(ec)
                J[:, a], EC[:, a] = j, ec[j]
        cut, nxt = 0, min(n_lev, 64)
        w_lazy = float(np.min(weights[free_arr[lazy]])) if lazy else 1.0
        while True:
            if lazy and nxt > cut:
                lo = starts[cut]
                hi = starts[nxt] if nxt < n_lev else len(C)
                idx, sg = order[lo:hi], seg[lo:hi]
                for a in lazy:
                    ec = np.asarray(curves[free[a]].c_at(S[idx]))
                    lex = np.lexsort((ec, sg))
                    first = np.ones(len(lex), dtype=bool)
                    first[1:] = sg[lex][1:] != sg[lex][:-1]
                    pick = lex[first]
                    J[sg[pick], a] = idx[pick]
                    EC[sg[pick], a] = ec[pick]
                cut = nxt
            cands = candidates(J, EC)
            if not lazy or cut >= n_lev:
                break
            _plan, settled = trial_plan(cands[4], cands[7], level_cost[cut] * w_lazy)
            if settled:
                break
            nxt = min(n_lev, cut * 4)
        lev_idx, a_idx, j_sel, d_sel, eff, ep_sel, ec_sel, rank = cands
        if len(rank) == 0:
            say(f"snap: round {st.n_rounds}, {M} free, no candidate predicted under budget (relax {relax:g})")
            return False

        tried_at_level: dict[float, int] = {}
        round_trials = 0
        skip = st.cur_trials_done if st.cur_relax == relax else 0
        st.cur_relax, st.cur_trials_done = relax, skip
        for k in rank:
            level = float(eff[k])
            if tried_at_level.get(level, 0) >= settings.top_k:
                continue
            if round_trials >= settings.max_round_trials:
                say(f"snap: round {st.n_rounds}, {M} free, {round_trials} trials without an acceptance (relax {relax:g})")
                return False
            tried_at_level[level] = tried_at_level.get(level, 0) + 1
            round_trials += 1
            if round_trials <= skip:
                continue
            a = int(a_idx[k])
            i = int(free_arr[a])
            c = float(level_cost[lev_idx[k]])
            d = float(d_sel[k])
            saved = st.x.copy()
            pin(i, st.x[i] + d)
            others = [f for f in free if f != i]
            st.x[others] += d * resp[a]
            st.n_trials += 1
            trial = reopt()
            if trial <= ceiling[a]:
                st.costs[i] = c
                st.n_hess += 1
                st.trace.append({"i": i, "stage": "hessian", "cost": c, "c_pre": float(ec_sel[k]),
                                 "c_pred": float(ep_sel[k]), "c_post": trial, "disp": abs(d)})
                st.history.append((st.x.copy(), st.snapped.copy(), st.costs.copy(), trial))
                st.cur_trials_done = 0
                return True
            unpin(i, saved)
            if settings.skip_failed:
                key = (i, int(j_sel[k]))
                st.failed[key] = max(st.failed.get(key, 0.0), relax)
            st.cur_trials_done = round_trials
            checkpoint_if_due()
        say(f"snap: round {st.n_rounds}, {M} free, {len(rank)} candidates over "
            f"{len(tried_at_level)} levels, none accepted (relax {relax:g})")
        return False

    # -------------------------------------------------------------------- loop
    t0 = time.time()
    if resumed is None:
        floor = reopt()
        say(f"snap: floor C_train = {floor:.3e}, budget = {eps:.3e}")
    else:
        floor = resumed["floor"]
        t0 -= resumed["elapsed"]
    steps = [2.0 ** k for k in range(12)] if settings.relax_ladder == "gentle" else [1.0, 4.0, 16.0, float("inf")]
    while not st.loop_done:
        mid = bool(st.cur_failed_relax) or st.cur_trials_done > 0
        nz = 0 if mid else zero_t_stage()
        n_free = int((~st.snapped).sum())
        if nz:
            checkpoint_if_due()
            say(f"snap: zero-T stage snapped {nz}, {n_free} free, C = {c_fn(st.x):.3e}, "
                f"T/q so far {0.5 * (weights * st.costs).sum():.1f} ({time.time() - t0:.0f}s)")
        if n_free == 0:
            break
        for relax in [r for r in steps if r >= st.relax and r not in st.cur_failed_relax]:
            if hessian_round(relax):
                st.relax = max(1.0, relax / 2.0) if settings.relax_ladder == "gentle" else relax
                st.cur_failed_relax = []
                break
            st.cur_failed_relax.append(relax)
            st.cur_trials_done = 0
            checkpoint(time.time() - t0, floor)
        else:
            st.cur_failed_relax = []
            break
        checkpoint(time.time() - t0, floor)
        if st.n_hess % 5 == 0:
            say(f"snap: {st.n_hess} Hessian snaps, {int((~st.snapped).sum())} free, "
                f"T/q so far {0.5 * (weights * st.costs).sum():.1f}, C = {c_fn(st.x):.3e} ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- end game
    def synthesis_allocation(curves, rest, residual):
        """Equal budget shares s_i e_i^2 = residual / K over K synthesized angles (s = curvature)."""
        share = max(residual, 1e-18) / len(rest)
        out = {}
        for i in rest:
            e = math.sqrt(share / curves[i].curvature())
            out[i] = (e, A_GEN * math.log2(1.0 / e) + B_GEN, None, curves[i].curvature() * e * e)
        return out

    def library_allocation(rest, curves, residual, c_now):
        """Each straggler takes a library angle or KN synthesis, by a Lagrangian sweep."""
        opts = {}
        for i in rest:
            cv = curves[i]
            m = _admissible_mask(cv, c_now + residual * (1.0 + 1e-9), S)
            cand = np.arange(len(S)) if m is None else np.flatnonzero(m)
            lib = []
            if len(cand):
                Jc, Ec = best_per_level_subset(cand, np.asarray(cv.c_at(S[cand])))
                best_dc = np.inf
                for lv in np.flatnonzero(Ec - c_now <= residual * (1.0 + 1e-9)):
                    dc = float(Ec[lv] - c_now)
                    if dc < best_dc:
                        lib.append((float(level_cost[lv]), dc, float(S[Jc[lv]])))
                        best_dc = dc
            opts[i] = (max(cv.curvature(), 1e-300), lib)

        def choose(lam):
            pick, tot = {}, 0.0
            for i, (sv, lib) in opts.items():
                w = float(weights[i])
                dcs = w * A_GEN / (2.0 * math.log(2.0) * lam)
                e = math.sqrt(dcs / sv)
                best = (w * (A_GEN * math.log2(1.0 / e) + B_GEN) + lam * dcs,
                        (e, A_GEN * math.log2(1.0 / e) + B_GEN, None, dcs))
                for c, dc, ang in lib:
                    v = w * c + lam * dc
                    if v < best[0]:
                        best = (v, (None, c, ang, dc))
                pick[i] = best[1]
                tot += best[1][3]
            return pick, tot

        lo, hi = 1e-6, 1e30
        pick, tot = choose(hi)
        if tot > residual:
            return pick
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            p2, t2 = choose(mid)
            if t2 <= residual:
                hi, pick = mid, p2
            else:
                lo = mid
            if hi / lo < 1.0 + 1e-6:
                break
        return pick

    def straggler_costs(x, snapped, c_now):
        rest = [int(i) for i in np.flatnonzero(~snapped)]
        if not rest:
            return {}
        curves = envs_at(x).curves(rest)
        residual = max(eps - c_now, 0.0)
        if settings.library_stragglers:
            return library_allocation(rest, curves, residual, c_now)
        return synthesis_allocation(curves, rest, residual)

    def total_teq(costs, alloc):
        paid = costs.copy()
        for i, e in alloc.items():
            paid[i] = e[1]
        return 0.5 * float((weights * paid).sum())

    if not st.loop_done:
        st.loop_done = True
        checkpoint(time.time() - t0, floor)
    K = len(st.history)
    eg = st.eg_teq
    if "full" not in eg:
        eg["full"] = total_teq(st.costs, straggler_costs(st.x, st.snapped, c_fn(st.x)))
        checkpoint_if_due()
    for k in range(K - 1, -1, -1):
        if k not in eg:
            xk, snk, ck, c_k = st.history[k]
            eg[k] = total_teq(ck, straggler_costs(xk, snk, c_k))
            checkpoint_if_due()
    checkpoint(time.time() - t0, floor)
    best_key = "full"
    for k in range(K - 1, -1, -1):
        if eg[k] < eg[best_key] - 1e-9:
            best_key = k
    if best_key == "full":
        n_prefix = K
    else:
        n_prefix = best_key + 1
        xk, snk, ck, _c = st.history[best_key]
        say(f"snap: best prefix keeps {n_prefix} of {K} snaps, T/q {eg[best_key]:.1f} against {eg['full']:.1f}")
        st.x[:], st.snapped[:], st.costs[:] = xk, snk, ck
    c_prefix = c_fn(st.x)
    alloc = straggler_costs(st.x, st.snapped, c_prefix)
    if any(e[2] is not None for e in alloc.values()):
        # The allocation adds plug-in rises one angle at a time; check the joint circuit.
        x0 = st.x.copy()
        room = max(eps - c_prefix, 0.0)
        residual, fits, xt = room, False, x0
        for _ in range(30):
            xt = x0.copy()
            for i, e in alloc.items():
                if e[2] is not None:
                    xt[i] = _unwrap(x0[i], e[2])
            used = c_fn(xt) - c_prefix + sum(e[3] for e in alloc.values() if e[2] is None)
            if used <= room * (1.0 + 1e-9):
                fits = True
                break
            residual *= 0.8 * room / max(used, 1e-300)
            rest = [int(i) for i in np.flatnonzero(~st.snapped)]
            alloc = library_allocation(rest, envs_at(x0).curves(rest), residual, c_prefix)
        if fits:
            st.x[:] = xt
        else:
            rest = [int(i) for i in np.flatnonzero(~st.snapped)]
            alloc = synthesis_allocation(envs_at(x0).curves(rest), rest, room)
            say("snap: library stragglers did not fit jointly; synthesizing all of them")
    lib_alloc = {i: e for i, e in alloc.items() if e[2] is not None}
    syn_alloc = {i: e for i, e in alloc.items() if e[2] is None}
    for i, e in lib_alloc.items():
        st.costs[i] = e[1]
        st.snapped[i] = True
    for i, e in syn_alloc.items():
        st.costs[i] = e[1]
    c_final = c_fn(st.x)
    tq = 0.5 * float((weights * st.costs).sum())
    say(f"snap: done, zero-T {st.n_zero}, Hessian {st.n_hess}, library stragglers {len(lib_alloc)}, "
        f"KN stragglers {len(syn_alloc)}, planned T/q {tq:.1f}, C_train {c_final:.3e} ({time.time() - t0:.0f}s)")
    return {
        "settings": asdict(settings),
        "floor": floor,
        "C_train": c_final,
        "angles": st.x.tolist(),
        "snapped": st.snapped.tolist(),
        "costs_plan": st.costs.tolist(),
        "multiplicity": weights.tolist(),
        "Tq_plan": tq,
        "stragglers": {str(i): {"eps": e[0], "T_plan": e[1], "dC_plan": e[3]} for i, e in syn_alloc.items()},
        "library_stragglers": {str(i): {"angle": e[2], "cost": e[1], "dC": e[3]} for i, e in lib_alloc.items()},
        "n_zero_t": st.n_zero, "n_hessian": st.n_hess, "n_rounds": st.n_rounds, "n_trials": st.n_trials,
        "n_snaps_kept": n_prefix, "n_snaps_trajectory": K, "Tq_full_trajectory": eg["full"],
        "trace": st.trace,
        "history": [{"x": xk.tolist(), "snapped": snk.tolist(), "costs": ck.tolist(), "c": c_k}
                    for xk, snk, ck, c_k in st.history],
        "seconds": time.time() - t0,
    }
