"""Open-chain SNAP: the Hessian-ranked protocol of snap.py with every cost on the open chain, by MPS.

The ring arm (snap.py, realize.py, edges.fit_edges) compiles on a 10-site ring with
state vectors. This module is the open-chain arm: it never builds a ring. Training,
snapping, re-optimization, the straggler budget checks and the edges all use the
cost mps_train.OpenChainCost on the open chain of N sites (N = 100) and the training
inputs of train_chain (product_vectors(N, 12, seed 0) against order-4 references).

Parameters. z = (bulk, edges) as in mps_train and train_chain: D n_angles bulk angles
(one set per sublayer, shared by its bonds), then 6 ZYZ angles per parity-1 sublayer
for the end gates on sites N-1 and 0. The bulk angles go through the whole protocol.
The edge angles are re-optimized with the bulk after every snap and are part of the
Hessian's free block (so the predicted response moves them too); they are not snap
candidates of the Hessian rounds, and enter the zero-T stage only with edge_zero_t. In
the end game they are stragglers with T weight edge_weight per angle (chain_point: 2/N,
an end gate appears once per circuit, a bulk gate N/2 times), each given a library angle
or a KN budget by the same Lagrangian allocation as the bulk stragglers. realize_chain
then simplifies the edges under the budget (identity gates, Rz for gates whose middle
angle is near 0, multiples of pi/4 within 1e-2, all checked on the chain) and prices
what is left by its allocation (library angles at their lattice cost, KN at the
allotted budget). The ring arm (RingProblem) has no edge angles in z.

Pieces
    ChainPool          G input groups x K replicas of single-process spawned workers. Each
                       worker holds its group's inputs and targets for the whole compile
                       and, while an environment is alive, its forward and backward caches.
    FarmCost           OpenChainCost whose inputs run on a ChainPool. value_and_grad is
                       OpenChainCost's own code, so c and dc/dz are those numbers exactly;
                       value_and_grad_many runs many points at once over all replicas.
    fd_hessian(_rows)  Central differences of FarmCost gradients over chosen rows.
    polish_chain       Trust-region Newton steps on that Hessian, the start's polish.
    ChainEnvironments  Forward states a_l (before sublayer l) and pulled-back bras b_l per
                       input, for the sublayers asked for, kept in the workers (layer blocks
                       over the K replicas) until release(). curve(i) is the exact C of the
                       chain against one bulk angle.
    compile_chain      The protocol of snap.compile_snap on these costs.
    realize_chain      Lattice pricing, KN stragglers checked on the chain, edges simplified
                       and synthesized.
    evaluate_chain     evaluate.mps_infidelity at N = 100 on the sweep's inputs, fallback
                       totals and T counts, without the ring side metric.
    ChainProblem       Wires pool, cost, Hessian and environments for one (model, t, D).
    chain_point        Depth pick, polish, compile, realize, evaluate and the record files.

Curves. Sublayer l applies one gate G(theta) to n_g bonds (N/2 for parity 0, N/2 - 1 for
parity 1; the end gates do not depend on bulk angles). <b_l| L_l(theta) |a_l> is
contracted directly, without truncation, so it is exactly a trigonometric polynomial
in theta/2 whose harmonics are n_g-fold sums of those of G (cells.py: +-1 for the
cos/sin slots, 0 and +-2 for the staggered slot). With harmonics h_min..h_max in
steps of g, M = n_g (h_max - h_min)/g + 1 equally spaced probes over 4 pi/g and a DFT
fix it (51 probes for a harmonic-1 slot at N = 100, 101 for the staggered one; the
generic bound 2 x degree + 1 would be 101 and 201). One more probe at the current
angle checks the fit (ChainCurve.residual; above RESIDUAL_MAX the curve is refused).
F_k(theta) = |<b_l|L_l|a_l>|^2 / (<a_l|a_l><b_l|b_l>), the analogue of OpenChainCost's
F, which divides by the norms; c = -mean expm1(log F / N), OpenChainCost's formula. The
sandwich at the current angle differs from the forward amplitude by the truncation of
sublayers l..D-1 (the cost's amp_dev), so curve(i).c_at(theta0) equals
OpenChainCost.value(z) to truncation accuracy, and exactly at small N without truncation.

Differences from snap.compile_snap (everything else is copied unchanged)
 1. Costs: objective, c_fn and Hessian are FarmCost (OpenChainCost) on the open chain's
    training inputs; curves come from ChainEnvironments (MPS sandwiches) instead of
    environment.Environments (ring state vectors). RingProblem runs the same code on
    compile_snap's ring costs.
 2. Parameters are z = (bulk, edges) (Parameters above): snap candidates, the reserve and
    share guards and the T accounting of the Hessian rounds run over the free bulk angles
    (n in the reserve cap is the number of bulk angles); the end game, and with
    edge_zero_t the zero-T stage, include the edges at weight edge_weight.
 3. The Hessian is over the free bulk and edge angles (only those rows, by central
    differences of gradients at FD_STEP on the chain), so the response x_f = -H_ff^+ H_fp,
    the surviving fraction r_p and the start of a trial include the edges. compile_snap
    differentiates all angles, snapped ones too, and keeps the free block; the block is
    the same.
 4. The pseudo-inverse in x_f uses rcond RESPONSE_RCOND = 1e-8, not 1e-10: the finite-
    difference Hessian is noisy at ~5e-8 of its largest entry, and below that the
    gauge-flat directions of the ZYZ edge angles gave edge moves of 20 rad (t = 1; r_p
    unchanged to 4 digits). legacy_settings restores 1e-10.
 5. The re-optimizer is ChainSettings.reopt, chosen explicitly (no default): "newton",
    "lbfgs-converged" or "coordinate" (ChainSettings docstring), none with an iteration or
    sweep cap and none a fallback for another. compile_snap re-optimizes with L-BFGS-B
    capped at 250 iterations (snap._REOPT); the tests reproduce production compiles by
    passing that re-optimization through compile_chain's ``reoptimizer`` argument, which
    nothing else uses.
 6. Trials of one round can run trial_workers at a time (not with "coordinate"), in rank
    order, each from the round's saved state; the first passing one in rank order is
    accepted, later ones are aborted. Every trial starts from the same state in
    compile_snap too (a failed trial restores it), so the accepted snap is the one
    compile_snap accepts; n_trials counts the trials compile_snap would have run,
    n_trials_run all that were started.
 7. Resume is exact. The state keeps the current round (its re-optimized point, eps0,
    the Hessian rows done so far, the response and the trial plan), so a resumed compile
    continues at the next Hessian chunk or trial and ends where an uninterrupted one
    ends; compile_snap re-runs the round's eps0 = reopt() on resume. The deadline (and
    SIGTERM, and an optional stop event) is checked at every cost evaluation of a
    re-optimization, between sublayers of a coordinate sweep and between Hessian chunks;
    an interrupted re-optimization or chunk is discarded and redone on resume, from the
    same point.
 8. Curves are cached by the exact parameter vector, built only for the sublayers that
    have free angles and released from the workers once made, and the end game's cost
    of prefix k is computed as soon as the zero-T stage has the curves at that prefix's
    point. The values are those compile_snap computes at the end (same point, same curves).
 9. ChainCurve.c_at uses OpenChainCost's formula -mean expm1(log F / N) instead of
    1 - mean F^(1/N) (the same number without the cancellation), and _admissible_mask
    keeps its grid basis between calls (same arithmetic).
10. Bookkeeping only: seconds per stage, re-optimizations and cost evaluations (timing),
    one reopt_log entry per re-optimization, Hessian statistics per round (negative
    eigenvalues, lowest eigenvalue, r_p, asymmetry), worker memory, n_trials_run, and
    records keep z, the compiled edges and n_bulk.
Outside compile_chain: chain_point polishes the trained z* with trust-region Newton steps on
the finite-difference Hessian (polish_chain), as train.py's polish does for the ring ladders,
because an unconverged start leaves negative curvature that the PSD projection turns into
r_p = 0 for every angle; unlike train.py's fixed steps it has no step cap and stops on the
convergence test of the newton re-optimizer (POLISH_TOL_REL x budget) or at the noise floor.
realize_chain simplifies the edges before pricing them (realize_chain docstring); the ring
arm's edges.fit_edges does the same near-pi/4 setting for its fitted edges.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import pickle
import resource
import signal
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import scipy.optimize as sopt

from snapevo.cells import Cell, get_cell
from snapevo.circuits import Layered
from snapevo.edges import edge_eps, pairs, u_zyz
from snapevo.environment import AngleCurve
from snapevo.library import QUARTER, TWO_PI, Library
from snapevo.mps import MPS, apply_layered
from snapevo.mps_train import OpenChainCost, _apply_dagger, _blocks, _jax_maps, _run_input, _snapshot, _threads_env
from snapevo.snap import _GRID, _REOPT, SnapSettings, _psd, _unwrap
from snapevo.synthesis import kn_law, synthesize

__all__ = ["FD_STEP", "RESPONSE_RCOND", "RESIDUAL_MAX", "POLISH_TOL_REL", "EDGE_WINDOW",
           "EDGE_ROOM_FRAC",
           "ChainPool", "FarmCost", "fd_hessian", "fd_hessian_rows", "polish_chain", "newton_minimize", "ChainCurve",
           "ChainEnvironments", "ChainProblem", "RingProblem", "ChainSettings", "legacy_settings", "REOPTIMIZERS",
           "compile_chain", "ring_point", "realize_chain", "realize_chain_edges",
           "simplify_edges", "evaluate_chain", "pick_depth", "chain_point", "DeadlineReached", "SnapSettings"]

FD_STEP = 1e-4
"""Step (rad) of the central-difference Hessian, as jaxcost.make_hessian's "fd"."""

RESPONSE_RCOND = 1e-8
"""rcond of the pseudo-inverse in the trial response (module docstring, difference 4)."""

RESIDUAL_MAX = 1e-8
"""Largest accepted relative misfit of a curve at its check probe (exact up to rounding, ~1e-14)."""

POLISH_TOL_REL = 1e-6
"""chain_point polishes until the predicted Newton decrease and the last step's are below this x budget."""

EDGE_WINDOW = 1e-2
"""realize_chain tries angles within this of a multiple of pi/4 (edges.fit_edges' window)."""

REOPTIMIZERS = ("newton", "lbfgs-converged", "coordinate")
"""ChainSettings.reopt choices (no default; ChainSettings docstring)."""

NEWTON_DELTA0, NEWTON_DELTA_MAX, NEWTON_DELTA_MIN = 1.0, 64.0, 1e-12
"""Initial, largest and collapsed ("noise floor") trust radius (rad) of the "newton" re-optimizer."""

COORD_GRID, COORD_REFINE = 2048, 3
"""The "coordinate" re-optimizer's grid: at least COORD_GRID points (and 32 per highest harmonic) over one
4 pi period of an angle's curve; the COORD_REFINE lowest grid minima are refined by bounded Brent."""

EDGE_ROOM_FRAC = 0.5
"""Share of the room left under the budget that realize_chain's edge simplification may use."""

_PROBE_BYTES = 1 << 28
"""Memory per probe batch in a worker; larger batches are split."""


@dataclass
class ChainSettings(SnapSettings):
    """snap.SnapSettings plus what compile_chain adds; every field is stored in the snap_ record.

    reopt: the re-optimizer of every re-optimization (round start, zero-T, trials, end game).
        There is no default: compile_chain refuses settings without one. Three modes, none
        a fallback for another, all without an iteration or sweep cap:
      "newton": trust-region Newton steps on the free parameters with the Hessian at hand (a
        trial: the round's, PSD-projected and restricted to the other free parameters; the
        round-start and zero-T re-optimizations: the latest one) and a fresh gradient each
        step. The Hessian is recomputed at the current point (a refresh: central
        differences on the chain, jaxcost's on the ring) when none is at hand, at the first
        of a run of steps that do not lower c (the trust region shrinks 4x per such step),
        and every newton_refresh_every accepted steps. It stops when the predicted Newton
        decrease g.H^+.g/2 on the free parameters and the last accepted step's decrease are
        both below newton_tol_rel x budget ("converged"), or when the trust radius has
        collapsed below NEWTON_DELTA_MIN after a refresh ("noise floor": steps no longer
        lower c, as at the MPS truncation floor).
      "lbfgs-converged": L-BFGS-B with snap._REOPT's ftol and gtol, stopped when c fell by
        less than lbfgs_tol_rel x budget over the last lbfgs_window iterations.
      "coordinate": sweeps over the free angles (edges too on the chain) in circuit order,
        each set to the global minimum of its exact single-angle curve (a grid of at least
        COORD_GRID points over the curve's 4 pi period, the COORD_REFINE lowest grid minima
        refined by bounded Brent), with the environments carried along the sweep; it stops
        when a sweep lowers c by less than newton_tol_rel x budget ("converged") or raises
        the exact c ("noise floor", the sweep undone). One trial at a time only.
      reopt_log records, per re-optimization, the mode's steps (refreshes, rejected steps
        and the final predicted decrease; iterations; sweeps, curves and moves), the stop
        reason, cost evaluations, c and seconds.
    response_rcond: rcond of the pseudo-inverse in the trial response (compile_snap: 1e-10).
    edge_weight: per-qubit T weight of one edge rotation in the end game, 2/N (an edge
        rotation appears once per circuit); 0 leaves the edges out of the end game.
    edge_zero_t: let the zero-T stage set edge angles to cost-0 points, accepted on the
        exact c of the circuit with that one angle changed, without a re-optimization.
    """

    reopt: str | None = None
    newton_tol_rel: float = 1e-6
    newton_refresh_every: int = 20
    lbfgs_window: int = 10
    lbfgs_tol_rel: float = 1e-4
    response_rcond: float = RESPONSE_RCOND
    edge_weight: float = 0.0
    edge_zero_t: bool = False


def legacy_settings(budget: float, **kw) -> ChainSettings:
    """compile_snap's choices of the shared code (rcond 1e-10, no edge stragglers), for the reproduction test."""
    return ChainSettings(budget=budget, **{"response_rcond": 1e-10, "edge_weight": 0.0, **kw})


class DeadlineReached(RuntimeError):
    """compile_chain (or polish_chain) stopped at its deadline, SIGTERM or stop event, after its checkpoint."""


class _Abort(Exception):
    """A speculative trial that is no longer needed."""


class _Stop(Exception):
    """The deadline passed inside a cost evaluation; the interrupted step is discarded."""


# ============================================================================ workers

_CW: dict = {}


def _cw_init(cell_name: str, inputs, targets, chi_max: int, cutoff: float) -> None:
    _CW.clear()
    _CW.update(cell=get_cell(cell_name), inputs=inputs, targets=targets, chi_max=int(chi_max), cutoff=float(cutoff),
               env=None)
    if mp.current_process().name != "MainProcess":
        # SLURM sends SIGTERM to every process of the job; the parent checkpoints on it and needs its
        # workers until then (they still exit when the parent shuts them down or dies)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _cw_ping() -> int:
    return os.getpid()


def _cw_run(x, e, grad: bool) -> list[dict]:
    """OpenChainCost's per-input pass (mps_train._run_input) for this worker's inputs."""
    w = _CW
    return [_run_input(w["cell"], x, e, v, tg, w["chi_max"], w["cutoff"], grad)
            for v, tg in zip(w["inputs"], w["targets"])]


def _layers_of(cell: Cell, x, e) -> list[tuple]:
    circuit = Layered.brickwork(cell, x)
    edge = iter(pairs(e))
    return [(p, a, next(edge) if p == 1 else None) for p, a in circuit.layers]


def _env_states(cell: Cell, layers, vec, target: MPS, chi_max: int, cutoff: float, want: set):
    """a_l and b_l for l in want, made exactly as mps_train._run_input makes them."""
    D = len(layers)
    lo, hi = min(want), max(want)
    a, b = {}, {}
    psi = MPS.product(vec, chi_max=chi_max, cutoff=cutoff)
    for l in range(hi + 1):
        if l in want:
            a[l] = _snapshot(psi)
        if l < hi:
            parity, angles, pair = layers[l]
            apply_layered(psi, Layered(cell, [(parity, angles)]), None if pair is None else [pair])
    nv = math.sqrt(abs(target.overlap(target)))
    bra = _snapshot(target)
    bra.A[bra.center] = bra.A[bra.center] / nv
    bra.chi_max, bra.cutoff, bra.discarded = int(chi_max), float(cutoff), 0.0
    bra.chi_peak = max(t.shape[2] for t in bra.A)
    for l in range(D - 1, lo - 1, -1):
        if l in want:
            b[l] = _snapshot(bra)
        if l > lo:
            parity, angles, pair = layers[l]
            _apply_dagger(bra, cell, parity, angles, pair)
    return a, b


def _rss_mb() -> float:
    """Peak resident memory of this process so far, MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _cw_env_build(key: str, x, e, want) -> dict:
    """Build this worker's caches for the sublayers in ``want`` (replacing any older ones)."""
    w = _CW
    w["env"] = None
    t0 = time.perf_counter()
    want = sorted(int(l) for l in want)
    layers = _layers_of(w["cell"], x, e)
    states = []
    for vec, tg in zip(w["inputs"], w["targets"]):
        a, b = _env_states(w["cell"], layers, vec, tg, w["chi_max"], w["cutoff"], set(want))
        norms = {l: math.sqrt(abs(a[l].overlap(a[l])) * abs(b[l].overlap(b[l]))) for l in want}
        states.append((a, b, norms))
    w["env"] = {"key": key, "layers": layers, "want": want, "states": states}
    mps_all = [m for a, b, _ in states for d in (a, b) for m in d.values()]
    return {"pid": os.getpid(), "seconds": time.perf_counter() - t0,
            "bytes": int(sum(t.nbytes for m in mps_all for t in m.A)),
            "chi": int(max((max(t.shape[2] for t in m.A) for m in mps_all), default=1)), "rss_mb": _rss_mb()}


def _sandwich(a: MPS, b: MPS, parity: int, G: np.ndarray, pair) -> np.ndarray:
    """<b| L_q |a> for every q, L_q the sublayer with bulk gate G[q] (Q, 4, 4) and end gates ``pair``.

    pair = (gate on N-1, gate on 0), each (2, 2) or one per q, (Q, 2, 2). No truncation:
    the gates of one sublayer act on disjoint blocks, contracted left to right with the
    left environment of every q in one batch.
    """
    Q = G.shape[0]
    L = np.ones((Q, 1, 1), dtype=complex)
    for i, w in _blocks(a.N, parity):
        if w == 2:
            ket = np.tensordot(a.A[i], a.A[i + 1], axes=(2, 0))
            bra = np.tensordot(b.A[i], b.A[i + 1], axes=(2, 0)).conj()
            l, r, lb, rb = ket.shape[0], ket.shape[3], bra.shape[0], bra.shape[3]
            Y = (L.reshape(Q * lb, l) @ ket.reshape(l, 4 * r)).reshape(Q, lb, 4, r)
            Y = np.matmul(G, Y.transpose(0, 2, 1, 3).reshape(Q, 4, lb * r))            # (Q, p, lb r)
            Y = Y.reshape(Q, 4, lb, r).transpose(2, 1, 0, 3).reshape(lb * 4, Q * r)
            L = (bra.reshape(lb * 4, rb).T @ Y).reshape(rb, Q, r).transpose(1, 0, 2)
        else:
            g = np.asarray(pair[1] if i == 0 else pair[0], dtype=complex)            # pair = (g on N-1, g on 0)
            bra = b.A[i].conj()
            if g.ndim == 2:
                ket = np.tensordot(g, a.A[i], axes=(1, 1)).transpose(1, 0, 2)
                l, r, lb, rb = ket.shape[0], ket.shape[2], bra.shape[0], bra.shape[2]
                Y = (L.reshape(Q * lb, l) @ ket.reshape(l, 2 * r)).reshape(Q, lb, 2, r)
            else:                                                    # one end gate per q
                ket = a.A[i]
                l, r, lb, rb = ket.shape[0], ket.shape[2], bra.shape[0], bra.shape[2]
                Y = (L.reshape(Q * lb, l) @ ket.reshape(l, 2 * r)).reshape(Q, lb, 2, r)
                Y = np.matmul(g, Y.transpose(0, 2, 1, 3).reshape(Q, 2, lb * r))
                Y = Y.reshape(Q, 2, lb, r).transpose(0, 2, 1, 3)
            Y = Y.transpose(1, 2, 0, 3).reshape(lb * 2, Q * r)
            L = (bra.reshape(lb * 2, rb).T @ Y).reshape(rb, Q, r).transpose(1, 0, 2)
    return L[:, 0, 0].copy()


def _cw_env_probe(key: str, jobs) -> list[np.ndarray]:
    """jobs: (layer, slot, thetas[, edge]). Per job, (len(thetas), n_local) amplitudes over sqrt(<a|a><b|b>).

    A bulk job sets the sublayer's angle ``slot`` to each theta. An edge job has edge =
    (side, j, abc): the end gate on site N-1 (side 0) or 0 (side 1) becomes
    Rz Ry Rz of abc with its angle j set to each theta.
    """
    env = _CW.get("env")
    if env is None or env["key"] != key:
        raise RuntimeError("this worker no longer holds that environment")
    cell, layers = _CW["cell"], env["layers"]
    n_loc = len(env["states"])
    out: list = [None] * len(jobs)
    by_layer: dict[int, list[int]] = {}
    for j, job in enumerate(jobs):
        if int(job[0]) not in env["states"][0][0]:
            raise RuntimeError(f"sublayer {job[0]} is not in this environment")
        by_layer.setdefault(int(job[0]), []).append(j)
    for l, idx in by_layer.items():
        parity, angles, pair = layers[l]
        G0 = cell.gate(angles)
        G, GL, GF, sizes = [], [], [], []
        for j in idx:
            slot, thetas = jobs[j][1], jobs[j][2]
            edge = jobs[j][3] if len(jobs[j]) > 3 else None
            for th in thetas:
                gl, gf = (None, None) if pair is None else pair
                if edge is None:
                    ang = np.array(angles, dtype=float)
                    ang[slot] = th
                    G.append(cell.gate(ang))
                else:
                    side, k, abc = edge
                    abc = np.array(abc, dtype=float)
                    abc[k] = th
                    if side == 0:
                        gl = u_zyz(*abc)
                    else:
                        gf = u_zyz(*abc)
                    G.append(G0)
                if pair is not None:
                    GL.append(gl)
                    GF.append(gf)
            sizes.append(len(thetas))
        G = np.asarray(G)
        GL = np.asarray(GL) if pair is not None else None
        GF = np.asarray(GF) if pair is not None else None
        res = np.empty((len(G), n_loc), dtype=complex)
        for k, (a, b, norms) in enumerate(env["states"]):
            chi = max(max(t.shape[2] for t in a[l].A), max(t.shape[2] for t in b[l].A))
            step = max(1, int(_PROBE_BYTES // (3 * 64 * chi * chi)))
            for lo in range(0, len(G), step):
                pq = None if pair is None else (GL[lo: lo + step], GF[lo: lo + step])
                res[lo: lo + step, k] = _sandwich(a[l], b[l], parity, G[lo: lo + step], pq) / norms[l]
        pos = np.cumsum([0] + sizes)
        for j, p0, p1 in zip(idx, pos[:-1], pos[1:]):
            out[j] = res[p0:p1]
    return out


def _cw_env_set_layer(key: str, l: int, angles, edge6=None) -> None:
    """Change sublayer l's bulk angles (and, for parity 1, its six edge angles) in this worker's environment."""
    env = _CW.get("env")
    if env is None or env["key"] != key:
        raise RuntimeError("this worker no longer holds that environment")
    parity, _a, pair = env["layers"][l]
    if edge6 is not None:
        pair = pairs(np.asarray(edge6, dtype=float))[0]
    env["layers"][l] = (parity, np.asarray(angles, dtype=float), pair)


def _cw_env_advance(key: str, l: int) -> None:
    """a_(l+1) = L_l a_l with sublayer l as it is now (truncated as apply_layered truncates)."""
    env = _CW.get("env")
    if env is None or env["key"] != key:
        raise RuntimeError("this worker no longer holds that environment")
    parity, angles, pair = env["layers"][l]
    for a, b, norms in env["states"]:
        if l + 1 in a:
            psi = _snapshot(a[l])
            apply_layered(psi, Layered(_CW["cell"], [(parity, angles)]), None if pair is None else [pair])
            a[l + 1] = psi
            norms[l + 1] = math.sqrt(abs(psi.overlap(psi)) * abs(b[l + 1].overlap(b[l + 1])))


def _cw_env_clear(key: str | None = None) -> None:
    """Drop this worker's caches (only those of ``key`` when given)."""
    env = _CW.get("env")
    if env is not None and (key is None or env["key"] == key):
        _CW["env"] = None


# ============================================================================ pool and cost

class _LocalWorker:
    """A worker that runs in this process, with its own worker state (tests and tiny problems).

    Calls run at submit, one at a time across all local workers (the worker functions
    keep their state in the module's _CW, which is swapped in and out around each call).
    """

    _lock = threading.RLock()

    def __init__(self, initargs):
        self.state: dict = {}
        self._run(_cw_init, *initargs)

    def _run(self, fn, *args):
        with _LocalWorker._lock:
            saved = dict(_CW)
            _CW.clear()
            _CW.update(self.state)
            try:
                return fn(*args)
            finally:
                self.state = dict(_CW)
                _CW.clear()
                _CW.update(saved)

    def submit(self, fn, *args) -> cf.Future:
        f: cf.Future = cf.Future()
        try:
            f.set_result(self._run(fn, *args))
        except BaseException as exc:                              # noqa: BLE001 (handed to the caller)
            f.set_exception(exc)
        return f

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.state = {}


class ChainPool:
    """G input groups x K replicas of single-process workers (spawned, ``threads`` BLAS threads each).

    Worker (g, r) holds inputs groups[g] (round-robin, as OpenChainCost splits them) and
    their targets. submit(g, ...) runs on the least busy replica of group g unless a
    replica is named (environments use replica r for layer block r). local=True runs
    the same worker functions in this process instead (no parallelism; for tests and
    problems so small that inter-process latency would dominate).
    """

    def __init__(self, cell: Cell, inputs, targets, chi_max: int, cutoff: float, groups: int | None = None,
                 replicas: int = 1, threads: int = 1, local: bool = False):
        inputs = np.asarray(inputs, dtype=complex)
        n = len(targets)
        G = max(1, min(int(groups or n), n))
        self.n, self.G, self.K = n, G, max(1, int(replicas))
        self.groups = [list(range(g, n, G)) for g in range(G)]
        self.chi_max, self.cutoff, self.threads, self.local = int(chi_max), float(cutoff), int(threads), bool(local)
        self._lock = threading.Lock()
        self._load = [[0] * self.K for _ in range(G)]
        args = [(cell.name, inputs[grp], [targets[k] for k in grp], self.chi_max, self.cutoff) for grp in self.groups]
        if self.local:
            self.workers = [[_LocalWorker(a) for _ in range(self.K)] for a in args]
            return
        ctx = mp.get_context("spawn")
        with _threads_env(threads):
            self.workers = [[cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_cw_init, initargs=a)
                             for _ in range(self.K)] for a in args]
            for f in [w.submit(_cw_ping) for row in self.workers for w in row]:
                f.result()

    @property
    def size(self) -> int:
        return self.G * self.K

    def submit(self, g: int, fn, *args, replica: int | None = None) -> cf.Future:
        with self._lock:
            r = int(np.argmin(self._load[g])) if replica is None else int(replica)
            self._load[g][r] += 1
        try:
            fut = self.workers[g][r].submit(fn, *args)
        except BaseException:
            self._done(g, r)
            raise
        fut.add_done_callback(lambda _f, g=g, r=r: self._done(g, r))
        return fut

    def _done(self, g: int, r: int) -> None:
        with self._lock:
            self._load[g][r] -= 1

    def rows(self, x, e, grad: bool) -> list[cf.Future]:
        return [self.submit(g, _cw_run, x, e, grad) for g in range(self.G)]

    def gather(self, futures) -> list[dict]:
        out: list = [None] * self.n
        for grp, f in zip(self.groups, futures):
            for k, r in zip(grp, f.result()):
                out[k] = r
        return out

    def close(self) -> None:
        for row in getattr(self, "workers", []):
            for w in row:
                w.shutdown(wait=True, cancel_futures=True)
        self.workers = []


class FarmCost(OpenChainCost):
    """OpenChainCost with its inputs on a ChainPool; thread-safe, and value_and_grad_many runs points at once.

    Only _results differs from OpenChainCost: the per-input rows come from the pool,
    and the rest of value and value_and_grad is OpenChainCost's code.
    """

    def __init__(self, model, cell: Cell, t: float, D: int, N: int, inputs, targets, pool: ChainPool):
        super().__init__(model, cell, t, D, N, inputs, targets, chi_max=pool.chi_max, cutoff=pool.cutoff, processes=1)
        if pool.n != self.n:
            raise ValueError(f"the pool holds {pool.n} inputs, the cost {self.n}")
        self.pool = pool
        self._tls = threading.local()
        self._gates, self._pull = _jax_maps(cell, self.D, self.n_odd)      # before any thread calls the cost

    def _xe(self, z):
        x, e = self.split(z)
        return x.reshape(-1), e

    def _results(self, z, grad: bool) -> list[dict]:
        pre = getattr(self._tls, "rows", None)
        if pre is not None:
            self._tls.rows = None
            return pre
        return self.pool.gather(self.pool.rows(*self._xe(z), grad))

    def value_and_grad_many(self, zs) -> list[tuple[float, np.ndarray]]:
        """[(c, dc/dz)] at every point, all submitted before any is gathered."""
        zs = [np.asarray(z, dtype=float).reshape(-1) for z in zs]
        handles = [self.pool.rows(*self._xe(z), True) for z in zs]
        out = []
        for z, h in zip(zs, handles):
            self._tls.rows = self.pool.gather(h)
            out.append(self.value_and_grad(z))
        return out

    def value_many(self, zs) -> list[float]:
        zs = [np.asarray(z, dtype=float).reshape(-1) for z in zs]
        handles = [self.pool.rows(*self._xe(z), False) for z in zs]
        out = []
        for z, h in zip(zs, handles):
            self._tls.rows = self.pool.gather(h)
            out.append(self.value(z))
        return out

    def close(self) -> None:                     # the pool belongs to whoever made it
        pass


def fd_hessian_rows(cost: FarmCost, z, rows, cols, h: float = FD_STEP) -> np.ndarray:
    """Raw central-difference rows R[a, b] = (g(z + h e_rows[a]) - g(z - h e_rows[a]))[cols[b]] / 2h.

    2 len(rows) gradients, all submitted at once.
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    r_idx = np.asarray(list(rows), dtype=int)
    c_idx = np.asarray(list(cols), dtype=int)
    pts = []
    for k in r_idx:
        e = np.zeros(z.size)
        e[k] = h
        pts += [z + e, z - e]
    res = cost.value_and_grad_many(pts)
    R = np.empty((r_idx.size, c_idx.size))
    for a in range(r_idx.size):
        R[a] = (res[2 * a][1][c_idx] - res[2 * a + 1][1][c_idx]) / (2.0 * h)
    return R


def _symmetrize(R: np.ndarray) -> tuple[np.ndarray, float]:
    asym = float(np.max(np.abs(R - R.T)) / max(float(np.max(np.abs(R))), 1e-300)) if R.size else 0.0
    return 0.5 * (R + R.T), asym


def fd_hessian(cost: FarmCost, z, rows, h: float = FD_STEP) -> tuple[np.ndarray, float]:
    """Hessian of c over the parameters ``rows``, by central differences of gradients, symmetrized.

    Returns (H, asym) with asym = max|R - R^T| / max|R| of the raw rows R, a measure of
    the finite-difference and truncation noise. 2 len(rows) gradients, all at once.
    """
    rows = list(rows)
    return _symmetrize(fd_hessian_rows(cost, z, rows, rows, h))


def _eig_stats(H: np.ndarray, nb_rows: int) -> dict:
    """Lowest eigenvalue and negative count of H and of its first nb_rows x nb_rows (bulk) block."""
    w = np.linalg.eigvalsh(H) if H.size else np.zeros(1)
    wb = np.linalg.eigvalsh(H[:nb_rows, :nb_rows]) if nb_rows else np.zeros(1)
    return {"eig_min": float(w[0]), "eig_max": float(w[-1]), "n_neg": int((w < 0).sum()),
            "eig_min_bulk": float(wb[0]), "n_neg_bulk": int((wb < 0).sum())}


# ============================================================================ curves

class ChainCurve(AngleCurve):
    """environment.AngleCurve whose C is OpenChainCost's formula, -mean expm1(log F / N)."""

    def c_at(self, theta):
        F = np.maximum(np.abs(self.overlap_at(theta)) ** 2, 1e-300)
        return -np.mean(np.expm1(np.log(F) / self.N), axis=-1)


@lru_cache(maxsize=None)
def _slot_modes(cell_name: str, slot: int) -> tuple[int, int, int]:
    """(lowest, highest, step) of the harmonics of theta/2 in the cell gate's dependence on one slot."""
    cell = get_cell(cell_name)
    h = int(cell.harmonic[slot])
    M0 = 4 * h + 3
    base = np.random.default_rng(1234).uniform(-math.pi, math.pi, cell.n_angles)
    th = 4.0 * math.pi * np.arange(M0) / M0
    Gs = []
    for v in th:
        ang = base.copy()
        ang[slot] = v
        Gs.append(cell.gate(ang))
    C = np.fft.fft(np.asarray(Gs), axis=0) / M0
    freqs = np.rint(np.fft.fftfreq(M0, d=1.0 / M0)).astype(int)
    amp = np.max(np.abs(C.reshape(M0, -1)), axis=1)
    sup = sorted(int(f) for f, a in zip(freqs, amp) if a > 1e-12)
    if not sup or max(abs(s) for s in sup) > h:
        raise ValueError(f"slot {slot} of {cell_name} has harmonics {sup}, beyond its declared {h}")
    step = int(np.gcd.reduce(np.diff(sup))) if len(sup) > 1 else 1
    return sup[0], sup[-1], max(step, 1)


class ChainEnvironments:
    """Forward and backward caches of the open-chain brickwork at one z, kept in the pool's workers.

    Only the sublayers in ``layers`` (default all) are cached. They are split into
    contiguous blocks, one per replica (at most ``blocks``); worker (g, r) holds a_l and
    b_l of its inputs for the sublayers of block r, made as OpenChainCost makes them.
    release() drops them from the workers; a newer ChainEnvironments on the same pool
    replaces them. Either way this one can no longer make curves.
    """

    def __init__(self, z, cost: FarmCost, blocks: int | None = None, layers=None):
        self.cost, self.pool = cost, cost.pool
        self.z = np.asarray(z, dtype=float).reshape(-1).copy()
        x, e = cost.split(self.z)
        self.x, self.e = x.reshape(-1), e
        self.cell, self.N, self.D = cost.cell, cost.N, cost.D
        want = sorted({int(l) for l in (range(self.D) if layers is None else layers)})
        if not want or want[0] < 0 or want[-1] >= self.D:
            raise ValueError(f"sublayers {want} outside 0..{self.D - 1}")
        nb = max(1, min(int(blocks or self.pool.K), self.pool.K, len(want)))
        self.blocks = [list(map(int, blk)) for blk in np.array_split(np.asarray(want), nb)]
        self.block_of = {l: r for r, blk in enumerate(self.blocks) for l in blk}
        self.key = uuid.uuid4().hex
        t0 = time.perf_counter()
        futs = [[self.pool.submit(g, _cw_env_build, self.key, self.x, self.e, blk, replica=r)
                 for r, blk in enumerate(self.blocks)] for g in range(self.pool.G)]
        self.info = [[f.result() for f in row] for row in futs]
        self.seconds_build = time.perf_counter() - t0
        self.released = False

    def _plan(self, index: int):
        """(layer, slot, offset, step, probes M, probe angles, edge or None) of one angle's curve.

        Bulk index: the sublayer's slot, shared by n_g gates. Edge index nb + 6 m + 3 side + j:
        angle j of the end gate on site N-1 (side 0) or 0 (side 1) of parity-1 sublayer
        2 m + 1, which enters one 2 x 2 gate through cos and sin of theta/2 (harmonics +-1, two
        probes).
        """
        na = self.cell.n_angles
        nb = na * self.D
        n_edge = 6 * (self.D // 2)
        if not 0 <= index < nb + n_edge:
            raise IndexError(f"angle index {index} outside 0..{nb + n_edge - 1}")
        edge = None
        if index < nb:
            layer, slot = divmod(index, na)
            n_g = self.N // 2 if layer % 2 == 0 else self.N // 2 - 1
            lo, hi, step = _slot_modes(self.cell.name, slot)
        else:
            m, r = divmod(index - nb, 6)
            side, j = divmod(r, 3)
            layer, slot = 2 * m + 1, -1
            edge = (side, j, tuple(float(v) for v in self.e[m, 3 * side: 3 * side + 3]))
            n_g, lo, hi, step = 1, -1, 1, 2
        if layer not in self.block_of:
            raise KeyError(f"sublayer {layer} (angle {index}) is not in this environment")
        off = n_g * lo
        M = n_g * (hi - lo) // step + 1
        probes = (4.0 * math.pi / step) * np.arange(M) / M
        return layer, slot, off, step, M, probes, edge

    def curves(self, indices) -> dict[int, ChainCurve]:
        """Curves of several bulk angles, all probes submitted at once."""
        if self.released:
            raise RuntimeError("this environment was released")
        idx = [int(i) for i in indices]
        plans = {i: self._plan(i) for i in idx}
        jobs: dict[int, list] = {}
        order: dict[int, list[int]] = {}
        for i, (layer, slot, _off, _st, _M, probes, edge) in plans.items():
            r = self.block_of[layer]
            jobs.setdefault(r, []).append((layer, slot, np.append(probes, self.z[i]), edge))
            order.setdefault(r, []).append(i)
        futs = {(g, r): self.pool.submit(g, _cw_env_probe, self.key, jobs[r], replica=r)
                for g in range(self.pool.G) for r in jobs}
        samples = {i: np.empty((plans[i][4] + 1, self.pool.n), dtype=complex) for i in idx}
        for (g, r), f in futs.items():
            cols = self.pool.groups[g]
            for i, s in zip(order[r], f.result()):
                samples[i][:, cols] = s
        out = {}
        for i in idx:
            layer, slot, off, step, M, probes, _edge = plans[i]
            s = samples[i]
            coeffs = np.fft.fft(s[:M] * np.exp(-0.5j * off * probes)[:, None], axis=0) / M
            modes = off + step * np.arange(M)
            cv = ChainCurve(i, coeffs, modes, self.N, float(self.z[i]), 0.0)
            scale = max(float(np.max(np.abs(s))), 1e-300)
            cv.residual = float(np.max(np.abs(cv.overlap_at(self.z[i]) - s[M]))) / scale
            if not cv.residual <= RESIDUAL_MAX:
                raise RuntimeError(f"curve of angle {i} misses its check probe by {cv.residual:.1e} "
                                   f"(> {RESIDUAL_MAX:.0e}): the probe count or mode set is wrong")
            out[i] = cv
        return out

    def curve(self, index: int) -> ChainCurve:
        return self.curves([index])[int(index)]

    def set_angle(self, index: int, theta: float) -> None:
        """Change one angle (bulk or edge) here and in the workers' copy of its sublayer."""
        if self.released:
            raise RuntimeError("this environment was released")
        na = self.cell.n_angles
        nb = na * self.D
        self.z[int(index)] = float(theta)
        layer = int(index) // na if index < nb else 2 * ((int(index) - nb) // 6) + 1
        edge6 = self.e[layer // 2].copy() if layer % 2 == 1 and self.e.size else None
        r = self.block_of[layer]
        futs = [self.pool.submit(g, _cw_env_set_layer, self.key, layer, self.x[na * layer: na * layer + na].copy(),
                                 edge6, replica=r) for g in range(self.pool.G)]
        for f in futs:
            f.result()

    def advance(self, layer: int) -> None:
        """Recompute the forward cache after ``layer`` from the sublayer as it is now (one block only)."""
        if layer + 1 not in self.block_of:
            return                                    # the next sublayer is not in this environment
        if self.block_of.get(layer) != self.block_of.get(layer + 1):
            raise ValueError("advance needs both sublayers in one block (blocks=1)")
        r = self.block_of[layer]
        futs = [self.pool.submit(g, _cw_env_advance, self.key, int(layer), replica=r) for g in range(self.pool.G)]
        for f in futs:
            f.result()

    def release(self) -> None:
        """Drop this environment's caches from the workers."""
        if self.released:
            return
        futs = [self.pool.submit(g, _cw_env_clear, self.key, replica=r)
                for g in range(self.pool.G) for r in range(len(self.blocks))]
        for f in futs:
            f.result()
        self.released = True

    def memory_bytes(self) -> int:
        return int(sum(d["bytes"] for row in self.info for d in row))

    def rss_max_mb(self) -> float:
        return float(max(d.get("rss_mb", 0.0) for row in self.info for d in row))


# ============================================================================ problem wiring

POOL_FACTORY = None
"""When set (a callable taking ChainPool's arguments: cell, inputs, targets, chi_max, cutoff and the keywords
groups, replicas, threads, local), ChainProblem builds its pool with it instead of ChainPool: chain_farm sets it
to lease GPU workers or workers on other nodes (cluster/chain_farm.py). None (the default): ChainPool."""


class ChainProblem:
    """Pool, FarmCost, Hessian and environments of one open-chain training problem (a context manager).

    groups x replicas workers (default one input per group; local=True runs them in this
    process). The cost's truncation is (chi_max, cutoff); use the ladder record's
    train_cutoff and train_chi. ``pool`` (or the module's POOL_FACTORY) supplies another
    pool with ChainPool's interface (chain_farm's GPU and multi-node pools); it is closed with
    the problem.
    """

    def __init__(self, model, cell: Cell, t: float, D: int, N: int, inputs, targets, chi_max: int, cutoff: float,
                 groups: int | None = None, replicas: int = 1, env_blocks: int | None = None,
                 fd_step: float = FD_STEP, threads: int = 1, local: bool = False, pool=None):
        if pool is None:
            maker = POOL_FACTORY if (POOL_FACTORY is not None and not local) else ChainPool
            pool = maker(cell, inputs, targets, chi_max, cutoff, groups=groups, replicas=replicas, threads=threads,
                         local=local)
        self.pool = pool
        try:
            self.cost = FarmCost(model, cell, t, D, N, inputs, targets, self.pool)
        except Exception:
            self.pool.close()
            raise
        self.env_blocks, self.fd_step = env_blocks, fd_step
        self.stats = {"hessian": [], "envs": [], "n_grad": 0, "n_value": 0}
        self._count = threading.Lock()

    def objective(self, z):
        with self._count:
            self.stats["n_grad"] += 1
        return self.cost.value_and_grad(z)

    def c_fn(self, z) -> float:
        with self._count:
            self.stats["n_value"] += 1
        return float(self.cost.value(z))

    def hessian(self, z, rows, cols=None) -> np.ndarray:
        """The symmetrized Hessian over ``rows``; with ``cols``, the raw rows over those columns."""
        t0 = time.perf_counter()
        rows = list(rows)
        if cols is None:
            out, asym = fd_hessian(self.cost, z, rows, self.fd_step)
        else:
            out, asym = fd_hessian_rows(self.cost, z, rows, cols, self.fd_step), None
        self.stats["hessian"].append({"rows": len(rows), "seconds": time.perf_counter() - t0, "asym": asym})
        return out

    def envs_at(self, z, layers=None, sweep: bool = False) -> ChainEnvironments:
        env = ChainEnvironments(z, self.cost, 1 if sweep else self.env_blocks, layers)
        self.stats["envs"].append({"seconds": env.seconds_build, "bytes": env.memory_bytes(),
                                   "rss_mb": env.rss_max_mb()})
        return env

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> ChainProblem:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class RingProblem:
    """The ring arm's costs for compile_chain: cli.cmd_compile's, on the 10-site ring.

    jaxcost.make_cost and make_hessian (method "auto": autodiff up to 96 angles, else
    central differences of the jitted gradient) on the production training set
    (circuits.TRAIN_SET), the ring C of cmd_compile's c_fn, and environment.Environments
    for the curves. z is the bulk angles alone (no edges on a ring). hessian(z, rows, cols)
    slices one full Hessian per point, as compile_snap's hessian(x)[free, free].
    """

    def __init__(self, model, cell: Cell, t: float, D: int, N: int = 10, hessian_method: str = "auto"):
        from snapevo.circuits import TRAIN_SET, state_pairs
        from snapevo.jaxcost import make_cost, make_hessian
        self.cell, self.N, self.D = cell, int(N), int(D)
        self.tin, self.tg = state_pairs(model, self.N, t, TRAIN_SET[1], TRAIN_SET[0])
        self._f = make_cost(cell, self.N, self.tin, self.tg, self.D)
        self._H = make_hessian(cell, self.N, self.tin, self.tg, self.D, method=hessian_method, grad=self._f)
        self._last = (None, None)
        self.stats = {"n_grad": 0, "n_value": 0, "n_hessian": 0}

    def objective(self, z):
        self.stats["n_grad"] += 1
        return self._f(z)

    def c_fn(self, z) -> float:
        from snapevo.circuits import local_infidelity
        self.stats["n_value"] += 1
        return local_infidelity(Layered.brickwork(self.cell, z).apply(self.tin, self.N), self.tg, self.N)

    def hessian(self, z, rows, cols=None) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        key = z.tobytes()
        if self._last[0] != key:
            self._last = (key, self._H(z))
            self.stats["n_hessian"] += 1
        H = self._last[1]
        rows = list(rows)
        return H[np.ix_(rows, list(cols) if cols is not None else rows)]

    def envs_at(self, z, layers=None, sweep: bool = False):
        from snapevo.environment import Environments
        return _RingEnv(Environments(np.asarray(z, dtype=float).copy(), self.cell, self.tin, self.tg, self.N))


class _RingEnv:
    """environment.Environments with the sweep methods of ChainEnvironments (set_angle, advance)."""

    def __init__(self, env):
        self.env = env

    def curves(self, indices):
        return self.env.curves(indices)

    def curve(self, index: int):
        return self.env.curve(index)

    def set_angle(self, index: int, theta: float) -> None:
        self.env.x[int(index)] = float(theta)

    def advance(self, layer: int) -> None:
        e, na = self.env, self.env.cell.n_angles
        s = e._apply(e._pre[layer], e.cell.gate(e.x[na * layer: na * layer + na]), layer)
        if layer + 1 < e.D:
            e._pre[layer + 1] = s
        else:
            e._out = s


class _Stopper:
    """Deadline, SIGTERM (main thread only) and an external stop event, as one test."""

    def __init__(self, deadline: float | None, stop: threading.Event | None):
        self.deadline, self.stop = deadline, stop
        self.term = threading.Event()
        self._old = None
        self._installed = False

    def __enter__(self) -> _Stopper:
        if threading.current_thread() is threading.main_thread():
            try:
                self._old = signal.signal(signal.SIGTERM, lambda _s, _f: self.term.set())
                self._installed = True
            except ValueError:                                   # not the main interpreter thread
                pass
        return self

    def __exit__(self, *exc) -> None:
        if self._installed:
            signal.signal(signal.SIGTERM, signal.SIG_DFL if self._old is None else self._old)
            self._installed = False

    def __call__(self) -> bool:
        return (self.term.is_set() or (self.stop is not None and self.stop.is_set())
                or (self.deadline is not None and time.time() > self.deadline))

    def why(self) -> str:
        if self.term.is_set():
            return "SIGTERM"
        if self.stop is not None and self.stop.is_set():
            return "stop event"
        return "deadline"


def polish_chain(z, objective, hessian, tol: float, n_bulk: int | None = None, log=None,
                 checkpoint: str | None = None, deadline: float | None = None,
                 stop: threading.Event | None = None) -> tuple[np.ndarray, dict]:
    """chain_point's polish of a trained z before compiling: newton_minimize to ``tol`` (POLISH_TOL_REL x budget)."""
    return newton_minimize(z, objective, hessian, tol, n_bulk=n_bulk, log=log, checkpoint=checkpoint,
                           deadline=deadline, stop=stop, label="polish")


def newton_minimize(z, objective, hessian, tol, n_bulk: int | None = None, log=None,
                    checkpoint: str | None = None, deadline: float | None = None,
                    stop: threading.Event | None = None, label: str = "newton") -> tuple[np.ndarray, dict]:
    """Trust-region Newton steps on the full Hessian until converged, with no step cap (the polish and the ladders).

    scipy's trust-exact on the full (indefinite) Hessian, a fresh Hessian at every accepted
    point, as train.py polishes the ring ladders, but with no step cap: it stops when the
    bulk block of the Hessian (the first n_bulk parameters) has no negative curvature beyond
    1e-8 of its largest eigenvalue and both the predicted Newton decrease g.H^+.g/2 (H
    PSD-projected) and the last step's decrease are below ``tol`` (a number, or a function of
    the current c: the ladders use 1e-6 c), "converged"; or when trust-exact stops because
    its model no longer predicts a decrease, "noise floor". One Hessian costs 2 len(z) chain
    gradients.
    The result is kept only if it lowers c. Returns (z, info) with c and the Hessian's lowest
    eigenvalue and negative count (all of z and the bulk block) before and after, steps,
    Hessians, the final predicted decrease and the stop reason. ``checkpoint`` (a JSON file)
    saves z after every step; a rerun continues from it (its trust radius starts afresh). At
    the deadline, on SIGTERM or on ``stop`` it saves and raises DeadlineReached.
    """
    say = log or (lambda _m: None)
    z0 = np.asarray(z, dtype=float).reshape(-1)
    nb = z0.size if n_bulk is None else int(n_bulk)
    rows = list(range(z0.size))
    tol_at = tol if callable(tol) else (lambda _c, v=float(tol): v)
    start, done, info = z0.copy(), 0, {"tol": None if callable(tol) else float(tol), "hessians": 0}
    if checkpoint and os.path.exists(checkpoint):
        ck = json.loads(open(checkpoint).read())
        if np.asarray(ck["z_in"]).size != z0.size or not np.array_equal(np.asarray(ck["z_in"]), z0):
            raise ValueError(f"{checkpoint} is for another start point")
        start, done, info = np.asarray(ck["z"], dtype=float), int(ck["steps"]), ck["info"]
        info["tol"] = None if callable(tol) else float(tol)
        say(f"{label}: resumed {os.path.basename(checkpoint)} after {done} steps")
    t0 = time.time() - info.get("seconds", 0.0)
    cache_f: OrderedDict = OrderedDict()
    last: dict = {"x": None, "H": None}

    def pred_of(H, g) -> float:
        w, V = np.linalg.eigh(0.5 * (H + H.T))
        bb = V.T @ g
        keep = w > 1e-12 * max(float(np.abs(w).max()), 1e-300)
        return 0.5 * float(np.sum(bb[keep] ** 2 / w[keep]))

    with _Stopper(deadline, stop) as stopping:
        def save(x, c) -> None:
            if checkpoint:
                tmp = f"{checkpoint}.{os.getpid()}.tmp"
                with open(tmp, "w") as fh:
                    json.dump({"z_in": z0.tolist(), "z": np.asarray(x, dtype=float).tolist(), "steps": done, "c": c,
                               "info": {**info, "seconds": time.time() - t0}}, fh)
                os.replace(tmp, checkpoint)

        def fun(x):
            key = np.asarray(x, dtype=float).tobytes()
            if key in cache_f:
                return cache_f[key]
            if stopping():
                raise _Stop
            v = objective(x)
            cache_f[key] = v
            while len(cache_f) > 4:
                cache_f.popitem(last=False)
            return v

        def hess(x):
            if last["x"] is not None and np.array_equal(last["x"], x):
                return last["H"]
            if stopping():
                raise _Stop
            H = np.asarray(hessian(x, rows))
            info["hessians"] += 1
            last.update(x=np.array(x, dtype=float), H=H)
            return H

        state = {"x": start, "c": None}

        def converged(x, last_dec) -> bool:
            c, g = fun(x)
            H = hess(x)
            st_ = _eig_stats(H, nb)
            info["pred"] = pred_of(H, np.asarray(g, dtype=float))
            tl = tol_at(float(c))
            info["tol_last"] = tl
            flat_ok = st_["eig_min_bulk"] > -1e-8 * max(abs(st_["eig_max"]), 1e-300)
            return flat_ok and info["pred"] < tl and (last_dec is None or last_dec < tl)

        def cb(intermediate_result):
            nonlocal done
            done += 1
            x_prev, c_prev = state["x"], (info["c_before"] if state["c"] is None else state["c"])
            state.update(x=np.array(intermediate_result.x, dtype=float), c=float(intermediate_result.fun))
            save(state["x"], state["c"])
            if not np.array_equal(x_prev, state["x"]) and converged(state["x"], c_prev - state["c"]):
                info["stop"] = "converged"
                raise StopIteration

        try:
            res = None
            if "c_before" not in info:
                info["c_before"] = float(fun(z0)[0])
                info.update({f"{k}_before": v for k, v in _eig_stats(hess(z0), nb).items()})
                say(f"{label}: c = {info['c_before']:.6e}, lowest eigenvalue {info['eig_min_before']:.2e} "
                    f"({info['n_neg_before']} negative; bulk {info['eig_min_bulk_before']:.2e}, "
                    f"{info['n_neg_bulk_before']} negative), tolerance "
                    f"{tol_at(info['c_before']):.1e}{' (relative to c)' if callable(tol) else ''}")
                save(start, info["c_before"])
            if converged(start, None):
                info["stop"] = "converged"
            else:
                info["stop"] = None
                res = sopt.minimize(fun, start, jac=True, hess=hess, method="trust-exact", callback=cb,
                                    options={"maxiter": 10 ** 9, "gtol": 0.0})
                if info["stop"] is None:
                    info["stop"] = "noise floor" if "predict improvement" in str(res.message) else str(res.message)
            c_new = float(res.fun) if res is not None else float(fun(start)[0])
            x_new = np.asarray(res.x if res is not None else start, dtype=float)
            better = c_new < info["c_before"]
            out = x_new if better else z0
            info.update({f"{k}_after": v for k, v in _eig_stats(hess(out), nb).items()})
        except _Stop:
            save(state["x"], state["c"])
            raise DeadlineReached(f"{label} stopped ({stopping.why()}) after {done} steps; "
                                  f"checkpoint {checkpoint}") from None
    info.update(c_after=c_new if better else info["c_before"], kept=bool(better), steps=done,
                message=None if res is None else str(res.message), seconds=time.time() - t0)
    say(f"{label}: {done} trust-region Newton steps ({info['stop']}), c {info['c_before']:.6e} -> "
        f"{info['c_after']:.6e}{'' if better else ' (kept the start)'}, lowest eigenvalue "
        f"{info['eig_min_before']:.2e} -> {info['eig_min_after']:.2e} (bulk {info['eig_min_bulk_before']:.2e} -> "
        f"{info['eig_min_bulk_after']:.2e}, {info['n_neg_bulk_after']} negative), predicted decrease "
        f"{info.get('pred', 0.0):.1e}, {info['hessians']} Hessians, {info['seconds']:.0f}s")
    return out, info


# ============================================================================ the compiler

@dataclass
class _ChainState:
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
    n_trials_run: int = 0
    timing: dict = field(default_factory=dict)
    round: dict | None = None
    hstats: list = field(default_factory=list)
    h_cur: tuple | None = None
    reopt_log: list = field(default_factory=list)


_BASIS: dict = {}


def _curve_min(cv, x_i: float) -> tuple[float, float, float]:
    """(angle, c there, c at x_i): the global minimum of one exact curve over the 4 pi period around x_i."""
    top = max(abs(float(np.max(np.real(cv.modes)))), abs(float(np.min(np.real(cv.modes)))), 1.0)
    n_grid = max(COORD_GRID, int(32 * top))
    th = x_i + 4.0 * math.pi * (np.arange(n_grid) / n_grid - 0.5)
    v = np.asarray(cv.c_at(th), dtype=float)
    loc = np.flatnonzero((v <= np.roll(v, 1)) & (v <= np.roll(v, -1)))
    loc = loc[np.argsort(v[loc], kind="stable")][:COORD_REFINE] if loc.size else np.array([int(np.argmin(v))])
    h = 4.0 * math.pi / n_grid
    c0 = float(cv.c_at(x_i))
    best = (x_i, c0)
    for k in loc:
        r = sopt.minimize_scalar(lambda u: float(cv.c_at(u)), bounds=(th[k] - h, th[k] + h), method="bounded",
                                 options={"xatol": 1e-12, "maxiter": 10 ** 9})
        for u, cu in ((float(r.x), float(r.fun)), (float(th[k]), float(v[k]))):
            if cu < best[1]:
                best = (u, cu)
    return best[0], best[1], c0


def _admissible_mask(curve, c_max: float, S: np.ndarray) -> np.ndarray | None:
    """snap._admissible_mask with its grid basis kept per mode set (same arithmetic)."""
    if not np.isfinite(c_max):
        return None
    c = np.asarray(curve.coeffs)
    m = 0.5 * np.asarray(curve.modes, dtype=float)
    n = c.shape[1]
    thr = 1.0 - curve.N * n * max(c_max, 0.0) * (1.0 + 1e-9) - 1e-12
    if thr <= 0.5:
        return None
    h = 2.0 * math.pi / _GRID
    key = m.tobytes()
    if key not in _BASIS:
        if len(_BASIS) > 8:
            _BASIS.clear()
        th = np.arange(_GRID + 1) * h
        _BASIS[key] = np.exp(1j * th[:, None] * m[None, :])
    basis = _BASIS[key]
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


def compile_chain(z_star, cell: Cell, D: int, settings: SnapSettings, objective, c_fn, hessian, envs_at,
                  library: Library, log=None, trial_workers: int = 1, deadline: float | None = None,
                  hessian_chunk: int | None = None, stop: threading.Event | None = None,
                  env_layers: int | None = None, reoptimizer=None) -> dict:
    """Run snap.compile_snap's protocol and return the compiled circuit as a JSON-ready dict.

    The open chain (ChainProblem) and the ring (RingProblem) run this same code; on the
    ring z has no edge angles.

    Args:
        z_star: Trained z: (bulk, edges), D n_angles + 6 (D // 2) angles (train_chain), or the
            D n_angles bulk angles alone (the ring).
        settings: ChainSettings (a plain SnapSettings takes ChainSettings' defaults).
        objective: ``z -> (c, dc/dz)`` on the training inputs; thread-safe when trial_workers > 1.
        c_fn: ``z -> c`` on the training inputs.
        hessian: ``(z, rows, cols) -> (len(rows), len(cols))`` rows of the Hessian of c
            (ChainProblem.hessian: central differences; RingProblem.hessian: jaxcost's);
            compile_chain symmetrizes them.
        envs_at: ``(z, layers) -> object with curves(indices) -> {index: curve}`` for the
            indices of those sublayers (edge indices too on the chain), and optionally release().
        trial_workers: Trials of one round evaluated at the same time (module docstring, 5).
        deadline: Epoch seconds. Past it, on SIGTERM or when ``stop`` is set, the compile
            writes its checkpoint and raises DeadlineReached (module docstring, 6).
        hessian_chunk: Hessian rows per call (default all at once); between chunks the
            rows done are kept in the checkpoint and the deadline is checked.
        env_layers: At most this many sublayers per environment (default all): the curves are
            made in batches of sublayers, each with its own forward and backward pass, which
            bounds the workers' cache memory (the curves are the same numbers).
        reoptimizer: None (settings.reopt chooses). Tests pass a function
            ``(f, x0, lower, upper) -> (x, c)`` here to run compile_snap's own re-optimization
            and check the shared code against the production compiles; nothing else does.
    """
    say = log or (lambda _m: None)
    z_star = np.asarray(z_star, dtype=float).reshape(-1)
    n = z_star.size
    na = cell.n_angles
    nb = D * na
    if n == nb:
        edge_idx: list[int] = []
    elif n == nb + 6 * (D // 2):
        edge_idx = list(range(nb, n))
    else:
        raise ValueError(f"z has {n} entries, expected {nb} or {nb + 6 * (D // 2)} for D = {D}")
    rcond = float(getattr(settings, "response_rcond", RESPONSE_RCOND))
    rule = getattr(settings, "reopt", None) if reoptimizer is None else "custom"
    if rule is None:
        raise ValueError(f"settings.reopt is not set; choose a re-optimizer from {REOPTIMIZERS} (there is no default)")
    if rule not in REOPTIMIZERS + ("custom",):
        raise ValueError(f"unknown re-optimizer {rule!r}; choose from {REOPTIMIZERS}")
    if rule == "coordinate" and int(trial_workers) > 1:
        raise ValueError("the coordinate re-optimizer keeps one environment in the workers: use trial_workers = 1")
    win_l = int(getattr(settings, "lbfgs_window", 10))
    tol_l = float(getattr(settings, "lbfgs_tol_rel", 1e-4)) * settings.budget
    edge_w = float(getattr(settings, "edge_weight", 0.0)) if edge_idx else 0.0
    edge_zt = bool(getattr(settings, "edge_zero_t", False)) and bool(edge_idx)
    need_edges = edge_w > 0.0 or edge_zt
    tol = float(getattr(settings, "newton_tol_rel", 1e-6)) * settings.budget
    n_every = int(getattr(settings, "newton_refresh_every", 20))
    weights = np.zeros(n)
    weights[:nb] = [cell.multiplicity[i % na] for i in range(nb)]
    weights[edge_idx] = edge_w
    B = max(1, int(trial_workers))
    A_GEN, B_GEN = kn_law()
    S, C = library.S, library.C
    S0 = S[C == 0.0]
    say(f"snap: {len(S)} library angles (F = {library.F:g}), {len(S0)} free, "
        f"{len(np.unique(C))} cost levels, budget {settings.budget:.3e}; {nb} bulk and "
        f"{len(edge_idx)} edge angles (end-game weight {edge_w:g}{', zero-T' if edge_zt else ''}), "
        f"re-optimizer {rule}, {B} trial(s) at a time")

    st = _ChainState(x=z_star.copy(), snapped=np.zeros(n, dtype=bool), costs=np.zeros(n),
                     lower=np.full(n, -np.inf), upper=np.full(n, np.inf))
    eps = settings.budget
    resumed = None
    if settings.checkpoint and os.path.exists(settings.checkpoint):
        with open(settings.checkpoint, "rb") as fh:
            resumed = pickle.load(fh)
        st = resumed["state"]
        for name, default in (("round", None), ("hstats", []), ("timing", {}), ("h_cur", None), ("reopt_log", [])):
            if not hasattr(st, name):
                setattr(st, name, default)
        if st.x.size != n:
            raise ValueError(f"{settings.checkpoint} is for {st.x.size} parameters, not {n}")
        say(f"snap: resumed {os.path.basename(settings.checkpoint)} ({st.n_zero} zero-T, "
            f"{st.n_hess} Hessian, {int((~st.snapped[:nb]).sum())} free, {resumed['elapsed']:.0f}s spent"
            + (f", inside round {st.n_rounds}" if st.round is not None else "") + ")")
    last_ckpt = [time.time()]
    t0 = time.time()
    floor = None
    stopper = _Stopper(deadline, stop)

    def tick(key: str, since: float) -> None:
        st.timing[key] = st.timing.get(key, 0.0) + time.time() - since

    def bump(key: str, by: float = 1) -> None:
        st.timing[key] = st.timing.get(key, 0) + by

    def checkpoint(elapsed: float, floor: float) -> None:
        if not settings.checkpoint or floor is None:
            return
        tmp = settings.checkpoint + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"state": st, "elapsed": elapsed, "floor": floor}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, settings.checkpoint)
        last_ckpt[0] = time.time()

    def checkpoint_if_due() -> None:
        late = stopper()
        if late or time.time() - last_ckpt[0] >= settings.checkpoint_every:
            checkpoint(time.time() - t0, floor)
        if late:
            raise DeadlineReached(f"stopped ({stopper.why()}); checkpoint {settings.checkpoint}")

    def cval(z) -> float:
        bump("n_value")
        return c_fn(z)

    def layer_of(i: int) -> int:
        return i // na if i < nb else 2 * ((i - nb) // 6) + 1

    # ------------------------------------------------------------- re-optimizers
    count_lock = threading.Lock()

    def lbfgs_converged(f, x0, lower, upper):
        """The comparison mode: L-BFGS-B to the budget-relative convergence test, no step cap."""
        hist: list = []

        def g(z):
            v = f(z)
            if not hist:
                hist.append(float(v[0]))                       # L-BFGS-B evaluates x0 first
            return v

        def cb(intermediate_result):
            hist.append(float(intermediate_result.fun))
            if len(hist) > win_l and hist[-win_l - 1] - hist[-1] < tol_l:
                raise StopIteration
        res = sopt.minimize(g, x0, bounds=list(zip(lower, upper)), callback=cb, jac=True, method="L-BFGS-B",
                            options={**_REOPT["options"], "maxiter": 10 ** 9, "maxfun": 10 ** 9})
        stop = "converged" if (len(hist) > win_l and hist[-win_l - 1] - hist[-1] < tol_l) else str(res.message)
        return np.asarray(res.x, dtype=float), float(res.fun), int(getattr(res, "nit", 0)), stop

    def tr_step(w, V, gF, delta):
        """Minimizer of g.p + p.H.p/2 over |p| <= delta for H = V diag(w) V^T, w >= 0."""
        b = V.T @ gF
        wmax = max(float(w.max()), 1e-300) if w.size else 1.0

        def p_of(lam):
            return -(V @ (b / (w + lam)))
        if w.size and float(w.min()) > 1e-14 * wmax:
            p = p_of(0.0)
            if np.linalg.norm(p) <= delta:
                return p
        lo, hi = 1e-18 * max(wmax, 1.0), float(np.linalg.norm(gF)) / delta + wmax
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if np.linalg.norm(p_of(mid)) > delta:
                lo = mid
            else:
                hi = mid
            if hi / lo < 1.0 + 1e-9:
                break
        return p_of(hi)

    def psd_rows(x, F):
        """The PSD-projected Hessian over F at x (a refresh); 2 len(F) gradients on the chain."""
        R = np.asarray(hessian(x, list(F), cols=list(F)), dtype=float)
        bump("newton_hess_grads", 2 * len(F))
        return _psd(_symmetrize(R)[0])

    def newton(f, x0, F, Hf):
        """Trust-region Newton steps on the parameters F from x0 with a given Hessian, refreshed on failure.

        No step cap: it stops when the predicted Newton decrease g.H^+.g/2 on F and the last
        accepted step's decrease are both below newton_tol_rel x budget ("converged"), or when
        the trust radius has collapsed below NEWTON_DELTA_MIN after a refresh ("noise floor").
        """
        x = np.array(x0, dtype=float)
        F = np.asarray(F, dtype=int)
        info = {"path": "newton", "steps": 0, "rejects": 0, "refreshes": 0, "stop": None, "pred": None, "H": None}
        c, g = f(x)
        c = float(c)
        if F.size == 0:
            info.update(stop="nothing free", pred=0.0)
            return x, c, info

        def eig(H):
            w, V = np.linalg.eigh(H)
            w = np.clip(w, 0.0, None)
            keep = w > 1e-12 * max(float(w.max()) if w.size else 0.0, 1e-300)
            return w, V, keep

        if Hf is None:
            Hf = psd_rows(x, F)
            info["refreshes"] += 1
            info["H"] = Hf
        w, V, keep = eig(Hf)
        delta, since, last_dec, refreshed = NEWTON_DELTA0, 0, None, False
        while True:
            gF = np.asarray(g, dtype=float)[F]
            b = V.T @ gF
            pred_newton = 0.5 * float(np.sum(b[keep] ** 2 / w[keep]))
            info["pred"] = pred_newton
            if pred_newton < tol and (last_dec is None or last_dec < tol):
                info["stop"] = "converged"
                break
            if delta < NEWTON_DELTA_MIN:
                if refreshed:
                    info["stop"] = "noise floor"
                    break
                Hf = psd_rows(x, F)                              # recompute here before giving up
                w, V, keep = eig(Hf)
                info["refreshes"] += 1
                info["H"] = Hf
                refreshed, since = True, 0
                continue
            p = tr_step(w, V, gF, delta)
            pred = -float(gF @ p + 0.5 * p @ (Hf @ p))
            xt = x.copy()
            xt[F] += p
            ct, gt = f(xt)
            ct = float(ct)
            if ct < c and pred > 0.0:
                rho = (c - ct) / pred
                last_dec = c - ct
                x, c, g = xt, ct, gt
                info["steps"] += 1
                since += 1
                refreshed = False
                if rho > 0.75 and np.linalg.norm(p) > 0.99 * delta:
                    delta = min(2.0 * delta, NEWTON_DELTA_MAX)
                elif rho < 0.25:
                    delta *= 0.25
                if since >= n_every:
                    Hf = psd_rows(x, F)
                    w, V, keep = eig(Hf)
                    info["refreshes"] += 1
                    info["H"] = Hf
                    refreshed, since = True, 0
            else:
                # a step that does not lower c: recompute the Hessian here (once per run of failures)
                # and shrink the trust region
                info["rejects"] += 1
                delta *= 0.25
                if not refreshed:
                    Hf = psd_rows(x, F)
                    w, V, keep = eig(Hf)
                    info["refreshes"] += 1
                    info["H"] = Hf
                    refreshed, since = True, 0
        return x, c, info

    def coordinate(f, x0, F, abort=None):
        """Sweeps over F in circuit order, each angle set to the global minimum of its exact curve.

        Each sweep builds environments at its start point and runs through the sublayers in
        order: the curve of each free angle (bulk or edge) of a sublayer is made with the
        angles changed so far, the angle goes to the curve's global minimum when that is
        lower than its current value, and the forward cache is carried past the sublayer
        (the backward caches of later sublayers are unchanged within a sweep). No sweep cap:
        it stops when a sweep lowers c by less than newton_tol_rel x budget ("converged"), or
        when a sweep raises the exact c (truncation; the sweep is undone, "noise floor").
        """
        x = np.array(x0, dtype=float)
        F = sorted(int(i) for i in F)
        info = {"path": "coordinate", "sweeps": 0, "curves": 0, "moves": 0, "stop": None, "gain": None,
                "env_seconds": 0.0}
        c = float(f(x)[0])
        if not F:
            info.update(stop="nothing free", gain=0.0)
            return x, c, info
        by_layer: dict[int, list[int]] = {}
        for i in F:
            by_layer.setdefault(layer_of(i), []).append(i)
        first, last = min(by_layer), max(by_layer)
        span = last - first + 1
        step = span if not env_layers else max(1, int(env_layers))
        while True:
            x_prev, c_prev = x.copy(), c
            for lo in range(first, last + 1, step):
                hi = min(lo + step, last + 1)
                s0 = time.time()
                env = envs_at(x.copy(), list(range(lo, hi)), sweep=True)
                info["env_seconds"] += time.time() - s0
                try:
                    for l in range(lo, hi):
                        if abort is not None and abort.is_set():
                            raise _Abort
                        if stopper():
                            raise _Stop
                        for i in by_layer.get(l, []):
                            cv = env.curve(i)
                            info["curves"] += 1
                            u, cu, c0 = _curve_min(cv, float(x[i]))
                            if cu < c0:
                                x[i] = u
                                env.set_angle(i, u)
                                info["moves"] += 1
                        if l + 1 < hi:
                            env.advance(l)
                finally:
                    if hasattr(env, "release"):
                        env.release()
            info["sweeps"] += 1
            c = float(f(x)[0])
            gain = c_prev - c
            info["gain"] = gain
            if gain < 0.0:
                x, c = x_prev, c_prev
                info["stop"] = "noise floor"
                break
            if gain < tol:
                info["stop"] = "converged"
                break
        return x, c, info

    def restrict(hc, F):
        if hc is None:
            return None
        idx, H = hc
        pos = {int(k): a for a, k in enumerate(idx)}
        if any(int(f) not in pos for f in F):
            return None
        p = [pos[int(f)] for f in F]
        return H[np.ix_(p, p)]

    def run_reopt(x0, lower, upper, abort=None, kind="reopt", F=None, Hf=None):
        """One re-optimization by the chosen rule. Returns (x, c, the Hessian of a Newton refresh or None)."""
        calls = [0]

        def f(z):
            if abort is not None and abort.is_set():
                raise _Abort
            if stopper():
                raise _Stop
            calls[0] += 1
            return objective(z)
        s0 = time.time()
        entry: dict = {"kind": kind}
        try:
            if rule == "newton":
                x, c, info = newton(f, x0, [] if F is None else F, Hf)
                newH = info.pop("H")
                entry.update(info)
            elif rule == "lbfgs-converged":
                x, c, nit, why = lbfgs_converged(f, x0, lower, upper)
                newH = None
                entry.update(path="lbfgs-converged", nit=nit, stop=why)
            elif rule == "coordinate":
                x, c, info = coordinate(f, x0, [] if F is None else F, abort)
                newH = None
                entry.update(info)
            else:                                                # tests only (module docstring)
                x, c = reoptimizer(f, x0, lower, upper)
                x, c, newH = np.asarray(x, dtype=float), float(c), None
                entry.update(path=getattr(reoptimizer, "__name__", "custom"))
        finally:
            with count_lock:
                st.timing[f"n_{kind}"] = st.timing.get(f"n_{kind}", 0) + 1
                st.timing[f"calls_{kind}"] = st.timing.get(f"calls_{kind}", 0) + calls[0]
        entry.update(evals=calls[0], c=c, seconds=time.time() - s0)
        with count_lock:
            st.reopt_log.append(entry)
            if entry["path"] == "newton":
                bump("newton_refreshes", entry.get("refreshes", 0))
                bump("newton_steps", entry.get("steps", 0))
                bump("newton_rejects", entry.get("rejects", 0))
            elif entry["path"] == "coordinate":
                bump("coord_sweeps", entry.get("sweeps", 0))
                bump("coord_curves", entry.get("curves", 0))
                bump("coord_env_seconds", entry.get("env_seconds", 0.0))
            else:
                bump(f"iters_{kind}", entry.get("nit", 0))
        return x, c, newH

    def free_now() -> list[int]:
        return [int(i) for i in np.flatnonzero(~st.snapped)]

    def reopt(kind: str = "reopt") -> float:
        """Re-optimize the free parameters in place; an interrupted one leaves the state as it was."""
        if st.snapped.all():
            return cval(st.x)
        s0 = time.time()
        F = free_now()
        x, fun, newH = run_reopt(st.x, st.lower, st.upper, kind=kind, F=F, Hf=restrict(st.h_cur, F))
        st.x[:] = x
        if newH is not None:
            st.h_cur = (F, newH)
        tick("reopt", s0)
        return fun

    def pin(i: int, angle: float) -> None:
        st.x[i] = angle
        st.lower[i] = st.upper[i] = angle
        st.snapped[i] = True

    def unpin(i: int, saved: np.ndarray) -> None:
        st.x[:] = saved
        st.lower[i], st.upper[i] = -np.inf, np.inf
        st.snapped[i] = False

    def free_indices() -> list[int]:
        return [int(i) for i in np.flatnonzero(~st.snapped[:nb])]

    def free_edges() -> list[int]:
        return [int(i) for i in edge_idx if not st.snapped[i]]

    curve_cache: OrderedDict = OrderedDict()

    def curves_at(x, indices) -> dict:
        """Curves at exactly x, cached by the bytes of x; environments only for the sublayers needed, then released."""
        key = np.asarray(x, dtype=float).tobytes()
        have = curve_cache.get(key, {})
        missing = [int(i) for i in indices if int(i) not in have]
        if missing:
            s0 = time.time()
            layers = sorted({layer_of(i) for i in missing})
            step = len(layers) if not env_layers else max(1, int(env_layers))
            for lo in range(0, len(layers), step):
                group = set(layers[lo: lo + step])
                env = envs_at(np.asarray(x, dtype=float).copy(), sorted(group))
                try:
                    have.update(env.curves([i for i in missing if layer_of(i) in group]))
                finally:
                    if hasattr(env, "release"):
                        env.release()
                if hasattr(env, "memory_bytes"):
                    st.timing["env_bytes_max"] = max(st.timing.get("env_bytes_max", 0), env.memory_bytes())
                if hasattr(env, "rss_max_mb"):
                    st.timing["worker_rss_mb_max"] = max(st.timing.get("worker_rss_mb_max", 0.0), env.rss_max_mb())
            bump("n_curve_sets")
            tick("curves", s0)
        curve_cache[key] = have
        curve_cache.move_to_end(key)
        while len(curve_cache) > 4:
            curve_cache.popitem(last=False)
        return {int(i): have[int(i)] for i in indices}

    e_res2 = (2.0 ** (-(settings.reserve_t - B_GEN) / A_GEN)) ** 2 if settings.reserve_t > 0 else 0.0

    def share_for(eps0: float, n_free: int, relax: float = 1.0) -> float:
        if settings.share_cap <= 0 or not np.isfinite(relax):
            return float("inf")
        return relax * settings.share_cap * max(eps - eps0, 0.0) / max(n_free, 1)

    def reserve_for(curv: dict, exclude: int) -> float:
        if e_res2 == 0.0:
            return 0.0
        need = sum(cv * e_res2 for i, cv in curv.items() if i != exclude)
        return min(need, settings.reserve_frac * eps * max(len(curv) - 1, 0) / nb)

    # ------------------------------------------------------------- level tables
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

    # ------------------------------------------------------------- end-game pieces
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

    def rest_of(snapped) -> list[int]:
        rest = [int(i) for i in np.flatnonzero(~snapped[:nb])]
        if edge_w > 0.0:
            rest += [int(i) for i in edge_idx if not snapped[i]]
        return rest

    def straggler_costs(x, snapped, c_now):
        rest = rest_of(snapped)
        if not rest:
            return {}
        curves = curves_at(x, rest)
        residual = max(eps - c_now, 0.0)
        if settings.library_stragglers:
            return library_allocation(rest, curves, residual, c_now)
        return synthesis_allocation(curves, rest, residual)

    def total_teq(costs, alloc):
        paid = costs.copy()
        for i, e in alloc.items():
            paid[i] = e[1]
        return 0.5 * float((weights * paid).sum())

    def eager_endgame() -> None:
        """The end game's cost of the newest prefix, while the curves at its point are at hand."""
        k = len(st.history) - 1
        if k < 0 or k in st.eg_teq:
            return
        xk, snk, ck, c_k = st.history[k]
        if np.array_equal(xk, st.x) and np.array_equal(snk, st.snapped):
            s0 = time.time()
            st.eg_teq[k] = total_teq(ck, straggler_costs(xk, snk, c_k))
            tick("endgame", s0)

    # ------------------------------------------------------------- zero-T stage
    def zero_t_stage() -> int:
        total = 0
        while True:
            free = free_indices()
            fe = free_edges() if need_edges else []
            if not free and not (edge_zt and fe):
                return total
            curves = curves_at(st.x, free + fe)
            eager_endgame()
            curv = {i: curves[i].curvature() for i in free}
            best = None
            for i in free + (fe if edge_zt else []):
                ec = np.asarray(curves[i].c_at(S0))
                j = int(np.argmin(ec))
                if ec[j] <= eps - reserve_for(curv, i) and (best is None or ec[j] < best[0]):
                    best = (float(ec[j]), i, float(S0[j]))
            if best is None:
                return total
            ec, i, ang = best
            saved = st.x.copy()
            c_before = cval(st.x)
            pin(i, _unwrap(st.x[i], ang))
            if i >= nb:
                c_after = cval(st.x)                      # an edge: the exact c of this one change
            else:
                try:
                    c_after = reopt()
                except _Stop:
                    unpin(i, saved)
                    raise
            if c_after > eps - reserve_for(curv, i) or c_after - c_before > share_for(c_before, len(free)):
                unpin(i, saved)
                return total
            st.costs[i] = 0.0
            st.n_zero += 1
            st.trace.append({"i": i, "stage": "zero", "cost": 0.0, "c_pre": ec, "c_post": c_after})
            st.history.append((st.x.copy(), st.snapped.copy(), st.costs.copy(), c_after))
            total += 1
            checkpoint_if_due()

    # ------------------------------------------------------------- Hessian stage
    pool_ex = cf.ThreadPoolExecutor(B) if B > 1 else None

    def run_batch(batch, saved, hidx, resp, H):
        """Trials of one batch from the saved state, in rank order; later ones aborted once one passes.

        An exception in any trial (the deadline included) aborts the others and is raised.
        """
        aborts = [threading.Event() for _ in batch]
        specs = []
        for (a, i, d, *_rest), ab in zip(batch, aborts):
            x0 = saved.copy()
            x0[i] = saved[i] + d
            others = [f for f in hidx if f != i]
            x0[others] += d * resp[a]
            lower, upper = st.lower.copy(), st.upper.copy()
            lower[i] = upper[i] = x0[i]
            keep = [k for k in range(len(hidx)) if k != a]
            specs.append((x0, lower, upper, ab, "trial", others, H[np.ix_(keep, keep)] if H is not None else None))
        if not specs[0][5]:
            # the trial pins the last free parameter: compile_snap's reopt() returns c_fn there
            out = []
            for sp, (_a, _i, _d, ceil, *_r) in zip(specs, batch):
                st.n_trials_run += 1
                out.append((sp[0], cval(sp[0])))
                if out[-1][1] <= ceil:
                    break
            return out
        if pool_ex is None:
            out = []
            for sp, (_a, _i, _d, ceil, *_r) in zip(specs, batch):
                st.n_trials_run += 1
                out.append(run_reopt(*sp)[:2])
                if out[-1][1] <= ceil:
                    break
            return out
        futs = [pool_ex.submit(run_reopt, *sp) for sp in specs]
        st.n_trials_run += len(futs)
        out, err = [], None
        for j, (fu, (_a, _i, _d, ceil, *_r)) in enumerate(zip(futs, batch)):
            try:
                r = fu.result()[:2]
            except BaseException as exc:                          # noqa: BLE001 (re-raised below)
                err = exc
                break
            out.append(r)
            if r[1] <= ceil:
                for ab in aborts[j + 1:]:
                    ab.set()
                for g in futs[j + 1:]:
                    try:
                        g.result()
                    except _Abort:
                        pass
                return out
        if err is not None:
            for ab in aborts:
                ab.set()
            for g in futs:
                try:
                    g.result()
                except BaseException:                             # noqa: BLE001 (the first one is raised)
                    pass
            raise err
        return out

    def round_hessian(rnd) -> tuple[np.ndarray, float]:
        """Rows of the round's Hessian, in chunks kept in the round state; symmetrized."""
        hidx, R = rnd["hidx"], rnd["R"]
        todo = [k for k in hidx if k not in R]
        size = len(todo) if not hessian_chunk else max(1, int(hessian_chunk))
        s0 = time.time()
        for lo in range(0, len(todo), max(size, 1)):
            if stopper():
                raise _Stop
            part = todo[lo: lo + size]
            rows = np.asarray(hessian(rnd["x"], part, cols=hidx), dtype=float)
            for k, row in zip(part, rows):
                R[k] = row
            bump("hess_grads", 2 * len(part))
            tick("hessian", s0)
            s0 = time.time()
            checkpoint_if_due()
        return _symmetrize(np.array([R[k] for k in hidx]).reshape(len(hidx), len(hidx)))

    def plan_round(rnd, relax: float) -> bool:
        """Hessian, response, curves, candidates and the trial plan of a round, into rnd. False: no candidate."""
        free, hidx, eps0 = rnd["free"], rnd["hidx"], rnd["eps0"]
        x = rnd["x"]
        M, P = len(free), len(hidx)
        s0 = time.time()
        Hraw, asym = round_hessian(rnd)
        H = _psd(Hraw)
        resp = np.zeros((M, P - 1))
        rfrac = np.zeros(M)
        for a in range(M):
            mask = np.ones(P, dtype=bool)
            mask[a] = False
            xf = (-np.linalg.pinv(H[np.ix_(mask, mask)], rcond=rcond) @ H[mask, a] if P > 1
                  else np.zeros(0))
            v = np.zeros(P)
            v[a] = 1.0
            v[mask] = xf
            rfrac[a] = min(max(float(v @ H @ v) / max(float(H[a, a]), 1e-300), 0.0), 1.0)
            resp[a] = xf
        max_xf = np.max(np.abs(resp), axis=1) if P > 1 else np.zeros(M)
        norm_xf = np.linalg.norm(resp, axis=1)
        hs = {"round": st.n_rounds, "relax": relax, "free": M, "rows": P, "asym": asym,
              "noise": asym * float(np.max(np.abs(Hraw))) if Hraw.size else 0.0, **_eig_stats(Hraw, M),
              "rp_median": float(np.median(rfrac)), "rp_min": float(np.min(rfrac)), "rp_max": float(np.max(rfrac)),
              "n_rp_zero": int((rfrac <= 1e-12).sum()),
              "max_edge_move": float(np.max(np.abs(resp[:, M - 1:]))) if P > M else 0.0}
        hs["n_neg_beyond_noise"] = int((np.linalg.eigvalsh(Hraw) < -hs["noise"]).sum()) if Hraw.size else 0
        say(f"snap: round {st.n_rounds} Hessian, {P} rows: lowest eigenvalue {hs['eig_min']:.2e} ({hs['n_neg']} "
            f"negative; bulk {hs['eig_min_bulk']:.2e}, {hs['n_neg_bulk']} negative), median r_p "
            f"{hs['rp_median']:.2e} ({hs['n_rp_zero']} of {M} at 0), asymmetry {asym:.1e}")

        if stopper():
            raise _Stop
        curves = curves_at(x, free)
        curv = {i: c.curvature() for i, c in curves.items()}
        ceiling = np.minimum(np.array([eps - reserve_for(curv, i) for i in free]),
                             eps0 + share_for(eps0, M, relax))
        free_arr = np.asarray(free)
        x_free = x[free_arr]
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

        s1 = time.time()
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
        tick("candidates", s1)
        hs["seconds"] = time.time() - s0
        if rule == "newton":
            st.h_cur = (list(hidx), H)
        if len(rank) == 0:
            st.hstats.append(hs)                                 # with the round's outcome, so a resume adds it once
            say(f"snap: round {st.n_rounds}, {M} free, no candidate predicted under budget (relax {relax:g})")
            return False
        # the trials compile_snap would run, in its order (top_k per level, max_round_trials in all)
        plan, tried_at_level, truncated = [], {}, False
        for k in rank:
            level = float(eff[k])
            if tried_at_level.get(level, 0) >= settings.top_k:
                continue
            if len(plan) >= settings.max_round_trials:
                truncated = True
                break
            tried_at_level[level] = tried_at_level.get(level, 0) + 1
            a = int(a_idx[k])
            plan.append((a, int(free_arr[a]), float(d_sel[k]), float(ceiling[a]), float(level_cost[lev_idx[k]]),
                         int(j_sel[k]), float(ec_sel[k]), float(ep_sel[k])))
        rnd.update(plan=plan, resp=resp, n_cand=int(len(rank)), n_levels=len(tried_at_level), truncated=truncated,
                   H=H if rule == "newton" else None)
        st.hstats.append(hs)
        return True

    def hessian_round(relax: float = 1.0) -> bool:
        free = free_indices()
        M = len(free)
        if M == 0:
            return False
        rnd = st.round if st.round is not None and st.round["relax"] == relax else None
        if rnd is None:
            eps0 = reopt()
            st.n_rounds += 1
            st.cur_relax, st.cur_trials_done = relax, 0
            rnd = st.round = {"relax": relax, "free": free, "hidx": free + free_edges(), "x": st.x.copy(),
                              "eps0": eps0, "R": {}}
            checkpoint(time.time() - t0, floor)
        else:
            st.x[:] = rnd["x"]
        if "plan" not in rnd:
            if not plan_round(rnd, relax):
                st.round = None
                return False
            checkpoint(time.time() - t0, floor)
        plan, saved = rnd["plan"], rnd["x"]
        done = st.cur_trials_done
        s0 = time.time()
        for lo in range(done, len(plan), B):
            batch = plan[lo: lo + B]
            try:
                results = run_batch(batch, saved, rnd["hidx"], rnd["resp"], rnd.get("H"))
            finally:
                tick("trials", s0)
                s0 = time.time()
            for (a, i, d, ceil, c, j, ec, ep), (xt, trial) in zip(batch, results):
                st.n_trials += 1
                if trial <= ceil:
                    pin(i, saved[i] + d)
                    st.x[:] = xt
                    st.costs[i] = c
                    st.n_hess += 1
                    st.trace.append({"i": i, "stage": "hessian", "cost": c, "c_pre": ec, "c_pred": ep,
                                     "c_post": trial, "disp": abs(d)})
                    st.history.append((st.x.copy(), st.snapped.copy(), st.costs.copy(), trial))
                    st.cur_trials_done = 0
                    st.round = None
                    return True
                if settings.skip_failed:
                    key = (i, j)
                    st.failed[key] = max(st.failed.get(key, 0.0), relax)
            done += len(batch)
            st.cur_trials_done = done
            checkpoint_if_due()
        st.round = None
        if rnd["truncated"]:
            say(f"snap: round {st.n_rounds}, {M} free, {len(plan)} trials without an acceptance (relax {relax:g})")
        else:
            say(f"snap: round {st.n_rounds}, {M} free, {rnd['n_cand']} candidates over "
                f"{rnd['n_levels']} levels, none accepted (relax {relax:g})")
        return False

    # -------------------------------------------------------------------- loop
    try:
        with stopper:
            try:
                if resumed is None:
                    floor = reopt("floor")
                    say(f"snap: floor C_train = {floor:.3e}, budget = {eps:.3e}")
                    checkpoint(time.time() - t0, floor)
                else:
                    floor = resumed["floor"]
                    t0 -= resumed["elapsed"]
                steps = ([2.0 ** k for k in range(12)] if settings.relax_ladder == "gentle"
                         else [1.0, 4.0, 16.0, float("inf")])
                while not st.loop_done:
                    mid = bool(st.cur_failed_relax) or st.cur_trials_done > 0 or st.round is not None
                    nz = 0 if mid else zero_t_stage()
                    n_free = int((~st.snapped[:nb]).sum())
                    if nz:
                        checkpoint_if_due()
                        say(f"snap: zero-T stage snapped {nz}, {n_free} free, C = {cval(st.x):.3e}, "
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
                    if stopper():
                        raise DeadlineReached(f"stopped ({stopper.why()}); checkpoint {settings.checkpoint}")
                    if st.n_hess % 5 == 0:
                        say(f"snap: {st.n_hess} Hessian snaps, {int((~st.snapped[:nb]).sum())} free, "
                            f"T/q so far {0.5 * (weights * st.costs).sum():.1f}, C = {cval(st.x):.3e} "
                            f"({time.time() - t0:.0f}s; " + ", ".join(
                                f"{k} {v:.0f}s" for k, v in st.timing.items()
                                if k in ("reopt", "trials", "hessian", "curves", "endgame", "candidates")) + ")")

                # ---------------------------------------------------- end game
                if not st.loop_done:
                    st.loop_done = True
                    checkpoint(time.time() - t0, floor)
                s_eg = time.time()
                K = len(st.history)
                eg = st.eg_teq
                if "full" not in eg:
                    eg["full"] = total_teq(st.costs, straggler_costs(st.x, st.snapped, cval(st.x)))
                    checkpoint_if_due()
                for k in range(K - 1, -1, -1):
                    if k not in eg:
                        xk, snk, ck, c_k = st.history[k]
                        eg[k] = total_teq(ck, straggler_costs(xk, snk, c_k))
                        checkpoint_if_due()
                checkpoint(time.time() - t0, floor)
            except _Stop:
                checkpoint(time.time() - t0, floor)
                raise DeadlineReached(f"stopped ({stopper.why()}) inside a re-optimization or Hessian; "
                                      f"checkpoint {settings.checkpoint}") from None
    finally:
        if pool_ex is not None:
            pool_ex.shutdown(wait=True)
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
    c_prefix = cval(st.x)
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
            used = cval(xt) - c_prefix + sum(e[3] for e in alloc.values() if e[2] is None)
            if used <= room * (1.0 + 1e-9):
                fits = True
                break
            residual *= 0.8 * room / max(used, 1e-300)
            rest = rest_of(st.snapped)
            alloc = library_allocation(rest, curves_at(x0, rest), residual, c_prefix)
        if fits:
            st.x[:] = xt
        else:
            rest = rest_of(st.snapped)
            alloc = synthesis_allocation(curves_at(x0, rest), rest, room)
            say("snap: library stragglers did not fit jointly; synthesizing all of them")
    lib_alloc = {i: e for i, e in alloc.items() if e[2] is not None}
    syn_alloc = {i: e for i, e in alloc.items() if e[2] is None}
    for i, e in lib_alloc.items():
        st.costs[i] = e[1]
        st.snapped[i] = True
    for i, e in syn_alloc.items():
        st.costs[i] = e[1]
    c_final = cval(st.x)
    tick("endgame", s_eg)
    tq = 0.5 * float((weights * st.costs).sum())
    tq_bulk = 0.5 * float((weights[:nb] * st.costs[:nb]).sum())
    n_grad = int(st.timing.get("calls_reopt", 0) + st.timing.get("calls_floor", 0) + st.timing.get("calls_trial", 0)
                 + st.timing.get("hess_grads", 0) + st.timing.get("newton_hess_grads", 0))
    say(f"snap: done, zero-T {st.n_zero}, Hessian {st.n_hess}, library stragglers "
        f"{sum(1 for i in lib_alloc if i < nb)} (+{sum(1 for i in lib_alloc if i >= nb)} edge), KN stragglers "
        f"{sum(1 for i in syn_alloc if i < nb)} (+{sum(1 for i in syn_alloc if i >= nb)} edge), planned T/q {tq:.2f} "
        f"(bulk {tq_bulk:.2f}), C_train {c_final:.3e} ({time.time() - t0:.0f}s, {n_grad} gradients)")
    return {
        "settings": asdict(settings),
        "reoptimizer": rule if reoptimizer is None else getattr(reoptimizer, "__name__", "custom"),
        "trial_workers": B,
        "floor": floor,
        "C_train": c_final,
        "angles": st.x[:nb].tolist(),
        "edges": st.x[nb:].tolist(),
        "z": st.x.tolist(),
        "n_bulk": nb,
        "snapped": st.snapped[:nb].tolist(),
        "costs_plan": st.costs[:nb].tolist(),
        "multiplicity": weights[:nb].tolist(),
        "Tq_plan": tq,
        "Tq_plan_bulk": tq_bulk,
        "stragglers": {str(i): {"eps": e[0], "T_plan": e[1], "dC_plan": e[3]} for i, e in syn_alloc.items() if i < nb},
        "library_stragglers": {str(i): {"angle": e[2], "cost": e[1], "dC": e[3]}
                               for i, e in lib_alloc.items() if i < nb},
        "edge_weight": edge_w,
        "edge_stragglers": {str(i - nb): {"eps": e[0], "T_plan": e[1], "dC_plan": e[3]}
                            for i, e in syn_alloc.items() if i >= nb},
        "edge_library": {str(i - nb): {"angle": e[2], "cost": e[1], "dC": e[3]}
                         for i, e in lib_alloc.items() if i >= nb},
        "edge_zero_t": [int(i - nb) for i in edge_idx if st.snapped[i] and i not in lib_alloc],
        "n_zero_t": st.n_zero, "n_hessian": st.n_hess, "n_rounds": st.n_rounds, "n_trials": st.n_trials,
        "n_trials_run": st.n_trials_run,
        "n_snaps_kept": n_prefix, "n_snaps_trajectory": K, "Tq_full_trajectory": eg["full"],
        "trace": st.trace,
        "history": [{"z": xk.tolist(), "snapped": snk[:nb].tolist(), "costs": ck[:nb].tolist(), "c": c_k}
                    for xk, snk, ck, c_k in st.history],
        "hessian_stats": st.hstats,
        "reopt_log": st.reopt_log,
        "n_grad": n_grad,
        "n_value": int(st.timing.get("n_value", 0)),
        "timing": dict(st.timing),
        "seconds": time.time() - t0,
    }


# ============================================================================ realize and evaluate

def simplify_edges(x_bulk, e, c_fn, allowance: float, window: float = EDGE_WINDOW, log=None) -> tuple[np.ndarray, dict]:
    """Make the compiled edge gates cheaper while c (bulk fixed) rises by at most ``allowance``.

    Gates are (alpha, beta, gamma) triples, Rz(alpha) Ry(beta) Rz(gamma), in z's order (site
    N-1, then site 0, per parity-1 sublayer). Three greedy passes, each change kept only
    if c of the whole circuit stays within c_start + allowance:
      1. identity: gates set to (0, 0, 0), smallest single-gate rise first;
      2. Rz: a gate whose beta is within ``window`` of a multiple of 2 pi becomes
         (alpha + gamma, 0, 0) (the same gate up to a phase when beta is exactly such a
         multiple; the ZYZ form is degenerate there, so trained near-identity gates drift
         to pairs like (0.94, 0.0016, -0.94));
      3. pi/4: angles within ``window`` of a multiple of pi/4, closest first, set to it
         (edges._snap_near_clifford_t's rule for the ring arm's fitted edges).
    Returns (angles, info).
    """
    say = log or (lambda _m: None)
    x_bulk = np.asarray(x_bulk, dtype=float).reshape(-1)
    g = np.asarray(e, dtype=float).reshape(-1, 3).copy()
    orig = g.copy()

    def c_of(gg) -> float:
        return float(c_fn(np.concatenate([x_bulk, gg.reshape(-1)])))

    c0 = c_of(g)
    limit, cur = c0 + max(allowance, 0.0), c0
    dropped, canonical, snapped = [], [], []
    single = []
    for k in range(len(g)):
        if np.any(g[k]):
            trial = g.copy()
            trial[k] = 0.0
            single.append((c_of(trial) - c0, k))
    for dc, k in sorted(single):
        if c0 + dc > limit:
            break
        trial = g.copy()
        trial[k] = 0.0
        v = c_of(trial)
        if v <= limit:
            g, cur = trial, v
            dropped.append(k)
    for k in range(len(g)):
        a, b, c = g[k]
        rem = math.remainder(b, TWO_PI)
        if b == 0.0 or abs(rem) > window:
            continue
        trial = g.copy()
        trial[k] = [a + c, 0.0, 0.0]
        v = c_of(trial)
        if v <= limit:
            g, cur = trial, v
            canonical.append(k)
    flat = g.reshape(-1)
    for off, i in sorted((abs(math.remainder(v, QUARTER)), i) for i, v in enumerate(flat)):
        if off == 0.0:
            continue
        if off > window:
            break
        trial = flat.copy()
        trial[i] -= math.remainder(trial[i], QUARTER)
        v = c_of(trial.reshape(-1, 3))
        if v <= limit:
            flat, cur = trial, v
            snapped.append(i)
    info = {"allowance": float(allowance), "window": window, "c_before": c0, "c_after": cur,
            "n_identity": len(dropped), "n_rz": len(canonical), "n_pi4": len(snapped),
            "identity_gates": dropped, "rz_gates": canonical, "pi4_angles": snapped,
            "single_gate_rise": {str(k): dc for dc, k in single}, "angles_compiled": orig.reshape(-1).tolist()}
    say(f"realize: edges simplified within {allowance:.2e}: {len(dropped)} of {len(g)} gates to identity, "
        f"{len(canonical)} to a single Rz, {len(snapped)} angles to multiples of pi/4, c {c0:.6e} -> {cur:.6e}")
    return flat.copy(), info


def realize_chain_edges(angles, eps, skip=()) -> dict:
    """KN for the edge angles at eps, multiples of pi/4 within eps set exactly (edges.realize_edges' rule).

    ``eps`` is one budget for all or {index: budget}; the indices in ``skip`` (lattice
    angles, priced with the library by realize_chain) are left as they are. The record has
    the keys of edges.realize_edges' record that evaluate.py and fallbacks.py read (use,
    eps, angles, angles_real, kn, T_total, ledger_total, ledger2_total, n_rotations); the
    edges are always used.
    """
    x = np.asarray(angles, dtype=float).reshape(-1)
    x_real = x.copy()
    T = ledger = ledger2 = 0.0
    kn = {}
    n_exact = 0
    skip = {int(k) for k in skip}
    eps_of = (lambda i: float(eps[i])) if isinstance(eps, dict) else (lambda i: float(eps))
    for i, theta in enumerate(x):
        if i in skip:
            continue
        rem = math.remainder(theta, 2 * math.pi)
        k = round(rem / QUARTER)
        if 2.0 * abs(math.sin((rem - k * QUARTER) / 2.0)) <= eps_of(i):
            x_real[i] = theta - rem + k * QUARTER
            T += 0.0 if k % 2 == 0 else 1.0
            n_exact += 1
            continue
        r = synthesize(theta, eps_of(i))
        kn[str(i)] = r
        x_real[i] = r["theta_s"]
        T += r["expected_t"]
        ledger += (1.0 - r["p"]) * r["d_branch"]
        ledger2 += (1.0 - r["p"]) * r["d_branch"] ** 2
    return {"source": "compiled (trained with the bulk on the open chain)", "use": True,
            "eps": float(eps) if not isinstance(eps, dict) else min(eps.values(), default=0.0),
            "eps_per_angle": {str(i): eps_of(i) for i in range(x.size) if i not in skip},
            "angles": x.tolist(), "angles_real": x_real.tolist(), "kn": kn, "n_rotations": int(x.size),
            "n_kn": len(kn), "n_clifford_t": n_exact, "n_lattice": len(skip), "T_total": T, "ledger_total": ledger,
            "ledger2_total": ledger2}


def realize_chain(compiled: dict, cell: Cell, c_fn, tighten: float = 0.5, max_tries: int = 6, log=print,
                  edge_room_frac: float = EDGE_ROOM_FRAC, edge_window: float = EDGE_WINDOW,
                  simplify: bool = True) -> dict:
    """realize.realize on the open chain: lattice pricing, KN stragglers, the edges by their allocation.

    Edges. The end game gave each edge angle a library angle (edge_library) or a KN budget
    (edge_stragglers), at T weight edge_weight = 2/N. First simplify_edges may use
    edge_room_frac of the room left under the budget (budget - C_train - the planned rise of
    every KN straggler, bulk and edge) for identity gates, single Rz and multiples of pi/4,
    each checked on the chain. Then each edge angle is priced: an exact multiple of pi/4 at
    0 or 1 T; an untouched library angle with the library at every F (in per_F, weight
    edge_weight, like a bulk angle of multiplicity edge_weight); an untouched KN straggler
    with KN at its allotted eps; an angle the clean-up changed (a merged Rz) with KN at the
    smallest eps allotted in its gate, else edges.edge_eps(budget) (also every angle when
    the compile had no edge stragglers). Bulk stragglers are synthesized at the compiler's
    eps. The realized training c (c_fn, on the chain, every rotation realized) must be at
    most the budget; if not, every KN eps (bulk and edge) is multiplied by ``tighten`` and the
    KN rotations are synthesized again, as realize.py does with the ring C. If the lattice
    angles and the edges alone (KN angles exact) are already over the budget, it says so
    and does not tighten, since no KN precision can help (over_budget_without_stragglers).
    """
    from snapevo.realize import F_GRID
    x = np.asarray(compiled["angles"], dtype=float)
    snapped = np.asarray(compiled["snapped"], dtype=bool)
    mult = np.asarray(compiled["multiplicity"], dtype=float)
    budget = float(compiled["settings"]["budget"])
    libs = {F: Library(F) for F in F_GRID}

    def price(theta):
        priced = {}
        for F, lib in libs.items():
            hit = lib.lookup(theta)
            if hit is None:
                raise ValueError(f"angle {theta!r} is not a lattice point")
            priced[F] = hit
        return {"kind": Library.kind(theta, priced[F_GRID[0]][0]),
                "cost": {f"{F:g}": c for F, (c, q) in priced.items()},
                "q": {f"{F:g}": q for F, (c, q) in priced.items()}}

    lib_rows = {int(i): price(x[i]) for i in np.flatnonzero(snapped)}

    e_comp = np.asarray(compiled["edges"], dtype=float)
    edge_w = float(compiled.get("edge_weight", 0.0))
    e_lib = {int(k): v for k, v in compiled.get("edge_library", {}).items()}
    e_kn = {int(k): float(v["eps"]) for k, v in compiled.get("edge_stragglers", {}).items()}
    kn_planned = sum(float(v.get("dC_plan", 0.0)) for v in compiled["stragglers"].values()) + \
        sum(float(v.get("dC_plan", 0.0)) for v in compiled.get("edge_stragglers", {}).values())
    room = budget - float(compiled["C_train"]) - kn_planned
    sinfo = None
    e_use = e_comp
    if simplify and e_comp.size:
        e_use, sinfo = simplify_edges(x, e_comp, c_fn, edge_room_frac * max(room, 0.0), edge_window, log=log)
        sinfo.update(room=room, kn_planned=kn_planned, room_frac=edge_room_frac)
    base = edge_eps(budget)
    edge_lat, eps_e = {}, {}
    for k in range(e_comp.size):
        same = e_use[k] == e_comp[k]
        quarter = abs(math.remainder(e_use[k], QUARTER)) < 1e-12
        if same and k in e_lib and not quarter:
            edge_lat[k] = price(e_use[k])
        elif same and k in e_kn:
            eps_e[k] = e_kn[k]
        else:
            gate = range(3 * (k // 3), 3 * (k // 3) + 3)
            eps_e[k] = min([e_kn[j] for j in gate if j in e_kn], default=base)
    c_fixed = float(c_fn(np.concatenate([x, e_use])))
    over = c_fixed > budget * (1 + 1e-9)
    if over:
        log(f"realize: WARNING the lattice angles and the edges alone give C_train {c_fixed:.3e} > budget "
            f"{budget:.3e} (compiled {float(compiled['C_train']):.3e}); tightening the KN rotations cannot fix it")
    eps_plan = {int(k): float(v["eps"]) for k, v in compiled["stragglers"].items()}
    scale = 1.0
    for _attempt in range(max_tries):
        kn_rows = {i: synthesize(x[i], e * scale) for i, e in eps_plan.items()}
        x_real = x.copy()
        for i, r in kn_rows.items():
            x_real[i] = r["theta_s"]
        E = realize_chain_edges(e_use, {k: e * scale for k, e in eps_e.items()}, skip=edge_lat)
        e_real = np.asarray(E["angles_real"], dtype=float)
        c_real = float(c_fn(np.concatenate([x_real, e_real])))
        log(f"realize: {len(lib_rows)} lattice angles, {len(kn_rows)} KN stragglers; edges: {len(edge_lat)} lattice, "
            f"{E['n_kn']} KN, {E['n_clifford_t']} multiples of pi/4 (eps x {scale:g}); C_train {c_real:.3e} on the "
            f"open chain vs budget {budget:.3e}")
        if c_real <= budget * (1 + 1e-9) or not (kn_rows or E["kn"]) or over:
            break
        scale *= tighten
    else:
        log("realize: realized C still above budget after tightening; keeping the last attempt")
    E["angles_compiled"] = e_comp.tolist()
    E["simplify"] = sinfo
    E["lattice"] = {str(k): r for k, r in edge_lat.items()}
    E["edge_weight"] = edge_w

    per_F = {}
    for F, lib in libs.items():
        cost, q, delta = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
        for i, row in lib_rows.items():
            cost[i], q[i], delta[i] = row["cost"][f"{F:g}"], row["q"][f"{F:g}"], lib.delta_fb
        for i, r in kn_rows.items():
            cost[i], q[i], delta[i] = r["expected_t"], 1.0 - r["p"], r["d_branch"]
        ec = np.array([edge_lat[k]["cost"][f"{F:g}"] for k in edge_lat])
        eq = np.array([edge_lat[k]["q"][f"{F:g}"] for k in edge_lat])
        per_F[f"{F:g}"] = {"Tq": 0.5 * float(mult @ cost) + 0.5 * edge_w * float(ec.sum()),
                           "ledger_per_qubit": 0.5 * float(mult @ (q * delta))
                           + 0.5 * edge_w * float((eq * lib.delta_fb).sum()),
                           "ledger2_per_qubit": 0.5 * float(mult @ (q * delta ** 2))
                           + 0.5 * edge_w * float((eq * lib.delta_fb ** 2).sum()),
                           "delta_fb": lib.delta_fb}
    rot_per_q = cell.rotations_per_sublayer_per_qubit * (len(x) // cell.n_angles)
    return {
        "budget": budget,
        "angles_compiled": x.tolist(),
        "angles_real": x_real.tolist(),
        "multiplicity": mult.tolist(),
        "C_train_real": c_real,
        "C_train_compiled": float(compiled["C_train"]),
        "C_train_lattice_edges": c_fixed,
        "over_budget_without_stragglers": bool(over),
        "straggler_eps_scale": scale,
        "lattice": {str(i): r for i, r in lib_rows.items()},
        "kn": {str(i): r for i, r in kn_rows.items()},
        "per_F": per_F,
        "rotations_per_qubit": rot_per_q,
        "n_free": int(sum(1 for r in lib_rows.values() if r["kind"] == "clifford")),
        "n_t": int(sum(1 for r in lib_rows.values() if r["kind"] == "t")),
        "n_rus": int(sum(1 for r in lib_rows.values() if r["kind"] == "rus")),
        "n_kn": len(kn_rows),
        "max_kn_offset": max((abs(r["theta_s"] - x[i]) for i, r in kn_rows.items()), default=0.0),
        "edges": E,
        "z_real": np.concatenate([x_real, e_real]).tolist(),
        "bc": "open",
    }


def _order(label: str) -> float:
    try:
        return float(label)
    except ValueError:
        return math.inf


def evaluate_chain(real: dict, model, cell: Cell, t: float, N: int = 100, f_tol: float = 0.01, log=print,
                   expect_N: int | None = 100, **mps_kw) -> dict:
    """evaluate.evaluate_circuit for a realize_chain record, without its ring held-out side metric.

    The circuit's MPS infidelity on the sweep's inputs (evaluate.mps_infidelity with
    ``mps_kw``: states, dt, chi_max, cutoff, cache_dir, check), the totals with the
    fallback bound per library F and the choice of F, and the T counts. In place of
    C_ring and infid_ring_pred the record carries C_train_real (the realized c on the
    open chain's training inputs) and infid_train_pred = 1 - (1 - C_train_real)^N. The
    reported error is at N = 100; any other N is refused unless expect_N is None (tests).
    """
    from snapevo.evaluate import fidelity_bound, mps_infidelity
    if expect_N is not None and int(N) != int(expect_N):
        raise ValueError(f"the reported evaluation is at N = {expect_N}, not {N}")
    E = real["edges"]
    circuit = Layered.brickwork(cell, real["angles_real"])
    m = mps_infidelity(circuit, pairs(E["angles_real"]), model, t, N=N, log=log, **mps_kw)
    pricing = real["per_F"]
    labels = sorted(pricing, key=_order)
    per = {}
    for label in labels:
        pf = pricing[label]
        L = N * pf["ledger_per_qubit"] + E["ledger_total"]
        Q2 = N * pf["ledger2_per_qubit"] / 2 + E["ledger2_total"] / 2
        total = fidelity_bound(m["infid"], L, Q2)
        per[label] = {"Tq": pf["Tq"] + E["T_total"] / N, "ledger": L, "Q2": Q2, "infid_total": total}
    floor = per[labels[-1]]["infid_total"]
    choice = next(lab for lab in labels if per[lab]["infid_total"] <= (1.0 + f_tol) * floor)
    best = per[choice]
    share = max(0.0, (best["infid_total"] - m["infid"]) / best["infid_total"]) if best["infid_total"] > 0 else 0.0
    rot = real["rotations_per_qubit"] + E["n_rotations"] / N
    C = float(real["C_train_real"])
    out = {"N": N, "mps": m, "C_train_real": C, "infid_train_pred": 1.0 - (1.0 - C) ** N, "per_F": per,
           "F": choice, "infid": m["infid"], "infid_total": best["infid_total"], "fallback_share": share,
           "fallback_flag": share > 0.1, "c_site": m["c_site"], "ledger": best["ledger"],
           "Tq": best["Tq"], "T_edges_per_qubit": E["T_total"] / N, "rotations_per_qubit": rot,
           "T_per_rotation": best["Tq"] / rot, "T_per_qubit_per_time": best["Tq"] / t, "bc": "open"}
    log(f"evaluate: N = {N}, F = {choice}: 1 - F = {m['infid']:.3e} (training c predicts "
        f"{out['infid_train_pred']:.3e}), with fallbacks {best['infid_total']:.3e} (share {share:.1%}), "
        f"T/q {best['Tq']:.1f} (edges {out['T_edges_per_qubit']:.2f}), T/rot {out['T_per_rotation']:.2f}")
    return out


def _floor(r: dict) -> float:
    return max(float(r["c_train"]), float(r.get("c_heldout") or 0.0))


def pick_depth(records, budget: float, margin: float) -> int:
    """Shallowest depth of a chain ladder whose floor times margin is under the budget, else the deepest.

    cli._pick_depth with the chain ladder's own floor, max(c_train, c_heldout), in place
    of the ring's max(C_train, C_val).
    """
    for r in sorted(records, key=lambda r: r["D"]):
        if _floor(r) * margin <= budget:
            return int(r["D"])
    return int(max(records, key=lambda r: r["D"])["D"])


EVAL_DEFAULTS = {"dt": 0.2, "chi_max": 1024, "cutoff": 1e-12, "check": True}

EDGE_FITS = ("newton", "lbfgs", "lbfgs-uncapped", "auto")
"""ring_point's edge fits. For the lowest T use "lbfgs-uncapped": at h_x = 1.6, t = 1, budget 1e-3, D = 5 the
Newton fit (lower fit infidelity, 0 of 12 angles near a multiple of pi/4) gave T/q 39.36 with 2.07 from the
edges, the L-BFGS-B fit (10 of 12 set to multiples of pi/4) 37.59 with 0.30, at the same c = 6.170e-4."""

EDGE_NEWTON_MAX_N = 12
"""ring_point's edge_fit "auto": the Newton edge fit up to this N_fit (edges.fit_size: t <= 2), production's
L-BFGS-B fit above. One exact Hessian of the edge fit costs about 3 gradients per edge angle, and a gradient at
N_fit = 18 and D ~ 60 about 15-20 s (production's fits there: median 20-30 min for 30-90 iterations), so a
Newton fit at t >= 6 would take tens of hours per point; at N_fit = 12 (t = 2, D = 17, 48 angles) a Hessian
takes 6 s."""
"""evaluate.mps_infidelity settings of the sweep's default evaluation (cli's eval options)."""


def _polished_start(P: ChainProblem, rec: dict, D: int, nb: int, out_dir, tol: float, deadline, log):
    """The depth's ladder z polished by polish_chain to tolerance ``tol``, cached in out_dir/polish_D<D>.json.

    The cache serves every budget whose tolerance is at least the one it reached; a tighter
    one continues polishing from the cached point.
    """
    from pathlib import Path

    from snapevo import io
    z_in = np.asarray(rec["z"], dtype=float)
    path = Path(out_dir) / f"polish_D{D}.json"
    want = {"train_cutoff": rec["train_cutoff"], "train_chi": rec["train_chi"]}
    start, prior = z_in, None
    if path.exists():
        doc = io.read_json(path)
        if np.array_equal(np.asarray(doc["z_in"]), z_in) and {k: doc.get(k) for k in want} == want:
            if float(doc["tol"]) <= tol:
                log(f"polish: {path.name} reached {doc['tol']:.1e}, c {doc['info']['c_before']:.6e} -> "
                    f"{doc['info']['c_after']:.6e}")
                return np.asarray(doc["z"], dtype=float), doc["info"]
            start, prior = np.asarray(doc["z"], dtype=float), doc
    z, info = polish_chain(start, P.objective, lambda zz, rows: P.hessian(zz, rows), tol, n_bulk=nb, log=log,
                           checkpoint=str(path.with_suffix(".ckpt.json")), deadline=deadline)
    if prior is not None:
        info = {**info, "c_before": prior["info"]["c_before"], "earlier": prior["info"]}
    io.write_json(path, {**want, "D": D, "tol": float(tol), "z_in": z_in.tolist(), "z": z.tolist(), "info": info})
    ck = path.with_suffix(".ckpt.json")
    if ck.exists():
        ck.unlink()
    return z, info


def _same_reopt(compiled: dict | None, settings_kw: dict | None, path) -> None:
    """Refuse to reuse a snap_ record made with another re-optimizer than the one asked for."""
    want = (settings_kw or {}).get("reopt")
    if compiled is None or want is None:
        return
    have = compiled.get("reoptimizer") or (compiled.get("settings") or {}).get("reopt")
    if have != want:
        raise ValueError(f"{path} was compiled with re-optimizer {have!r}, not {want!r}: use another runs or "
                         f"output folder per re-optimizer, or fresh=True")


def chain_point(model, t: float, budget: float, runs, D: int | None = None, margins=(3.0, 30.0),
                ref_cache=None, groups: int | None = None, replicas: int = 1, trial_workers: int = 1,
                env_blocks: int | None = None, settings_kw: dict | None = None, eval_kw: dict | None = None,
                steps=("compile", "realize", "evaluate"), fresh: bool = False, deadline: float | None = None,
                local: bool = False, polish: bool = True, hessian_chunk: int | None = None,
                env_layers: int | None = None, check_N: bool = True, out_sub: str | None = None,
                all_depths: bool = False, log=print) -> list[dict]:
    """Compile, realize and evaluate one budget on the open chain, at one depth per floor margin.

    Reads runs/<tag>/t<T>/chain/ladder.json (train_chain) and writes snap_/real_/eval_
    b<budget>_D<D>.json next to it, in the format of the ring arm's files (module
    docstring). As cli.cmd_point: each margin m picks the shallowest depth whose floor
    (here the ladder's max(c_train, c_heldout)) is at most budget/m, a depth whose floor
    is above 2 budget/3 is skipped, and existing files are reused unless ``fresh``. Before
    compiling, the depth's z is polished (polish_chain, to POLISH_TOL_REL x budget; polish=False
    skips it), cached as polish_D<D>.json for every budget it is tight enough for. A killed compile resumes from
    its checkpoint snap_..ckpt.pkl, exactly. The training problem is the ladder's (N,
    train_set) with its references from ``ref_cache``; the compile's truncation is the
    depth's train_cutoff and train_chi. Hessian rows go hessian_chunk at a time (default
    8 per replica), and environments hold at most env_layers sublayers (default all). The
    evaluation runs on the sweep's inputs at N = 100 (check_N) against runs/<tag>/t<T>/ref,
    with EVAL_DEFAULTS updated by eval_kw. out_sub puts the snap_/real_/eval_ records (and the
    compile checkpoint) in chain/<out_sub>/, e.g. one folder per re-optimizer; the ladder and
    the polish cache stay in chain/ (the polish does not depend on the re-optimizer). all_depths
    compiles every ladder depth whose floor is at most budget/1.5 instead of one per margin.
    """
    import json as _json
    from pathlib import Path

    from snapevo import io
    from snapevo.cells import cell_for
    from snapevo.evaluate import MPS_SET
    from snapevo.mps_train import open_chain_problem

    cell = cell_for(model)
    d = Path(runs) / model.tag / f"t{t:g}"
    out_dir = d / "chain"
    rec_dir = out_dir / out_sub if out_sub else out_dir
    lad = _json.loads((out_dir / "ladder.json").read_text())
    if lad.get("cell") != cell.name:
        raise ValueError(f"the ladder was trained with {lad.get('cell')}, not {cell.name}")
    recs = lad["results"]
    if D:
        depths = [int(D)]
    elif all_depths:
        depths = sorted(int(r["D"]) for r in recs if _floor(r) * 1.5 <= budget)
    else:
        depths = sorted({pick_depth(recs, budget, float(m)) for m in margins})
    N = int(lad["N"])
    if check_N and N != 100:
        raise ValueError(f"the ladder is at N = {N}; the open-chain arm reports N = 100")
    seed, n_in = lad["train_set"]
    problem = None
    results = []
    for D_ in depths:
        stem = f"{io.budget_tag(budget)}_D{D_}"
        p_snap, p_real, p_eval = (rec_dir / f"{k}_{stem}.json" for k in ("snap", "real", "eval"))
        if p_eval.exists() and not fresh and "evaluate" in steps:
            log(f"chain point: {p_eval.name} exists (fresh=True to redo)")
            results.append(io.read_json(p_eval))
            continue
        rec = next(r for r in recs if int(r["D"]) == D_)
        floor = _floor(rec)
        if floor * 1.5 > budget:
            log(f"chain point: skip, the D = {D_} floor {floor:.2e} leaves less than a third of the budget "
                f"{budget:.1e} for snapping (train deeper)")
            continue
        need_cost = ("compile" in steps and (fresh or not p_snap.exists())) or \
                    ("realize" in steps and (fresh or not p_real.exists()))
        compiled = io.read_json(p_snap) if p_snap.exists() and not fresh else None
        _same_reopt(compiled, settings_kw, p_snap)
        real = io.read_json(p_real) if p_real.exists() and not fresh else None
        if need_cost:
            if problem is None:
                problem = open_chain_problem(model, t, N, n_in, seed, ref_cache, log=log)
            vecs, targets = problem
            with ChainProblem(model, cell, t, D_, N, vecs, targets, chi_max=int(rec["train_chi"]),
                              cutoff=float(rec["train_cutoff"]), groups=groups, replicas=replicas,
                              env_blocks=env_blocks, local=local) as P:
                if compiled is None and "compile" in steps:
                    nb = D_ * cell.n_angles
                    z0, pinfo = np.asarray(rec["z"], dtype=float), None
                    if polish:
                        z0, pinfo = _polished_start(P, rec, D_, nb, out_dir, POLISH_TOL_REL * budget, deadline, log)
                    ck = p_snap.with_suffix(".ckpt.pkl")
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    if fresh and ck.exists():
                        ck.unlink()
                    settings = ChainSettings(budget=budget, **{"checkpoint": str(ck), "edge_weight": 2.0 / N,
                                                               **(settings_kw or {})})
                    chunk = hessian_chunk or 8 * P.pool.K
                    log(f"chain compile: {model.describe()}, cell {cell.name}, t = {t:g}, D = {D_}, budget "
                        f"{budget:.1e}, N = {N}, {P.pool.size} workers ({P.pool.G} groups x {P.pool.K}), "
                        f"ladder floor {floor:.3e}")
                    compiled = compile_chain(z0, cell, D_, settings, P.objective, P.c_fn, P.hessian, P.envs_at,
                                             Library(settings.F), log=log, trial_workers=trial_workers,
                                             deadline=deadline, hessian_chunk=chunk, env_layers=env_layers)
                    compiled.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D_, "N": N,
                                     "bc": "open", "train_set": [seed, n_in], "train_cutoff": rec["train_cutoff"],
                                     "train_chi": rec["train_chi"], "ladder_c_train": float(rec["c_train"]),
                                     "ladder_c_heldout": rec.get("c_heldout"), "polish": pinfo,
                                     "workers": [P.pool.G, P.pool.K], "hessian_chunk": chunk})
                    io.write_json(p_snap, compiled)
                    if ck.exists():
                        ck.unlink()
                    log(f"chain compile: wrote {p_snap}")
                if real is None and compiled is not None and "realize" in steps:
                    real = realize_chain(compiled, cell, P.c_fn, log=log)
                    real.update({"model": model.to_dict(), "cell": cell.name, "D": D_, "t": t})
                    io.write_json(p_real, real)
        if real is not None and "evaluate" in steps:
            kw = {"states": MPS_SET, **EVAL_DEFAULTS, **(eval_kw or {}), "cache_dir": d / "ref"}
            ev = evaluate_chain(real, model, cell, t, N=N, log=log, expect_N=100 if check_N else None, **kw)
            ev.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D_, "budget": budget,
                       "n_free": real["n_free"], "n_t": real["n_t"], "n_rus": real["n_rus"], "n_kn": real["n_kn"]})
            io.write_json(p_eval, ev)
            results.append(ev)
    return results


def _ring_pick_depth(records, budget: float, margin: float) -> int:
    """cli._pick_depth: the shallowest ring depth whose max(C_train, C_val) times margin is under the budget."""
    for r in sorted(records, key=lambda r: r["D"]):
        if max(r["C_train"], r.get("C_val", 0.0)) * margin <= budget:
            return int(r["D"])
    return int(max(records, key=lambda r: r["D"])["D"])


def ring_point(model, t: float, budget: float, runs, out_sub: str = "ring2", D: int | None = None,
               margins=(3.0, 30.0), settings_kw: dict | None = None, eval_kw: dict | None = None,
               steps=("compile", "realize", "evaluate"), fresh: bool = False, deadline: float | None = None,
               cells: str = "cells.json", edge_fit: str | None = None, all_depths: bool = False,
               log=print) -> list[dict]:
    """The ring arm with compile_chain's protocol: cli.cmd_point with compile_chain(RingProblem) as the compile.

    Depths as cmd_point (runs/<tag>/t<T>/<cells>, by default the sweep's cells.json, or a
    Newton ladder such as ring2/ladder.json; each margin's shallowest depth with
    max(C_train, C_val) <= budget/margin; a floor above 2 budget/3 is skipped). The compile
    is compile_chain on RingProblem with ChainSettings updated by settings_kw (which must name
    the re-optimizer, reopt); realize is realize.realize with the ring C, the edges are fitted
    by edge_fit (required: "lbfgs", edges.fit_edges, production's; "newton",
    edges_newton.fit_edges_newton, the same problem and record by trust-region Newton;
    "lbfgs-uncapped", edges_newton.fit_edges_lbfgs, fit_edges without its iteration cap; "auto",
    newton when the fit chain has at most EDGE_NEWTON_MAX_N sites, lbfgs above) and
    synthesized by edges.realize_edges at edge_eps(budget), and the evaluation is
    evaluate.evaluate_circuit at N = 100 against runs/<tag>/t<T>/ref, i.e. the ring arm's
    own realize and evaluate. Records go to runs/<tag>/t<T>/<out_sub>/ in the production
    format; the production files are not touched. A killed compile resumes from its
    checkpoint. all_depths compiles every ladder depth whose floor is at most budget/1.5
    instead of one depth per margin.
    """
    from pathlib import Path

    from snapevo import io
    from snapevo.cells import cell_for
    from snapevo.edges import fit_edges, fit_size, realize_edges
    from snapevo.edges_newton import fit_edges_lbfgs, fit_edges_newton
    from snapevo.evaluate import MPS_SET, evaluate_circuit
    from snapevo.realize import realize
    from snapevo.train import cell_records, load_cell

    cell = cell_for(model)
    d = Path(runs) / model.tag / f"t{t:g}"
    out_dir = d / out_sub
    recs = cell_records(d / cells)
    if D:
        depths = [int(D)]
    elif all_depths:
        depths = sorted(int(r["D"]) for r in recs if max(r["C_train"], r.get("C_val", 0.0)) * 1.5 <= budget)
    else:
        depths = sorted({_ring_pick_depth(recs, budget, float(m)) for m in margins})
    if edge_fit not in EDGE_FITS:
        raise ValueError(f"edge_fit must be one of {EDGE_FITS}, not {edge_fit!r}")
    results = []
    for D_ in depths:
        stem = f"{io.budget_tag(budget)}_D{D_}"
        p_snap, p_real, p_eval = (out_dir / f"{k}_{stem}.json" for k in ("snap", "real", "eval"))
        if p_eval.exists() and not fresh and "evaluate" in steps:
            log(f"ring point: {p_eval.name} exists (fresh=True to redo)")
            results.append(io.read_json(p_eval))
            continue
        rec = next(r for r in recs if int(r["D"]) == D_)
        floor = max(rec["C_train"], rec.get("C_val", 0.0))
        if floor * 1.5 > budget:
            log(f"ring point: skip, the D = {D_} floor {floor:.2e} leaves less than a third of the budget "
                f"{budget:.1e} for snapping")
            continue
        compiled = io.read_json(p_snap) if p_snap.exists() and not fresh else None
        _same_reopt(compiled, settings_kw, p_snap)
        if compiled is not None and compiled.get("cells", "cells.json") != cells:
            raise ValueError(f"{p_snap} was compiled from {compiled.get('cells', 'cells.json')}, not {cells}: use "
                             f"another output folder per ladder, or fresh=True")
        if compiled is None and "compile" in steps:
            out_dir.mkdir(parents=True, exist_ok=True)
            ck = p_snap.with_suffix(".ckpt.pkl")
            if fresh and ck.exists():
                ck.unlink()
            x0, header = load_cell(d / cells, D_)
            if header["cell"] != cell.name:
                raise ValueError(f"cells.json was trained with {header['cell']}, not {cell.name}")
            settings = ChainSettings(budget=budget, **{"checkpoint": str(ck), **(settings_kw or {})})
            P = RingProblem(model, cell, t, D_)
            log(f"ring compile: {model.describe()}, cell {cell.name}, t = {t:g}, D = {D_}, budget {budget:.1e}")
            compiled = compile_chain(x0, cell, D_, settings, P.objective, P.c_fn, P.hessian, P.envs_at,
                                     Library(settings.F), log=log, deadline=deadline)
            compiled.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D_, "N": P.N, "bc": "ring",
                             "arm": "ring2", "ring_hessians": P.stats["n_hessian"], "cells": cells,
                             "ladder_C_train": rec["C_train"], "ladder_C_val": rec.get("C_val"),
                             "ladder_start": rec.get("start"), "ladder_optimizer": rec.get("optimizer")})
            io.write_json(p_snap, compiled)
            if ck.exists():
                ck.unlink()
            log(f"ring compile: wrote {p_snap}")
        real = io.read_json(p_real) if p_real.exists() and not fresh else None
        if real is None and compiled is not None and "realize" in steps:
            real = realize(compiled, model, cell, t, log=log)
            real.update({"model": model.to_dict(), "cell": cell.name, "D": D_})
            kind = edge_fit if edge_fit != "auto" else ("newton" if fit_size(t) <= EDGE_NEWTON_MAX_N else "lbfgs")
            fitter = {"newton": fit_edges_newton, "lbfgs": fit_edges, "lbfgs-uncapped": fit_edges_lbfgs}[kind]
            fit = fitter(Layered.brickwork(cell, real["angles_real"]), model, t, log=log)
            real["edge_fit"] = kind
            real["edges"] = realize_edges(fit, edge_eps(budget))
            io.write_json(p_real, real)
        if real is not None and "evaluate" in steps:
            kw = {"states": MPS_SET, **EVAL_DEFAULTS, **(eval_kw or {}), "cache_dir": d / "ref"}
            ev = evaluate_circuit(Layered.brickwork(cell, real["angles_real"]), real["edges"], real["per_F"],
                                  real["rotations_per_qubit"], model, t, N=100, log=log, **kw)
            ev.update({"model": model.to_dict(), "cell": cell.name, "t": t, "D": D_, "budget": budget,
                       "n_free": real["n_free"], "n_t": real["n_t"], "n_rus": real["n_rus"], "n_kn": real["n_kn"],
                       "arm": "ring2"})
            io.write_json(p_eval, ev)
            results.append(ev)
    return results
