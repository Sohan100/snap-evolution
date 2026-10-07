"""Edge rotations: what a ring-trained brickwork needs at the two ends of an open chain.

Cells are trained on a ring. On an open chain the parity-1 sublayers lose the
wrap bond (N-1, 0), and with it whatever correction the trained gates relied on
there. Each lost gate is replaced by a free rotation on each end site,

    U = Rz(alpha) Ry(beta) Rz(gamma),

six angles per parity-1 sublayer. We fit them with the bulk angles fixed, on an
open chain of N_fit sites, to the mean infidelity over random product inputs
(state vectors, exact gradients from jax). The start point is the cell's own
single-qubit factor, the wrap gate with its coupling removed. For the
zero-field cell at D = 11 this takes the boundary error from 4e-4 to below one
bulk site's worth (docs/validation.md).

After the fit, angles within 1e-2 of a multiple of pi/4 are set to it, closest
first, while the fit infidelity rises by less than a tenth of one bulk site's
worth in all. Those cost 0 or 1 T and need no synthesis.

We keep the fitted edges only when they remove at least one bulk site's worth of
error at N_fit (worth_it). Otherwise the open chain uses the cell's own factor,
which costs nothing extra. Kept angles are synthesized with KN non-mixed, like
stragglers. Ry(beta) = S H Rz(beta) H S^dag, so every edge angle costs one Rz.
The edges are a fixed cost, at most 6 rotations per parity-1 sublayer, so their
T count per qubit falls as 1/N.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from scipy.optimize import minimize

from snapevo.circuits import Layered, random_product_states
from snapevo.hamiltonian import XXZ, evolve
from snapevo.library import QUARTER
from snapevo.synthesis import synthesize

__all__ = ["EDGE_SET", "u_zyz", "zyz_angles", "fit_size", "edge_eps", "start_angles", "pairs",
           "fit_edges", "worth_it", "realize_edges"]

EDGE_SET: tuple[int, int] = (31415, 12)
"""(seed, n_states) of the product inputs the edges are fitted on."""


def u_zyz(a: float, b: float, c: float) -> np.ndarray:
    """Rz(a) Ry(b) Rz(c)."""
    rz = lambda x: np.diag([np.exp(-0.5j * x), np.exp(0.5j * x)])        # noqa: E731
    ry = np.array([[math.cos(b / 2), -math.sin(b / 2)], [math.sin(b / 2), math.cos(b / 2)]], dtype=complex)
    return rz(a) @ ry @ rz(c)


def zyz_angles(U: np.ndarray) -> tuple[float, float, float]:
    """(a, b, c) with U = e^{i phase} Rz(a) Ry(b) Rz(c)."""
    V = U / np.sqrt(np.linalg.det(U))
    x, y = V[0, 0], V[1, 0]
    b = 2.0 * math.atan2(abs(y), abs(x))
    s = -2.0 * np.angle(x) if abs(x) > 1e-14 else 0.0       # a + c
    d = 2.0 * np.angle(y) if abs(y) > 1e-14 else 0.0        # a - c
    return float((s + d) / 2), float(b), float((s - d) / 2)


def fit_size(t: float) -> int:
    """Default N_fit: the end regions must not overlap much, so it grows with t (cap 18)."""
    return int(min(18, max(12, 2 * math.ceil(t) + 8)))


def edge_eps(budget: float) -> float:
    """KN budget for the edge rotations, small enough that they never matter."""
    return float(min(1e-4, 0.1 * math.sqrt(budget)))


def start_angles(circuit: Layered) -> np.ndarray:
    """ZYZ angles of the cell's single-qubit factor on the wrap bond, the fit's start point."""
    out = []
    for g_last, g_first in circuit.edge_pairs(None):
        out += [*zyz_angles(g_last), *zyz_angles(g_first)]
    return np.array(out, dtype=float)


def pairs(angles) -> list[tuple[np.ndarray, np.ndarray]]:
    """Edge pairs (gate on N-1, gate on 0) from the flat angle vector, 6 per parity-1 sublayer."""
    x = np.asarray(angles, dtype=float).reshape(-1, 6)
    return [(u_zyz(*r[:3]), u_zyz(*r[3:])) for r in x]


def _objective(circuit: Layered, N: int, inputs: np.ndarray, targets: np.ndarray):
    """Jitted x -> (mean infidelity, gradient) on the open chain, bulk fixed."""
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    n = inputs.shape[0]
    s0 = jnp.asarray(inputs.reshape((n,) + (2,) * N))
    tg = jnp.asarray(targets)
    gates = [(p, jnp.asarray(circuit.cell.gate(a).reshape(2, 2, 2, 2))) for p, a in circuit.layers]

    def g2(s, U, i):
        s = jnp.tensordot(s, U, axes=([i + 1, i + 2], [2, 3]))
        return jnp.moveaxis(s, [-2, -1], [i + 1, i + 2])

    def g1(s, g, i):
        return jnp.moveaxis(jnp.tensordot(s, g, axes=([i + 1], [1])), -1, i + 1)

    def zyz(a, b, c):
        e = jnp.exp(-0.5j * a), jnp.exp(-0.5j * c)
        ry = jnp.array([[jnp.cos(b / 2), -jnp.sin(b / 2)], [jnp.sin(b / 2), jnp.cos(b / 2)]])
        return jnp.diag(jnp.array([e[0], jnp.conj(e[0])])) @ ry @ jnp.diag(jnp.array([e[1], jnp.conj(e[1])]))

    @jax.checkpoint
    def even(s, U):
        for i in range(0, N - 1, 2):
            s = g2(s, U, i)
        return s

    @jax.checkpoint
    def odd(s, U, p):
        for i in range(1, N - 1, 2):
            s = g2(s, U, i)
        return g1(g1(s, zyz(p[0], p[1], p[2]), N - 1), zyz(p[3], p[4], p[5]), 0)

    def cost(x):
        s, j = s0, 0
        for parity, U in gates:
            if parity == 1:
                s, j = odd(s, U, x[6 * j: 6 * j + 6]), j + 1
            else:
                s = even(s, U)
        ov = jnp.sum(jnp.conj(tg) * s.reshape(n, -1), axis=1)
        return 1.0 - jnp.mean(jnp.abs(ov) ** 2)

    vg = jax.jit(jax.value_and_grad(cost))

    def f(x):
        v, g = vg(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    return f


def fit_edges(circuit: Layered, model: XXZ, t: float, N_fit: int | None = None,
              states: tuple[int, int] = EDGE_SET, maxiter: int = 1000, log=print) -> dict:
    """Fit the edge rotations of a realized brickwork on an open chain of N_fit sites.

    Returns the angles and three infidelities at N_fit on the fit states: with
    the start point (no fit), with the fitted edges, and the same circuit on a
    ring against the ring evolution (the bulk alone, for comparison). The fit
    stops once an iteration gains less than a thousandth of one bulk site's
    worth of infidelity.
    """
    N = N_fit or fit_size(t)
    t0 = time.time()
    x_in = random_product_states(N, states[1], states[0])
    ring_t = evolve(model.sparse(N, "ring"), x_in, t)
    ring = 1.0 - float(np.mean(np.abs(np.sum(np.conj(ring_t) * circuit.apply(x_in, N), axis=1)) ** 2))
    target = evolve(model.sparse(N, "open"), x_in, t)
    f = _objective(circuit, N, x_in, target)
    x0 = start_angles(circuit)
    v0 = f(x0)[0]
    res = minimize(f, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-3 * ring / N, "gtol": 1e-13})
    x, v, n_snap = _snap_near_clifford_t(f, res.x, float(res.fun), 0.1 * ring / N)
    out = {"N_fit": N, "states": list(states), "angles": x.tolist(), "infid_start": v0,
           "infid_fit": v, "infid_fit_unsnapped": float(res.fun), "n_snapped": n_snap, "infid_ring": ring,
           "nit": int(res.nit), "seconds": time.time() - t0}
    log(f"edges: N_fit = {N}, {x0.size} angles, infidelity {v0:.3e} -> {res.fun:.3e} "
        f"(ring {ring:.3e}), {res.nit} iterations, {n_snap} set to multiples of pi/4 ({v:.3e}), "
        f"{out['seconds']:.0f}s")
    return out


def _snap_near_clifford_t(f, x: np.ndarray, v: float, allowance: float, window: float = 1e-2):
    """Set fitted angles to the nearest multiple of pi/4 while the fit infidelity allows it.

    Many fitted angles sit within 1e-4 of 0 or pi/2: corrections the fit
    barely needs. Synthesizing them costs as much as any other angle, and
    KN's search is slow near multiples of pi/4. Angles within ``window`` of
    one are tried from the closest, and a snap is kept while the total rise
    of the fit infidelity stays below ``allowance`` (a tenth of one bulk
    site's worth, see fit_edges). Returns the angles, the new infidelity and
    the number of snaps.
    """
    x = np.array(x, dtype=float)
    limit = v + allowance
    offsets = [(abs(math.remainder(a, QUARTER)), i) for i, a in enumerate(x)]
    n = 0
    for off, i in sorted(offsets):
        if off == 0.0:
            continue
        if off > window:
            break
        trial = x.copy()
        trial[i] -= math.remainder(trial[i], QUARTER)
        vt = f(trial)[0]
        if vt <= limit:
            x, v, n = trial, vt, n + 1
    return x, v, n


def worth_it(fit: dict) -> bool:
    """Keep fitted edges when they remove at least one bulk site's worth of error at N_fit."""
    return fit["infid_start"] - fit["infid_fit"] > fit["infid_ring"] / fit["N_fit"]


def realize_edges(fit: dict, eps: float) -> dict:
    """Synthesize the kept edge angles with KN at eps. Multiples of pi/4 cost 0 or 1 T.

    An angle within diamond distance eps of a multiple of pi/4 is set to it,
    since that is cheaper than any KN circuit and as accurate as the budget
    asks. Many fitted edge angles are like that (corrections of 1e-5 near 0 or
    pi/2), and KN's search is very slow on them. When the fit is not worth it,
    the record says so (use = False) and the open chain keeps the cell's own
    single-qubit factor at no extra cost.
    """
    if not worth_it(fit):
        return {**fit, "use": False, "eps": eps, "angles_real": None, "kn": {}, "n_rotations": 0,
                "n_kn": 0, "n_clifford_t": 0, "T_total": 0.0, "ledger_total": 0.0, "ledger2_total": 0.0}
    x = np.asarray(fit["angles"], dtype=float)
    x_real = x.copy()
    T = ledger = ledger2 = 0.0
    kn = {}
    n_exact = 0
    for i, theta in enumerate(x):
        rem = math.remainder(theta, 2 * math.pi)
        k = round(rem / QUARTER)
        if 2.0 * abs(math.sin((rem - k * QUARTER) / 2.0)) <= eps:
            x_real[i] = theta - rem + k * QUARTER
            T += 0.0 if k % 2 == 0 else 1.0
            n_exact += 1
            continue
        r = synthesize(theta, eps)
        kn[str(i)] = r
        x_real[i] = r["theta_s"]
        T += r["expected_t"]
        ledger += (1.0 - r["p"]) * r["d_branch"]
        ledger2 += (1.0 - r["p"]) * r["d_branch"] ** 2
    return {**fit, "use": True, "eps": eps, "angles_real": x_real.tolist(), "kn": kn, "n_rotations": int(x.size),
            "n_kn": len(kn), "n_clifford_t": n_exact, "T_total": T, "ledger_total": ledger, "ledger2_total": ledger2}
