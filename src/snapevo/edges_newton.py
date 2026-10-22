"""The ring arm's edge fit by trust-region Newton on the exact Hessian (edges.fit_edges uses L-BFGS-B).

Same problem as edges.fit_edges: the edge rotations of a realized brickwork (six ZYZ angles
per parity-1 sublayer) fitted with the bulk fixed, on an open chain of N_fit sites, to the
mean infidelity over edges.EDGE_SET product inputs against exact open-chain evolution, from
the cell's own single-qubit factor (edges.start_angles). The objective is the one of
edges._objective, rebuilt here with its exact Hessian (forward-over-reverse autodiff,
HESSIAN_CHUNK tangents at a time) so that the optimizer is
chain_snap.newton_minimize (scipy's trust-exact, a fresh exact Hessian at every accepted
point, no step cap). It stops when the predicted Newton decrease and the last step's are
both below fit_edges' tolerance, a thousandth of one bulk site's worth of infidelity
(1e-3 x ring / N_fit), with no negative curvature beyond 1e-8 of the largest eigenvalue;
or at the noise floor. Then, as fit_edges: angles within 1e-2 of a multiple of pi/4 are set
to it while the fit infidelity rises by less than a tenth of one bulk site's worth
(edges._snap_near_clifford_t), and the record has fit_edges' keys, so edges.worth_it and
edges.realize_edges take it as they are.
"""

from __future__ import annotations

import os
import time

import numpy as np

from snapevo.circuits import Layered, random_product_states
from snapevo.edges import EDGE_SET, _objective, _snap_near_clifford_t, fit_size, start_angles
from snapevo.hamiltonian import XXZ, evolve

__all__ = ["edge_objective", "fit_edges_newton", "fit_edges_lbfgs", "HESSIAN_CHUNK"]

HESSIAN_CHUNK = 8
"""Tangents per forward-over-reverse pass of the edge objective's Hessian (as jaxcost.make_hessian)."""


def edge_objective(circuit: Layered, N: int, inputs: np.ndarray, targets: np.ndarray):
    """(x -> (mean infidelity, gradient), x -> exact Hessian), jitted, bulk fixed (edges._objective's cost)."""
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
    gfun = jax.grad(cost)

    @jax.jit
    def hvp_rows(x, V):
        return jax.vmap(lambda v: jax.jvp(gfun, (x,), (v,))[1])(V)

    def f(x):
        v, g = vg(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    def h(x, rows=None):
        # forward-over-reverse, HESSIAN_CHUNK tangents at a time (jax.hessian at once needs one state
        # vector of 2^N_fit amplitudes per angle and tangent, tens of GB at N_fit = 18 and D ~ 60)
        x = jnp.asarray(x, dtype=jnp.float64)
        n = x.shape[0]
        eye = np.eye(n)
        H = np.zeros((n, n))
        for lo in range(0, n, HESSIAN_CHUNK):
            V = jnp.asarray(eye[lo: lo + HESSIAN_CHUNK])
            if V.shape[0] < HESSIAN_CHUNK:
                V = jnp.concatenate([V, jnp.zeros((HESSIAN_CHUNK - V.shape[0], n))])
            H[lo: lo + HESSIAN_CHUNK] = np.asarray(hvp_rows(x, V))[: min(HESSIAN_CHUNK, n - lo)]
        H = 0.5 * (H + H.T)
        return H if rows is None else H[np.ix_(list(rows), list(rows))]

    return f, h


def fit_edges_newton(circuit: Layered, model: XXZ, t: float, N_fit: int | None = None,
                     states: tuple[int, int] = EDGE_SET, log=print) -> dict:
    """edges.fit_edges with trust-region Newton on the exact Hessian in place of L-BFGS-B (module docstring)."""
    from snapevo.chain_snap import newton_minimize
    N = N_fit or fit_size(t)
    t0 = time.time()
    x_in = random_product_states(N, states[1], states[0])
    ring_t = evolve(model.sparse(N, "ring"), x_in, t)
    ring = 1.0 - float(np.mean(np.abs(np.sum(np.conj(ring_t) * circuit.apply(x_in, N), axis=1)) ** 2))
    target = evolve(model.sparse(N, "open"), x_in, t)
    f, h = edge_objective(circuit, N, x_in, target)
    x0 = start_angles(circuit)
    v0 = f(x0)[0]
    tol = 1e-3 * ring / N
    if x0.size:
        x_fit, info = newton_minimize(x0, f, h, tol, n_bulk=None, log=lambda s: log(f"edges: {s}"), label="newton")
    else:
        x_fit, info = x0, {"c_after": v0, "steps": 0, "hessians": 0, "stop": "nothing to fit", "pred": 0.0}
    v_fit = float(info["c_after"])
    x, v, n_snap = _snap_near_clifford_t(f, np.asarray(x_fit, dtype=float), v_fit, 0.1 * ring / N)
    out = {"N_fit": N, "states": list(states), "angles": np.asarray(x).tolist(), "infid_start": v0,
           "infid_fit": v, "infid_fit_unsnapped": v_fit, "n_snapped": n_snap, "infid_ring": ring,
           "nit": int(info["steps"]), "optimizer": "newton", "newton": {k: info.get(k) for k in
                                                                       ("steps", "hessians", "stop", "pred")},
           "seconds": time.time() - t0}
    log(f"edges: N_fit = {N}, {x0.size} angles, infidelity {v0:.3e} -> {v_fit:.3e} (ring {ring:.3e}), "
        f"{info['steps']} Newton steps ({info['stop']}), {n_snap} set to multiples of pi/4 ({v:.3e}), "
        f"{out['seconds']:.0f}s")
    return out


def fit_edges_lbfgs(circuit: Layered, model: XXZ, t: float, N_fit: int | None = None,
                    states: tuple[int, int] = EDGE_SET, log=print) -> dict:
    """edges.fit_edges with no iteration or evaluation cap (production's: maxiter 1000, maxfun 15000).

    Same objective (edges._objective), start, tolerance (L-BFGS-B's ftol = 1e-3 x ring / N_fit:
    an iteration gaining less than a thousandth of one bulk site's worth of infidelity) and near-pi/4
    setting, and the same record; identical to fit_edges wherever that stopped before its cap (all but
    a few of the production fits).
    """
    import scipy.optimize as sopt
    N = N_fit or fit_size(t)
    t0 = time.time()
    x_in = random_product_states(N, states[1], states[0])
    ring_t = evolve(model.sparse(N, "ring"), x_in, t)
    ring = 1.0 - float(np.mean(np.abs(np.sum(np.conj(ring_t) * circuit.apply(x_in, N), axis=1)) ** 2))
    target = evolve(model.sparse(N, "open"), x_in, t)
    f = _objective(circuit, N, x_in, target)
    x0 = start_angles(circuit)
    v0 = f(x0)[0]
    res = sopt.minimize(f, x0, jac=True, method="L-BFGS-B",
                        options={"maxiter": 10 ** 9, "maxfun": 10 ** 9, "ftol": 1e-3 * ring / N, "gtol": 1e-13})
    x, v, n_snap = _snap_near_clifford_t(f, res.x, float(res.fun), 0.1 * ring / N)
    out = {"N_fit": N, "states": list(states), "angles": x.tolist(), "infid_start": v0,
           "infid_fit": v, "infid_fit_unsnapped": float(res.fun), "n_snapped": n_snap, "infid_ring": ring,
           "nit": int(res.nit), "optimizer": "lbfgs (uncapped)", "message": str(res.message),
           "seconds": time.time() - t0}
    log(f"edges: N_fit = {N}, {x0.size} angles, infidelity {v0:.3e} -> {res.fun:.3e} "
        f"(ring {ring:.3e}), {res.nit} iterations (uncapped: {res.message}), {n_snap} set to multiples of pi/4 "
        f"({v:.3e}), {out['seconds']:.0f}s")
    return out
