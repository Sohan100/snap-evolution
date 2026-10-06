"""JAX value-and-gradient of C for the brickwork, and its Hessian.

The circuit structure (depth, bonds, batch) is fixed at trace time, so the
jitted graph is a flat chain of D N/2 contractions. Build once per run and
reuse: tracing costs about a second at D = 13 and N = 10.

x64 must be on before jax is imported; this module does that itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from snapevo.cells import Cell  # noqa: E402

jax.config.update("jax_enable_x64", True)

__all__ = ["make_cost", "make_hessian"]


def make_cost(cell: Cell, N: int, inputs: np.ndarray, targets: np.ndarray, D: int,
              precise: bool = False):
    """Jitted ``x -> (C, dC/dx)`` returning float and float64 numpy arrays.

    ``precise`` evaluates each state's infidelity as the squared norm of the
    output's component orthogonal to the target, which keeps ~1e-11 relative
    accuracy at C ~ 1e-9 instead of ~1e-7.
    """
    na = cell.n_angles
    tin = jnp.asarray(inputs.reshape((inputs.shape[0],) + (2,) * N))
    tg = jnp.asarray(targets)
    bonds = [[(i, (i + 1) % N) for i in range(l % 2, N, 2)] for l in range(D)]

    def apply_gate(s, U, i, j):
        s = jnp.tensordot(s, U.reshape(2, 2, 2, 2), axes=([i + 1, j + 1], [2, 3]))
        return jnp.moveaxis(s, [-2, -1], [i + 1, j + 1])

    def cost(x):
        p = x.reshape(D, na)
        s = tin
        for l in range(D):
            U = cell.jax_gate(p[l])
            for i, j in bonds[l]:
                s = apply_gate(s, U, i, j)
        out = s.reshape(tg.shape[0], -1)
        ov = jnp.sum(jnp.conj(tg) * out, axis=1)
        if precise:
            r = out - ov[:, None] * tg
            rho = jnp.sum(jnp.real(r) ** 2 + jnp.imag(r) ** 2, axis=1)
            return jnp.mean(-jnp.expm1(jnp.log1p(-rho) / N))
        return 1.0 - jnp.mean((jnp.abs(ov) ** 2) ** (1.0 / N))

    jitted = jax.jit(jax.value_and_grad(cost))

    def value_and_grad(x):
        v, g = jitted(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    value_and_grad.cost = cost
    return value_and_grad


def make_hessian(cell: Cell, N: int, inputs: np.ndarray, targets: np.ndarray, D: int,
                 method: str = "auto", autodiff_max: int = 96, grad=None):
    """Hessian of C over the flat angles, as a function ``x -> (n, n)`` array.

    ``"jax"`` is forward-over-reverse autodiff, chunked eight tangents at a
    time. ``"fd"`` is a central difference of the jitted gradient at 1e-4 rad,
    which agrees to ~5e-7 and avoids the XLA compile of very deep graphs.
    ``"auto"`` uses autodiff up to ``autodiff_max`` angles.
    """
    n_angles = cell.n_angles * D
    if method == "auto":
        method = "jax" if n_angles <= autodiff_max else "fd"
    if method == "fd":
        g = grad or make_cost(cell, N, inputs, targets, D)

        def hessian_fd(x, h: float = 1e-4):
            x = np.asarray(x, dtype=np.float64)
            H = np.zeros((x.size, x.size))
            for k in range(x.size):
                e = np.zeros(x.size)
                e[k] = h
                H[k] = (g(x + e)[1] - g(x - e)[1]) / (2 * h)
            return 0.5 * (H + H.T)

        return hessian_fd
    if method != "jax":
        raise ValueError(f"unknown Hessian method {method!r}")
    cost = make_cost(cell, N, inputs, targets, D).cost
    gfun = jax.grad(cost)
    chunk = 8

    @jax.jit
    def rows(x, V):
        return jax.vmap(lambda v: jax.jvp(gfun, (x,), (v,))[1])(V)

    def hessian(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        n = x.shape[0]
        eye = np.eye(n)
        H = np.zeros((n, n))
        for lo in range(0, n, chunk):
            V = jnp.asarray(eye[lo: lo + chunk])
            if V.shape[0] < chunk:
                V = jnp.concatenate([V, jnp.zeros((chunk - V.shape[0], n))])
            H[lo: lo + chunk] = np.asarray(rows(x, V))[: min(chunk, n - lo)]
        return 0.5 * (H + H.T)

    return hessian
