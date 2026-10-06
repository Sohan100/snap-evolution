import numpy as np
import scipy.sparse as sp

from snapevo.cells import CELLS
from snapevo.circuits import Layered, local_infidelity, random_product_states, state_pairs
from snapevo.environment import Environments
from snapevo.hamiltonian import XXZ

X = sp.csr_matrix(np.array([[0, 1], [1, 0]], complex))
Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]], complex))
Z = sp.csr_matrix(np.array([[1, 0], [0, -1]], complex))


def _pauli_ring(N, delta):
    def emb(ops):
        out = None
        for s in range(N):
            m = ops.get(s, sp.identity(2, format="csr", dtype=complex))
            out = m if out is None else sp.kron(out, m, format="csr")
        return out
    H = sum(emb({i: X, (i + 1) % N: X}) + emb({i: Y, (i + 1) % N: Y}) + delta * emb({i: Z, (i + 1) % N: Z})
            for i in range(N))
    F = sum(emb({i: X}) for i in range(N))
    return H, F


def test_spin_units():
    H, F = _pauli_ring(6, 0.25)
    assert abs(XXZ(0.25, 0.0).sparse(6) - H / 4).max() < 1e-14
    assert abs(XXZ(0.25, 1.6).sparse(6) - (H / 4 + 0.8 * F)).max() < 1e-14


def test_jax_gate_matches_numpy():
    rng = np.random.default_rng(1)
    for cell in CELLS.values():
        a = rng.normal(size=cell.n_angles)
        assert np.abs(np.asarray(cell.jax_gate(a)) - cell.gate(a)).max() < 1e-13


def test_dagger_inverts():
    cell = CELLS["xxz+x"]
    c = Layered.brickwork(cell, np.random.default_rng(2).normal(size=5 * 4))
    s = random_product_states(6, 3, 5)
    assert np.abs(c.apply(c.apply(s, 6), 6, dagger=True) - s).max() < 1e-12


def test_trotter_converges():
    model = XXZ(0.25, 1.6)
    cell = CELLS["xxz+x"]
    tin, tg = state_pairs(model, 6, 1.0, 4, 3)
    err = [local_infidelity(Layered.brickwork(cell, cell.trotter_angles(model, 1.0, D)).apply(tin, 6), tg, 6)
           for D in (8, 16, 32)]
    assert err[0] > err[1] > err[2] and err[1] / err[2] > 3


def test_environment_curve_is_exact():
    model = XXZ(0.25, 0.0)
    cell = CELLS["xxz+stag"]
    x = cell.trotter_angles(model, 1.0, 4) + 0.01
    tin, tg = state_pairs(model, 6, 1.0, 4, 3)
    env = Environments(x, cell, tin, tg, 6)
    for i in (0, 3, 9, 15):                       # includes the staggered slot (harmonic 2)
        cv = env.curve(i)
        assert cv.residual < 1e-10
        y = x.copy()
        y[i] = 0.37
        direct = local_infidelity(Layered.brickwork(cell, y).apply(tin, 6), tg, 6)
        assert abs(cv.c_at(0.37) - direct) < 1e-12
