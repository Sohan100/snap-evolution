import numpy as np

from snapevo.cells import CELLS
from snapevo.circuits import Layered, random_product_states
from snapevo.edges import fit_edges, pairs, start_angles, u_zyz, worth_it, zyz_angles
from snapevo.hamiltonian import XXZ, evolve


def test_zyz_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        assert abs(abs(np.vdot(u_zyz(*zyz_angles(q)).ravel(), q.ravel())) - 2) < 1e-12


def test_start_point_is_the_cells_own_factor():
    for cell_name, model in (("xxz+stag", XXZ(0.25, 0.0)), ("xxz+x", XXZ(0.25, 1.6))):
        cell = CELLS[cell_name]
        x = cell.trotter_angles(model, 1.0, 5) + 0.1 * np.random.default_rng(1).normal(size=5 * cell.n_angles)
        c = Layered.brickwork(cell, x)
        s = random_product_states(8, 2, 0)
        a = c.apply(s, 8, bc="open")
        b = c.apply(s, 8, bc="open", edges=pairs(start_angles(c)))
        assert np.allclose(np.abs(np.sum(np.conj(a) * b, axis=1)), 1.0, atol=1e-12)


def test_fit_lowers_the_open_chain_infidelity():
    model = XXZ(0.25, 0.0)
    cell = CELLS["xxz+stag"]
    x = cell.trotter_angles(model, 1.0, 5).reshape(5, 4)
    x[:, 3] = [0.02, -0.01, 0.015, -0.02, 0.01]
    c = Layered.brickwork(cell, x.ravel())
    fit = fit_edges(c, model, 1.0, N_fit=8, states=(3, 6), maxiter=200, log=lambda *_: None)
    assert fit["infid_fit"] < fit["infid_start"]
    assert worth_it({**fit, "infid_fit": 0.0}) and not worth_it({**fit, "infid_fit": fit["infid_start"]})
    s = random_product_states(8, 6, 3)
    tg = evolve(model.sparse(8, "open"), s, 1.0)
    F = np.abs(np.sum(np.conj(tg) * c.apply(s, 8, bc="open", edges=pairs(fit["angles"])), axis=1)) ** 2
    assert np.isclose(1 - F.mean(), fit["infid_fit"], rtol=1e-8)
