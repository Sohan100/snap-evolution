import numpy as np
import pytest

from snapevo.cells import cell_for
from snapevo.circuits import Layered, product_vectors, random_product_states
from snapevo.edges import pairs
from snapevo.hamiltonian import XXZ, evolve
from snapevo.mps import MPS, apply_layered
from snapevo.mps_train import _run_input, mps_from_dense, sublayer_environments
from snapevo.train_chain import trotter_z


def _gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:                                             # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="no GPU here")


def _random_mps(N, chi, seed):
    rng = np.random.default_rng(seed)
    dims = [1] + [min(chi, 2 ** min(i, N - i)) for i in range(1, N)] + [1]
    A = [rng.normal(size=(dims[i], 2, dims[i + 1])) + 1j * rng.normal(size=(dims[i], 2, dims[i + 1]))
         for i in range(N)]
    v = MPS(A).to_dense()
    return mps_from_dense(v / np.linalg.norm(v))


@pytest.mark.parametrize("move", ("right", "left"))
@pytest.mark.parametrize("cutoff,chi_max", ((0.0, 10 ** 6), (1e-6, 10 ** 6), (1e-12, 5)))
def test_two_qubit_gate_truncates_as_the_cpu_code(move, cutoff, chi_max):
    from snapevo.mps_gpu import GMPS
    rng = np.random.default_rng(3)
    psi = _random_mps(10, 16, 1)
    psi.chi_max, psi.cutoff = chi_max, cutoff
    g = GMPS.from_cpu(psi)
    U = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    for i in (4, 2, 6, 3):
        psi.apply_2q(U, i, move=move)
        g.apply_2q(U, i, move=move)
        assert g.center == psi.center and g.bond_dims == psi.bond_dims
    assert g.discarded == pytest.approx(psi.discarded, rel=1e-8, abs=1e-15)
    a, b = psi.to_dense(), g.to_cpu().to_dense()
    assert abs(abs(np.vdot(a, b)) - 1.0) < 1e-12
    c = g.A[g.center].get()
    assert abs(np.vdot(c, c).real - 1.0) < 1e-12                  # canonical about the center, normalized


@pytest.mark.parametrize("hx", (0.0, 1.6))
def test_run_input_and_environments_match_the_cpu_code(hx):
    from snapevo.mps_gpu import GMPS, apply_layered_gpu, run_input_gpu, sublayer_environments_gpu
    model = XXZ(0.25, hx)
    cell = cell_for(model)
    N, t, D = 10, 1.0, 6
    vecs = product_vectors(N, 2, 9)
    states = random_product_states(N, 2, 9)
    targets = [mps_from_dense(v) for v in evolve(model.sparse(N, "open"), states, t)]
    z = trotter_z(model, cell, t, D) + 0.04
    nb = D * cell.n_angles
    x, e = z[:nb], z[nb:].reshape(-1, 6)
    for cutoff, chi in ((0.0, 10 ** 6), (1e-10, 12)):
        for k in range(2):
            a = _run_input(cell, x, e, vecs[k], targets[k], chi, cutoff, True)
            b = run_input_gpu(cell, x, e, vecs[k], targets[k], chi, cutoff, True)
            assert b["F"] == pytest.approx(a["F"], rel=1e-11)
            for key in ("E_bulk", "E_last", "E_first"):
                assert np.allclose(b[key], a[key], rtol=0, atol=1e-10 * np.abs(a[key]).max())
            assert b["chi_u"] == a["chi_u"] and b["discarded_u"] == pytest.approx(a["discarded_u"], rel=1e-6,
                                                                                    abs=1e-18)
    # one sublayer's environments directly
    circ = Layered.brickwork(cell, x)
    psi = MPS.product(vecs[0], chi_max=10 ** 6, cutoff=0.0)
    apply_layered(psi, Layered(cell, circ.layers[:3]), pairs(e)[:1])
    gp = GMPS.product(vecs[0], chi_max=10 ** 6, cutoff=0.0)
    apply_layered_gpu(gp, Layered(cell, circ.layers[:3]), pairs(e)[:1])
    par, ang = circ.layers[3]
    pr = pairs(e)[1]
    Ea = sublayer_environments(psi, targets[0], par, cell.gate(ang), pr)
    Eb = sublayer_environments_gpu(gp, GMPS.from_cpu(targets[0]), par, cell.gate(ang), pr)
    for u, v in zip(Ea, Eb):
        assert np.allclose(np.asarray(v), np.asarray(u), rtol=0, atol=1e-11)
