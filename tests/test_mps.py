import math

import numpy as np
from scipy.sparse.linalg import expm_multiply

from snapevo.cells import CELLS
from snapevo.circuits import Layered, product_vectors, random_product_states
from snapevo.evaluate import evaluate_circuit
from snapevo.hamiltonian import XXZ
from snapevo.mps import MPS, apply_layered, reference_circuit


def _random_u2(rng):
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    return q


def _circuit(cell_name, model, t, D, seed=0):
    cell = CELLS[cell_name]
    x = cell.trotter_angles(model, t, D)
    x = x + 0.05 * np.random.default_rng(seed).normal(size=x.size)      # stag and fields nonzero
    return Layered.brickwork(cell, x)


def test_mps_equals_state_vector_on_open_chain():
    rng = np.random.default_rng(1)
    for cell_name, model in (("xxz+stag", XXZ(0.25, 0.0)), ("xxz+x", XXZ(0.25, 1.6))):
        c = _circuit(cell_name, model, 1.0, 5)
        odd = sum(1 for p, _ in c.layers if p == 1)
        for edges in (None, [(_random_u2(rng), _random_u2(rng)) for _ in range(odd)]):
            v = product_vectors(8, 1, 3)[0]
            psi = apply_layered(MPS.product(v, chi_max=10 ** 6, cutoff=0.0), c, edges)
            dense = c.apply(random_product_states(8, 1, 3), 8, bc="open", edges=edges)[0]
            assert np.abs(psi.to_dense() - dense).max() < 1e-12


def test_reference_matches_exact_evolution():
    for model in (XXZ(0.25, 0.0), XXZ(0.25, 1.6)):
        x = random_product_states(8, 2, 4)
        exact = expm_multiply(-1j * 1.5 * model.sparse(8, "open"), x.T).T
        y = reference_circuit(model, 1.5, 0.2).apply(x, 8, bc="open")
        F = np.abs(np.sum(np.conj(exact) * y, axis=1)) ** 2
        assert 1 - F.min() < 1e-10


def test_truncation_error_bar_holds():
    model = XXZ(0.25, 1.6)
    c = _circuit("xxz+x", model, 2.0, 7)
    v = product_vectors(10, 1, 5)[0]
    exact_u = apply_layered(MPS.product(v, chi_max=10 ** 6, cutoff=0.0), c)
    exact_v = apply_layered(MPS.product(v, chi_max=10 ** 6, cutoff=0.0), reference_circuit(model, 2.0, 0.2))
    u = apply_layered(MPS.product(v, chi_max=4, cutoff=0.0), c)
    w = apply_layered(MPS.product(v, chi_max=4, cutoff=0.0), reference_circuit(model, 2.0, 0.2))
    assert u.discarded > 0 and w.discarded > 0
    T_exact = math.sqrt(1 - abs(exact_v.overlap(exact_u)) ** 2)
    T_trunc = math.sqrt(max(0.0, 1 - abs(w.overlap(u)) ** 2))
    assert abs(T_trunc - T_exact) <= math.sqrt(u.discarded) + math.sqrt(w.discarded)


def test_save_load_roundtrip(tmp_path):
    model = XXZ(0.25, 0.0)
    psi = apply_layered(MPS.product(product_vectors(6, 1, 2)[0]), reference_circuit(model, 1.0, 0.25))
    psi.save(tmp_path / "ref.npz", {"t": 1.0})
    back, meta = MPS.load(tmp_path / "ref.npz")
    assert meta["t"] == 1.0 and back.center == psi.center
    assert abs(back.overlap(psi) - 1) < 1e-12


def test_evaluate_matches_dense_fidelity():
    model = XXZ(0.25, 0.0)
    c = _circuit("xxz+stag", model, 1.0, 5)
    pricing = {"18": {"Tq": 10.0, "ledger_per_qubit": 0.0, "ledger2_per_qubit": 0.0}}
    ev = evaluate_circuit(c, None, pricing, 5.0, model, 1.0, N=8, states=(7, 3), dt=0.1, cutoff=0.0,
                          log=lambda *_: None)
    x = random_product_states(8, 3, 7)
    exact = expm_multiply(-1j * 1.0 * model.sparse(8, "open"), x.T).T
    F = np.abs(np.sum(np.conj(exact) * c.apply(x, 8, bc="open"), axis=1)) ** 2
    assert math.isclose(ev["infid"], 1 - F.mean(), rel_tol=1e-6)        # reference error ~1e-8 relative
    assert math.isclose(ev["infid_total"], ev["infid"], rel_tol=1e-12) and ev["F"] == "18"


def test_F_is_the_cheapest_converged_fallback_cost():
    model = XXZ(0.25, 0.0)
    c = _circuit("xxz+stag", model, 1.0, 5)
    pricing = {f: {"Tq": 10.0 + i, "ledger_per_qubit": led, "ledger2_per_qubit": 0.0}
               for i, (f, led) in enumerate((("18", 1e-3), ("22", 2e-6), ("30", 1e-6)))}
    ev = evaluate_circuit(c, None, pricing, 5.0, model, 1.0, N=8, states=(7, 2), dt=0.1, check=False,
                          log=lambda *_: None)
    tot = {f: ev["per_F"][f]["infid_total"] for f in pricing}
    assert tot["18"] > 1.01 * tot["30"] and tot["22"] <= 1.01 * tot["30"]
    assert ev["F"] == "22" and math.isclose(ev["Tq"], 11.0)
    assert 0 < ev["fallback_share"] < 0.1
