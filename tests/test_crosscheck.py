import math

import numpy as np

from snapevo.baseline import rotation_tally, trotter_circuit
from snapevo.cells import CELLS
from snapevo.circuits import Layered
from snapevo.hamiltonian import XXZ
from snapevo.crosscheck import arc_width, arc_width_dense, extrapolate


def test_lanczos_matches_dense():
    model = XXZ(0.25, 1.6)
    cell = CELLS["xxz+x"]
    c = Layered.brickwork(cell, cell.trotter_angles(model, 0.5, 6))
    for N in (6, 8):
        w_dense = arc_width_dense(c, model, 0.5, N)
        r = arc_width(c, model, 0.5, N)
        assert abs(r["w"] - w_dense) < 1e-7 * max(1.0, w_dense)
        assert math.isclose(r["D_coh"], 2 * math.sin(w_dense / 2), rel_tol=1e-6)


def test_extrapolation_affine():
    e = extrapolate([8, 10, 12], [0.08, 0.10, 0.12], 100)
    assert math.isclose(e["w_affine"], 1.0, rel_tol=1e-9)
    assert e["w"] >= e["w_power"] - 1e-12


def test_trotter_circuit_order2_tally():
    model = XXZ(0.25, 1.6)
    c = trotter_circuit(model, 1.0, 2, 3)
    # field layers merge across steps: X(dt/2) E O E X(dt) E O E X(dt) E O E X(dt/2)
    kinds = [("X" if a[0] == 0 and a[3] != 0 else "B") for _p, a in c.layers]
    assert kinds.count("X") == 4
    tally = rotation_tally(c)
    assert all(v > 0 for v in tally.values())
    np.testing.assert_allclose(sum(a[3] for _p, a in c.layers if a[3]), 1.6 * 1.0)


def test_process_fidelity_exact_vs_haar():
    from snapevo.crosscheck import process_fidelity
    model = XXZ(0.25, 1.6)
    cell = CELLS["xxz+x"]
    c = Layered.brickwork(cell, cell.trotter_angles(model, 0.5, 6))
    ex = process_fidelity(c, model, 0.5, 8)
    hr = process_fidelity(c, model, 0.5, 8, exact_max=6, n_haar=64)
    assert ex["method"] == "exact" and hr["method"] == "haar64"
    assert abs(hr["infid"] - ex["infid"]) < 4 * hr["stderr"] + 0.05 * ex["infid"]
    # a circuit equal to the target up to Trotter error has a small, positive infidelity
    assert 0 < ex["infid"] < 1e-2


def test_fidelity_bound():
    from snapevo.evaluate import fidelity_bound
    assert fidelity_bound(1e-4, 0.0, 0.0) == 1e-4
    assert math.isclose(fidelity_bound(0.0, 2e-3, 0.0), 1e-6)
    assert math.isclose(fidelity_bound(1e-4, 2e-3, 1e-7), (1e-2 + 1e-3) ** 2 + 1e-7)
    assert fidelity_bound(0.9, 1.0, 0.5) == 1.0
