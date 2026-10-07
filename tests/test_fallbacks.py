import numpy as np

from snapevo.baseline import realize_baseline, trotter_circuit
from snapevo.cells import CELLS
from snapevo.circuits import Layered
from snapevo.edges import u_zyz
from snapevo.evaluate import mps_infidelity
from snapevo.fallbacks import _rz, sample_baseline, sample_circuit, slot_gates
from snapevo.hamiltonian import XXZ

QUIET = dict(log=lambda *_: None)


def test_slot_gates_with_rz_are_the_cell_rotations():
    cell = CELLS["xxz+x"]
    th = 0.37
    g = slot_gates(_rz(th))
    assert np.allclose(g["zz"], cell.two_qubit_part([0, 0, th, 0, 0]), atol=1e-14)
    assert np.allclose(g["yy"], cell.two_qubit_part([0, th, 0, 0, 0]), atol=1e-14)
    assert np.allclose(g["xx"], cell.two_qubit_part([th, 0, 0, 0, 0]), atol=1e-14)
    assert np.allclose(g["x"], cell.one_qubit_factor([0, 0, 0, th, 0])[0], atol=1e-14)
    assert np.allclose(_rz(0.2) @ g["y"] @ _rz(-0.5), u_zyz(0.2, th, -0.5), atol=1e-14)


def test_baseline_success_branch_draws_match_the_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("SNAPEVO_RUNS", str(tmp_path))
    model, t, N = XXZ(0.25, 1.6), 0.5, 8
    circ = trotter_circuit(model, t, 2, 1)
    rb = realize_baseline(circ, 1e-2)
    kw = dict(N=N, states=(5, 3), dt=0.1, chi_max=64, cutoff=1e-14)
    ref = mps_infidelity(rb["circuit"], None, model, t, check=False, **QUIET, **kw)
    s = sample_baseline(circ, rb["circuit"], rb["synth"], 1e-2, model, t, samples=2, success_only=True, **QUIET, **kw)
    assert abs(s["infid"] - ref["infid"]) < 1e-12 and s["fallbacks_mean"] == 0
    d = sample_baseline(circ, rb["circuit"], rb["synth"], 1e-2, model, t, samples=6, **QUIET, **kw)
    assert d["fallbacks_mean"] > 0 and d["infid"] > 0


def test_brickwork_with_edges_matches_the_evaluation():
    """Both cells, random angles, fitted-style edges: the drawn circuit with no fallbacks is the evaluated one."""
    rng = np.random.default_rng(4)
    kw = dict(N=8, states=(9, 2), dt=0.1, chi_max=64, cutoff=1e-14)
    for name, model in (("xxz+stag", XXZ(0.25, 0.0)), ("xxz+x", XXZ(0.25, 1.6))):
        cell = CELLS[name]
        real = Layered.brickwork(cell, rng.uniform(-0.6, 0.6, 4 * cell.n_angles))
        for edges in (None, {"use": True, "angles_real": rng.uniform(-1, 1, 12).tolist(), "kn": {}}):
            from snapevo.edges import pairs
            ref = mps_infidelity(real, pairs(edges["angles_real"]) if edges else None, model, 0.5, check=False,
                                 **QUIET, **kw)
            s = sample_circuit(real, lambda l, s: None, edges, model, 0.5, samples=1, **QUIET, **kw)
            assert abs(s["infid"] - ref["infid"]) < 1e-12, (name, edges is not None)
