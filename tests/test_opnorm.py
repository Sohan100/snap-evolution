import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse.linalg as spla
from scipy.optimize import minimize_scalar

from snapevo import opnorm as on
from snapevo.circuits import held_out, local_infidelity, product_vectors, state_pairs
from snapevo.crosscheck import arc_width, process_fidelity
from snapevo.hamiltonian import get_model
from snapevo.mps import MPS, apply_layered

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
T = 2.0
FIELDS = {"1.6": "A", "0": "B"}            # snapevo's model keys: A is h_x = 1.6, B is h_x = 0
SNAP = {"1.6": "xxz_d0.25_hx1.6/t2/real_b1.0e-06_D17.json",   # both keep fitted edges
        "0": "xxz_d0.25_hx0/t2/real_b1.0e-05_D11.json"}
BASELINE = "xxz_d0.25_hx0/t2/baseline_o2_r4_e1e-04.json"
SCRATCH_OUT = Path(os.environ.get("OPNORM_TEST_OUT", "/pscratch/sd/s/sohan100/snapevo-scratch/opnorm2/pytest"))


def _snap(field):
    p = RUNS / SNAP[field]
    if not p.exists():
        pytest.skip(f"{p} not in this checkout")
    c = on.snap_circuit(json.loads(p.read_text()))
    assert c.edges is not None                       # the open chain runs with fitted edges
    return c


def _circuit(field, kind):
    model = get_model(FIELDS[field])
    return model, (_snap(field) if kind == "snap" else on.trotter(model, T, 2, 3))


@pytest.mark.parametrize("field", ["1.6", "0"])
@pytest.mark.parametrize("kind", ["snap", "trotter"])
@pytest.mark.parametrize("bc", ["ring", "open"])
@pytest.mark.parametrize("N", [6, 8])
def test_lanczos_matches_full_diagonalization(field, kind, bc, N):
    model, c = _circuit(field, kind)
    a = on.arc(c, model, T, N, bc)
    d = on.dense_metrics(c, model, T, N, bc)
    assert abs(a["w"] - d["w"]) < 1e-8
    assert not a["saturated"] and a["w"] < 1.0
    assert max(a["res"]) <= max(1e-10 * max(map(abs, a["lambda"])), 1e-13) * 10
    assert abs(a["w_phase"] - a["w"]) < 1e-8                  # the Ritz vectors' phases give the same arc
    assert math.isclose(a["eps_op"], 2 * math.sin(d["w"] / 4), rel_tol=1e-7)
    assert math.isclose(a["D_coh"], 2 * math.sin(d["w"] / 2), rel_tol=1e-7)


def _bruteforce_eps(U, V):
    f = lambda phi: np.linalg.norm(U - np.exp(1j * phi) * V, 2)          # noqa: E731
    grid = np.linspace(-math.pi, math.pi, 721)
    k = int(np.argmin([f(p) for p in grid]))
    r = minimize_scalar(f, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, 720)]), method="bounded",
                        options={"xatol": 1e-12})
    return float(r.fun)


@pytest.mark.parametrize("field", ["1.6", "0"])
@pytest.mark.parametrize("kind", ["snap", "trotter"])
@pytest.mark.parametrize("bc", ["ring", "open"])
def test_eps_op_is_the_min_over_phase_of_the_spectral_norm(field, kind, bc):
    N = 6
    model, c = _circuit(field, kind)
    U, V = on.dense_unitaries(c, model, T, N, bc)
    brute = _bruteforce_eps(U, V)
    assert abs(on.arc(c, model, T, N, bc)["eps_op"] - brute) < 1e-8
    assert abs(on.dense_metrics(c, model, T, N, bc)["eps_op"] - brute) < 1e-8


def test_wide_arc_dense_exact_and_lanczos_flags_saturation():
    """A crude circuit (one first-order step for t = 2) has w > pi at N = 8: the dense path still gives
    eps_op exactly, and the sine-based Lanczos path flags it instead of returning a wrong arc."""
    model = get_model("A")
    c = on.trotter(model, T, 1, 1)
    N = 8
    d = on.dense_metrics(c, model, T, N, "ring")
    assert d["w"] > math.pi
    U, V = on.dense_unitaries(c, model, T, N, "ring")
    assert abs(d["eps_op"] - _bruteforce_eps(U, V)) < 1e-8
    a = on.arc(c, model, T, N, "ring")
    assert a["saturated"]


@pytest.mark.parametrize("bc", ["ring", "open"])
def test_C_matches_local_infidelity(bc):
    model, c = _circuit("1.6", "snap")
    N = 8
    f = on.infidelities(c, model, T, N, bc)
    seed, n = on.PRODUCT_SET
    tin, tg = state_pairs(model, N, T, n, seed, bc=bc)
    ref = local_infidelity(c.layered.apply(tin, N, bc=bc, edges=c.edges if bc == "open" else None), tg, N)
    assert math.isclose(f["C"], ref, rel_tol=1e-8)
    assert f["max_infid"] >= f["mean_infid"] > 0
    if bc == "ring":                              # same Haar draws as crosscheck.process_fidelity
        p = process_fidelity(c.layered, model, T, N, exact_max=0, n_haar=16, seed=7)
        assert math.isclose(f["c_e"], p["c_e"], rel_tol=1e-6)


def test_baseline_rebuild_reproduces_recorded_C_ring():
    p = RUNS / BASELINE
    if not p.exists():
        pytest.skip(f"{p} not in this checkout")
    rec = json.loads(p.read_text())
    model = get_model("B")
    c = on.baseline_circuit(rec, model)
    assert math.isclose(held_out(c.layered, model, T, sizes=(10,))["worst"], rec["C_ring"], rel_tol=1e-10)
    pure = on.trotter(model, T, rec["order"], rec["steps"]).layered
    assert any(not np.allclose(a, b) for (_, a), (_, b) in zip(c.layered.layers, pure.layers))


@pytest.mark.parametrize("field", ["1.6", "0"])
@pytest.mark.parametrize("N", [6, 8])
def test_open_chain_unitary_matches_layered_apply(field, N):
    model, c = _circuit(field, "snap")
    d = 2 ** N
    eye = np.eye(d, dtype=complex)
    for bc, edges in (("open", c.edges), ("open", None), ("ring", None)):
        ops = on.layered_ops(c.layered, N, bc, edges)
        U = on.apply_ops(ops, eye, N)
        ref = c.layered.apply(eye, N, bc=bc, edges=edges).T             # columns U|e_k>
        np.testing.assert_allclose(U, ref, atol=1e-12)
        np.testing.assert_allclose(on.apply_ops(ops, eye, N, dagger=True), ref.conj().T, atol=1e-12)
    # and the MPS of the 100-site evaluation applies the same open-chain circuit
    vec = product_vectors(N, 1, 5)[0]
    psi = apply_layered(MPS.product(vec, chi_max=64, cutoff=0.0), c.layered, c.edges)
    x = np.array([1.0 + 0j])
    for v in vec:
        x = np.kron(x, v)
    np.testing.assert_allclose(psi.to_dense(), on.apply_ops(c.ops(N, "open"), x.reshape(-1, 1), N).ravel(), atol=1e-10)


@pytest.mark.parametrize("field", ["1.6", "0"])
def test_crosscheck_arc_width_agrees_on_the_ring(field):
    model, c = _circuit(field, "snap")
    N = 8
    a = on.arc(c, model, T, N, "ring")
    x = arc_width(c.layered, model, T, N)
    assert abs(a["w"] - x["w"]) < 1e-8
    assert abs(a["phi0"] - x["phi0"]) < 1e-9                  # the same phi0 convention
    assert a["matvecs"] < x["matvecs"]


@pytest.mark.parametrize("bc", ["ring", "open"])
def test_propagator_is_exact(bc):
    model = get_model("A")
    for N in (6, 10):
        p = on.Propagator(model, N, bc, 4.0)
        assert p.check() < 1e-12
        rng = np.random.default_rng(3)
        X = rng.normal(size=(2 ** N, 3)) + 1j * rng.normal(size=(2 ** N, 3))
        Y = p.apply(X, [-1.0, 1.0, -1.0])
        H = model.sparse(N, bc)
        for k, s in enumerate((-1.0, 1.0, -1.0)):
            ref = spla.expm_multiply(s * 1j * 4.0 * H, X[:, k], traceA=0.0)
            assert np.linalg.norm(Y[:, k] - ref) < 1e-12 * np.linalg.norm(X[:, k])


def test_gate_lists_and_builders():
    model, c = _circuit("0", "snap")
    N = 6

    class G:                                                  # duck-typed like baseline_lit.Gate
        def __init__(self, sites, U):
            self.sites, self._U = sites, U

        def matrix(self):
            return self._U

    ops = c.ops(N, "open")
    w = on.arc(c, model, T, N, "open")["w"]
    fixed = on.Circuit(gates=ops, N_fixed=N, bcs=("open",))
    built = on.Circuit(gates=lambda n, bc: [G(s, U) for s, U in c.ops(n, bc)], bcs=("open",))
    assert abs(on.arc(fixed, model, T, N, "open")["w"] - w) < 1e-10
    assert abs(on.arc(built, model, T, N, "open")["w"] - w) < 1e-10
    with pytest.raises(ValueError):
        fixed.ops(8, "open")
    with pytest.raises(ValueError):
        built.ops(N, "ring")
    # a non-adjacent pair goes through the generic path
    rng = np.random.default_rng(0)
    U4 = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    X = rng.normal(size=(2 ** N, 2)) + 0j
    Y = on.apply_ops([((1, 4), U4)], X, N)
    swap = np.eye(4)[[0, 2, 1, 3]]
    ref = on.apply_ops([((3, 4), swap), ((2, 3), swap), ((1, 2), U4), ((2, 3), swap), ((3, 4), swap)], X, N)
    np.testing.assert_allclose(Y, ref, atol=1e-12)


def test_measure_floor_and_dense_check():
    model, c = _circuit("1.6", "snap")
    m = on.measure(c, model, T, 8, "open")
    assert m["floor_ok"] and m["floor"] <= m["floor_max"] <= m["eps_op"]
    assert abs(m["dense_dw"]) < 1e-8
    assert m["prop"]["check"] < 1e-12
    assert 0.5 < m["kappa"] < 5


def test_extrapolate():
    e = on.extrapolate([8, 10, 12, 14, 18], [0.17, 0.21, 0.25, 0.29, 0.37], targets=(50, 100), c=1e-6,
                       checks={18: 0.37})
    assert e["fit_sizes"] == [10, 12, 14]
    assert math.isclose(e["alpha"], 0.02) and math.isclose(e["beta"], 0.01, abs_tol=1e-12)
    assert math.isclose(e["targets"]["100"]["w"], 2.01)
    assert not e["targets"]["100"]["saturated"]
    assert e["targets"]["100"]["floor"] <= e["targets"]["100"]["eps_op"]
    assert abs(e["checks"]["18"]["rel_err"]) < 1e-12
    s = on.extrapolate([10, 12, 14], [0.5, 0.6, 0.7], targets=(100,))
    assert s["targets"]["100"]["saturated"] and s["targets"]["100"]["D_coh"] == 2.0
    assert math.isclose(on.floor_eps(1e-4, 100), math.sqrt(1 - (1 - 1e-4) ** 100))


def test_runner_task_and_table():
    p = RUNS / SNAP["0"]
    if not p.exists():
        pytest.skip(f"{p} not in this checkout")
    out = SCRATCH_OUT / f"run_{os.getpid()}"
    py = [sys.executable, str(REPO / "cluster" / "opnorm_run.py")]
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    name = "snap:0:2:b1.0e-05_D11@6,8"
    r = subprocess.run(py + ["task", name, "--out", str(out)], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    files = sorted((out / "xxz_d0.25_hx0" / "t2" / "opnorm").glob("*.json"))
    assert len(files) == 4                                   # N = 6, 8, ring and open
    rec = json.loads(files[0].read_text())
    assert rec["circuit"]["source"] == "snap" and rec["result"]["floor_ok"]
    r = subprocess.run(py + ["table", "--out", str(out), "--fit-N", "8"], capture_output=True, text=True, env=env,
                       timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "b1.0e-05_D11" in r.stdout
    r = subprocess.run(py + ["task", name], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode != 0 and "--real" in (r.stdout + r.stderr)      # never writes runs/ by default
