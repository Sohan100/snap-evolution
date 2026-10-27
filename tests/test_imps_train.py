"""The infinite-chain trainer (snapevo.imps_train) and its pieces in snapevo.imps: random iMPS inputs in canonical
form, the gate layer inside the transfer matrix, the exact gradient, the difference Hessian, the exact-circuit
limit, and (as a code check only) c_inf against exact state vectors of the same inputs on a 16-site ring.

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0.
"""

import numpy as np
import pytest

from snapevo import imps as im
from snapevo import imps_train as it
from snapevo.cells import cell_for
from snapevo.circuits import Layered
from snapevo.hamiltonian import XXZ, evolve
from snapevo.mps import reference_circuit

FIELDS = (1.6, 0.0)


def _ref(model, t, phi):
    psi, _ = im.reference_state(model, t, phi, 0.1, 2048, 1e-14)
    return psi


def test_random_imps_canonical_and_normalized():
    a = im.random_imps(2, 2, seed=(it.TRAIN_SEED, 0))
    b = im.random_imps(2, 2, seed=(it.TRAIN_SEED, 1))
    for psi in (a, b):
        for j in range(2):
            B = psi.B[j]
            M = B.reshape(B.shape[0], -1)
            assert np.allclose(M @ M.conj().T, np.eye(B.shape[0]), atol=1e-13)                 # right isometry
            rho = np.einsum("asb,a,asd->bd", B.conj(), psi.S[j] ** 2, B)
            assert np.allclose(rho, np.diag(psi.S[(j + 1) % 2] ** 2), atol=1e-13)             # left condition
            assert abs(np.sum(psi.S[j] ** 2) - 1) < 1e-13
        assert abs(psi.norm_eigenvalue() - 1) < 1e-12
        r = psi.rdm(0, 2)
        assert abs(np.trace(r) - 1) < 1e-12 and np.linalg.eigvalsh(r).min() > -1e-13
    assert not np.allclose(a.B[0], b.B[0])


@pytest.mark.parametrize("hx", FIELDS)
def test_gate_derivatives_exact(hx):
    cell = cell_for(XXZ(0.25, hx))
    a = np.random.default_rng(1).normal(size=cell.n_angles)
    for k, dU in enumerate(it.gate_derivatives(cell, a)):
        e = np.zeros_like(a)
        e[k] = 1e-6
        fd = (cell.gate(a + e) - cell.gate(a - e)) / 2e-6
        assert np.abs(fd - dU).max() < 1e-8


def test_gate_transfer_identity_and_arpack():
    psi = im.random_imps(2, 2, seed=(it.TRAIN_SEED, 2))
    phi = im.random_imps(2, 2, seed=(it.TRAIN_SEED, 3))
    rng = np.random.default_rng(4)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    for p in (0, 1):
        assert abs(im.GateTransfer(psi, psi, np.eye(4), p).eig()["lambda"] - 1) < 1e-12
        T = im.GateTransfer(phi, psi, q, p)
        d, s = T.eig(), T.eig(dense_max=0)
        assert abs(d["lambda"] - s["lambda"]) < 1e-12 and s["residual"] < 1e-10
        G = T.env(d["l"], d["r"])                                   # lambda l^dag r = sum U * G
        assert abs(np.sum(T.U4 * G) / d["lr"] - d["lambda"]) < 1e-12


@pytest.mark.parametrize("hx", FIELDS)
def test_gradient_matches_finite_differences(hx):
    model = XXZ(0.25, hx)
    cell = cell_for(model)
    t, D = 0.5, 5
    phi = it.make_input("train", 0)
    psi = _ref(model, t, phi)
    x = cell.trotter_angles(model, t, D) + 0.05 * np.random.default_rng(2).normal(size=cell.n_angles * D)
    r = it.input_eval(cell, Layered.brickwork(cell, x).layers, phi, psi, grad=True)
    v = it.input_eval(cell, Layered.brickwork(cell, x).layers, phi, psi, grad=False)
    assert abs(r["abs"] - v["abs"]) < 1e-13
    g = r["grad"].reshape(-1)
    h = 1e-5
    for j in range(0, x.size, 4):
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        fd = (it.input_eval(cell, Layered.brickwork(cell, xp).layers, phi, psi, grad=False)["abs"]
              - it.input_eval(cell, Layered.brickwork(cell, xm).layers, phi, psi, grad=False)["abs"]) / (2 * h)
        assert abs(g[j] - fd) < 1e-5 * max(abs(fd), 1e-3)


@pytest.mark.parametrize("hx", FIELDS)
def test_exact_circuit_gives_zero(hx):
    model = XXZ(0.25, hx)
    t = 0.5
    phi = it.make_input("train", 1)
    psi = _ref(model, t, phi)
    ref = reference_circuit(model, t, 0.1)                          # the same formula as the reference
    v = it.input_eval(ref.cell, ref.layers, phi, psi, grad=False)
    assert 1 - v["abs"] < 1e-12


@pytest.mark.parametrize("hx", FIELDS)
def test_c_inf_against_exact_ring(hx):
    """Code check only: the same chi = 2 cell as a periodic MPS on 12 and 16 sites, exact e^{-iHt}."""
    model = XXZ(0.25, hx)
    cell = cell_for(model)
    t, D = 0.5, 5
    lay = Layered.brickwork(cell, cell.trotter_angles(model, t, D))
    for k in range(2):
        phi = it.make_input("train", k)
        psi = _ref(model, t, phi)
        cinf = 1 - it.input_eval(cell, lay.layers, phi, psi, grad=False)["abs"]
        err = []
        for N in (12, 16):
            s = im.ring_state_imps(phi, N)[None, :]
            F = abs(np.vdot(evolve(model.sparse(N, "ring"), s, t)[0], lay.apply(s, N, bc="ring")[0])) ** 2
            err.append(abs((1 - F ** (1 / N)) / cinf - 1))
        assert err[1] < 2e-2 and err[1] < err[0] + 1e-4


def test_hessian_matches_second_differences(tmp_path):
    model = XXZ(0.25, 1.6)
    t, D = 0.5, 3
    P = it.Problem(model, t, workers=2, n_train=2, n_test=2, cache=tmp_path)
    try:
        P.build_references()
        x = P.cell.trotter_angles(model, t, D) + 0.03 * np.random.default_rng(3).normal(size=P.cell.n_angles * D)
        c, g = P.cost_grad(x)
        assert abs(c - P.value(x)["c"]) < 1e-13
        H = P.hessian(x)
        assert np.allclose(H, H.T)
        rng = np.random.default_rng(5)
        for _ in range(2):
            d = rng.normal(size=x.size)
            d /= np.linalg.norm(d)
            e = 1e-3
            c2 = (P.value(x + e * d)["c"] - 2 * c + P.value(x - e * d)["c"]) / e ** 2
            assert abs(d @ H @ d / c2 - 1) < 1e-5
    finally:
        P.close()


def test_lbfgs_ladder_writes_its_own_arm(tmp_path):
    import json
    model = XXZ(0.25, 1.6)
    runs = tmp_path / "runs"
    out = it.train_imps_ladder(model, 0.5, [2, 3], runs=runs, workers=2, n_train=2, n_test=2, stall=None,
                               target=None, optimizer="lbfgs", log=lambda s: None)
    assert out.parent.name == "imps_train_n2_lbfgs"
    doc = json.loads(out.read_text())
    assert doc["optimizer"].startswith("L-BFGS-B")
    for r in doc["results"]:
        assert r["optimizer"] == "lbfgs" and r["C_val"] is not None
        for v in r["runs"].values():
            assert v["stop"] in ("converged", "noise floor", "stalled") and v["hessians"] >= 2
            assert v["n_neg_after"] is not None
    with pytest.raises(ValueError):
        it.train_imps_ladder(model, 0.5, [2], runs=runs, workers=1, n_train=2, n_test=0, optimizer="bfgs")


def test_split_starts_then_combine_match_the_normal_ladder(tmp_path):
    import json
    model = XXZ(0.25, 1.6)
    kw = dict(workers=2, n_train=2, n_test=2, stall=None, target=None, log=lambda s: None)
    full = json.loads(it.train_imps_ladder(model, 0.5, [2, 3], runs=tmp_path / "a", **kw).read_text())["results"]
    runs = tmp_path / "b"
    ck = runs / model.tag / "t0.5" / "imps_train_n2" / "ckpt"
    steps = {}
    for D in (2, 3):
        st = it.ladder_status(model, 0.5, [2, 3], runs=runs, n_train=2, stall=None, target=None)
        assert not st["finished"] and st["D"] == D
        assert st["starts"] == (["trotter", "random"] if D == 2 else ["trotter", "warm", "random"])
        for s in st["starts"]:
            it.train_imps_ladder(model, 0.5, [2, 3], runs=runs, only_start=s, hold_lock=False, **kw)
            steps[(D, s)] = json.loads((ck / f"D{D}_{s}.json").read_text())["steps"]
        it.train_imps_ladder(model, 0.5, [d for d in (2, 3) if d <= D], runs=runs, hold_lock=False, **kw)
    split = json.loads((runs / model.tag / "t0.5" / "imps_train_n2" / "ladder.json").read_text())["results"]
    assert [r["D"] for r in split] == [r["D"] for r in full] == [2, 3]
    for a, b in zip(full, split):
        # the combine reused every single-start run as it was (no further steps)
        assert {k: v["steps"] for k, v in b["runs"].items()} == {s: steps[(b["D"], s)] for s in b["runs"]}
        # and reached the normal ladder's floor (step counts can differ: the eigensolver warm starts depend on
        # which worker ran which input before)
        assert abs(a["C_train"] - b["C_train"]) <= 1e-4 * a["C_train"]
    assert it.ladder_status(model, 0.5, [2, 3], runs=runs, n_train=2, stall=None, target=None)["finished"]
