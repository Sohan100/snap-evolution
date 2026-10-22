import json
import math

import numpy as np
import pytest
import scipy.optimize as sopt

from snapevo.cells import cell_for
from snapevo.circuits import Layered, local_infidelity, product_vectors, random_product_states
from snapevo.edges import pairs, start_angles
from snapevo.hamiltonian import XXZ, evolve
from snapevo.mps import MPS, apply_layered
from snapevo.mps_train import (RESTART_GRACE, TRAIN_OPTIONS, OpenChainCost, init_from_ring, mps_from_dense,
                               open_chain_problem, train_open)

FIELDS = (XXZ(0.25, 0.0), XXZ(0.25, 1.6))
IDS = ("hx0", "hx1.6")


def _exact_problem(model, N, t, n, seed):
    """MPS inputs, dense inputs and exact open-chain outputs, as MPSs and dense."""
    states = random_product_states(N, n, seed)
    out = evolve(model.sparse(N, "open"), states, t)
    return product_vectors(N, n, seed), states, out, [mps_from_dense(v) for v in out]


def _exact_cost(cell, D, N, states, targets, z):
    """The open-chain cost by state vectors, with the same edge convention."""
    na = cell.n_angles
    out = Layered.brickwork(cell, z[: D * na]).apply(states, N, bc="open", edges=pairs(z[D * na:]))
    return local_infidelity(out, targets, N)


def _z_near_trotter(model, cell, t, D, rng, noise=0.05):
    x = cell.trotter_angles(model, t, D) + noise * rng.normal(size=D * cell.n_angles)
    return np.concatenate([x, start_angles(Layered.brickwork(cell, x)) + noise * rng.normal(size=6 * (D // 2))])


def _z_random(cell, D, rng):
    return rng.uniform(-math.pi, math.pi, size=D * cell.n_angles + 6 * (D // 2))


def _central(f, z, h, order=4):
    g = np.zeros(z.size)
    for i in range(z.size):
        e = np.zeros(z.size)
        e[i] = h
        if order == 2:
            g[i] = (f(z + e) - f(z - e)) / (2 * h)
        else:
            g[i] = (8 * (f(z + e) - f(z - e)) - (f(z + 2 * e) - f(z - 2 * e))) / (12 * h)
    return g


@pytest.mark.parametrize("N", [8, 10])
def test_mps_cost_equals_state_vector_cost(N):
    rng = np.random.default_rng(N)
    t, D, n = 1.0, 6, 3
    for model in FIELDS:
        cell = cell_for(model)
        vecs, states, dense, targets = _exact_problem(model, N, t, n, seed=N)
        cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
        for z in (_z_near_trotter(model, cell, t, D, rng), _z_random(cell, D, rng)):
            c = cost.value(z)
            exact = _exact_cost(cell, D, N, states, dense, z)
            assert abs(c - exact) <= 1e-10 * exact
            assert cost.diagnostics()["discarded_u"] == 0.0
            # the forward pass is apply_layered on the whole circuit, number for number
            circuit, edges = cost.circuit(z)
            for k in range(n):
                u = apply_layered(MPS.product(vecs[k], chi_max=10 ** 6, cutoff=0.0), circuit, edges)
                F = abs(targets[k].overlap(u)) ** 2 / abs(targets[k].overlap(targets[k]) * u.overlap(u))
                assert F == cost.last["F"][k]
            assert cost.value_and_grad(z)[0] == c


@pytest.mark.parametrize("N", [8, 10])
@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_gradient_matches_finite_differences_of_exact_cost(model, N):
    rng = np.random.default_rng(7 + N)
    t, D, n = 1.0, 6, 3
    cell = cell_for(model)
    na = cell.n_angles
    vecs, states, dense, targets = _exact_problem(model, N, t, n, seed=3)
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    z = _z_random(cell, D, rng)
    z1q = z.copy()
    z1q[2 * na: 2 * na + 3] = z1q[3 * na: 3 * na + 3] = 0.0    # sublayers 2 and 3 single-qubit only
    for zz in (z, z1q):
        g = cost.value_and_grad(zz)[1]
        fd = _central(lambda y: _exact_cost(cell, D, N, states, dense, y), zz, 1e-3)
        assert np.linalg.norm(g - fd) <= 1e-6 * np.linalg.norm(fd)
        assert np.all(np.abs(g - fd) <= 1e-6 * np.abs(fd))
        assert cost.diagnostics()["amp_dev"] < 1e-12


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_truncated_gradient_matches_mps_finite_differences(model, tmp_path):
    rng = np.random.default_rng(3)
    N, t, D, n = 20, 1.0, 8, 2
    cell = cell_for(model)
    vecs, targets = open_chain_problem(model, t, N, n, 11, tmp_path)
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, cutoff=1e-12)
    z = _z_near_trotter(model, cell, t, D, rng, noise=0.1)
    g = cost.value_and_grad(z)[1]
    diag = cost.diagnostics()
    assert diag["discarded_u"] > 0 and diag["discarded_b"] > 0
    fd = _central(cost.value, z, 1e-4, order=2)
    assert np.linalg.norm(g - fd) <= 1e-4 * np.linalg.norm(fd)
    assert np.all(np.abs(g - fd) <= 1e-4 * np.abs(fd))


def test_references_are_cached_and_checked(tmp_path):
    model = XXZ(0.25, 1.6)
    vecs, targets = open_chain_problem(model, 0.5, 8, 2, 4, tmp_path)
    files = sorted((tmp_path / model.tag / "t0.5").glob("ref_N8_s4_k*_dt0.1_c1e-14_x2048.npz"))
    assert len(files) == 2
    vecs3, again = open_chain_problem(model, 0.5, 8, 3, 4, tmp_path)         # k = 0, 1 from the cache
    assert np.array_equal(vecs3[:2], vecs) and abs(again[1].overlap(targets[1]) - 1) < 1e-14
    exact = evolve(model.sparse(8, "open"), random_product_states(8, 3, 4), 0.5)
    for k in range(3):
        assert 1 - abs(np.vdot(exact[k], again[k].to_dense())) ** 2 < 1e-10
    (tmp_path / XXZ(0.25, 0.0).tag / "t0.5").mkdir(parents=True)
    for f in files:                                     # a reference filed under the wrong model is refused
        f.rename(tmp_path / XXZ(0.25, 0.0).tag / "t0.5" / f.name)
    with pytest.raises(ValueError):
        open_chain_problem(XXZ(0.25, 0.0), 0.5, 8, 2, 4, tmp_path)


def _scaled(psi, s):
    out = MPS([a.copy() for a in psi.A], chi_max=psi.chi_max, cutoff=psi.cutoff)
    out.center = psi.center
    out.A[out.center] = out.A[out.center] * s
    return out


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_gradient_does_not_depend_on_the_norm_or_phase_of_the_targets(model):
    # regression: the bra used to keep |v| until its first apply_2q renormalized it, so every sublayer
    # below the last got environments off by |v| (gradient error 45-48% at |v| = 2)
    N, t, n = 8, 1.0, 2
    cell = cell_for(model)
    vecs, _, dense, _ = _exact_problem(model, N, t, n, seed=9)
    for D in (6, 7):                                   # last sublayer parity 1 and parity 0
        z = _z_near_trotter(model, cell, t, D, np.random.default_rng(D), noise=0.1)
        for center in (N // 2, N - 1):                # away from site 0, where the bra's norm must not be put
            unit = [mps_from_dense(v) for v in dense]
            for p in unit:
                p._move_to(center)
            c0, g0 = OpenChainCost(model, cell, t, D, N, vecs, unit, chi_max=10 ** 6, cutoff=0.0).value_and_grad(z)
            for s in (2.0, 0.5 * np.exp(1.1j)):
                cost = OpenChainCost(model, cell, t, D, N, vecs, [_scaled(p, s) for p in unit],
                                     chi_max=10 ** 6, cutoff=0.0)
                c, g = cost.value_and_grad(z)
                assert abs(c - c0) <= 1e-13 * c0
                assert np.linalg.norm(g - g0) <= 1e-12 * np.linalg.norm(g0)
                assert cost.diagnostics()["amp_dev"] < 1e-12


def test_targets_must_be_canonical_about_their_center():
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N = 8
    vecs, _, dense, targets = _exact_problem(model, N, 1.0, 2, 1)
    OpenChainCost(model, cell, 1.0, 6, N, vecs, targets)
    bad = MPS([a.copy() for a in targets[1].A])
    bad.center = N - 1
    bad.A[0] = 2 * bad.A[0]                            # the norm is no longer at the center
    with pytest.raises(ValueError, match="target 1"):
        OpenChainCost(model, cell, 1.0, 6, N, vecs, [targets[0], bad])
    hand = MPS([a.copy() for a in targets[1].A])       # MPS() says center 0, but the norm sits at site N-1
    with pytest.raises(ValueError, match="canonical"):
        OpenChainCost(model, cell, 1.0, 6, N, vecs, [targets[0], hand])
    zero = _scaled(targets[1], 0.0)
    with pytest.raises(ValueError, match="zero"):
        OpenChainCost(model, cell, 1.0, 6, N, vecs, [targets[0], zero])


def test_cached_reference_without_its_time_is_refused(tmp_path):
    # regression: float(None) raised a TypeError instead of the ValueError for a mismatched file
    model = XXZ(0.25, 1.6)
    open_chain_problem(model, 0.5, 8, 1, 4, tmp_path)
    (f,) = (tmp_path / model.tag / "t0.5").glob("ref_*.npz")
    psi, meta = MPS.load(f)
    psi.save(f, {k: v for k, v in meta.items() if k != "t"})
    with pytest.raises(ValueError):
        open_chain_problem(model, 0.5, 8, 1, 4, tmp_path)
    psi.save(f, {**meta, "t": "half"})
    with pytest.raises(ValueError):
        open_chain_problem(model, 0.5, 8, 1, 4, tmp_path)


def test_jax_gates_are_the_numpy_gates():
    for model in FIELDS:
        cell = cell_for(model)
        vecs, _, _, targets = _exact_problem(model, 4, 0.3, 1, 0)
        cost = OpenChainCost(model, cell, 0.3, 5, 4, vecs, targets)
        z = _z_random(cell, 5, np.random.default_rng(2))
        G, g_last, g_first = cost.jax_gates(z)
        circuit, edges = cost.circuit(z)
        assert max(np.abs(G[l] - cell.gate(a)).max() for l, (_, a) in enumerate(circuit.layers)) < 1e-14
        assert max(np.abs(g_last[m] - e[0]).max() + np.abs(g_first[m] - e[1]).max()
                   for m, e in enumerate(edges)) < 1e-14


def test_parallel_workers_match_one_process():
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    vecs, _, _, targets = _exact_problem(model, 10, 1.0, 3, 5)
    z = _z_near_trotter(model, cell, 1.0, 6, np.random.default_rng(4))
    c1, g1 = OpenChainCost(model, cell, 1.0, 6, 10, vecs, targets).value_and_grad(z)
    with OpenChainCost(model, cell, 1.0, 6, 10, vecs, targets, processes=2) as cost:
        c2, g2 = cost.value_and_grad(z)
        assert cost.value(z) == c2
    assert abs(c2 - c1) <= 1e-12 * c1 and np.linalg.norm(g2 - g1) <= 1e-12 * np.linalg.norm(g1)


def test_threads_option_gives_the_same_cost_and_gradient(tmp_path):
    # random angles at N = 16: bond dimension about 250, large enough for OpenBLAS to split its products
    # (the two thread counts then differ at the 1e-14 level, not bit for bit)
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N, t, D = 16, 2.0, 10
    vecs, targets = open_chain_problem(model, t, N, 2, 3, tmp_path)
    z = _z_random(cell, D, np.random.default_rng(1))
    out = {}
    for threads in (1, 2):
        with OpenChainCost(model, cell, t, D, N, vecs, targets, cutoff=1e-12, processes=2, threads=threads) as cost:
            out[threads] = cost.value_and_grad(z)
            info = cost.worker_threads()
            assert len(info) == 2 and all(w["env"] == str(threads) for w in info)
            assert all(w["blas"] in (None, threads) for w in info)
            assert cost.diagnostics()["chi_u"] >= 128
    (c1, g1), (c2, g2) = out[1], out[2]
    assert abs(c2 - c1) <= 1e-12 * c1 and np.linalg.norm(g2 - g1) <= 1e-12 * np.linalg.norm(g1)
    with OpenChainCost(model, cell, t, D, N, vecs, targets, cutoff=1e-12, threads=2) as cost:   # one worker
        c3, g3 = cost.value_and_grad(z)
        assert [w["env"] for w in cost.worker_threads()] == ["2"]
    assert abs(c3 - c1) <= 1e-12 * c1 and np.linalg.norm(g3 - g1) <= 1e-12 * np.linalg.norm(g1)
    assert OpenChainCost(model, cell, t, D, N, vecs, targets).worker_threads() == []   # default: this process


def _small_problem(hx=1.6, N=6, t=0.3, D=3):
    model = XXZ(0.25, hx)
    cell = cell_for(model)
    vecs, _, _, targets = _exact_problem(model, N, t, 3, 2)
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    return cost, init_from_ring(cell, cell.trotter_angles(model, t, D))


def test_train_open_applies_the_reduction_test_itself_after_a_restart(tmp_path):
    cost, z0 = _small_problem()
    ref = sopt.minimize(cost.value_and_grad, z0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, **TRAIN_OPTIONS})
    assert ref.message.startswith("CONVERGENCE: RELATIVE REDUCTION") and ref.nit > 50
    ck = tmp_path / "train.json"
    res, _ = train_open(cost, z0, 5000, log=lambda *_: None, checkpoint=ck)
    # the test inside train_open stops where L-BFGS-B's own does, at the same point
    assert res.stop == "ftol" and res.nit == ref.nit and res.nfev == ref.nfev and np.array_equal(res.x, ref.x)
    assert res.status == 0 and res.restart_nit == 0
    # a fresh L-BFGS-B start from the converged point stops within its first iterations
    again = sopt.minimize(cost.value_and_grad, res.x, jac=True, method="L-BFGS-B",
                          options={"maxiter": 5000, **TRAIN_OPTIONS})
    assert again.nit <= RESTART_GRACE and again.message.startswith("CONVERGENCE: RELATIVE REDUCTION")
    # a resume from it does not take that stop: the test applies from the RESTART_GRACE + 1-th iteration
    res2, _ = train_open(cost, z0, 6000, log=lambda *_: None, checkpoint=ck)
    assert res2.restart_nit == res.nit and res2.nit - res.nit > RESTART_GRACE and res2.fun <= res.fun
    assert res2.stop in ("ftol", "maxiter") or res2.stop.startswith("lbfgsb: ")


def test_train_open_on_iteration_hook_stops_without_losing_iterations(tmp_path):
    cost, z0 = _small_problem()
    ck = tmp_path / "train.json"
    seen = []

    def hook(nit, x, c, info):
        seen.append((nit, c))
        assert x.shape == z0.shape and c == cost.value(x) and info["nfev"] >= nit and info["seconds"] >= 0
        return "enough" if nit == 7 else None
    res, trace = train_open(cost, z0, 40, log=lambda *_: None, checkpoint=ck, on_iteration=hook)
    assert [n for n, _ in seen] == list(range(1, 8)) and res.nit == 7 and res.stop == "hook: enough"
    saved = json.loads(ck.read_text())
    assert saved["nit"] == 7 and np.array_equal(saved["z"], res.x) and trace[-1][0] == 7
    # an uninterrupted run passes the same iterates; the resumed one goes on from iteration 7
    full, _ = train_open(cost, z0, 7, log=lambda *_: None)
    assert np.array_equal(full.x, res.x)
    res2, trace2 = train_open(cost, z0, 12, log=lambda *_: None, checkpoint=ck, on_iteration=lambda *a: None)
    assert res2.nit == 12 and res2.stop == "maxiter" and res2.restart_nit == 7 and trace2[-1][0] == 12


def test_init_from_ring_starts_at_the_cells_own_factor():
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N, D = 10, 5
    x = cell.trotter_angles(model, 1.0, D) + 0.05 * np.random.default_rng(1).normal(size=D * cell.n_angles)
    vecs, _, _, targets = _exact_problem(model, N, 1.0, 2, 6)
    cost = OpenChainCost(model, cell, 1.0, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    z0 = init_from_ring(cell, x)
    assert np.array_equal(z0[: x.size], x) and z0.size == cost.n_params
    circuit = Layered.brickwork(cell, x)
    F = [abs(targets[k].overlap(apply_layered(MPS.product(vecs[k], chi_max=10 ** 6, cutoff=0.0), circuit))) ** 2
         for k in range(2)]
    assert abs(cost.value(z0) - (1 - np.mean(np.array(F) ** (1 / N)))) < 1e-14
    e = np.arange(6.0 * (D // 2))
    assert np.array_equal(init_from_ring(cell, x, {"angles": e.tolist()})[x.size:], e)
    with pytest.raises(ValueError):
        init_from_ring(cell, x, e[:6])


def test_train_open_lowers_the_cost_and_traces_every_25_iterations():
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 8, 0.5, 4
    vecs, _, _, targets = _exact_problem(model, N, t, 3, 2)
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    z0 = init_from_ring(cell, cell.trotter_angles(model, t, D))
    res, trace = train_open(cost, z0, maxiter=60, log=lambda *_: None)
    assert [it for it, _, _ in trace][:3] == [0, 25, 50] and trace[-1][0] == res.nit
    assert trace[0][1] == cost.value(z0) and res.fun < 0.5 * trace[0][1]
    assert all(b[1] <= a[1] for a, b in zip(trace, trace[1:]))


class _Killed(Exception):
    pass


def test_train_open_checkpoints_and_resumes(tmp_path):
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 8, 0.5, 4
    vecs, _, _, targets = _exact_problem(model, N, t, 3, 2)
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    z0 = init_from_ring(cell, cell.trotter_angles(model, t, D))
    ck = tmp_path / "train.json"

    def killed_after_25(msg):
        if "iteration 25," in msg:
            raise _Killed
    with pytest.raises(_Killed):                       # the job dies right after the iteration-25 checkpoint
        train_open(cost, z0, maxiter=60, log=killed_after_25, checkpoint=ck)
    saved = json.loads(ck.read_text())
    assert saved["nit"] == 25 and [r[0] for r in saved["trace"]] == [0, 25]
    assert cost.value(np.array(saved["z"])) == saved["c"] == saved["trace"][-1][1]

    res, trace = train_open(cost, np.zeros_like(z0), maxiter=60, log=lambda *_: None, checkpoint=ck)
    assert [list(r) for r in trace[:2]] == saved["trace"] and [it for it, _, _ in trace] == [0, 25, 50, 60]
    assert res.nit == 60 and res.nfev > saved["nfev"] and all(b[1] <= a[1] for a, b in zip(trace, trace[1:]))
    final = json.loads(ck.read_text())
    assert final["nit"] == 60 and final["nfev"] == res.nfev and np.array_equal(final["z"], res.x)

    calls = cost.ncalls                                # maxiter reached: nothing is evaluated
    res2, trace2 = train_open(cost, z0, maxiter=60, log=lambda *_: None, checkpoint=ck)
    assert cost.ncalls == calls and res2.nit == 60 and np.array_equal(res2.x, res.x) and trace2 == trace

    other = OpenChainCost(model, cell, t, D + 1, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    with pytest.raises(ValueError):                    # a checkpoint of another problem is refused
        train_open(other, init_from_ring(cell, cell.trotter_angles(model, t, D + 1)), 60, checkpoint=ck)
