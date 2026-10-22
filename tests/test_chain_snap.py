import json
import math
import os
import pickle
import signal
import threading

import numpy as np
import pytest
import scipy.optimize as sopt

import snapevo.chain_snap as cs
import snapevo.circuits
import snapevo.edges
import snapevo.environment
import snapevo.evaluate
import snapevo.jaxcost
import snapevo.realize
import snapevo.snap
from snapevo.cells import cell_for
from snapevo.chain_snap import (ChainProblem, ChainSettings, DeadlineReached, FarmCost, RingProblem, SnapSettings,
                                chain_point, compile_chain, legacy_settings, newton_minimize,
                                evaluate_chain, fd_hessian, fd_hessian_rows, pick_depth, polish_chain, realize_chain,
                                simplify_edges)
from snapevo.circuits import Layered, local_infidelity, product_vectors, random_product_states
from snapevo.edges import pairs, start_angles, u_zyz
from snapevo.evaluate import ref_path
from snapevo.hamiltonian import XXZ, evolve
from snapevo.library import Library
from snapevo.mps import reference_state
from snapevo.mps_train import OpenChainCost, mps_from_dense, open_chain_problem
from snapevo.train_chain import EVAL_DT, EVAL_SET, train_chain_ladder, trotter_z

FIELDS = (XXZ(0.25, 0.0), XXZ(0.25, 1.6))
IDS = ("hx0", "hx1.6")
KEYS = ("angles", "edges", "snapped", "costs_plan", "Tq_plan", "C_train", "n_trials", "n_rounds", "trace")


def _exact_problem(model, N, t, n, seed):
    states = random_product_states(N, n, seed)
    out = evolve(model.sparse(N, "open"), states, t)
    return product_vectors(N, n, seed), states, out, [mps_from_dense(v) for v in out]


def _dense_cost(cell, D, N, states, targets, z):
    nb = D * cell.n_angles
    out = Layered.brickwork(cell, z[:nb]).apply(states, N, bc="open", edges=pairs(z[nb:]))
    return local_infidelity(out, targets, N)


def _z_near_trotter(model, cell, t, D, rng, noise=0.05):
    x = cell.trotter_angles(model, t, D) + noise * rng.normal(size=D * cell.n_angles)
    return np.concatenate([x, start_angles(Layered.brickwork(cell, x)) + noise * rng.normal(size=6 * (D // 2))])


def _forbid_ring(monkeypatch):
    def no(*_a, **_k):
        raise AssertionError("the open-chain arm touched the ring")

    real_apply = Layered.apply

    def apply(self, states, N, dagger=False, bc="ring", edges=None):
        if bc != "open":
            no()
        return real_apply(self, states, N, dagger, bc, edges)

    monkeypatch.setattr(Layered, "apply", apply)
    monkeypatch.setattr(snapevo.jaxcost, "make_cost", no)
    monkeypatch.setattr(snapevo.jaxcost, "make_hessian", no)
    monkeypatch.setattr(snapevo.environment, "Environments", no)
    monkeypatch.setattr(snapevo.snap, "compile_snap", no)
    monkeypatch.setattr(snapevo.edges, "fit_edges", no)
    monkeypatch.setattr(snapevo.realize, "realize", no)
    monkeypatch.setattr(snapevo.circuits, "held_out", no)
    monkeypatch.setattr(snapevo.evaluate, "held_out", no)
    monkeypatch.setattr(snapevo.evaluate, "evaluate_circuit", no)


def _fake_kn(monkeypatch):
    """KN stand-in: theta_s off by eps/2, the calibrated cost, p = 0.99 (keeps tests off the shared cache)."""
    calls = []

    def synth(theta, eps, **_kw):
        calls.append((float(theta), float(eps)))
        return {"theta": theta, "eps": eps, "expected_t": 1.1 * math.log2(1.0 / eps) + 4.4, "t_v": 10, "t_b": 20,
                "p": 0.99, "theta_s": theta + 0.5 * eps, "d0": 0.5 * eps, "d1": eps, "d_branch": 0.6 * eps,
                "bound": eps}

    monkeypatch.setattr(cs, "synthesize", synth)
    return calls


_TRAINED: dict = {}


def _trained(model, N, t, D, n, seed, iters=300):
    """A small problem and its open-chain-trained z (exact MPS, no truncation), cached per arguments."""
    key = (model.hx, N, t, D, n, seed, iters)
    if key not in _TRAINED:
        _TRAINED[key] = _train(model, N, t, D, n, seed, iters)
    return _TRAINED[key]


def _train(model, N, t, D, n, seed, iters):
    """Newton-trained z (iters > 0) or the Trotter start itself (iters == 0, far from converged)."""
    cell = cell_for(model)
    vecs, states, dense, targets = _exact_problem(model, N, t, n, seed)
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    z0 = trotter_z(model, cell, t, D)
    if not iters:
        return cell, vecs, states, dense, targets, z0, float(cost.value(z0))

    def hess(z, rows, h=1e-4):
        rows = list(rows)
        R = np.empty((len(rows), len(rows)))
        for a, k in enumerate(rows):
            e = np.zeros(z.size)
            e[k] = h
            R[a] = (cost.value_and_grad(z + e)[1][rows] - cost.value_and_grad(z - e)[1][rows]) / (2 * h)
        return 0.5 * (R + R.T)
    z, info = newton_minimize(z0, cost.value_and_grad, hess, lambda c: 1e-6 * c, n_bulk=D * cell.n_angles)
    return cell, vecs, states, dense, targets, np.asarray(z), float(info["c_after"])


# ----------------------------------------------------------------------------- curves

@pytest.mark.parametrize("N", [8, 10])
@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_curves_equal_the_dense_open_chain_cost(model, N):
    rng = np.random.default_rng(N + 3)
    t, D, n = 1.0, 5, 3
    cell = cell_for(model)
    vecs, states, dense, targets = _exact_problem(model, N, t, n, seed=N)
    z = _z_near_trotter(model, cell, t, D, rng)
    nb = D * cell.n_angles
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, replicas=2) as P:
        env = P.envs_at(z)
        assert len(env.blocks) == 2
        curves = env.curves(range(nb))
        for i, cv in curves.items():
            assert cv.residual < 1e-12
            layer, slot = divmod(i, cell.n_angles)
            n_g = N // 2 if layer % 2 == 0 else N // 2 - 1
            assert len(cv.modes) == (2 * n_g + 1 if cell.harmonic[slot] == 2 else n_g + 1)
            for th in [z[i], *rng.uniform(-2 * math.pi, 2 * math.pi, 3)]:
                zz = z.copy()
                zz[i] = th
                exact = _dense_cost(cell, D, N, states, dense, zz)
                assert abs(cv.c_at(th) - exact) <= 1e-10 * exact
        # a single curve is the same number as the batch, and c_at takes arrays
        one = env.curve(nb - 1)
        th = rng.uniform(0, 2 * math.pi, 5)
        assert np.allclose(one.c_at(th), curves[nb - 1].c_at(th), rtol=1e-13, atol=0)
        # a newer environment replaces the caches of the older one in the workers
        P.envs_at(z)
        with pytest.raises(RuntimeError):
            env.curves([0])


def test_environments_cache_only_the_sublayers_asked_for_and_release_them():
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 8, 1.0, 5
    vecs, _, _, targets = _exact_problem(model, N, t, 3, 4)
    z = _z_near_trotter(model, cell, t, D, np.random.default_rng(9))
    na = cell.n_angles
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, replicas=2,
                      local=True) as P:
        full = P.envs_at(z)
        ref = full.curves([na + 1, 3 * na + 2])
        full.release()
        part = P.envs_at(z, [1, 3])
        assert part.memory_bytes() < full.memory_bytes()
        got = part.curves([na + 1, 3 * na + 2])
        for i in got:
            th = np.linspace(0, 2 * math.pi, 7)
            assert np.allclose(got[i].c_at(th), ref[i].c_at(th), rtol=1e-12, atol=0)
        with pytest.raises(KeyError):
            part.curves([0])                                     # sublayer 0 was not cached
        part.release()
        with pytest.raises(RuntimeError):
            part.curves([na + 1])
        assert all(w.state.get("env") is None for row in P.pool.workers for w in row)


def test_a_curve_that_misses_its_check_probe_is_refused(monkeypatch):
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N, t, D = 8, 1.0, 3
    vecs, _, _, targets = _exact_problem(model, N, t, 2, 4)
    z = _z_near_trotter(model, cell, t, D, np.random.default_rng(2))
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2, local=True) as P:
        env = P.envs_at(z)
        monkeypatch.setattr(cs, "_slot_modes", lambda _name, _slot: (0, 1, 1))     # too few modes
        with pytest.raises(RuntimeError, match="check probe"):
            env.curves([0])


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_curves_match_the_truncated_cost_to_truncation_accuracy(model, tmp_path):
    rng = np.random.default_rng(5)
    N, t, D, n = 20, 1.0, 6, 2
    cell = cell_for(model)
    vecs, targets = open_chain_problem(model, t, N, n, 11, tmp_path)
    z = _z_near_trotter(model, cell, t, D, rng, noise=0.1)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=1024, cutoff=1e-12, groups=2) as P:
        c0 = P.c_fn(z)
        assert P.cost.diagnostics()["discarded_u"] > 0
        nb = D * cell.n_angles
        curves = P.envs_at(z).curves([0, nb // 2, nb - 1])
        for i, cv in curves.items():
            assert abs(cv.c_at(z[i]) - c0) <= 1e-7 * c0
            for d in (0.02, -0.3):
                zz = z.copy()
                zz[i] += d
                v = P.c_fn(zz)
                assert abs(cv.c_at(zz[i]) - v) <= 1e-7 * v


# ----------------------------------------------------------------------------- cost and Hessian

def test_farm_cost_is_open_chain_cost_number_for_number():
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 8, 1.0, 4
    vecs, _, _, targets = _exact_problem(model, N, t, 4, 2)
    rng = np.random.default_rng(1)
    zs = [_z_near_trotter(model, cell, t, D, rng) for _ in range(3)]
    ref = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=64, cutoff=1e-12)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=64, cutoff=1e-12, groups=3, replicas=2) as P:
        assert isinstance(P.cost, FarmCost)
        many = P.cost.value_and_grad_many(zs)
        vals = P.cost.value_many(zs)
        for z, (c, g), v in zip(zs, many, vals):
            c0, g0 = ref.value_and_grad(z)
            assert c == c0 and np.array_equal(g, g0)
            assert P.objective(z)[0] == c0 and v == ref.value(z) == P.c_fn(z)


def test_hessian_is_symmetric_and_matches_second_differences():
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N, t, D = 8, 1.0, 4
    vecs, _, _, targets = _exact_problem(model, N, t, 3, 4)
    z = _z_near_trotter(model, cell, t, D, np.random.default_rng(2))
    nb = D * cell.n_angles
    rows = [0, 3, 5, nb - 1, nb, nb + 4, z.size - 1]                     # bulk and edge parameters
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3) as P:
        H, asym = fd_hessian(P.cost, z, rows)
        assert asym < 1e-7 and np.array_equal(H, H.T)
        assert np.array_equal(P.hessian(z, rows), H)
        # rows in chunks over all the columns are the same numbers
        R = np.vstack([P.hessian(z, rows[:3], cols=rows), P.hessian(z, rows[3:], cols=rows)])
        assert np.array_equal(0.5 * (R + R.T), H) and np.array_equal(R, fd_hessian_rows(P.cost, z, rows, rows))
        h = 1e-3
        f = P.cost.value
        for a, i in enumerate(rows):
            for b, j in enumerate(rows):
                ei, ej = np.zeros(z.size), np.zeros(z.size)
                ei[i], ej[j] = h, h
                second = (f(z + ei + ej) - f(z + ei - ej) - f(z - ei + ej) + f(z - ei - ej)) / (4 * h * h)
                assert abs(H[a, b] - second) <= 1e-5 * np.max(np.abs(H))


def test_polish_converges_records_curvature_and_resumes(tmp_path):
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, _ = _trained(model, N, t, D, 3, 3, iters=0)         # the Trotter start
    nb = D * cell.n_angles
    tol = 1e-6 * 1e-4
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, local=True) as P:
        zp, info = polish_chain(z, P.objective, P.hessian, tol, n_bulk=nb)
        assert info["kept"] and info["c_after"] < info["c_before"] and info["steps"] >= 1
        assert info["stop"] in ("converged", "noise floor") and info["hessians"] >= 2
        if info["stop"] == "converged":
            assert info["pred"] < tol and info["eig_min_bulk_after"] > 0
        assert P.c_fn(zp) == pytest.approx(info["c_after"], rel=1e-12)
        for k in ("eig_min", "n_neg", "eig_min_bulk", "n_neg_bulk"):
            assert f"{k}_before" in info and f"{k}_after" in info
        # polished again to the same tolerance, it stops at once
        _zz, again = polish_chain(zp, P.objective, P.hessian, tol, n_bulk=nb)
        assert again["steps"] == 0 or again["stop"] == "noise floor"
        # stopped after a few steps, it continues from its checkpoint to the same test
        ck = str(tmp_path / "polish.json")
        stop, calls = threading.Event(), [0]

        def obj(zz):
            calls[0] += 1
            if calls[0] == 4:
                stop.set()
            return P.objective(zz)
        with pytest.raises(DeadlineReached):
            polish_chain(z, obj, P.hessian, tol, n_bulk=nb, checkpoint=ck, stop=stop)
        saved = json.loads(open(ck).read())
        assert saved["steps"] >= 1
        zr, info_r = polish_chain(z, P.objective, P.hessian, tol, n_bulk=nb, checkpoint=ck)
        assert info_r["steps"] >= saved["steps"] and info_r["c_after"] < info_r["c_before"] == info["c_before"]
        assert info_r["stop"] in ("converged", "noise floor")


# ----------------------------------------------------------------------------- compile

@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_compile_meets_the_budget_on_the_open_chain_only(model, tmp_path, monkeypatch):
    _forbid_ring(monkeypatch)
    N, t, D = 6, 0.5, 3
    cell, vecs, states, dense, targets, z, floor = _trained(model, N, t, D, 3, 3)
    budget = 3 * max(floor, 1e-6)
    nb = D * cell.n_angles
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2, replicas=2,
                      local=True) as P:
        settings = ChainSettings(budget=budget, max_round_trials=8, checkpoint=str(tmp_path / "ck.pkl"),
                                 edge_weight=2.0 / N, reopt="newton")
        res = compile_chain(z, cell, D, settings, P.objective, P.c_fn, P.hessian, P.envs_at, Library(22.0))
        zc = np.asarray(res["z"])
        assert res["C_train"] <= budget * (1 + 1e-9)
        assert res["C_train"] == P.c_fn(zc)
        # the dense 1 - mean F^(1/N) loses ~1e-16 to cancellation at these c
        assert abs(_dense_cost(cell, D, N, states, dense, zc) - res["C_train"]) <= 1e-8 * res["C_train"] + 1e-15
        assert all(w.state.get("env") is None for row in P.pool.workers for w in row)    # caches released
    assert len(res["angles"]) == len(res["snapped"]) == nb and len(res["edges"]) == 6 * (D // 2)
    assert np.array_equal(zc, np.concatenate([res["angles"], res["edges"]]))
    lib = Library(22.0)
    for i, s in enumerate(res["snapped"]):
        if s:
            assert lib.lookup(res["angles"][i]) is not None
    assert res["n_zero_t"] + res["n_hessian"] >= 1
    assert np.isfinite(res["Tq_plan"])
    assert set(res["timing"]) >= {"reopt", "curves", "hess_grads", "n_value"}
    tm = res["timing"]
    assert res["n_grad"] == sum(tm.get(k, 0) for k in ("calls_floor", "calls_reopt", "calls_trial", "hess_grads",
                                                       "newton_hess_grads"))
    # every re-optimization is Newton's, with its steps, refreshes and stop reason recorded
    assert res["reoptimizer"] == "newton" and res["reopt_log"]
    assert all(e["path"] == "newton" and e["stop"] in ("converged", "noise floor", "nothing free")
               and "refreshes" in e and e["pred"] is not None for e in res["reopt_log"])
    tol = 1e-6 * budget
    assert all(e["pred"] < tol for e in res["reopt_log"] if e["stop"] == "converged")
    # the edges are stragglers of the end game: each has a library angle or a KN budget
    alloc = set(res["edge_library"]) | set(res["edge_stragglers"])
    assert alloc == {str(k) for k in range(6 * (D // 2))} and res["edge_weight"] == 2.0 / N
    assert res["Tq_plan"] >= res["Tq_plan_bulk"]
    for k, v in res["edge_library"].items():
        assert res["edges"][int(k)] == pytest.approx(math.remainder(v["angle"] - res["edges"][int(k)], 2 * math.pi)
                                                    + res["edges"][int(k)], abs=1e-12)
    # one Hessian record per round, with its curvature and response diagnostics
    assert len(res["hessian_stats"]) == res["n_rounds"] >= 1
    for h in res["hessian_stats"]:
        assert {"n_neg", "eig_min", "n_neg_bulk", "eig_min_bulk", "rp_median", "asym", "rows"} <= set(h)
    # the edges are re-optimized with the bulk, never pinned
    assert not np.array_equal(res["edges"], z[nb:])


def test_parallel_trials_and_layer_batches_give_the_same_compile_and_a_failing_trial_stops():
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, floor = _trained(model, N, t, D, 3, 3)
    budget = 3 * max(floor, 1e-6)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2, replicas=3,
                      local=True) as P:
        def run(objective=P.objective, hessian=P.hessian, **kw):
            settings = ChainSettings(budget=budget, max_round_trials=8, edge_weight=2.0 / N, reopt="newton")
            return compile_chain(z, cell, D, settings, objective, P.c_fn, hessian, P.envs_at, Library(22.0), **kw)

        one = run()
        par = run(trial_workers=3)
        batched = run(env_layers=1)                              # one sublayer per environment: same curves
        for k in KEYS:
            assert par[k] == one[k] and batched[k] == one[k], k
        assert par["n_trials_run"] >= one["n_trials_run"] == one["n_trials"]
        armed = threading.Event()

        def hessian(zz, rows, cols=None):
            armed.set()                                          # every later evaluation is a trial's
            return P.hessian(zz, rows, cols)

        def objective(zz):
            if armed.is_set():
                raise RuntimeError("worker failed")
            return P.objective(zz)
        with pytest.raises(RuntimeError, match="worker failed"):
            run(objective, hessian, trial_workers=3)


def test_resume_after_a_kill_mid_round_is_exact(tmp_path):
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, floor = _trained(model, N, t, D, 3, 3)
    budget = 3 * max(floor, 1e-6)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2, local=True) as P:
        calls = {"obj": 0, "hess": 0}
        trip = {}

        def objective(zz):
            calls["obj"] += 1
            if calls["obj"] == trip.get("obj"):
                trip["action"]()
            return P.objective(zz)

        marks = []                                               # objective calls made before each Hessian chunk

        def hessian(zz, rows, cols=None):
            calls["hess"] += 1
            marks.append(calls["obj"])
            if calls["hess"] == trip.get("hess"):
                trip["action"]()
            return P.hessian(zz, rows, cols)

        def run(ck=None, stop=None):
            settings = ChainSettings(budget=budget, max_round_trials=8, checkpoint=ck, edge_weight=2.0 / N,
                                     reopt="newton")
            return compile_chain(z, cell, D, settings, objective, P.c_fn, hessian, P.envs_at, Library(22.0),
                                 hessian_chunk=4, stop=stop)

        ref = run()
        n_obj, n_hess = calls["obj"], calls["hess"]
        assert ref["n_rounds"] >= 2
        first_trials = sorted(set(marks))[:3]                    # the next evaluations belong to a round's trials
        kills = [("obj", int(f * n_obj), "stop") for f in (0.15, 0.5, 0.85)]
        kills += [("obj", m + 3, "stop") for m in first_trials]
        kills += [("hess", max(2, n_hess // 2), "stop"), ("obj", first_trials[-1] + 2, "sigterm")]
        kills = [(w, at, how) for w, at, how in kills if 1 <= at < (n_obj - 2 if w == "obj" else n_hess)]
        old = signal.getsignal(signal.SIGTERM)
        inside = []
        for where, at, how in kills:
            ck = str(tmp_path / f"ck_{where}_{at}_{how}.pkl")
            stop = threading.Event()
            calls.update(obj=0, hess=0)
            trip.clear()
            trip[where] = at
            trip["action"] = stop.set if how == "stop" else (lambda: os.kill(os.getpid(), signal.SIGTERM))
            with pytest.raises(DeadlineReached):
                run(ck, stop)
            if os.path.exists(ck):                         # none if stopped inside the floor's re-optimization
                with open(ck, "rb") as fh:
                    inside.append(pickle.load(fh)["state"].round is not None)
            trip.clear()
            res = run(ck)
            for k in KEYS:
                assert res[k] == ref[k], (where, at, how, k)
            strip = [[{k: v for k, v in h.items() if k != "seconds"} for h in r["hessian_stats"]] for r in (res, ref)]
            assert strip[0] == strip[1], (where, at, how)
        assert sum(inside) >= 1                                       # kills mid-round, not only between rounds
        assert signal.getsignal(signal.SIGTERM) == old               # the handler is put back


# ----------------------------------------------------------------------------- realize and evaluate

def _eval_refs(model, t, N, runs, n=None):
    d = runs / model.tag / f"t{t:g}" / "ref"
    d.mkdir(parents=True, exist_ok=True)
    seed, n_all = EVAL_SET
    vecs = product_vectors(N, n or n_all, seed)
    for k in range(n or n_all):
        for c in (1e-12, 1e-14):
            reference_state(model, t, vecs[k], EVAL_DT, 1024, c, ref_path(d, N, seed, k, EVAL_DT, c, 1024))
    return d


def test_realize_and_evaluate_on_the_open_chain(tmp_path, monkeypatch):
    _forbid_ring(monkeypatch)
    calls = _fake_kn(monkeypatch)
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, floor = _trained(model, N, t, D, 3, 3)
    nb = D * cell.n_angles
    lib = Library(22.0)
    x, e = z[:nb].copy(), z[nb:].copy()
    snapped = np.zeros(nb, dtype=bool)
    for i in (0, 2, 5):                                     # three bulk angles on lattice points
        j = int(np.argmin(np.abs((lib.S - x[i] + math.pi) % (2 * math.pi) - math.pi)))
        x[i] = lib.S[j]
        snapped[i] = True
    e[1] = math.pi / 2 + 1e-9                                # an edge angle within eps of a multiple of pi/4
    budget = 0.9                       # moving that edge angle spoils the circuit; this test checks the bookkeeping
    compiled = {"angles": x.tolist(), "edges": e.tolist(), "snapped": snapped.tolist(),
                "multiplicity": [float(cell.multiplicity[i % cell.n_angles]) for i in range(nb)],
                "settings": {"budget": budget}, "C_train": floor,
                "stragglers": {str(i): {"eps": 1e-3} for i in range(nb) if not snapped[i]}}
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    real = realize_chain(compiled, cell, cost.value, log=lambda _s: None, simplify=False)
    E = real["edges"]
    assert E["use"] and E["eps"] == snapevo.edges.edge_eps(budget) and E["simplify"] is None
    assert E["angles_real"][1] == math.pi / 2 and "1" not in E["kn"] and E["n_clifford_t"] >= 1
    assert E["n_kn"] + E["n_clifford_t"] == E["n_rotations"] == 6 * (D // 2)
    assert set(real["lattice"]) == {"0", "2", "5"} and real["n_kn"] == nb - 3
    assert real["C_train_real"] == cost.value(np.asarray(real["z_real"]))
    assert np.array_equal(real["z_real"], np.concatenate([real["angles_real"], E["angles_real"]]))
    assert real["straggler_eps_scale"] == 1.0 and len(calls) == real["n_kn"] + E["n_kn"]
    assert all(eps == 1e-3 for _th, eps in calls[:real["n_kn"]])     # bulk stragglers first, then edges
    assert not real["over_budget_without_stragglers"]
    # evaluation on the sweep's inputs at N against cached references, without the ring side metric
    runs = tmp_path / "runs"
    d = _eval_refs(model, t, N, runs, n=3)
    kw = dict(states=(EVAL_SET[0], 3), cache_dir=d, cutoff=1e-12, chi_max=1024, dt=EVAL_DT)
    with pytest.raises(ValueError, match="N = 100"):
        evaluate_chain(real, model, cell, t, N=N, log=lambda _s: None, **kw)
    ev = evaluate_chain(real, model, cell, t, N=N, log=lambda _s: None, expect_N=None, **kw)
    assert "C_ring" not in ev and ev["C_train_real"] == real["C_train_real"]
    direct = snapevo.evaluate.mps_infidelity(Layered.brickwork(cell, real["angles_real"]), pairs(E["angles_real"]),
                                             model, t, N=N, states=(EVAL_SET[0], 3), cache_dir=d, check=False,
                                             log=lambda _s: None)
    assert ev["infid"] == direct["infid"]
    best = ev["per_F"][ev["F"]]
    assert best["Tq"] == pytest.approx(real["per_F"][ev["F"]]["Tq"] + E["T_total"] / N)
    assert ev["rotations_per_qubit"] == pytest.approx(real["rotations_per_qubit"] + E["n_rotations"] / N)
    assert ev["infid_total"] >= ev["infid"]


def _gate_dist(a, b):
    U, V = u_zyz(*a), u_zyz(*b)
    return 1.0 - abs(np.trace(U.conj().T @ V)) / 2.0


def test_edge_simplification_keeps_c_within_its_allowance():
    # a stand-in cost: each gate pays w_g x (phase-free distance to its reference gate)
    ref = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.3, 0.7, 0.1), (1.0, 0.4, -0.2)]
    w = [1e-3, 1.0, 1e-4, 1.0]
    e = np.array([0.94, 0.0016, -0.94,                       # drifted near-identity gate: becomes the identity
                  0.3, 1e-3, 0.2,                             # middle angle near 0: one Rz of 0.5
                  math.pi / 4 + 3e-3, 0.7, 0.1,               # first angle near pi/4, cheap to set
                  1.0, 0.4, -0.2 + 4e-3])                     # third angle near 0 but expensive to set
    ref = [(math.pi / 4 + 3e-3, 0.7, 0.1) if k == 2 else r for k, r in enumerate(ref)]
    x_bulk = np.array([0.1, 0.2])

    def c_fn(zz):
        g = np.asarray(zz)[2:].reshape(-1, 3)
        return 1e-6 + sum(wk * _gate_dist(gk, rk) for gk, rk, wk in zip(g, ref, w))

    allowance = 2e-6
    out, info = simplify_edges(x_bulk, e, c_fn, allowance)
    g = out.reshape(-1, 3)
    assert info["identity_gates"] == [0] and np.all(g[0] == 0.0)
    assert info["rz_gates"] == [1] and g[1][0] == pytest.approx(0.5) and g[1][1] == g[1][2] == 0.0
    assert g[2][0] == math.pi / 4 and 6 in info["pi4_angles"]
    assert g[3][2] != 0.0 and 11 not in info["pi4_angles"]
    assert info["c_after"] <= info["c_before"] + allowance and info["c_after"] == c_fn(np.concatenate([x_bulk, out]))
    # no room: the changes together may not raise c (gates 0 and 1 move onto their references, which pays for
    # the pi/4 setting of angle 6)
    same, info0 = simplify_edges(x_bulk, e, c_fn, 0.0)
    assert info0["c_after"] <= info0["c_before"] and 11 not in info0["pi4_angles"]


def test_realize_says_when_edges_and_lattice_angles_alone_break_the_budget(monkeypatch):
    calls = _fake_kn(monkeypatch)
    cell = cell_for(XXZ(0.25, 0.0))
    nb = 3 * cell.n_angles
    lib = Library(22.0)
    x = np.full(nb, lib.S[100])
    compiled = {"angles": x.tolist(), "edges": [0.0] * 6, "snapped": [True] * (nb - 2) + [False] * 2,
                "multiplicity": [float(cell.multiplicity[i % cell.n_angles]) for i in range(nb)],
                "settings": {"budget": 1e-4}, "C_train": 2e-4,
                "stragglers": {str(k): {"eps": 1e-3, "dC_plan": 1e-6} for k in (nb - 2, nb - 1)}}
    real = realize_chain(compiled, cell, lambda _z: 2e-4, log=lambda _s: None)
    assert real["over_budget_without_stragglers"] and real["straggler_eps_scale"] == 1.0
    assert len(calls) == 2                                       # no tightening rounds


def test_chain_point_polishes_writes_the_three_records_and_resumes(tmp_path, monkeypatch):
    _forbid_ring(monkeypatch)
    _fake_kn(monkeypatch)
    model, N, t = XXZ(0.25, 0.0), 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    train_chain_ladder(model, t, [2, 3], runs=runs, N=N, n_inputs=3, maxiter=30, processes=1,
                       ref_cache=tmp_path / "refs", parallel_starts=False, stall=None, log=lambda _s: None)
    lad = json.loads((runs / model.tag / f"t{t:g}" / "chain" / "ladder.json").read_text())
    floors = {r["D"]: max(r["c_train"], r["c_heldout"]) for r in lad["results"]}
    budget = 3.0 * floors[3]
    assert pick_depth(lad["results"], budget, 3.0) == (2 if 3 * floors[2] <= budget else 3)
    with pytest.raises(ValueError, match="N = 100"):
        chain_point(model, t, budget, runs, D=3, ref_cache=tmp_path / "refs", local=True, log=lambda _s: None)
    lines = []
    kw = dict(D=3, ref_cache=tmp_path / "refs", groups=3, log=lines.append, local=True, polish=True, check_N=False)
    out = chain_point(model, t, budget, runs, settings_kw={"max_round_trials": 4, "reopt": "newton"}, **kw)
    assert len(out) == 1
    d = runs / model.tag / f"t{t:g}" / "chain"
    stem = f"b{budget:.1e}_D3"
    snap, real, ev = (json.loads((d / f"{k}_{stem}.json").read_text()) for k in ("snap", "real", "eval"))
    assert not (d / f"snap_{stem}.ckpt.pkl").exists()
    assert snap["bc"] == "open" and snap["N"] == N and snap["C_train"] <= budget * (1 + 1e-9)
    assert snap["polish"]["stop"] in ("converged", "noise floor") and (d / "polish_D3.json").exists()
    assert snap["floor"] <= snap["polish"]["c_after"] + 1e-15
    assert real["C_train_real"] <= budget * (1 + 1e-9) or real["n_kn"] == 0
    assert real["edges"]["simplify"]["c_after"] <= real["edges"]["simplify"]["c_before"] + \
        real["edges"]["simplify"]["allowance"]
    assert ev["D"] == 3 and ev["budget"] == budget and ev["mps"]["N"] == N and "trunc_check" in ev["mps"]
    # a rerun reuses the evaluation; another budget at this depth reuses the polish
    monkeypatch.setattr(cs, "compile_chain", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    again = chain_point(model, t, budget, runs, **kw)
    assert again[0]["infid"] == ev["infid"]
    monkeypatch.setattr(cs, "polish_chain", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("polished again")))
    with pytest.raises(AssertionError):                          # reached compile_chain, past the cached polish
        chain_point(model, t, 2 * budget, runs, **kw)


def test_pick_depth_uses_the_ladders_own_floor():
    recs = [{"D": 3, "c_train": 1e-3}, {"D": 5, "c_train": 1e-5}, {"D": 7, "c_train": 1e-7}]
    assert pick_depth(recs, 1e-4, 3.0) == 5
    assert pick_depth(recs, 1e-4, 30.0) == 7
    assert pick_depth(recs, 1e-9, 3.0) == 7
    assert pick_depth(recs, 1e-2, 3.0) == 3
    recs[1]["c_heldout"] = 5e-5                                  # the held-out floor counts too
    assert pick_depth(recs, 1e-4, 3.0) == 7


# ----------------------------------------------------------------------------- edges as stragglers

@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_edge_curves_equal_the_dense_open_chain_cost(model):
    rng = np.random.default_rng(13)
    N, t, D = 8, 1.0, 5
    cell = cell_for(model)
    vecs, states, dense, targets = _exact_problem(model, N, t, 3, 6)
    z = _z_near_trotter(model, cell, t, D, rng)
    nb = D * cell.n_angles
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, replicas=2,
                      local=True) as P:
        env = P.envs_at(z, [1, 3])
        curves = env.curves(range(nb, z.size))
        for i, cv in curves.items():
            assert len(cv.modes) == 2 and cv.residual < 1e-12                # two harmonics of theta/2
            for th in [z[i], *rng.uniform(-2 * math.pi, 2 * math.pi, 3)]:
                zz = z.copy()
                zz[i] = th
                exact = _dense_cost(cell, D, N, states, dense, zz)
                assert abs(cv.c_at(th) - exact) <= 1e-10 * exact


def test_edge_zero_t_sets_cost_free_edge_angles_without_reoptimizing(monkeypatch):
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, floor = _trained(model, N, t, D, 3, 3)
    budget = 30 * max(floor, 1e-6)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, local=True) as P:
        settings = ChainSettings(budget=budget, max_round_trials=8, edge_weight=2.0 / N, edge_zero_t=True,
                                 reopt="newton")
        res = compile_chain(z, cell, D, settings, P.objective, P.c_fn, P.hessian, P.envs_at, Library(22.0))
        assert res["C_train"] <= budget * (1 + 1e-9) and res["C_train"] == P.c_fn(np.asarray(res["z"]))
    nb = D * cell.n_angles
    edge_steps = [e for e in res["trace"] if e["i"] >= nb]
    assert edge_steps and all(e["stage"] == "zero" for e in edge_steps)
    for e in edge_steps:                                            # cost-0 points: multiples of pi/2
        assert abs(math.remainder(res["z"][e["i"]], math.pi / 2)) < 1e-12
    assert {e["i"] - nb for e in edge_steps} >= set(res["edge_zero_t"])


def test_realize_prices_edges_by_their_allocation(monkeypatch):
    calls = _fake_kn(monkeypatch)
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, floor = _trained(model, N, t, D, 3, 3)
    nb = D * cell.n_angles
    lib = Library(22.0)
    x, e = z[:nb].copy(), z[nb:].copy()
    for i in range(nb):                                              # every bulk angle on a lattice point
        x[i] = lib.S[int(np.argmin(np.abs((lib.S - x[i] + math.pi) % (2 * math.pi) - math.pi)))]
    j_lat = 1                                                        # a lattice point away from multiples of pi/4
    e[j_lat] = lib.S[int(np.argmin(np.abs((lib.S - 0.3 + math.pi) % (2 * math.pi) - math.pi)))]
    cost = OpenChainCost(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0)
    budget = 0.9
    compiled = {"angles": x.tolist(), "edges": e.tolist(), "snapped": [True] * nb,
                "multiplicity": [float(cell.multiplicity[i % cell.n_angles]) for i in range(nb)],
                "settings": {"budget": budget}, "C_train": cost.value(np.concatenate([x, e])), "stragglers": {},
                "edge_weight": 2.0 / N,
                "edge_library": {str(j_lat): {"angle": float(e[j_lat]), "cost": 1.0, "dC": 0.0}},
                "edge_stragglers": {str(k): {"eps": 3e-6, "dC_plan": 0.0} for k in range(6) if k != j_lat}}
    real = realize_chain(compiled, cell, cost.value, log=lambda _s: None, simplify=False)
    E = real["edges"]
    assert set(E["lattice"]) == {str(j_lat)} and str(j_lat) not in E["kn"] and E["n_lattice"] == 1
    assert all(eps == 3e-6 for _th, eps in calls)                    # KN at the allotted budgets
    for F, pf in real["per_F"].items():                              # the lattice edge angle at weight 2/N
        bulk = 0.5 * sum(m * r["cost"][F] for m, r in zip(compiled["multiplicity"], real["lattice"].values()))
        assert pf["Tq"] == pytest.approx(bulk + 0.5 * (2.0 / N) * E["lattice"][str(j_lat)]["cost"][F])
    assert real["C_train_real"] == cost.value(np.asarray(real["z_real"]))


# ----------------------------------------------------------------------------- the ring backend

def compile_snap_lbfgs(f, x0, lower, upper):
    """compile_snap's own re-optimization (snap._REOPT), only to check the shared code against compile_snap."""
    from snapevo.snap import _REOPT
    res = sopt.minimize(f, x0, bounds=list(zip(lower, upper)), **_REOPT)
    return res.x, res.fun


def test_ring_backend_reproduces_compile_snap_and_runs_every_reoptimizer():
    from snapevo.snap import compile_snap
    model, t, D, N = XXZ(0.25, 0.0), 0.5, 4, 6
    cell = cell_for(model)
    P = RingProblem(model, cell, t, D, N=N, hessian_method="jax")
    x = sopt.minimize(P.objective, cell.trotter_angles(model, t, D), jac=True, method="trust-exact",
                      hess=lambda z: P.hessian(z, range(z.size))).x
    budget = 3 * max(P.objective(x)[0], 1e-6)
    ref = compile_snap(x, cell, SnapSettings(budget=budget, max_round_trials=8), P.objective, P.c_fn, P._H,
                       P.tin, P.tg, N, Library(22.0))
    res = compile_chain(x, cell, D, legacy_settings(budget, max_round_trials=8), P.objective, P.c_fn, P.hessian,
                        P.envs_at, Library(22.0), reoptimizer=compile_snap_lbfgs)
    for k in ("angles", "snapped", "costs_plan", "Tq_plan", "C_train", "floor", "trace", "n_trials", "n_rounds",
              "stragglers", "library_stragglers"):
        assert res[k] == ref[k], k
    for mode in cs.REOPTIMIZERS:
        new = compile_chain(x, cell, D, ChainSettings(budget=budget, max_round_trials=8, reopt=mode), P.objective,
                            P.c_fn, P.hessian, P.envs_at, Library(22.0))
        assert new["C_train"] <= budget * (1 + 1e-9) and new["edges"] == [], mode
        assert new["C_train"] == P.c_fn(np.asarray(new["z"])), mode
        assert new["reoptimizer"] == mode and new["settings"]["reopt"] == mode
        # L-BFGS-B may also stop on its own ftol/gtol tests (snap._REOPT's) or a failed line search at the
        # rounding floor (ABNORMAL), never on an iteration count
        assert all(e["path"] == mode and (e["stop"] in ("converged", "noise floor", "nothing free")
                                          or mode == "lbfgs-converged" and e["stop"].startswith(("CONVERGENCE",
                                                                                                 "ABNORMAL")))
                   for e in new["reopt_log"]), (mode, {e["stop"] for e in new["reopt_log"]})
        if mode == "newton":
            assert new["n_grad"] < res["n_grad"]
        if mode == "coordinate":
            assert all(e["sweeps"] >= 1 and e["gain"] < 1e-6 * budget for e in new["reopt_log"]
                       if e["stop"] == "converged")
            assert new["timing"]["coord_curves"] >= 1


def test_shared_code_reproduces_a_production_compile():
    """compile_chain on the ring with compile_snap's choices reproduces a production compile (h_x = 1.6, t = 1)."""
    from pathlib import Path

    from snapevo.train import load_cell
    d = Path(__file__).resolve().parent.parent / "runs" / "xxz_d0.25_hx1.6" / "t1"
    if not (d / "snap_b1.0e-03_D5.json").exists():
        pytest.skip("the production compile is not in this checkout")
    model, t, D = XXZ(0.25, 1.6), 1.0, 5
    cell = cell_for(model)
    prod = json.loads((d / "snap_b1.0e-03_D5.json").read_text())
    x0, _ = load_cell(d / "cells.json", D)
    ps = {k: v for k, v in prod["settings"].items() if k not in ("checkpoint", "checkpoint_every")}
    P = RingProblem(model, cell, t, D)
    res = compile_chain(x0, cell, D, legacy_settings(**ps), P.objective, P.c_fn, P.hessian, P.envs_at,
                        Library(ps["F"]), reoptimizer=compile_snap_lbfgs)
    trace = [(e["i"], e["stage"], e["cost"]) for e in prod["trace"]]
    assert [(e["i"], e["stage"], e["cost"]) for e in res["trace"]] == trace
    for k in ("snapped", "costs_plan", "Tq_plan", "n_zero_t", "n_hessian", "n_rounds", "n_trials"):
        assert res[k] == prod[k], k
    assert np.max(np.abs(np.asarray(res["angles"]) - np.asarray(prod["angles"]))) < 1e-12
    assert res["C_train"] == pytest.approx(prod["C_train"], rel=1e-9)        # floating point of another machine


# ----------------------------------------------------------------------------- re-optimizer choice and sweeps

def test_compile_needs_a_reoptimizer_and_coordinate_runs_one_trial_at_a_time():
    model = XXZ(0.25, 0.0)
    N, t, D = 6, 0.5, 3
    cell, vecs, _, _, targets, z, floor = _trained(model, N, t, D, 3, 3)
    budget = 3 * max(floor, 1e-6)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2, local=True) as P:
        args = (P.objective, P.c_fn, P.hessian, P.envs_at, Library(22.0))
        with pytest.raises(ValueError, match="no default"):
            compile_chain(z, cell, D, ChainSettings(budget=budget), *args)
        with pytest.raises(ValueError, match="no default"):
            compile_chain(z, cell, D, SnapSettings(budget=budget), *args)
        with pytest.raises(ValueError, match="unknown re-optimizer"):
            compile_chain(z, cell, D, ChainSettings(budget=budget, reopt="lbfgs-cap250"), *args)
        with pytest.raises(ValueError, match="trial_workers = 1"):
            compile_chain(z, cell, D, ChainSettings(budget=budget, reopt="coordinate"), *args, trial_workers=2)


def test_chain_sweep_environments_give_the_curves_of_fresh_ones():
    """set_angle and advance carry the forward caches along a sweep: same curves as environments built anew."""
    rng = np.random.default_rng(21)
    model = XXZ(0.25, 1.6)
    N, t, D = 8, 1.0, 5
    cell = cell_for(model)
    na, nb = cell.n_angles, D * cell.n_angles
    vecs, states, dense, targets = _exact_problem(model, N, t, 3, 7)
    z = _z_near_trotter(model, cell, t, D, rng)
    kw = dict(chi_max=10 ** 6, cutoff=0.0, groups=3, replicas=2, local=True)
    # P holds the sweep's environment; Q, another pool, builds the fresh ones (a new environment on a pool
    # replaces the older one in its workers)
    with ChainProblem(model, cell, t, D, N, vecs, targets, **kw) as P, \
            ChainProblem(model, cell, t, D, N, vecs, targets, **kw) as Q:
        for layers in (list(range(D)), [1, 2, 3]):
            zz = z.copy()
            env = P.envs_at(zz, layers, sweep=True)
            assert len(env.blocks) == 1
            for l in layers:
                # a bulk angle of every sublayer, and an edge angle of the parity-1 ones
                change = [na * l + int(rng.integers(na))] + ([nb + 6 * (l // 2) + int(rng.integers(6))]
                                                              if l % 2 else [])
                for i in change:
                    th = float(rng.uniform(-math.pi, math.pi))
                    zz[i] = th
                    env.set_angle(i, th)
                    fresh = Q.envs_at(zz.copy(), [l])
                    idx = list(range(na * l, na * l + na)) + (list(range(nb + 6 * (l // 2), nb + 6 * (l // 2) + 6))
                                                              if l % 2 else [])
                    got, want = env.curves(idx), fresh.curves(idx)
                    fresh.release()
                    for j in idx:
                        assert np.allclose(got[j].coeffs, want[j].coeffs, rtol=0, atol=1e-13), (layers, l, i, j)
                env.advance(l)
            # the sweep ends in the state of zz: a curve of the last sublayer equals the dense cost
            i = na * layers[-1] + 2
            cv = env.curve(i)
            for th in (zz[i], 0.7):
                zt = zz.copy()
                zt[i] = th
                exact = _dense_cost(cell, D, N, states, dense, zt)
                assert abs(cv.c_at(th) - exact) <= 1e-10 * exact
            env.release()


def test_ring_sweep_environment_gives_the_curves_of_a_fresh_one():
    from snapevo.environment import Environments
    model, t, D, N = XXZ(0.25, 1.6), 0.5, 4, 6
    cell = cell_for(model)
    na = cell.n_angles
    P = RingProblem(model, cell, t, D, N=N, hessian_method="jax")
    rng = np.random.default_rng(5)
    x = cell.trotter_angles(model, t, D) + 0.1 * rng.normal(size=D * na)
    env = P.envs_at(x, None, sweep=True)
    xx = x.copy()
    for l in range(D):
        i = na * l + int(rng.integers(na))
        xx[i] = float(rng.uniform(-math.pi, math.pi))
        env.set_angle(i, xx[i])
        fresh = Environments(xx.copy(), cell, P.tin, P.tg, N)
        for j in range(na * l, na * l + na):
            assert np.allclose(env.curve(j).coeffs, fresh.curve(j).coeffs, rtol=0, atol=1e-13)
        env.advance(l)
    fresh = Environments(xx.copy(), cell, P.tin, P.tg, N)
    assert np.allclose(env.env._out, fresh._out, rtol=0, atol=1e-13)


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_coordinate_compile_meets_the_budget_and_layer_batches_change_nothing(model, monkeypatch):
    _forbid_ring(monkeypatch)
    N, t, D = 6, 0.5, 3
    cell, vecs, states, dense, targets, z, floor = _trained(model, N, t, D, 3, 3)
    budget = 3 * max(floor, 1e-6)
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=2, replicas=2,
                      local=True) as P:
        def run(**kw):
            settings = ChainSettings(budget=budget, max_round_trials=8, edge_weight=2.0 / N, reopt="coordinate")
            return compile_chain(z, cell, D, settings, P.objective, P.c_fn, P.hessian, P.envs_at, Library(22.0),
                                 **kw)
        res = run()
        assert all(w.state.get("env") is None for row in P.pool.workers for w in row)    # caches released
        batched = run(env_layers=1)
        zc = np.asarray(res["z"])
        assert res["C_train"] <= budget * (1 + 1e-9) and res["C_train"] == P.c_fn(zc)
        assert abs(_dense_cost(cell, D, N, states, dense, zc) - res["C_train"]) <= 1e-8 * res["C_train"] + 1e-15
    for k in KEYS:
        assert batched[k] == res[k], k
    assert res["reoptimizer"] == "coordinate" and res["reopt_log"]
    for e in res["reopt_log"]:
        assert e["path"] == "coordinate" and e["stop"] in ("converged", "noise floor", "nothing free")
        if e["stop"] == "converged":
            assert e["sweeps"] >= 1 and 0.0 <= e["gain"] < 1e-6 * budget
    # the sweeps move the edges too
    assert not np.array_equal(res["edges"], z[D * cell.n_angles:])
    assert res["timing"]["coord_sweeps"] >= 1 and res["timing"]["coord_curves"] >= 1


def test_curve_minimum_is_global():
    """The coordinate step finds the global minimum of a curve with several local minima."""
    from snapevo.environment import AngleCurve
    rng = np.random.default_rng(2)
    modes = np.arange(-6, 7)
    coeffs = (rng.normal(size=(13, 4)) + 1j * rng.normal(size=(13, 4))) / 13
    cv = AngleCurve(0, coeffs, modes, 10, 0.3, 0.0)
    th = np.linspace(-2 * math.pi, 2 * math.pi, 400001)
    best = float(np.min(cv.c_at(th)))
    u, cu, c0 = cs._curve_min(cv, 0.3)
    assert cu <= best + 1e-12 and c0 == pytest.approx(float(cv.c_at(0.3)))
    assert abs(float(cv.c_at(u)) - cu) < 1e-14
