import json

import numpy as np
import pytest

from snapevo.cells import cell_for
from snapevo.chain_snap import ChainProblem, DeadlineReached, newton_minimize
from snapevo.circuits import product_vectors, random_product_states
from snapevo.hamiltonian import XXZ, evolve
from snapevo.mps_train import mps_from_dense
from snapevo.newton_fast import minimize_newton, newton_cg_minimize, newton_reuse_minimize
from snapevo.train_chain import trotter_z


def _problem(model, N=6, t=0.6, n=3, seed=4):
    states = random_product_states(N, n, seed)
    targets = [mps_from_dense(v) for v in evolve(model.sparse(N, "open"), states, t)]
    return product_vectors(N, n, seed), targets


@pytest.mark.parametrize("hx", (0.0, 1.6))
def test_reuse_and_cg_reach_the_full_newton_floor_with_fewer_gradients(hx, tmp_path):
    model = XXZ(0.25, hx)
    cell = cell_for(model)
    N, t, D = 6, 0.6, 4
    vecs, targets = _problem(model, N, t)
    z0 = trotter_z(model, cell, t, D) + 0.05 * np.random.default_rng(2).normal(size=trotter_z(model, cell, t, D).size)
    nb = D * cell.n_angles
    out = {}
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, local=True) as P:
        hess = lambda zz, rows: P.hessian(zz, rows)              # noqa: E731
        tol = lambda c: 1e-6 * c                                 # noqa: E731
        n0 = P.stats["n_grad"]
        z, info = newton_minimize(z0, P.objective, hess, tol, n_bulk=nb, checkpoint=str(tmp_path / "full.json"))
        out["full"] = (info["c_after"], info["hessians"] * 2 * z0.size + (P.stats["n_grad"] - n0), info)
        z, info = newton_reuse_minimize(z0, P.objective, hess, tol, n_bulk=nb, checkpoint=str(tmp_path / "r.json"))
        out["reuse"] = (info["c_after"], info["gradients"], info)
        z, info = newton_cg_minimize(z0, P.objective, hess, tol, n_bulk=nb, checkpoint=str(tmp_path / "cg.json"),
                                     grad_many=P.cost.value_and_grad_many)
        out["cg"] = (info["c_after"], info["gradients"], info)
    c_full = out["full"][0]
    for mode in ("reuse", "cg"):
        c, grads, info = out[mode]
        assert c <= c_full * (1 + 1e-3), (mode, c, c_full)
        assert info["stop"] in ("converged", "noise floor") and info["kept"]
        if info["stop"] == "converged":
            assert info["pred"] < 1e-6 * c * (1 + 1e-9) and info["eig_min_bulk_after"] > -1e-8 * info["eig_max_after"]
        assert grads < out["full"][1], (mode, grads, out["full"][1])
    assert out["cg"][2]["hvps"] > 0 and out["cg"][2]["mode"] == "cg"


def test_a_stopped_run_resumes_and_modes_share_checkpoints(tmp_path):
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    N, t, D = 6, 0.6, 4
    vecs, targets = _problem(model, N, t)
    z0 = trotter_z(model, cell, t, D) + 0.05
    nb = D * cell.n_angles
    ck = str(tmp_path / "run.json")
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, local=True) as P:
        hess = lambda zz, rows: P.hessian(zz, rows)              # noqa: E731
        tol = lambda c: 1e-6 * c                                 # noqa: E731
        ref = newton_reuse_minimize(z0, P.objective, hess, tol, n_bulk=nb)[1]["c_after"]
        calls = {"n": 0}

        def objective(zz):
            calls["n"] += 1
            if calls["n"] == 6:
                raise_stop["on"] = True
            return P.objective(zz)

        import threading
        ev = threading.Event()
        raise_stop = {"on": False}

        def obj2(zz):
            out = objective(zz)
            if raise_stop["on"]:
                ev.set()
            return out
        with pytest.raises(DeadlineReached):
            newton_reuse_minimize(z0, obj2, hess, tol, n_bulk=nb, checkpoint=ck, stop=ev)
        doc = json.loads(open(ck).read())
        assert doc["steps"] >= 1 and doc["info"]["mode"] == "reuse"
        # continue with another mode from the same checkpoint
        z, info = minimize_newton("cg", z0, P.objective, hess, tol, grad_many=P.cost.value_and_grad_many, n_bulk=nb,
                                  checkpoint=ck)
        assert info["c_after"] <= ref * (1 + 1e-3) and info["stop"] in ("converged", "noise floor")
        z, info = minimize_newton("full", z0, P.objective, hess, tol, n_bulk=nb, checkpoint=ck)
        assert info["stop"] == "converged"


@pytest.mark.parametrize("mode", ("reuse", "cg"))
def test_ladder_and_precompute_run_the_cheaper_newton_modes(mode, tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_train_chain import _eval_refs

    import snapevo.train_chain_best as tcb
    model, N, t = XXZ(0.25, 1.6), 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    kw = dict(runs=runs, N=N, n_inputs=3, ref_cache=tmp_path / "refs", local=True, log=lambda s: None,
              optimizers=("newton",), newton_mode=mode)
    assert tcb.precompute_chain_runs(model, t, [2], tasks=1, **kw) == 2
    out = tcb.train_chain_ladder_best(model, t, [2, 3], stall=None, target=None, heldout_processes=1, **kw)
    doc = json.loads(out.read_text())
    assert doc["newton_mode"] == mode
    for r in doc["results"]:
        for v in r["runs"].values():
            assert v["stop"] in ("converged", "noise floor")
    ck = sorted((runs / model.tag / f"t{t:g}" / "chain" / "ckpt").glob("D3_warm_newton_*.json"))[0]
    assert json.loads(ck.read_text())["info"]["mode"] == mode


def test_a_full_newton_checkpoint_resumes_under_reuse(tmp_path):
    import threading
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    N, t, D = 6, 0.6, 4
    vecs, targets = _problem(model, N, t)
    z0 = trotter_z(model, cell, t, D) + 0.05
    nb = D * cell.n_angles
    ck = str(tmp_path / "run.json")
    with ChainProblem(model, cell, t, D, N, vecs, targets, chi_max=10 ** 6, cutoff=0.0, groups=3, local=True) as P:
        hess = lambda zz, rows: P.hessian(zz, rows)              # noqa: E731
        tol = lambda c: 1e-6 * c                                 # noqa: E731
        ref = newton_minimize(z0, P.objective, hess, tol, n_bulk=nb)[1]["c_after"]
        ev, calls = threading.Event(), {"n": 0}

        def obj(zz):
            calls["n"] += 1
            if calls["n"] == 8:
                ev.set()
            return P.objective(zz)
        with pytest.raises(DeadlineReached):
            newton_minimize(z0, obj, hess, tol, n_bulk=nb, checkpoint=ck, stop=ev)
        steps_before = json.loads(open(ck).read())["steps"]
        z, info = newton_reuse_minimize(z0, P.objective, hess, tol, n_bulk=nb, checkpoint=ck)
    assert steps_before >= 1 and info["steps"] >= steps_before and info["c_before"] > info["c_after"]
    assert info["stop"] == "converged" and info["c_after"] <= ref * (1 + 1e-3)
