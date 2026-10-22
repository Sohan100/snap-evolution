import json

import numpy as np
import pytest

import snapevo.train_chain_best as tcb
from snapevo.cells import cell_for
from snapevo.hamiltonian import XXZ
from test_train_chain import _eval_refs, _forbid_ring

FIELDS = (XXZ(0.25, 0.0), XXZ(0.25, 1.6))
IDS = ("hx0", "hx1.6")


def test_lbfgs_minimize_runs_to_convergence_without_a_cap_and_resumes(tmp_path):
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    A = A @ A.T + 0.1 * np.eye(6)

    def obj(z):
        return 1e-6 + 0.5 * z @ A @ z, A @ z

    z0 = rng.normal(size=6)
    ck = str(tmp_path / "ck.json")
    z, info = tcb.lbfgs_minimize(z0, obj, tol_rel=1e-6, checkpoint=ck)
    assert info["stop"] in ("converged", "noise floor") or info["stop"].startswith("lbfgsb")
    assert info["c_after"] < 1e-6 * (1 + 1e-5)
    z2, info2 = tcb.lbfgs_minimize(z0, obj, tol_rel=1e-6, checkpoint=ck)      # finished: returned from the checkpoint
    assert np.array_equal(z, z2) and info2["c_after"] == info["c_after"]


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_best_ladder_keeps_the_lowest_run_on_the_open_chain_only(model, tmp_path, monkeypatch):
    _forbid_ring(monkeypatch)
    N, t = 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    out = tcb.train_chain_ladder_best(model, t, [2, 3], runs=runs, N=N, n_inputs=3, ref_cache=tmp_path / "refs",
                                      stall=None, target=None, local=True, heldout_processes=1, log=lambda s: None)
    doc = json.loads(out.read_text())
    recs = {r["D"]: r for r in doc["results"]}
    assert sorted(recs) == [2, 3]
    assert set(recs[3]["runs"]) == {f"{s}/{o}" for s in ("trotter", "warm", "random") for o in ("newton", "lbfgs")}
    cell = cell_for(model)
    for D, r in recs.items():
        assert len(r["z"]) == D * cell.n_angles + 6 * (D // 2)
        assert r["c_train"] == min(x["c"] for x in r["runs"].values())
        assert r["optimizer"] in ("newton", "lbfgs") and 0.0 <= r["c_heldout"] < 1.0
    monkeypatch.setattr(tcb, "newton_minimize", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(tcb, "lbfgs_minimize", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    tcb.train_chain_ladder_best(model, t, [2, 3], runs=runs, N=N, n_inputs=3, ref_cache=tmp_path / "refs",
                                stall=None, target=None, local=True, heldout_processes=1, log=lambda s: None)


def test_precomputed_runs_are_taken_over_by_the_ladder(tmp_path, monkeypatch):
    model, N, t = XXZ(0.25, 1.6), 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    kw = dict(runs=runs, N=N, n_inputs=3, ref_cache=tmp_path / "refs", local=True, log=lambda s: None)
    n = tcb.precompute_chain_runs(model, t, [2, 3], tasks=1, **kw)
    assert n == 2 * 2 * 2                                                 # 2 depths x (trotter, random) x 2 optimizers
    assert tcb.precompute_chain_runs(model, t, [2, 3], tasks=1, **kw) == 0          # nothing left to do
    calls = []
    real_newton, real_lbfgs = tcb.newton_minimize, tcb.lbfgs_minimize
    monkeypatch.setattr(tcb, "newton_minimize", lambda z, *a, **k: calls.append("newton") or real_newton(z, *a, **k))
    monkeypatch.setattr(tcb, "lbfgs_minimize", lambda z, *a, **k: calls.append("lbfgs") or real_lbfgs(z, *a, **k))
    tcb.train_chain_ladder_best(model, t, [2, 3], stall=None, target=None, heldout_processes=1, **kw)
    assert sorted(calls) == ["lbfgs", "newton"]                          # only the warm start of D = 3 was run
    doc = json.loads((runs / model.tag / f"t{t:g}" / "chain" / "ladder.json").read_text())
    assert [r["D"] for r in doc["results"]] == [2, 3]
