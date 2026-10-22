import json
import math

import numpy as np
import pytest
import scipy.optimize

import snapevo.edges
import snapevo.jaxcost
import snapevo.mps_train
import snapevo.train
import snapevo.train_chain
import snapevo.train_newton as tn
from snapevo.cells import cell_for
from snapevo.circuits import Layered, product_vectors
from snapevo.edges import realize_edges, worth_it
from snapevo.edges_newton import fit_edges_newton
from snapevo.evaluate import ref_path
from snapevo.hamiltonian import XXZ
from snapevo.mps import reference_state
from snapevo.train import load_cell
from snapevo.train_chain import EVAL_DT, EVAL_SET

FIELDS = (XXZ(0.25, 0.0), XXZ(0.25, 1.6))
IDS = ("hx0", "hx1.6")


@pytest.fixture
def no_lbfgs(monkeypatch):
    """Fail on any L-BFGS-B call; count the other optimizer calls."""
    real = scipy.optimize.minimize
    methods = []

    def minimize(*a, **k):
        method = str(k.get("method", "")).lower()
        if method in ("", "l-bfgs-b", "lbfgsb"):
            raise AssertionError(f"an optimizer other than Newton was called: {k.get('method')!r}")
        methods.append(method)
        return real(*a, **k)

    monkeypatch.setattr(scipy.optimize, "minimize", minimize)
    return methods


def _eval_refs(model, t, N, runs):
    d = runs / model.tag / f"t{t:g}" / "ref"
    d.mkdir(parents=True, exist_ok=True)
    seed, n = EVAL_SET
    vecs = product_vectors(N, n, seed)
    for k in range(n):
        for c in (1e-12, 1e-14):
            reference_state(model, t, vecs[k], EVAL_DT, 1024, c, ref_path(d, N, seed, k, EVAL_DT, c, 1024))


def _forbid_ring(monkeypatch):
    def no(*_a, **_k):
        raise AssertionError("the chain trainer touched the ring or an L-BFGS trainer")

    real_apply = Layered.apply

    def apply(self, states, N, dagger=False, bc="ring", edges=None):
        if bc != "open":
            no()
        return real_apply(self, states, N, dagger, bc, edges)

    monkeypatch.setattr(Layered, "apply", apply)
    monkeypatch.setattr(snapevo.train, "train_ladder", no)
    monkeypatch.setattr(snapevo.train, "load_cell", no)
    monkeypatch.setattr(snapevo.jaxcost, "make_cost", no)
    monkeypatch.setattr(snapevo.jaxcost, "make_hessian", no)
    monkeypatch.setattr(snapevo.mps_train, "ring_angles", no)
    monkeypatch.setattr(snapevo.mps_train, "init_from_ring", no)
    monkeypatch.setattr(snapevo.mps_train, "train_open", no)
    monkeypatch.setattr(snapevo.train_chain, "train_chain_ladder", no)
    monkeypatch.setattr(snapevo.edges, "fit_edges", no)


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_chain_ladder_trains_by_newton_on_the_open_chain_only_and_resumes(model, tmp_path, monkeypatch, no_lbfgs):
    _forbid_ring(monkeypatch)
    N, t = 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    lines = []
    out = tn.train_chain_ladder_newton(model, t, [2, 3], runs=runs, N=N, n_inputs=3, groups=3, local=True,
                                       ref_cache=tmp_path / "refs", parallel_starts=False, stall=None,
                                       target=None, heldout_processes=1, log=lines.append)
    assert no_lbfgs and set(no_lbfgs) == {"trust-exact"}
    doc = json.loads(out.read_text())
    assert doc["bc"] == "open" and doc["optimizer"] == "newton" and doc["N"] == N
    recs = {r["D"]: r for r in doc["results"]}
    assert sorted(recs) == [2, 3]
    assert set(recs[2]["starts"]) == {"trotter", "random"} and set(recs[3]["starts"]) == {"trotter", "warm", "random"}
    cell = cell_for(model)
    for D, r in recs.items():
        assert len(r["z"]) == D * cell.n_angles + 6 * (D // 2)
        assert r["optimizer"] == "newton" and r["c_train"] == min(s["c"] for s in r["starts"].values())
        for s in r["starts"].values():
            assert s["stop"] in ("converged", "noise floor") and s["c"] <= s["c_start"] and s["steps"] >= 0
            assert s["hessians"] >= 1
            if s["stop"] == "converged":
                assert s["pred"] < 1e-6 * s["c"] * (1 + 1e-9)
        assert 0.0 <= r["c_heldout"] < 1.0
    # the warm start begins at the previous depth's trained circuit
    warm = json.loads(next((runs / model.tag / f"t{t:g}" / "chain" / "ckpt").glob("D3_warm_newton_*.json")).read_text())
    assert warm["info"]["c_before"] == pytest.approx(recs[2]["c_train"], rel=1e-9)
    # a rerun trains nothing
    monkeypatch.setattr(tn, "newton_minimize", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    tn.train_chain_ladder_newton(model, t, [2, 3], runs=runs, N=N, n_inputs=3, groups=3, local=True,
                                 ref_cache=tmp_path / "refs", parallel_starts=False, stall=None, target=None,
                                 heldout_processes=1, log=lines.append)


def test_chain_ladder_stops_at_its_deadline_and_continues(tmp_path, monkeypatch):
    model, N, t = XXZ(0.25, 1.6), 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    kw = dict(runs=runs, N=N, n_inputs=3, groups=3, local=True, ref_cache=tmp_path / "refs", parallel_starts=False,
              stall=None, target=None, heldout_processes=1, log=lambda _s: None)
    out = tn.train_chain_ladder_newton(model, t, [2], deadline=0.0, **kw)
    assert not out.exists() or not json.loads(out.read_text())["results"]
    out = tn.train_chain_ladder_newton(model, t, [2], **kw)
    assert [r["D"] for r in json.loads(out.read_text())["results"]] == [2]


def test_ring_ladder_trains_by_newton_into_ring2(tmp_path, no_lbfgs):
    model, t = XXZ(0.25, 0.0), 0.5
    runs = tmp_path / "runs"
    out = tn.train_ring_ladder_newton(model, t, [2, 3], runs=runs, stall=None, target=None, log=lambda _s: None)
    assert out == runs / model.tag / f"t{t:g}" / "ring2" / "ladder.json"
    assert set(no_lbfgs) == {"trust-exact"}
    doc = json.loads(out.read_text())
    assert doc["optimizer"] == "newton" and doc["N"] == 10
    cell = cell_for(model)
    for r in doc["results"]:
        assert np.asarray(r["params"]).shape == (r["D"], cell.n_angles)
        assert r["C_train"] == min(s["c"] for s in r["starts"].values()) and 0 < r["C_val"] < 1
        for s in r["starts"].values():
            assert s["stop"] in ("converged", "noise floor")
    x, header = load_cell(out, 3)                          # cells.json's format
    assert x.size == 3 * cell.n_angles and header["cell"] == cell.name
    f = snapevo.jaxcost.make_cost(cell, 10, *_ring_pairs(model, t), 3)
    assert f(x)[0] == pytest.approx(doc["results"][1]["C_train"], rel=1e-10)


def _ring_pairs(model, t):
    from snapevo.circuits import TRAIN_SET, state_pairs
    return state_pairs(model, 10, t, TRAIN_SET[1], TRAIN_SET[0])


def test_newton_edge_fit_has_fit_edges_record_and_never_calls_lbfgs(no_lbfgs):
    model, t, D = XXZ(0.25, 0.0), 1.0, 5
    cell = cell_for(model)
    x = cell.trotter_angles(model, t, D) + 0.05 * np.random.default_rng(3).normal(size=D * cell.n_angles)
    circuit = Layered.brickwork(cell, x)
    fit = fit_edges_newton(circuit, model, t, N_fit=8, log=lambda _s: None)
    assert set(no_lbfgs) == {"trust-exact"}
    for k in ("N_fit", "states", "angles", "infid_start", "infid_fit", "infid_fit_unsnapped", "n_snapped",
              "infid_ring", "nit", "seconds"):
        assert k in fit
    assert fit["infid_fit_unsnapped"] <= fit["infid_start"] and fit["newton"]["stop"] in ("converged", "noise floor")
    assert fit["infid_fit"] <= fit["infid_fit_unsnapped"] + 0.1 * fit["infid_ring"] / 8 + 1e-15
    assert worth_it(fit) in (True, False)                            # edges.worth_it and realize_edges take it
    if not worth_it(fit):
        assert realize_edges(fit, 1e-4)["use"] is False
    for a in fit["angles"]:
        assert math.isfinite(a)
