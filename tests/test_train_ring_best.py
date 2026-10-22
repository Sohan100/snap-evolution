import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

import snapevo.train_ring_best as trb
from snapevo.cells import cell_for
from snapevo.chain_snap import ring_point
from snapevo.circuits import TRAIN_SET, Layered, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ
from snapevo.jaxcost import make_cost
from snapevo.train import L_TRAIN, VAL_SET, cell_records, load_cell

T = 0.5
LADDERS: dict = {}


def _ladder(tmp_path_factory, model, depths=(2, 3)):
    """A small best-of ring ladder (in-process runs), trained once per model."""
    key = (model.hx, tuple(depths))
    if key not in LADDERS:
        runs = tmp_path_factory.mktemp("runs")
        lines = []
        out = trb.train_ring_ladder_best(model, T, list(depths), runs=runs, workers=1, stall=None, target=None,
                                         log=lines.append)
        LADDERS[key] = (runs, out, lines)
    return LADDERS[key]


@pytest.mark.parametrize("model", (XXZ(0.25, 0.0), XXZ(0.25, 1.6)), ids=("hx0", "hx1.6"))
def test_best_of_ladder_keeps_the_lowest_run_in_cells_format(model, tmp_path_factory, monkeypatch):
    runs, out, _lines = _ladder(tmp_path_factory, model)
    cell = cell_for(model)
    assert out == runs / model.tag / f"t{T:g}" / "ring2" / "ladder.json"
    doc = json.loads(out.read_text())
    assert doc["N"] == L_TRAIN and doc["train_set"] == list(TRAIN_SET) and doc["val_set"] == list(VAL_SET)
    assert doc["cell"] == cell.name and doc["model"] == model.to_dict() and doc["optimizer"] == "best of newton, lbfgs"
    recs = {r["D"]: r for r in cell_records(out)}
    assert sorted(recs) == [2, 3]
    assert set(recs[2]["runs"]) == {"trotter/newton", "trotter/lbfgs", "random/newton", "random/lbfgs"}
    assert set(recs[3]["runs"]) == {f"{s}/{o}" for s in trb.STARTS for o in trb.OPTIMIZERS}
    tin, tg = state_pairs(model, L_TRAIN, T, TRAIN_SET[1], TRAIN_SET[0])
    vin, vtg = state_pairs(model, L_TRAIN, T, VAL_SET[1], VAL_SET[0])
    for D, r in recs.items():
        x, header = load_cell(out, D)                       # train.load_cell reads it as it reads cells.json
        assert x.size == D * cell.n_angles and header["cell"] == cell.name
        assert r["C_train"] == min(v["c"] for v in r["runs"].values())
        best = r["runs"][f"{r['start']}/{r['optimizer']}"]
        assert best["c"] == r["C_train"]
        assert make_cost(cell, L_TRAIN, tin, tg, D, precise=True)(x)[0] == pytest.approx(r["C_train"], rel=1e-9)
        assert r["C_val"] == pytest.approx(local_infidelity(Layered.brickwork(cell, x).apply(vin, L_TRAIN), vtg,
                                                            L_TRAIN), rel=1e-12)
        for v in r["runs"].values():
            assert v["c"] <= v["c_start"] and v["stop"] is not None and "TOTAL NO." not in str(v["stop"])
    # the warm start of D = 3 is D = 2 padded with an identity sublayer
    warm = json.loads((runs / model.tag / f"t{T:g}" / "ring2" / "ckpt" / "D3_warm_newton.json").read_text())
    p2 = np.asarray(recs[2]["params"])
    assert np.array_equal(np.asarray(warm["z_in"]), np.vstack([p2, np.zeros((1, cell.n_angles))]).reshape(-1))
    # a rerun trains nothing
    monkeypatch.setattr(trb, "ring_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("trained again")))
    trb.train_ring_ladder_best(model, T, [2, 3], runs=runs, workers=1, stall=None, target=None, log=lambda _s: None)


def test_spawned_workers_give_the_in_process_ladder(tmp_path_factory, tmp_path):
    model = XXZ(0.25, 0.0)
    runs, out, _ = _ladder(tmp_path_factory, model)
    out2 = trb.train_ring_ladder_best(model, T, [2], runs=tmp_path, workers=4, stall=None, target=None,
                                      log=lambda _s: None)
    a = cell_records(out)[0]
    b = cell_records(out2)[0]
    assert b["D"] == 2 and b["C_train"] == pytest.approx(a["C_train"], rel=1e-12, abs=1e-18)
    assert b["start"] == a["start"] and b["optimizer"] == a["optimizer"]


def test_a_run_stopped_at_its_deadline_resumes_from_its_checkpoint(tmp_path):
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    z0 = cell.trotter_angles(model, T, 3)
    for opt in trb.OPTIMIZERS:
        ck = tmp_path / f"{opt}.json"
        spec = {"model": model.to_dict(), "t": T, "D": 3, "start": "trotter", "z0": z0.tolist(), "optimizer": opt,
                "checkpoint": str(ck), "deadline": None, "tol_rel": trb.TOL_REL}
        ref = trb.ring_run({**spec, "checkpoint": str(tmp_path / f"{opt}_ref.json")})
        assert ref["status"] == "done"
        # a deadline already past: newton stops at its first evaluation, L-BFGS-B after its first iteration
        stopped = trb.ring_run({**spec, "deadline": time.time() - 1.0})
        assert stopped["status"] == "deadline" and ck.exists()
        again = trb.ring_run(spec)
        assert again["status"] == "done"
        assert again["info"]["c_after"] == pytest.approx(ref["info"]["c_after"], rel=1e-3)
        # a finished run returns at once from its checkpoint
        assert trb.ring_run(spec)["info"]["c_after"] == pytest.approx(again["info"]["c_after"], rel=1e-12)


def test_deadline_stops_the_ladder_before_a_depth_and_a_rerun_continues(tmp_path):
    model = XXZ(0.25, 0.0)
    out = trb.train_ring_ladder_best(model, T, [2], runs=tmp_path, workers=1, deadline=0.0, stall=None,
                                     target=None, log=lambda _s: None)
    assert not out.exists()
    out = trb.train_ring_ladder_best(model, T, [2], runs=tmp_path, workers=1, stall=None, target=None,
                                     log=lambda _s: None)
    assert [r["D"] for r in cell_records(out)] == [2]


def test_depth_grid_is_the_production_grid_plus_cells_depths():
    runs = Path(__file__).resolve().parent.parent / "runs"
    if not (runs / "xxz_d0.25_hx0" / "t4" / "cells.json").exists():
        pytest.skip("production cells.json not in this checkout")
    g = trb.depth_grid(XXZ(0.25, 0.0), 4, runs)
    assert 19 in g and g[0] == 3 and g[-1] == 97 and g == sorted(set(g))
    assert trb.depth_grid(XXZ(0.25, 1.6), 10, runs)[-1] == 161


def _fake_kn(monkeypatch):
    """KN stand-in for realize and realize_edges (keeps the test off the shared synthesis cache)."""
    import math

    import snapevo.edges
    import snapevo.realize

    def synth(theta, eps, **_kw):
        return {"theta": theta, "eps": eps, "expected_t": 1.1 * math.log2(1.0 / eps) + 4.4, "t_v": 10, "t_b": 20,
                "p": 0.99, "theta_s": theta + 0.5 * eps, "d0": 0.5 * eps, "d1": eps, "d_branch": 0.6 * eps,
                "bound": eps}
    monkeypatch.setattr(snapevo.realize, "synthesize", synth)
    if hasattr(snapevo.edges, "synthesize"):
        monkeypatch.setattr(snapevo.edges, "synthesize", synth)


def test_ring_point_compiles_every_depth_of_the_ring2_ladder_with_newton_and_the_newton_edge_fit(
        tmp_path_factory, tmp_path, monkeypatch):
    _fake_kn(monkeypatch)
    model = XXZ(0.25, 0.0)
    runs0, out, _ = _ladder(tmp_path_factory, model)
    runs = tmp_path / "runs"
    shutil.copytree(runs0, runs)
    recs = cell_records(out)
    budget = 3.0 * max(max(r["C_train"], r["C_val"]) for r in recs)
    lines = []
    res = ring_point(model, T, budget, runs, out_sub="ring2/protocol", cells="ring2/ladder.json",
                     settings_kw={"reopt": "newton", "max_round_trials": 8}, steps=("compile", "realize"),
                     edge_fit="newton", all_depths=True, log=lines.append)
    assert res == []                                              # no evaluation asked for
    d = runs / model.tag / f"t{T:g}" / "ring2" / "protocol"
    for r in recs:
        snap = json.loads((d / f"snap_b{budget:.1e}_D{r['D']}.json").read_text())
        real = json.loads((d / f"real_b{budget:.1e}_D{r['D']}.json").read_text())
        assert snap["cells"] == "ring2/ladder.json" and snap["reoptimizer"] == "newton"
        assert snap["ladder_C_train"] == r["C_train"] and snap["C_train"] <= budget * (1 + 1e-9)
        assert all(e["path"] == "newton" for e in snap["reopt_log"])
        assert real["edge_fit"] == "newton" and "newton" in real["edges"]
    # records of another ladder are not reused
    shutil.copy(out.parent / "ladder.json", runs / model.tag / f"t{T:g}" / "ring2" / "other.json")
    with pytest.raises(ValueError, match="compiled from"):
        ring_point(model, T, budget, runs, out_sub="ring2/protocol", cells="ring2/other.json",
                   settings_kw={"reopt": "newton"}, steps=("compile",), edge_fit="newton", all_depths=True,
                   log=lambda _s: None)


def test_newton_edge_fit_hessian_is_the_chunked_exact_hessian():
    from snapevo.circuits import random_product_states
    from snapevo.edges import start_angles
    from snapevo.edges_newton import HESSIAN_CHUNK, edge_objective
    from snapevo.hamiltonian import evolve
    model, t, D, N = XXZ(0.25, 1.6), 0.7, 7, 6
    cell = cell_for(model)
    x = cell.trotter_angles(model, t, D) + 0.05 * np.random.default_rng(1).normal(size=D * cell.n_angles)
    circ = Layered.brickwork(cell, x)
    xin = random_product_states(N, 3, 4)
    f, h = edge_objective(circ, N, xin, evolve(model.sparse(N, "open"), xin, t))
    z = start_angles(circ) + 0.1
    assert z.size > HESSIAN_CHUNK                                 # several chunks, the last one padded
    H = h(z)
    eps = 1e-5
    fd = np.array([(f(z + eps * e)[1] - f(z - eps * e)[1]) / (2 * eps) for e in np.eye(z.size)])
    assert np.allclose(H, 0.5 * (fd + fd.T), rtol=0, atol=1e-7 * np.abs(H).max())
    assert np.array_equal(h(z, rows=[1, 3]), H[np.ix_([1, 3], [1, 3])])


def test_newton_runs_switch_to_difference_hessians_above_the_exact_limit(tmp_path):
    model = XXZ(0.25, 0.0)
    cell = cell_for(model)
    z0 = cell.trotter_angles(model, T, 3)
    base = {"model": model.to_dict(), "t": T, "D": 3, "start": "trotter", "z0": z0.tolist(), "optimizer": "newton",
            "deadline": None, "tol_rel": trb.TOL_REL}
    exact = trb.ring_run({**base, "checkpoint": str(tmp_path / "a.json")})
    fd = trb.ring_run({**base, "checkpoint": str(tmp_path / "b.json"), "hessian_exact_max": 0})
    assert exact["info"]["hessian"] == "jax" and fd["info"]["hessian"] == "fd"
    assert fd["info"]["c_after"] == pytest.approx(exact["info"]["c_after"], rel=1e-6)
    tin, tg = state_pairs(model, L_TRAIN, T, TRAIN_SET[1], TRAIN_SET[0])
    f = make_cost(cell, L_TRAIN, tin, tg, 3, precise=True)
    x = np.asarray(exact["z"])
    assert np.allclose(trb.make_fd_hessian(f)(x), trb.make_precise_hessian(cell, L_TRAIN, tin, tg, 3)(x),
                       rtol=0, atol=1e-6 * np.abs(trb.make_precise_hessian(cell, L_TRAIN, tin, tg, 3)(x)).max())


def test_parallel_difference_hessian_equals_the_serial_one(tmp_path):
    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    D = 3
    tin, tg = state_pairs(model, L_TRAIN, T, TRAIN_SET[1], TRAIN_SET[0])
    f = make_cost(cell, L_TRAIN, tin, tg, D, precise=True)
    x = cell.trotter_angles(model, T, D) + 0.03
    H, ex = trb.make_parallel_fd_hessian(model, T, D, 3)
    try:
        assert np.array_equal(H(x), trb.make_fd_hessian(f)(x))
    finally:
        ex.shutdown()
    base = {"model": model.to_dict(), "t": T, "D": D, "start": "trotter", "z0": cell.trotter_angles(model, T, D).tolist(),
            "optimizer": "newton", "deadline": None, "tol_rel": trb.TOL_REL, "hessian_exact_max": 0}
    one = trb.ring_run({**base, "checkpoint": str(tmp_path / "a.json")})
    par = trb.ring_run({**base, "checkpoint": str(tmp_path / "b.json"), "hessian_workers": 2})
    assert par["status"] == "done" and par["info"]["hessian"] == "fd"
    assert par["info"]["c_after"] == one["info"]["c_after"] and par["z"] == one["z"]


def test_uncapped_lbfgs_edge_fit_is_fit_edges_when_that_stops_before_its_cap():
    from snapevo.edges import fit_edges
    from snapevo.edges_newton import fit_edges_lbfgs
    model, t, D = XXZ(0.25, 1.6), 0.7, 5
    cell = cell_for(model)
    x = cell.trotter_angles(model, t, D) + 0.05 * np.random.default_rng(2).normal(size=D * cell.n_angles)
    circ = Layered.brickwork(cell, x)
    a = fit_edges(circ, model, t, N_fit=8, log=lambda _s: None)
    b = fit_edges_lbfgs(circ, model, t, N_fit=8, log=lambda _s: None)
    assert a["nit"] < 1000
    for k in ("angles", "infid_start", "infid_fit", "infid_fit_unsnapped", "n_snapped", "infid_ring", "nit"):
        assert a[k] == b[k], k
