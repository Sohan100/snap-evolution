import json

import numpy as np
import pytest

import snapevo.edges
import snapevo.jaxcost
import snapevo.mps_train
import snapevo.train
from snapevo.cells import cell_for
from snapevo.circuits import Layered, local_infidelity, product_vectors, random_product_states
from snapevo.edges import pairs
from snapevo.evaluate import ref_path
from snapevo.hamiltonian import XXZ, evolve
from snapevo.mps import reference_state
from snapevo.train_chain import EVAL_DT, EVAL_SET, pad_z, split_z, train_chain_ladder, trotter_z

FIELDS = (XXZ(0.25, 0.0), XXZ(0.25, 1.6))
IDS = ("hx0", "hx1.6")


def _dense_cost(cell, D, N, states, targets, z):
    x, e = split_z(cell, z, D)
    out = Layered.brickwork(cell, x).apply(states, N, bc="open", edges=pairs(e))
    return local_infidelity(out, targets, N)


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_trotter_z_is_the_open_chain_trotter_brickwork(model):
    cell, N, t, D = cell_for(model), 8, 0.5, 5
    states = random_product_states(N, 3, 7)
    targets = evolve(model.sparse(N, "open"), states, t)
    x = cell.trotter_angles(model, t, D)
    own = Layered.brickwork(cell, x).apply(states, N, bc="open", edges=None)    # the cell's own end factor
    assert _dense_cost(cell, D, N, states, targets, trotter_z(model, cell, t, D)) == pytest.approx(
        local_infidelity(own, targets, N), rel=1e-10, abs=1e-14)


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
@pytest.mark.parametrize("D_old,D_new", [(2, 3), (3, 6), (4, 7)])
def test_padding_with_identity_sublayers_keeps_the_circuit(model, D_old, D_new):
    cell, N = cell_for(model), 8
    rng = np.random.default_rng(D_old * 10 + D_new)
    states = random_product_states(N, 3, 3)
    targets = evolve(model.sparse(N, "open"), states, 0.4)
    z = rng.uniform(-np.pi, np.pi, size=D_old * cell.n_angles + 6 * (D_old // 2))
    zp = pad_z(cell, z, D_old, D_new)
    assert zp.size == D_new * cell.n_angles + 6 * (D_new // 2)
    assert _dense_cost(cell, D_new, N, states, targets, zp) == pytest.approx(
        _dense_cost(cell, D_old, N, states, targets, z), rel=1e-11, abs=1e-14)


def _eval_refs(model, t, N, runs):
    d = runs / model.tag / f"t{t:g}" / "ref"
    d.mkdir(parents=True)
    seed, n = EVAL_SET
    vecs = product_vectors(N, n, seed)
    for k in range(n):
        for c in (1e-12, 1e-14):
            reference_state(model, t, vecs[k], EVAL_DT, 1024, c, ref_path(d, N, seed, k, EVAL_DT, c, 1024))


def _forbid_ring(monkeypatch):
    def no(*_a, **_k):
        raise AssertionError("the open-chain arm touched the ring")

    real_apply = Layered.apply

    def apply(self, states, N, dagger=False, bc="ring", edges=None):
        if bc != "open":
            no()
        return real_apply(self, states, N, dagger, bc, edges)

    monkeypatch.setattr(Layered, "apply", apply)
    monkeypatch.setattr(snapevo.train, "train_ladder", no)
    monkeypatch.setattr(snapevo.train, "load_cell", no)
    monkeypatch.setattr(snapevo.jaxcost, "make_cost", no)
    monkeypatch.setattr(snapevo.mps_train, "ring_angles", no)
    monkeypatch.setattr(snapevo.mps_train, "init_from_ring", no)
    monkeypatch.setattr(snapevo.edges, "fit_edges", no)


@pytest.mark.parametrize("model", FIELDS, ids=IDS)
def test_ladder_trains_on_the_open_chain_only_and_resumes(model, tmp_path, monkeypatch):
    _forbid_ring(monkeypatch)
    N, t = 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    lines = []
    out = train_chain_ladder(model, t, [2, 3], runs=runs, N=N, n_inputs=3, maxiter=8, processes=1,
                             ref_cache=tmp_path / "refs", parallel_starts=False, stall=None, log=lines.append)
    doc = json.loads(out.read_text())
    assert doc["bc"] == "open" and doc["N"] == N
    recs = {r["D"]: r for r in doc["results"]}
    assert sorted(recs) == [2, 3]
    assert set(recs[2]["starts"]) == {"trotter", "random"}                  # no previous depth to warm-start from
    assert set(recs[3]["starts"]) == {"trotter", "warm", "random"}
    cell = cell_for(model)
    for D, r in recs.items():
        assert len(r["z"]) == D * cell.n_angles + 6 * (D // 2)
        assert r["c_train"] == min(s["c"] for s in r["starts"].values())
        assert 0.0 <= r["c_heldout"] < 1.0
        assert r["maxiter"] == 8 and r["heldout_cutoff"] == r["train_cutoff"] == 1e-14
    # the warm start begins at the previous depth's trained circuit
    warm = json.loads((runs / model.tag / f"t{t:g}" / "chain" / "ckpt" / "D3_warm_c1e-14.json").read_text())
    assert warm["trace"][0][1] == pytest.approx(recs[2]["c_train"], rel=1e-9)
    # a rerun trains nothing
    monkeypatch.setattr(snapevo.mps_train, "train_open", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    train_chain_ladder(model, t, [2, 3], runs=runs, N=N, n_inputs=3, maxiter=8, processes=1,
                       ref_cache=tmp_path / "refs", parallel_starts=False, stall=None, log=lines.append)


def test_a_rerun_respects_the_stall_rule_of_finished_depths(tmp_path, monkeypatch):
    model, N, t = XXZ(0.25, 0.0), 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    kw = dict(runs=runs, N=N, n_inputs=3, maxiter=4, processes=1, ref_cache=tmp_path / "refs", parallel_starts=False,
              log=lambda s: None)
    train_chain_ladder(model, t, [2, 3], stall=None, **kw)
    out = runs / model.tag / f"t{t:g}" / "chain" / "ladder.json"
    doc = json.loads(out.read_text())
    doc["results"][1]["c_train"] = doc["results"][0]["c_train"]            # D = 3 no better than D = 2: stalled
    out.write_text(json.dumps(doc))
    monkeypatch.setattr(snapevo.mps_train, "train_open", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    train_chain_ladder(model, t, [2, 3, 5], stall=(2.0, 1), **kw)          # must stop at D = 3, not train D = 5


def test_two_runs_of_one_ladder_cannot_overlap(tmp_path):
    import fcntl
    model, N, t = XXZ(0.25, 0.0), 6, 0.3
    runs = tmp_path / "runs"
    _eval_refs(model, t, N, runs)
    d = runs / model.tag / f"t{t:g}" / "chain"
    d.mkdir(parents=True)
    held = open(d / "lock", "w")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(RuntimeError, match="another run"):
        train_chain_ladder(model, t, [2], runs=runs, N=N, n_inputs=3, maxiter=2, processes=1,
                           ref_cache=tmp_path / "refs", parallel_starts=False, log=lambda s: None)
