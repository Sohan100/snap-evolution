import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

import snapevo.baseline as base
from snapevo import baseline_lit as bl
from snapevo.cells import CELLS
from snapevo.circuits import product_vectors, random_product_states
from snapevo.hamiltonian import XXZ, get_model
from snapevo.mps import MPS

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def fake_synth(theta, eps):
    """Deterministic stand-in for KN with the same record fields (no files, no search)."""
    u = math.sin(1e3 * theta)
    return {"expected_t": 1.1 * math.log2(1 / eps) + 4.4 + 0.5 * u, "p": 0.95 + 0.01 * u,
            "theta_s": theta + 0.3 * eps * u, "d_branch": eps * (1 + u * u)}


def _eye(N):
    return np.eye(2 ** N, dtype=complex)


def _err(f, N, H, t):
    return np.linalg.norm(bl.unitary(f.gates(N), N) - expm(-1j * t * H), 2)


# ------------------------------------------------------------------ the brickwall is baseline.py's formula

@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_brickwall_unitary_equals_baseline(name, order):
    model, N = get_model(name), 8
    for r in (1, 3):
        Ub = base.trotter_circuit(model, 2.0, order, r).apply(_eye(N), N, bc="open").T
        f = bl.Formula(model, 2.0, order, r, "BW10", "XEO")
        assert np.abs(bl.unitary(f.gates(N), N) - Ub).max() < 1e-13


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_brickwall_counts_and_T_equal_baseline(monkeypatch, name, order):
    monkeypatch.setattr(base, "synthesize", fake_synth)
    model, N = get_model(name), 8
    for r in (1, 3):
        circ = base.trotter_circuit(model, 2.0, order, r)
        f = bl.Formula(model, 2.0, order, r, "BW10", "XEO")
        tb, tm = base.rotation_tally(circ), bl.bulk_tally(f)["density"]
        assert set(tb) == set(tm) and all(abs(tb[k] - tm[k]) < 1e-12 for k in tb)
        for eps in (1e-3, 1e-6):
            rb, rm = base.realize_baseline(circ, eps), bl.realize(f, eps, N, fake_synth)
            for key in ("Tq", "ledger_per_qubit", "ledger2_per_qubit", "rotations_per_qubit"):
                assert abs(rb[key] - rm[key]) <= 1e-12 * max(1.0, abs(rb[key])), key
            assert rb["n_distinct"] == rm["n_distinct"]
            assert abs(base.plan_cost(circ, eps) - bl.plan_cost(f, eps)) < 1e-9
            Ub = rb["circuit"].apply(_eye(N), N, bc="open").T
            assert np.abs(bl.unitary(rm["gates"], N) - Ub).max() < 1e-13


def test_angle_keys_round_like_baseline():
    """baseline.py keys np.float64 angles with numpy's round; Python's float round differs at decimal ties."""
    th = 0.2947489932760005                      # a field angle of h_x = 1.6, order 4, t = 4, r = 9
    assert round(th, 15) != float(np.round(np.float64(th), 15))
    assert bl._key(th, False) == float(np.round(np.float64(th), 15))
    model = get_model("A")
    for t, r in ((4.0, 9), (8.0, 9)):
        tb = base.rotation_tally(base.trotter_circuit(model, t, 4, r))
        tm = bl.bulk_tally(bl.Formula(model, t, 4, r, "BW10", "XEO"))["density"]
        assert set(tb) == set(tm) and all(abs(tb[k] - tm[k]) < 1e-12 for k in tb)


_RUNS = Path(__file__).resolve().parents[1] / "runs"


def _saved_baselines(ts=(2, 4)):
    return sorted(p for tag in ("xxz_d0.25_hx0", "xxz_d0.25_hx1.6") for t in ts
                  for p in (_RUNS / tag / f"t{t}").glob("baseline_o*_r*_e*.json"))


def _replay(d):
    """synth that returns the KN records saved with a baseline.py result (no search, no cache files)."""
    recs = d["synth"]
    return lambda theta, eps: recs[f"{theta:.15f}"]


@pytest.mark.skipif(not _saved_baselines(), reason="no saved baseline.py results under runs/")
def test_saved_baseline_points_reprice_exactly():
    """Every saved baseline.py point at t = 2, 4 (runs/, read only), rebuilt as BW10/XEO and priced with its
    own KN records, has the same T per qubit, ledgers, rotations and number of distinct angles."""
    for p in _saved_baselines():
        d = json.loads(p.read_text())
        f = bl.Formula(XXZ.from_dict(d["model"]), float(d["t"]), d["order"], d["steps"], "BW10", "XEO")
        rb = bl.realize(f, d["eps"], 12, _replay(d))
        for key in ("Tq", "ledger_per_qubit", "ledger2_per_qubit", "rotations_per_qubit"):
            assert abs(rb[key] - d[key]) <= 1e-12 * max(1.0, abs(d[key])), (p.name, key)
        assert rb["n_distinct"] == d["n_distinct"], p.name


@pytest.mark.parametrize("tag,name,rel", [("xxz_d0.25_hx0", "baseline_o2_r6_e1e-03.json", 1e-9),
                                          ("xxz_d0.25_hx1.6", "baseline_o2_r4_e3e-04.json", 1e-5)])
def test_mps_at_100_sites_reproduces_saved_evaluation(tag, name, rel):
    """bl.mps_infidelity at N = 100 on a baseline.py point equals its saved evaluate.mps_infidelity.
    With a separate field the gates are fused differently (field into the bond gates), so the
    truncations at cutoff 1e-12 differ slightly: measured 3.5e-7 relative."""
    p = _RUNS / tag / "t2" / name
    ref = _RUNS / tag / "t2" / "ref"
    if not p.exists() or not ref.exists():
        pytest.skip("saved result or reference states missing")
    d = json.loads(p.read_text())
    model, t = XXZ.from_dict(d["model"]), float(d["t"])
    rb = bl.realize(bl.Formula(model, t, d["order"], d["steps"], "BW10", "XEO"), d["eps"], 100, _replay(d))
    m = bl.mps_infidelity(rb["gates"], model, t, N=100, ref_dirs=[ref], check=False, log=lambda s: None)
    assert abs(m["infid"] / d["mps"]["infid"] - 1) < rel
    assert abs(m["c_site"] / d["mps"]["c_site"] - 1) < rel


# ------------------------------------------------------------------ orders and orderings

@pytest.mark.parametrize("ordering", ["BW10", "BW01", "SEQ1", "SEQ0", "WAVE3", "WAVE3R", "WAVE4", "C:110"])
def test_sweep_has_its_pattern(ordering):
    """Def. 1 (Eq. 3): reading c off the stage's time order gives back the ordering's pattern."""
    n = 13
    pos = {k: i for i, k in enumerate(bl.sweep_order(ordering, n))}
    c = [1 if pos[k] < pos[k + 1] else 0 for k in range(n - 1)]
    assert c == bl.c_pattern(ordering, n)
    assert max(bl.lemma8_layers(c)) == 1 + max(len(list(g)) for _, g in itertools.groupby(c))   # Eq. 78


def test_paper_pattern_example():
    """Def. 1's example: n = 5, pi(1..5) = (5, 1, 2, 4, 3) has c = 1100. Bonds 0-indexed here."""
    lay = bl.lemma8_layers([1, 1, 0, 0])
    order = sorted(range(5), key=lambda k: (lay[k], k))
    pos = {k: i for i, k in enumerate(order)}
    assert [1 if pos[k] < pos[k + 1] else 0 for k in range(4)] == [1, 1, 0, 0]
    assert bl.c_pattern("C:1100", 5) == [1, 1, 0, 0]


def test_order6_step_is_symmetric_and_order1_is_first_order():
    """S^(6) is a symmetric formula (S(tau) S(-tau) = 1), and order 1 converges as tau^2 per step."""
    model, N = get_model("A"), 6
    for ordering, nesting in (("SEQ1", "XB"), ("WAVE3", "[B,X]")):
        U = bl.unitary(bl.Formula(model, 0.3, 6, 1, ordering, nesting).gates(N), N)
        V = bl.unitary(bl.Formula(model, -0.3, 6, 1, ordering, nesting).gates(N), N)
        assert np.abs(U @ V - np.eye(2 ** N)).max() < 1e-12
    H = model.sparse(N, "open").toarray()
    e = [_err(bl.Formula(model, tau, 1, 1, "SEQ1", "XB"), N, H, tau) for tau in (0.02, 0.01)]
    assert 1.9 < math.log2(e[0] / e[1]) < 2.1

@pytest.mark.parametrize("ordering,nesting,fold", [("BW10", "XEO", False), ("SEQ1", "XB", False),
                                                   ("WAVE3", "[B,X]", False), ("SEQ0", "B", True)])
def test_order6_converges(ordering, nesting, fold):
    model, N = get_model("A"), 6
    H = model.sparse(N, "open").toarray()
    e = [_err(bl.Formula(model, tau, 6, 1, ordering, nesting, fold), N, H, tau) for tau in (0.4, 0.2)]
    assert 6.7 < math.log2(e[0] / e[1]) < 7.3                    # tau^7 per step
    g = [_err(bl.Formula(model, 2.0, 6, r, ordering, nesting, fold), N, H, 2.0) for r in (4, 8)]
    assert 5.6 < math.log2(g[0] / g[1]) < 6.6                    # tau^6 at fixed t


def test_stage_coefficients():
    for p in (2, 4, 6):
        w = bl.stage_coeffs(p)
        assert len(w) == 5 ** (p // 2 - 1) and abs(sum(w) - 1) < 1e-14
    mu6 = 1 / (4 - 4 ** 0.2)
    assert abs(bl.stage_coeffs(6)[0] - mu6 / (4 - 4 ** (1 / 3))) < 1e-15


def test_lemma8_depths_and_stage_count():
    n = 11
    depth = {o: max(bl.lemma8_layers(bl.c_pattern(o, n))) for o in ("BW10", "BW01", "SEQ1", "SEQ0", "WAVE3",
                                                                    "WAVE3R", "WAVE4")}
    assert depth == {"BW10": 2, "BW01": 2, "SEQ1": n, "SEQ0": n, "WAVE3": 3, "WAVE3R": 3, "WAVE4": 4}
    assert bl.sweep_order("BW10", 6) == [0, 2, 4, 1, 3, 5]       # baseline.py's E before O
    assert bl.c_pattern("WAVE3", 9) == [0, 0, 1, 1, 0, 0, 1, 1]
    ch = bl.Chain.generic(np.eye(4), 10)                         # n = 9 terms, Sec. VI.1: 401 stages
    assert bl.count_stages(bl.product_sequence(ch, "B", "SEQ1", 6, 0.1, 1)) == 401


def _paper_err(h2, ordering, order, N=10, tau=0.1):
    ch = bl.Chain.generic(h2 / np.linalg.norm(h2, 2), N)
    seq = bl.merge_sequence(bl.product_sequence(ch, "B", ordering, order, tau, 1), ch.support)
    U = bl.unitary([ch.terms[lab].gate(d) for lab, d in seq], N)
    return np.linalg.norm(U - expm(-1j * tau * ch.hamiltonian()), 2)


@pytest.mark.parametrize("model,ratio", [("Heisenberg", 7.72), ("TFIM", 0.624)])
def test_paper_bw_seq_ratio(model, ratio):
    """Liu et al. Sec. VI.1, Fig. 7: order 4, tau = 0.1, 10 qubits, direction-averaged BW/SEQ."""
    h2 = (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)) if model == "Heisenberg" else np.kron(X, X) + np.kron(Z, I2)
    e = {o: _paper_err(h2, o, 4) for o in ("BW10", "BW01", "SEQ1", "SEQ0")}
    got = (e["BW10"] + e["BW01"]) / (e["SEQ1"] + e["SEQ0"])
    assert abs(got / ratio - 1) < 0.01


# Single-step spectral-norm errors at tau = 0.1, N = 10, order 4, from the earlier independent
# script (snapevo-scratch/scripts/ordering_check.py, ordering_check/n10.json, 13 Oct).
EARLIER_N10 = {
    ("B", False): {"BW01": 2.017868e-08, "BW10": 1.864329e-08, "SEQ1": 2.478602e-09, "SEQ0": 2.478602e-09,
                   "WAVE3": 1.011912e-08, "WAVE3R": 6.081423e-09},
    ("A", True): {"BW01": 5.642983e-07, "BW10": 5.543288e-07, "SEQ1": 4.208246e-07, "SEQ0": 4.548256e-07,
                  "WAVE3": 4.801199e-07, "WAVE3R": 4.734196e-07},
    ("A", False): {"BW01": 7.760466e-07, "BW10": 7.856024e-07, "SEQ1": 7.837954e-07, "SEQ0": 7.837954e-07,
                   "WAVE3": 7.877525e-07, "WAVE3R": 7.822831e-07},
}


@pytest.mark.parametrize("name,fold", list(EARLIER_N10))
def test_orderings_reproduce_earlier_errors(name, fold):
    model, N, tau = get_model(name), 10, 0.1
    H = model.sparse(N, "open").toarray()
    for ordering, ref in EARLIER_N10[(name, fold)].items():
        e = _err(bl.Formula(model, tau, 4, 1, ordering, "XB", fold), N, H, tau)
        assert abs(e / ref - 1) < 0.01, ordering


# ------------------------------------------------------------------ terms, decompositions, pricing

@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("fold", [False, True])
def test_chain_terms_sum_to_model(name, fold):
    model = get_model(name)
    ch = bl.Chain.xxz(model, 6, fold)
    assert np.abs(ch.hamiltonian() - model.sparse(6, "open").toarray()).max() < 1e-14


def _phase_dist(A, B):
    ph = np.vdot(A.ravel(), B.ravel())
    return np.abs(A * ph / abs(ph) - B).max()


def test_kak_and_euler_reconstruct():
    rng = np.random.default_rng(7)
    for _ in range(20):
        U = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
        A1, B1, abc, A2, B2 = bl.kak(U)
        R = np.kron(A1, B1) @ CELLS["xxz+x"].two_qubit_part(abc) @ np.kron(A2, B2)
        assert _phase_dist(R, U) < 1e-12
        V = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        a, b, c = bl.zxz_angles(V)
        assert _phase_dist(bl._rz(a) @ bl._rx(b) @ bl._rz(c), V) < 1e-12
    assert bl.zxz_angles(bl._rz(0.3))[1:] == (0.0, 0.0)            # diagonal: one rotation
    U = expm(-0.37j * (np.kron(X, X) + 0.5 * np.kron(Z, Z)))       # degenerate spectrum
    A1, B1, abc, A2, B2 = bl.kak(U)
    assert _phase_dist(np.kron(A1, B1) @ CELLS["xxz+x"].two_qubit_part(abc) @ np.kron(A2, B2), U) < 1e-12


@pytest.mark.parametrize("ordering", ["BW10", "SEQ1", "WAVE3"])
def test_folded_compiles_exactly(ordering):
    model, N = get_model("A"), 8
    f = bl.Formula(model, 1.0, 4, 2, ordering, fold=True)
    assert _phase_dist(bl.unitary(f.compiled(N), N), bl.unitary(f.gates(N), N)) < 1e-12
    assert all(g.kind in ("xxz", "zxz") for g in f.compiled(N))


@pytest.mark.parametrize("ordering,nesting,fold", [("SEQ1", "XB", False), ("WAVE3", "[B,X]", False),
                                                   ("WAVE3", "B", True), ("BW10", "[XE,O]", False)])
def test_bulk_density_does_not_depend_on_sizes(ordering, nesting, fold):
    f = bl.Formula(get_model("A"), 2.0, 4, 2, ordering, nesting, fold)
    a = bl.bulk_tally(f)
    b = bl.bulk_tally(f, N_min=a["sizes"][1])
    assert a["sizes"] != b["sizes"]
    assert set(a["density"]) == set(b["density"])
    assert all(abs(a["density"][k] - b["density"][k]) < 1e-12 for k in a["density"])
    assert abs(a["rotations"] - b["rotations"]) < 1e-12


def test_sequential_needs_about_twice_the_rotations():
    model = get_model("B")
    bw = bl.bulk_tally(bl.Formula(model, 2.0, 4, 8, "BW10"))["rotations"]
    seq = bl.bulk_tally(bl.Formula(model, 2.0, 4, 8, "SEQ1"))["rotations"]
    assert 1.9 < seq / bw < 2.1


def test_nestings_parse_and_differ():
    model, N = get_model("A"), 6
    us = [bl.unitary(bl.Formula(model, 0.5, 2, 1, "BW10", n).gates(N), N) for n in bl.BW_NESTINGS]
    for i in range(len(us)):
        for j in range(i):
            assert np.abs(us[i] - us[j]).max() > 1e-6
    with pytest.raises(ValueError):
        bl.parse_nesting("[XE,O")
    with pytest.raises(ValueError):
        bl.Formula(model, 0.5, 1, 1, "BW10", "[XE,O]").gates(N)  # order 1 needs a flat sweep


# ------------------------------------------------------------------ MPS and the scan

@pytest.mark.parametrize("ordering,nesting,fold,order", [("BW10", "XEO", False, 4), ("SEQ1", "[B,X]", False, 2),
                                                         ("WAVE3", "B", True, 2)])
def test_mps_apply_equals_dense(ordering, nesting, fold, order):
    model, N = get_model("A"), 10
    f = bl.Formula(model, 1.0, order, 2, ordering, nesting, fold)
    gates = bl.realize(f, 1e-3, N, fake_synth)["gates"]
    vec = product_vectors(N, 1, 11)[0]
    psi = bl.apply_mps(MPS.product(vec, chi_max=64, cutoff=0.0), gates)     # untruncated at N = 10
    dense = bl.apply_dense(gates, random_product_states(N, 1, 11), N)[0]
    assert psi.discarded == 0.0
    assert np.abs(psi.to_dense() - dense).max() < 1e-12


def test_scan_family_smoke():
    model = get_model("A")
    pts = bl.scan_family(model, 0.5, "SEQ1", 2, "XB", N=6, states=(5, 4), c_floor=1e-6, synth=fake_synth,
                         log=lambda s: None)
    assert pts and {"C", "C_trotter", "C_bound", "Tq", "steps", "eps"} <= set(pts[0])
    front = bl.pareto(pts)
    assert all(a["Tq"] <= b["Tq"] and a["C"] > b["C"] for a, b in zip(front, front[1:]))


def test_default_synth_refuses_runs_cache(monkeypatch):
    monkeypatch.delenv("SNAPEVO_RUNS", raising=False)
    with pytest.raises(RuntimeError):
        bl.realize(bl.Formula(get_model("B"), 1.0, 2, 1, "SEQ1"), 1e-3, 6)


def test_refine_family_resolves_targets():
    model = get_model("B")
    kw = dict(N=6, states=(5, 4), synth=fake_synth, log=lambda s: None)
    pts = bl.scan_family(model, 0.5, "BW10", 4, c_floor=1e-9, **kw)
    new = bl.refine_family(model, 0.5, "BW10", 4, pts, (1e-4,), **kw)
    assert all(p["refine"] and p["eps"] in bl.EPS_GRID for p in new)
    allp = pts + new
    best = min((p for p in allp if p["C"] <= 1e-4), key=lambda p: p["Tq"])
    looser = [e for e in bl.EPS_GRID if e > best["eps"]]
    if looser:                          # the next looser budget at that r was tried and fails the target
        nxt = [p for p in allp if p["steps"] == best["steps"] and p["eps"] == min(looser)]
        assert nxt and nxt[0]["C"] > 1e-4


def test_scan_family_resumes():
    model = get_model("A")
    kw = dict(N=6, states=(5, 4), synth=fake_synth, log=lambda s: None, c_floor=1e-7)
    full = bl.scan_family(model, 0.5, "SEQ1", 2, "XB", **kw)
    steps = []
    part = bl.scan_family(model, 0.5, "SEQ1", 2, "XB", r_max=3, on_step=lambda p: steps.append(len(p)), **kw)
    assert len(steps) == 3
    res = bl.scan_family(model, 0.5, "SEQ1", 2, "XB", done=part, **kw)
    assert [(p["steps"], p["eps"], p["C"]) for p in res] == [(p["steps"], p["eps"], p["C"]) for p in full]


def test_small_derived_angles_snap_or_split():
    """Folded circuits have tiny Euler angles: snapped within eps, split into two generic rotations above it."""
    model, N = get_model("A"), 8
    f = bl.Formula(model, 2.0, 2, 32, "SEQ1", fold=True)
    calls = []

    def spy(theta, eps):
        calls.append(theta)
        return fake_synth(theta, eps)

    loose, tight = bl.realize(f, 1e-3, N, spy), bl.realize(f, 1e-8, N, spy)
    assert loose["n_snapped"] > tight["n_snapped"] and tight["n_split"] > 0
    q = math.pi / 4
    assert all(abs(math.remainder(th, q)) >= bl.SMALL_OFFSET or abs(abs(th) - bl.SPLIT_ANGLE) < bl.SMALL_OFFSET
               for th in calls)
    for rb in (loose, tight):                    # the realized circuit stays close to the exact one
        U = bl.unitary(f.compiled(N), N)
        V = bl.unitary(rb["gates"], N)
        assert abs(abs(np.trace(U.conj().T @ V)) / 2 ** N - 1) < 1e-2


def test_default_families_match_the_screen():
    """The screened families: 12 at h_x = 0, 67 at h_x = 1.6 (litbase/scripts/launch.py), all buildable."""
    for name, n in (("B", 12), ("A", 67)):
        model = get_model(name)
        fams = bl.default_families(model)
        assert len(fams) == n
        labels = {bl.Formula(model, 1.0, f["order"], 1, f["ordering"], f.get("nesting", "XB"),
                             f.get("fold", False)).family for f in fams}
        assert len(labels) == n
        for f in fams[:: max(1, n // 8)]:
            bl.Formula(model, 1.0, f["order"], 1, f["ordering"], f.get("nesting", "XB"), f.get("fold", False)).gates(6)


def test_scan_returns_fronts():
    model = get_model("B")
    res = bl.scan(model, 0.5, [dict(ordering="BW10", order=2), dict(ordering="SEQ1", order=2)], N=6,
                  states=(5, 4), c_floor=1e-6, synth=fake_synth, log=lambda s: None)
    assert {p["family"] for p in res["points"]} == {"BW10/o2", "SEQ1/o2"}
    for key, front in (("C", res["front"]), ("C_bound", res["front_bound"])):
        assert front and all(a["Tq"] <= b["Tq"] and a[key] > b[key] for a, b in zip(front, front[1:]))
        assert all(not (q["Tq"] <= p["Tq"] and q[key] < p[key]) for p in front for q in res["points"] if q is not p)
