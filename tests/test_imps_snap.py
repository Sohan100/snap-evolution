"""SNAP on the infinite chain (snapevo.imps_snap): the eigenvalue curves against the exact cost, and a small
compile, realize and evaluate end to end.

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0.
"""

import json

import numpy as np
import pytest

from snapevo import imps_snap as isn
from snapevo import imps_train as it
from snapevo.hamiltonian import XXZ


@pytest.mark.parametrize("hx", (1.6, 0.0))
def test_curves_match_the_exact_cost(hx):
    model = XXZ(0.25, hx)
    t, D, budget = 0.5, 3, 1e-2
    with isn.TIProblem(model, t, D, n_train=2, n_test=0, workers=2, budget=budget) as P:
        na = P.cell.n_angles
        z = P.cell.trotter_angles(model, t, D) + 0.02 * np.random.default_rng(3).normal(size=na * D)
        idx = [0, na + 1, 2 * na + na - 1]                          # one slot in each sublayer, the last slot too
        curves = P.curves(z, idx)
        assert sorted(curves) == sorted(idx)
        c0 = P.c_fn(z)
        rng = np.random.default_rng(5)
        for i, cv in curves.items():
            assert cv.residual < 1e-12, (i, cv.residual)
            # the curve is exact along its own route (environments meeting at sublayer i // na); the cost function
            # runs forward through every sublayer, which truncates differently, by ~1e-8 of c
            L = P._lambdas(z, P._want({i: [z[i]]}))[(i // na, i % na)]
            assert abs(float(cv.c_at(z[i])) - float(isn._cost(L)[0])) < 1e-13
            assert abs(float(cv.c_at(z[i])) - c0) < 1e-7 * c0
            a, b, _ = cv.windows[0]
            for th in list(rng.uniform(a, b, 4)) + [0.5 * (a + b) + 2 * np.pi]:    # inside, and a period away
                zz = z.copy()
                zz[i] = th
                assert abs(float(cv.c_at(th)) - P.c_fn(zz)) < 1e-7 * P.c_fn(zz), (i, th)
            for th in rng.uniform(z[i] + 0.5, z[i] + 2 * np.pi - 0.5, 4):   # well outside: never below the cap
                zz = z.copy()
                zz[i] = th
                if all(not (wa <= cv._reduce(th) <= wb) for wa, wb, _ in cv.windows):
                    assert cv.c_at(th) >= P.cap and P.c_fn(zz) >= 0.5 * P.cap
            h = 1e-3                                             # the curvature against an exact difference
            zp, zm = z.copy(), z.copy()
            zp[i] += h
            zm[i] -= h
            exact = (P.c_fn(zp) + P.c_fn(zm) - 2 * c0) / h ** 2
            assert abs(cv.curvature() - exact) < 1e-3 * abs(exact) + 1e-6, (i, cv.curvature(), exact)
            S = np.linspace(0, 2 * np.pi, 20001)
            m = cv.admissible(0.5 * (c0 + P.cap), S)
            assert m.any() and np.all(np.abs(np.asarray(cv._reduce(S[m])) - z[i]) <= b - z[i] + 1e-12)
            # a threshold above the cap extends the curve, which stays exact where it is used
            n0 = len(cv.windows)
            m = cv.admissible(0.2, S)
            assert cv.cap >= 0.2 and len(cv.windows) >= n0
            for th in rng.choice(S[m], 4, replace=False):
                zz = z.copy()
                zz[i] = th
                assert abs(float(cv.c_at(th)) - P.c_fn(zz)) < P.tol + 1e-7 * P.c_fn(zz), (i, th)   # its tolerance
            cv.extend(np.inf)                                    # the lazy angles' case: exact everywhere
            for th in rng.uniform(0, 2 * np.pi, 4):
                zz = z.copy()
                zz[i] = th
                assert abs(float(cv.c_at(th)) - P.c_fn(zz)) < P.tol + 1e-7 * P.c_fn(zz), (i, th)   # its tolerance


def test_a_small_compile_realizes_and_evaluates_on_the_infinite_chain(tmp_path):
    model = XXZ(0.25, 1.6)
    t = 0.5
    it.train_imps_ladder(model, t, [2, 3, 5], runs=tmp_path, workers=2, n_train=2, n_test=4, stall=None,
                         target=None, log=lambda s: None)
    out = isn.ti_point(model, t, 1e-2, tmp_path, n_train=2, workers=2, margins=(3.0,),
                       settings_kw={"reopt": "newton"}, log=lambda s: None)
    assert len(out) == 1
    ev = out[0]
    d = tmp_path / model.tag / "t0.5" / "imps_snap_n2"
    real = json.loads(next(d.glob("real_*.json")).read_text())
    snap = json.loads(next(d.glob("snap_*.json")).read_text())
    assert snap["bc"] == "infinite" and snap["curve_max_residual"] <= snap["curve_tol"]
    assert real["C_train_real"] <= 1e-2 * (1 + 1e-9)
    assert ev["c"] > 0 and ev["Tq"] > 0 and ev["T_edges_per_qubit"] == 0.0 and not ev["over_budget"]
    for v in snap["stragglers"].values():                     # a sane precision for every unsnapped angle
        assert 0 < v["eps"] < 1
    assert ev["T_per_rotation"] == pytest.approx(ev["Tq"] / ev["rotations_per_qubit"])


def test_a_resumed_compile_reuses_its_saved_curves_and_ends_the_same(tmp_path):
    import shutil
    import time
    from snapevo.chain_snap import DeadlineReached
    model = XXZ(0.25, 1.6)
    t = 0.5
    a, b = tmp_path / "a", tmp_path / "b"
    it.train_imps_ladder(model, t, [2, 3, 5], runs=a, workers=2, n_train=2, n_test=4, stall=None, target=None,
                         log=lambda s: None)
    shutil.copytree(a, b)
    kw = dict(n_train=2, workers=2, margins=(3.0,), settings_kw={"reopt": "newton"})
    whole = isn.ti_point(model, t, 1e-2, a, log=lambda s: None, **kw)[0]
    d = b / model.tag / "t0.5" / "imps_snap_n2"
    for wait in (2, 4, 8, 16, 32, 64):                        # stop it once a checkpoint and its curves are on disk
        try:
            isn.ti_point(model, t, 1e-2, b, deadline=time.time() + wait, log=lambda s: None, **kw)
        except DeadlineReached:
            if list(d.glob("*.ckpt.pkl.curves")):
                break
    else:
        pytest.skip("the compile finished before it could be stopped with its curves saved")
    lines = []
    resumed = isn.ti_point(model, t, 1e-2, b, log=lines.append, **kw)[0]
    assert any("reused" in s and "saved curves" in s for s in lines)
    assert not list(d.glob("*.ckpt.pkl*"))                   # the finished point leaves no checkpoint files
    sa = json.loads(next((a / model.tag / "t0.5" / "imps_snap_n2").glob("snap_*.json")).read_text())
    sb = json.loads(next(d.glob("snap_*.json")).read_text())
    assert sa["snapped"] == sb["snapped"]
    assert np.allclose(sa["angles"], sb["angles"], atol=1e-8)
    assert resumed["Tq"] == pytest.approx(whole["Tq"]) and resumed["c"] == pytest.approx(whole["c"], rel=1e-6)


def test_fallback_defects_match_a_ring_with_explicit_fallbacks():
    """One fallback at one bond: the infinite chain's rho against a 12-site ring with exact state vectors; and the
    expected change of the log-amplitude per cell against random draws on that ring."""
    import math
    from functools import reduce

    from snapevo import fallbacks as fb
    from snapevo import imps as im
    from snapevo.cells import cell_for
    from snapevo.circuits import Layered, apply_2q
    from snapevo.hamiltonian import evolve
    from snapevo.synthesis import synthesize

    model = XXZ(0.25, 1.6)
    cell = cell_for(model)
    t, D, N, na = 0.5, 3, 12, cell.n_angles
    z = cell.trotter_angles(model, t, D) + 0.03 * np.random.default_rng(1).normal(size=na * D)
    recs = {i: synthesize(float(z[i]), 3e-2, unitaries=True) for i in range(z.size)}
    lay = Layered.brickwork(cell, np.array([recs[i]["theta_s"] for i in range(z.size)])).layers
    fl = isn.fallback_layers(cell.name, lay, lambda l, s: recs.get(l * na + s))
    v = np.array([[0.8, 0.6j], [0.6, -0.8]], dtype=complex)
    phi = im.IMPS.product(v, chi_max=it.CHI_MAX, cutoff=it.CUTOFF)
    psi, _ = im.reference_state(model, t, phi, it.DT, it.CHI_MAX, it.CUTOFF)
    st = reduce(np.kron, [v[i % 2] for i in range(N)])[None, :]
    ref = evolve(model.sparse(N, "ring"), st, t)

    def ring(gl):
        s = st
        for par, bonds in gl:
            for i in range(par, N, 2):
                s = apply_2q(s, bonds[i], i, (i + 1) % N, N)
        return complex(np.vdot(ref[0], s[0]))

    base = [(par, {i: U for i in range(par, N, 2)}) for par, U, _ in fl]
    ov0 = ring(base)
    n_checked = 0
    for l, (par, U, inst) in enumerate(fl):
        for kind, q, variants in inst[:2]:
            one = [(p2, U2, [(kind, 1.0, variants)] if l2 == l else []) for l2, (p2, U2, _) in enumerate(fl)]
            rho_inf = math.exp(isn.fallback_terms(one, phi, psi)["kn"])
            gl = [(p2, dict(b)) for p2, b in base]
            gl[l][1][par] = variants[0]
            assert abs(rho_inf - abs(ring(gl) / ov0)) < 1e-7
            n_checked += 1
    assert n_checked >= 4
    pred = isn.fallback_terms(fl, phi, psi)["kn"]
    rng = np.random.default_rng(7)
    d = []
    for _ in range(300):
        gl = []
        for l, (par, U, inst) in enumerate(fl):
            r = fb._layer_rots(cell.name, np.asarray(lay[l][1]), lambda s, l=l: recs.get(l * na + s))
            bonds = {}
            for i in range(par, N, 2):
                over = {}
                for name in fb._ORDER[cell.name]:
                    u = r[name].draw(rng)
                    if u is not None:
                        over[name] = u
                bonds[i] = fb._gate(cell.name, r, over)
            gl.append((par, bonds))
        d.append(math.log(abs(ring(gl) / ov0)) / (N / 2))
    d = np.array(d)
    assert abs(d.mean() - pred) < 3 * d.std(ddof=1) / math.sqrt(d.size) + 0.1 * abs(pred)
