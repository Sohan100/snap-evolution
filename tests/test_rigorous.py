"""The ordering-aware Trotter bound of Liu et al. (arXiv:2610.03672) and the rigorous T estimate (rigorous.py)."""

import math
import os

import numpy as np
import pytest
import scipy.linalg as sl

from snapevo import rigorous as rg
from snapevo.baseline import trotter_circuit
from snapevo.circuits import random_product_states
from snapevo.hamiltonian import get_model

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2)
SLOW = bool(os.environ.get("SNAPEVO_SLOW"))


def test_suzuki_coefficients():
    for p, m in ((2, 1), (4, 5), (6, 25)):
        c = rg.suzuki_coefficients(p)
        assert len(c) == m and math.isclose(sum(c), 1.0, abs_tol=1e-14)


def test_orderings_follow_definition_1():
    for name in rg.ORDERINGS:
        c = rg.ordering_bits(name, 9)
        pi = rg.sweep(c)
        pos = {j: i for i, j in enumerate(pi)}
        assert sorted(pi) == list(range(1, 10))
        assert all(c[k - 1] == int(pos[k] < pos[k + 1]) for k in range(1, 9))


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("order", [2, 4])
def test_layer_formula_is_baseline_circuit(name, order):
    """field="layer", ordering "bw" is baseline.trotter_circuit on the open chain, gate for gate."""
    model, N, t, r = get_model(name), 8, 0.7, 3
    ch = rg.xxz_chain(model, N)
    psi = random_product_states(N, 3, 5)
    U = rg.formula_unitary(ch, rg.build_formula(ch, order, "bw", "layer", steps=r), t / r)
    out = trotter_circuit(model, t, order, r).apply(psi, N, bc="open")
    assert np.abs(out - psi @ U.T).max() < 1e-12


def test_chain_is_the_open_hamiltonian():
    for name in "AB":
        model = get_model(name)
        ch = rg.xxz_chain(model, 6)
        assert np.abs(rg.dense_hamiltonian(ch) - model.sparse(6, "open").toarray()).max() < 1e-14


# ------------------------------------------------------------------ an independent brute-force reference

def _full(op4, j, N):
    """A 4x4 operator on qubits (j, j+1) of an N-qubit chain, labels from 1."""
    return np.kron(np.kron(np.eye(2 ** (j - 1)), op4), np.eye(2 ** (N - j - 1)))


def _reference(chain, formula, tau, q=2, v=0, nodes=48):
    """Eq. 68 with full-chain matrices and no slicing tricks, plus the true quantities it bounds.

    Returns the windows' parts and the quantities each inequality of the chain bounds:
    ||F_kq - F^[L]_kq|| (Eq. 63), ||C_kq - C^[L]_kq|| (Eq. 67), ||C_kq - F_kq|| (Eq. 62) and
    ||U_k - S_k|| - ||U_{k-q} - S_{k-q}|| (Eq. 60). beta_C includes the field gates of label k-L.
    """
    N, n, L = chain.N, chain.n, formula.order + q + v
    H = [None] + [_full(chain.term(j), j, N) for j in range(1, n + 1)]
    Hs = lambda a, b: sum((H[j] for j in range(a, b + 1)), np.zeros((2 ** N, 2 ** N), complex))   # noqa: E731
    gates = []                                             # fields are 4x4 pieces too (h_x/2 X x I or I x X)
    for j, name, a in formula.gates:
        gates.append((j, _full(sl.expm(-1j * a * tau * chain.pieces[j - 1][name]), j, N)))
    S = lambda lo, hi: _prod([G for j, G in gates if lo < j <= hi], N)                            # noqa: E731
    U = lambda k: sl.expm(-1j * tau * Hs(1, k))                                                   # noqa: E731
    k0 = n if n <= L else next(k for k in range(L - q + 1, L + 1) if (n - k) % q == 0)
    head = np.linalg.norm(U(k0) - S(0, k0), 2)
    out = []
    for k in range(k0 + q, n + 1, q):
        lo = k - L
        A, B = Hs(lo + 1, k - q), Hs(k - q + 1, k)
        FL = sl.expm(1j * tau * A) @ sl.expm(-1j * tau * (A + B))
        CL = S(lo, k - q).conj().T @ S(lo, k)
        F = U(k - q).conj().T @ U(k)
        C = S(0, k - q).conj().T @ S(0, k)
        W, beta_C, field_comm = np.eye(2 ** N), 0.0, 0.0
        for (j, G), (_, name, _a) in zip(gates, formula.gates):
            if j > k:
                continue
            if j > k - q:
                W = W @ G
            elif j > lo:
                W = G.conj().T @ W @ G
            elif j == lo:
                c = np.linalg.norm(W @ G - G @ W, 2)
                if chain.support(j, name) == (j,):
                    field_comm = max(field_comm, c)
                beta_C += c
        x, w = np.polynomial.legendre.leggauss(nodes)
        s, w = (x + 1) * tau / 2, w * tau / 2
        beta_F = sum(wi * (tau - si) * np.linalg.norm(
            H[lo] @ (Xs := sl.expm(1j * si * A) @ B @ sl.expm(-1j * si * A)) - Xs @ H[lo], 2) for si, wi in zip(s, w))
        out.append({"k": k, "local": np.linalg.norm(CL - FL, 2), "beta_C": beta_C, "beta_F": beta_F,
                    "field_comm": field_comm, "F_err": np.linalg.norm(F - FL, 2), "C_err": np.linalg.norm(C - CL, 2),
                    "inc": np.linalg.norm(C - F, 2),
                    "step": np.linalg.norm(U(k) - S(0, k), 2) - np.linalg.norm(U(k - q) - S(0, k - q), 2)})
    return head, k0, out


def _prod(Gs, N):
    P = np.eye(2 ** N, dtype=complex)
    for G in Gs:
        P = P @ G
    return P


@pytest.mark.parametrize("name,order,ordering,field,N,tau,q,v", [
    ("A", 2, "bw", "layer", 10, 0.3, 2, 0),
    ("A", 2, "seq", "layer", 10, 0.5, 2, 1),
    ("B", 2, "wave", "layer", 10, 0.4, 1, 0),
    ("A", 4, "bw", "layer", 8, 0.5, 2, 0),
    ("A", 4, "seq_b", "layer", 9, 0.4, 2, 0),
    ("A", 4, "bw_eo", "fold", 9, 0.5, 2, 0),
    ("B", 4, "seq", "layer", 9, 0.6, 1, 0),
])
def test_against_brute_force_reference(name, order, ordering, field, N, tau, q, v):
    """Each part of Eq. 68 matches a full-chain computation, and each inequality of its derivation holds."""
    ch = rg.xxz_chain(get_model(name), N)
    f = rg.build_formula(ch, order, ordering, field)
    b = rg.step_bound(ch, f, tau, q=q, v=v, check=True)
    head, k0, ref = _reference(ch, f, tau, q, v)
    assert b.k0 == k0 and len(b.windows) == len(ref) >= 1
    assert math.isclose(b.exact_head, head, rel_tol=1e-9, abs_tol=1e-15)
    for w, r in zip(b.windows, ref):
        assert w["k"] == r["k"]
        assert math.isclose(w["local"], r["local"], rel_tol=1e-8)
        assert math.isclose(w["beta_C"], r["beta_C"], rel_tol=1e-6, abs_tol=1e-6 * w["local"])
        assert r["field_comm"] < 1e-13                         # field gates of label k-L commute (module docs)
        assert math.isclose(w["beta_F_quad"], r["beta_F"], rel_tol=1e-6)
        assert r["beta_F"] <= w["beta_F"] <= 1.35 * r["beta_F"]   # the rigorous beta_F, and its slack
        assert r["F_err"] <= w["beta_F"] * (1 + 1e-9)             # Eq. 63
        assert r["C_err"] <= w["beta_C"] * (1 + 1e-9) + 1e-14     # Eq. 67
        assert r["inc"] <= (w["local"] + w["beta_F"] + w["beta_C"]) * (1 + 1e-9)   # Eq. 62
        assert r["step"] <= r["inc"] + 1e-14                      # Eq. 60


@pytest.mark.parametrize("name", ["A", "B"])
@pytest.mark.parametrize("N", [6, 8, 10])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("tau", [0.1, 0.5])
def test_bound_is_valid_and_within_2x_for_the_baseline_formula(name, N, order, tau):
    """The (2, 0) bound on baseline.py's formula. At N = 6 and order 4 the chain fits in one slice (exact)."""
    ch = rg.xxz_chain(get_model(name), N)
    f = rg.build_formula(ch, order, "bw", "layer")
    b = rg.step_bound(ch, f, tau)
    e = rg.exact_step_error(ch, f, tau)
    assert e <= b.total * (1 + 1e-9)
    assert b.total <= 2.0 * e


@pytest.mark.parametrize("name,ordering,field", [("A", "seq", "layer"), ("B", "seq", "layer"),
                                                 ("A", "bw", "fold"), ("A", "wave", "layer"),
                                                 ("B", "wave_1", "layer"), ("A", "bw_eo", "layer")])
@pytest.mark.parametrize("N", [8, 10])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("tau", [0.1, 0.5])
def test_bound_is_valid_for_other_orderings_and_the_folded_field(name, ordering, field, N, order, tau):
    """Valid everywhere. Sequential at h_x = 0 is the loose case (2.2x to 2.8x at N = 10, order 4)."""
    ch = rg.xxz_chain(get_model(name), N)
    f = rg.build_formula(ch, order, ordering, field)
    b = rg.step_bound(ch, f, tau)
    e = rg.exact_step_error(ch, f, tau)
    assert e <= b.total * (1 + 1e-9)
    assert b.total <= 3.0 * e


def test_v_and_q_variants_are_valid():
    ch = rg.xxz_chain(get_model("B"), 10)
    f = rg.build_formula(ch, 2, "seq", "layer")
    e = rg.exact_step_error(ch, f, 0.3)
    for q, v in ((1, 0), (2, 0), (2, 1), (2, 2), (3, 0)):
        assert e <= rg.step_bound(ch, f, 0.3, q=q, v=v).total * (1 + 1e-9)


def test_beta_F_quadrature_converged_and_rigorous_value_above_it():
    ch = rg.xxz_chain(get_model("A"), 12)
    for order, tau in ((4, 0.5), (4, 1.2), (2, 0.05)):
        f = rg.build_formula(ch, order, "seq", "layer")
        b = rg.step_bound(ch, f, tau, check=True)
        for w in b.windows:
            assert abs(w["beta_F_quad"] - w["beta_F_quad_fine"]) <= 1e-5 * w["beta_F_quad_fine"]
            assert w["beta_F_quad_fine"] <= w["beta_F"] <= 1.35 * w["beta_F_quad_fine"]


def _paper_ratio(h, order, beta_f):
    """BW/SEQ ratios at n = 10 qubits, tau = 0.1, each family averaged over its two variants (Sec. VI.1)."""
    ch = rg.ti_chain(h / np.linalg.norm(h, 2), 10)
    res = {}
    for o in ("bw", "bw_eo", "seq", "seq_b"):
        f = rg.build_formula(ch, order, o, "fold")
        res[o] = (rg.step_bound(ch, f, 0.1, beta_f=beta_f).total, rg.exact_step_error(ch, f, 0.1))
    bw = [(res["bw"][i] + res["bw_eo"][i]) / 2 for i in (0, 1)]
    sq = [(res["seq"][i] + res["seq_b"][i]) / 2 for i in (0, 1)]
    return bw[1] / sq[1], bw[0] / sq[0], res


@pytest.mark.parametrize("beta_f", ["quad", "taylor"])
def test_reproduces_paper_fig7_fourth_order(beta_f):
    """Sec. VI.1: at fourth order the actual and q = 2 predicted BW/SEQ ratios are 7.72 and 5.44
    (Heisenberg) and 0.624 and 0.636 (TFIM), normalized terms, 10 qubits, tau = 0.1."""
    heis = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)
    actual, bound, res = _paper_ratio(heis, 4, beta_f)
    assert abs(actual - 7.72) < 0.01
    assert abs(bound - 5.44) / 5.44 < 0.01
    assert all(e <= b <= 2.1 * e for b, e in res.values())
    tfim = np.kron(X, X) + np.kron(Z, I2)                 # field on the left site (Table 1)
    actual, bound, res = _paper_ratio(tfim, 4, beta_f)
    assert abs(actual - 0.624) < 0.001
    assert abs(bound - 0.636) < 0.002
    assert all(e <= b <= 1.2 * e for b, e in res.values())


@pytest.mark.skipif(not SLOW, reason="about 4 min; set SNAPEVO_SLOW=1")
@pytest.mark.parametrize("ordering,r", [("bw_eo", 60), ("seq", 49)])
def test_reproduces_paper_fig10_sixth_order_step_counts(ordering, r):
    """Sec. VI.1 / Fig. 10: q = 2 (v = 0) step counts 60 (even-odd BW) and 49 (forward SEQ) for the
    unnormalized Heisenberg chain on 12 qubits, sixth order, t = n + 1 = 12, target 1e-3."""
    ch = rg.ti_chain(np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z), 12)
    f = rg.build_formula(ch, 6, ordering, "fold")
    bound = lambda k: k * rg.step_bound(ch, f, 12.0 / k).total          # noqa: E731
    assert bound(r) <= 1e-3 < bound(r - 1)


def test_steps_needed_brackets_the_target():
    ch = rg.xxz_chain(get_model("A"), 8)
    f = rg.build_formula(ch, 2, "bw", "layer")
    t, dT = 2.0, 2e-3

    def bound(r):
        return r * rg.step_bound(ch, f, t / r).total

    r = rg.steps_needed(bound, dT)
    assert r > 1 and bound(r) <= dT < bound(r - 1)
    # the multi-step bound holds for the r-step circuit
    U = np.eye(2 ** 8) + rg._expm1_herm(rg.dense_hamiltonian(ch), t)
    V = rg.formula_unitary(ch, rg.build_formula(ch, 2, "bw", "layer", steps=r), t / r)
    assert np.linalg.norm(U - V, 2) <= bound(r)
    assert rg.steps_needed(lambda k: 1.0 / k, 0.25) == 4 and rg.steps_needed(lambda k: 0.1, 0.2) == 1


def test_delta_conversion():
    cs = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    for N in (50, 100):
        d = [rg.delta_for_c(c, N) for c in cs]
        assert all(a < b for a, b in zip(d, d[1:]))
        for c, x in zip(cs, d):
            assert math.isclose(rg.c_for_delta(x, N), c, rel_tol=1e-9)
            assert math.isclose(x, math.sqrt(N * c), rel_tol=0.3)
    assert rg.delta_for_c(1e-6, 100) > rg.delta_for_c(1e-6, 50)


def test_fidelity_relation_on_a_tiny_chain():
    """1 - F <= delta^2 for every input, against the exact worst case over all states.

    For unitaries U, V with eigenphase spread w of U^dag V (w < pi), the phase-optimized operator distance is
    delta = 2 sin(w/4) and the worst-case infidelity over all states is sin^2(w/2) = delta^2 cos^2(w/4).
    """
    N = 4
    ch = rg.xxz_chain(get_model("A"), N)
    for tau in (0.2, 0.6):
        U = np.eye(2 ** N) + rg._expm1_herm(rg.dense_hamiltonian(ch), tau)
        f = rg.build_formula(ch, 2, "bw", "layer")
        V = rg.formula_unitary(ch, f, tau)
        ph = np.angle(np.linalg.eigvals(U.conj().T @ V))
        ph = np.sort(np.mod(ph - ph[0] + np.pi, 2 * np.pi))
        gaps = np.diff(np.concatenate([ph, [ph[0] + 2 * np.pi]]))
        w = 2 * np.pi - gaps.max()                         # smallest arc holding every eigenphase
        delta = 2 * math.sin(w / 4)
        worst = math.sin(w / 2) ** 2
        b = rg.step_bound(ch, f, tau).total
        assert delta <= np.linalg.norm(U - V, 2) + 1e-12 <= b + 1e-12
        assert worst <= delta ** 2 and worst >= delta ** 2 * math.cos(w / 4) ** 2 - 1e-12
        psi = random_product_states(N, 50, 3)
        F = np.abs(np.sum(np.conj(psi @ U.T) * (psi @ V.T), axis=1)) ** 2
        assert (1 - F).max() <= worst + 1e-12
        c_site = 1 - np.mean(F ** (1 / N))
        assert c_site <= rg.c_for_delta(delta, N) + 1e-12 <= rg.c_for_delta(b, N) + 1e-12


def test_rotation_count_matches_baseline_layers():
    for name in "AB":
        model = get_model(name)
        for order in (2, 4):
            N, t, r = 10, 1.3, 4
            got = rg.rotation_count(model, N, order, "bw", t, r)
            want = 0
            for parity, angles in trotter_circuit(model, t, order, r).layers:
                if angles[3]:
                    want += N                                   # a field layer: one X rotation per site
                else:
                    want += 3 * (N // 2 - parity)               # open chain: no wrap bond
            assert got["R"] == want and got["exact"] == 0
    # sequential needs about twice the bond rotations of brickwall without a field (2n - 2 vs n per stage)
    m = get_model("B")
    bw, sq = (rg.rotation_count(m, 50, 4, o, 1.3, 10)["per_step"] for o in ("bw", "seq"))
    assert 1.9 < sq / bw < 2.0


def test_condition_c_and_cost():
    # eps = delta_S / R = 1e-6: D0 <= 0.9 eps, D1 <= 0.1 eps / (1 - p)
    recs = [{"p": 0.99, "d0": 0.8e-6, "d1": 1e-5}, {"p": 1.0, "d0": 0.5e-6, "d1": 0.0}]
    out = rg.condition_c(recs, 2e-3, counts=[1000, 1000])
    assert out["ok"] and out["mu"] <= 0.521 * 2e-3
    rare = [{"p": 1 - 1e-7, "d0": 0.5e-6, "d1": 0.1e-6 / 1e-7}]          # a rare, very wrong fallback
    assert not rg.condition_c(rare, 2e-3, counts=[2000])["ok"]
    assert rg.condition_c(rare, 2e-1, counts=[200000])["ok"]            # fine once R is large
    assert rg.t_per_qubit(0, 1e-3, 10) == 0.0
    from snapevo.synthesis import kn_cost
    assert math.isclose(rg.t_per_qubit(200, 1e-3, 10), 20 * kn_cost(5e-6))
    best = rg.best_split(1e-3, 10, [(1, 2e-3, 10), (2, 4e-4, 20), (4, 5e-5, 40)])
    assert best["r"] in (2, 4) and math.isclose(best["delta_S"], 1e-3 - best["delta_T"])
