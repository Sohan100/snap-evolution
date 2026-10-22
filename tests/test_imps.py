"""The infinite-chain evaluator (snapevo.imps): exact cases, the gate convention against the ring, the
thermodynamic-limit per-site infidelity against the ring C(N), and its truncation convergence.

Models: XXZ chain, Delta = 0.25, J = 1, spin units, at h_x = 1.6 and h_x = 0 (snapevo's model keys A and B).
"""

import math

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from snapevo import imps as im
from snapevo.baseline import trotter_circuit
from snapevo.cells import get_cell
from snapevo.circuits import Layered, product_vectors
from snapevo.hamiltonian import get_model
from snapevo.mps import reference_circuit
from snapevo.opnorm import apply_ops

FIELDS = {"1.6": "A", "0": "B"}            # snapevo's model keys: A is h_x = 1.6, B is h_x = 0
CELL_OF = {"1.6": "xxz+x", "0": "xxz+stag"}


def _unit(v):
    v = np.asarray(v, dtype=complex)
    return v / np.linalg.norm(v)


def _haar_4(rng):
    Z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def _ring_rdm(psi, N, sites):
    t = np.asarray(psi).reshape((2,) * N)
    others = [k for k in range(N) if k not in sites]
    m = np.transpose(t, list(sites) + others).reshape(2 ** len(sites), -1)
    return m @ m.conj().T


def _ring_C(circuit, model, t, vec, N):
    """1 - F^{1/N} on the N-site ring for the unit cell placed periodically, against exact e^{-iHt}."""
    v = im.ring_state(vec, N)
    exact = spla.expm_multiply(-1j * t * model.sparse(N, "ring"), v)
    u = circuit.apply(v[None, :], N, bc="ring")[0]
    return -math.expm1(math.log(abs(np.vdot(exact, u)) ** 2) / N)


# --------------------------------------------------------------------------- exact cases

@pytest.mark.parametrize("L", [2, 4, 8])
def test_product_states_exact(L):
    a, b = product_vectors(L, 2, 31)
    A, B = im.IMPS.product(a), im.IMPS.product(b)
    exact = 1 - np.prod([abs(np.vdot(_unit(b[j]), _unit(a[j]))) ** 2 for j in range(L)]) ** (1 / L)
    f = im.fidelity(A, B)
    assert f["method"] == "dense" and f["dim"] == 1
    assert abs(f["c_inf"] - exact) < 1e-14
    assert abs(im.fidelity(A, A)["c_inf"]) < 1e-15
    assert abs(A.norm_eigenvalue() - 1) < 1e-15


@pytest.mark.parametrize("cname", ["xxz+x", "xxz+stag"])
def test_trivial_circuits_exact(cname):
    cell = get_cell(cname)
    vec = product_vectors(4, 1, 5)[0]
    ident = Layered.brickwork(cell, np.zeros(6 * cell.n_angles))
    psi = im.apply_layered(im.IMPS.product(vec), ident)
    assert psi.chi == [1, 1, 1, 1]                                      # the identity creates no entanglement
    assert abs(im.fidelity(psi, im.IMPS.product(vec))["c_inf"]) < 1e-15
    # a single-qubit sublayer (no coupling) moves each site exactly
    ang = np.zeros(cell.n_angles)
    ang[3:] = [0.37, -1.1][: cell.n_angles - 3]
    for parity in (0, 1):
        psi = im.apply_layered(im.IMPS.product(vec), Layered(cell, [(parity, ang)]))
        g1, g2 = cell.one_qubit_factor(ang)
        moved = [(g1 if (j - parity) % 2 == 0 else g2) @ _unit(vec[j]) for j in range(4)]
        assert psi.chi == [1, 1, 1, 1]
        assert abs(im.fidelity(psi, im.IMPS.product(moved))["c_inf"]) < 1e-15
        assert psi.updates == 0


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("L", [2, 4])
def test_one_layer_overlap_exact(parity, L):
    """One layer of U against one layer of W on the same product input: the overlap is a product over pairs."""
    rng = np.random.default_rng(7 + parity + L)
    U, W = _haar_4(rng), _haar_4(rng)
    vec = [_unit(v) for v in product_vectors(L, 1, 11)[0]]
    a = im.IMPS.product(vec, cutoff=0.0)
    b = im.IMPS.product(vec, cutoff=0.0)
    a.apply_layer(U, parity)
    b.apply_layer(W, parity)
    amp = 1.0
    for i in range(parity, L + parity, 2):                         # pairs (i, i+1) of the cell, the last wrapping
        x = np.kron(vec[i % L], vec[(i + 1) % L])
        amp *= abs(np.vdot(W @ x, U @ x))
    f = im.fidelity(a, b)
    assert abs(f["c_inf"] - (1 - amp ** (2 / L))) < 1e-13
    assert max(a.chi) <= 4 and a.discarded == 0.0


@pytest.mark.parametrize("field", ["1.6", "0"])
@pytest.mark.parametrize("L", [2, 4])
def test_gate_convention_matches_the_ring(field, L):
    """Inside the light cone the iMPS and the ring agree exactly: every 1-, 2- and 3-site density matrix of a
    depth-4 brickwork (the field's cell at random angles, then a random gate on the odd bonds) matches
    Layered.apply(bc="ring") and opnorm.apply_ops at N = 16 (3 + 2 x 5 sublayers < 16)."""
    rng = np.random.default_rng(3 if field == "1.6" else 4)
    cell = get_cell(CELL_OF[field])
    lay = Layered.brickwork(cell, rng.normal(size=4 * cell.n_angles))
    vec = product_vectors(L, 1, 13)[0]
    N = 16
    psi = im.apply_layered(im.IMPS.product(vec, cutoff=0.0), lay)
    ring = lay.apply(im.ring_state(vec, N)[None, :], N, bc="ring")[0]
    G = _haar_4(rng)                                               # not swap-symmetric: tests the orientation
    psi.apply_layer(G, 1)
    ring = apply_ops([((i, (i + 1) % N), G) for i in range(1, N, 2)], ring[:, None], N)[:, 0]
    for j in range(L):
        for m in (1, 2, 3):
            r_inf = psi.rdm(j, m)
            r_ring = _ring_rdm(ring, N, [(j + k) % N for k in range(m)])
            assert np.abs(r_inf - r_ring).max() < 1e-12
    assert abs(psi.norm_eigenvalue() - 1) < 1e-12


def test_period_two_input_in_a_four_site_cell():
    """A period-2 input written with a 4-site cell gives the same c_inf as the 2-site cell."""
    model = get_model("A")
    vec2 = product_vectors(2, 1, 17)[0]
    vec4 = np.concatenate([vec2, vec2])
    circ = trotter_circuit(model, 1.0, 2, 2)
    r2 = im.c_inf(circ, model, 1.0, vec2, cutoff=1e-14)
    r4 = im.c_inf(circ, model, 1.0, vec4, cutoff=1e-14)
    assert abs(r4["c_inf"] / r2["c_inf"] - 1) < 1e-8


def test_save_load_roundtrip(tmp_path):
    model = get_model("B")
    vec = product_vectors(2, 1, 19)[0]
    p = tmp_path / "ref.npz"
    v, meta = im.reference_state(model, 1.0, vec, dt=0.1, cache=p)
    w, meta2 = im.IMPS.load(p)
    assert w.chi == v.chi and w.discarded == v.discarded and meta2["t"] == 1.0
    assert abs(im.fidelity(v, w)["c_inf"]) < 1e-14


# --------------------------------------------------------------------------- the thermodynamic limit

@pytest.mark.parametrize("field", ["1.6", "0"])
@pytest.mark.parametrize("t", [1.0, 2.0])
@pytest.mark.parametrize("L", [2, 4])
def test_trotter_c_inf_matches_the_ring(field, t, L):
    """c_inf of a product formula against the iTEBD reference equals the ring C(16) (exact e^{-iHt}, the same
    unit cell placed periodically) to well within a few percent; measured, 1e-5 relative or better."""
    model = get_model(FIELDS[field])
    circ = trotter_circuit(model, t, 2, 2)
    vec = product_vectors(L, 1, 21)[0]
    r = im.c_inf(circ, model, t, vec, dt=0.1, cutoff=1e-14)
    C16 = _ring_C(circ, model, t, vec, 16)
    assert abs(r["c_inf"] / C16 - 1) < 0.01
    assert r["residual"] < 1e-12


def test_ring_converges_to_c_inf():
    """C(N) of the ring approaches c_inf as N grows (h_x = 1.6, t = 2, one order-1 step: a large light cone)."""
    model = get_model("A")
    t = 2.0
    circ = trotter_circuit(model, t, 1, 1)
    vec = product_vectors(2, 1, 23)[0]
    c = im.c_inf(circ, model, t, vec, dt=0.1, cutoff=1e-14)["c_inf"]
    d = [abs(_ring_C(circ, model, t, vec, N) / c - 1) for N in (8, 12, 16)]
    assert d[2] < 1e-3 and d[2] <= d[0]


@pytest.mark.parametrize("field", ["1.6", "0"])
def test_cutoff_and_bond_dimension_convergence(field):
    """At t = 2 the per-site infidelity of a product formula is converged in the cutoff (1e-10 to 1e-14) and in
    chi_max once chi_max reaches the untruncated bond dimension; chi_max below it moves c_inf."""
    model = get_model(FIELDS[field])
    t = 2.0
    circ = trotter_circuit(model, t, 2, 3)
    vec = product_vectors(2, 1, 29)[0]
    cs = {cut: im.c_inf(circ, model, t, vec, cutoff=cut) for cut in (1e-10, 1e-12, 1e-14)}
    ref = cs[1e-14]["c_inf"]
    assert abs(cs[1e-12]["c_inf"] / ref - 1) < 1e-4
    assert abs(cs[1e-10]["c_inf"] / ref - 1) < 1e-2
    chi = max(cs[1e-14]["chi_u"], cs[1e-14]["chi_v"])
    full = im.c_inf(circ, model, t, vec, cutoff=1e-14, chi_max=chi + 8)["c_inf"]
    assert abs(full / ref - 1) < 1e-10
    small = im.c_inf(circ, model, t, vec, cutoff=1e-14, chi_max=max(2, chi // 4))
    assert small["discarded_v"] > 1e-8 and abs(small["c_inf"] / ref - 1) > 1e-3


def test_arpack_path_matches_dense():
    """The matrix-free ARPACK eigenvalue equals the dense one."""
    model = get_model("A")
    t = 2.0
    vec = product_vectors(2, 1, 37)[0]
    v, _ = im.reference_state(model, t, vec, dt=0.1, cutoff=1e-14)
    u = im.apply_layered(im.IMPS.product(vec, cutoff=1e-14), trotter_circuit(model, t, 2, 2))
    dense = im.leading_eigenvalue(u, v, dense_max=10 ** 6)
    arp = im.leading_eigenvalue(u, v, dense_max=0)
    assert dense["method"] == "dense" and arp["method"] == "arpack"
    assert abs(dense["lambda"] - arp["lambda"]) < 1e-13
