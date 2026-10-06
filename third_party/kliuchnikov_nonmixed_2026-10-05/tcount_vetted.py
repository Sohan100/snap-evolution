"""Vetted T-counting.

A T-count is a property of an EXPLICIT CIRCUIT that has been VERIFIED to
implement the intended operation -- never of a formula. Every number this
module returns comes with the circuit it counts, and that circuit has been
checked against the target by a simulator written here, independent of the
synthesizer that produced it (the synthesizer does not grade itself).

Why this exists: the RUS gadget lattice was priced with upstream's sde(z, y, L),
documented as "the exact T-count of V". It is the smallest denominator exponent
k of V's entries. Exact synthesis of all 2,290 gadgets gives T(V) = 2k-2 or 2k,
never k (Giles-Selinger). Every RUS T-count in this project was understated.

The procedure, for each primitive:

  1. CONSTRUCT an explicit Clifford+T circuit.
     - exact single-qubit unitary U over Z[omega]/sqrt2^k: Matsumoto-Amano
       normal form via pygridsynth's exact synthesis (T-optimal for single-qubit
       Clifford+T: Matsumoto-Amano 2008, Giles-Selinger 2013).
     - RUS gadget (z, y, L): ancilla |0>, CNOT(s->a), V on a, CNOT(s->a),
       measure a in Z. V's circuit is expanded gate by gate on the ancilla, so
       the T-count is read off the actual two-qubit circuit. (Clifford-
       equivalent to the X-measurement form; Cliffords are free.)
  2. VERIFY by simulation, with gate matrices defined here:
     - 1q: circuit product == U up to global phase, to 1e-12; plus an exact
       ring-arithmetic check through pygridsynth.
     - RUS: Kraus operators of the two measurement outcomes. Success must be
       state-independent (K0^dag K0 = p I), must equal Rz(+-theta_claim) up to
       phase, with p = |z|^2 / 2^L. Failure must be a diagonal unitary F.
  3. COUNT T and T^dag gates in the verified circuit (achievable, so an upper
     bound; equal to the optimum for single-qubit normal forms).
  4. CORRECTION from the verified failure branch, not a lookup table: F is
     undone exactly at 0 T (Clifford) or 1 T when it is Rz(j pi/4); otherwise
     it has no exact Clifford+T form and costs the fallback (flagged MODEL,
     since that cost comes from a synthesis formula, not a circuit).
  5. EXPECTED COST  E[T] = T_att/p + T_corr (1-p)/p, and the single-shot
     fallback option t_att + (1-p) fallback -- the same structure as
     rus_lattice._gadget_cost, now fed only verified inputs.

Run `python tcount_vetted.py` for the known-answer suite.
"""
import cmath
import math
import sys

import numpy as np

try:
    import pygridsynth as pg
except ImportError as e:                                    # pragma: no cover
    raise SystemExit("tcount_vetted needs pygridsynth (pip install pygridsynth)") from e

OMEGA = cmath.exp(1j * math.pi / 4)

# ---------------------------------------------------------------- gate set
# Defined here, independently of the synthesizer.
GATES_1Q = {
    "H": np.array([[1, 1], [1, -1]], complex) / math.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], complex),
    "Sdg": np.array([[1, 0], [0, -1j]], complex),
    "X": np.array([[0, 1], [1, 0]], complex),
    "Z": np.array([[1, 0], [0, -1]], complex),
    "T": np.array([[1, 0], [0, OMEGA]], complex),
    "Tdg": np.array([[1, 0], [0, OMEGA.conjugate()]], complex),
    "W": OMEGA * np.eye(2, dtype=complex),                  # global phase
}
T_GATES = {"T", "Tdg"}
_PG_NAME = {"HGate": "H", "SGate": "S", "SXGate": "X", "TGate": "T",
            "WGate": "W"}


def count_t(word):
    return sum(g in T_GATES for g in word)


def word_matrix(word):
    """Matrix of a gate list in TIME order (word[0] acts first)."""
    M = np.eye(2, dtype=complex)
    for g in word:
        M = GATES_1Q[g] @ M
    return M


def equal_up_to_phase(A, B, tol=1e-12):
    ph = np.vdot(A.ravel(), B.ravel())
    if abs(ph) < 1e-15:
        return False
    return float(np.abs(A * (ph / abs(ph)) - B).max()) < tol


# ---------------------------------------------------------------- 1-qubit exact
def _pg_to_word(circ):
    """pygridsynth circuits are in MATRIX order (circ[0] is leftmost); return
    TIME order. The order is re-checked by every verification below, so a
    wrong assumption here fails loudly rather than miscounting."""
    names = []
    for gate in circ:
        n = type(gate).__name__
        if n not in _PG_NAME:
            raise ValueError(f"unexpected gate from synthesizer: {n}")
        names.append(_PG_NAME[n])
    return names[::-1]


def domega_unitary(z, y, L, n=0):
    """V = (1/sqrt2^L) [[z, -y* w^n], [y, z* w^n]], z/y as dicts {a,b,c,d} in
    the (a w^3 + b w^2 + c w + d) basis shared by upstream and pygridsynth."""
    zz = pg.ZOmega(z["a"], z["b"], z["c"], z["d"])
    yy = pg.ZOmega(y["a"], y["b"], y["c"], y["d"])
    return pg.DOmegaUnitary(pg.DOmega(zz, L), pg.DOmega(yy, L), n).reduce_denomexp()


def _dunitary_matrix(U):
    return np.array(U.to_complex_matrix.tolist(), complex)


def synth_1q_exact(U):
    """Exact Clifford+T circuit for a DOmegaUnitary, verified two ways.

    Returns dict(word, t_count, k): word in TIME order, k = lowest-terms
    denominator exponent (reported so the T = 2k-2 / 2k relation can be
    checked, never used as a cost)."""
    U = U.reduce_denomexp()
    circ = pg.decompose_domega_unitary(U, wires=[0])
    word = _pg_to_word(circ)
    target = _dunitary_matrix(U)
    if not equal_up_to_phase(word_matrix(word), target):
        raise AssertionError("synthesized circuit does not implement U "
                             "(independent simulator)")
    # exact ring-arithmetic cross-check (no floating point)
    back = pg.DOmegaUnitary.from_circuit(circ).reduce_denomexp()
    if not equal_up_to_phase(_dunitary_matrix(back), target):
        raise AssertionError("exact re-multiplication disagrees with U")
    return dict(word=word, t_count=count_t(word), k=U.k)


# ---------------------------------------------------------------- 2-qubit sim
def _embed(G, q):
    """1q gate on qubit q of (system=0, ancilla=1), basis |s a>."""
    return np.kron(G, np.eye(2)) if q == 0 else np.kron(np.eye(2), G)


CNOT_SA = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                   complex)                            # control s, target a


def rus_circuit(v_word):
    """Explicit two-qubit RUS circuit as a TIME-ordered list of (gate, qubit)."""
    return ([("CNOT", None)] + [(g, 1) for g in v_word]
            + [("CNOT", None)] + [("MEASURE_Z", 1)])


def simulate_rus(circuit):
    """Kraus operators (on the system) for ancilla outcomes 0 and 1."""
    U = np.eye(4, dtype=complex)
    for g, q in circuit:
        if g == "CNOT":
            U = CNOT_SA @ U
        elif g == "MEASURE_Z":
            break
        else:
            U = _embed(GATES_1Q[g], q) @ U
    # ancilla starts |0>: columns with a=0; outcome m selects rows with a=m
    return [U[np.ix_([m, 2 + m], [0, 2])] for m in (0, 1)]


def _diag_angle(D):
    """theta with D proportional to Rz(theta) = diag(e^{-i theta/2}, e^{i theta/2})."""
    return (cmath.phase(D[1, 1]) - cmath.phase(D[0, 0])) % (2 * math.pi)


def _ang_close(a, b, tol=1e-9):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi) < tol


def certify_rus(z, y, L, angle_claim=None, p_claim=None, fallback_t=18.0):
    """Certify one BRS gadget from its algebraic data. Raises on any mismatch.

    Returns a record whose every cost field is read off a verified circuit,
    except T_corr for a non-exact failure, which is flagged corr_kind='MODEL'."""
    V = domega_unitary(z, y, L)
    s = synth_1q_exact(V)
    circ = rus_circuit(s["word"])
    K0, K1 = simulate_rus(circ)
    p = float(np.real(np.trace(K0.conj().T @ K0)) / 2)
    # failure weight from K1 itself: 1 - p is ill-conditioned when p ~ 1
    # (p = 0.999998 loses ~10 digits), which once tripped the undo check
    q = float(np.real(np.trace(K1.conj().T @ K1)) / 2)
    if abs(p + q - 1) > 1e-12:
        raise AssertionError(f"outcome probabilities do not sum to 1: {p + q}")
    if np.abs(K0.conj().T @ K0 - p * np.eye(2)).max() > 1e-10:
        raise AssertionError("success probability is state-dependent")
    if np.abs(K1.conj().T @ K1 - q * np.eye(2)).max() > 1e-10:
        raise AssertionError("failure branch is not a scaled unitary")
    if abs(K0[0, 1]) + abs(K0[1, 0]) > 1e-10:
        raise AssertionError("success branch is not a z-rotation")
    theta = _diag_angle(K0)
    if angle_claim is not None and not (_ang_close(theta, angle_claim)
                                        or _ang_close(theta, -angle_claim)):
        raise AssertionError(f"angle mismatch: implemented {theta}, "
                             f"claimed {angle_claim}")
    if p_claim is not None and abs(p - p_claim) > 1e-9:
        raise AssertionError(f"p mismatch: {p} vs claimed {p_claim}")
    t_att = sum(g in T_GATES for g, _q in circ)

    # deterministic gadget (y = 0, p = 1): there is no failure branch
    if q < 1e-14:
        if np.abs(K1).max() > 1e-10:
            raise AssertionError("p = 1 but failure branch is nonzero")
        return dict(angle=theta, p=1.0, t_att=t_att, k=s["k"], t_corr=0.0,
                    corr_kind="deterministic", expected_t=float(t_att),
                    n_gates=len(circ))

    # failure: F = K1/sqrt(1-p), must be diagonal; undo it exactly if we can
    F = K1 / math.sqrt(q)
    if abs(F[0, 1]) + abs(F[1, 0]) > 1e-10:
        raise AssertionError("failure branch is not diagonal")
    phi = _diag_angle(F)
    j = phi / (math.pi / 4)
    if abs(j - round(j)) < 1e-9:
        j = int(round(j)) % 8
        # F is proportional to T^j; undo = Tdg^j = Sdg^(j//2) Tdg^(j%2)
        undo = ["Sdg"] * (j // 2) + (["Tdg"] if j % 2 else [])
        if not equal_up_to_phase(word_matrix(undo) @ F, np.eye(2), tol=1e-10):
            raise AssertionError("failure-undo circuit does not invert F")
        t_corr, corr_kind = float(count_t(undo)), ("Clifford" if j % 2 == 0 else "T")
    else:
        t_corr, corr_kind = float(fallback_t), "MODEL"
    repeat = t_att / p + t_corr * q / p
    single = t_att + q * fallback_t
    return dict(angle=theta, p=p, t_att=t_att, k=s["k"], t_corr=t_corr,
                corr_kind=corr_kind, expected_t=min(repeat, single),
                n_gates=len(circ))


# ---------------------------------------------------------------- known answers
def selftest(verbose=True):
    """Known-answer suite. Every expected value is fixed by theory, not by the
    code under test."""
    fails = []

    def check(name, cond, detail=""):
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")
        if not cond:
            fails.append(name)

    def tc_of(word):
        return synth_1q_exact(word_to_domega(word))["t_count"]

    # Cliffords are free
    for w in (["H"], ["S"], ["X"], ["H", "S", "H"], ["S", "S", "H", "S"]):
        check(f"Clifford {''.join(w)} -> 0 T", tc_of(w) == 0)
    # single T and T^dag
    check("T -> 1", tc_of(["T"]) == 1)
    check("Tdg -> 1", tc_of(["Tdg"]) == 1)
    check("T T = S -> 0", tc_of(["T", "T"]) == 0)
    # (H T)^n is a Matsumoto-Amano normal form, hence T-optimal: exactly n
    for n in (1, 2, 3, 5, 8, 12):
        got = tc_of(["H", "T"] * n)
        check(f"(HT)^{n} -> {n} (normal form is optimal)", got == n, f"got {got}")
    # REGRESSION for the bug this module exists to fix: the lowest-terms
    # denominator exponent k is NOT the T-count. For (HT)^n the T-count is n
    # while k is about n/2; a counter that returned k would fail here.
    diffs = []
    for n in (4, 6, 10):
        r = synth_1q_exact(word_to_domega(["H", "T"] * n))
        diffs.append((n, r["k"], r["t_count"]))
    check("denominator exponent k is NOT the T-count ((HT)^n, n=4,6,10)",
          all(t == n and k != t for n, k, t in diffs),
          " ".join(f"n={n}:k={k},T={t}" for n, k, t in diffs))
    # random Clifford+T words: never more T than the word, same parity
    rng = np.random.default_rng(0)
    bad_up, bad_par = 0, 0
    for _ in range(300):
        w = list(rng.choice(["H", "S", "T", "Tdg", "X"], size=rng.integers(1, 30)))
        n_w, n_s = count_t(w), tc_of(w)
        bad_up += n_s > n_w
        bad_par += (n_s - n_w) % 2 != 0
    check("300 random words: synthesized T <= word T", bad_up == 0, f"{bad_up} violations")
    check("300 random words: T-count parity invariant", bad_par == 0, f"{bad_par} violations")
    # gridsynth's own output for Rz(theta) is a normal form: exact synthesis of
    # its unitary must reproduce the same T-count (two routes must agree)
    agree = 0
    for th in (0.1, 0.37, 1.1, 2.9):
        gates = pg.gridsynth_gates(pg.mpmath.mpf(th), pg.mpmath.mpf("1e-4"))
        w = [c for c in str(gates)[::-1] if c in "HSTXW"]
        agree += tc_of(w) == count_t(w)
    check("gridsynth outputs are already T-optimal (4 angles)", agree == 4, f"{agree}/4")
    return fails


def word_to_domega(word):
    """Exact DOmegaUnitary of a TIME-ordered word, via pygridsynth's exact
    ring arithmetic, cross-checked against this module's own simulator."""
    expand = {"H": "H", "S": "S", "X": "X", "T": "T", "W": "W",
              "Sdg": "SSS", "Tdg": "SSST", "Z": "SS"}
    U = pg.DOmegaUnitary.from_gates("".join(expand[g] for g in word[::-1]))
    if not equal_up_to_phase(_dunitary_matrix(U), word_matrix(word)):
        raise AssertionError("exact word unitary disagrees with the simulator")
    return U


if __name__ == "__main__":
    print("tcount_vetted known-answer suite")
    f = selftest()
    print(f"\n{'ALL PASSED' if not f else f'{len(f)} FAILED: {f}'}")
    sys.exit(1 if f else 0)
