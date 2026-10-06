"""Kliuchnikov et al. (arXiv:2203.10064) NON-MIXED FALLBACK synthesis of Rz(theta):
angle and epsilon in, verified circuit and T count out.

    python synthesize.py --angle 0.7 --eps 1e-4            # prints JSON
    python synthesize.py --angle 0.7 --eps 1e-4 --circuit  # also prints the gate lists

    from synthesize import synthesize
    r = synthesize(0.7, 1e-4)
    r["expected_t"], r["circuit"]

Convention: Rz(theta) = diag(e^{-i theta/2}, e^{+i theta/2}). epsilon is the DIAMOND-norm error
budget of the whole protocol (success + fallback branches).

The protocol (paper Fig. 1, Problem 3.8, Props. 3.9-3.10):
  1. projective rotation: ancilla |0>; CNOT(data->anc); V on anc; CNOT(data->anc); measure anc.
     Outcome 0 (prob p >= q): the data gets a diagonal U0 with D(U0, Rz(theta)) <= (1-s) eps.
  2. outcome 1 only (prob 1-p): the data carries a known diagonal F; apply the Ross-Selinger word B
     with D(B F, Rz(theta)) <= s eps / (1-p).
  Total diamond error <= p D0 + (1-p) D1 <= eps. Expected T = T(V) + (1-p) T(B), single shot.
Every T is counted in a circuit simulated by tcount_vetted (an independent simulator); every branch
error is MEASURED on the built circuit and asserted against its budget. The best of q in
{0.99, 0.95, 0.9} is returned (s = 0.1).
"""
import argparse
import json
import math

import numpy as np

import kliuchnikov_fallback as KF
import tcount_vetted as tv


def synthesize(theta, eps, q_options=(0.99, 0.95, 0.9), split=0.1):
    """Returns a dict with:
      expected_t      T(V) + (1-p) T(B)          (expected T count per use)
      t_v, t_b        T count of the ancilla word V and of the fallback word B
      p               success probability of the projective rotation
      diamond_success measured D(U0, Rz(theta));  diamond_fallback measured D(B F, Rz(theta))
      diamond_bound   p * diamond_success + (1-p) * diamond_fallback   (<= eps, asserted)
      circuit         explicit gate lists (time order; gates H S T X W, W = global phase omega):
                        "ancilla_V": V on the ancilla, inside CNOT(data->anc) ... CNOT(data->anc)
                        "fallback_B": applied to the data only if the ancilla reads 1 ([] if never)
      unitaries       U0 (success) and U1 = B F (fallback) on the data, as [[re, im], ...] rows
    """
    r = KF.best_protocol(float(theta), float(eps), q_options=q_options, split=split)
    V = KF.projective_search(float(theta) % (2 * math.pi), (1 - split) * float(eps), r["q"])
    s = tv.synth_1q_exact(V)
    word_v = s["word"]
    # rebuild B exactly as fallback_protocol does, so the returned circuit is the one that was priced
    K0, K1 = tv.simulate_rus(tv.rus_circuit(word_v))
    p = float(np.real(np.trace(K0.conj().T @ K0)) / 2)
    qf = 1 - p
    word_b = []
    U1 = None
    if qf > 1e-14:
        F = K1 / math.sqrt(qf)
        phiB = tv._diag_angle(KF.rz(theta) @ F.conj().T)
        word_b = KF._gridsynth_word(phiB, r["eps2"])
        U1 = tv.word_matrix(word_b) @ F
    assert tv.count_t(word_v) == r["t_v"] and tv.count_t(word_b) == r["t_b"], "rebuilt circuit != priced circuit"
    U0 = K0 / math.sqrt(p)
    m = lambda M: None if M is None else [[[float(z.real), float(z.imag)] for z in row] for row in M]
    return dict(theta=float(theta), eps=float(eps), expected_t=float(r["expected_t"]), t_v=int(r["t_v"]),
                t_b=int(r["t_b"]), p=p, q_threshold=r["q"], diamond_success=float(r["d0"]),
                diamond_fallback=float(r["d1"]), diamond_bound=float(r["bound"]),
                circuit=dict(ancilla_V=word_v, fallback_B=word_b), unitaries=dict(U0=m(U0), U1=m(U1)))


def main():
    ap = argparse.ArgumentParser(description="Kliuchnikov non-mixed fallback synthesis of Rz(theta)")
    ap.add_argument("--angle", type=float, required=True, help="theta in radians")
    ap.add_argument("--eps", type=float, required=True, help="diamond-norm error budget")
    ap.add_argument("--circuit", action="store_true", help="include gate lists and unitaries in the output")
    a = ap.parse_args()
    r = synthesize(a.angle, a.eps)
    if not a.circuit:
        r = {k: v for k, v in r.items() if k not in ("circuit", "unitaries")}
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
