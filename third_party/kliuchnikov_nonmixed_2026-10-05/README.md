# Kliuchnikov non-mixed fallback synthesis of Rz(theta)

Kliuchnikov, Lauter, Minko, Paetznick, Petit (arXiv:2203.10064), the FALLBACK protocol, non-mixed,
built as explicit Clifford+T circuits. Angle and epsilon in, verified circuit and T count out.

## Install
    python -m venv venv && . venv/bin/activate
    pip install -r requirements.txt          # numpy, mpmath, pygridsynth

## Use
    python synthesize.py --angle 0.7 --eps 1e-4             # JSON: expected T, T(V), T(B), p, errors
    python synthesize.py --angle 0.7 --eps 1e-4 --circuit   # + the gate lists and branch unitaries

    from synthesize import synthesize
    r = synthesize(0.7, 1e-4)
    r["expected_t"]          # 17.08 for this example
    r["circuit"]             # {"ancilla_V": [...], "fallback_B": [...]}

Convention: Rz(theta) = diag(e^{-i theta/2}, e^{+i theta/2}). eps is the DIAMOND-norm error of the
whole protocol. Gates in the gate lists: H S T X W (W = global phase e^{i pi/4}), in time order.

## The protocol
1. Projective rotation:
   - ancilla |0>, CNOT(data->ancilla), V on the ancilla, CNOT(data->ancilla), measure the ancilla;
   - outcome 0 (probability p >= 0.9-0.99): the data gets a diagonal unitary U0 with
     D(U0, Rz(theta)) <= 0.9 eps.
2. Fallback, outcome 1 only (probability 1-p):
   - the data carries a known diagonal F;
   - apply the Ross-Selinger word B with D(B F, Rz(theta)) <= 0.1 eps / (1-p).
3. Total error <= p D0 + (1-p) D1 <= eps (convexity of the diamond norm).
4. Cost (single shot): expected T = T(V) + (1-p) T(B).

`synthesize` returns the cheapest of q in {0.99, 0.95, 0.9} (the minimum success probability the
search for V enforces).

## What is verified
- Every T count is counted in the explicit gate list, which is simulated by an independent
  simulator (tcount_vetted.py).
- Both branch errors are MEASURED on the built circuit and asserted against their budgets.
- `python kliuchnikov_fallback.py --selftest`: region and ellipse containment, Clifford targets
  (p = 1, 0 T), branch budgets at 12 (theta, eps) pairs, T parity. ALL PASSED.
- `python kliuchnikov_fallback.py --calibrate` reproduces the paper's Table 1 (Clifford+T,
  Fallback, mean, q = 0.99), 40 random angles per eps:

  | eps | ours (mean) | paper | gap |
  |---|---|---|---|
  | 1e-4 | 18.74 | 19.44 | -0.70 |
  | 1e-5 | 23.06 | 22.86 | +0.20 |
  | 1e-6 | 26.25 | 26.28 | -0.03 |
  | 1e-7 | 29.81 | 29.70 | +0.11 |

  Fit 1.10 log2(1/eps) + 4.44 (paper: 1.03 log2(1/eps) + 5.75). PASS.

## Measuring the error of a whole circuit directly
The synthesised rotation is a two-branch channel, not one unitary: U0 with probability p (>= 0.9),
and U1 = B F with probability 1-p. Both unitaries are returned (`--circuit`, key "unitaries").
For a direct whole-circuit error measurement, use one of:
- simulate the circuit with every rotation replaced by its U0, and add sum (1-p) * D1 as a
  rigorous term for the rare fallback branches; or
- sample the branches.
The per-rotation diamond bound `diamond_bound` (<= eps) is also returned.

## Files
- synthesize.py            entry point (function + CLI)
- kliuchnikov_fallback.py  protocol construction, selftest, calibration
- tcount_vetted.py         exact synthesis + independent simulator / T counter
- kliuchnikov_calibration.json   calibration output
