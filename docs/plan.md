# Simulation plan: XXZ time evolution, unmixed, no catalysis

This is the plan for the time-evolution results of paper 1. It follows the call of 5 Oct 2026 (Isaac, David, Sohan), Isaac's corrected library of the same day, and Sohan's decisions of 6 Oct.

## Decisions this plan rests on

1. The library is Isaac's corrected RUS file (`third_party/rz_library_corrected_2026-10-05`), with T counts from exact synthesis. There is no Gidney catalysis and there are no $k\pi/12$ angles.
2. Synthesis is deterministic (unmixed) everywhere. Mixed synthesis appears only as a remark in the text (about 2x).
3. The error is computed on the actual Clifford+T circuits at the full size, not bounded and not extrapolated. It is the state infidelity on random product inputs at $N = 100$, by MPS (`docs/error_metric.md`).
4. The 100 sites form an open chain, which MPS can follow to $t = 10$. A ring would stop at about $t = 4$. Cells are still trained on a ring, and edge rotations handle the two ends.
5. The baseline is Kliuchnikov non-mixed (Isaac's implementation, `third_party/kliuchnikov_nonmixed_2026-10-05`), not Ross-Selinger.

## Models

Both models are in spin units, $S = \sigma/2$ (`hamiltonian.py`):

$$
H = \sum_i \left( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta\, S^z_i S^z_{i+1} \right) + h_x \sum_i S^x_i ,\qquad \Delta = 0.25 .
$$

| key | $h_x$ | cell | angles per sublayer | Rz per qubit per sublayer |
|---|---|---|---|---|
| A | 1.6 | `xxz+x`: $e^{-i(aXX + bYY + cZZ)/2}\,(R_x(e) \otimes R_x(f))$ | 5 | 2.5 |
| B | 0 | `xxz+stag`: $e^{-i(aXX + bYY + cZZ)/2}\,(R_z(d) \otimes R_z(-d))$ | 4 | 2.5 |

Model B is the Sep 2026 Hamiltonian divided by 4. A spin-unit time $t$ is Pauli-unit time $t/4$, so the Sep runs at Pauli $t = 0.5, 1, 2$ correspond to model B at $t = 2, 4, 8$. In spin units the magnon velocity is about 1, so the light cone at $t = 10$ spans about 20 sites, well inside the 100-site chain.

David has not posted his Hamiltonian on Slack yet. If his convention differs, only `MODELS` in `hamiltonian.py` changes.

## The sweep

Models A and B at $t = 1, 2, 4, 6, 8, 10$, with SNAP at training budgets $B = 10^{-4}, 10^{-5}, 10^{-6}$. In fidelity these are 99% to 99.99% at $N = 100$, since $1 - F \approx 100\,C$. Each point and each time also gets a baseline curve. `cluster/make_jobs.py` holds the grid, including the training depths per time.

## Stages

We run the stages as follows:

1. **Train** a depth ladder per model and time on the 10-site ring (`snapevo train`). Each depth starts from Trotter, from the previous depth padded with identity sublayers, and from a random perturbation, and the best of the three is kept. The ladder stops when the floor no longer falls by at least 2x per four sublayers.
2. **Compile** (`snapevo compile`). The depth is the shallowest one whose floor is at most $B/3$. The library is priced at $F = 22$ while compiling.
3. **Realize** (`snapevo realize`). Lattice angles are priced on the $F$ grid, stragglers are synthesized with KN, and the edge rotations are fitted and synthesized.
4. **References** (`snapevo reference`). MPS of $e^{-iHt}$ on the 8 inputs at $N = 100$, cached per model and time and shared by every circuit.
5. **Evaluate** (`snapevo evaluate`). MPS of the circuit on the same inputs, the infidelity with its error bars, the fallback bound, and $F$.
6. **Baseline** (`snapevo baseline-scan --evaluate`). Product formulas of order 2 and 4 with KN, screened by ring $C$, then the Pareto front evaluated like SNAP.
7. **End-to-end** (with David, not in this repo yet). The dynamical structure factor on the 100-site chain, compiled circuit against exact evolution.

`snapevo point` runs stages 2, 3 and 5. `snapevo report` prints one model and time.

## Where it runs

The cloud box (2 cores) runs $t = 1$ and $2$, the references up to $t = 6$, and the first baselines. The cluster runs the rest from `cluster/` (`cluster/README.md`). Training and compiling at $t \ge 4$ is the expensive part: the Sep runs took up to 16 h for a single compile of a 61-sublayer cell. The MPS references cost seconds up to $t = 6$, minutes at $t = 8$ and about an hour per input at $t = 10$.

## What we report per point

The reported quantities are the following:

- $1 - \bar F$ at $N = 100$ for the success branch and with fallbacks, its statistical error, its truncation bar
- the per-site value $c$, next to the ring's held-out $C$
- T per qubit per circuit, T per rotation, T per qubit per unit time, with the edge share
- counts of free (Clifford), T, RUS and KN rotations, the fallback cost $F$
- catalyst qubits, always 0 here

## Figures

T per rotation and T per qubit per unit time against $1 - F$ at $N = 100$, SNAP and baseline on the same axes, one panel per time. A second figure shows T per qubit per unit time against $t$ at fixed $1 - F$.
