# Simulation plan: XXZ time evolution, unmixed, no catalysis

This is the plan for the time-evolution results of paper 1. It follows the call of 5 Oct 2026 (Isaac, David, Sohan) and Isaac's corrected library of the same day.

## Decisions this plan rests on

1. The library is Isaac's corrected RUS file (`third_party/rz_library_corrected_2026-10-05`), with T counts from exact synthesis. There is no Gidney catalysis and there are no $k\pi/12$ angles.
2. Synthesis is deterministic (unmixed) everywhere. Mixed synthesis appears only as a remark in the text (about 2x).
3. We measure errors by plugging in the actual Clifford+T circuits, not with analytic bounds. We report the operator norm (diamond for channels) and the fidelity, on product states and as a process fidelity. `docs/error_metric.md` defines both.
4. The baseline is Kliuchnikov non-mixed (Isaac's implementation, `third_party/kliuchnikov_nonmixed_2026-10-05`), not Ross-Selinger.
5. The end-to-end demonstration is a 100-site ring, so every error is reported at $N = 100$.

## Models

Both models are in spin units, $S = \sigma/2$, on a ring (`hamiltonian.py`):

$$
H = \sum_i \left( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta\, S^z_i S^z_{i+1} \right) + h_x \sum_i S^x_i ,\qquad \Delta = 0.25 .
$$

| key | $h_x$ | cell | angles per sublayer | Rz per qubit per sublayer |
|---|---|---|---|---|
| A | 1.6 | `xxz+x`: $e^{-i(aXX + bYY + cZZ)/2}\,(R_x(e) \otimes R_x(f))$ | 5 | 2.5 |
| B | 0 | `xxz+stag`: $e^{-i(aXX + bYY + cZZ)/2}\,(R_z(d) \otimes R_z(-d))$ | 4 | 2.5 |

Model B is the Sep 2026 Hamiltonian divided by 4. A spin-unit time $t$ is Pauli-unit time $t/4$, so the old cells at Pauli $t = 0.5, 1, 2$ are cells for model B at $t = 2, 4, 8$. That gives a direct check of what the corrected library alone does to the same circuits. In spin units the magnon velocity is about 1, so a 100-site ring has no wrap-around before $t \approx 50$.

David has not posted his Hamiltonian on Slack yet. If his convention differs, only `MODELS` in `hamiltonian.py` changes.

## Stages

We run the stages as follows:

1. **Train** a depth ladder per model and time at $N = 10$ (`snapevo train`). Each depth starts from Trotter, from the previous depth padded with identity sublayers, and from a random perturbation. The best of the three is kept. The ladder stops when the floor no longer falls by at least 2x per four sublayers.
2. **Compile** (`snapevo compile`). The depth is the shallowest one whose floor is at most $B/3$, plus the next depth for the tight half of the budgets, and the cheaper of the two is kept. The library is priced at $F = 22$ while compiling. The smoke run showed $D_{\rm coh}(N) \approx 2\sin(N\sqrt{C})$ (`docs/error_metric.md`), so the budget range depends on which error is the headline. There are two ranges:
   - the local range, $B \in \{10^{-4}, 3\times10^{-5}, 10^{-5}, 3\times10^{-6}, 10^{-6}\}$, which is fidelity 99% to 99.99% at $N = 100$, plus $\alpha$ and the DSF-type question. These are cheap, short compiles.
   - the global range, $B \in \{3\times10^{-7}, 10^{-7}, 3\times10^{-8}, 10^{-8}, 3\times10^{-9}\}$, for $D_{\rm total}(100)$ between about 0.1 and 0.01. These need cells of depth 15 to 27 at $t = 2$ and are the expensive, wall-limited points.
3. **Realize** (`snapevo realize`). Lattice angles are priced on the $F$ grid and stragglers are synthesized with KN.
4. **Evaluate** (`snapevo evaluate`). This computes held-out $C$ at $N = 10, 12$, the exact $w(N)$ and process fidelity for $N = 6$ to $16$, the totals at $N = 100$ with the fallbacks included, and picks $F$.
5. **Baseline** (`snapevo baseline`). Product formulas of order 2 and 4 are combined with KN at $\epsilon \in \{10^{-3}, 3\times10^{-4}, \dots, 10^{-7}\}$ and $r$ chosen so that the Trotter and synthesis errors are of the same size. Every point is measured, and the lower envelope is the baseline curve.
6. **End-to-end** (with David, not in this repo yet). DSF on the 100-site ring with MPS, compiled circuit against exact evolution. The question is whether the local observable error tracks held-out $C$ rather than $D_{\rm total}(100)$.

## Order of work

1. Model B at $t = 2$. This was the smoke run, done 2026-10-06 (below). Next comes the full budget ladder, compared point by point with the Sep 2026 runs.
2. Model A at $t = 2$: ladder, compile, evaluate.
3. Baselines for A and B at $t = 2$.
4. Both models at $t = 4$ and $t = 8$.
5. Figures: T per rotation and T per qubit per unit time against each error metric ($D_{\rm total}(100)$ and the state infidelity at $N = 100$), with SNAP and baseline on the same axes, one panel per time. A diagnostic panel shows $w(N)/N$ against $\sqrt{C}$ to test the square-root relation from the meeting.

## What we report per point

The reported quantities are the following:

- $\alpha = w/N$ at the largest measured $N$ (operator-norm error rate per site)
- $D_{\rm total}(100)$, $D_{\rm coh}(100)$ from both fits, the ledger and $F$
- state infidelity at $N = 10$ and $N = 100$, process infidelity at $N = 100$, per-site $c_e$
- held-out $C$ (worst of 8 sets) and the training budget
- T per qubit per circuit, T per rotation, T per qubit per unit time
- counts of free (Clifford), T, RUS and KN rotations
- catalyst qubits, always 0 here

## Compute

The cloud container has 2 cores. Smoke-run timings for model B, $t = 2$, are below. Earlier deep points took up to 16 h each (Pauli $t = 2$, $D = 61$). The compile checkpoints after every accepted snap and resumes from the checkpoint, so long runs can move between machines. Set `SNAPEVO_RUNS` to keep run files outside OneDrive.

## Smoke run (2026-10-06)

The smoke run used model B, $t = 2$, $N = 10$, budget $10^{-4}$, $D = 7$. Results are in `docs/validation.md`.
