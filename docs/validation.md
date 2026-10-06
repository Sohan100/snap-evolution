# Validation log

Checks run on this code and its inputs, newest first. Every number here can be reproduced with the commands given.

## First 100-site runs, 2026-10-06

Every number below is the state infidelity on the 8 random product inputs at $N = 100$ on the open chain, by MPS, cutoff $10^{-12}$ (`python -m snapevo summary` writes them all to `runs/summary.csv`). The cloud box ran them with 2 cores shared by up to four jobs.

### SNAP points

| model | $t$ | budget | $D$ | $1 - \bar F$ | with fallbacks | ring $1-(1-C)^{100}$ | T/q | T/rot | T/q/t | $\chi$ | change at cutoff/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 1 | $10^{-4}$ | 5 | 9.64e-03 | 9.71e-03 | 1.02e-02 | 66.2 | 5.30 | 66.2 | 11 | 2e-06 |
| A | 1 | $10^{-5}$ | 5 | 1.01e-03 | 1.02e-03 | 1.07e-03 | 103.3 | 8.27 | 103.3 | 11 | 6e-06 |
| A | 1 | $10^{-6}$ | 7 | 9.64e-05 | 1.12e-04 | 1.02e-04 | 146.2 | 8.35 | 146.2 | 10 | 5e-05 |
| A | 2 | $10^{-4}$ | 7 | 9.80e-03 | 9.83e-03 | 9.99e-03 | 93.8 | 5.36 | 46.9 | 20 | 5e-06 |
| A | 2 | $10^{-5}$ | 11 | 9.57e-04 | 1.24e-03 | 9.98e-04 | 160.7 | 5.85 | 80.4 | 19 | 5e-06 |
| B | 1 | $10^{-4}$ | 5 | 9.54e-03 | 9.61e-03 | 1.01e-02 | 36.2 | 2.90 | 36.2 | 11 | 5e-06 |
| B | 1 | $10^{-5}$ | 5 | 8.64e-04 | 8.71e-04 | 8.92e-04 | 68.2 | 5.46 | 68.2 | 11 | 1e-05 |
| B | 1 | $10^{-6}$ | 7 | 9.74e-05 | 9.77e-05 | 1.02e-04 | 98.4 | 5.62 | 98.4 | 10 | 3e-05 |
| B | 2 | $10^{-4}$ | 7 | 9.73e-03 | 9.76e-03 | 1.02e-02 | 70.6 | 4.03 | 35.3 | 21 | 6e-07 |
| B | 2 | $10^{-5}$ | 11 | 6.62e-04 | 6.72e-04 | 7.09e-04 | 183.2 | 6.59 | 91.6 | 20 | 3e-05 |
| B | 4 | $10^{-4}$ | 13 | 9.46e-03 | 9.49e-03 | 9.63e-03 | 106.5 | 3.28 | 26.6 | 58 | 9e-06 |

The statistical error is 0.8% to 2.1% of the infidelity. Redoing input 0 at a cutoff of $10^{-14}$ changes its infidelity by at most $5\times10^{-5}$ relative, so truncation does not matter at these times. The open chain comes out 2% to 7% below the ring prediction, as the boundary fits of the last section say it should.

Only one point kept fitted edges, model B at $t = 2$ and $B = 10^{-5}$. It is the $D = 11$ cell with large staggered angles. At $N_{\rm fit} = 12$ the fit took the open-chain infidelity from $8.3\times10^{-4}$ to $6.3\times10^{-5}$, below the ring's $8.0\times10^{-5}$. Its 30 KN rotations cost 545 T in all, 5.45 T per qubit at $N = 100$, about 3% of the point. Without them the boundary would roughly double the 100-site infidelity.

### Against the product-formula baseline

`baseline-scan` screened orders 2 and 4 with KN at the two budgets around the coherent estimate, then ran the Pareto front through the same MPS evaluation. Interpolating the front (log-log) at the SNAP point's infidelity gives the following T-count ratios, baseline over SNAP:

| model | $t$ | $B = 10^{-4}$ | $10^{-5}$ | $10^{-6}$ |
|---|---|---|---|---|
| A | 1 | below 2.0 (outside the front) | 3.0 | 4.7 |
| A | 2 | 3.8 | 4.6 | |
| B | 1 | 1.7 | 2.0 | 2.5 |
| B | 2 | 2.2 | 1.9 | |
| B | 4 | 3.2 | | |

This compares success branches. With the fallback bound on both sides the ratios grow (B 2.1 to 3.7, A 3.7 to 6.8), because the bound charges the baseline 10% to 90% of its total. The bound lets every fallback line up with the coherent error, so these totals are upper limits. The scan grids are coarse (order 2 and 4, $r$ up 1.5x per step, $\epsilon$ in steps of about 3), so a finer baseline search could move the ratios somewhat.

### Timings on the cloud box

Training ladders: $t = 1$, $D = 3$ to 9, 6 min per model. $t = 4$, $D = 9$ to 21 (model B), 55 min. Compiles: 1 min ($t = 1$, $D = 5$), 12 to 14 min ($t = 2$, $D = 11$, $B = 10^{-5}$, and $t = 4$, $D = 13$, $B = 10^{-4}$). Edge fits: 9 s to 4 min. A 100-site evaluation takes seconds up to $t = 4$ once the references are cached.

The 100-site reference at $t = 10$ needs bond dimension 448 at cutoff $10^{-10}$ (discarded weight $1.6\times10^{-6}$, 28 min on a shared core). At $10^{-12}$ it had not finished after an hour. The $t = 10$ points belong on the cluster.

## MPS evaluation on the open chain, 2026-10-06

### Entanglement of random product inputs (exact, $N = 20$)

Half-chain entropy and the bond dimension that keeps the discarded weight below $\epsilon$, for one random product input, spin units:

| $t$ | B chain $S$ | B chain $\chi(10^{-6})$, $\chi(10^{-10})$ | B ring $\chi(10^{-6})$, $\chi(10^{-10})$ | A chain $S$ | A ring $\chi(10^{-10})$ |
|---|---|---|---|---|---|
| 1 | 0.14 | 4, 7 | 12, 25 | 0.16 | 26 |
| 2 | 0.52 | 7, 12 | 28, 83 | 0.40 | 83 |
| 4 | 1.02 | 17, 35 | 171, 520 | 0.94 | 444 |
| 6 | 1.82 | 42, 87 | 651, 977 | 1.61 | 918 |

The ring's half has two cuts, so its bond dimension is about the square of the chain's once the two cuts are independent. At $N = 20$ and $t \ge 4$ the ring numbers are already capped by the system size. This is why the 100-site runs use an open chain (`docs/error_metric.md`).

### MPS code against state vectors (`tests/test_mps.py`)

With no truncation, the MPS of the realized smoke circuit and of a trained model A cell at $N = 10$ agree with the state-vector simulation to $8\times10^{-15}$, with the default edges and with random edge gates. MPS overlaps agree with dense ones to $10^{-17}$.

### The reference $e^{-iHt}$

The reference is the order-4 Suzuki formula. Its infidelity per site against exact evolution, open chain, $N = 12$, 4 inputs:

| model | $t$ | step 0.2 | step 0.1 | step 0.05 |
|---|---|---|---|---|
| B | 2 | $1.6\times10^{-13}$ | below $10^{-13}$ | below $10^{-13}$ |
| B | 10 | $4.2\times10^{-13}$ | below $10^{-13}$ | below $10^{-13}$ |
| A | 2 | $1.0\times10^{-11}$ | below $10^{-13}$ | below $10^{-13}$ |
| A | 10 | $2.6\times10^{-11}$ | $1.7\times10^{-13}$ | below $10^{-13}$ |

We use step 0.2. At $N = 100$ that is at most $3\times10^{-9}$, far below any circuit we evaluate.

### Cost of the 100-site references (model B, one input, cutoff $10^{-12}$, one core)

| $t$ | peak bond dimension | summed discarded weight | seconds |
|---|---|---|---|
| 1 | 9 | $5.6\times10^{-10}$ | 0.3 |
| 2 | 16 | $1.5\times10^{-9}$ | 0.8 |
| 4 | 43 | $4.4\times10^{-9}$ | 5 |
| 6 | 109 | $8.0\times10^{-9}$ | 36 |
| 8 | 275 | $1.2\times10^{-8}$ | 364 |

### Boundary of the ring-trained cells

Infidelity on 24 random product inputs at $t = 2$, ring against open chain. On the chain the wrap gate keeps only its single-qubit factor ("1q") or is dropped entirely ("drop"). The last two columns fit $1 - F = cN + b$ to $N = 12, 14, 16$ and give $b/c$, the boundary in bulk sites:

| cell | $N = 8$ ring, 1q, drop | $N = 16$ ring, 1q, drop | $b/c$ ring | $b/c$ 1q |
|---|---|---|---|---|
| B, SNAP $D = 7$, $10^{-4}$ | $8.0, 5.1, 5.1 \times10^{-4}$ | $1.6, 1.3, 1.3 \times10^{-3}$ | $-1.1$ | $-3.2$ |
| B, trained $D = 7$ | $1.5, 0.88, 0.88 \times10^{-4}$ | $3.1, 2.5, 2.5 \times10^{-4}$ | $-0.6$ | $-3.6$ |
| B, trained $D = 11$ | $7.9\times10^{-6}$, $3.8\times10^{-4}$, $8.9\times10^{-3}$ | $1.6\times10^{-5}$, $3.8\times10^{-4}$, $9.1\times10^{-3}$ | $-1.5$ | no fit |
| A, trained $D = 7$ | $1.7, 1.6 \times10^{-4}$, 0.41 | $3.3, 3.3 \times10^{-4}$, 0.40 | $-0.3$ | $-0.5$ |
| A, trained $D = 11$ | $1.3, 1.8 \times10^{-5}$, 0.61 | $2.5, 3.0 \times10^{-5}$, 0.60 | $-0.5$ | $2.4$ |

Dropping the wrap gate removes half the field from the end sites of model A, so the single-qubit factor has to stay. With it, model A and the shallow model B cells are fine. The $D = 11$ zero-field cell has large staggered angles (up to 0.5 rad, against 0.002 at $D = 7$) and a boundary error of about $4\times10^{-4}$ that does not grow with $N$.

Fitting the edge rotations of that cell at $N_{\rm fit} = 12$ (30 angles, 89 s) takes the fit infidelity from $4.1\times10^{-4}$ to $8.6\times10^{-6}$, below the ring's $1.2\times10^{-5}$. On fresh inputs the open chain with fitted edges gives $5.0\times10^{-6}$, $8.9\times10^{-6}$ and $1.3\times10^{-5}$ at $N = 8, 12, 16$, against $7.9\times10^{-6}$, $1.2\times10^{-5}$ and $1.6\times10^{-5}$ on the ring. For the smoke circuit the fit gains less than one bulk site's worth, so its edges stay at the cell's own factor and cost nothing.

### KN on negative angles

Isaac's `synthesize` failed its own check ("rebuilt circuit != priced circuit") at $\theta = -0.4429453$, $\epsilon = 10^{-4}$, robustly under perturbations of $10^{-8}$ in $\theta$ and 0.1% in $\epsilon$. It prices at $\theta$ and rebuilds at $\theta \bmod 2\pi$, and the two searches can differ for negative angles. Passing $\theta \bmod 2\pi$ fixes it (15.9 T expected). The vendored package is unchanged, and `synthesis.py` now always passes $[0, 2\pi)$. Worth telling Isaac.

## Smoke run, 2026-10-06

The smoke run used model B ($h_x = 0$), $t = 2$ in spin units (Pauli $t = 0.5$), on the cloud box with 2 cores shared by four jobs.

### Training (`snapevo train --model B --t 2 --depths 7,9,11,13`)

The new training reproduces the Sep 2026 Pauli-unit ladder:

| D | C_train, new | C_train, Sep 2026 |
|---|---|---|
| 7 | 1.743e-5 | 1.74e-5 |
| 9 | 4.172e-6 | 4.17e-6 |
| 11 | 9.209e-7 | 9.21e-7 |

Model A ($h_x = 1.6$, cell `xxz+x`) trains to about the same floors: $1.975\times10^{-5}$ at $D = 7$, $4.341\times10^{-6}$ at $D = 9$ and $1.442\times10^{-6}$ at $D = 11$. The last one stopped at the iteration limit.

### One SNAP point (`snapevo point --model B --t 2 --budget 1e-4 --D 7`)

| quantity | value |
|---|---|
| compile | 7 zero-T snaps, 21 Hessian snaps, 13 library stragglers, 0 KN stragglers, 3.5 min |
| realized training C | 9.834e-5 (budget 1e-4) |
| held-out C, N = 10 | 1.024e-4, 9.92e-5, 9.94e-5, 1.000e-4 |
| held-out C, N = 12 | 9.88e-5, 1.002e-4, 9.94e-5, 1.015e-4 |
| rotations | 8 Clifford, 0 T, 20 RUS, 0 KN, out of 28 angles |
| w/N at N = 6, 8, 10, 12, 14 | 0.0177, 0.0200, 0.0192, 0.0202, 0.0198 |
| alpha / sqrt(C) | 1.95 |
| extrapolated w(100) | 2.10 (affine), 2.40 (power) |
| D_coh(100) | 1.86, saturated |
| per-site process infidelity $c_e$, N = 6 to 14 | 1.04e-4 (exact to N = 10, Haar above) |
| state infidelity, N = 10 / N = 100 | 1.02e-3 / 1.03e-2 (fallbacks included at N = 100) |
| process infidelity, N = 100 | 1.05e-2 |
| ledger at N = 100 | 1.06e-3 at F = 18 (8.4e-5 at F = 22) |
| T per qubit (F = 18) | 70.5 |
| T per rotation | 4.03 |
| T per qubit per unit time | 35.2 |

At the same time and budget, the Sep 2026 compile cost 48.5 T per qubit under the Aug library without catalysis, and 33.4 with three catalysis levels and the $k\pi/12$ angles. 70.5 / 48.5 = 1.45, which matches the median repricing ratio of 1.46 that the 2026-10-06 library check found for level-0 circuits.

Lanczos timings for the exact operator norm at $D = 7$, single core under load, were 0 s, 2 s, 4 s and 7 s at $N = 6$ to $12$, and 100 s at $N = 14$.

### One baseline point (`snapevo baseline --model B --t 2 --order 2 --steps 4 --eps 1e-4`)

| quantity | value |
|---|---|
| distinct angles synthesized | 4 (KN costs 16.0 to 19.3 T each) |
| held-out C, Trotter only / realized | 3.85e-5 / 3.89e-5 |
| alpha = w/N at N = 14 | 0.0116 |
| D_total(100) | 1.11 (ledger 6.9e-3) |
| state / process infidelity, N = 100 | 4.34e-3 / 4.37e-3 (Trotter alone 3.85e-3 / 3.87e-3) |
| T per qubit | 231.8 |
| T per rotation | 17.2 (13.5 rotations per qubit) |

This point is about 2.5 times more accurate in $C$ than the SNAP point, so the two do not compare directly. The KN fallbacks raise its state infidelity bound at $N = 100$ by 12%, against 1% for the SNAP point at $F = 18$. Matched comparisons need the full budget ladders. The constant $\alpha/\sqrt{C} = 1.86$ is close to SNAP's.

## Port check, 2026-10-06

We compiled the Sep 2026 cell (model B, Pauli $t = 0.5$, $D = 7$, budget $10^{-4}$) twice. The first run used the old `tcount_tfim.escalation_hessian`, with the corrected library injected and catalysis and $k\pi/12$ off. The second run used `snapevo.snap`. Both runs used the same states and the old straggler cost law. Every final angle and cost agrees exactly, as do $C$ ($9.834\times10^{-5}$), T per qubit (70.68 at $F = 22$) and every counter: 7 zero-T, 21 Hessian, 13 library stragglers. The script is not part of the repo because it needs the old package.

## Library file, 2026-10-06 (`python -m snapevo verify-library`)

The run printed the following:

```
SO(3) exponent vs BFS T count, all 18384 Clifford+T elements with T <= 8: 0 mismatches
4580 gadget rows checked: all checks pass
F = 18 lattice vs shipped rz_library_cat0_kn18.csv: max cost difference 5.0e-09
```

A longer check the same day, in the claude.ai project doc `rz_library_corrected_check_2026-10-06`, also found the following:

- Pygridsynth's exact synthesis gives the same T counts on all 4,580 rows once $n$ is supplied.
- Isaac's README figures hold. The new library is 1.654x the old at the median, and 1.447x to 1.739x from the 10th to the 90th percentile.
- The corrected costs are never above Isaac's Oct 1 v2 costs, and are 2 T lower at 12,929 angles.
- The $2k-2$ rule holds for all 25,550 $L = 10, 11$ gadgets as well.
- The zip carries a stale `.pyc` of the old builder, which has identical function bytecode and was dropped here.

## Tests (`pytest`, 27 tests, about 2 min)

The suite checks the following:

- The library rows, the shipped lattice and the BFS T-count rule
- The fallback error law and the two pricing strategies
- Spin against Pauli units
- JAX against numpy gates
- That the inverse circuit undoes the circuit
- First-order Trotter convergence
- Exact single-angle curves, including the staggered slot
- Lanczos against dense arc widths
- Exact against Haar-estimated process fidelity, and the fidelity bound
- MPS against state vectors on the open chain, with default and random edges
- The order-4 reference against exact evolution
- The truncation error bar, MPS save and load, and the MPS evaluation against dense fidelities
- The choice of the fallback cost F
- ZYZ angles, the edge start point, and an edge fit that lowers the open-chain infidelity
- Affine extrapolation
- The order-2 Trotter layout
- KN against its README example and the cache
- A small compile that meets its budget
