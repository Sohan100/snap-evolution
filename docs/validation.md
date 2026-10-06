# Validation log

Checks run on this code and its inputs, newest first. Every number here can be reproduced with the commands given.

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
- Isaac's README figures hold: the new library is 1.654x the old at the median, and 1.447x to 1.739x from the 10th to the 90th percentile.
- The corrected costs are never above Isaac's Oct 1 v2 costs, and are 2 T lower at 12,929 angles.
- The $2k-2$ rule holds for all 25,550 $L = 10, 11$ gadgets as well.
- The zip carries a stale `.pyc` of the old builder, which has identical function bytecode and was dropped here.

## Tests (`pytest`, 18 tests, about 1 min)

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
- Affine extrapolation
- The order-2 Trotter layout
- KN against its README example and the cache
- A small compile that meets its budget
