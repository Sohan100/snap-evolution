# Validation log

Checks run on this code and its inputs, newest first. Every number here can be reproduced with the commands given.

## Budget sweep at $t = 1$ and 2, 2026-10-07

SNAP at 15 training budgets in half decades from $10^{-1}$ to $10^{-8}$, for both models, against the rebuilt baseline below. Same evaluation as the first runs, 8 random product inputs at $N = 100$ on the open chain, cutoff $10^{-12}$. `python -m snapevo report --model B --t 2` lists every point of one model and time, `python -m snapevo compare` prints the comparison tables below, and `python -m snapevo plot` draws `runs/figures/sweep_*.png` three ways (success branch, with the fallback bound, with the fallbacks drawn).

### SNAP at $t = 1$

| model | budget | $D$ | $1 - \bar F$ | with the bound | drawn | $c$ | T/q | edges T/q | T/rot | T/q/t | $F$ | $\chi$ | change at cutoff/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | $10^{-1}$ | 2 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 9.77e-02 | 0.0 | 0.00 | 0.00 | 0.0 | 18 | 9 | 2e-11 |
| A | $3\times10^{-2}$ | 2 | 9.36e-01 | 9.36e-01 | 9.36e-01 | 2.72e-02 | 9.0 | 0.00 | 1.79 | 9.0 | 18 | 9 | 1e-08 |
| A | $10^{-2}$ | 3 | 5.58e-01 | 5.58e-01 | 5.58e-01 | 8.14e-03 | 16.3 | 0.00 | 2.17 | 16.3 | 18 | 9 | 1e-07 |
| A | $3\times10^{-3}$ | 3 | 2.20e-01 | 2.20e-01 | 2.20e-01 | 2.48e-03 | 20.7 | 0.00 | 2.76 | 20.7 | 18 | 9 | 7e-07 |
| A | $10^{-3}$ | 5 | 5.99e-02 | 5.99e-02 | 5.99e-02 | 6.17e-04 | 37.6 | 0.30 | 2.98 | 37.6 | 18 | 11 | 3e-08 |
| A | $3\times10^{-4}$ | 5 | 2.82e-02 | 2.82e-02 | 2.82e-02 | 2.86e-04 | 46.8 | 0.00 | 3.74 | 46.8 | 18 | 11 | 2e-07 |
| A | $10^{-4}$ | 5 | 9.64e-03 | 9.71e-03 | 9.71e-03 | 9.69e-05 | 66.2 | 0.00 | 5.30 | 66.2 | 20 | 11 | 2e-06 |
| A | $3\times10^{-5}$ | 5 | 2.81e-03 | 2.82e-03 | 2.82e-03 | 2.81e-05 | 83.5 | 0.00 | 6.68 | 83.5 | 22 | 10 | 6e-06 |
| A | $10^{-5}$ | 5 | 1.01e-03 | 1.02e-03 | 1.02e-03 | 1.01e-05 | 103.3 | 0.00 | 8.27 | 103.3 | 22 | 11 | 6e-06 |
| A | $3\times10^{-6}$ | 7 | 2.99e-04 | 3.01e-04 | 3.01e-04 | 2.99e-06 | 119.7 | 0.00 | 6.84 | 119.7 | 24 | 11 | 2e-05 |
| A | $10^{-6}$ | 7 | 9.64e-05 | 1.12e-04 | 9.96e-05 | 9.64e-07 | 146.2 | 0.00 | 8.35 | 146.2 | 24 | 10 | 5e-05 |
| A | $10^{-6}$ | 11 | 9.64e-05 | 9.68e-05 | 9.68e-05 | 9.64e-07 | 146.8 | 0.00 | 5.34 | 146.8 | 26 | 10 | 3e-05 |
| A | $3\times10^{-7}$ | 9 | 3.09e-05 | 3.11e-05 | 3.11e-05 | 3.09e-07 | 175.4 | 0.00 | 7.80 | 175.4 | 26 | 10 | 6e-05 |
| A | $10^{-7}$ | 11 | 9.00e-06 | 1.09e-05 | 9.08e-06 | 9.00e-08 | 206.6 | 0.79 | 7.43 | 206.6 | 28 | 10 | 3e-04 |
| A | $10^{-7}$ | 15 | 9.77e-06 | 1.38e-05 | 9.93e-06 | 9.77e-08 | 262.2 | 5.96 | 6.91 | 262.2 | 28 | 10 | 2e-04 |
| A | $3\times10^{-8}$ | 15 | 2.88e-06 | 4.04e-06 | 2.92e-06 | 2.88e-08 | 275.2 | 6.42 | 7.26 | 275.2 | 28 | 10 | 1e-03 |
| A | $10^{-8}$ | 15 | 9.45e-07 | 1.44e-06 | 9.70e-07 | 9.45e-09 | 317.7 | 2.07 | 8.38 | 317.7 | 30 | 10 | 2e-03 |
| B | $10^{-1}$ | 2 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 8.80e-02 | 0.5 | 0.00 | 0.10 | 0.5 | 18 | 10 | 1e-10 |
| B | $3\times10^{-2}$ | 2 | 9.20e-01 | 9.20e-01 | 9.20e-01 | 2.50e-02 | 5.4 | 0.00 | 1.07 | 5.4 | 18 | 10 | 9e-09 |
| B | $10^{-2}$ | 3 | 5.62e-01 | 5.62e-01 | 5.62e-01 | 8.24e-03 | 9.9 | 0.00 | 1.33 | 9.9 | 18 | 10 | 2e-07 |
| B | $3\times10^{-3}$ | 3 | 1.96e-01 | 1.96e-01 | 1.96e-01 | 2.18e-03 | 17.8 | 0.00 | 2.37 | 17.8 | 18 | 10 | 6e-07 |
| B | $10^{-3}$ | 3 | 8.39e-02 | 8.39e-02 | 8.39e-02 | 8.76e-04 | 18.6 | 0.00 | 2.48 | 18.6 | 18 | 10 | 2e-06 |
| B | $3\times10^{-4}$ | 5 | 2.88e-02 | 2.89e-02 | 2.89e-02 | 2.92e-04 | 30.6 | 0.00 | 2.45 | 30.6 | 18 | 10 | 4e-06 |
| B | $10^{-4}$ | 5 | 9.54e-03 | 9.61e-03 | 9.61e-03 | 9.58e-05 | 36.2 | 0.00 | 2.90 | 36.2 | 18 | 11 | 5e-06 |
| B | $3\times10^{-5}$ | 5 | 1.56e-03 | 1.57e-03 | 1.57e-03 | 1.56e-05 | 54.3 | 0.00 | 4.34 | 54.3 | 22 | 12 | 2e-05 |
| B | $10^{-5}$ | 5 | 8.64e-04 | 8.71e-04 | 8.71e-04 | 8.64e-06 | 68.2 | 0.00 | 5.46 | 68.2 | 20 | 11 | 1e-05 |
| B | $3\times10^{-6}$ | 7 | 2.83e-04 | 2.86e-04 | 2.86e-04 | 2.83e-06 | 78.4 | 0.00 | 4.48 | 78.4 | 22 | 11 | 2e-05 |
| B | $10^{-6}$ | 7 | 9.74e-05 | 9.77e-05 | 9.77e-05 | 9.74e-07 | 98.4 | 0.00 | 5.62 | 98.4 | 26 | 10 | 3e-05 |
| B | $10^{-6}$ | 11 | 8.63e-05 | 8.66e-05 | 8.66e-05 | 8.63e-07 | 96.2 | 0.00 | 3.50 | 96.2 | 26 | 10 | 3e-05 |
| B | $3\times10^{-7}$ | 9 | 2.95e-05 | 3.16e-05 | 2.98e-05 | 2.95e-07 | 195.5 | 3.56 | 8.60 | 195.5 | 26 | 11 | 4e-05 |
| B | $3\times10^{-7}$ | 13 | 2.85e-05 | 2.94e-05 | 2.86e-05 | 2.85e-07 | 173.4 | 5.83 | 5.28 | 173.4 | 28 | 11 | 8e-05 |
| B | $10^{-7}$ | 11 | 9.60e-06 | 9.84e-06 | 9.63e-06 | 9.60e-08 | 216.2 | 4.57 | 7.78 | 216.2 | 28 | 11 | 2e-04 |
| B | $10^{-7}$ | 15 | 8.69e-06 | 8.74e-06 | 8.74e-06 | 8.69e-08 | 141.8 | 0.00 | 3.78 | 141.8 | 28 | 10 | 3e-04 |
| B | $3\times10^{-8}$ | 13 | 2.99e-06 | 4.03e-06 | 3.03e-06 | 2.99e-08 | 265.4 | 7.14 | 8.08 | 265.4 | 28 | 10 | 9e-04 |
| B | $10^{-8}$ | 15 | 9.98e-07 | 1.11e-06 | 1.01e-06 | 9.98e-09 | 175.1 | 0.00 | 4.67 | 175.1 | 28 | 11 | 1e-03 |

A few things stand out.

- At $10^{-1}$ model A compiles to a Clifford circuit with no T at all, and model B to 0.5 T per qubit. Their $1 - F$ at $N = 100$ is 1 to three digits, so the loose end only means something per site.
- The realized $c$ is at or below the budget everywhere, 2% to 7% below the ring prediction as before. Model B at $3\times10^{-5}$ lands at $1.6\times10^{-5}$, well under its budget.
- Model B seems to hit a wall between $10^{-6}$ and $3\times10^{-7}$, where T per qubit doubles (98 to 196) as the compile moves from $D = 7$ to 9. That is the depth rule, not SNAP. Compiled at the shallowest depth with 30 times more room under the budget (`--margins 30`), model B needs 96 T per qubit at $10^{-6}$ ($D = 11$), 173 at $3\times10^{-7}$ ($D = 13$) and 142 at $10^{-7}$ ($D = 15$), against 98, 196 and 216 at the shallowest depths that fit. The deeper cells have more rotations, but with more room each comes out cheaper, 3.8 T per rotation at $10^{-7}$ against 7.8. For model A the deeper cell does not pay. At $10^{-7}$ it needs 262 T per qubit at $D = 15$ against 207 at $D = 11$, and at $10^{-6}$ the two are equal. So the cluster jobs compile every budget at both depths (`--margins 3,30`), and the plots and the comparison keep the cheaper.
- Model A grows by up to 30 T per qubit per half decade down to $10^{-7}$, and by 40 to 70 below, where its ladder runs out of room ($D = 15$, floor $5.5\times10^{-9}$).
- Model B reaches $10^{-8}$ at 175 T per qubit ($D = 15$), about what it needs at $3\times10^{-7}$ on the deeper cell. Its $3\times10^{-8}$ point ($D = 13$, 265 T per qubit) sits off the envelope for the same reason as the wall.
- Truncation never matters here. Redoing input 0 at a 100 times smaller cutoff changes its infidelity by at most $2\times10^{-3}$ relative.
- By the bound the fallbacks are under 10% of the error except for model A at $10^{-6}$ (14%), and from $10^{-7}$ down, where the KN stragglers and edge rotations bring it to 17% to 34%. Drawn, every point is within 4% of its success branch.

### SNAP at $t = 2$

| model | budget | $D$ | $1 - \bar F$ | with the bound | drawn | $c$ | T/q | edges T/q | T/rot | T/q/t | $F$ | $\chi$ | change at cutoff/100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | $10^{-1}$ | 3 | 1.00e+00 | 1.00e+00 | 1.00e+00 | 9.69e-02 | 3.5 | 0.00 | 0.47 | 1.8 | 18 | 16 | 2e-11 |
| A | $3\times10^{-2}$ | 5 | 9.41e-01 | 9.41e-01 | 9.41e-01 | 2.80e-02 | 14.1 | 0.00 | 1.12 | 7.0 | 18 | 16 | 5e-08 |
| A | $10^{-2}$ | 5 | 5.65e-01 | 5.65e-01 | 5.65e-01 | 8.29e-03 | 24.8 | 0.00 | 1.99 | 12.4 | 18 | 16 | 2e-07 |
| A | $3\times10^{-3}$ | 5 | 2.60e-01 | 2.60e-01 | 2.60e-01 | 3.01e-03 | 38.5 | 0.00 | 3.08 | 19.3 | 18 | 20 | 5e-07 |
| A | $10^{-3}$ | 7 | 8.96e-02 | 8.98e-02 | 8.98e-02 | 9.39e-04 | 64.0 | 0.00 | 3.66 | 32.0 | 18 | 20 | 7e-08 |
| A | $3\times10^{-4}$ | 7 | 3.06e-02 | 3.09e-02 | 3.09e-02 | 3.11e-04 | 76.1 | 0.00 | 4.35 | 38.1 | 18 | 20 | 1e-06 |
| A | $10^{-4}$ | 7 | 9.80e-03 | 9.83e-03 | 9.83e-03 | 9.85e-05 | 93.8 | 0.00 | 5.36 | 46.9 | 20 | 20 | 5e-06 |
| A | $3\times10^{-5}$ | 9 | 2.65e-03 | 2.67e-03 | 2.67e-03 | 2.66e-05 | 126.9 | 0.00 | 5.64 | 63.4 | 22 | 19 | 7e-06 |
| A | $10^{-5}$ | 11 | 9.57e-04 | 1.24e-03 | 9.96e-04 | 9.57e-06 | 160.7 | 0.00 | 5.85 | 80.4 | 22 | 19 | 5e-06 |
| A | $3\times10^{-6}$ | 13 | 2.76e-04 | 3.09e-04 | 2.78e-04 | 2.76e-06 | 199.6 | 1.06 | 6.07 | 99.8 | 24 | 19 | 6e-05 |
| A | $10^{-6}$ | 17 | 9.97e-05 | 1.02e-04 | 1.00e-04 | 9.97e-07 | 255.7 | 4.09 | 5.95 | 127.9 | 26 | 19 | 1e-04 |
| A | $3\times10^{-7}$ | 21 | 2.95e-05 | 3.08e-05 | 2.98e-05 | 2.95e-07 | 290.0 | 6.47 | 5.46 | 145.0 | 26 | 18 | 3e-04 |
| B | $10^{-1}$ | 3 | 9.99e-01 | 9.99e-01 | 9.99e-01 | 7.34e-02 | 2.0 | 0.00 | 0.27 | 1.0 | 18 | 16 | 3e-10 |
| B | $3\times10^{-2}$ | 3 | 9.50e-01 | 9.50e-01 | 9.50e-01 | 2.96e-02 | 6.1 | 0.00 | 0.81 | 3.0 | 18 | 16 | 6e-08 |
| B | $10^{-2}$ | 5 | 5.83e-01 | 5.83e-01 | 5.83e-01 | 8.71e-03 | 15.6 | 0.00 | 1.24 | 7.8 | 18 | 16 | 5e-07 |
| B | $3\times10^{-3}$ | 5 | 2.35e-01 | 2.35e-01 | 2.35e-01 | 2.68e-03 | 24.6 | 0.00 | 1.97 | 12.3 | 18 | 19 | 1e-06 |
| B | $10^{-3}$ | 5 | 9.48e-02 | 9.48e-02 | 9.48e-02 | 9.96e-04 | 29.4 | 0.00 | 2.35 | 14.7 | 18 | 20 | 1e-06 |
| B | $3\times10^{-4}$ | 7 | 2.96e-02 | 2.97e-02 | 2.97e-02 | 3.00e-04 | 40.0 | 0.00 | 2.28 | 20.0 | 18 | 21 | 2e-06 |
| B | $10^{-4}$ | 7 | 9.73e-03 | 9.76e-03 | 9.76e-03 | 9.77e-05 | 70.6 | 0.00 | 4.03 | 35.3 | 20 | 21 | 6e-07 |
| B | $3\times10^{-5}$ | 9 | 2.94e-03 | 2.95e-03 | 2.95e-03 | 2.94e-05 | 77.5 | 0.00 | 3.45 | 38.8 | 22 | 19 | 3e-06 |
| B | $10^{-5}$ | 11 | 6.63e-04 | 6.70e-04 | 6.66e-04 | 6.63e-06 | 182.2 | 4.40 | 6.55 | 91.1 | 24 | 20 | 2e-05 |
| B | $3\times10^{-6}$ | 11 | 2.29e-04 | 4.16e-04 | 2.56e-04 | 2.29e-06 | 215.1 | 3.42 | 7.74 | 107.5 | 24 | 19 | 6e-05 |
| B | $10^{-6}$ | 13 | 8.99e-05 | 9.05e-05 | 9.05e-05 | 8.99e-07 | 151.8 | 0.00 | 4.67 | 75.9 | 26 | 19 | 2e-04 |
| B | $3\times10^{-7}$ | 15 | 2.32e-05 | 2.33e-05 | 2.33e-05 | 2.32e-07 | 184.6 | 0.00 | 4.92 | 92.3 | 26 | 18 | 4e-04 |

The budgets from $10^{-7}$ down are left for the cluster. Here the compile of model B at $10^{-7}$ ($D = 17$) took about 4 min per snap, so 4 to 5 hours, and the deeper ones would take longer. The ladders reach $D = 25$ with floors of $2.1\times10^{-9}$ (B) and $2.6\times10^{-8}$ (A), so model A cannot go below about $10^{-7}$ at this depth.

- T per qubit per unit time falls with time, as the plan expects. At $10^{-6}$ it goes from 146 to 128 for model A and from 98 to 76 for model B between $t = 1$ and 2.
- The shallowest depth that fits is not always the cheapest. Model B at $10^{-5}$ and $3\times10^{-6}$ compiles at $D = 11$ to 182 and 215 T per qubit, while $10^{-6}$ at $D = 13$ needs 152 and $3\times10^{-7}$ at $D = 15$ needs 185. The $D = 11$ cell has large staggered angles, needs edge rotations and leaves less room under the budget. The two-depth compile of the cluster jobs (`--margins 3,30`) takes care of this. The plots draw such points faint, off the line.
- Model B at $3\times10^{-6}$ has a fallback share of 45% by the bound, mostly from two KN stragglers at $\epsilon \approx 7\times10^{-4}$, one of them with a fallback branch $3\times10^{-2}$ away from its success branch. Drawn, it is 11% above its success branch.
- Truncation is still harmless, at most $4\times10^{-4}$ relative at a 100 times smaller cutoff.

### The baseline, rebuilt

The old scan (orders 2 and 4, the two KN budgets around $\epsilon^*$) had two problems. Its front started at $c \approx 3\times10^{-4}$ for model B and $3\times10^{-3}$ for model A at $t = 1$, so the loose half of the sweep had nothing to compare against. And $\epsilon^*$ is pessimistic for formulas with many distinct angles. Order 4 at $r = 1$ reaches $C = 1.8\times10^{-7}$ at $\epsilon = 3\times10^{-4}$ for model B, where the estimate predicts $5\times10^{-6}$ and the old scan used $3\times10^{-5}$.

Scan version 3 (`baseline.py`) adds order 1, tries the two grid budgets above $\epsilon^*$ instead of one, and at $r = 1$ every looser budget up to $10^{-1}$. That makes the baseline cheaper at equal success-branch infidelity, by 15% to 20% near $10^{-7}$ for model B at $t = 1$ and much more at the loose end. It keeps a second front ranked by the per-site infidelity with the fallback bound (`bound_per_site`), and each budget gets the point of each front nearest it.

### Drawing the fallbacks

At loose KN budgets the success branch is far better than $\epsilon$, while the fallback branches use their whole share of it. By the bound, the fallbacks are then 30% to 97% of most baseline points' error, against under 10% for most SNAP points. At $N = 100$ a baseline circuit has thousands of KN rotations, and the median one falls back 1.6% of the time, so nearly every run takes dozens of fallbacks. Neither the success branch nor the bound settles what the circuit does.

`snapevo sample` (`fallbacks.py`) draws them. Each KN rotation of the 100-site circuit falls back on its own with probability $1 - p$, the circuit runs with the fallback unitary there, and the infidelity is averaged over 4 draws for each of the 8 inputs. That is every rotation of the baseline, and the stragglers and edge rotations of SNAP. The library gadgets' own fallbacks stay a bound on top, a small one. With no fallbacks drawn the sampled circuit reproduces the evaluated one to $10^{-12}$ for both cells, with and without edges (`tests/test_fallbacks.py`).

Over the 82 baseline points at $t = 1$ and 2 the drawn infidelity is a median 3% above the success branch and at most 4.9 times above it, where the bound is a median 3.6 and at most 41 times above it. For the 17 SNAP points with KN rotations the drawn value is at most 11% above the success branch, where the bound is up to 1.8 times above it. So the success branch is a fair guide for both methods, and the bound overcharges the baseline badly.

### SNAP against the baseline

Each cell is the baseline's T per qubit at the SNAP point's per-site infidelity, interpolated log-log along the lower envelope of the evaluated baseline points, and in brackets its ratio to SNAP. The three pairs of columns compare success branches, both with the fallback bound, and both with the fallbacks drawn. A dash means the SNAP point is outside the baseline's range.

$t = 1$:

| model | budget | SNAP T/q | $c$, success | baseline | $c$, bound | baseline | $c$, drawn | baseline |
|---|---|---|---|---|---|---|---|---|
| A | $10^{-1}$ | 0.0 | 9.77e-02 | - | 9.61e-02 | - | 9.61e-02 | - |
| A | $3\times10^{-2}$ | 9.0 | 2.72e-02 | 13.5 (1.51x) | 2.71e-02 | - | 2.71e-02 | 13.6 (1.51x) |
| A | $10^{-2}$ | 16.3 | 8.14e-03 | 21.7 (1.33x) | 8.13e-03 | 79.9 (4.92x) | 8.13e-03 | 22.5 (1.38x) |
| A | $3\times10^{-3}$ | 20.7 | 2.48e-03 | 67.8 (3.28x) | 2.48e-03 | 120.6 (5.83x) | 2.48e-03 | 70.9 (3.43x) |
| A | $10^{-3}$ | 37.6 | 6.17e-04 | 98.3 (2.62x) | 6.17e-04 | 134.2 (3.57x) | 6.17e-04 | 114.4 (3.04x) |
| A | $3\times10^{-4}$ | 46.8 | 2.86e-04 | 115.9 (2.48x) | 2.86e-04 | 145.7 (3.12x) | 2.86e-04 | 123.5 (2.64x) |
| A | $10^{-4}$ | 66.2 | 9.69e-05 | 129.5 (1.96x) | 9.76e-05 | 243.4 (3.68x) | 9.76e-05 | 151.0 (2.28x) |
| A | $3\times10^{-5}$ | 83.5 | 2.81e-05 | 199.8 (2.39x) | 2.82e-05 | 332.7 (3.98x) | 2.82e-05 | 214.6 (2.57x) |
| A | $10^{-5}$ | 103.3 | 1.01e-05 | 293.4 (2.84x) | 1.02e-05 | 459.5 (4.45x) | 1.02e-05 | 321.0 (3.11x) |
| A | $3\times10^{-6}$ | 119.7 | 2.99e-06 | 478.8 (4.00x) | 3.01e-06 | 813.9 (6.80x) | 3.01e-06 | 480.3 (4.01x) |
| A | $10^{-6}$ | 146.8 | 9.64e-07 | 685.4 (4.67x) | 9.68e-07 | 1053.4 (7.18x) | 9.68e-07 | 701.6 (4.78x) |
| A | $10^{-6}$ | 146.2 | 9.64e-07 | 685.4 (4.69x) | 1.12e-06 | 1025.0 (7.01x) | 9.96e-07 | 689.8 (4.72x) |
| A | $3\times10^{-7}$ | 175.4 | 3.09e-07 | 847.2 (4.83x) | 3.11e-07 | 1095.0 (6.24x) | 3.11e-07 | 957.5 (5.46x) |
| A | $10^{-7}$ | 206.6 | 9.00e-08 | 1035.9 (5.01x) | 1.09e-07 | 1155.5 (5.59x) | 9.08e-08 | 1069.7 (5.18x) |
| A | $10^{-7}$ | 262.2 | 9.77e-08 | 1023.6 (3.90x) | 1.38e-07 | 1141.3 (4.35x) | 9.93e-08 | 1060.2 (4.04x) |
| A | $3\times10^{-8}$ | 275.2 | 2.88e-08 | 1218.9 (4.43x) | 4.04e-08 | 1389.5 (5.05x) | 2.92e-08 | 1218.4 (4.43x) |
| A | $10^{-8}$ | 317.7 | 9.45e-09 | 1460.7 (4.60x) | 1.44e-08 | - | 9.70e-09 | 1464.1 (4.61x) |
| B | $10^{-1}$ | 0.5 | 8.80e-02 | - | 8.73e-02 | - | 8.73e-02 | - |
| B | $3\times10^{-2}$ | 5.4 | 2.50e-02 | - | 2.50e-02 | - | 2.50e-02 | - |
| B | $10^{-2}$ | 9.9 | 8.24e-03 | 13.7 (1.38x) | 8.23e-03 | - | 8.23e-03 | 13.8 (1.38x) |
| B | $3\times10^{-3}$ | 17.8 | 2.18e-03 | 19.1 (1.08x) | 2.18e-03 | 37.0 (2.08x) | 2.18e-03 | 19.5 (1.10x) |
| B | $10^{-3}$ | 18.6 | 8.76e-04 | 24.0 (1.29x) | 8.76e-04 | 49.3 (2.65x) | 8.76e-04 | 24.9 (1.34x) |
| B | $3\times10^{-4}$ | 30.6 | 2.92e-04 | 42.6 (1.39x) | 2.93e-04 | 86.1 (2.81x) | 2.93e-04 | 44.4 (1.45x) |
| B | $10^{-4}$ | 36.2 | 9.58e-05 | 55.3 (1.52x) | 9.66e-05 | 100.1 (2.76x) | 9.66e-05 | 75.0 (2.07x) |
| B | $3\times10^{-5}$ | 54.3 | 1.56e-05 | 84.4 (1.55x) | 1.57e-05 | 145.3 (2.68x) | 1.57e-05 | 119.9 (2.21x) |
| B | $10^{-5}$ | 68.2 | 8.64e-06 | 110.5 (1.62x) | 8.72e-06 | 174.6 (2.56x) | 8.72e-06 | 138.1 (2.02x) |
| B | $3\times10^{-6}$ | 78.4 | 2.83e-06 | 192.4 (2.45x) | 2.86e-06 | 218.9 (2.79x) | 2.86e-06 | 192.3 (2.45x) |
| B | $10^{-6}$ | 96.2 | 8.63e-07 | 242.2 (2.52x) | 8.66e-07 | 287.1 (2.98x) | 8.66e-07 | 242.6 (2.52x) |
| B | $10^{-6}$ | 98.4 | 9.74e-07 | 240.6 (2.45x) | 9.77e-07 | 283.7 (2.88x) | 9.77e-07 | 241.1 (2.45x) |
| B | $3\times10^{-7}$ | 173.4 | 2.85e-07 | 254.4 (1.47x) | 2.94e-07 | 319.4 (1.84x) | 2.86e-07 | 257.1 (1.48x) |
| B | $3\times10^{-7}$ | 195.5 | 2.95e-07 | 254.1 (1.30x) | 3.16e-07 | 317.1 (1.62x) | 2.98e-07 | 256.6 (1.31x) |
| B | $10^{-7}$ | 216.2 | 9.60e-08 | 274.0 (1.27x) | 9.84e-08 | 554.3 (2.56x) | 9.63e-08 | 312.3 (1.44x) |
| B | $10^{-7}$ | 141.8 | 8.69e-08 | 276.4 (1.95x) | 8.74e-08 | 557.8 (3.94x) | 8.74e-08 | 320.3 (2.26x) |
| B | $3\times10^{-8}$ | 265.4 | 2.99e-08 | 560.7 (2.11x) | 4.03e-08 | 581.2 (2.19x) | 3.03e-08 | 561.8 (2.12x) |
| B | $10^{-8}$ | 175.1 | 9.98e-09 | 585.1 (3.34x) | 1.11e-08 | 622.5 (3.55x) | 1.01e-08 | 588.3 (3.36x) |

$t = 2$:

| model | budget | SNAP T/q | $c$, success | baseline | $c$, bound | baseline | $c$, drawn | baseline |
|---|---|---|---|---|---|---|---|---|
| A | $10^{-1}$ | 3.5 | 9.69e-02 | 15.2 (4.35x) | 9.60e-02 | - | 9.60e-02 | - |
| A | $3\times10^{-2}$ | 14.1 | 2.80e-02 | 45.3 (3.22x) | 2.78e-02 | - | 2.78e-02 | 45.4 (3.23x) |
| A | $10^{-2}$ | 24.8 | 8.29e-03 | 57.1 (2.30x) | 8.28e-03 | - | 8.28e-03 | 57.5 (2.32x) |
| A | $3\times10^{-3}$ | 38.5 | 3.01e-03 | 136.7 (3.55x) | 3.01e-03 | 169.4 (4.40x) | 3.01e-03 | 137.2 (3.56x) |
| A | $10^{-3}$ | 64.0 | 9.39e-04 | 162.5 (2.54x) | 9.41e-04 | 223.3 (3.49x) | 9.41e-04 | 162.7 (2.54x) |
| A | $3\times10^{-4}$ | 76.1 | 3.11e-04 | 245.0 (3.22x) | 3.14e-04 | 313.7 (4.12x) | 3.14e-04 | 246.5 (3.24x) |
| A | $10^{-4}$ | 93.8 | 9.85e-05 | 253.2 (2.70x) | 9.88e-05 | 462.1 (4.92x) | 9.88e-05 | 353.4 (3.77x) |
| A | $3\times10^{-5}$ | 126.9 | 2.66e-05 | 438.1 (3.45x) | 2.67e-05 | 712.3 (5.61x) | 2.67e-05 | 477.9 (3.77x) |
| A | $10^{-5}$ | 160.7 | 9.57e-06 | 699.5 (4.35x) | 1.24e-05 | 948.3 (5.90x) | 9.96e-06 | 712.1 (4.43x) |
| A | $3\times10^{-6}$ | 199.6 | 2.76e-06 | 891.6 (4.47x) | 3.09e-06 | 1350.5 (6.77x) | 2.78e-06 | 1014.9 (5.09x) |
| A | $10^{-6}$ | 255.7 | 9.97e-07 | 1218.5 (4.77x) | 1.02e-06 | 1685.0 (6.59x) | 1.00e-06 | 1262.0 (4.94x) |
| A | $3\times10^{-7}$ | 290.0 | 2.95e-07 | 1667.3 (5.75x) | 3.08e-07 | 2209.9 (7.62x) | 2.98e-07 | 1678.1 (5.79x) |
| B | $10^{-1}$ | 2.0 | 7.34e-02 | 12.4 (6.22x) | 7.28e-02 | - | 7.28e-02 | - |
| B | $3\times10^{-2}$ | 6.1 | 2.96e-02 | 15.2 (2.49x) | 2.96e-02 | - | 2.96e-02 | - |
| B | $10^{-2}$ | 15.6 | 8.71e-03 | 20.5 (1.32x) | 8.70e-03 | - | 8.70e-03 | 21.0 (1.35x) |
| B | $3\times10^{-3}$ | 24.6 | 2.68e-03 | 30.3 (1.23x) | 2.68e-03 | 82.1 (3.34x) | 2.68e-03 | 31.7 (1.29x) |
| B | $10^{-3}$ | 29.4 | 9.96e-04 | 47.6 (1.62x) | 9.96e-04 | 100.6 (3.42x) | 9.96e-04 | 54.7 (1.86x) |
| B | $3\times10^{-4}$ | 40.0 | 3.00e-04 | 102.9 (2.57x) | 3.01e-04 | 133.4 (3.34x) | 3.01e-04 | 103.4 (2.59x) |
| B | $10^{-4}$ | 70.6 | 9.77e-05 | 151.9 (2.15x) | 9.80e-05 | 209.3 (2.97x) | 9.80e-05 | 152.2 (2.16x) |
| B | $3\times10^{-5}$ | 77.5 | 2.94e-05 | 229.1 (2.95x) | 2.95e-05 | 317.7 (4.10x) | 2.95e-05 | 234.0 (3.02x) |
| B | $10^{-5}$ | 182.2 | 6.63e-06 | 353.3 (1.94x) | 6.71e-06 | 435.5 (2.39x) | 6.66e-06 | 356.4 (1.96x) |
| B | $3\times10^{-6}$ | 215.1 | 2.29e-06 | 474.4 (2.21x) | 4.16e-06 | 535.6 (2.49x) | 2.56e-06 | 514.9 (2.39x) |
| B | $10^{-6}$ | 151.8 | 8.99e-07 | 518.0 (3.41x) | 9.05e-07 | 602.4 (3.97x) | 9.05e-07 | 523.0 (3.45x) |
| B | $3\times10^{-7}$ | 184.6 | 2.32e-07 | 529.9 (2.87x) | 2.33e-07 | 650.8 (3.53x) | 2.33e-07 | 557.1 (3.02x) |

Along SNAP's lower envelope, with the fallbacks drawn, SNAP needs 1.4 to 5.5 times fewer T than the baseline for model A at $t = 1$ and 2.3 to 5.8 times at $t = 2$. For model B it is 1.1 to 3.4 times at $t = 1$ and 1.3 to 3.5 times at $t = 2$, and 3.1 times at $t = 4$ for its one point ($10^{-4}$). The gap grows with time and toward tight budgets. For model B it is smallest at the loose end, where one order-2 step at a loose KN budget is hard to beat. The rows at the shallow depths of model B at $t = 1$ ($3\times10^{-7}$ to $3\times10^{-8}$) are off SNAP's envelope and show the old wall. Comparing success branches gives ratios a median 2% and at most 30% lower than the drawn ones. Comparing bounds overstates SNAP's advantage, by a median 32% and up to 3.6 times.

### Faster synthesis, snapped edges, safer caches

Three changes came out of these runs.

- KN was the slow step at tight budgets. Isaac's `synthesize()` runs `best_protocol` and then repeats the search to rebuild the gate words, which we never use, so `synthesis.py` now calls `best_protocol` directly. Inside the search, each candidate's T count came from `tcount_vetted.synth_1q_exact`, which builds an mpmath matrix for every gate and verifies the circuit twice. A count from pygridsynth's own reduction steps, without the gate objects, matched it on all 2,392 candidates of a test set at 1/15 of the cost. The winner still goes through the vetted synthesis, which is checked against the fast count. Results are unchanged to the last bit, and one synthesis at $10^{-4}$ went from 8 to 15 s down to about 1 s. The cache now also keeps the branch unitaries, which the draws need.
- Many fitted edge angles sit within $10^{-4}$ of 0 or $\pi/2$, and KN took 10 min or more on some of those. After the fit they are now set to the multiple of $\pi/4$ while the fit infidelity rises by less than a tenth of one bulk site's worth in all. That took 11% to 87% of the edge angles in the 14 points with edges and cut the edge cost of model A at $t = 1$, $10^{-7}$ from 4.0 to 0.8 T per qubit, with the 100-site infidelity unchanged. The five earlier points with edges were redone this way.
- Three jobs writing `runs/kn_cache.json` at once could lose entries or, in the worst case, rename a half-written file into place. The cache is now locked with `flock`, and every JSON and MPS file goes through a per-process temporary name. This matters on the cluster, where hundreds of jobs share the cache.

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

## Tests (`pytest`, 35 tests, about 2 min)

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
- ZYZ angles, the edge start point, an edge fit that lowers the open-chain infidelity, and the snaps of edge angles to multiples of $\pi/4$
- The early stop of the training ladder
- Affine extrapolation
- The order-2 Trotter layout
- KN against its README example and the cache, the cache lock across processes, and the per-file claim that keeps two jobs off the same point
- The drawn circuit with no fallbacks against the evaluated one, for the baseline and for both cells with and without edges, and the slot gates against the cell's rotations
- A small compile that meets its budget
