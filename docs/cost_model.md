# Cost model

Every T count in this repo comes from one of two vendored packages by Isaac Kim, used unmodified. `third_party/*/SHA256SUMS` records the files as received on 2026-10-05.

## Exact-angle library (RUS gadgets)

`third_party/rz_library_corrected_2026-10-05` holds Isaac's corrected Rz library, which replaces Sohan's Aug 2026 `rz_library.zip`. Its README lists the changes. The ones that matter here are these:

1. `t_exact` is now the T count of one gadget attempt, read off an exactly synthesized circuit. The old column held the denominator exponent $k$, about half the true count.
2. Each gadget has two rows, the cheapest same-angle dressing and the cheapest odd-shift dressing ($\pm\pi/4$). The dressing is $V' = \mathrm{diag}(1, \omega^a)\, V\, \mathrm{diag}(1, \omega^b)$.
3. No catalysis and no $k\pi/12$ angles. The library holds RUS gadgets and the eight $k\pi/4$ anchors only.

The file covers the 2,290 gadgets with denominator exponent $L \le 9$ and success probability $p \ge 0.9$. The $L = 10, 11$ set used by Isaac's Oct 1 cost model is not included.

### Pricing

A gadget with T count $t$, success probability $p$ and correction class $r \in \{\text{Clifford}, \text{T}, \text{2+T}\}$ costs the cheaper of

$$
\text{repeat: } \frac{t}{p} + c_r \frac{1-p}{p}, \qquad \text{single shot: } t + (1-p) F ,
$$

with $c_r = 0, 1, F$. $F$ is the expected T count of the KN synthesis that corrects a 2+T failure (`library.gadget_cost`). Next to the cost we keep $q$, the expected number of approximate KN corrections per use: $(1-p)/p$ for repeat with a 2+T correction, $1-p$ for single shot, 0 otherwise. Each correction has diamond error $\delta_{\rm fb}(F)$. These are the library's entries in the error ledger (`docs/error_metric.md`).

At $F = 18$, 19 gadgets with an exact (Clifford or T) correction come out cheaper single shot, by less than 0.1 T. The pricing keeps that choice and charges its $q$ to the ledger, so the accounting stays consistent. None do at $F \ge 22$.

### Dressing

Each priced row contributes $\pm\theta + k\pi/4$ for $k = 0, \dots, 7$, with one extra T for odd $k$. The eight anchors $k\pi/4$ cost 0 or 1 T. Where several entries land on the same angle, the cheapest wins. At $F = 18$ this lattice equals the shipped `rz_library_cat0_kn18.csv` (27,499 rows), as the tests check.

### What we verified about the file

`python -m snapevo verify-library` rebuilds every row from its ring elements, simulates the gadget, and recomputes the T count by an independent method. `docs/validation.md` has the results. Two format points matter for anyone reusing the file:

1. The file does not store the $\omega$ exponent $n$ of the dressed unitary $V' = \frac{1}{\sqrt2^{L}}\begin{pmatrix} z & -\omega^n y^* \\ y & \omega^n z^* \end{pmatrix}$. Same-angle rows have $n = 0$ and odd-shift rows $n = 1$. Isaac's Oct 1 `certify_rus` fixes $n = 0$ and therefore rejects every odd-shift row.
2. The gadget circuit (CNOT, $V'$, CNOT, measure the ancilla, outcome 0) applies $R_z(-\theta)$ with $\theta$ = `angle_rad`. The sign does not matter for the library, since $R_z(-\theta) = X R_z(\theta) X$.

## KN non-mixed synthesis (stragglers, baseline, fallback corrections)

`third_party/kliuchnikov_nonmixed_2026-10-05` is Isaac's implementation of the fallback protocol of Kliuchnikov, Lauter, Minko, Paetznick and Petit (arXiv:2203.10064), non-mixed. For a target $R_z(\theta)$ and a diamond budget $\epsilon$ it builds the following:

- A projective rotation (CNOT, $V$, CNOT, measure) whose success branch, with probability $p \ge 0.9$, is exactly $R_z(\theta_s)$ with $D(R_z(\theta_s), R_z(\theta)) \le 0.9\,\epsilon$.
- A fallback, applied only on failure, $U_1 = B F$ with $B$ a gridsynth word and $D(U_1, R_z(\theta)) \le 0.1\,\epsilon/(1-p)$.

The expected T count is $T(V) + (1-p) T(B)$. `synthesis.synthesize` calls it, caches the result in `runs/kn_cache.json`, and returns $\theta_s$ and $D(U_1, U_0)$. Calls take a few seconds each.

The package's calibration, at 40 random angles per $\epsilon$ and $q = 0.99$, gives a mean cost of

$$
\bar T(\epsilon) = 1.096 \log_2(1/\epsilon) + 4.44 ,
$$

against $1.03 \log_2(1/\epsilon) + 5.75$ in the paper's Table 1. We use the calibrated law in three places: to plan straggler precision inside the compiler, to set $\delta_{\rm fb}(F) = 2^{-(F - 4.44)/1.096}$ (for example $1.9\times10^{-4}$ at $F = 18$ and $1.5\times10^{-5}$ at $F = 22$), and to choose which baseline points to measure. Every reported straggler and baseline cost comes from an actual circuit.

## What is not in the model

The model leaves out the following:

- catalysis of any kind, so catalyst qubits are 0
- mixed (probabilistic) synthesis
- the $L = 10, 11$ gadgets. The corrected file stops at $L = 9$. We checked that the same dressing rule holds for all 25,550 of the larger gadgets, so a corrected dense library could be built if it turns out to matter.
