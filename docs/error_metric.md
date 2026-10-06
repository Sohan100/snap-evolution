# Error metrics for the unmixed method

This note defines the errors we report for a compiled time-evolution circuit, for SNAP and for the baseline alike. It follows the decisions of the 5 Oct 2026 call. We synthesize deterministically (no mixing) and we measure errors directly by plugging in the actual Clifford+T circuits. We report two kinds of error for every point. The worst case is the operator norm and the diamond norm. The average case is the fidelity, on product states and as a process fidelity. The code is `realize.py` and `metrics.py`.

## What a realized rotation is

Every rotation of a compiled circuit ends up as one of the following:

1. **A Clifford or T angle** ($\theta \in \tfrac{\pi}{4}\mathbb{Z}$). Exact, 0 or 1 T.
2. **A library gadget** (corrected RUS file, `docs/cost_model.md`). On success it applies $R_z(\theta)$ exactly. On failure it applies a known $R_z(\theta_f)$. If $\theta_f \in \tfrac{\pi}{4}\mathbb{Z}$ we undo it exactly and retry. Otherwise the correction is synthesized with KN at fallback cost $F$, which is approximate, with diamond error $\delta_{\rm fb}(F)$.
3. **A straggler**, synthesized with KN non-mixed at diamond budget $\epsilon$. Its success branch (probability $p$) is exactly $R_z(\theta_s)$ with $|\theta_s - \theta| \le 0.9\,\epsilon$. Its fallback branch (probability $1-p$) is a gridsynth word $B$ times a known diagonal $F$.

Each rotation instance $i$ therefore has a success unitary $R_i^{s}$, which is an exact $z$ rotation, and a heralded fallback unitary $R_i^{f}$ that runs with probability $q_i$ per use. For gadgets, $q_i$ is $1-p$ (single shot) or $(1-p)/p$ (repeat with a 2+T correction). The success probabilities do not depend on the input state, since $K_0^\dagger K_0 = p\,\mathbb{1}$ for every gadget. So the realized operation is a fixed channel

$$
\mathcal{E} = \sum_b P_b\, \mathcal{U}_b ,
$$

a mixture over branch patterns $b$, where $\mathcal{U}_b$ is the unitary channel of the circuit with fallbacks at the rotations in $b$. Let $U_s$ be the circuit with every rotation on its success branch and $V = e^{-iHt}$. Every metric below splits into a coherent part, measured on $U_s$, and a fallback part, which is exact bookkeeping over the instances. Per qubit per circuit the bookkeeping is

$$
\ell = \tfrac12 \sum_{\text{angles } i} m_i\, q_i\, \delta_i , \qquad \ell_2 = \tfrac12 \sum_{\text{angles } i} m_i\, q_i\, \delta_i^2 ,
$$

with $m_i$ the Rz rotations the angle costs per bond and $\delta_i = \| \mathcal{R}_i^f - \mathcal{R}_i^s \|_\diamond$. For library gadgets $\delta_i = \delta_{\rm fb}(F)$ from the calibrated KN law. For KN stragglers $\delta_i$ comes from Isaac's returned unitaries. On $N$ sites, $L = N\ell$ and $Q_2 = N\ell_2/2$.

## Worst case: operator norm and diamond

For a pattern $b$ with fallbacks at the set $F_b$, unitary invariance and the triangle inequality give $\| \mathcal{U}_b - \mathcal{U}_s \|_\diamond \le \sum_{i \in F_b} \delta_i$. Averaging over patterns,

$$
\| \mathcal{E} - \mathcal{V} \|_\diamond  \le  \underbrace{\| \mathcal{U}_s - \mathcal{V} \|_\diamond}_{D_{\rm coh}(N)}  +  L  =  D_{\rm total}(N).
$$

We use the full diamond norm, with values in $[0, 2]$. For any observable, $|\mathrm{tr}\,O(\mathcal{E}(\rho) - \mathcal{V}(\rho))| \le \|O\|\, D_{\rm total}$.

Let the eigenphases of $V^\dagger U_s$ span an arc of width $w$ once the global phase is removed. Then

$$
\epsilon_{\rm op} = \min_\phi \| U_s - e^{i\phi} V \| = 2 \sin\frac{w}{4}, \qquad
D_{\rm coh} = \| \mathcal{U}_s - \mathcal{V} \|_\diamond = 2 \sin\frac{w}{2} \quad (w < \pi).
$$

We compute $w$ exactly with Lanczos on $G = \mathrm{Im}(e^{-i\phi_0} V^\dagger U_s)$. Its extreme eigenvalues are the sines of the arc ends, and every matrix-vector product is one circuit application plus one Krylov exponential. This runs at $N = 6, 8, \dots, 16$ routinely and at 18 for selected points. A test checks it against full diagonalization at $N \le 8$.

We need $N = 100$, so we extrapolate $w$, not $D$. If $V^\dagger U_s \approx e^{-iK}$ with $K = \sum_j k_j$ a sum of local error terms, then $w = \lambda_{\max}(K) - \lambda_{\min}(K)$ grows linearly with $N$. $D_{\rm coh} = 2\sin(w/2)$ then saturates at 2, which is the right behavior for a fixed error per site on a large ring. We fit the three largest sizes two ways, affine $w = \alpha N + \beta$ and power law $w = c N^{a}$, and report the larger prediction at $N = 100$. Both fits are stored. We also report $\alpha = w/N$ at the largest measured size, which does not depend on $N$.

## Average case: fidelity

We measure two fidelities of the success branch.

1. **State infidelity on product inputs.** For random product states $|\phi_k\rangle$ and exact outputs $|\psi_k\rangle$, the per-site infidelity is $C = 1 - \frac1n\sum_k |\langle\psi_k|U_s|\phi_k\rangle|^{2/N}$, measured on the four held-out sets at $N = 10, 12$ (worst of the 8). The whole-ring infidelity is
$$
1 - F_s(N) = 1 - (1 - C)^N .
$$
At $N = 10$ this is the Sep 2026 "whole-circuit infidelity". We report it there for continuity, and at $N = 100$.
2. **Process infidelity.** This is the entanglement fidelity, the fidelity on one half of a maximally entangled state,
$$
F_e(N) = \left| \tfrac1d \mathrm{tr}\, V^\dagger U_s \right|^2, \qquad d = 2^N ,
$$
and the average gate fidelity $\bar F = (d F_e + 1)/(d+1)$ follows from it. We compute it exactly for $N \le 10$. Above that we use 16 Haar-random states, whose mean fidelity estimates $\bar F$. The estimator agrees with the exact value to within 0.6% at $N = 8$ and 12. The per-site value $c_e = 1 - F_e^{1/N}$ is flat in $N$, so the whole-ring value is $1 - (1 - c_e)^N$.

Both per-site numbers stop depending on $N$ once the ring is longer than the light cone. The error is a sum of local terms, so the log of either fidelity adds over distant regions. The smoke data show it, with $c_e$ flat from $N = 6$ to 14. This is why $C$ from $N = 10, 12$ carries over to $N = 100$ with no fit.

### Fallbacks in the fidelity

Fidelity is linear in the channel, $F(\mathcal{E}) = \sum_b P_b F_b$, so a fallback cannot be absorbed into the success branch the way mixing would require. It has to be bounded. The trace distance between pure states, $\sqrt{1 - |\langle a|b\rangle|^2}$, is a metric. A fallback at instance $i$ moves the output by at most $\delta_i/2$ in that distance. So for every pattern $\sqrt{1 - F_b} \le \sqrt{1 - F_s} + X_b$ with $X_b = \sum_{i\in F_b} \delta_i/2$, where an instance counts once per fallback it runs. The fallbacks of different instances are independent, so $\mathbb{E}X = L/2$ and $\mathrm{Var}\,X = \sum_i (\delta_i/2)^2\, \mathrm{Var}\, n_i$, with $n_i$ the number of fallbacks at instance $i$. A repeat gadget runs a geometric number of them, with variance $q_i(1 + q_i)$, and a single-shot gadget or a KN straggler runs at most one, with variance $q_i(1 - q_i)$. Every library gadget has $p \ge 0.9$, so $q_i \le 1$ and $\mathrm{Var}\,n_i \le 2q_i$ in all cases. That gives $\mathbb{E}X^2 \le (L/2)^2 + Q_2$, and averaging gives

$$
1 - F(\mathcal{E}) \le \left( \sqrt{1 - F_s} + \frac{L}{2} \right)^2 + Q_2 .
$$

This holds for every input state. Averaging over inputs with Jensen's inequality keeps the same form with the mean infidelity, and the same argument with a maximally entangled input gives the process version. `metrics.fidelity_bound` implements it.

## What the smoke run shows

The smoke run used model B, $t = 2$, $D = 7$, held-out $C = 1.02\times10^{-4}$. The results are as follows:

- $\alpha = w/N$ is flat from $N = 8$ on, at $0.0200, 0.0192, 0.0202, 0.0198$ for $N = 8, 10, 12, 14$, and $\alpha \approx 1.95\sqrt{C}$.
- The per-site process infidelity is $c_e = 1.04\times10^{-4}$ at $N = 6$ to $14$, the same as $C$ to within 2%.
- At $N = 100$ with $F = 18$, we get $D_{\rm total} = 1.86$, state infidelity $1.03\times10^{-2}$ and process infidelity $1.05\times10^{-2}$.

So the same circuit is a 1% error on 100 sites by fidelity, and useless by operator norm. The two scale as

$$
1 - F(N) \approx N C, \qquad D_{\rm coh}(N) \approx 2 \sin\!\left(N\sqrt{C}\right),
$$

which is the square-root relation mentioned at the meeting, now with its constant. At $N = 100$, $D_{\rm coh} \le 0.1$ needs $C \lesssim 2.5\times10^{-7}$, and $D_{\rm coh} \le 0.01$ needs $C \lesssim 2.5\times10^{-9}$. A fidelity of 99% at $N = 100$ needs only $C \approx 10^{-4}$, and 99.99% needs $C \approx 10^{-6}$.

## How the budget is spent

We use the steps as follows:

1. **Compile.** The compiler works on the training $C$ at $N = 10$, over 12 random product states. The budget $B$ on $C$ is a knob. It is not the reported error.
2. **Realize.** We synthesize the stragglers with KN at the precision the compiler allotted. The success-branch circuit then has to satisfy $C \le B$ on the training set. If it does not, every straggler $\epsilon$ is halved and the stragglers are synthesized again.
3. **Measure.** We compute held-out $C$ at $N = 10, 12$, and $w(N)$ and $F_e(N)$ for $N = 6$ to $16$, and take both to $N = 100$.
4. **Pick $F$.** The library prices are recomputed on $F \in \{18, 20, \dots, 30\}$. We keep the smallest $F$ at which the fallbacks are at most a tenth of the total, both in $D_{\rm total}(100)$ and in the state infidelity at $N = 100$. A larger $F$ costs about $(1-p)\,\Delta F$ more T per gadget, a few tenths of a T.
5. **Report.** Each point carries $\alpha$, $D_{\rm total}(100)$, the state and process infidelities at $N = 100$ (and the state infidelity at $N = 10$), held-out $C$, and T per rotation and per qubit per unit time. `snapevo report` prints them side by side.

## The baseline is measured the same way

The baseline is a product formula (order 1, 2 or 4, $r$ steps) with every non-Clifford rotation synthesized by KN non-mixed at budget $\epsilon$ (`baseline.py`). Every distinct angle is synthesized once, and every copy uses the same circuit. That is what deterministic synthesis means in a translation-invariant circuit, and it makes the synthesis errors add coherently over sites and steps. The Sep 2026 comparison assumed incoherent addition, with per-rotation precision $\sqrt{\epsilon/R}$, which flatters the baseline. Direct measurement removes that assumption for both methods, in both metrics. The baseline curve is the lower envelope over $(\text{order}, r, \epsilon)$ of measured points, one envelope per metric. The calibrated KN law is used only to choose which points to measure.

## Checks on every point

1. The realized training $C$ is at most the budget.
2. The held-out sets agree to within 5%, and $C$ at $N = 12$ is within 3% of $N = 10$ (as in Sep 2026).
3. The affine and power-law predictions of $w(100)$ agree to within a factor of 2. Otherwise we add $N = 18$ before quoting the point.
4. The per-site process infidelity $c_e$ agrees across the measured sizes to within 5%, and the Haar estimate has a standard error below 2% of the infidelity.
5. At the chosen $F$ the fallbacks are at most 10% of $D_{\rm total}(100)$ and of the state infidelity at $N = 100$.

## Open points for the team

1. **Which metric is the headline?** The candidates are $D_{\rm total}(100)$ (or $\alpha$), the state infidelity at $N = 100$, and the process infidelity at $N = 100$. The smoke run shows they rank the same circuit very differently. For DSF we need local two-point functions. By Lieb-Robinson their error depends only on the gates in the light cone, so it does not grow with $N$, and it should track the fidelity numbers rather than $D_{\rm total}(100)$. The end-to-end DSF run with David can settle this.
2. **Which accuracy range matters?** The meeting argued that about 1% error is enough for DSF on NISQ hardware, and that a 1% total budget at $N = 100$ leaves 10 to 20% of the rotations Clifford. In fidelity, 1% at $N = 100$ is $C \approx 10^{-4}$, the loose end of our local range. In operator norm it is $C \approx 2.5\times10^{-9}$, at the edge of what the zero-field cells reach at $t = 2$ (floor about $2\times10^{-9}$ at $D = 27$) and where the Sep runs hit the wall.
3. **Mixed synthesis.** We leave it out of the comparison and note it in the text as a further factor of about 2 in operator norm. It cannot help the fidelity, which is linear in the channel (Isaac, 1 Oct). If it comes back, the dephasing part enters the ledger as $\tfrac12|\Delta_1 \Delta_2|$ per rotation.
4. **How loose is the fallback term in the fidelity?** The cross term $\sqrt{1 - F_s}\,L$ assumes every fallback error lines up with the coherent error. For SNAP at $F = 18$ the whole fallback term is 1% of the bound, so it does not matter. For the KN baseline it is about 10%, because KN gives its fallbacks a tenth of $\epsilon$ in diamond norm and the cross term turns that into a linear term. A Monte Carlo over branch patterns at $N \le 12$, with the branch unitaries KN returns, would give the actual value.
