# Error metric for the unmixed method

This note defines the error we report for a compiled time-evolution circuit, for SNAP and for the baseline alike. It follows the decisions of the 5 Oct 2026 call. We synthesize deterministically (no mixing), we measure errors directly by plugging in the actual Clifford+T circuits, and we use the operator norm as the physical measure. The code is `realize.py` and `metrics.py`.

## What a realized rotation is

Every rotation of a compiled circuit ends up as one of the following:

1. **A Clifford or T angle** ($\theta \in \tfrac{\pi}{4}\mathbb{Z}$). Exact, 0 or 1 T.
2. **A library gadget** (corrected RUS file, `docs/cost_model.md`). On success it applies $R_z(\theta)$ exactly. On failure it applies a known $R_z(\theta_f)$. If $\theta_f \in \tfrac{\pi}{4}\mathbb{Z}$ we undo it exactly and retry. Otherwise the correction is synthesized with KN at fallback cost $F$, which is approximate, with diamond error $\delta_{\rm fb}(F)$.
3. **A straggler**, synthesized with KN non-mixed at diamond budget $\epsilon$. Its success branch (probability $p$) is exactly $R_z(\theta_s)$ with $|\theta_s - \theta| \le 0.9\,\epsilon$. Its fallback branch (probability $1-p$) is a gridsynth word $B$ times a known diagonal $F$.

Each rotation instance $i$ therefore has a success unitary $R_i^{s}$, which is an exact $z$ rotation, and a heralded fallback unitary $R_i^{f}$ that runs with probability $q_i$ per use. For gadgets, $q_i$ is $1-p$ (single shot) or $(1-p)/p$ (repeat with a 2+T correction). The success probabilities do not depend on the input state, since $K_0^\dagger K_0 = p\,\mathbb{1}$ for every gadget. So the realized operation is a fixed channel

$$
\mathcal{E} = \sum_b P_b\, \mathcal{U}_b ,
$$

a mixture over branch patterns $b$, where $\mathcal{U}_b$ is the unitary channel of the circuit with fallbacks at the rotations in $b$.

## The bound we report

Let $U_s$ be the circuit with every rotation on its success branch, and $V = e^{-iHt}$. For a pattern $b$ with fallbacks at the set $F_b$, unitary invariance and the triangle inequality give $\| \mathcal{U}_b - \mathcal{U}_s \|_\diamond \le \sum_{i \in F_b} \delta_i$ with $\delta_i = \| \mathcal{R}_i^f - \mathcal{R}_i^s \|_\diamond$. Averaging over patterns,

$$
\| \mathcal{E} - \mathcal{V} \|_\diamond  \le  \underbrace{\| \mathcal{U}_s - \mathcal{V} \|_\diamond}_{D_{\rm coh}(N)}  +  \underbrace{\sum_i q_i\, \delta_i}_{N \ell}  =  D_{\rm total}(N).
$$

The second term is exact bookkeeping. Each instance contributes its expected number of fallbacks times their error, and nothing about it is extrapolated. Per qubit per circuit the ledger is

$$
\ell = \tfrac12 \sum_{\text{angles } i} m_i\, q_i\, \delta_i ,
$$

with $m_i$ the Rz rotations the angle costs per bond. For library gadgets $\delta_i = \delta_{\rm fb}(F)$ from the calibrated KN law. For KN stragglers $\delta_i = \| \mathcal{U}_1 - \mathcal{U}_0 \|_\diamond$, computed from Isaac's returned unitaries.

We use the full diamond norm, with values in $[0, 2]$. For any observable, $|\mathrm{tr}\,O(\mathcal{E}(\rho) - \mathcal{V}(\rho))| \le \|O\|\, D_{\rm total}$.

## The coherent part

Let the eigenphases of $V^\dagger U_s$ span an arc of width $w$ once the global phase is removed. Then

$$
\epsilon_{\rm op} = \min_\phi \| U_s - e^{i\phi} V \| = 2 \sin\frac{w}{4}, \qquad
D_{\rm coh} = \| \mathcal{U}_s - \mathcal{V} \|_\diamond = 2 \sin\frac{w}{2} \quad (w < \pi).
$$

We compute $w$ exactly with Lanczos on $G = \mathrm{Im}(e^{-i\phi_0} V^\dagger U_s)$. Its extreme eigenvalues are the sines of the arc ends, and every matrix-vector product is one circuit application plus one Krylov exponential. This is matrix free and runs at $N = 6, 8, \dots, 16$ routinely, and at 18 for selected points. A test checks it against full diagonalization at $N \le 8$.

We need $N = 100$, so we extrapolate $w$, not $D$. If $V^\dagger U_s \approx e^{-iK}$ with $K = \sum_j k_j$ a sum of local error terms, then $w = \lambda_{\max}(K) - \lambda_{\min}(K)$ grows linearly with $N$. $D_{\rm coh} = 2\sin(w/2)$ then saturates at 2, which is the right behavior for a fixed error per site on a large ring. We fit the three largest sizes two ways, affine $w = \alpha N + \beta$ and power law $w = c N^{a}$, and report the larger prediction at $N = 100$. Both fits are stored, so the spread between them is visible for every point.

### What the smoke run shows

The smoke run used model B, $t = 2$, $D = 7$ and held-out $C = 1.0\times10^{-4}$. There the arc width per site, $\alpha = w/N$, is flat from $N = 8$ on: $0.0200, 0.0192, 0.0202, 0.0198$ at $N = 8, 10, 12, 14$. It sits at $\alpha \approx 1.95\sqrt{C}$. So for these circuits

$$
D_{\rm coh}(N) \approx 2 \sin\!\left(N\sqrt{C}\right),
$$

which is the square-root relation mentioned at the meeting, now with its constant. The consequence is large. At $N = 100$, $D_{\rm coh} \le 0.1$ needs $C \lesssim 2.5\times10^{-7}$, and $D_{\rm coh} \le 0.01$ needs $C \lesssim 2.5\times10^{-9}$. The Sep 2026 budgets, $C = 10^{-4}$ to $10^{-7}$, map to $D_{\rm coh}(100)$ from 1.9 (saturated) down to about 0.06. The intensive number $\alpha$ does not depend on $N$, so we report it with every point next to $D_{\rm total}(100)$. For any ring size, $D_{\rm total}(N) \approx 2\sin(\alpha N/2) + N\ell$.

## How the budget is spent

We use the steps as follows:

1. **Compile.** The compiler works on the training local infidelity at $N = 10$, $C = 1 - \frac{1}{n}\sum_k |\langle \psi_k | U | \phi_k \rangle|^{2/N}$ over 12 random product states. The budget $B$ on $C$ is a knob. It is not the reported error.
2. **Realize.** We synthesize the stragglers with KN at the precision the compiler allotted. The success-branch circuit then has to satisfy $C \le B$ on the training set. If it does not, every straggler $\epsilon$ is halved and the stragglers are synthesized again.
3. **Measure.** We compute held-out $C$ at $N = 10, 12$ (four disjoint sets), $w(N)$ for $N = 6$ to $16$, and the extrapolation to $N = 100$.
4. **Pick $F$.** The library prices are recomputed on $F \in \{18, 20, \dots, 30\}$. We keep the smallest $F$ whose ledger at $N = 100$ is at most a tenth of $D_{\rm total}(100)$. A larger $F$ costs about $(1-p)\,\Delta F$ more T per gadget, a few tenths of a T.
5. **Report.** Each point is plotted at its measured $D_{\rm total}(100)$, with T per rotation and T per qubit per unit time on the other axis. Held-out $C$, $D_{\rm coh}$, the ledger and both fits go in the table.

## The baseline is measured the same way

The baseline is a product formula (order 1, 2 or 4, $r$ steps) with every non-Clifford rotation synthesized by KN non-mixed at budget $\epsilon$ (`baseline.py`). Every distinct angle is synthesized once, and every copy uses the same circuit. That is what deterministic synthesis means in a translation-invariant circuit, and it makes the synthesis errors add coherently over sites and steps. The Sep 2026 comparison assumed incoherent addition, with per-rotation precision $\sqrt{\epsilon/R}$, which flatters the baseline. Direct measurement removes that assumption for both methods. The baseline curve is the lower envelope over $(\text{order}, r, \epsilon)$ of measured points. The calibrated KN law is used only to choose which points to measure.

## Checks on every point

1. The realized training $C$ is at most the budget.
2. The held-out sets agree to within 5%, and $C$ at $N = 12$ is within 3% of $N = 10$ (as in Sep 2026).
3. The affine and power-law predictions at $N = 100$ agree to within a factor of 2. Otherwise we add $N = 18$ before quoting the point.
4. The ledger is at most 10% of $D_{\rm total}(100)$ at the chosen $F$.

## Open points for the team

1. **Is $D_{\rm total}(100)$ the right headline?** For DSF we need local two-point functions. By Lieb-Robinson their error depends only on the gates in the light cone, so it does not grow with $N$. The smoke run makes the gap concrete. A circuit with $10^{-4}$ infidelity per site is useless in global operator norm at $N = 100$ ($D_{\rm coh} \approx 1.9$), while its local error is small. We report three numbers: $\alpha$ (intensive operator-norm rate), $D_{\rm total}(100)$, and held-out $C$. The paper has to pick one as the x axis.
2. **Which accuracy range matters?** The meeting argued that about 1% per site is enough for DSF on NISQ hardware, and that a 1% total budget at $N = 100$ leaves 10 to 20% of the rotations Clifford. A 1% budget in $D_{\rm total}(100)$ means $C \approx 2.5\times10^{-9}$. At $t = 2$ the zero-field cells floor at about $2\times10^{-9}$ by $D = 27$ (Sep 2026 ladder), so that target sits at the edge of what this cell family reaches, and in the region where the Sep runs hit the wall. A local target allows budgets around $10^{-4}$ to $10^{-6}$, where SNAP does best.
3. **Mixed synthesis.** We leave it out of the comparison and note it in the text as a further factor of about 2. If it comes back, the dephasing part enters the ledger exactly as above, as $\tfrac12|\Delta_1 \Delta_2|$ per rotation.
