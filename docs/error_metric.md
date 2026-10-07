# Error metric for the unmixed method

This note defines the error we report for a compiled time-evolution circuit, for SNAP and for the baseline alike. We synthesize deterministically (no mixing), and we compute the error of the actual Clifford+T circuits at the full system size. The decisions come from the call of 5 Oct 2026 and from Sohan on 6 Oct.

The reported error is the state infidelity on random product inputs at $N = 100$, computed with matrix product states on an open chain. Nothing is extrapolated from small systems. The code is `evaluate.py`, `mps.py` and `edges.py`.

## The quantity

For an input $|\phi\rangle = |a_1\rangle \otimes \cdots \otimes |a_N\rangle$, with every $|a_j\rangle$ a random point on the Bloch sphere, let $|u\rangle = U|\phi\rangle$ be the output of the compiled circuit and $|v\rangle = e^{-iHt}|\phi\rangle$ the exact output. We compute

$$
F_k = |\langle v_k | u_k \rangle|^2
$$

for 8 inputs and report the infidelity $1 - \bar F$, its statistical error, and the per-site value

$$
c = 1 - \overline{F_k^{1/N}} ,
$$

which is defined like the training $C$. The per-site value is just another way of writing the same number, $F \approx (1-c)^N$. Comparing $c$ at $N = 100$ with $C$ from the 10-site ring checks that the per-site error really carries over.

## Why an open chain

The cells are trained on a ring, but we evaluate on a 100-site open chain. The reason is entanglement. Random product states are infinite-temperature-like inputs, and their entanglement grows linearly in $t$. On a ring every half of the system has two cuts, so the bond dimension is roughly the square of the open chain's. We measured it at $N = 20$ with exact state vectors (spin units, both models behave alike):

| $t$ | chain, one cut: $S$ (nats) | $\chi$ for $10^{-10}$ | ring, two cuts: $\chi$ for $10^{-10}$ |
|---|---|---|---|
| 2 | 0.52 | 12 | 83 |
| 4 | 1.02 | 35 | 520 |
| 6 | 1.82 | 87 | 977 (saturated at $N = 20$) |

On a 100-site ring the two cuts are independent, so at $t = 6$ the ring needs about $87^2 \approx 7500$ and at $t = 10$ about $10^5$. The open chain needs a few hundred at $t = 10$ (`docs/validation.md` has the measured values at $N = 100$). So the ring stops at about $t = 4$, and the chain reaches $t = 10$.

## Edges

On the open chain, the parity-1 sublayers lose the wrap bond $(N-1, 0)$. By default we keep the wrap gate's single-qubit factor, $R_z(d) \otimes R_z(-d)$ or $R_x(e) \otimes R_x(f)$, and drop its coupling. That makes an open-chain product formula exact at the ends, and it is all model A needs. For model A at $D = 7$ and $11$ the boundary costs at most 2.4 bulk sites' worth of infidelity.

The zero-field cells are different once their staggered angles are large. They lean on the wrap bond, and at $D = 11$ the open chain picks up a boundary error of $4\times10^{-4}$, against $1.1\times10^{-6}$ per site in the bulk. We fix it with edge rotations. Every lost wrap gate is replaced by a free $U = R_z(\alpha) R_y(\beta) R_z(\gamma)$ on each end site, six angles per parity-1 sublayer. We fit them with the bulk fixed, on an open chain of $N_{\rm fit}$ sites (12 to 18, growing with $t$), using exact gradients. At $D = 11$ this removes the boundary error. The open chain then has a lower infidelity than the ring at every size we tried ($N = 8$ to 16).

We keep the fitted edges only when they remove at least one bulk site's worth of error at $N_{\rm fit}$. Otherwise the cell's own factor stays, at no extra cost. Kept angles are synthesized with KN non-mixed at $\min(10^{-4}, 0.1\sqrt{B})$, so their synthesis error never matters. They cost about 17 T each, and there are 6 per parity-1 sublayer in total, not per qubit. That adds about $51 D/N$ T per qubit, so about $0.5 D$ at $N = 100$.

## How we compute it

Both states are MPSs on the open chain (`mps.py`). A two-qubit gate on neighbours is followed by an SVD that keeps the fewest singular values whose discarded weight is at most $10^{-12}$, with at most 1024 of them, and renormalizes. The reference $e^{-iHt}$ is the order-4 Suzuki product formula with step $0.2$, run through the same code. Against exact evolution at $N = 12$, its error is below $3\times10^{-11}$ per site at $t = 10$, so below $3\times10^{-9}$ at $N = 100$.

Truncation. Each MPS carries the sum of its discarded weights, $\epsilon_u$ and $\epsilon_v$. The usual TEBD estimate puts each state at a trace distance of about $\sqrt{\epsilon}$ from its untruncated version. If both errors lined up with the circuit's own error, $\sqrt{1 - F}$ would move by $\sqrt{\epsilon_u} + \sqrt{\epsilon_v}$. We report that pessimistic bar, and we also redo input 0 at a cutoff 100 times smaller and report how much its infidelity changes. The second number is the real test. A point is flagged when either is above 5%. At the smoke point the bar is 0.1% and the change is below $10^{-4}$.

A reference depends only on the model, the time and the input, so it is cached (`runs/<model>/t<t>/ref/`) and shared by every SNAP and baseline circuit at that time. The references dominate the cost from $t = 8$ on.

## What a realized rotation is

Every rotation of a compiled circuit ends up as one of the following:

1. **A Clifford or T angle** ($\theta \in \tfrac{\pi}{4}\mathbb{Z}$). Exact, 0 or 1 T.
2. **A library gadget** (corrected RUS file, `docs/cost_model.md`). On success it applies $R_z(\theta)$ exactly. On failure it applies a known $R_z(\theta_f)$. If $\theta_f \in \tfrac{\pi}{4}\mathbb{Z}$ we undo it exactly and retry. Otherwise the correction is synthesized with KN at fallback cost $F$, which is approximate, with diamond error $\delta_{\rm fb}(F)$.
3. **A straggler**, synthesized with KN non-mixed at diamond budget $\epsilon$. Its success branch (probability $p$) is exactly $R_z(\theta_s)$ with $|\theta_s - \theta| \le 0.9\,\epsilon$. Its fallback branch (probability $1-p$) is a gridsynth word times a known diagonal.

The MPS runs the circuit with every rotation on its success branch. Each instance $i$ also has a heralded fallback that runs with probability $q_i$ per use ($1-p$ single shot, $(1-p)/p$ for a repeat gadget with a 2+T correction) and has diamond error $\delta_i$ relative to the success branch. The success probabilities do not depend on the input, since $K_0^\dagger K_0 = p\,\mathbb{1}$ for every gadget.

## Fallbacks

Fidelity is linear in the channel, so a fallback cannot be absorbed into the success branch. We bound it. With $L = \sum_i q_i \delta_i$ and $Q_2 = \tfrac12 \sum_i q_i \delta_i^2$ over all instances (bulk and edges),

$$
1 - F(\mathcal{E}) \le \left( \sqrt{1 - F_s} + \frac{L}{2} \right)^2 + Q_2 .
$$

The proof is short. The pure-state trace distance $\sqrt{1 - F}$ is a metric, and a fallback at instance $i$ moves the output by at most $\delta_i/2$ in it. So for a branch pattern $b$, $\sqrt{1 - F_b} \le \sqrt{1 - F_s} + X_b$, with $X_b$ the sum of $\delta_i/2$ over the fallbacks it runs. The fallbacks of different instances are independent, $\mathbb{E}X = L/2$, and $\mathrm{Var}\,X \le Q_2$ because every library gadget has $p \ge 0.9$, so the variance of a fallback count is at most $2q_i$. Averaging over patterns gives the bound, and Jensen's inequality carries it to the mean over inputs.

The library is priced on $F \in \{18, 20, \dots, 30\}$. A larger $F$ only shrinks the library's part of $L$, and costs about $(1-p)\,\Delta F$ more T per gadget, a few tenths of a T. We keep the smallest $F$ whose total is within 1% of the total at $F = 30$. What remains comes mostly from the KN stragglers, and we report it as the fallback share of the total.

The term is worst case, since it lets every fallback line up with the coherent error. It is linear in $L$ while the coherent infidelity is quadratic in the errors, so its share grows at tight budgets. It is 1% at the smoke point, 13% for model A at $t = 1$ and $B = 10^{-6}$, and about 10% for the KN baseline, whose fallbacks get a tenth of $\epsilon$.

## Steps for one point

We run the steps as follows:

1. **Compile.** The compiler works on the training $C$ at $N = 10$ on the ring, over 12 random product states. The budget $B$ on $C$ is a knob. It is not the reported error.
2. **Realize.** We synthesize the stragglers with KN at the precision the compiler allotted, check $C \le B$ on the training set (halving every straggler $\epsilon$ if not), then fit and synthesize the edges.
3. **Evaluate.** MPS at $N = 100$ on 8 inputs, the fallback bound, the choice of $F$.
4. **Report.** Each point carries $1 - \bar F$ at $N = 100$ with and without fallbacks, its statistical and truncation error bars, $c$, the ring $C$ for comparison, and T per qubit, T per rotation and T per qubit per unit time. `snapevo report` prints them side by side.

The baseline is a product formula (order 1, 2 or 4, $r$ steps) with every non-Clifford rotation synthesized by KN non-mixed at budget $\epsilon$ (`baseline.py`). Every distinct angle is synthesized once and every copy uses the same circuit, so the synthesis errors add coherently, which the direct computation captures. `baseline-scan` screens $(\text{order}, r, \epsilon)$ by ring $C$ at $N = 10$ and runs the MPS evaluation only on the Pareto front. For each order and $r$ it tries the first grid budget at or below the coherent estimate $\epsilon^*$, where $(0.45 R \epsilon^*)^2$ is a third of the Trotter $C$ for $R$ rotations per qubit, and the two budgets above it. At $r = 1$ it tries every looser budget up to $10^{-1}$ as well, so the front reaches the loose targets. The estimate is pessimistic when a formula has many distinct angles, whose errors partly cancel. Order 4 at $r = 1$, for example, reaches $C = 1.8 \times 10^{-7}$ at $\epsilon = 3 \times 10^{-4}$, where the estimate predicts about $5 \times 10^{-6}$.

## Checks on every point

1. The realized training $C$ is at most the budget.
2. The statistical error on $1 - \bar F$ is below 5%, and so are the truncation bar and the change at a 100 times smaller cutoff.
3. $c$ at $N = 100$ is within 10% of the ring's held-out $C$. A larger gap means a boundary problem.
4. The fallback share is at most 10%. Above that, the bound rather than the circuit sets the reported number, and sampling the fallbacks (open point 3) is worth doing.

## Small-ring cross-checks

Before the MPS runs we measured the success branch on small rings and extrapolated to $N = 100$. That code is now `crosscheck.py` (`snapevo crosscheck`), kept as a check at short times. It computes three numbers:

1. **Operator norm and diamond norm.** If the eigenphases of $V^\dagger U_s$ span an arc of width $w$, then $\epsilon_{\rm op} = 2\sin(w/4)$ and $D_{\rm coh} = 2\sin(w/2)$. Lanczos gives $w$ exactly for $N \le 16$, and we extrapolate it to $N = 100$.
2. **State infidelity** $1 - (1 - C)^N$ from the held-out $C$.
3. **Process infidelity** $1 - |\mathrm{tr}(V^\dagger U_s)/d|^2$, exact up to $N = 10$ and from 16 Haar states above.

At the smoke point (model B, $t = 2$, $D = 7$) these give $D_{\rm total}(100) = 1.86$, state infidelity $1.03\times10^{-2}$ and process infidelity $1.05\times10^{-2}$ on the ring. The MPS gives $9.73\times10^{-3}$ on the open chain. The worst-case and average-case numbers scale as

$$
1 - F(N) \approx N C, \qquad D_{\rm coh}(N) \approx 2 \sin\!\left(N\sqrt{C}\right),
$$

so a 1% error at $N = 100$ needs $C \approx 10^{-4}$ in fidelity but $C \approx 2.5\times10^{-9}$ in operator norm.

## Open points for the team

1. **David's Hamiltonian and geometry.** The open chain matches what an MPS code for the dynamical structure factor normally uses. If the end-to-end run needs a ring after all, $t$ stops at about 4 for random product inputs.
2. **Mixed synthesis.** We leave it out and note it in the text as a further factor of about 2 in operator norm. It cannot help the fidelity, which is linear in the channel (Isaac, 1 Oct).
3. **How loose is the fallback term?** Sampling the gadget outcomes in the MPS runs would replace the bound by the actual average. Each sample is one more circuit MPS, cheap next to the reference. It matters for the baseline and for SNAP at tight budgets, where the fallback share reaches 10% or more.
4. **Cheaper edges.** The edge angles go to KN at about 17 T each. Snapping them to the library would cut that by about 4x.
