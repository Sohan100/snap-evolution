# Per-site infidelity for translation-invariant time evolution in the literature

Survey written 23 Oct 2026. It records how published papers define and compute a per-site (local) fidelity or infidelity for translation-invariant (TI) chains, infinite or periodic, when they simulate or compile $e^{-iHt}$. It reports what the papers do. Our own conventions appear only in Section 4, for comparison.

Sources are the latest arXiv TeX source of each paper (version in the table). Each archive sits in its own directory under `/pscratch/sd/s/sohan100/snapevo-scratch/lit_ti/<arXiv id>/`. Quotes are copied from the TeX with `\cite{}` removed and paper-specific macros (`\thv`, `\ket`, `\Tr`, `\Uvar` and the like) expanded into standard LaTeX; `[...]` marks a cut and `[sic]` a typo in the source. Equation numbers come from the arXiv HTML rendering where it exists (2506.02105, 2409.16346, 2409.16361, 2312.14245, 2501.08872). For the other papers I counted numbered displays in the TeX, so those numbers can be off by one (see Section 5).

Notation used in my own sentences: "amp" means the overlap amplitude $|\langle a|b\rangle|$, "sq" means $|\langle a|b\rangle|^2$, $\lambda_0$ is the leading eigenvalue of a mixed transfer matrix (TM), and a "cell" is the TM's unit cell.

## 1. Summary

- **The standard TI quantity is the fidelity per site of Zhou, Orús and Vidal.** $\ln d = \lim_{N\to\infty} \frac{1}{N}\ln|\langle\psi_1|\psi_2\rangle|$. It is built on the amplitude, not its square. It is per lattice site, assumes normalized states, and equals the leading eigenvalue of the single-site mixed TM (ZOV 0709.4596 Eqs. 1-2; Zhou 0704.2945; Vanderstraeten, Haegeman, Verstraete 1810.07006 Sec. 2.1, which calls $|\lambda(E)|$ "the fidelity per site"). For an $N$-site unit cell, Karrasch and Schuricht (1302.3893 App. B) take "the $N$-th root of the dominant eigenvalue".
- **Loschmidt-echo papers square the overlap and take a log.** The Loschmidt rate is $l(t) = -\frac{1}{N}\ln|\langle\psi_0|e^{-iHt}|\psi_0\rangle|^2$, which is $-2\ln d$ for the return overlap (Heyl, Polkovnikov, Kehrein 1206.2505; Karrasch and Schuricht Eq. 10).
- **Only two papers compile circuits on an infinite chain and report an error density.** Astrakhantsev, Lin, Pollmann, Smith (2210.03751) compile states with a 1-site cell. They maximize the "overlap density" $|\lambda|$ (amp per site) and report $1-\mathcal F$, with $\mathcal F$ the "squared overlap per unit cell". Gibbs and Cincio (2506.02105) compile $e^{-iHt}$ with a 2-site cell and report $C_3 = 1 - \frac{1}{N}\sum_k|\lambda_0|^2$, with $\lambda_0$ taken from the 2-site mixed TM. The paper calls this a "local measure of fidelity per qubit". It is the squared overlap per 2-site cell, so it is about twice a per-site squared infidelity. I checked the TM diagram in their TeX (two ket and two bra site tensors); the claim we had is correct.
- **Barratt et al. (2003.12087) and Dborin et al. (2205.12996)** use the leading eigenvalue of the mixed TM as the objective of each TDVP-like step on an infinite chain, and plot the log of the Loschmidt overlap. Neither defines a per-site accuracy metric for the compiled evolution.
- **Finite-system compilation papers mostly report whole-system operator quantities, with no per-site normalization.** These are the Hilbert-Schmidt test $1-|\mathrm{Tr}\,U^\dagger V|^2/4^n$ (Gibbs and Cincio 2409.16361; Le, Sun, Mendl; Mizuta et al.; Zhang et al. Eq. 1), the normalized Frobenius distance (Mansuroglu et al.; Mc Keever and Lubasch; Tepaske, Hahn, Luitz; Kotil et al. as objective) and the spectral norm (Kotil et al. as reported error).
- **Per-site normalizations in finite systems exist but differ from paper to paper.**
  - Zhang et al. Eq. 9 take $1-(1-C)^{1/n}$ of the input-averaged infidelity $C$, a root of the mean.
  - Gibbs and Cincio Eq. 13 average $|\langle R_k|V^\dagger U|R_k\rangle|^{2/n}$ over Haar-random states, a mean of roots.
  - Causer et al. Eq. 14 define the operator error density $\delta = \sqrt{2-(\mathrm{Re\,Tr}\,U^\dagger V)^{1/N}}$.
  - De Franco et al. (2607.14296) use the operator log-fidelity density $-\frac1N\log_2(|\mathrm{Tr}\,U^\dagger V|/2^N)$.
  - Van Damme et al. (2302.14181) use the log-amplitude per site $\frac1N\log\langle\Psi_{\rm bench}|\Psi\rangle$.
- **So there is no single standard for a compilation error.** The tensor-network fidelity literature uses the amplitude per site. The circuit-compilation papers use the squared overlap, per site or per cell. The log (rate) form appears in the Loschmidt work and in Van Damme et al. and De Franco et al.
- **Inputs.**
  - On infinite chains, every state-compilation and Loschmidt paper evolves one physical initial state at a time: Néel, fully polarized, or a ground state.
  - The only infinite-chain unitary compilation (Gibbs and Cincio 2506.02105) trains on 4 random iMPS of bond dimension $\chi=2$ with a 2-site cell and tests on 100 of them. It calls $\chi=1$ (product) inputs the choice of "previous work" and reports that they generalize worse.
  - For finite systems the standard training inputs are Haar-random single-qubit product states (Caro et al.; Gibbs et al. 2204.10269; Zhang et al., who test on 100). Haar-random global states also appear (Gibbs and Cincio Eq. 13 with 8 states; Tepaske et al.), and Mansuroglu et al. sample "random input vectors" without saying from which distribution.
- **I found no paper whose input states have a longer unit cell than the circuit's period, as our 8-site product cells do.**

## 2. Table

"Whole" means a whole-system quantity with no per-site normalization. Formulas avoid the pipe character so the table renders; `abs(...)` is the modulus.

| Paper (arXiv version, journal) | System | Quantity: formula; amp or sq; per site or cell | How computed | Inputs: type, number, cell | Where |
|---|---|---|---|---|---|
| Zhou, Orús, Vidal, 0709.4596 v2, PRL 100, 080601 (2008) | TI lattice, infinite or finite PBC, "invariant under translations by one lattice site" | $\ln d=\lim \frac1N \ln F$, $F$ = abs(overlap). Amp, per site, defined via log, reported as $d$ | Leading eigenvalue $\mu_0$ of the 1-site mixed TM $E=\sum_s A^{s*}(\lambda_2)\otimes A^s(\lambda_1)$; corrections exponentially small in $L_x$ | Ground states; no dynamics | Eqs. (1), (2); "Fidelity per lattice site from tensor network representations", case (i) |
| Zhou, Barjaktarevič, cond-mat/0701608 v1; Zhou, 0704.2945 v1 | 1D chains | $F$ = abs(overlap) $\sim d^L$, $\ln d = \lim \ln F/L$. Amp, per site | $d$ = largest eigenvalue of $E$ (0704.2945) | Ground states | 0701608 Proposition 2 and footnote "definition"; 0704.2945, text after the MPS fidelity equation |
| Vanderstraeten, Haegeman, Verstraete, 1810.07006 v3, SciPost Phys. Lect. Notes 7 (2019) | Uniform MPS, 1-site cell | "fidelity per site" $f$ = abs($\lambda(E^1_2)$). Amp, per site | Largest-magnitude eigenvalue of the mixed TM | n/a | Sec. 2.1 |
| Heyl, Polkovnikov, Kehrein, 1206.2505 v2, PRL 110, 135704 (2013) | Infinite TFIM chain | $f(z)=-\lim\frac1N\ln Z(z)$; $L(t)$ = abs($G$)$^2=e^{-N l(t)}$, $l = f(it)+f(-it)$. Sq, per site ("number of degrees of freedom"), log | Exact (free fermions) | Ground state of the pre-quench $H$ | Eqs. (2), (4); paragraph "The free energy density (4) is just the rate function" |
| Karrasch, Schuricht, 1302.3893 v2, PRB 87, 195104 (2013) | Infinite chain with $N$-site cell ($N=2$ for Ising); finite chains as check | $l(t)=-\frac1L \ln$ abs($G$)$^2$. Sq, per site, log | Infinite: "$N$-th root of the dominant eignvalue [sic]" of the $N$-site mixed TM; finite: "$L$-th root of the overlap" | Ground states, polarized states | Eq. (10); App. B |
| Barratt et al., 2003.12087 v2, npj QI 7, 79 (2021) | Infinite TI chain, circuit iMPS with $U\in SU(dD)$ | Per-step objective abs(overlap)$^2$ of $e^{iHdt}$-evolved and new state, "given as the largest eigenvalue of the mixed transfer matrix"; Loschmidt as "the log of this overlap". No per-site accuracy metric | Variational eigenvalue search on the mixed TM; the TDVP-step TM figure holds two state unitaries | TFIM ground state at $g=1$, evolved at $g=0.2$; PXP states | Supp. "Time-evolving Quantum Circuit MPS", "Finding the Overlap"; main Sec. II.C |
| Dborin et al., 2205.12996 v1 | Infinite TI chain, $D=2$ circuits on Google Rainbow | Overlap $\to\lambda^N$, $\lambda$ the principal TM eigenvalue (amp per TM block); circuits estimate $\lambda^2$; Loschmidt plotted as $-\log$ abs(overlap) | Power method $\lambda=\lim C_n/C_{n-1}$; $C_n$ divided by $C_n(U_A,U_A)$ to undo depolarization | TFIM ground state at $g=1.5$, evolved at $g=0.2$ | Sec. II.C Eq. (2); Sec. II.D Eq. (3); Fig. 4 caption |
| Astrakhantsev, Lin, Pollmann, Smith, 2210.03751 v4, PRR 5, 033187 (2023) | Infinite chain; uniform sequential circuit with 1-site cell; iTEBD reference has a 2-site cell | Overlap density abs($\lambda$), $\log$ abs($\lambda$) $=\lim\frac1N\log$ abs($\xi$) (amp, per cell = per site); fidelity density $\mathcal F$ = "squared overlap per unit cell", error $1-\mathcal F$; accumulated $\mathcal M = 1-\prod_i$ abs($\lambda_i$)$^2$ | Leading eigenvalue of the mixed TM by exact diagonalization or power method | Fully polarized state; one input | Secs. II.A, II.C, III.A; Apps. B.1, E |
| Gibbs, Cincio, 2506.02105 v1 (infinite chain, $C_3$) | Infinite TI chain, iMPS with 2-site cell; circuit layers of period 2 | $C_3=1-\frac1N\sum_k$ abs($\lambda_0$)$^2$, $\lambda_0$ of the 2-site mixed TM. Sq, per 2-site cell, about 2x per site; text says "per qubit" | Arnoldi (Arpack.jl) on the mixed TM; targets by iTEBD, $\chi_{\max}=128$ | Random iMPS, $\chi=2$, 2-site cell: 4 train, 100 test; $\chi_{\rm Train}=1$ studied in App. A; sampling distribution not stated | Sec. II.2 Eq. (1); Sec. III.3 Eq. (12); Sec. III.4; App. A |
| Gibbs, Cincio, 2506.02105 v1 (ring check, $C_4$) | Periodic rings, exact state vectors | $C_4 = 1-\frac1N\sum_k$ abs(overlap)$^{2/n}$. Sq, per qubit, mean of roots | Exact state vectors | 8 Haar-random global states | Sec. III.5 Eq. (13) |
| Van Damme, Haegeman, McCulloch, Vanderstraeten, 2302.14181 v1 | Infinite chain (MPO for $e^{-iH\delta t}$); finite cylinder | Phase error $p$ = abs($\lambda - iE_0\delta t$), $\lambda=\lim\frac1N\log\langle\Psi_0\vert U\vert\Psi_0\rangle$ (complex log-amp per site); "fidelity per site" $f=\frac1N\log\langle\Psi_{\rm bench}\vert\Psi_n\rangle$ | Leading eigenvalue of the triple-layer TM | Ground state (infinite); ground state with a spin flip (cylinder) | Secs. 2.1, 8.1, 8.2 |
| Lin, Dilip, Green, Smith, Pollmann, 2008.10322 v2, PRX Quantum 2, 010342 (2021) | Finite open chain, $N=31$ ($N=11$ for the evolution algorithm) | $\mathcal F$ = abs(overlap)$^2$, whole; threshold $1-10^{-4}$ | TEBD MPS, $\chi=1024$ reference | Polarized all-up state | Sec. II.A Eq. (2); Sec. III.A Eq. (3) |
| Haghshenas, Gray, Potter, Chan, 2107.01307 v2, PRX 12, 011047 (2022) | Finite open chains, $L=16$ to 32; ground states | $1-$abs(overlap), amp, whole; relative energy error | DMRG reference | n/a | "Local Optimization versus Global Optimization" |
| Mansuroglu et al., 2106.03680 v5, QST 8, 025006 (2023) | Periodic rings (also open, 2D); trained on 6 qubits, applied up to 24 | $C=\frac{1}{2^N}$(Frobenius distance)$^2$; sampled form $\frac{1}{\lvert\mathcal V\rvert}\sum(1-$abs(overlap)$)$, amp. Whole | Exact up to 12 qubits, sampling beyond | Operator level; "random input vectors" (number and distribution not stated) | Sec. 3 Eq. (8); Sec. 5 Eq. (15) |
| Mc Keever, Lubasch, 2205.11427 v5, PRR 5, 023146 (2023) | Finite open chain, $n\le14$ for exact errors | $\epsilon_{\rm approx}=\sqrt{1-\mathrm{Re\,Tr}[U^\dagger e^{-itH}]/2^n}$; phase error; spectral norm. Whole (inset divided by $n$) | MPO, Taylor expansion, Newton method | Operator level; App. F: 50 random bitstrings at $n=12$ | Sec. 2 Eqs. (2), (3); Sec. 3; Apps. B, F |
| Kotil, Banerjee, Huang, Mendl, 2212.07556 v2, J. Phys. A 57, 135303 (2024) | Periodic ring, trained at $L=6$, applied to larger $L$ | Objective: Frobenius distance$^2$; reported: spectral norm. Whole | Exact; Riemannian trust region | Operator level | Secs. 3, 5, 6.1 |
| Mizuta, Nakagawa, Mitarai, Fujii, 2203.15484 v1, PRX Quantum 3, 030402 (2022) | TI with PBC; compiled at $\tilde L=20$, applied at $L=40$ | $C_{\rm HST}$ (whole, sq); local $C_{\rm LHST}=\frac1L\sum_j C^{(j)}$, a per-site average of 1-site Bell-pair infidelities; $\bar F \ge 1-\frac{2^L}{2^L+1}L\,C_{\rm LHST}$ | TEBD on the subsystem | Operator level | Eqs. (1), (7), (8), (12), (13); Secs. 2.1, 4.1, 5 |
| Tepaske, Hahn, Luitz, 2205.03445 v2, SciPost Phys. 14, 073 (2023) | Finite open chains, $L=8,12,16$ | $\epsilon = 1-\mathrm{Re\,Tr}[\mathcal U^\dagger\mathcal C]/2^L$. Whole | Exact, or trace estimated with Haar-random states | Operator level; $N_\psi$ Haar-random states | Secs. 2.3, 2.5 |
| Causer, Jung, Mitra, Pollmann, Gammon-Smith, 2312.14245 v2, PRR 6, 033062 (2024) | Finite open chains, $N=8$ to 512; gates optimized one by one (not TI) | $\delta=\sqrt{2-(\mathrm{Re\,Tr}\,U^\dagger_{\rm MPO}U_{\rm BW})^{1/N}}$: operator, per site, linear in the overlap. $F^{\rm a}=\frac{1}{2^N}\sum_\sigma$ abs(overlap)$^2$, whole | MPO contractions | Operator level; $F^{\rm a}$ sums over all bitstrings | Sec. IV.1 Eq. (14); Sec. V.2 Eq. (23) |
| Zhang, Wiersema, Carrasquilla, Cincio, Kim, 2409.16346 v1 | Finite open chains, $n$ up to 80; quasi-2D | $C=1-\frac{1}{N_s}\sum_i$ abs(overlap)$^2$ (whole); "per-site" $\tilde C=1-(1-C)^{1/n}$, root of the mean | TEBD on product inputs | Haar-random single-qubit product states; $N_s$ train ("$N_s<50$ suffices"; 16 in App. C), 100 test | Sec. II.1 Eq. (5); Sec. II.2 Eq. (9) |
| Gibbs, Cincio, 2409.16361 v2, Quantum 9, 1789 (2025) | Finite open chains (TFIM 200 qubits, Hubbard 100, $J_1$-$J_2$ 40) | $C_{\rm HST}=1-$abs(Tr)$^2/2^{2n}$. Sq, whole | MPOs mapped to normalized MPSs on a doubled space | Operator level | Sec. 2.1 Eqs. (1)-(3) |
| Le, Sun, Mendl, 2501.08872 v4, Quantum 9, 1833 (2025) | Finite open chains, $N=20, 50$; gates independent | $\mathcal C_{HS}=1-$abs(Tr)$^2/d^2$, $d=2^N$. Sq, whole | MPO | Operator level | Sec. 2 Eq. (2) |
| De Franco, Kennes, Luitz, Rizzi, Schmitt, 2607.14296 v1 | Finite random Floquet brickwork, $N=100$, not TI | $f=-\frac1N\log_2 F$, $F$ = abs(Tr)$/2^N$. Amp, per site, log | MPO | Operator level | Sec. 4.1, Eq. (logf); App. C |
| Caro et al., 2204.10268 v3, Nat. Commun. 14, 3751 (2023) | Finite $n$, theory | Training cost $1-\frac1N\sum$ abs(overlap)$^2$, whole; risk bounds | n/a | Locally scrambled ensembles, e.g. Haar or stabilizer product states | Eq. (4); Lemma 1; Theorem 1 |
| Gibbs et al., 2204.10269 v3, PRR 6, 013241 (2024) | Finite: 4-qubit Heisenberg with PBC; XY with OBC, $n\le12$ | Global cost $1-\frac1N\sum$ abs(overlap)$^2$; local cost | Exact | Haar product states; 1 to 5 sufficed | Eqs. (5), (6); numerics section |

## 3. Per-paper notes

### 3.1 Fidelity per site: Zhou, Orús, Vidal (0709.4596) and Zhou's earlier papers

The definition, from the "Generalities" paragraph:

> the ground state fidelity $F(\lambda_1,\lambda_2) = | \langle \psi(\lambda_2)|\psi(\lambda_1)\rangle|$ asymptotically scales as $F(\lambda_1,\lambda_2) \sim {d(\lambda_1,\lambda_2)}^N$, with $N$ the number of sites in the lattice. [...] Physically, the scaling parameter $d(\lambda_1,\lambda_2)$ is the *averaged* fidelity per lattice site,
> $$\ln d(\lambda_1,\lambda_2) \equiv \lim_{N \rightarrow \infty} \frac {\ln F(\lambda_1,\lambda_2)}{N}, \qquad (1)$$
> which is seen to be well defined in the thermodynamic limit even if $F(\lambda_1,\lambda_2)$ becomes trivially zero. It satisfies [...] (i) normalization $d(\lambda,\lambda)=1$

The finite-size version is Eq. (2), $\ln d_N \equiv \ln F/N$. The overlap is an amplitude, the states are normalized, and the definition is per site.

On computation: the authors "specialize to a $D$-dimensional lattice system that is invariant under translations by one lattice site". They write $F = \mathrm{Tr}(T^{L_x})$ and conclude

> $d_N(\lambda_1,\lambda_2)$ is given by the largest eigenvalue $\mu_0$ of $T$ up to corrections that decay exponentially in the linear system size $L_x$.

For 1D, case (i): "The zero-dimensional transfer matrix $T$ is given by $E_{\tilde{\alpha}\tilde{\beta}}(\lambda_1,\lambda_2) \equiv \sum_s (A^{s}_{\alpha' \beta'}(\lambda_2))^* A_{\alpha \beta}^{s}(\lambda_1)$". This is a 1-site TM. The paper never treats a multi-site cell.

On why per site, the introduction says the fidelity per site "is naturally interpreted as the free energy per site of this $D$-dimensional classical statistical vertex lattice model, and as such it is well defined in the thermodynamic limit (even though the fidelity itself becomes zero)." The paper credits Ref. [zhou] with the quantity: "The fidelity per lattice site, introduced in Ref. [zhou]". That reference is H.-Q. Zhou and J. P. Barjaktarevič, cond-mat/0701608, plus 0704.2940 and 0704.2945.

- Zhou and Barjaktarevič (cond-mat/0701608 v1) define "the fidelity $F(\psi, \phi) \equiv | \langle \psi | \phi \rangle |$". Their Proposition 2 states "the fidelity $F(\lambda, \lambda')$ scales as $[d(\lambda, \lambda')]^L$". The footnote reads "Formally, $d(\lambda,\lambda')$ may be defined as $\ln d(\lambda,\lambda') = \lim_{L \rightarrow \infty} \ln F(\lambda,\lambda') /L$ in the thermodynamic limit."
- Zhou (0704.2945 v1) writes $F = \mathrm{Tr}(E^L)$ with "the (generalized) transfer matrix $E(\lambda,\lambda') = \sum _{s=1}^d A^{s*}(\lambda') \otimes A^s(\lambda)$". It adds: "In fact, the scaling parameter $d(\lambda,\lambda)$ [sic] is the largest eigenvalue of the transfer matrix $E(\lambda,\lambda')$, if there is no pair of eigenvalues that are complex conjugate each other".

These follow-ups fix the same convention: amplitude, per site, leading TM eigenvalue.

### 3.2 Vanderstraeten, Haegeman, Verstraete (1810.07006), uniform MPS lecture notes

Sec. 2.1, for "two properly normalized MPS":

> This result is known as the orthogonality catastrophe, according to which states in the thermodynamic limit are either equal or orthogonal. [...] A more physical quantity to express whether two MPS in the thermodynamic limit are `close', is the fidelity per site, which exactly corresponds to
> $$f \left( A_1,A_2 \right) = \left| \lambda(E^1_2) \right|.$$

This is a 1-site uniform MPS, the amplitude, and the leading eigenvalue of the mixed TM. It is the same as ZOV.

### 3.3 Heyl, Polkovnikov, Kehrein (1206.2505), Loschmidt rate function

Eq. (2) is $G(t)=\langle \Psi_i| e^{-iHt} |\Psi_i\rangle$. Eq. (4) is $f(z)=-\lim_{N\rightarrow\infty} \frac{1}{N}\,\ln\,Z(z)$, "where $N$ is the number of degrees of freedom". The text then says:

> The free energy density (4) is just the rate function of the return amplitude $G(t) = \exp[-N\,f(it)]$. Likewise for the return probability (Loschmidt echo) $L(t)\stackrel{\rm def}{=} |G(t)|^2 = \exp(-N\,l(t))$ one has $l(t)=f(it)+f(-it)$.

There are two rates: one for the amplitude, $\mathrm{Re}\,f(it)$, and one for the squared overlap, $l=2\,\mathrm{Re}\,f(it)$. Both are per site and in log form. The paper computes them exactly for the TFIM. The input is the ground state of the pre-quench Hamiltonian. This is a physical observable, not a compilation error, but the per-site log of the squared overlap became the usual way to report infinite-chain overlaps in dynamics.

### 3.4 Karrasch and Schuricht (1302.3893), rate function on infinite MPS with an $N$-site cell

Eq. (10) is $l(t)=-\frac{1}{L}\ln\big|G(t)\big|^2=2\,\mathfrak{Re}\,f(i t)$. App. B says how to compute it:

> The matrices $A^{[1\ldots N]\sigma_j}$ are associated with a unit cell of size $N$ (e.g., $N=2$ for the Ising model). [...] For a finite system described by Eq. (mps1), the return amplitude per site can be computed straightforwardly as the $L$-th root of the overlap between $|\Psi_0\rangle$ and $\exp(-i H_1t)|\Psi_0\rangle$. The overlap per site of two infinite MPS [described by matrices $A$ and $\tilde A$ in Eq. (mps2)] is given by the $N$-th root of the dominant eignvalue [sic] of the transfer matrix

The TM in that equation multiplies all $N$ site tensors of the cell. This is the clearest statement I found of the cell-length root: take the leading eigenvalue of the $N$-site mixed TM, then its $N$-th root, to get a per-site amplitude. The App. B figure caption calls $l(t)$ "the rate function per site". The inputs are ground states of $H_0$ and polarized states.

### 3.5 Barratt, Dborin, Bal, Stojevic, Pollmann, Green (2003.12087)

System: an infinite TI chain whose state is a circuit iMPS. "$U \in SU(dD)$ with $d$ the local Hilbert space dimension and $D=2^N$ the bond order" (Fig. 1 caption).

The time-evolution step (Supplementary, "Time-evolving Quantum Circuit MPS") finds parameters "whose fidelity with the time-evolution of the state $|\psi (X(t))\rangle$ is optimised: $\max_{X(t+dt)} \langle \psi (X(t+dt))| e^{i H dt} |\psi (X(t))\rangle|^2$" (the opening modulus bar is missing in the source). The overlap itself (Supplementary, "Finding the Overlap"):

> In the infinite translationally invariant case, the definition of the overlap requires care. It can be given as the largest eigenvalue of the *mixed transfer matrix*. [...] The overlap is the result of applying the matrix of SEq. (MixedTransferMatrix) an infinite number of times to boundary vectors. The result (with appropriate normalisation) is equal to the largest eigenvalue.

The TM is drawn only as a figure. I rendered `EqB16Fig.pdf` with Ghostscript. For the $D=4$ step it contains two state unitaries, $U_\alpha$ and $U_{\alpha-1}$, together with $W_{\rm odd}$ and $W_{\rm even}$, so the block spans two physical sites. The text gives no root by cell length.

Loschmidt echo, main Sec. II.C: "The ground state of the transverse field Ising model with $g=1$ is prepared. This state is then evolved under the same Hamiltonian but with $g=0.2$. At each step of the trajectory, the overlap of the current state with the original state is recorded. The log of this overlap is shown for the whole trajectory in Fig. 3c." The paper reports no per-site infidelity of the evolution against an exact reference.

### 3.6 Dborin et al. (2205.12996), the same framework on hardware

Sec. II.C:

> For states parametrized by circuit unitaries $U_A$ and $U_B$ this overlap is given by an infinitely wide and deep version of the circuit shown in Fig. 3a. It is formally zero. However, the distance between states can be quantified by the rate at which this overlap tends to zero as the length of the system, $N$, is taken to infinity. This occurs as $\lambda^N$, where $\lambda$ is the principal eigenvalue of the transfer matrix

Eq. (2) is the power method, $\lambda = \lim_{n\to\infty} C_n(U_A,U_B)/C_{n-1}(U_A,U_B)$. On hardware, "we divide $C_n(U_A,U_B)$ by a Loschmidt echo $C_n(U_A,U_A)$, whose value is one in the depolarisation-free case." This normalizes by the self-overlap, for noise.

The time-evolution update is Eq. (3), $U(t+dt) = \mathrm{argmax}_{W} | \langle \psi(W)| e^{i {\cal H} dt } | \psi(U(t)) \rangle|$. The Fig. 4 caption says the measured probability "gives an approximation to $\lambda^2$". The DQPT plot is $-\log|\langle\psi(0)|\psi(t)\rangle|$ for an infinite chain, which is in effect a log-amplitude per TM block: "Starting with the parametrised groundstate at $g=1.5$ and evolving with $g=0.2$, the logarithm of the overlap of the initial state with the time evolved state, $- \log |\langle \psi(0) | \psi(t) \rangle |$" (Sec. II.D).

### 3.7 Astrakhantsev, Lin, Pollmann, Smith (2210.03751), uniform sequential circuits

System: an infinite chain. Sec. II.B: "Due to the sequential decomposition, $\hat{u}(\delta t)$ and hence $\hat{U}_T$ are translationally invariant with a single site unit cell."

Quantities. Sec. II.A: "The absolute value of the eigenvalue $\lvert \lambda \rvert \leqslant 1$ defines the overlap density between the two states, and $|\lambda| = 1$ if and only if the states are identical." App. B.1:

> Therefore, a better quantity to consider is instead the overlap density, which is equal to the absolute value of the leading eigenvalue of the transfer matrix $\lvert \lambda \rvert$, satisfying the relation
> $$\log \lvert \lambda \rvert = \lim_{N \rightarrow \infty} \frac{1}{N}\log{\lvert \xi \rvert}.$$

The reason for a density, Sec. II.C: "The direct measure of the closeness is the fidelity, i.e., squared overlap, between the two states [...]. This quantity is either $1$ or $0$ in the thermodynamic limit and cannot be used for posing the optimization problem. Instead, we define the mixed transfer matrix [...] we maximize the absolute value of the overlap density $\lvert \lambda \rvert$".

The reported accuracy, Sec. III.A:

> we define the reachable time $t^*$ of the given ansatz as the time when the error in fidelity density with the quasi-exact (iTEBD) state crosses the threshold value $1-\mathcal{F} = 10^{-4}$. Here, $\mathcal{F}$ is the fidelity density, i.e., squared overlap per unit cell, between the USC state and the iTEBD wave function.

They also use an accumulated measure, $\mathcal{M}(t) = 1 - \prod_{i < t} |\lambda_{i}|^2$. App. E notes that the reference "breaks translation symmetry to two-site emergent unit cell". The paper does not say whether $\mathcal F$ against that reference is per site or per 2-site cell.

The input is one state: "The initial wave function is taken as a fully-magnetized state $|\psi_0 \rangle = |\ldots 000\ldots\rangle$ in the $\sigma^z$ basis" (Sec. III), for the TFIM with a longitudinal field.

### 3.8 Gibbs and Cincio (2506.02105), learning circuits with infinite tensor networks

System. Sec. II.1: "Throughout this work, we focus on compiling translation invariant states and unitaries with a 2-site unit cell, however our method can be easily extended to larger unit cells." Sec. II.3 defines layers $U_0=\bigotimes_r U^{[2r,2r+1]}$ and $U_1=\bigotimes_r U^{[2r-1,2r]}$, the same period-2 structure as our brickwork.

Quantity, Sec. II.2:

> Due to the orthogonality catastrophe in exponentially large Hilbert spaces, the global fidelity does not provide useful information on infinite systems, and in fact takes a binary value of 0 or 1 for normalized states. Therefore, we instead compute a local measure of fidelity per qubit. [...] the local fidelity between two normalized iMPS $|\psi_1\rangle$ and $|\psi_2\rangle$ is given by the leading eigenvalue of the corresponding mixed transfer matrix: If $|\psi_1\rangle$ and $|\psi_2\rangle$ are described by the iMPS site tensors $\{A,B\}$ and $\{C,D\}$ respectively, the following tensor network gives their mixed transfer matrix. [diagram] Consequently, the largest magnitude eigenvalue of this matrix gives the local fidelity between the two states, which we denote by
> $$f_\text{local} = \lambda_0(|\psi_1\rangle, |\psi_2 \rangle). \qquad (1)$$
> Throughout this work we ensure all iMPS remain normalized, by imposing that $\big|\lambda_0(|\psi\rangle, |\psi \rangle) \big| = 1$.

Verification of the 2-site cell. The diagram is the TikZ macro `\transfermatrix` in `main.tex`. It draws ket tensors $A^*, B^*$ above bra tensors $C, D$, with two vertical physical bonds and open virtual bonds at both ends. So $\lambda_0$ is the overlap amplitude per 2-site cell. The iTEBD bond matrices "are absorbed into the site tensors" (Sec. II.3), so $\{A,B\}$ are exactly the two physical sites.

The cost for $e^{-iHt}$ is Eq. (12), Sec. III.3:

> $$C_{3}(\boldsymbol{\theta}) = 1 - \frac{1}{N} \sum_{k=1}^{N}\big|\lambda_0(V(\boldsymbol{\theta}) |\Phi^\chi_k \rangle, e^{-iHt}|\Phi^\chi_k \rangle)\big|^2.$$

It is the squared overlap per 2-site cell. A per-qubit squared fidelity would be $|\lambda_0|^{2/2}=|\lambda_0|$. So $C_3$ per input is $1-|\lambda_0|^2 = (1-|\lambda_0|)(1+|\lambda_0|) \approx 2(1-|\lambda_0|)$, about twice the per-site squared infidelity, despite the words "per qubit". The general cost Eq. (3) and the state costs Eqs. (10) and (11) have the same form. Eq. (11) evolves from the Néel state $|10\rangle^{\otimes\infty}$ (XXZ in Pauli operators with $J_x=1.0$, $J_z=0.5$ to $t=1.5$, and Thirring to $t=8.0$).

Inputs and their justification, Sec. III.3:

> In the above definition, the state $|\Phi^{\chi}\rangle$ denotes a random iMPS with bond dimension $\chi$, and our training set is composed of $N$ of these random states. Previous work approached this compilation using unentangled input states e.g. $\chi_{\text{Train}}=1$. However, here we use $\chi_{\text{Train}}=2$ as this larger set of states has improved generalization properties, with a minimal increase in computational cost.

> Consequently, using the results of Ref. [caro], only $\mathcal{O}(L \ \mathrm{log}(L))$ training states are required for a faithful compilation. For our numerical analysis, we set $N_{\mathrm{Train}}=4$, which we find to be sufficient for all circuit depths tested.

> To verify our compilation is successfully generalizing, we compute the cost function using an unseen set of $N_{\mathrm{Test}}=100$ random testing states of $\chi_\text{Test}=2$ [...] even the smallest increase from $\chi_\text{Train}=1$ to 2 finds significantly improved trainability, with the onset of generalization (the simultaneous decrease of the testing cost with the training cost) found at only 2 training states.

App. A (TFIM, $h=1$, $t=2$, 11 layers) varies $N_{\rm Train}$ for $\chi_{\rm Train}=1$ and 2, always testing on "100 unseen $\chi_{\text{Test}}=2$ random iMPS", and reports "For $N_\text{Train}=1$, generalization is not observed". The cell of the random iMPS is the same 2-site cell as the circuit. The paper does not say how the random tensors are drawn.

The T-count study (Sec. III.4, TFIM $h=1$, $t=2$) uses $C_3$ as the error axis: "the iMPS-optimized circuit with 795 $T$ gates per qubit and corresponding approximation error $3.082 \cdot 10^{-7}$, compared to [...] a 4th-order Trotterization with 4140 $T$ gates per qubit and approximation error $3.208\cdot 10^{-7}$". Each $R_z$ goes through Ross-Selinger with $\|U - R_z(\theta)\|\leq\epsilon$.

The finite-ring check, Sec. III.5, says "Here it is most natural to study a 1D system with periodic boundary conditions (PBC). We use exact state vector simulation". Eq. (13):

> $$C_4(V) = 1 - \frac{1}{N} \sum_{k=1}^{N} \big| \langle\mathcal{R}_k |V^{\dagger}e^{-iHt} | \mathcal{R}_k\rangle \big|^{2/{n_{\text{qubits}}}}.$$
> [...] We modify the calculation to compute a local infidelity for a more similar measure to our earlier numerical results. We average over $N=8$ Haar random states.

The exponent $2/n_{\rm qubits}$ makes $C_4$ per qubit, while $C_3$ is per 2-site cell. For the same per-site error, $C_4 \approx C_3/2$. The paper does not comment on the factor.

### 3.9 Van Damme, Haegeman, McCulloch, Vanderstraeten (2302.14181), MPOs for time evolution

Sec. 2.1: "the expectation value of this MPO with respect to an MPS is characterized by the leading eigenvalue of the triple-layer transfer matrix [...] such that we can evaluate $\lambda = \lim_{N\to\infty} \frac{1}{N} \log \left( \langle \Psi| O |\Psi\rangle \right)$". The operator accuracy test on the infinite chain, Sec. 8.1:

> $$p = \left| \lambda - i E_0 \delta t \right|, \qquad \lambda = \lim_{N\to\infty} \frac{1}{N} \log \langle \Psi_0| U(\delta t) |\Psi_0\rangle ,$$
> where $\lambda$ can be evaluated directly in the thermodynamic limit (see Sec. 2) and $E_0$ is the ground-state energy per site.

This is a complex log-amplitude per site on one input, the ground state, so it tests only the phase the MPO gives that eigenstate. Sec. 8.2, on a finite cylinder: "we evaluate the fidelity per site with respect to a benchmark time-evolved state [...] $f = \frac{1}{N} \log \langle \Psi_{\text{bench}}(T) | \Psi_n(T) \rangle$, with $N$ the number of sites."

### 3.10 Finite-chain state compilation: Lin et al. (2008.10322) and Haghshenas et al. (2107.01307)

Lin et al. use a finite open chain, $\hat{H} = - J [\sum_{j=1}^{N-1} \hat{\sigma}^x_j \hat{\sigma}^x_{j+1} + \dots]$, with $N=31$ and the "polarized initial state $|\Psi\rangle = |\cdots \!\uparrow \uparrow\uparrow\! \cdots\rangle$". The error is the whole-chain squared fidelity, Eq. (2), $\mathcal{F} = |\langle \Psi^M_\mathrm{qc}(t) | \Psi_\mathrm{mps}(t)\rangle|^2$. They define the reachable time as the one "at which the fidelity $\mathcal{F}$ drops below $1-10^{-4}$". For the variational evolution ($N=11$), "We can estimate the total error by monitoring the fidelity at the end of each optimization $\prod_i \mathcal{F}_i$." Nothing is per site. Astrakhantsev et al., with Lin as a coauthor, use the same $10^{-4}$ threshold on the infinite chain, applied there to a fidelity density.

Haghshenas et al. study ground states on open chains ($L=16$ to 32) with "the infidelity cost function ${\bar{\mathcal{F}}} = 1 - | \langle \Psi | \psi \rangle |$". That is amplitude and whole-system, plus relative energy errors. There is no TI dynamics.

### 3.11 TI operator compilation on small rings: Mansuroglu et al., Kotil et al., Mizuta et al.

Mansuroglu et al. (2106.03680), Sec. 3, Eq. (8):

> $$C(U_\text{var}, \vec{\theta}) = \frac{1}{2^N} ||U_{\rm var}(\vec{\theta}) - U_{\rm ex}||_F^2 [...] \in [0, 1],$$
> [...] The additional factor $\frac{1}{2^N}$ eliminates the scaling of $C(\vec \theta)$ in system size $N$. The exact operator of time evolution $U_{\rm ex}$ serves as a benchmark which is agnostic of an initial state.

The $1/2^N$ is the Hilbert-space dimension, not a per-site root. Their own data show the cost growing with $N$ (Sec. 5.1): "For the one-dimensional system, optimization was performed on a 6-qubit chain. An upscaling to 24 qubits only raises the cost value by a factor of 3." Beyond 12 qubits they sample, Eq. (15): $C_{var} \approx \frac{1}{|\mathcal{V}|} \sum_{\psi \in \mathcal{V}} ( 1 - |\langle \psi| U_{\rm ex}^\dagger U_{\rm var} | \psi\rangle| )$, an amplitude, "by sampling over a set of random input vectors". The number and distribution of the vectors are not given in the text.

Kotil et al. (2212.07556) use a periodic ring $\mathbb{Z}_{/(L)}$, trained at $L=6$. The objective is $\|W-U\|_F^2$ (Sec. 5). The reported error is whole-system (Sec. 6): "We fix a time $t$ and quantify the approximation error by the spectral norm distance". On transfer to larger rings (Sec. 6.1): "As expected from the light cone picture, the error hardly increases for larger $L$." They justify the small-ring training in Sec. 3. The light cone at time $t$ must fit inside the ring "such that the light cone does not interfere with itself given the periodic boundary conditions", and the circuit is then extended "simply by extending it with copies of the same gate $G_\ell$ in layer $\ell$".

Mizuta et al. (2203.15484) define the global $C_{\rm HST} = 1 - \frac{1}{4^L} | \mathrm{Tr} [U^\dagger V] | ^2$ (Eq. 1) and a local cost

> $$C_\mathrm{LHST}(U,V) = \frac{1}{L} \sum_{j=1}^L C_\mathrm{LHST}^{(j)}(U,V), \qquad C_\mathrm{LHST}^{(j)}(U,V) = 1 - \mathrm{Tr} [\Pi_j \rho_\mathrm{AB}(U,V)],$$

where $\Pi_j$ projects the $j$-th system/ancilla pair onto a Bell pair (Eqs. 7, 8). This is a per-site operator-level quantity in the sense of an average of single-site terms, not a root of a global fidelity. Its relation to the Haar-averaged gate fidelity is extensive, Eq. (13): $\bar{F}(U,V) \geq 1 - \frac{2^{|\Lambda|}}{2^{|\Lambda|}+1} \cdot |\Lambda| C_\mathrm{LHST} (U,V)$. In Sec. 4.1 they impose "translation invariance and PBC also on the ansatz". In Sec. 5 they compile at $\tilde L = 20$ with TEBD, apply the circuit at $L=40$, and report a whole-system bound, $\bar{F} \geq 0.9977$.

### 3.12 Finite open chains with MPOs: Gibbs and Cincio (2409.16361), Le, Sun, Mendl (2501.08872), Tepaske et al. (2205.03445), Mc Keever and Lubasch (2205.11427)

All four report whole-system operator errors.

- Gibbs and Cincio 2409.16361, Sec. 2.1, Eq. (1): $C_{\text{HST}}(U, V) = 1-\frac{1}{2^{2n}}|\text{Tr}(U^{\dagger}V)|^2$, "directly related to the average state infidelity when applying the two unitaries to a random state" through Eq. (2), $C_{\text{HST}} = \frac{2^n + 1}{2^n}(1 - \mathbb{E}_{\text{Haar}_n} |\langle\psi|U^{\dagger}V|\psi\rangle|^2)$. Open chains, TFIM on 200 qubits.
- Le, Sun, Mendl, Sec. 2: "A common way [Kotil, Causer] to quantify how well a quantum circuit $W$ of $N$ qubits approximates a reference unitary $U_{\rm ref}$ is given by the normalized Frobenius norm", $\mathcal C_F = 1-\frac1d\mathrm{Re\,Tr}(U_{\rm ref}^\dagger W)$. They then use the phase-insensitive Eq. (2), $\mathcal{C}_{HS} = 1 - \frac{1}{d^2} |\mathcal T|^2$. This is the only explicit "common way" statement I found, and it concerns a whole-system quantity. Open chains, $N=20$ and 50.
- Tepaske et al., Sec. 2.3: "the 'infidelity' $\epsilon$, given by $\epsilon = 1 - \mathrm{Re}\,\mathrm{tr}[\mathcal{U}^{\dagger}\mathcal{C}]/2^L$". For large $L$, Sec. 2.5 replaces the trace by an average over $N_\psi$ Haar-random states ("we use typicality").
- Mc Keever and Lubasch, Sec. 2, Eq. (2): $\epsilon_\text{approx} = \sqrt{1 - \Re \{ \mathrm{tr}[ U^{\dagger} \exp(-\text{i} t H) ] \} / 2^{n}}$ for $n\le14$. Eq. (3) is a phase error $|\lambda + i e_0 t|$ with $\lambda = \log\langle\psi_0|U|\psi_0\rangle$ on the DMRG ground state, the finite, unnormalized form of Van Damme's test. Sec. 3 reports that "when they are normalized by the number of qubits, the data collapse onto a single curve [...] indicating a linear scaling of the approximation error with qubit number." App. F gives the state infidelity $1-|\langle\psi_\text{exact}|\psi_\text{approx}\rangle|^2$ "averaged over a set of $50$ initial states $|\psi\rangle_\text{init}$ which are drawn from the set of random bitstrings" at $n=12$.

### 3.13 Per-site operator densities on finite chains: Causer et al. (2312.14245) and De Franco et al. (2607.14296)

Causer et al., Sec. IV.1, Eq. (14):

> One instructive measure is the error density of the circuit with respect to the MPO [footnote: In practice, we expect that $\mathrm{Tr} [ U^{\dagger}_{\rm MPO}(\Delta t) U_{\rm BW} (\Delta t)] = (2 - \delta(\Delta t)^{2})^N$; (error_density) follows directly from this.],
> $$\delta(\Delta t) = \sqrt{2 - {\rm Re}\left(\mathrm{Tr} \left[ U^{\dagger}_{\rm MPO}(\Delta t) U_{\rm BW} (\Delta t) \right]\right)^{1/N}}.$$

The trace is not divided by $2^N$ inside the root. With $f_{\rm op} = (\mathrm{Re\,Tr}/2^N)^{1/N}$, the per-site Hilbert-Schmidt amplitude, $\delta^2 = 2(1-f_{\rm op})$. It is linear in the overlap and phase-sensitive. The chains are open, $N=8$ to 512, and the gates are optimized individually, not TI. For comparison with hardware they also use a whole-system state average over all bitstrings, Eq. (23): $F_{\Delta t}^{\rm a} = \frac{1}{2^{N}} \sum_{\bm \sigma} |\langle {\bm \sigma} | U_{\rm MPO}^{\dagger}(\Delta t) U_{\rm BW}(\Delta t) | {\bm \sigma}\rangle|^{2}$.

De Franco et al., Sec. 4.1, cite ZOV for the scaling argument:

> A standard choice for the cost function is the fidelity between the target operator $(U^t)$ and an Ansatz circuit $(C)$ of reduced depth, $F = \dfrac{1}{2^{N}}\left|\text{Tr}\left[\left(U^{t}\right)^{\dagger}C\right]\right|$. [...] as in the case of state fidelity [Zhou_2008, ...], $F$ exhibits exponential scaling with the system size $N$ [...] To remove the scaling with $N$, we use as cost function the logarithmic fidelity density $f = -\frac{1}{N}\log_{2} F$ which is expected to be independent of the system size.

This is the ZOV convention applied to operators: amplitude, per site, log, base 2. App. C derives $F \sim F_1^N$ for product unitaries. The system is a disordered Floquet circuit at $N=100$, so it is not TI.

### 3.14 Finite-system QML compilation and the input-state theory: Zhang et al. (2409.16346), Caro et al. (2204.10268), Gibbs et al. (2204.10269)

Zhang et al. use the empirical risk on Haar-random product states, Eq. (5): $C_{\mathcal{D}_\mathcal{Q}}(\boldsymbol{\theta}) = 1 - \frac{1}{N_s} \sum_{i=1}^{N_s} |\langle \psi_i|U^\dagger V(\boldsymbol{\theta})|\psi_i\rangle|^2$, with $\mathcal Q = \mathcal{S}_{\mathrm{Haar}_1^{\otimes n}}$. The per-site version is Sec. II.2, Eq. (9):

> Lastly, fixing the total evolution time and PQC depth while varying the system size, we plot a "per-site" risk: $\tilde{C}_{\mathcal{D}_\mathcal{Q}} = 1- (1-C_{\mathcal{D}_\mathcal{Q}})^{1/n}$, against $n$. We find that the per-site risk function flattens already at $n>20$, showing that the QML method can be extended to larger systems even in the presence of next-neighbor interaction.

Their numbers of inputs: "validate afterwards on a data set $\mathcal{D}_\mathrm{Test}$ of $100$ unseen states"; "in practice we find that a small number of states ($N_s<50$) suffices"; App. C uses $N_s = 16$ at $n=80$. The chains are open: "despite the open boundary condition of the $1$D chains, the translation invariant PQC's generalization outperforms that of the non-translation invariant one due to its fewer parameters."

Caro et al. treat finite $n$ and whole-system risks. Lemma 1: for any $\mathcal{Q} \in \mathbb{S}_{\rm LS}^{(2)}$, $\frac{1}{2} R_{\mathcal{S}_{{\rm Haar}_n}} \leq \frac{d}{d+1} R_{\mathcal{Q}} \leq R_{\mathcal{S}_{{\rm Haar}_n}}$ with $d=2^n$. Theorem 1 states $\frac12 R_{\mathcal Q}\le R_{\mathcal P}\le 2R_{\mathcal Q}$ for two such ensembles. The definition the theorems need:

> An ensemble of $n$-qubit unitaries is called locally scrambled if it is invariant under pre-processing by tensor products of arbitrary local unitaries.

On which inputs to use: "as random stabilizer states and random product states can be prepared using only a single layer of single-qubit gates, it makes practical sense to train using the ensembles $\mathcal{S}_{{\rm Haar}_1^{\otimes n}}$ or $\mathcal{S}_{{\rm Stab}_1^{\otimes n}}$."

Gibbs et al. (2204.10269) train on $|\Psi_{\rm P}^{(j)}\rangle = \bigotimes_{i=1}^n |\psi_i^{(j)}\rangle$ "where the states [...] are independently drawn from the single-qubit Haar distribution", and note that a 1-qubit 2-design is enough. Their numerics need very few inputs: "We found that 5 input states were sufficient to perform a full Hilbert space diagonalization" (4-qubit Heisenberg with PBC), and "only a single training state was required to diagonalize over the entire Hilbert space" (XY with OBC, $n=4$ to 12, local cost).

### 3.15 Checked and not on point

- Zaletel, Mong, Karrasch, Moore, Pollmann (1407.1832, PRB 91, 165112) say the $W^{I/II}$ MPOs have "a constant error per site in the thermodynamic limit". They compare at $L=20, 40$ from Néel and random states, but the TeX gives no error formula.
- Mc Keever and Lubasch (2311.05544): adiabatic state preparation, not TI.
- McCulloch (0804.2509): iDMRG "orthogonality fidelity" between iterations, not per site.
- Liu, Wei, Foss-Feig, Gorshkov, Childs, "Ordering-Aware Theory of Trotter Error" (2610.03672): spectral-norm Trotter bounds on TI open chains. Whole-system, no fidelity density.

## 4. How the literature conventions relate to ours

Our conventions, for comparison:

- Finite system: $c = 1 - \overline{F_k^{1/N}}$, with $F_k = |\langle{\rm target}_k|V\phi_k\rangle|^2$ and $\phi_k$ Haar-random product states.
- Infinite chain: $c_{\rm inf} = 1 - |\lambda_0|^{2/L}$, with $\lambda_0$ the leading eigenvalue of the mixed TM between the circuit's iMPS and the reference iMPS over an $L$-site cell. The inputs are 32 random product states on an 8-site cell, repeated along the chain ($L_u=8$).

Write $x = |\lambda_0|^{2/L}$ for our per-site squared fidelity of one input, so $c_{\rm inf} = 1-x$. Conversions for one input, to first order in $c_{\rm inf}$:

- ZOV, Zhou, Vanderstraeten, Karrasch (amplitude per site): $d = |\lambda_0|^{1/L} = \sqrt{x}$, so $1-d = 1-\sqrt{1-c_{\rm inf}} \approx c_{\rm inf}/2$. Our $c_{\rm inf}$ is twice the amplitude-based per-site infidelity.
- Astrakhantsev et al.: the overlap density $|\lambda|$ is $d$. The fidelity density "squared overlap per unit cell" with their 1-site cell is $\mathcal F = x$, so $1-\mathcal F = c_{\rm inf}$ exactly. This is the closest published match to our $c_{\rm inf}$.
- Gibbs and Cincio $C_3$ (per input, 2-site cell): $1-|\lambda_0|^2 = 1-x^2 = c_{\rm inf}(2-c_{\rm inf}) \approx 2c_{\rm inf}$. Their Clifford+T example, $C_3 = 3.082\times10^{-7}$, corresponds to $c_{\rm inf}\approx 1.54\times10^{-7}$ in our convention, for the TFIM at $h=1$, $t=2$. Any comparison of our infinite-chain numbers with their figures needs this factor of 2.
- Gibbs and Cincio $C_4$ (rings): the same formula as our finite $c$, a mean over inputs of $F_k^{1/N}$ with $F_k$ squared. Only the inputs differ: theirs are 8 Haar-random global states, ours Haar-random product states.
- Zhang et al. $\tilde C = 1-(\overline{F_k})^{1/n}$ against our $c = 1-\overline{F_k^{1/n}}$. Because $y\mapsto y^{1/n}$ is concave, $\overline{F_k^{1/n}} \le (\overline{F_k})^{1/n}$, so $c \ge \tilde C$, with equality when all $F_k$ are equal. When $n c_k \ll 1$ for every input both reduce to $\overline{c_k}$. When $n c_k$ is not small, $\tilde C$ is pulled toward the best inputs. The inputs are the same ensemble as ours.
- Heyl and Karrasch rate, applied to the error overlap: $l = -\frac1N\ln|\langle a|b\rangle|^2 = -\ln x = c_{\rm inf} + c_{\rm inf}^2/2 + \dots$. To first order the rate form and our $1-f$ form agree.
- Van Damme et al.: $\mathrm{Re}\,f = \ln d = \frac12\ln x \approx -c_{\rm inf}/2$.
- De Franco et al.: $f = -\log_2 f_{\rm op}\approx (1-f_{\rm op})/\ln 2$, with $f_{\rm op}$ the per-site Hilbert-Schmidt amplitude.
- Causer et al.: $\delta^2 = 2(1-f_{\rm op})$.

The Hilbert-Schmidt-based operator densities correspond to Haar-random global inputs, not product inputs. The Haar-averaged fidelity is $\bar F = (2^N|\mathrm{Tr}\,U^\dagger V/2^N|^2+1)/(2^N+1)$ (Mizuta Eq. 12; Gibbs and Cincio 2409.16361 Eq. 2). So $1-\bar F^{1/N}\approx 1-f_{\rm op}^2 \approx \delta^2$ when $2^N f_{\rm op}^{2N}\gg1$ (taking the trace real and positive, as Causer's Re does). None of the papers gives a per-site relation between product-state and Haar-state values. Caro's factor-2 bounds are on whole-system risks, which tend to 1 when $N c\gg1$. My reading is that they do not carry over to per-site numbers; no paper I read states this either way.

### Which of our choices have precedent

- **Leading eigenvalue of the mixed TM, root by cell length.** Precedent. ZOV, Vanderstraeten and Astrakhantsev use the eigenvalue for a 1-site cell. Karrasch and Schuricht take "the $N$-th root of the dominant eigenvalue" for an $N$-site cell. Gibbs and Cincio take no root, which is why their number is per 2-site cell.
- **Squared overlap per site.** Precedent, but not universal. Astrakhantsev's fidelity density, the Heyl and Karrasch rates in log form, Zhang's $\tilde C$ and Gibbs and Cincio's $C_4$ are all squared and per site. The TN fidelity literature (ZOV, Vanderstraeten, Van Damme) and the operator densities (Causer, De Franco) use the amplitude instead. Our $c_{\rm inf}$ is about twice an amplitude-based per-site infidelity and about half of Gibbs and Cincio's $C_3$.
- **Many random inputs on the infinite chain, not one physical initial state.** Precedent only in Gibbs and Cincio 2506.02105: random iMPS with $\chi=2$ on a 2-site cell, 4 train and 100 test. They mention $\chi=1$ as "previous work" and report it generalizes worse in training. They never test on $\chi=1$ states; their test set is always $\chi=2$. All other infinite-chain papers use one physical state (Néel, polarized, ground state).
- **Product states on an 8-site cell with a period-2 circuit.** No precedent found. In every infinite-chain paper the input cell equals the state or circuit cell (1 or 2 sites). Gibbs and Cincio's $\chi=1$ random iMPS on a 2-site cell, which are product states with period 2, are the nearest case. With an 8-site input cell the joint TM spans 8 sites, and taking $|\lambda_0|^{2/8}$ follows the Karrasch and Schuricht prescription. On whether product inputs are enough, the theory papers (Caro; Gibbs 2204.10269) cover finite $n$ and locally scrambled ensembles. Under Caro's definition, a TI ensemble of states repeated with a fixed period is not invariant under arbitrary site-dependent local unitaries. It does not match a locally scrambled ensemble up to second moments either, because sites $i$ and $i+8$ carry the same random state. So I read the theorems as not covering our ensemble or Gibbs and Cincio's random iMPS; this is my reading, not a statement in any paper. Gibbs and Cincio still cite Caro for their $\mathcal{O}(L\log L)$ count.
- **32 inputs.** Inside the published range, though no paper derives a number. Gibbs and Cincio use 4 train and 100 test on the infinite chain and 8 on rings. Zhang et al. use 100 test and say $N_s<50$ suffices for training, with sampling error $\mathcal O(N_s^{-1/2})$. Mc Keever and Lubasch use 50 bitstrings. Gibbs et al. 2204.10269 needed 1 to 5.
- **Rings as stand-ins.** Gibbs and Cincio check on PBC rings with exact state vectors and 8 Haar states. Kotil, Mansuroglu and Mizuta train on small rings (6, 6, 20 sites) and move to larger systems. All of those report whole-system errors, not per-site ones.
- **Finite $c$ on Haar product states.** The same formula as Gibbs and Cincio's $C_4$ (different inputs) and the same inputs as Zhang et al. (root of the mean rather than mean of roots).

### Explicit statements of "standard" or of reasons

- Le, Sun, Mendl: "A common way [...] is given by the normalized Frobenius norm". This is whole-system.
- De Franco et al.: "A standard choice for the cost function is the fidelity between the target operator [...] and an Ansatz circuit". They then switch to the per-site log density "To remove the scaling with $N$".
- ZOV: the fidelity per site "is well defined in the thermodynamic limit (even though the fidelity itself becomes zero)".
- Vanderstraeten et al.: "A more physical quantity to express whether two MPS in the thermodynamic limit are `close', is the fidelity per site".
- Gibbs and Cincio: the global fidelity "takes a binary value of 0 or 1 for normalized states. Therefore, we instead compute a local measure of fidelity per qubit". On inputs, $\chi_{\rm Train}=2$ "has improved generalization properties, with a minimal increase in computational cost". On rings, "We modify the calculation to compute a local infidelity for a more similar measure to our earlier numerical results."
- Astrakhantsev et al.: the global fidelity "is either $1$ or $0$ in the thermodynamic limit and cannot be used for posing the optimization problem".
- Dborin et al.: the overlap "is formally zero. However, the distance between states can be quantified by the rate at which this overlap tends to zero".
- Caro et al.: training on Haar or stabilizer product states "makes practical sense" because they need one layer of single-qubit gates.

None of these papers calls a particular per-site convention, such as amplitude against square, or per site against per cell, the standard one.

## 5. What I could not access or verify

- I could not search systematically for papers that cite 2506.02105. The Semantic Scholar API returned HTTP 429 four times. arXiv API keyword searches ("fidelity density", "infidelity per site", "local infidelity" with "infinite", iMPS with compilation) found no further infinite-chain compilation paper. Newer work may exist that I missed.
- Barratt et al. draw the mixed TM only as a figure. I rendered `EqB16Fig.pdf` with Ghostscript to see that the TDVP-step block holds two state unitaries. The text states no normalization by cell length.
- Gibbs and Cincio 2506.02105 do not say how the random iMPS tensors are drawn. I did not look for their code.
- Astrakhantsev et al. do not say whether their fidelity density against the 2-site iTEBD reference is per site or per 2-site cell.
- Mansuroglu et al. give neither the number nor the distribution of their "random input vectors". Kotil et al. give the larger ring sizes only in a figure, which I did not render.
- Dborin et al. (2205.12996) has no journal reference on arXiv. I did not check a published version. The same holds for Zhou and Barjaktarevič (cond-mat/0701608), Van Damme et al. (2302.14181) and De Franco et al. (2607.14296). For all papers I read the latest arXiv TeX, not the journal text.
- Equation numbers for papers without arXiv HTML (0709.4596, 1206.2505, 1302.3893, 2210.03751, 2106.03680, 2205.11427, 2212.07556, 2203.15484, 2204.10268, 2204.10269, 2205.12996) come from counting numbered displays in the TeX and may be off by one. For Van Damme et al. and De Franco et al. I give section numbers and TeX labels only, because the counts were unreliable.
- Heyl et al. evaluate their rate exactly; I did not check how follow-up iMPS works on DQPTs (for example Halimeh and Zauner-Stauber) normalize by cell length, beyond Karrasch and Schuricht.
