# Conventions and file formats

## Gates

$$
R_z(\theta) = e^{-i\theta Z/2} = \mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2}), \qquad R_x(\theta) = e^{-i\theta X/2}, \qquad T = \mathrm{diag}(1, e^{i\pi/4}) .
$$

Every library cost and every synthesis cost is for $R_z(\theta)$ at the angle $\theta$ that appears in the formula above.

## Units

The Hamiltonians are in spin units, $S = \sigma/2$ (`hamiltonian.py`). In Pauli operators the bond coefficients are $J/4$ on $XX$ and $YY$ and $J\Delta/4$ on $ZZ$, and the field coefficient is $h_x/2$ on $X$. At $h_x = 0$ the Sep 2026 Pauli-unit Hamiltonian is four times ours, so Pauli time $\tau$ is spin time $4\tau$.

## Brickwork

Qubit 0 is the most significant bit of a state index. Sublayer $l$ applies one cell gate to every bond of parity $l \bmod 2$. Even sublayers use bonds $(0,1), (2,3), \dots$ and odd sublayers use $(1,2), \dots, (N-1, 0)$. $N$ must be even. The flat angle vector holds the angles of sublayer 0, then sublayer 1, and so on, in the cell's slot order:

| cell | slots | Rz per bond per slot |
|---|---|---|
| `xxz+stag` | xx, yy, zz, stag | 1, 1, 1, 2 |
| `xxz+x` | xx, yy, zz, x1, x2 | 1, 1, 1, 1, 1 |

The gate is $e^{-i(aXX + bYY + cZZ)/2}$ times the single-qubit part, and the single-qubit part acts first. The staggered part is $R_z(d) \otimes R_z(-d)$, which costs two rotations of the same library price. The field part is $R_x(e) \otimes R_x(f)$.

On the open chain (the evaluation) the bond $(N-1, 0)$ does not exist. In each odd sublayer its gate is replaced by a single-qubit gate on site $N-1$ and one on site 0. By default these are the cell's single-qubit factor, the wrap gate with its coupling removed. Fitted edges replace them by $R_z(\alpha) R_y(\beta) R_z(\gamma)$, stored as six angles per odd sublayer, site $N-1$ first.

A first-order Trotter step of length $\tau$ is two sublayers with $a = b = J\tau/2$, $c = J\Delta\tau/2$ and $e = f = h_x \tau/2$. A depth-$D$ cell for time $t$ starts from $\tau = 2t/D$.

## State sets

Every input is a product of random single-qubit states, drawn site by site from one seed (`circuits.product_vectors`). The sets are as follows:

| use | seed | states | size |
|---|---|---|---|
| training and compiling | 0 | 12 | ring, $N = 10$ |
| ladder `C_val` | 999 | 32 | ring, $N = 10$ |
| held-out $C$ | 54321, 777, 2468, 13579 | 16, 64, 64, 64 | ring, $N = 10$ and 12 |
| edge fit | 31415 | 12 | open chain, $N_{\rm fit}$ |
| reported error | 2026 | 8 | open chain, $N = 100$ |

The draws are the same as in the Sep 2026 code, so old and new numbers compare state for state.

## Signs in the library file

The literal gadget circuit applies $R_z(-\theta)$ for `angle_rad` $= \theta$. Odd-shift rows list $\theta_{\rm same} - \pi/4$. `theta_f_pi` is $\pi$ minus the failure branch's angle. Only the first fact affects anything, and only if someone builds the circuits literally (`docs/cost_model.md`).

## Run files

All run files live in `runs/<model tag>/t<t>/` (`io.py`). The JSON keys are as follows:

`cells.json`
: `model`, `cell`, `t`, `N`, `train_set`, `val_set`, and `results` as a list of `{D, params (D x slots), C_train, C_val, start}`.

`snap_b<B>_D<D>.json`
: `settings`, `floor`, `C_train`, `angles`, `snapped`, `costs_plan`, `multiplicity`, `Tq_plan`, `stragglers {i: {eps, T_plan, dC_plan}}`, `library_stragglers`, counters, `trace`, `history` (every accepted prefix), `seconds`, `model`, `cell`, `t`, `D`, `N`.

`real_b<B>_D<D>.json`
: `angles_compiled`, `angles_real`, `multiplicity`, `C_train_real`, `straggler_eps_scale`, `lattice {i: {kind, cost[F], q[F]}}`, `kn {i: synthesis record}`, `per_F {F: {Tq, ledger_per_qubit, ledger2_per_qubit, delta_fb}}`, `rotations_per_qubit`, counts, and `edges` (below).

`edges` (inside the realized file)
: `use`, `N_fit`, `states`, `angles` (fitted, then snapped), `angles_real` (after KN, or null when not kept), `infid_start`, `infid_fit_unsnapped`, `infid_fit`, `infid_ring` (all at $N_{\rm fit}$ on the fit states), `n_snapped`, `eps`, `kn {i: synthesis record}`, `n_rotations`, `n_kn`, `n_clifford_t` (angles at multiples of $\pi/4$), `T_total`, `ledger_total`, `ledger2_total`.

`eval_b<B>_D<D>.json`
: `N`, `mps` (per input: `F`, `infid`, `infid_lo`, `infid_hi`, discarded weights, bond dimensions, seconds, then `infid`, `stderr`, `c_site`, `trunc_rel`, `trunc_flag`, `chi_peak` and the MPS settings), `C_ring` and `infid_ring_pred`, `per_F {F: {Tq, ledger, Q2, infid_total}}`, the chosen `F`, `infid`, `infid_total`, `fallback_share`, `fallback_flag`, `c_site`, `Tq`, `T_edges_per_qubit`, `rotations_per_qubit`, `T_per_rotation`, `T_per_qubit_per_time`, rotation counts.

`baseline_o<order>_r<steps>_e<eps>.json`
: the realized baseline (`Tq`, ledgers, `rotations_per_qubit`, every synthesis record) and the same evaluation keys as an eval file, with the single pricing label `KN`. After `snapevo sample`, also `sampled` (`infid`, `stderr`, `c_site`, `samples`, `seed`, `fallbacks_mean`, `infid_draws`) and its copies `infid_sampled`, `c_sampled`. With `--crosscheck`, also `crosscheck {trotter, realized}`.

`baseline_scan.json`
: every screened `(order, steps, eps)` with its ring `C`, Trotter-only `C_trotter` and `Tq`, and the Pareto `front`.

`xcheck_b<B>_D<D>.json`
: the small-ring cross-checks: `held_out`, `arcs` (per $N$: $w$, $\epsilon_{\rm op}$, $D_{\rm coh}$), `process` (per $N$: $F_e$, $c_e$), `alpha`, `extrapolation`, `fidelity`, and the totals `D_total`, `state_infid_total`, `process_infid_total` at $N = 100$ on the ring.

`ref/ref_N<N>_s<seed>_k<k>_dt<dt>_c<cutoff>_x<chi_max>.npz`
: the cached MPS of $e^{-iHt}$ on input $k$: tensors `A0`, `A1`, ... and a JSON `meta` with the settings, discarded weight, peak bond dimension, center and seconds.

`Tq` is expected T per qubit for the whole circuit (time $t$), with the edge rotations divided over the $N$ qubits. T per rotation divides by the rotations per qubit, 2.5 per sublayer times $D$ plus the edge rotations over $N$.
