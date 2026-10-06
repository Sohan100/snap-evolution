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

A first-order Trotter step of length $\tau$ is two sublayers with $a = b = J\tau/2$, $c = J\Delta\tau/2$ and $e = f = h_x \tau/2$. A depth-$D$ cell for time $t$ starts from $\tau = 2t/D$.

## State sets

Training uses 12 product states with seed 0 at $N = 10$. The ladder's `C_val` uses 32 states with seed 999. Held-out sets are seeds 54321 (16 states), 777, 2468 and 13579 (64 states each), at $N = 10$ and $12$. The draws are the same as in the Sep 2026 code, so old and new numbers compare state for state.

## Signs in the library file

The literal gadget circuit applies $R_z(-\theta)$ for `angle_rad` $= \theta$. Odd-shift rows list $\theta_{\rm same} - \pi/4$. `theta_f_pi` is $\pi$ minus the failure branch's angle. Only the first fact affects anything, and only if someone builds the circuits literally (`docs/cost_model.md`).

## Run files

All run files live in `runs/<model tag>/t<t>/` (`io.py`). The JSON keys are as follows:

`cells.json`
: `model`, `cell`, `t`, `N`, `train_set`, `val_set`, and `results` as a list of `{D, params (D x slots), C_train, C_val, start}`.

`snap_b<B>_D<D>.json`
: `settings`, `floor`, `C_train`, `angles`, `snapped`, `costs_plan`, `multiplicity`, `Tq_plan`, `stragglers {i: {eps, T_plan, dC_plan}}`, `library_stragglers`, counters, `trace`, `history` (every accepted prefix), `seconds`, `model`, `cell`, `t`, `D`, `N`.

`real_b<B>_D<D>.json`
: `angles_compiled`, `angles_real`, `multiplicity`, `C_train_real`, `straggler_eps_scale`, `lattice {i: {kind, cost[F], q[F]}}`, `kn {i: synthesis record}`, `per_F {F: {Tq, ledger_per_qubit, ledger2_per_qubit, delta_fb}}`, `rotations_per_qubit`, counts.

`eval_b<B>_D<D>.json`
: `held_out`, `arcs` (per $N$: $w$, $\epsilon_{\rm op}$, $D_{\rm coh}$, timing), `process` (per $N$: $F_e$, per-site $c_e$, method), `alpha`, `extrapolation`, `fidelity` (success branch: `C`, `c_e`, `state_infid_10`, `state_infid`, `process_infid`), `per_F`, the chosen `F`, and the totals `D_total`, `state_infid_total`, `process_infid_total`, with `ledger`, `Q2`, `D_coh`, `Tq`, `T_per_rotation`, `T_per_qubit_per_time`.

`baseline_o<order>_r<steps>_e<eps>.json`
: `trotter` and `realized`, each a full measurement as in the eval file, then `Tq`, the ledgers, the same three totals, and every synthesis record.

`Tq` is expected T per qubit for the whole circuit (time $t$). T per rotation divides by 2.5 Rz per qubit per sublayer times $D$.
