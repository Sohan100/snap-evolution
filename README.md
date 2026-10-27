# snap-evolution

Compilation of quantum time evolution into Clifford+T circuits with SNAP, applied to the one-dimensional XXZ model. The repository contains the circuit training, the SNAP compiler, the synthesis of the remaining rotations, the error evaluation and the product-formula baselines behind the time-evolution results, together with the run outputs and figures.

## Overview

The target is the evolution operator $U = e^{-iHt}$ of the XXZ chain

$$H = \sum_i \left( S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + \Delta\, S^z_i S^z_{i+1} \right) + h_x \sum_i S^x_i ,$$

with $\Delta = 0.25$ and $J = 1$ in spin units, at $h_x = 0$ and $h_x = 1.6$, for evolution times $t = 1$ to $4$. Every stage works on the infinite chain, so the circuits and their errors are translation invariant and independent of system size.

1. **Training.** A brickwork circuit with a two-site unit cell, in which every bond of a sublayer carries the same gate, is trained depth by depth to approximate $U$. The cost is the infidelity per site, obtained from the leading eigenvalue of the mixed transfer matrix between the circuit output and the exactly evolved state (iTEBD), averaged over four random iMPS of bond dimension 2. Each depth is optimized by trust-region Newton with the exact gradient and Hessian, from three starting points.
2. **Compilation.** For a given error budget, SNAP moves rotation angles onto exact Clifford+T angles from a precomputed library, starting with Clifford angles at zero T cost. Candidates are ranked with the Hessian, and the remaining angles are re-optimized after every accepted move. Rotations that are not moved are synthesized individually, each at a precision set by its sensitivity.
3. **Synthesis.** Remaining rotations use probabilistic synthesis with a fallback branch [1]. Fallback events enter in expectation, as independent local defects, which keeps the evaluation translation invariant.
4. **Evaluation.** Every circuit is scored by its infidelity per site, $1 - F$, averaged over 100 random iMPS that are not used in training, together with its expected T count per qubit and per rotation.
5. **Baselines.** Product formulas of order 1, 2 and 4, scanned over the number of steps and the synthesis precision, with every rotation synthesized either by the same fallback synthesis or by GridSynth [2]; and uniform-precision synthesis of the trained circuits [3]. All baselines are evaluated with the same metric on the same test states.

## Installation

Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,plot]"
pytest
```

## Usage

Each command takes the field $h_x$ (`1.6` or `0`) and the evolution time $t$, and resumes from its checkpoints when rerun.

```
python cluster/imps_train.py   1.6 3 --workers 32     # train the depth ladder
python cluster/imps_snap.py snap    1.6 3 --workers 32   # SNAP at the error budgets 1e-1 ... 1e-6
python cluster/imps_snap.py uniform 1.6 3 --workers 32   # uniform-precision synthesis of the same circuits
python cluster/imps_snap.py reeval  1.6 3 --workers 32   # re-evaluate every realized circuit
python cluster/ti_trotter.py    1.6 3 --workers 32     # product formulas with fallback synthesis
python cluster/ti_trotter_gs.py 1.6 3 --workers 32     # the same formulas with GridSynth
python cluster/ti_paper_figs.py --times 1,2,3,4        # figures
```

Results are written to `runs/xxz_d0.25_hx<h_x>/t<t>/`: the trained ladder in `imps_train/`, the compiled, realized and evaluated circuits in `imps_snap/`, and the uniform-precision circuits in `imps_uniform/`. The product-formula scans are in `runs/ti/imps_trotter/` and `runs/ti/imps_trotter_gs/`, and the figures in `runs/figures/ti/paper/`.

### Running at scale

The scripts in `cluster/` run the same steps under Slurm. `imps_chain.slurm` compiles one ladder on a compute node, and `imps_points.slurm` adds further workers to the same ladder: each budget and depth is locked while it is compiled, so any number of workers share the remaining points without duplicating work. `imps_split.py` trains the starting points of a depth on separate nodes. Jobs stop shortly before their time limit, checkpoint, and resubmit themselves.

## Repository layout

```
src/snapevo/
  hamiltonian.py      XXZ model, exact evolution
  cells.py            brickwork cells and their gates
  imps.py             iMPS, mixed transfer matrices and environments
  imps_train.py       infinite-chain training: cost, gradient, Hessian, depth ladder
  newton_fast.py      trust-region Newton and the L-BFGS comparison optimizer
  imps_snap.py        SNAP on the infinite chain: single-angle cost curves, fallback defects, evaluation
  chain_snap.py       the SNAP compiler
  library.py          exact-angle Clifford+T library and its pricing
  synthesis.py        synthesis wrapper with a shared cache
  realize.py          substitution of the Clifford+T circuits
  fallbacks.py        fallback branches of the synthesized rotations
  baseline.py         product formulas
cluster/              command-line entry points and Slurm scripts
tests/                test suite
docs/                 conventions, cost model, error metric and validation notes
third_party/          the rotation library and the synthesis package, unmodified, with SHA256 checksums
runs/                 run outputs and figures
```

The finite-system workflow used earlier in the project (ring and open-chain training, MPS evaluation) remains available through `python -m snapevo` and is documented in `docs/`.

## References

1. arXiv:2203.10064, probabilistic single-qubit synthesis with fallback.
2. arXiv:1403.2975, optimal ancilla-free Clifford+T approximation of $z$-rotations (GridSynth).
3. arXiv:2506.02105, compression and synthesis of time-evolution circuits on the infinite chain.
