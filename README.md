# snap-evolution

SNAP compilation of XXZ time evolution onto Isaac's corrected exact-angle library, with errors measured by plugging in the Clifford+T circuits. This is the time-evolution code for paper 1. It uses no catalysis and only unmixed (deterministic) synthesis, as decided on the 5 Oct 2026 call.

The pipeline trains a translation-invariant brickwork for $e^{-iHt}$ on a ring and snaps its rotations onto the RUS library with the Hessian-ranked protocol. It then synthesizes the leftover rotations with Kliuchnikov non-mixed, fits rotations for the two ends of an open chain, and computes the state infidelity of the realized circuit on random product inputs at $N = 100$ with MPS, against an MPS of the exact evolution. Nothing is extrapolated, and the heralded fallback branches are bounded exactly. A product-formula baseline goes through the same evaluation. The small-ring metrics we used before (operator norm, diamond norm, process fidelity) remain as cross-checks.

## Install

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate            (on Linux or macOS: source .venv/bin/activate)
pip install -e .[test]
python -m snapevo verify-library  (independent check of the library file, ~30 s)
pytest                            (27 tests, ~2 min)
```

## Run one point

```
python -m snapevo train         --model B --t 2 --depths 7,9,11,13
python -m snapevo point         --model B --t 2 --budget 1e-4 --D 7
python -m snapevo baseline-scan --model B --t 2 --evaluate
python -m snapevo report        --model B --t 2
python -m snapevo summary                          (every point of every model and time, one CSV)
```

`--model A` has $h_x = 1.6$ and `--model B` has no field (spin units, $\Delta = 0.25$). Times are in spin units. `point` runs `compile`, `realize` and `evaluate`, which also exist as separate commands, and `evaluate` builds the MPS references on first use (`reference` builds them ahead). Output goes to `runs/` (`docs/conventions.md` lists the files and their keys). The full sweep up to $t = 10$ runs on a cluster from `cluster/` (`cluster/README.md`).

## Layout

```
docs/plan.md            the simulation plan: models, the t sweep, stages, where it runs
docs/error_metric.md    the reported error (MPS state infidelity at N = 100), edges, fallbacks, cross-checks
docs/cost_model.md      library and KN pricing, provenance, what is left out
docs/conventions.md     units, gates, angle order, file formats
docs/validation.md      checks run on the library, the port, the MPS code and the first runs
src/snapevo/
  hamiltonian.py        models A and B on a ring or an open chain, exact evolution
  cells.py              brickwork cells (xxz+stag, xxz+x)
  circuits.py           state-vector simulation (ring or open chain), C, held-out C, state sets
  jaxcost.py            JAX value, gradient and Hessian of C
  train.py              depth-ladder training
  library.py            corrected RUS library, pricing at fallback cost F
  synthesis.py          KN non-mixed wrapper, calibrated cost law, cache
  environment.py        exact single-angle C curves
  snap.py               Hessian-ranked SNAP compiler
  realize.py            plug in the Clifford+T circuits, error ledger
  edges.py              edge rotations for the open chain: fit, keep or drop, KN
  mps.py                open-chain MPS, circuits and the order-4 reference, cached references
  evaluate.py           MPS state infidelity at N = 100, fallback bound, choice of F
  baseline.py           product formulas + KN, scan and Pareto front
  crosscheck.py         small-ring op norm, diamond and process fidelity (checks)
  library_check.py      independent verification of the library file
  cli.py, io.py         command line, run-file layout
third_party/            Isaac's two packages from 2026-10-05, unmodified, with SHA256SUMS
cluster/                job lists, SLURM array script, submit and local runners for the t sweep
tests/                  pytest suite
```

## Provenance

`snap.py`, `environment.py`, `train.py` and the JAX cost are ported from `tcount_tfim` (work\rust, Sep 2026). On the same cell, states and library the ported compiler reproduces the old one exactly (`docs/validation.md`). Catalysis, the $k\pi/12$ angles, the environment-criterion escalation and the tier ladders were left behind. The library and the KN synthesizer are Isaac Kim's, unchanged.
