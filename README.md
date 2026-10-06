# snap-evolution

SNAP compilation of XXZ time evolution onto Isaac's corrected exact-angle library, with errors measured by plugging in the Clifford+T circuits. This is the time-evolution code for paper 1. It uses no catalysis and only unmixed (deterministic) synthesis, as decided on the 5 Oct 2026 call.

The pipeline trains a translation-invariant brickwork for $e^{-iHt}$ and snaps its rotations onto the RUS library with the Hessian-ranked protocol. It then synthesizes the leftover rotations with Kliuchnikov non-mixed and measures the realized circuit's operator-norm and diamond error at $N = 6$ to $16$. That error is extrapolated to the 100-site ring. A product-formula baseline goes through the same measurement.

## Install

Python 3.10 or newer.

```
python -m venv .venv
.venv\Scripts\activate            (on Linux or macOS: source .venv/bin/activate)
pip install -e .[test]
python -m snapevo verify-library  (independent check of the library file, ~30 s)
pytest                            (16 tests, ~1 min)
```

## Run one point

```
python -m snapevo train    --model B --t 2 --depths 7,9,11,13
python -m snapevo point    --model B --t 2 --budget 1e-4 --D 7
python -m snapevo baseline --model B --t 2 --order 2 --steps 4 --eps 1e-4
python -m snapevo report   --model B --t 2
```

`--model A` is the ring with $h_x = 1.6$ and `--model B` the one without (spin units, $\Delta = 0.25$). Times are in spin units. `point` runs `compile`, `realize` and `evaluate`, which also exist as separate commands. Output goes to `runs/` (`docs/conventions.md` lists the files and their keys).

## Layout

```
docs/plan.md            the simulation plan: models, stages, budgets, order of work
docs/error_metric.md    what error we report and why (unmixed method)
docs/cost_model.md      library and KN pricing, provenance, what is left out
docs/conventions.md     units, gates, angle order, file formats
docs/validation.md      checks run on the library, the port and the smoke run
src/snapevo/
  hamiltonian.py        models A and B, exact evolution
  cells.py              brickwork cells (xxz+stag, xxz+x)
  circuits.py           state-vector simulation, local infidelity C, state sets
  jaxcost.py            JAX value, gradient and Hessian of C
  train.py              depth-ladder training
  library.py            corrected RUS library, pricing at fallback cost F
  synthesis.py          KN non-mixed wrapper, calibrated cost law, cache
  environment.py        exact single-angle C curves
  snap.py               Hessian-ranked SNAP compiler
  realize.py            plug in the Clifford+T circuits, error ledger
  metrics.py            held-out C, exact op norm by Lanczos, N extrapolation
  baseline.py           product formulas + KN, measured the same way
  library_check.py      independent verification of the library file
  cli.py, io.py         command line, run-file layout
third_party/            Isaac's two packages from 2026-10-05, unmodified, with SHA256SUMS
tests/                  pytest suite
```

## Provenance

`snap.py`, `environment.py`, `train.py` and the JAX cost are ported from `tcount_tfim` (work\rust, Sep 2026). On the same cell, states and library the ported compiler reproduces the old one exactly (`docs/validation.md`). Catalysis, the $k\pi/12$ angles, the environment-criterion escalation and the tier ladders were left behind. The library and the KN synthesizer are Isaac Kim's, unchanged.
