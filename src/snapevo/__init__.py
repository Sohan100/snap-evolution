"""SNAP compilation of XXZ time evolution with the corrected exact-angle library.

Pipeline (docs/plan.md): train a translation-invariant brickwork on a ring
(train.py), snap its rotations onto Isaac's corrected RUS library with the
Hessian-ranked protocol (snap.py), plug in the Clifford+T circuits and fit the
open-chain edge rotations (realize.py, edges.py), and measure the state
infidelity at N = 100 with MPS (evaluate.py, mps.py). baseline.py builds the
product-formula + KN comparison the same way. crosscheck.py keeps the
small-ring metrics as checks.
"""

__version__ = "0.1.0"
