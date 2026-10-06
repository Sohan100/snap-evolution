"""SNAP compilation of XXZ time evolution with the corrected exact-angle library.

Pipeline (docs/plan.md): train a translation-invariant brickwork (train.py),
snap its rotations onto Isaac's corrected RUS library with the Hessian-ranked
protocol (snap.py), plug in the Clifford+T circuits (realize.py), and measure
the error directly (metrics.py). baseline.py builds the product-formula + KN
comparison the same way.
"""

__version__ = "0.1.0"
