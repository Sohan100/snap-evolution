import numpy as np
import scipy.optimize as sopt

from snapevo.cells import CELLS
from snapevo.circuits import Layered, local_infidelity, state_pairs
from snapevo.hamiltonian import XXZ
from snapevo.jaxcost import make_cost, make_hessian
from snapevo.library import Library
from snapevo.snap import SnapSettings, compile_snap


def test_small_compile_meets_budget():
    model, cell, N, D, t = XXZ(0.25, 0.0), CELLS["xxz+stag"], 6, 4, 0.5
    tin, tg = state_pairs(model, N, t, 6, 0)
    f = make_cost(cell, N, tin, tg, D)
    x = sopt.minimize(f, cell.trotter_angles(model, t, D), jac=True, method="L-BFGS-B").x
    budget = 3 * max(f(x)[0], 1e-6)

    def c_fn(y):
        return local_infidelity(Layered.brickwork(cell, y).apply(tin, N), tg, N)

    res = compile_snap(x, cell, SnapSettings(budget=budget, max_round_trials=8), f, c_fn,
                       make_hessian(cell, N, tin, tg, D, method="jax"), tin, tg, N, Library(22.0))
    assert res["C_train"] <= budget * (1 + 1e-9)
    lib = Library(22.0)
    for i, s in enumerate(res["snapped"]):
        if s:
            assert lib.lookup(res["angles"][i]) is not None
    assert np.isfinite(res["Tq_plan"])
