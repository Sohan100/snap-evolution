import math

from snapevo import library_check as lc
from snapevo.library import Library, gadget_cost, load_gadgets
from snapevo.synthesis import kn_eps


def test_rows():
    g = load_gadgets()
    assert len(g) == 4580
    assert sum(x.source.endswith("_oddshift") for x in g) == 2290
    # evaluate.fidelity_bound needs q <= 1 (variance of the fallback count <= 2q)
    assert min(x.p for x in g) >= 0.9 and Library(18.0).Q.max() <= 1.0


def test_lattice_matches_shipped_file_at_F18():
    assert lc.check_shipped_library() < 1e-6


def test_every_row_checks_out():
    bad = lc.check_rows()
    assert bad.pop("rows checked") == 4580
    assert not {k: v for k, v in bad.items() if v}


def test_so3_exponent_is_the_t_count():
    n, mism = lc.bfs_check(6)
    assert n == 24 * (3 * 2 ** 6 - 2) and mism == 0


def test_fallback_error_law():
    assert math.isclose(kn_eps(18.0), 1.889e-4, rel_tol=2e-3)
    assert Library(18.0).delta_fb > Library(22.0).delta_fb


def test_gadget_cost_strategies():
    c, q = gadget_cost(16.0, 0.95, "2+T", 18.0)          # single-shot wins for 2+T
    assert math.isclose(c, 16.0 + 0.05 * 18.0) and math.isclose(q, 0.05)
    c, q = gadget_cost(10.0, 0.95, "Clifford", 18.0)     # repeat, exact
    assert math.isclose(c, 10.0 / 0.95) and q == 0.0
