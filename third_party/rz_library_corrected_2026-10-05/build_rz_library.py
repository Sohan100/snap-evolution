"""Build the searchable Rz library from the gadget CSVs.

Standalone, numpy only.

    import build_rz_library as rl
    S, C = rl.build("rz_rus_gadgets.csv", "rz_catalyst_angles.csv",
                    cat_levels=3)
    ang, cost = rl.snap_nearest(S, C, 0.7371)          # nearest, any cost
    ang, cost = rl.snap_cheapest(S, C, 0.7371, 5e-3)   # cheapest within +-w

Convention: Rz(theta) = diag(e^{-i theta/2}, e^{+i theta/2}). A library angle
is implemented exactly, so the only error of a snap is the known
over-rotation theta - theta_lib.
"""

import csv
import math

import numpy as np

TWO_PI = 2.0 * math.pi
QUARTER = math.pi / 4.0


def load_entries(gadget_path, catalyst_path=None, fallback_t=18.0):
    """Return undressed [(angle, expected_T), ...].

    fallback_t prices the generic correction of a "2+T" failure branch
    (18 = Kliuchnikov non-mixed, 50 = conservative Ross-Selinger).
    """
    out = []
    with open(gadget_path, newline="") as fh:
        for e in csv.DictReader(fh):
            t, p = float(e["t_exact"]), float(e["p"])
            reset = {"Clifford": 0.0, "T": 1.0, "2+T": fallback_t}[e["correction"]]
            repeat = t / p + reset * (1 - p) / p
            single = t + (1 - p) * fallback_t
            out.append((float(e["angle_rad"]), min(repeat, single)))
    if catalyst_path is not None:
        with open(catalyst_path, newline="") as fh:
            for e in csv.DictReader(fh):
                out.append((float(e["angle_rad"]), float(e["expected_T"])))
    return out


def add_catalysis(entries, levels=1):
    """Gidney rotation doubling: each level adds half-angles at 2 + c/2."""
    merged = {}
    def add(a, c):
        a = round(a % TWO_PI, 12)
        if a not in merged or c < merged[a]:
            merged[a] = c
    for a, c in entries:
        add(a, c)
    cur = list(entries)
    for _ in range(levels):
        nxt = []
        for a, c in cur:
            h = 2.0 + c / 2.0
            nxt.append((a / 2.0, h))
            nxt.append((a / 2.0 + math.pi, h))
        for a, c in nxt:
            add(a, c)
        cur = nxt
    return list(merged.items())


def dress(entries, max_cost=float("inf")):
    """Clifford closure: +-angle + k*pi/4 (odd k adds 1 T) plus the bare
    k*pi/4 anchors at 0/1 T. Returns sorted (angles, costs) arrays."""
    lat = {}
    def add(a, c):
        a = round(a % TWO_PI, 12)
        if a not in lat or c < lat[a]:
            lat[a] = c
    for ang, c in entries:
        if c > max_cost:
            continue
        for s in (+1, -1):
            for k in range(8):
                add(s * ang + k * QUARTER, c + (0.0 if k % 2 == 0 else 1.0))
    for k in range(8):
        add(k * QUARTER, 0.0 if k % 2 == 0 else 1.0)
    S = np.array(sorted(lat))
    return S, np.array([lat[a] for a in S])


def build(gadget_path, catalyst_path=None, fallback_t=18.0, cat_levels=0,
          max_cost=float("inf")):
    """Gadget CSVs -> sorted (angles, costs). Reproduces the shipped
    rz_library_cat*_kn18.csv files."""
    e = load_entries(gadget_path, catalyst_path, fallback_t)
    if cat_levels:
        e = add_catalysis(e, cat_levels)
    return dress(e, max_cost)


def snap_nearest(S, C, theta):
    """Nearest library angle, unwrapped onto theta's branch."""
    d = np.abs((S - theta + math.pi) % TWO_PI - math.pi)
    i = int(np.argmin(d))
    return theta - ((theta - S[i] + math.pi) % TWO_PI - math.pi), float(C[i])


def snap_cheapest(S, C, theta, window):
    """Cheapest library angle within +-window; nearest if the window is empty."""
    x0 = theta % TWO_PI
    lo, hi = np.searchsorted(S, x0 - window), np.searchsorted(S, x0 + window)
    cands = list(range(lo, hi))
    if x0 - window < 0:
        cands += list(range(np.searchsorted(S, x0 - window + TWO_PI), len(S)))
    if x0 + window > TWO_PI:
        cands += list(range(0, np.searchsorted(S, x0 + window - TWO_PI)))
    if not cands:
        return snap_nearest(S, C, theta)
    j = min(cands, key=lambda k: C[k])
    return theta - ((theta - S[j] + math.pi) % TWO_PI - math.pi), float(C[j])


if __name__ == "__main__":
    # corrected package (2026-10-05): no catalyst angles, no catalysis levels (paper 1)
    S, C = build("rz_rus_gadgets.csv", None, cat_levels=0)
    print(f"levels=0: {len(S)} library angles, cost {C.min():.1f}-{C.max():.2f}")
