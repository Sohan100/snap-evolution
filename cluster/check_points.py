"""Apply the checks on every point (docs/error_metric.md) to every evaluated file.

    python cluster/check_points.py [--all]

Reads every eval_*.json and baseline_o*.json under $SNAPEVO_RUNS (default runs/)
and prints one markdown row per failed check, or every point with --all. The checks are:

1. stat   the statistical error on 1 - F is below 5%
2. conv   the change of input 0 at a 100x smaller cutoff (mps.trunc_check.rel_change) is below 5%
3. bar    the pessimistic truncation bar (mps.trunc_rel) is below 5%. Reported only, conv decides.
4. c      the per-site c at N = 100 is within 10% of the ring's held-out C
5. drawn  the point has its fallback draws (infid_sampled)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("SNAPEVO_RUNS", REPO / "runs"))


def checks(e: dict) -> dict:
    m = e["mps"]
    infid = e["infid"]
    stat = m["stderr"] / infid if infid > 0 else 0.0
    conv = m["trunc_check"]["rel_change"] if "trunc_check" in m else None
    c_gap = e["c_site"] / e["C_ring"] - 1 if e.get("C_ring") else None
    return {
        "stat": (stat, stat < 0.05),
        "conv": (conv, conv is not None and conv < 0.05),
        "bar": (m["trunc_rel"], m["trunc_rel"] < 0.05),
        "c": (c_gap, c_gap is not None and abs(c_gap) <= 0.10),
        "drawn": ("infid_sampled" in e, "infid_sampled" in e),
    }


def label(path: Path, e: dict) -> tuple[str, str, str]:
    model = "A" if e["model"]["hx"] else "B"
    if path.name.startswith("eval_"):
        what = f"SNAP {e['budget']:.0e}, D = {e['D']}"
    else:
        what = f"baseline o{e['order']} r{e['steps']} eps {e['eps']:.0e}"
    return model, f"{e['t']:g}", what


def main() -> None:
    every = "--all" in sys.argv
    files = sorted(RUNS.glob("*/t*/eval_*.json")) + sorted(RUNS.glob("*/t*/baseline_o*.json"))
    print("| model | $t$ | point | $1 - \\bar F$ | stat | change at cutoff/100 | bar | $c / C - 1$ | drawn | fails |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    n_fail = 0
    for p in files:
        e = json.loads(p.read_text())
        if "mps" not in e:
            continue
        ch = checks(e)
        bad = [k for k, (_, ok) in ch.items() if not ok]
        if bad:
            n_fail += 1
        if not bad and not every:
            continue
        model, t, what = label(p, e)
        conv, cg = ch["conv"][0], ch["c"][0]
        print(f"| {model} | {t} | {what} | {e['infid']:.2e} | {ch['stat'][0]:.1%} | "
              f"{'-' if conv is None else format(conv, '.0e')} | {ch['bar'][0]:.1%} | "
              f"{'-' if cg is None else format(cg, '+.1%')} | {'yes' if ch['drawn'][0] else 'no'} | {', '.join(bad)} |")
    print(f"\n{len(files)} points, {n_fail} with at least one failed check", file=sys.stderr)


if __name__ == "__main__":
    main()
