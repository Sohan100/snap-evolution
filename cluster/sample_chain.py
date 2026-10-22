"""Draw the KN fallbacks of open-chain SNAP points: the main metric with the fallbacks drawn.

    python cluster/sample_chain.py FILE [FILE ...] [--samples S] [--fresh]

FILE is an open-chain evaluation, runs/<tag>/t<T>/chain/<out>/eval_b*_D*.json, next to its real_ record. This is
cli.cmd_sample's SNAP branch for the open-chain records, whose edge rotations are always applied (chain_snap's
evaluate_chain) and carry their own KN budgets (eps_per_angle), so fallbacks.sample_snap cannot be used as is:
it would leave the edges out. Every KN rotation, bulk stragglers and edges alike, falls back on its own with
probability 1 - p (fallbacks.sample_circuit, same seeds), on the evaluation's own inputs, step, cutoff and chi,
against the cached references in runs/<tag>/t<T>/ref. On top of the drawn average comes the bound for the
library gadgets, bulk (fallbacks.snap_total's ledger) and edges, at the F the evaluation chose. The record gets
"sampled", "infid_sampled_kn" and "infid_sampled", as the sweep's points do. Draws per input: 4 at t <= 4,
2 beyond (cluster/make_jobs.py). A file is claimed while it is drawn (io.claim), so parallel runs skip it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sample_file(p: Path, samples: int | None = None, fresh: bool = False, log=print) -> dict | None:
    from snapevo import io
    from snapevo.cells import get_cell
    from snapevo.circuits import Layered
    from snapevo.evaluate import fidelity_bound
    from snapevo.fallbacks import sample_circuit
    from snapevo.hamiltonian import XXZ
    from snapevo.synthesis import synthesize

    with io.claim(p) as mine:
        if not mine:
            log(f"{p.name}: another process is drawing it")
            return None
        e = io.read_json(p)
        if "sampled" in e and not fresh:
            return e
        real = io.read_json(p.with_name(p.name.replace("eval_", "real_", 1)))
        model = XXZ(**real["model"]) if isinstance(real["model"], dict) else XXZ(0.25, float(e["model"]["hx"]))
        cell, t, N = get_cell(real["cell"]), float(real["t"]), int(e["N"])
        F = str(e["F"])
        m = e["mps"]
        S = samples or (4 if t <= 4 else 2)
        x = np.asarray(real["angles_compiled"], dtype=float)
        kn = {int(i): (r if "U0" in r else synthesize(float(x[int(i)]), r["eps"], unitaries=True))
              for i, r in real["kn"].items()}
        E = real["edges"]
        xe = np.asarray(E["angles"], dtype=float)
        eps_e = E.get("eps_per_angle") or {}
        ekn = {str(i): (r if "U0" in r else synthesize(float(xe[int(i)]), float(eps_e.get(str(i), E["eps"])),
                                                        unitaries=True))
               for i, r in E["kn"].items()}
        edges = {**E, "use": True, "kn": ekn}
        na = cell.n_angles
        n_kn = len(kn) + len(ekn)
        if e["infid"] > 0.999 or n_kn == 0:
            sm = {"infid": e["infid"], "drawn": False, "n_kn": n_kn}
        else:
            log(f"sample: {p.parents[3].name}/{p.parents[2].name} {p.stem}, {len(kn)} bulk and {len(ekn)} "
                f"edge KN rotations, {S} draws per input")
            sm = {**sample_circuit(Layered.brickwork(cell, real["angles_real"]), lambda l, s: kn.get(l * na + s),
                                   edges, model, t, N=N, states=tuple(m["states"]), samples=S, dt=m["dt"],
                                   chi_max=m["chi_max"], cutoff=m["cutoff"],
                                   cache_dir=p.parents[2] / "ref", log=log),
                  "drawn": True, "n_kn": n_kn}
        # library gadgets stay a bound: bulk (per qubit, x N) and edges (per circuit), at the evaluation's F
        delta = real["per_F"][F]["delta_fb"]
        mult = np.asarray(real["multiplicity"], dtype=float)
        L = sum(0.5 * mult[int(i)] * row["q"][F] * delta for i, row in real["lattice"].items()) * N
        L2 = sum(0.5 * mult[int(i)] * row["q"][F] * delta ** 2 for i, row in real["lattice"].items()) * N
        L += sum(row["q"][F] * delta for row in (E.get("lattice") or {}).values())
        L2 += sum(row["q"][F] * delta ** 2 for row in (E.get("lattice") or {}).values())
        e["sampled"] = sm
        e["infid_sampled_kn"] = sm["infid"]
        e["infid_sampled"] = fidelity_bound(sm["infid"], L, L2 / 2)
        io.write_json(p, e)
        log(f"{p.name}: 1 - F {e['infid']:.4e} (success), {e['infid_total']:.4e} (bound), "
            f"{e['infid_sampled']:.4e} (drawn)")
        return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    for f in a.files:
        sample_file(Path(f).resolve(), a.samples, a.fresh)


if __name__ == "__main__":
    main()
