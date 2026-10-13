"""Write the job lists of the t sweep, one shell command per line.

    python cluster/make_jobs.py

writes cluster/jobs/{train,reference,scan,points,baselines,samples}.txt. Edit the grid
below and rerun to change the sweep. submit.sh sends the lists to SLURM as job
arrays with dependencies, and run_local.sh runs one list in order on any machine.

The SNAP points cover 15 training budgets, half decades from 1e-1 to 1e-8, each
compiled at two depths: the shallowest whose floor is at most a third of the
budget, and the shallowest whose floor is at most a thirtieth (or the deepest).
The deeper cell has more rotations, but with more room under the budget they
come out much cheaper, and at tight budgets it usually wins (model B at t = 1,
1e-7: 142 T per qubit at D = 15 against 216 at D = 11). A depth whose floor
leaves less than a third of the budget for snapping is skipped at compile time,
so the tight end at large t needs the ladder to reach deep enough first. The baseline scan of each model and time runs once, and its
Pareto fronts are then evaluated by MPS at the points nearest each budget, one
job per budget, and the sample jobs draw the fallback branches of those points
and of the SNAP points of the same budget (fallbacks.py).

Every command uses the default MPS settings (N = 100, 8 inputs, seed 2026,
dt 0.2, cutoff 1e-12, chi_max 1024), so the points and the baselines find the
references the reference jobs cached. The extra reference of input 0 at cutoff
1e-14 is the truncation check of every evaluation. Change the settings
everywhere or nowhere.
"""

from pathlib import Path

MODELS = ("A", "B")
TIMES = (1, 2, 4, 6, 8, 10)                  # spin units
BUDGETS = ("1e-1", "3e-2", "1e-2", "3e-3", "1e-3", "3e-4", "1e-4", "3e-5",    # training C of the SNAP points
           "1e-5", "3e-6", "1e-6", "3e-7", "1e-7", "3e-8", "1e-8")
DEPTHS = {                                   # training ladders, each stops early when its floor stalls
    1: "2,3,5,7,9,11,13,15",
    2: "2,3,5,7,9,11,13,15,17,19,21,25",
    4: "3,5,7,9,11,13,15,17,21,25,29,33,37,41",
    6: "3,5,7,9,13,17,21,25,29,33,37,41,45,49,53",
    8: "5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73",
    10: "5,7,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,73,81",
}
STATES = 8
SPLIT_FROM = 6                               # from this t on, one reference job per input
SAMPLES = {1: 4, 2: 4, 4: 4, 6: 2, 8: 2, 10: 2}   # fallback draws per input, each one more circuit MPS
MARGINS = "3,30"   # one compile at the shallowest depth with floor <= B/3 and one with floor <= B/30
# L-BFGS iterations per start. At 6000 the deepest depths of every ladder at t >= 4 hit the cap and
# stopped as stalled, with the three starts up to 2.7x apart (Sohan, 12 Oct: extend them at 30000).
# Training resumes, so the depths already trained stay and the ladders continue past the stop.
MAXITER = {1: 6000, 2: 6000, 4: 30000, 6: 30000, 8: 30000, 10: 30000}


def main() -> None:
    py = "python -m snapevo"
    jobs = {
        "train": [f"{py} train --model {m} --t {t} --depths {DEPTHS[t]}"
                  + (f" --maxiter {MAXITER[t]}" if MAXITER[t] != 6000 else "") for m in MODELS for t in TIMES],
        "reference": [],
        "scan": [f"{py} baseline-scan --model {m} --t {t}" for m in MODELS for t in TIMES],
        "points": [f"{py} point --model {m} --t {t} --budget {b} --margins {MARGINS}"
                   for m in MODELS for t in TIMES for b in BUDGETS],
        "baselines": [f"{py} baseline-scan --model {m} --t {t} --evaluate --targets {b}"
                      for m in MODELS for t in TIMES for b in BUDGETS],
        "samples": [f"{py} sample --model {m} --t {t} --targets {b} --budget {b} --samples {SAMPLES[t]}"
                    for m in MODELS for t in TIMES for b in BUDGETS],
    }
    for m in MODELS:
        for t in TIMES:
            if t < SPLIT_FROM:
                jobs["reference"].append(f"{py} reference --model {m} --t {t}")
            else:
                jobs["reference"] += [f"{py} reference --model {m} --t {t} --only {k}" for k in range(STATES)]
            # input 0 at a 100x smaller cutoff, for the truncation check of every evaluation
            jobs["reference"].append(f"{py} reference --model {m} --t {t} --only 0 --cutoff 1e-14")
    out = Path(__file__).resolve().parent / "jobs"
    out.mkdir(exist_ok=True)
    for name, lines in jobs.items():
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{name:10s} {len(lines):3d} jobs -> {out / f'{name}.txt'}")


if __name__ == "__main__":
    main()
