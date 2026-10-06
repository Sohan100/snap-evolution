"""Write the job lists of the t sweep, one shell command per line.

    python cluster/make_jobs.py

writes cluster/jobs/{train,reference,points,baselines}.txt. Edit the grid below
and rerun to change the sweep. submit.sh sends the lists to SLURM as job arrays
with dependencies, and run_local.sh runs one list in order on any machine.

Every command uses the default MPS settings (N = 100, 8 inputs, seed 2026,
dt 0.2, cutoff 1e-12, chi_max 1024), so the points and the baselines find the
references the reference jobs cached. The extra reference of input 0 at cutoff
1e-14 is the truncation check of every evaluation. Change the settings
everywhere or nowhere.
"""

from pathlib import Path

MODELS = ("A", "B")
TIMES = (1, 2, 4, 6, 8, 10)                  # spin units
BUDGETS = ("1e-4", "1e-5", "1e-6")           # training C of the SNAP points
DEPTHS = {                                   # training ladders, each stops early when its floor stalls
    1: "3,5,7,9,11",
    2: "7,9,11,13,15,17",
    4: "9,11,13,15,17,21,25,29",
    6: "13,17,21,25,29,33,37",
    8: "17,21,25,29,33,37,41,45",
    10: "21,25,29,33,37,41,45,49,53",
}
STATES = 8
SPLIT_FROM = 6                               # from this t on, one reference job per input


def main() -> None:
    py = "python -m snapevo"
    jobs = {
        "train": [f"{py} train --model {m} --t {t} --depths {DEPTHS[t]}" for m in MODELS for t in TIMES],
        "reference": [],
        "points": [f"{py} point --model {m} --t {t} --budget {b}" for m in MODELS for t in TIMES for b in BUDGETS],
        "baselines": [f"{py} baseline-scan --model {m} --t {t} --evaluate" for m in MODELS for t in TIMES],
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
