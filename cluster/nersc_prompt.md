# Prompt for an agent on NERSC Perlmutter: finish the snap-evolution sweep

You are running the rest of the snap-evolution simulation sweep on NERSC Perlmutter, then analysing it and writing it up. The repo is a Python package (`snapevo`) that compiles XXZ time evolution into Clifford+T circuits with SNAP and compares it against a product-formula baseline. Every error is a state infidelity computed by MPS on a 100-site open chain. Work from the repo's own docs, which are the single source of truth.

- `docs/plan.md` covers the models, the sweep grid and the stages.
- `docs/error_metric.md` covers the error metric, the fallback bound and draws, and the checks every point must pass.
- `cluster/README.md` covers the job arrays, dependencies and measured costs.
- `docs/validation.md` holds the results so far. The top section, "Budget sweep at t = 1 and 2", is the model for your write-up.

`python -m snapevo -h` lists the commands. Read those four docs before step 1.

## What is already done

The repo already holds SNAP at all 15 budgets at $t = 1$ and at $10^{-1}$ to $3\times10^{-7}$ at $t = 2$ for both models. It also holds the baseline at all budgets for $t = 1$ and 2, a few model B points at $t = 4$, and fallback draws for all of it. Every job skips work whose result file already exists, so you submit the full grid and only the missing work runs. What remains is as follows:

1. Training ladders at $t = 4, 6, 8, 10$.
2. References at every $t$. The `ref/` caches are not in git.
3. SNAP points at $t \ge 4$, the tight end of $t = 2$, and the second, deeper compile at $t = 1$ and 2.
4. Baseline scans and evaluations at $t = 4$ to 10.
5. Fallback draws for everything new.

## Hard rules

- Leave `third_party/` and the grid in `cluster/make_jobs.py` (models, times, budgets, depths, margins, draws) as they are. If the grid looks wrong, stop and ask Sohan.
- Keep the MPS settings at their defaults everywhere ($N = 100$, 8 inputs, seed 2026, $dt = 0.2$, cutoff $10^{-12}$, $\chi_{\max} = 1024$), so that every job finds the cached references.
- Keep compile checkpoints (`*.ckpt.pkl`). A resubmitted compile resumes from them.
- Write prose in Sohan's voice: "we" sentences, plain declaratives, numbered steps after "as follows:", math in LaTeX. Use commas and periods where you would reach for an em dash or a semicolon.

## Step 1. Set up

As follows:

1. Clone the repo to `$SCRATCH/snap-evolution` from the bundle Sohan copied over (`git clone snap-evolution.bundle snap-evolution`). Check that `git log -1` shows "Budget sweep at t = 1 and 2 with drawn fallbacks and two-depth compiles".
2. Run `module load python`, then create a venv in the repo (`python -m venv .venv && source .venv/bin/activate`) and run `pip install -e .[test,plot]`.
3. Run `python -m snapevo verify-library` and `pytest`.
4. Run `export SNAPEVO_RUNS=$SCRATCH/snap-evolution/runs` so the runs stay inside the repo on scratch. Lustre on Perlmutter supports `flock`, which the shared KN cache uses.

Step 1 is done when the library check passes and all 35 tests pass.

## Step 2. Adapt the SLURM scripts to Perlmutter

`cluster/array.slurm` and `cluster/submit.sh` are written for a generic cluster. Make these changes:

1. Add `#SBATCH -A <account>` (ask Sohan for the NERSC project, `m....`), `#SBATCH -C cpu` and `#SBATCH -q shared` to `array.slurm`. The tasks are single-process Python jobs, so the shared QOS fits them and charges only for the cores they use.
2. In the site-setup lines, load the module, activate `.venv` and export `SNAPEVO_RUNS`.
3. Perlmutter caps shared and regular jobs at 48 h. Lower every `--time` in `submit.sh` to `48:00:00` or less. Training, compiles and reference builds resume after a timeout, so long tasks finish by resubmission.
4. Ask for 4 CPUs and 16G for the reference, baseline and sample arrays (already set), and 2 CPUs for the rest.
5. Check `sacctmgr show qos shared format=name,maxwall,maxsubmitpu` and confirm the arrays (about 630 tasks in all) fit the per-user submit limit. If they do not, submit the arrays in waves.

Step 2 is done when a smoke test passes. Run one line of `cluster/jobs/train.txt` for $t = 4$ through `array.slurm` on the debug QOS (`-q debug -t 00:30:00`). It passes when its log shows the ladder training its first depth with no import or path errors. Then cancel it.

## Step 3. Submit and keep the sweep moving

As follows:

1. Run `python cluster/make_jobs.py`, then `cluster/submit.sh`. Record the six array job IDs it prints in `cluster/logs/submitted.txt`.
2. Check progress about once a day with `squeue -u $USER` and `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`.
3. Resubmit every task that ended TIMEOUT. Use the same list and line number, for example `sbatch --array=17 --export=ALL,JOBS=$PWD/jobs/points.txt array.slurm`, from `cluster/`. Nothing is lost, because compiles resume from their checkpoint.
4. For a task that FAILED, read its log in `cluster/logs/`, fix the cause if it is environmental (memory, module, path), and resubmit. If the cause is a bug in `snapevo`, stop and report it to Sohan with the traceback. Do not patch the physics code on your own.
5. When the points and baseline arrays are both done, check that the sample array ran after them. Resubmit any sample tasks that started early.

Step 3 is done when every task of all six arrays has ended COMPLETED, either first time or after resubmission, or has a reason recorded in `cluster/logs/submitted.txt`. A point that compile skipped because its ladder floor is too high is a valid outcome, and so is a ladder that stopped early because its floor stalled.

## Step 4. Check every point

Apply the checks in `docs/error_metric.md` ("Checks on every point") to every `eval_*.json` and `baseline_o*.json` at every $t$. The fields are listed in `docs/conventions.md`.

1. The statistical error on $1 - \bar F$ is below 5%.
2. The truncation check (`mps.trunc_check.rel_change`) is below 5%. Expect the pessimistic bar `trunc_rel` to exceed 5% for some tight points at large $t$. Report it, but the convergence check decides.
3. The per-site $c$ at $N = 100$ is within 10% of the ring's $C$.
4. Every evaluated point has its fallback draws (`infid_sampled`).

Step 4 is done when every failing point is listed with its values in a table. For each failure, either rerun the point (for example, at a smaller cutoff for truncation) or state why it stands.

## Step 5. Analyse

As follows:

1. Run `python -m snapevo summary`, `python -m snapevo compare` and `python -m snapevo plot`. The figures go to `runs/figures/`.
2. For each model and $t$, find the range of the baseline-to-SNAP T ratio along SNAP's lower envelope, using the drawn fallbacks. Take it from the "drawn" column of `compare`, keeping only rows on SNAP's envelope.
3. At $10^{-6}$ and $10^{-4}$, tabulate T per qubit per unit time against $t$ for SNAP and the baseline. The plan expects SNAP's value to fall with $t$.
4. Record which of the two compile depths won at each budget, and where each model's wall sits, meaning the budget where T per qubit jumps.
5. For the baseline, compare the drawn infidelity with the success branch and with the bound, as the $t = 1, 2$ section does.

Step 5 is done when every model and time has its ratio range, and the T per qubit per unit time table covers all six times.

## Step 6. Write up and commit

As follows:

1. Add a section "Cluster sweep, t = 1 to 10" at the top of `docs/validation.md`, built like the $t = 1, 2$ section. It needs the SNAP tables per $t$ (`snapevo report`), the comparison tables, the findings of step 5 and the failures of step 4. Point to the figures by path.
2. Update "Where it runs" in `docs/plan.md` and the measured-costs table in `cluster/README.md` with the real times from `sacct`.
3. Commit the results and docs with a message that says what ran. `runs/` is tracked, and `ref/`, logs, checkpoints and lock files are ignored by `.gitignore`. Then make a fresh bundle, `git bundle create $SCRATCH/snap-evolution-results.bundle --all`, for Sohan to pull back.

Step 6 is done when `git status` is clean, the bundle exists, and the new section reads with no gaps for any model and time.
