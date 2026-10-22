# Handoff: t = 6 of the open-chain SNAP arm, for Isaac's workstation

From Sohan, October 2026. This package runs one self-contained slice of paper 1's time-evolution simulations on a
48-core, 256 GB workstation: **the open-chain arm at evolution time t = 6, both fields**, from training through the
paper's main metric. It is written for you and for any agent you point at it. Read it to the end before starting.

## What this computes

Paper 1 compiles e^{-iHt} for the XXZ chain onto your corrected exact-angle RUS library with SNAP:

    H = sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1} + 0.25 S^z_i S^z_{i+1}) + h_x sum_i S^x_i      (spin units)

for two fields, h_x = 1.6 and h_x = 0. (Always name the field; the code's "model A/B" labels are h_x = 1.6 / 0.)

The open-chain arm does everything on the 100-site open chain with MPS, never on a ring:

1. **Training.** A depth ladder of translation-invariant brickwork cells (one set of angles per sublayer, plus
   the edge gates at the two ends), trained on 12 random product inputs against MPS references of e^{-iHt}.
   Each depth starts from Trotter, from the previous depth padded with identity layers, and from Trotter plus
   noise; each start is optimized by trust-region Newton with Hessian reuse until converged (no iteration cap).
   The lowest floor is kept.
2. **SNAP compile.** At every budget from 1e-1 to 1e-9 and every depth whose floor is at most budget/1.5:
   Hessian-ranked snapping onto the library, Newton re-optimization after each snap, edge rotations snapped too.
3. **Realize and evaluate.** Lattice angles priced, leftover rotations synthesized with Kliuchnikov non-mixed,
   and the **per-site infidelity c = 1 - mean_k F_k^(1/N)** of the realized circuit measured on 8 held-out
   product inputs at **N = 100** by MPS. This is the paper's main error metric.
4. **Fallback draws.** The same metric with the KN fallback branches drawn (the paper's headline curves).

NERSC runs t = 1, 2, 4, 8 and 10 the same way. t = 6 comes to you because it needs more memory than NERSC's
login nodes allow (30 GB per user): a 12-worker t = 6 ladder peaks at about 32 GB, and the compiles and
tight-budget evaluations need 10-40 GB each. NERSC's compute-node hours go to t = 1, 2 and 4.

## What is in the package

    HANDOFF.md            this file
    PROVENANCE.txt        when and from what it was built, and the t = 6 depths already trained on NERSC
    MANIFEST.sha256       checksums (setup.sh checks them)
    requirements.txt      pinned Python packages
    snap-evolution/       the code (src/snapevo, cluster/, tests/, third_party/ with your two packages)
      runs/kn_cache.json  KN syntheses so far (new ones are added as it runs)
      runs/xxz_d0.25_hx{0,1.6}/t6/ref/    the N = 100 evaluation references (8 inputs, all cutoffs)
      runs/xxz_d0.25_hx{0,1.6}/t6/chain/  the t = 6 ladders so far (ladder.json, checkpoints)
    data/snapevo-scratch/open_train/refs/xxz_d0.25_hx{0,1.6}/t6/   the 12 training references per field
    scripts/setup.sh  run_t6.sh  status.sh  pack_results.sh  env.sh

About 5 GB unpacked. Results grow by a few GB at most.

## Steps

    unzip -q handoff_isaac_t6.zip && cd handoff_isaac_t6
    bash scripts/setup.sh              # venv, packages, library check, tests (~10 min)
    bash scripts/run_t6.sh start       # everything starts in the background; safe to rerun at any time
    bash scripts/status.sh             # depths and floors, points evaluated, draws done, what is running

setup.sh needs Python >= 3.10 and internet access for `pip install` (the packages in requirements.txt). It
checks every file against MANIFEST.sha256 first; a mismatch means a damaged copy.

Running a command by hand (an agent checking something): `source scripts/env.sh` first, so the references and
thread settings are right.

Stopping (for a reboot, or to free the machine): `bash scripts/run_t6.sh stop`. Everything checkpoints;
`run_t6.sh start` continues where it stopped. After a crash or power loss, just run `start` again.

Sending results: `bash scripts/pack_results.sh` makes `results_t6_<date>.tar.gz` (tens of MB). Please send one
every few days, and when everything has finished.

## What runs, and the resources

| piece | default | cores | memory |
|---|---|---|---|
| ladder h_x = 0, ladder h_x = 1.6 | 12 workers each | 24 | up to ~2.6 GB per worker (~32 GB per ladder) at shallow depths, under 1 GB per worker from D ~ 33 |
| compile + N = 100 evaluation | 1 at a time (`COMPILE_SLOTS`) | 12 | up to ~40 GB |
| fallback draws | 2 at a time (`SAMPLE_SLOTS`) | 8 | up to ~40 GB each |

The compile loop starts compiling as soon as finished depths allow a budget, cheapest first, so results appear
early and keep coming; deeper depths add the tighter budgets later. When a ladder log says `stopping the ladder`,
its cores are free: restart with more compile slots, e.g. `COMPILE_SLOTS=3 bash scripts/run_t6.sh start`.

Timing, from NERSC measurements (estimates): one input's gradient takes 1.5-3 minutes at t = 6, and a Newton
step needs two gradients per angle of the circuit, spread over the 12 workers. A depth takes hours below D ~ 10
and one to two weeks from D ~ 20 on; a compile takes hours at loose budgets and a day or more at tight ones.
Expect the first evaluated points in one to two weeks, most of the curve (budgets down to ~1e-6) in two to
three months, and the tightest budgets (1e-8, 1e-9) in four to six months or more.

For the first week or two nothing is ready to compile yet (no depth has a floor below 0.067), so the ladders
can have all 48 cores: `bash scripts/run_t6.sh stop`, then `LADDER_WORKERS=24 bash scripts/run_t6.sh start`
(about 125 GB). That roughly halves this stretch. Go back to the default 12 the same way once `status.sh`
shows points being evaluated.

## Rules (for agents especially)

- **Do not edit anything under `snap-evolution/src/`, `cluster/` or `third_party/`, and do not change the
  protocol settings in the scripts** (Newton with Hessian reuse, no iteration caps, `--edge-zero-t`, every depth,
  every budget, the inputs, cutoffs and references). The results must be comparable with NERSC's. If something
  fails, collect the log lines and report them to Sohan instead of patching code.
- **Never delete or overwrite files under `snap-evolution/runs/`**, and never pass `--fresh`. Every command skips
  finished work and resumes from checkpoints; deleting a checkpoint throws away days of work.
- Keep within the machine: the defaults use ~44 cores. Only raise `COMPILE_SLOTS` / `SAMPLE_SLOTS` when a ladder
  has finished or the machine is otherwise idle, and watch memory (`free -g`): stay under ~220 GB.
- No ring, no L-BFGS, no caps: these are deliberate decisions for this paper, not oversights.
- In anything you write, name the models by their field (h_x = 1.6, h_x = 0), not "model A/B".

## Troubleshooting

- `FileNotFoundError: ... references missing`: the package is incomplete; check `sha256sum -c MANIFEST.sha256`
  and tell Sohan which file.
- A process killed by the OOM killer (a ladder log says `BrokenProcessPool`): stop, restart with `SAMPLE_SLOTS=1`
  and/or `COMPILE_SLOTS=1`. The ladder resumes from its last Newton step.
- Worker processes left behind: pool workers ignore SIGTERM on purpose (their parent saves its checkpoint
  first), so a parent killed by the OOM killer or `kill -9` leaves its workers running and holding memory.
  `run_t6.sh start` kills such orphans before starting anything; it never touches workers of a live run.
- `another run holds .../chain/lock`: a ladder for that field is already running (look with `status.sh`).
- A compile that fails repeatedly: its lines in `logs/compile_*.log`; the loop moves on to other points and
  retries it on the next pass. Report persistent failures.

## Who to ask

Sohan. When you send results, include `bash scripts/status.sh > status.txt`.
