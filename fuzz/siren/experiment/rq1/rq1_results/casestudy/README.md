# eta case study — one attack, swept demand

Sweeps the safety filter's demand parameter `eta` on a single saved insertion
attack, holding the scene and the goal sequence fixed, and records what the
robot actually did.

All inputs and outputs live in `data/`: the attack the study is run on, the
swept CSVs, the provenance JSON, and the figures. The scripts read and write
there by default.

## Files

| file | what it is |
|---|---|
| `sweep_eta.py` | runs the eta sweep, writes `data/eta_sweep.csv` + `data/eta_sweep_meta.json` |
| `plot_eta_steps_task.py` | reads `data/eta_sweep.csv`, writes `data/eta_steps_task.pdf` (and a `.png` preview) |
| `sweep_eta20_authority.py` | records per-step control authority at `eta=20`, writes `data/eta20_authority.csv` |
| `plot_eta20.py` | reads `data/eta20_authority.csv`, writes `data/casestudy_eta20.pdf` |
| `plot_eta20_authority_steps.py` | reads `data/eta20_authority.csv`, writes `data/eta20_authority_steps.pdf` (and a `.png`) |
| `data/pssa_0_1.json` | the attack the study is run on (scene, obstacles, goal triple) |
| `data/eta_sweep.csv` | the sweep data |
| `data/eta_sweep_meta.json` | provenance: attack file, fixed scene parameters, obstacles |
| `data/eta20_authority.csv` | per-step authority trace at `eta=20` |
| `diagnostics/` | step-level probes into why the QP goes infeasible (see its own README) |

## Reproducing

```bash
cd /Users/tiffanyb/Fun/robot/spark

# REQUIRED: two OpenMP runtimes load in this process (the conda env's libomp and
# torch's bundled copy). KMP_DUPLICATE_LIB_OK silences the abort but leaves the
# race live, and it currently segfaults in __kmp_suspend_initialize_thread.
# Forcing one runtime to load first fixes it.
export DYLD_INSERT_LIBRARIES=$CONDA_PREFIX/lib/libomp.dylib
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=.

# eta sweep -> data/eta_sweep.csv -> data/eta_steps_task.pdf
python -m fuzz.siren.constructed.rq1_results.casestudy.sweep_eta
python -m fuzz.siren.constructed.rq1_results.casestudy.plot_eta_steps_task

# eta=20 authority -> data/eta20_authority.csv -> the two authority figures
python -m fuzz.siren.constructed.rq1_results.casestudy.sweep_eta20_authority
python -m fuzz.siren.constructed.rq1_results.casestudy.plot_eta20
python -m fuzz.siren.constructed.rq1_results.casestudy.plot_eta20_authority_steps
```

Sweep a different attack:

```bash
python -m fuzz.siren.constructed.rq1_results.casestudy.sweep_eta \
    --attack fuzz/siren/constructed/rq1_results/stock_attacks/rssa_0_5.json
```

## What is held fixed

Everything except `eta`: the seed, the test case, the obstacle positions and
radius, `d_min`, `phi_k`, `lambda`, and the goal triple `G0 -> G1' -> G1` all
come from the attack record and are not varied. So every difference between rows
is caused by the demand alone.

## Columns

| column | meaning |
|---|---|
| `min_clearance` | SPARK's surface-to-surface distance, minimum over the run. **Negative = the robot penetrated the obstacle.** |
| `collision` / `attack_succeeds` | `min_clearance < 0`. This is the safety failure. |
| `reached_final` | the robot finished the attacker's `[G0, G1', G1]` schedule |
| `contact_leg` | which leg the contact happened on: `2` = the return leg (a true insertion), `1` = the outbound leg (not an insertion) |
| `engaged_steps` | steps where the filter was active (`phi >= 0` on some masked pair) |
| `warmup_giveups` | infeasible QPs during `Harness.reset()`'s 10 warm-up acts, before the episode |
| `episode_giveups` | infeasible QPs during the episode itself |
| `giveups_instrumented` | 0 means the give-up counter was NOT installed for that filter |

Three of these need care when reading:

**`reached_final` is not the complement of `collision`.** A run can end with no
collision *and* no goal — the filter deflected the arm so hard the task never
completed. That is a task failure, not a successful defence, and the figure
encodes it as a hollow marker rather than folding it into "attack failed".

**`warmup_giveups` is separate on purpose.** `Harness.reset()` runs 10 warm-up
`algo.act` calls at the seeded initial pose. Counting those as episode
infeasibility inflates the give-up column and was an earlier error in this
analysis.

**`episode_giveups` is blank for the slack filters.** `probe.install_probe`
installs the counter only for `BasicSafeSetAlgorithm`, so a `0` there would mean
"nothing was watching", not "never infeasible". p-SSA cannot go infeasible by
construction anyway — its phase I solves for the nearest feasible relaxation —
so the quantity to report for it is the slack magnitude `s*`, which this sweep
does not record.
