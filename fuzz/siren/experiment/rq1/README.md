# RQ1 — constructed goal-insertion attacks

Constructs static-obstacle **insertion attacks** against SPARK's safety filters
at the stock 0.05 m obstacle radius (the size the SPARK benchmark ships), for the
seven value-based filters, and renders each one as a video. An insertion attack
is a scene where the legitimate task `G0 -> G1` reaches safely, but inserting one
admissible goal `G1'` makes the filtered robot collide on the return leg.

Everything runs from the **repo root** (`spark/`). Paths and module names below
assume that.

## Requirements

- **Interpreter:** the conda `spark` env — `/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python`.
  The system `python3` is x86_64 and its numpy will not import on this arm64
  machine, so the conda interpreter is not optional. `run.sh` hardcodes it.
- **Environment** (only needed when invoking modules directly; `run.sh` sets these itself):
  ```bash
  export KMP_DUPLICATE_LIB_OK=TRUE        # MuJoCo + OSQP each load an OpenMP runtime
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export PYTHONPATH=.
  ```
- **Config:** the pipeline loads `fuzz/siren/pipeline/configs/config2.yml` (this
  study's tuning: `eta=0.02`, `d_min=0.02`). `config1.yml` holds SPARK's shipped
  values.

## Quick start

```bash
cd /Users/tiffanyb/Fun/robot/spark
fuzz/siren/experiment/rq1/run.sh          # spots -> hunt -> render (~3 h, one seed)
```

`run.sh` cd's to the repo root itself, pins the interpreter, sets the OpenMP
environment, and refuses to start if another `fuzz.siren` job is already running
(the stages are memory-hungry; one at a time by default).

```bash
./run.sh --list            # stages, runtimes, outputs
./run.sh spots hunt        # run named stages only
FORCE=1 ./run.sh spots     # re-run a stage whose output already exists
SEED=3 ./run.sh            # a different scene (default SEED=1)
./run.sh --sweep 1-10 spots hunt   # fan out over seeds, JOBS at a time (default 6)
```

A stage is skipped when its output exists **and** nothing it depends on is newer,
so an interrupted run can simply be restarted.

## Stages

| stage | ~min | module | output |
|---|---|---|---|
| `spots`  | 20  | `run_rq1 --phase spots` | `rq1_results/stock_spots_<seed>.json` — placement corpus (surface-separation metric) |
| `hunt`   | 140 | `run_rq1 --phase hunt`  | `rq1_results/stock_attacks/*.json` — per-filter confirmed attacks |
| `render` | 10  | `render_stock`          | `stock_visualizations/*.mp4` + `*.png` — attack and legitimate-task videos |

Each stage is also a plain module you can run directly (from the repo root, with
the environment above):

```bash
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
$PY -m fuzz.siren.experiment.rq1.run_rq1 --phase spots --seed 1 --grid 5 \
    --gap-lo -0.050 --gap-hi -0.005 --per-goal 8
$PY -m fuzz.siren.experiment.rq1.run_rq1 --phase hunt --seed 1 --want 1 \
    --algos ssa,rssa,pssa,cbf,rcbf,sss,rsss
$PY -m fuzz.siren.experiment.rq1.render_stock          # renders rq1_results/stock_attacks/*.json
```

## Full coverage across seeds

`run.sh`'s skip logic keys on one check-file per stage, which cannot express
"all seven filters covered". To drive at least one attack for every
`(seed, filter)` over seeds 1–10, use the coverage driver instead:

```bash
./drive_goal.sh              # build corpora, then hunt only the still-missing filters, bounded-parallel
./run_goal_wrapped.sh        # same, but records an exit line even on an unexpected crash
```

Check coverage at any time:

```bash
$PY -m fuzz.siren.experiment.rq1.coverage --seeds 1-10
```

## Layout

```
run.sh               main runner (spots / hunt / render)
drive_goal.sh        coverage-driven multi-seed hunt; run_goal_wrapped.sh wraps it
run_rq1.py           the search: --phase spots (placements) / --phase hunt (attacks)
separated_search.py  predecessor search; provides CASE, goal grid, swept-volume legs
big_obstacle.py      run_pinned / evaluate — the c(x) insertion primitive
swept.py             measured swept-volume geometry (surface_gap, volume_radii, ...)
render_stock.py      renders attack + baseline rollouts to mp4/png
coverage.py          "does every (seed, filter) have an attack?" — drives drive_goal.sh
reach_probe.py       diagnostic: how near the return leg comes to a placement
triage.py            diagnostic: classify each confirmed attack by contact mechanism
rq1_results/         corpora (stock_spots_*.json), attacks (stock_attacks/), and casestudy/
stock_visualizations/  rendered videos
tests/               unit tests (pytest; the slow ones build one MuJoCo world)
trash/               superseded predecessors — not part of the pipeline
```

## Case study

`rq1_results/casestudy/` sweeps the demand parameter `eta` on a single saved
attack and plots what the robot actually did. It has its own README; run it with:

```bash
$PY -m fuzz.siren.experiment.rq1.rq1_results.casestudy.sweep_eta
$PY -m fuzz.siren.experiment.rq1.rq1_results.casestudy.plot_eta_steps_task
```

## Outputs

- `rq1_results/stock_spots_<seed>.json` — placement corpus per scene
- `rq1_results/stock_attacks/*.json` — confirmed attacks (one file per filter/seed)
- `stock_visualizations/*.mp4`, `*.png` — rendered attack and baseline rollouts
- `meta.json` alongside each output directory records the UTC timestamp, argv,
  the config text, and SHA-256 prefixes of every input file and source module.
