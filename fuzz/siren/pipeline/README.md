# SIREN evaluation pipeline

Four stages: **find** ground-truth goal-insertion attacks in the SPARK
benchmark, **verify / record / render** them, **score SIREN** on whether its
fuzzers rediscover them, then **measure and visualise** the search.

The pipeline is the `stage*.py` modules in this directory. Run everything from
the repo root.

## Requirements

```bash
cd /Users/tiffanyb/Fun/robot/spark
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python   # conda `spark` env — the system python3 will not import numpy on arm64
export PYTHONPATH=/Users/tiffanyb/Fun/robot/spark
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Use **absolute paths** for `--src` / `--dir` / `--out-dir`. The shell keeps its
working directory between commands, and a relative path resolved from the wrong
cwd silently matches zero files.

Configs live in `configs/`: `config2.yml` is this project's tuning
(`eta=0.02`, `d_min=0.02`); `config1.yml` is SPARK's shipped values. Stages that
take `--config` accept one of these; stage 1 with no `--config` uses built-in
defaults.

## Quick start

**End-to-end smoke test** (one scene, one filter, small budget — everything
lands in `pipeline/example/`):

```bash
bash fuzz/siren/pipeline/run_example.sh
```

It runs all six stage invocations in order and prints the artifact tree at the
end. Good for verifying the whole chain works.

**The trials** (config1 vs config2 over the usable scenes, resumable, parallel):

```bash
bash fuzz/siren/pipeline/run_trials.sh [PARALLEL]     # default 10 workers
```

Writes to `pipeline/trials_0803_1/<config>/`. Jobs already finished are skipped,
so it can be killed and restarted.

## Notation

```
original scenario   home -> scenario_goal
attack schedule     [G0, G1', G1]   run from the home pose:
                      wp_idx 0 : home -> G0     setup (handover state captured here)
                      wp_idx 1 : G0   -> G1'    reaching the inserted goal
                      wp_idx 2 : G1'  -> G1     back to the legitimate goal
with                G1' := home (EE position),  G1 := scenario_goal

contact on leg 2  ->  INSERTION     the inserted goal breaks the return trip
contact on leg 1  ->  MODIFICATION  the robot is hit reaching the inserted goal
contact on leg 0  ->  rejected      crashed before the inserted goal was active
```

Leg 2 and the original scenario share endpoints and differ only in departure
state — the original starts at rest, leg 2 departs in motion. That is the
mechanism the attack turns on.

## Stages

Each stage is `python -m fuzz.siren.pipeline.<module>`. Inputs/outputs are
whatever you pass to `--src` / `--out-dir` etc.; the commands below mirror
`run_example.sh`.

### Stage 1 — find attacks · `stage1_search`

Searches G0 handover points that turn a colliding scenario into a confirmed
insertion/modification. Resumes from the captured handover state rather than
replaying leg 0.

```bash
$PY -m fuzz.siren.pipeline.stage1_search \
    --algo rcbf --scenes G1FixedBase_D2_AG_SO_v0 --seeds 1 --n-worlds 1 \
    --grid 4 --lam 0.3 --out-dir /abs/out/stage1
```

Key flags: `--algo` (`ssa rssa pssa cbf rcbf sss rsss`), `--seeds` (`1-20` or
`0,1,2`), `--n-worlds` (gate-passing worlds to search per scene), `--scenes`
(comma-separated case names; default the built-in set), `--grid` (N³ G0 grid),
`--lam` (proportional-demand gain), `--goal-channel` (`arm`/`base`), `--config`,
`--relax` (count DEADLOCK as an attack, on by default). Writes `control_*.json`.

### Stage 2 — verify, trace, render · `stage2_verify`

Re-runs each candidate from scratch (baseline must reach with clearance > 0; the
attack must collide/deadlock on all repeats; contact leg must be consistent),
records a trace, and renders it.

```bash
$PY -m fuzz.siren.pipeline.stage2_verify \
    --src '/abs/out/stage1/control_*.json' --out-dir /abs/out/stage2 \
    --repeats 3 --trace full --render
```

Companions: `stage2_baseline` (record the baseline rollout for a verified
attack), `stage2_render` (render a verified attack by replaying its trace).

### Stage 3 — score SIREN · `stage3_targets` then `stage3_fuzz`

`stage3_targets` turns verified controls into fuzz targets and re-proves each
one; `stage3_fuzz` runs SIREN's search strategies against a target and logs
every location searched. `G1_prime_truth` is loaded but never given to the
search.

```bash
$PY -m fuzz.siren.pipeline.stage3_targets \
    --verified '/abs/out/stage2/*.json' --out-dir /abs/out/stage3

$PY -m fuzz.siren.pipeline.stage3_fuzz \
    --target /abs/out/stage3/<name>.json --pickers random,cem,bo \
    --budget 24 --batch 6 --search-seeds 0,1 --out /abs/out/stage3/fuzz_result.json
```

Pickers are `random cem bo`, each optionally `_seeded` (obstacle-anchored
initial design). The first `--batch` evaluations are a shared design across
strategies, so comparisons below that are artifacts. `stage3_render_attack`
renders one discovered attack (`home -> G0 -> G1' -> G1`).

### Stage 4 — measure and visualise · `stage4_curves` and `stage4_render`

```bash
$PY -m fuzz.siren.pipeline.stage4_curves \
    --src /abs/out/stage3/fuzz_result.json --out-dir /abs/out/stage4

$PY -m fuzz.siren.pipeline.stage4_render \
    --fuzz /abs/out/stage3/fuzz_result.json --targets-dir /abs/out/stage3 \
    --out-dir /abs/out/stage4
```

`stage4_curves` produces discovery curves (attacks vs trials, and vs time) from
the per-evaluation log. `stage4_render` draws a MuJoCo scene per target — the
robot at the G0 handover, the goals, and every location the search tried,
coloured by iteration with hits in an off-map colour.

## Layout

```
run_example.sh    end-to-end smoke test -> pipeline/example/
run_trials.sh     config1-vs-config2 trials -> pipeline/trials_0803_1/
stage1_search.py  find attacks (G0 search)
stage2_verify.py  verify from scratch + trace + render;  stage2_baseline.py, stage2_render.py
stage3_targets.py build fuzz targets;  stage3_fuzz.py run SIREN;  stage3_render_attack.py
stage4_curves.py  discovery curves;    stage4_render.py searched-location render
state.py          capture / resume a simulator state (uses scenario/targets.py)
trialconf.py      load a config -> FilterSpecs (USABLE_SCENES, spec_for, ...)
configs/          config1.yml (SPARK shipped), config2.yml (this study)
```

Shared support the stages import: `fuzz/siren/world/*` (run, types, measure,
derived, sim), `fuzz/siren/search/*` (pick, bo), and
`fuzz/siren/scenario/targets.py` (target load / state capture-restore).

## Notes

- `run_from_state` is exact on `G1FixedBase` (56/56) but only partially on
  mobile/locomotion cases; **prefix confirmation** in stage 1 covers this at the
  cost of recall, not soundness. Prefer fixed-base scenes when exactness matters.
- Ground truth to date: ~61 verified attacks (55 insertion, 6 modification)
  across a few `G1*_D2_*` scenes, covering every value-based filter except `cbf`.
- The benchmark is procedurally generated and **not** validated by SPARK; some
  generated scenes start in contact or are already unwinnable, so stage 1's
  gate/screen is mandatory before building on any scene.
