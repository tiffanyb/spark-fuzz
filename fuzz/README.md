# `fuzz/` — Phase-A single-arm adversarial goal-insertion harness

A self-contained module that tests whether a single **admissible** intermediate
goal `G1'`, inserted before a legitimate goal `G1`, can trap SPARK's reactive
safe controller so the arm can no longer reach `G1` safely.

Background and rationale: `../RESEARCH_PLAN_0611.md` §15.

## Idea

```
baseline:   G0 ──────────────► G1        reached safely
attack:     G0 ──► G1' ──────► G1        cannot reach G1 safely
                   (admissible)
```

Every `G1'` is admissible (within the arm workspace, clear of obstacles by the
same keepout SPARK uses when sampling its own goals). The contribution is the
metric SPARK/the SSA paper lack: a **goal-trap / deadlock** detector — a run that
stays safe but never reaches `G1`.

## Layout (fuzzing kept separate from core SPARK)

| File | Role |
|---|---|
| `goal_insertion_task.py` | `SingleArmGoalInsertionTask(BenchmarkTask)` — drives the right-arm goal along a scripted `[G1', G1]` schedule; self-registers into `spark_task` (no core edits) |
| `admissibility.py` | `is_admissible` / `sample_admissible` — bounds + obstacle clearance (+ optional IK) |
| `metrics.py` | `StepRecord`, `TrialOutcome`, `classify_trial` → REACHED / COLLISION / DEADLOCK / TIMEOUT |
| `config.py` | `build_single_arm_config` — FixedBase, single-arm, fixed scene, headless, r-SSA |
| `harness.py` | `SingleArmHarness` — builds env+algo, runs one trial, returns a `TrialOutcome` |
| `fuzzer.py` | `GoalInsertionFuzzer` — random-admissible search; ranks attacks |
| `run_phase_a.py` | CLI entry point |

The module touches **no core SPARK file**: the custom task is registered into the
`spark_task` namespace at import time so `initialize_class` can resolve it.

## Run

From the spark repo root, inside the SPARK conda env (MuJoCo required):

```bash
mjpython -m fuzz.run_phase_a --safe-algo rssa --seed 0 --candidates 50
```

Key flags: `--safe-algo {ssa,rssa,cbf,rcbf,sss}`, `--test-case`, `--seed`,
`--candidates`, `--max-steps`, `--ik-check`, `--viewer`, `--out report.json`.

## Design notes

- **Fixed scene across trials** via `seed_list=[seed]`, so baseline and every
  candidate share the same obstacles/start/`G1`.
- **`G1` = the benchmark's own sampled right goal.** A **hard baseline gate**
  runs `[G1]` first and aborts the whole search if `G0->G1` is not reached
  (an invalid scene would make any "attack" meaningless). See `TODO.md` item 1
  for making `G1` reachable-by-construction instead.
- **Two-hop only — one-hop traps are rejected.** Each candidate `G1'` must first
  pass a screen (`[G1']` alone reaches safely from `G0`); only then is the attack
  `[G1', G1]` evaluated. This rules out trivial traps where `G1'` itself is the
  problem (e.g. a goal placed where reaching it alone already fails). The
  contribution is purely **sequence-induced** loss of controllability: each goal
  is individually safe, the order is not.
- **r-SSA is the default controller** because its QP exposes a graded slack
  signal (`action_info["violation"]`); `ssa`/`cbf` only report slack on
  infeasibility.
- **Kinematic, headless** by default (`use_sim_dynamics=False`, no viewer) for
  determinism and speed; the trap is a controller property, not a physics one.

## Status / next

- Phase A: single arm, random-admissible search (this module).
- Follow-ons (not yet implemented): greedy/myopic + random baselines for the
  non-triviality comparison; p-SSA (not shipped in SPARK — needs porting) to
  attack the paper's strongest method; Phase B dual-arm (inter-arm trap).
