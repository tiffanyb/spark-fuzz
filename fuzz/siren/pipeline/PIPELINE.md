# SIREN evaluation pipeline

Three stages: **find** ground-truth attacks in the SPARK benchmark, **verify and
record** them, then **score SIREN** on whether it rediscovers them.

Everything is run from the repo root with

```bash
cd /Users/tiffanyb/Fun/robot/spark
export PYTHONPATH=/Users/tiffanyb/Fun/robot/spark
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
```

Use **absolute paths** for `--dir` / `--out-dir`. The shell keeps its working
directory between commands, and a relative path resolved from the wrong cwd
silently matches zero files — that already cost one full render pass.

---

## Notation

```
original scenario   home -> scenario_goal
attack schedule     [G0, G1', G1]   run from the home pose, so
                      wp_idx 0 : home -> G0     setup
                      wp_idx 1 : G0   -> G1'    reaching the inserted goal
                      wp_idx 2 : G1'  -> G1     back to the legitimate goal
with                G1' := home (EE position),  G1 := scenario_goal
```

Leg 2 and the original scenario share endpoints; they differ only in departure
state — the original starts at **rest**, leg 2 departs **in motion**. That is
the mechanism the whole thing turns on.

```
contact on leg 2  ->  INSERTION     the inserted goal breaks the return trip
contact on leg 1  ->  MODIFICATION  the robot is destroyed reaching the inserted goal
contact on leg 0  ->  rejected      crashed before the inserted goal was active
```

---

## Stage 1 — find attacks

**Script:** `fuzz/siren/scenario/g0_search.py`

```bash
$PY -m fuzz.siren.scenario.g0_search \
    --algo sss --seeds 1-20 --n-worlds 3 --grid 5 --lam 0.3 \
    --out-dir /abs/path/to/out
```

| flag | meaning |
|---|---|
| `--algo` | one of `ssa rssa pssa cbf rcbf sss rsss` |
| `--seeds` | seeds to **scan**, `1-20` or `0,1,2` |
| `--n-worlds` | how many gate-**passing** worlds to search per scenario |
| `--scenes` | any benchmark case names, comma separated; default is the built-in six |
| `--grid` | N³ grid of G0 candidates over the workspace box |
| `--lam` | CBF-family demand gain (see *gain* below) |
| `--relax` | count DEADLOCK as an attack (on by default) |

### Gate

```
run [scenario_goal] from home  ->  must COLLIDE or DEADLOCK
```

A failing seed is **skipped and the next tried**; only passers count toward
`--n-worlds`. Each output records `seeds_scanned`, a per-seed `gate_log`, and
`world_index`, so any result traces back to its exact world.

The gate is a **heuristic, not a precondition** — it is leg 2 *from rest*. Not
necessary (leg 2 in motion may collide where rest reaches), not sufficient. A
failed gate means "no cheap raw material", **not** "no attack exists".

### Per candidate

```
C1   [g0, G1] from home must REACH *and* keep clearance > 0 at every step
     -- capture s0 the instant wp_idx goes 0 -> 1 (the true handover:
        g0 is INTERMEDIATE, so the arm passes through in motion)
C1b  s0 legitimate: clearance > 0 and not already infeasible (mu <= 0)
C2   resume from s0, run [G1', G1]; must COLLIDE or DEADLOCK
C3   failing leg -> 0 = MODIFICATION, 1 = INSERTION
     (indices shift: home->g0 is not simulated in the resumed run)
prefix confirm   re-run [G0, G1', G1] from home; only confirmed hits are written
```

**C1 must test clearance, not just `reached_final`.** A rollout can penetrate an
obstacle and still arrive, and `measure.classify_run` calls that COLLISION
because collision is tested *before* reached. Using `reached_final` alone
produced 12 bogus cbf controls and 4 bogus sss ones.

**Prefix confirmation is required**, because `run_from_state` is exact only on
some robots (below). On one mobile-base scene it rejected 20 of 32 candidates.

### Choosing the gain

`cbf rcbf sss rsss` use a **proportional** demand `lambda*phi`; `ssa rssa pssa`
use a **constant** demand `eta`. At the shipped `lambda = 10` the proportional
family is often infeasible — `lambda*phi > C` — and then collisions are
artifacts of the gain, not attacks. Sweep `lambda` first and search only where
the filter is feasible *and* still fails.

**Support files:** `world/run.py`, `world/types.py`, `world/measure.py`,
`world/derived.py`, `world/sim/*`, `search/pick.py` (`is_admissible`),
`scenario/state.py`.

---

## Stage 2 — verify, render, record

### 2a. Collect and re-verify

**Script:** `fuzz/siren/scenario/collect_verified.py`

```bash
$PY -m fuzz.siren.scenario.collect_verified \
    --src '/abs/path/g0_*/control_*.json' \
    --out-dir /abs/path/verified_attacks
```

Re-runs everything **from scratch** rather than trusting the search:

```
baseline [G0,G1]       REACHES with clearance > 0 at every step
attack [G0,G1',G1]     COLLIDES or DEADLOCKS on all 3 repeats
contact leg            consistent across repeats -> INSERTION or MODIFICATION
```

3 repeats matter because these contacts are 20–500 µm deep and a 0.155 mm
perturbation was measured to flip an outcome.

Writes one flat JSON per attack plus `INDEX.json` (kept and rejected, with
reasons).

`verify_modification.py` does the same for leg-1 attacks alone;
`verify_control.py` is the older insertion-only checker and **rejects
modification attacks by design** — do not use it to judge them.

### 2b. Render

**Script:** `fuzz/siren/scenario/render_verified.py`

```bash
$PY -m fuzz.siren.scenario.render_verified --dir /abs/path/verified_attacks
```

Writes `<tag>_attack.mp4` and `<tag>_contact.png` per attack: leg label, live
clearance, nearest volume/obstacle pair, red border and solid-red obstacle at
contact. `--skip-existing` is on; `--only <substring>` filters; `--stride N`
renders every Nth idle step but never skips once contact is detected.

### 2c. Save the full trace — **NOT IMPLEMENTED**

Currently only a summary JSON and the video are kept. See *Missing code*.

---

## Stage 3 — score SIREN against ground truth

The point: an empty search is ambiguous between "no attack exists" and "the
search cannot find one". A ground-truth attack at known coordinates removes
that. `G1_prime_truth` is loaded but **never given to the search**.

### 3a. Build targets

**Script:** `fuzz/siren/scenario/make_targets.py`

The stage-2 output already carries every field `make_targets.py` reads
(`case, algo, seed, index, max_steps, d_min, eta, lam, k, G0, G1_prime, G1,
bounds, keepout, obstacles_world`), so it can be pointed straight at it:

```bash
$PY -m fuzz.siren.scenario.make_targets \
    --verified '/abs/path/verified_attacks/*.json' \
    --out-dir  /abs/path/targets
```

It drives the robot to G0, captures the handover state, writes the target, then
**reloads it and re-proves the attack**. A target that fails the round trip is
deleted rather than kept.

Caveat: it will also pick up `INDEX.json` — filter it out or the run errors.

### 3b. Run SIREN

**Script:** `fuzz/siren/scenario/run_fuzzer.py`

```bash
$PY -m fuzz.siren.scenario.run_fuzzer \
    --target /abs/path/targets/<name>.json \
    --pickers random,random_seeded,cem,cem_seeded,bo,bo_seeded \
    --budget 60 --search-seeds 0,1,2 \
    --out /abs/path/fuzz_<name>.json
```

Six search strategies: `random cem bo` × plain/`_seeded` (obstacle-anchored initial design).
All strategies spend the same first `_N_INIT = 8` evaluations on a shared design, so
comparisons below 8 are meaningless and "first hit at eval 8" is an artifact.

Reports per strategy: attacks found, evaluations-to-first-hit, closest approach to
the planted `G1'`. Every evaluation is logged, not just hits.

The evaluation contract per candidate is `fuzz/siren/pipeline.py` (steps 0–6).
Note two documented deviations: `mu` does **not** predict contact here
(corr −0.14 over 9936 candidates), so step 4 scores on **braking margin**; and
step 5's `mu`-based Kind-1/Kind-2 classifier should be replaced by the
braking-distance test (83% validated) — still open.

The fuzzing evaluation should record all searched location. For each searched location, record:
  - a UUID for this location
  - time of the searched location
  - its "father" location's UUID
  - the score of this search
  - whether it is an attack
  - and other information if you see fits, to justify the effectiveness and efficiency of the fuzzing process.

---

## Stage 4 — measure and visualise the search

Consumes the stage-3b output (`fuzz_<target>.json`), one entry per target. Two
deliverables: **discovery curves** and a **searched-location render**.

### 4a. Discovery curves

**Script:** `fuzz/siren/scenario/search_curves.py` — *not implemented*

Per target, per `(strategy, search_seed)`, two cumulative curves:

```
attacks vs trials   x = evaluation index 1..budget,  y = cumulative attacks
attacks vs time     x = wall-clock seconds,          y = cumulative attacks
```

Both are step functions from the per-evaluation log, so no re-simulation is
needed — the data is already in `evaluations`.

*Trials* is the primary axis. It is the honest measure of search efficiency
because one evaluation is one full rollout regardless of which strategy proposed
it, so it is invariant to machine load and to how many searches were running
concurrently.

*Time* is the secondary axis and is **not currently computable**: only a single
`elapsed_s` per strategy run is stored, not a per-evaluation timestamp. That is
Missing-code #8. Until it lands, the time curve can only be approximated as
`elapsed_s x eval/n_eval`, which assumes uniform cost per evaluation — false
here, since a rollout aborts at contact but a deadlock runs the full horizon, so
attacks are *systematically cheaper* than misses and the approximation biases
the curve in the flattering direction. Do not publish the approximate version.

Aggregate across search seeds with median and inter-quartile band rather than a
mean: with 3 seeds a single unlucky replicate distorts a mean badly.

Report alongside each curve, since a curve alone hides them:

```
evaluations-to-first-attack       (None if never found)
total attacks / budget            the hit rate
n_inadmissible                    proposals rejected on geometry, which cost
                                  nothing and so do not appear on either axis
```

The first 8 evaluations are a shared initial design identical across strategies,
so **every curve is identical below eval 9 by construction**. Mark that region
on the plot; otherwise it reads as agreement between strategies when it is an
artifact.

### 4b. Searched-location render

**Script:** extend `fuzz/siren/render_fuzz_case.py` — *not implemented*

One 3-D scene per target: the robot posed at the G0 handover, obstacles, `G0`,
`G1`, the planted `G1'`, and **every location the search tried** — not only the
hits.

```
tried locations   rainbow-coloured by search iteration
                  first iteration -> violet ... last iteration -> red
identified attacks   a single distinct colour, outside the rainbow
                     (white or black), and drawn larger
G0 / G1 / planted G1'   the existing blue / green / yellow markers
```

Rendered as a MuJoCo scene with the robot present — orbit mp4 plus a still —
matching the existing renderers, not a matplotlib scatter.

Two decisions to settle before implementing:

**What "iteration" means.** Colour by **ask/tell round**, not by evaluation
index. With `--budget 60 --batch 10` there are 6 rounds, which is a readable
number of distinct hues; 60 is not. The round is also the unit at which CEM
updates its mean/std and BO refits, so it is the meaningful step. Store the
round index explicitly rather than deriving it as `eval // batch`, which breaks
whenever the final batch is short.

**Attack colour must not be a rainbow hue.** If attacks were, say, red, they
would be indistinguishable from late-iteration misses. Use a colour outside the
map entirely.

The existing `render_fuzz_case.py` already draws the robot at `state_at_G0`,
the obstacles and the goal markers, and colours discovered goals by self-check
verdict. What it does **not** do is read the full `evaluations` list — it reads
`hits` only, so it currently cannot draw misses at all. That is the extension.

---

## Missing code

| # | what | why it matters |
|---|---|---|
| 1 | **Full-trace recorder** — per-step `qpos/qvel`, command state, `phi`, `mu`, clearance, engaged, `u`/`u_ref`, dumped alongside each verified attack | stage 2c is unimplemented; today only a summary and a video survive, so any post-hoc question needs a re-run |
| 2 | **`run_fuzzer.py` C1 fix** — its baseline check has the same `reached_final` weakness stage 1 had | it can currently accept a target whose baseline collides |
| 3 | **Modification support in `run_fuzzer.py`** — it counts an attack only when contact is on leg 2 | 6 of 61 ground-truth attacks are modification; SIREN is currently unscored against them |
| 4 | **`make_targets.py` INDEX filter** and support for a `kind` field | otherwise stage 3a errors on `INDEX.json` and loses insertion/modification labelling |
| 5 | **Stage-3 report script** — aggregate hit rate, evals-to-first-hit, distance-to-truth across targets and strategies, driven off the provenance log in #8 | `run_fuzzer` reports per target only; no cross-target summary exists |
| 6 | **Deadlock leg attribution in `collect_verified.py`** — DEADLOCK is accepted as INSERTION without a leg test | safe today only because `classify_run` requires `on_final_leg`; fragile if that changes |
| 7 | **cbf coverage** — no verified cbf attack exists | 6 of 7 filters are covered; cbf is the gap |
| 8 | **Search-provenance recorder** — per-location UUID, wall-clock timestamp, and parent UUID | required by the stage-3 spec above; none of the three exist today. The timestamp is also what makes stage 4a's time curve computable |
| 9 | **Picker-side lineage reporting** — `ask()` must return, per proposal, which prior evaluation(s) it descends from | prerequisite for #8; no picker exposes this, so the parent link cannot be reconstructed after the fact |
| 10 | **`search_curves.py`** — cumulative attacks vs trials and vs time, per target and `(strategy, seed)`, with median/IQR across seeds | stage 4a; no discovery-curve computation exists |
| 11 | **Round index in the evaluation log** — record the ask/tell round explicitly | stage 4b colours by round; deriving it as `eval // batch` is wrong whenever the final batch is short |
| 12 | **`render_fuzz_case.py` extension** — read the full `evaluations` list, colour by round on a rainbow map, attacks in an off-map colour | stage 4b; it currently reads `hits` only and so cannot draw the misses at all |

### On #8 and #9

`run_fuzzer.py` already logs every evaluation — `eval, cand, score, is_attack,
label, contact_leg, min_clearance, n_steps, n_gave_up, engaged, dist_to_truth`,
plus a lighter record for candidates rejected as inadmissible. Missing are the
**UUID**, the **timestamp**, and the **parent UUID**. The first two are
mechanical. The third is not, for two reasons.

*No picker reports lineage.* `Picker.ask(n)` returns bare coordinates. Nothing
in `pick.py` or `bo.py` records what a proposal came from, and it cannot be
recovered afterwards from the coordinates alone.

*"Parent" is not well defined for two of the three strategies.* A decision is
needed before implementing, and it should be recorded in the schema rather than
left implicit:

| strategy | what a proposal actually descends from |
|---|---|
| `random` | nothing — every draw is independent; parent is `null` |
| `cem` | the whole elite set (top ~1/3) that produced the current mean and std, not one point. Candidates: the elite set as a list, or the single best elite, or the distribution's identity |
| `bo` | the entire observation history through the GP posterior. There is no single parent; the honest link is to the round, or to the nearest observed neighbour |

A defensible schema is: `parent_ids` as a **list** (empty for `random`, the
elite set for `cem`, the round's full history or a nearest-neighbour reference
for `bo`), plus a `round` index. That keeps the genealogy honest instead of
inventing a single ancestor where none exists.

Worth recording alongside, to support the efficiency claims: wall-clock and
simulated-step cost per evaluation, the round index, the strategy and search
seed, whether the candidate came from the shared initial design, and — for
`cem` — the current mean and std, since those are what the lineage is really
tracking.

---

## Known limitations

**`run_from_state` is not universally exact.** After the fix in `state.py`
(`refresh_task_cache` — `get_info()` does not refresh the task's cached
`robot_base_frame`, so a resume built goals from a stale base pose):

```
G1FixedBase   all 8 variants      exact      56/56
G1MobileBase  6 of 8 variants     exact
G1MobileBase  2 variants          1 mismatch each (horizon-limited runs)
G1SportMode                       0/7        locomotion state uncaptured
IIWA14Single  D1, D2              0/10
Gen3Single, LRMate, R1LiteUpper   mixed within a single case
```

Mixed-within-a-case means the carrier is **path-dependent** and still
unidentified. Prefix confirmation covers this — it costs recall, not soundness.

**The benchmark is procedurally generated.** A "scenario" is a family; obstacles
and goals are sampled per seed from `benchmark_test_case_generator.py`. Home is
identical across every G1 case (`[0.2498, -0.2405, 0.1228]`); only obstacles and
goals move.

**Generated scenes are not validated by SPARK.** Some start in contact, some
start infeasible, some are already unwinnable (D1 + dynamic obstacles: the
obstacle closes at 0.400 m/s against 0.026 m/s of retreat authority, and the
first-order index hardcodes `L_f = 0` so the filter cannot see it move).
Screening is mandatory before building on any scene.

**Ground truth as of this writing:** 61 verified attacks — 55 insertion,
6 modification — from `G1MobileBase_D2_WG_DO_v1` s0, `G1FixedBase_D2_AG_SO_v0`
s1, and `G1MobileBase_D2_WG_SO_v1` s2. Covering ssa, pssa, sss, rcbf, rsss.
None for cbf.

Stage-1 searches for `cbf`, `rsss` and `rssa` (seeds 1–20, 3 gate-passing worlds
each) are still running and have reported further candidates — 8 each for `rsss`
and `rssa`, none for `cbf`. Those are **not** in the count above: a candidate
only becomes ground truth after `collect_verified.py` re-runs it from scratch,
and on past evidence a meaningful fraction does not survive that.
