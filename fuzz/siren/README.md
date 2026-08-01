# SIREN

**S**afe-control **I**nfeasibility via **R**eachable-goal **E**xploitation **N**exus

Finds goals that look legitimate but drive a humanoid into states where its
safety filter has no safe control left — producing either a collision (filters
that give up) or a lock-up (filters that yield).

For the full explanation of the maths and every attack, with no jargon assumed,
see **[SIREN_EXPLAINED.md](SIREN_EXPLAINED.md)** (and its PDF).

---

## Layout

Two halves, with a two-call boundary between them:

```
siren/
├── world/                  ground truth: what actually happens
│   ├── types.py            Scene, FilterSpec              (inputs)
│   ├── measure.py          StepMeasurement, RunRecord     (output)
│   ├── derived.py          pure math: authority C, margin g
│   ├── run.py              THE LOOP — orchestrates the three below
│   └── sim/                the SPARK adapter (the only spark_*/mujoco imports)
│       ├── config.py  harness.py  task.py  projected_ssa.py  probe.py
│
├── search/                 the attacker: what to try, how good was it
│   ├── attacks.py          insertion vs modification
│   ├── pick.py             admissibility + random / CEM pickers
│   ├── threat.py           what the attacker may know, see, and want
│   ├── loop.py             ensemble evaluation, budget, ranking
│   └── cli.py              entry point
│
└── test_siren.py           three-layer test suite
```

The entire contract between them:

```python
world.scene()                     -> Scene
world.run(schedule, filter_spec)  -> RunRecord
```

`search/` never imports SPARK or MuJoCo. If it needs something from the
simulator, that capability belongs behind `World` instead.

**We implement no physics and no safety filters.** SPARK supplies the MuJoCo
environment, the robot, the benchmark scenes, the distance computation and all
eight safety algorithms. `world/sim/` adds only: a Task that follows a waypoint
schedule, read-only probes, and config plumbing.

---

## Running it

Always set the thread limits — otherwise the BLAS stack segfaults:

```bash
cd /Users/tiffanyb/Fun/robot/spark
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/Users/tiffanyb/Tools/miniconda3/envs/spark/bin/python
```

### A search

```bash
$PY -m fuzz.siren.search.cli \
    --attack insertion --threat white --picker cem \
    --seed 20 --budget 60 --out /tmp/siren.json
```

Four independent axes — any combination is legal:

| Flag | Options |
|---|---|
| `--attack` | `insertion` · `modification` |
| `--threat` | `white` · `gray` · `gray-proportional` · `weak-black` · `strict-black` · `random` |
| `--observability` | `full` · `coarse` |
| `--picker` | `random` · `cem` |

Other useful flags: `--algo` (the real deployed filter), `--eta` / `--lam`,
`--d-min`, `--index {distance,velocity}`, `--test-case`, `--beta` (penetration
hedge weight), `--stop-on-first`.

### The tests

```bash
$PY -m fuzz.siren.test_siren                      # layers 1-2, seconds
$PY -m fuzz.siren.test_siren --spark              # + real SPARK
$PY -m fuzz.siren.test_siren --spark --all-cases  # + all 8 in-scope cases
```

- **L1 — pure math.** `derived.py` and `measure.py` against hand-made arrays.
  No MuJoCo, milliseconds. This is where the authority/margin identities and the
  DEADLOCK criteria are pinned down.
- **L2 — wiring.** Attacks, pickers, the knowledge budget and the threat presets
  against a fake world. Still no MuJoCo.
- **L3 — SPARK.** Every in-scope benchmark case, all seven filters, the probes,
  and end-to-end searches for all five tiers.

---

## What `--threat` selects

One flag bundles three things, all *derived* from a capability declaration
rather than hand-picked (see `search/threat.py`):

| Threat | Aims at | Ensemble | Penetration hedge |
|---|---|---|---|
| `white` | the exact margin `g` | 1 | no |
| `gray` | smallest `C` (same target as white) | 1 | no |
| `gray-proportional` | smallest `C`, deepest penetration | 1 | yes |
| `weak-black` | smallest `C`, deepest penetration | 3 | yes |
| `strict-black` | smallest **retreat capacity**, deepest penetration | 6 | yes |
| `random` | nothing — the baseline | 1 | — |

Black-box rolls every candidate out against its whole ensemble and keeps the
**worst** score, so a goal only rates highly if it works against *every*
plausible filter.

---

## Three things to know before trusting a result

**The demand must be physically achievable, and this is now enforced.** `g = C −
demand`. If the demanded retreat rate exceeds the authority available where the
filter engages, the filter is infeasible every time it activates: it never
operates, "attack" degenerates to "get close enough to engage", and the margin
is a saturated constant that can rank nothing. Measured on seed 20 with
`d_min=0.02`: at **η=0.5 the filter found a safe control at only 4%** of engaged
steps; at **η=0.02, 100%**. `search()` therefore probes the demand up front and
**aborts with `demand_not_met`** below 50%, unless you pass
`allow_infeasible_demand=True` to study that regime deliberately. This is the
same failure as the old `d_min=0.10` inert filter, reached by a different route.

**Two guidance families, selectable via `SIREN_GUIDANCE`.**

| | reads | status |
|---|---|---|
| `proximity` *(default)* | clearance at handover, fraction engaged, exposure | effect size **2.13** on a labelled set |
| `margin` | `min g` / `min C` over the trajectory | effect size **0.0004** — ranked 0/6 attacks in the top 6, worse than chance |

The margin family is kept so the negative result stays reproducible and the two
can be compared directly. Note that `min g` is now taken over **engaged steps
only** — reporting a margin for a constraint the filter is not enforcing is
meaningless and was *inverted* (candidates that never engaged scored as the most
dangerous of all).

**Soundness is one-directional.** A realized failure confirms a vulnerability.
Finding nothing within budget does **not** certify safety — the search may
simply have missed it.

---

## Scope

In scope: the eight `G1FixedBase_*` benchmark cases. The schedule-driven task
steers the G1's right wrist, so it needs a G1 with arm goals.

Out of scope, by name in `test_siren.py` rather than silently skipped:
`G1MobileBase_*` and `G1SportMode_*` drive a base goal rather than an arm goal;
`Gen3` / `IIWA14` / `LRMate200iD3f` / `R1LiteUpper` are different robots. A test
asserts this list stays in sync with what SPARK actually ships, so a new
benchmark case cannot appear unnoticed.

The velocity-augmented (second-order) safety index requires an
acceleration-controlled `D2` robot — SPARK asserts this — so on a `D1` world
only the distance index runs, and ensemble members that need the other index are
skipped **and reported**, never silently dropped.
