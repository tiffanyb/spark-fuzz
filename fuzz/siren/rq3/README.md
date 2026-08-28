# RQ3 — re-validated attack dataset

Every attack record stored anywhere under `fuzz/siren/` was re-simulated from
scratch. This directory holds the ones that still hold, and the evidence for the
ones that do not.

    dataset/         265 symlinks to attacks that survived, + INDEX.json
    visualization/   265 mp4 + 265 png, one per surviving attack
    logs/verdicts/   337 per-attack verdicts (the full record, pass or fail)
    logs/identities.json   the dedup map: 455 files -> 337 distinct attacks

## Why re-validate

These records were produced over months by several different searches, writing
two different schemas, against code that has since changed — the gap-metric fix
(which was off by the robot's own collision radius), the fail-open finding, the
D2/index rework. **A stored record is a claim, not a result.** Re-running is the
only way to know which claims still stand.

## The standard applied

Taken from `pipeline/stage2_verify.py`, which is stricter than "the label said
COLLISION":

    baseline [G0, G1]       must REACH *and* hold clearance > 0 throughout.
                            reached_final alone is not enough -- a run can
                            penetrate an obstacle and still arrive.
    attack [G0, G1', G1]    must be unsafe on EVERY repeat (2x).
    contact leg             2 -> INSERTION, 1 -> MODIFICATION, 0 -> reject
                            (a leg-0 contact means the robot crashed before the
                            inserted goal, so the goal is incidental), and the
                            leg must agree across repeats.

Repeats are not ceremony: these contacts are tens of microns deep, and a
0.155 mm perturbation was previously measured to flip an outcome. A knife-edge
that fires once is not an attack.

Records carrying `obstacles_world` have those obstacles PINNED — the scene is
part of the claim. Records without them (the teleop hunts) run on the scene's
own obstacles for that seed.

## Results

    455 files  ->  337 distinct attacks  (84 duplicate copies collapsed)

    VALID     265    INSERTION 134   MODIFICATION 124   DEADLOCK 7
    INVALID    70    attack did not reproduce 55
                     baseline was itself unsafe 15
    ERROR       2    SPARK whole-body IK cannot solve the scene

**21% of stored attacks no longer hold.** The two failure modes are worth
separating:

* **55 did not reproduce.** The attack ran but did not come out unsafe on every
  repeat. Given the micron-scale contact depths, these are knife-edges that the
  original search caught once.
* **15 had an unsafe baseline.** The legitimate task `[G0, G1]` itself collides
  in that scene, so there was never an attack to observe — the robot was going
  to crash with or without the inserted goal. This is the failure mode that the
  "baseline must REACH *and* never penetrate" clause exists to catch, and it is
  exactly the bug that invalidated 12 controls once before.

**6 attacks survive but were reclassified**: 4 INSERTION -> DEADLOCK and
2 INSERTION -> MODIFICATION. They are still attacks; the recorded kind was wrong.
`INDEX.json` flags these with `kind_matches_claim: false`.

**2 are unverifiable, not invalid.** Both are `G1MobileBase_D2_WG_SO_v0`
base-channel scenes where SPARK's casadi whole-body IK exits with
`Maximum_Iterations_Exceeded`. Deterministic across retries. They are excluded
from the dataset because no verdict could be reached, which is not the same as
having been refuted.

### Surviving attacks by filter

    sss 65    cbf 63    rcbf 54    pssa 25    rssa 24    ssa 22    rsss 12

All seven filters are represented, across 8 scenario cases.

## Reproducing

    python -m fuzz.siren.rq3.revalidate --shard 0 --of 8    # verdicts
    python -m fuzz.siren.rq3.build_dataset                  # symlinks
    python -m fuzz.siren.rq3.render --shard 0 --of 8        # videos

`dataset/` holds symlinks, never copies, so a link cannot drift from the record
it points at. `INDEX.json` carries every surviving attack's identity, its
original source, the duplicate copies that collapsed into it, and the measured
baseline/attack numbers from the re-run.
