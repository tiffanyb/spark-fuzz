# Known bugs

## BUG-1 (FIXED 2026-07-31, HIGH) — `clearance()` counted contacts on links the filter is told to ignore

**Symptom.** Collisions were recorded where the safety index simultaneously
reported safety: `clearance = -0.0001` (contact) while `phi = -0.066`
(comfortably safe), `engaged = False`, `gave_up = 0`, on every sampled run.
Two numbers that are supposed to describe the same event disagreed completely.

**Cause.** They were describing *different pairs*.

```
collision checker (Harness.clearance)  25 volumes x 5 obstacles = 125 pairs
safety index (phi)                     425 entries = 125 env + 300 self-collision
                                       of the 125 env pairs, only 95 are monitored
```

SPARK deliberately excludes six robot volumes from *environment* collision
checking via `env_collision_vol_ignore`:

```
waist_yaw_joint, waist_roll_joint, waist_pitch_joint,
pelvis_link_1,   pelvis_link_2,    pelvis_link_3
```

(125 - 95 = 30 = 6 ignored volumes x 5 obstacles.) These are structurally near
the base and would otherwise trip constantly.

`Harness.clearance()` iterates **all** of `robot_cfg.CollisionVol`, including
those six. In every collision examined the offending volume was
**`pelvis_link_3` — an ignored one.** Meanwhile `max phi` over ALL pairs was
`+0.110`, so the index does see danger; that entry is simply outside the
monitored subset.

**Impact.** An unknown fraction of reported "attacks" are pelvis/waist contacts,
not safety-filter failures. In the sample examined it was *every* collision.
Potentially affected:

* the 22% D2 attack rate,
* the 70% vs 25% proximity-vs-random comparison,
* the seed screen (seeds 0 and 5 ranked "high-rate"),
* every `is_attack_success()` verdict that depended on `clearance < 0`.

The *relative* comparison between guidance strategies may survive, since all
arms shared the same faulty detector — but the absolute rates, and the claim
that these are *filter* failures, do not.

**Fix.** `clearance()` must respect `env_collision_vol_ignore` and measure only
the pairs the filter is accountable for. Then re-run every affected result.

**Regression guard to add at the same time.** A permanent assertion that the two
views agree:

```
if clearance < 0:  assert max(phi over MASKED env pairs) > 0
```

Had that existed it would have surfaced immediately rather than after several
rounds of conclusions were drawn on top of it. Any future measurement pair that
is meant to describe the same event should get the same treatment.

**Repro.** `python -m fuzz.siren.diagnose_kind0 --seed 5 --case G1FixedBase_D2_AG_SO_v0`
and `python -m fuzz.siren.diagnose_blindspot --seed 5 --case G1FixedBase_D2_AG_SO_v0`.

### Resolution (2026-07-31)

`Harness.clearance()` now takes `guarded_only=True` and masks the ignored pairs to
`+inf` via the safety index's own `env_collision_mask`, so the collision check and
`phi` describe the same set of pairs. `world/run.py` carries the regression guard:
it warns once if any step reports `clearance < 0` while `phi <= 0`. Tests 60/60.

**Measured impact — the fix mattered, and unevenly.** Attack rates before and
after, unguided (random picker), insertion attack, `eta=0.02`:

| scene | pre-fix | post-fix | verdict |
|---|---|---|---|
| D2 seed 0 | 65% (13/20) | 67% (20/30) | real, survives |
| D2 seed 1 | 60% (12/20) | 47% (14/30) | real, survives |
| D2 seed 5 | — | 0/30, filter never engages | inert scene |
| D1 seed 0 | — | 97% (29/30) | near-saturated, see below |
| D1 seed 5 | 23% | **0/30** | was entirely artefact |
| D1 seed 20 | ~25% (random arm of the guidance test) | **0/30** | was entirely artefact |

So the D2 attacks are real and essentially unchanged, while **the D1 seed-5 and
seed-20 attack rates collapse to zero** — on those scenes every "attack" was a
pelvis/waist contact the filter was never asked to prevent.

**Consequence: the guidance comparison must be re-run.** The
"proximity 28/40 = 70% vs random 10/40 = 25%" result was measured by
`validate_fixes.py` on its default scene, **D1 seed 20** — which now yields 0/30.
That comparison is therefore void, not merely rescaled: both arms were ranking
candidates by an artefact. Re-running on D2 seed 1, which retains a genuine 47%
base rate. The *direction* of the earlier conclusion (proximity beats the
margin/certificate objective) may well survive, but it has to be re-established
rather than assumed.

**D1 seed 0 is a separate trap.** 97% is not a strong attack, it is a scene with
almost no authority to begin with: `C_min = 0.024` against `eta = 0.02`, so the
filter sits on the edge of its budget everywhere and nearly any inserted goal
tips it. `demand_achievable` passes it (`frac_feasible = 1.0`) because feasibility
is judged at engaged steps only. Unsuitable for comparing search strategies —
there is nothing to search for — and it should probably be excluded the same way
inert scenes are.

---

## BUG-2 (OPEN, LOW) — mechanism probe can disagree with the insertion gate

`classify_mechanism()` re-runs the one-hop leg independently of the search's
gate, so a candidate admitted by the gate can show `onehop = COLLISION` in the
mechanism report (observed in the seed-5 eta=0.02 run: 2 of 5). Gate and probe
should share one screening result rather than each rolling their own.
