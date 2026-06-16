# `fuzz/` — TODO / known limitations

## 1. Constrain `G1_base` sampling so reachability is guaranteed by construction

**Current behavior.** `G1` is whatever the benchmark's own random sampler
produced for the right arm (`harness.scene_info()` reads
`task.robot_goal_right.frame`). Its only validity guarantee is *static*: in
workspace bounds and `arm_goal_keepout` from obstacle centers. Whether the
**controller can actually drive G0 -> G1 safely** is unknown until we run it.

We therefore gate it at runtime: `GoalInsertionFuzzer.run()` runs the baseline
`[G1]` first and **aborts without fuzzing** if `G1` is not reached
(`abort_reason = "baseline_unreachable"`). Correct, but wasteful — a bad seed
burns a full trial and produces no data.

**Ideal future behavior.** Generate `G1` from a *bounded constraint set* that is
reachable-from-`G0` by construction, so the baseline gate becomes a formality
rather than a filter. Sketch:

- Characterize an admissible-and-reachable region `R(G0)` — the set of goals for
  which the reactive controller provably (or empirically, with high coverage)
  reaches from `G0` without collision. Candidates:
  - a certified inner approximation (e.g. a box / ellipsoid in EE space that
    stays inside the arm's collision-free reachable set given the fixed scene),
  - or an empirical region built by sweeping goals on a grid once per scene and
    keeping those that reach, then sampling `G1` (and `G1'`) only from it.
- Sample both `G1` and the inserted `G1'` from `R(G0)`. Then *every* individual
  goal is reachable by definition, and the only thing under test is the
  **sequence** G0 -> G1' -> G1 — which is exactly the attack we care about.
- Bonus: this removes the need for the per-candidate one-hop screen (below),
  since one-hop reachability would be guaranteed, not checked.

This turns "reject after running" into "never propose an invalid goal," which is
both cleaner and much cheaper at scale.

## 2. One-hop screen is currently empirical (one trial), not certified

To keep inserted goals non-trivial we run `[G1']` alone and require it to reach
before testing `[G1', G1]` (rejects "trivial one-hop traps"). This is an
empirical, single-rollout check. If item 1 lands (reachable-by-construction
region), the screen is redundant. Until then, consider >1 rollout or a tolerance
band if controller determinism is ever in doubt.

## 3. Admissibility ignores QP-feasibility / phi>=0 at the issue state

`is_admissible` checks bounds + center-based obstacle clearance + optional IK.
A stronger notion (noted in RESEARCH_PLAN_0611.md §15) is to also require the
safety index `phi >= 0` and the QP to be feasible at the moment `G1'` is issued.
`ik_check` is the current kinematic stand-in; layering the safety-index check on
top is a follow-on.

## 4. Center-based clearance inherits SPARK's geometry-agnostic convention

`_obstacle_clearance` matches SPARK's `_init_goal` exactly (center-to-center vs
`arm_goal_keepout`, no radius subtraction). This is intentional — it keeps `G1'`
indistinguishable from a benchmark-sampled goal — but it means a large obstacle
could admit a goal near/inside its surface. If we ever want physically-strict
clearance, that's a *deliberate divergence* from SPARK, not a bug fix.
