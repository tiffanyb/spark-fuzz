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

## 5. [RESEARCH — revisit later] Coverage-guided fuzzing over the controller's failure / QP-infeasibility regions

Idea (deferred — discussed 2026-06-17): replace/augment the random + targeted-CEM
search with a COVERAGE-GUIDED fuzzer whose coverage metric is tied to the *safety
filter's failure surface*, not generic state-space coverage. Concretely, reward
admissible goals (or goal sequences) that drive the SSA/p-SSA/CBF QP to NEW,
DIVERSE parts of its infeasibility / give-up region (the feasibility boundary) —
i.e. cover the distinct ways an admissible goal can push the certified-safe
controller to the edge where no safe control exists.

Why it matters: (a) it answers "isn't this just an optimization problem?" by
making attack synthesis a *systematic, coverage-driven* method rather than ad hoc
single-objective optimization; (b) the coverage metric is security-relevant (it
maps the attack surface = the controller's failure modes), unlike generic
coverage which would just make it a falsification/testing paper.

IMPORTANT framing note (also discussed): coverage-guided fuzzing is a *method*
upgrade and a *capability* story for the attacker — it does NOT by itself turn
this from a safety paper into a security paper. The security identity must come
from the THREAT MODEL (realistic goal channel: compromised LLM/learned planner,
teleop MITM, or perception spoofing) + DEFENSE EVASION (admissible goals evade
validation; the certified safe controller is the defeated defense) + IMPACT
(collision = integrity violation; deadlock = denial-of-service). Use the fuzzer
as the adversary's automated attack-synthesis engine, not as the thesis.

Open design questions to revisit: how to define a tractable coverage metric over
the QP-infeasibility region (e.g. bucketize by which body/obstacle constraint
saturates, by slack magnitude, by the binding active-set); single-goal vs
goal-sequence fuzzing; how to keep candidates admissible while maximizing
coverage.

## 6. [DEMO PLAN] "An admissible goal defeats a certified-safe humanoid"

Goal: a RoboPAIR-style impact demo, but for the goal-insertion attack — show that a
single *benign, admissible* goal defeats a *certified-safe* controller, with no
jailbreak. Contrast with RoboPAIR/BadRobot (which jailbreak the LLM's alignment to
elicit *harmful* actions): ours needs no jailbreak; the input is a legitimate goal
that slips past intent filters and defeats the *physical safety layer*.

SCENARIO: bedside caregiver / assistive humanoid handing water or medication to a
seated person. The person is exactly what the safety filter (SSA/CBF) exists to
protect. "Bring me the water" → reach the cup, stay clear of the person.

ATTACK: one admissible subgoal inserted first ("reposition your hand near the tray")
— passes every plausibility/ethical check because it is not harmful — leaves the arm
in a config from which the certified controller, completing the *same* handover,
either strikes the person (SSA/p-SSA -> COLLISION = integrity/physical harm) or
freezes mid-reach (r-SSA -> DEADLOCK = denial-of-care / DoS).

DELIVERABLE (Tier 1, buildable now from existing seed-20 clips):
- Annotated TRIPTYCH from ONE inserted goal G1'=[0.164,-0.245,0.008]:
  (a) baseline safe handover, (b) SSA/p-SSA collision, (c) r-SSA 900-step deadlock.
- Relabel the colliding obstacle as "PERSON (safety filter must avoid)", G1' as
  "injected benign waypoint", G1 as "the task"; banner "one admissible goal defeats
  the certified safe controller". Side-by-side image (key frames) + side-by-side video.
- Clips already rendered (clean az=205 angle): cmp_seed20_ssa/pssa/rssa(_900),
  seed20_dec2_baseline, seed20_dec3_feasible.

DELIVERABLE (Tier 2, aspirational, the "real-world" credibility):
- Real Unitree G1 + SPARK/p-SSA (the paper's exact hardware setup; same robot family
  RoboPAIR used). Reproduce the attack on hardware via AVP/teleop or an LLM-planner
  front-end. Even a MINIMAL LLM planner that, under an injected instruction, emits the
  admissible G1' as a subgoal -> makes the channel concrete and pre-empts "is the
  threat real?". Indirect prompt injection (text the VLM reads) is the cleanest vector.

FRAMING TO BAKE IN: no jailbreak needed (benign input); two impact classes
(collision=integrity, deadlock=DoS); attack target is deployed on real commercial
humanoid hardware (Unitree G1) and is the safety-filter family being adopted for
next-gen human-facing robots; preemptive disclosure before it ships/certifies.

## 7. [NEXT ITERATION] Attacks under reduced attacker knowledge (gray-box + black-box)

Full write-up: `NEXT_reduced_knowledge_attacks.md` (PDF in
`docs/NEXT_reduced_knowledge_attacks.pdf`).

Threat-model upgrade in two stages, both anchored on the fact that the ruler
`c(x)` (worst-blend `C(x)=min_y c_y(x)`) is a physical property of the
robot+obstacle — agnostic to both the demand coefficient AND the algorithm.

TIER 1 — GRAY-BOX (know algorithm, not coefficient). The attacker's objective is
**coefficient-free**: `C(x)` depends only on actuator limits, kinematics `J`, and
safety-index geometry `n` — never on `eta`/`lambda`. Fuzz `min_t C(x_t)` lower; the
missing coefficient only sets where the failure threshold sits, and the robot's
own give-up (hard QP infeasible -> `u_ref`; soft -> slack>0) is the oracle. The
per-seed attackability threshold `c_min* = min_t C(x_t)` is environment-intrinsic;
the attack succeeds iff `eta > c_min*`. Collapses to essentially the white-box
attack at a more realistic knowledge level.

TIER 2 — BLACK-BOX (don't even know the algorithm). `C(x)` still transfers
(algorithm-agnostic), but two things degrade: (Loss 1) the demand *shape* is
unknown — constant `eta` is defeated at a GRAZING boundary contact, proportional
`lambda*phi` only by a DEEP one (`phi > c/lambda`) — the 6-8-seeds vs 2-seeds gap.
So "minimize C" alone is insufficient; the robust hedge is to steer to states that
are simultaneously LOW-authority (`C` small) AND DEEP/fast-closing (`phi` large),
the universal worst case for any demand model (scalarize `min_t (C - kappa*phi)`,
sweep `kappa`, or a 2-objective frontier). (Loss 2) the payload isn't predictable
— hard->COLLISION, relaxed->DEADLOCK/push-through, heuristic->underbrake — but the
behavioral oracle still fires and EITHER outcome is a success (integrity vs DoS).
Boundary condition: premise needs a MYOPIC one-step value-based/heuristic filter; a
predictive/recursively-feasible filter with a real control-invariant set would
never enter `C<d` and wouldn't bite. Algorithm-agnostic across the myopic family,
not across all safety layers — which is fine, that family is what humanoids deploy.

Build deltas: (a) coefficient-free scorer `min_t C(x_t)` (reuse
`authority_policies.py`, split off the C-only path, no demand term read);
(b) black-box depth/closing term `phi(x)` + scalarization weight `kappa`;
(c) behavioral oracle labeling COLLISION vs DEADLOCK (hard-filter infeasibility
flag alongside `action_info["violation"]`); (d) report `c_min*` + whether/which
failure fired; (e) pairs with the coverage-guided engine in item 5. Soundness is
one-directional (all tiers): a failure **confirms** vulnerability, but bottoming
out `C`/depth with none does NOT certify safety — report "couldn't within budget,"
never "can't." Scoping: Tier 1 needs the safety index `phi` known (only the rate
unknown); if index-shaping coefficients were also unknown, fall back to purely
behavioral fuzzing.
