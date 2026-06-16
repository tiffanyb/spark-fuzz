"""
Per-step records and trial-outcome classification.

This is the metric layer the SSA-paper evaluation lacks (RESEARCH_PLAN_0611.md
§15.2): their metrics (J tracking, S collision, R_Feas feasibility) never detect
a *goal-trap* — a run that stays safe but never reaches G1. We add exactly that.

Outcome labels:
  REACHED   - the end-effector reached the final goal G1.
  COLLISION - the robot penetrated an obstacle (min robot-obstacle dist < margin).
  DEADLOCK  - never reached G1, never collided, and the EE is GENUINELY trapped:
              the trial ran a full horizon, the EE is still far from G1, and over a
              long trailing window it made negligible net progress and barely moved.
  TIMEOUT   - never reached G1, never collided, but NOT a confirmed trap (still
              approaching, or too close to call, or the horizon was too short).

Hardening (why the strict DEADLOCK test): a short horizon that cuts off during a
brief near-stall just before arrival used to be mislabelled DEADLOCK. The seed-4
attack is the canonical false positive -- it "deadlocked" at 160 steps but REACHES
at step 164. DEADLOCK now requires (a) a full-length trial, (b) the EE still far
from the goal, and (c) a persistent freeze (low net progress AND low motion) over
a long window -- so "slow but arriving" is TIMEOUT, not a trap. Only DEADLOCK and
COLLISION count as attack successes; TIMEOUT is explicitly inconclusive.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class StepRecord:
    step: int
    wp_idx: int               # which scheduled waypoint is active
    dist_final: float         # EE -> final goal G1 (world)
    reached_final: bool
    min_dist_env: float       # min robot-obstacle distance (<0 = penetration)
    peak_slack: float         # max QP slack this step (relaxed controllers)
    trigger_safe: bool        # safety filter engaged this step
    collided: bool


@dataclass
class TrialOutcome:
    label: str                       # REACHED | COLLISION | DEADLOCK | TIMEOUT
    reached_final: bool
    steps_to_final: Optional[int]    # step index of first reach, else None
    collided: bool
    worst_min_dist_env: float        # most-negative (deepest penetration / closest)
    peak_slack: float                # worst slack over the trial
    mean_slack: float
    n_steps: int
    final_dist: float = np.inf       # EE -> G1 distance at the last step
    tail_progress: float = 0.0       # net approach over the stall window (>0 = approaching)
    tail_std: float = 0.0            # std of EE -> G1 over the stall window (motion proxy)
    schedule: List = field(default_factory=list)   # waypoints used (base frame xyz)

    def is_attack_success(self) -> bool:
        """An inserted goal 'succeeds' only if it produces a CONFIRMED unsafe
        outcome: a genuine deadlock or an actual collision. TIMEOUT is
        inconclusive and does NOT count."""
        return self.label in ("DEADLOCK", "COLLISION")


def classify_trial(records: List[StepRecord], schedule=None,
                   collision_margin: float = 0.0,
                   reach_radius: float = 0.05,
                   stall_window: int = 100,
                   stall_std_eps: float = 3e-3,
                   progress_eps: float = 1e-2,
                   far_factor: float = 2.0) -> TrialOutcome:
    """Reduce per-step records to a single labelled outcome.

    DEADLOCK requires ALL of:
      - not reached and not collided,
      - the trial ran at least `stall_window` steps (a full-length attempt),
      - the EE ends FAR from G1 (final_dist > far_factor * reach_radius), and
      - a persistent FREEZE over the last `stall_window` steps: both the spread
        (std < stall_std_eps) and the net approach (|progress| < progress_eps) are
        tiny -- i.e. it is neither jittering toward the goal nor creeping in.
    Anything else that didn't reach/collide is TIMEOUT (inconclusive).
    """
    n = len(records)
    schedule = list(schedule) if schedule is not None else []

    if n == 0:
        return TrialOutcome("TIMEOUT", False, None, False, np.inf, 0.0, 0.0, 0,
                            schedule=schedule)

    slacks = np.array([r.peak_slack for r in records], dtype=float)
    min_dists = np.array([r.min_dist_env for r in records], dtype=float)
    dist_final = np.array([r.dist_final for r in records], dtype=float)

    collided = bool(np.any(min_dists < -abs(collision_margin)))

    steps_to_final = None
    for r in records:
        if r.reached_final:
            steps_to_final = r.step
            break
    reached_final = steps_to_final is not None

    W = int(min(stall_window, n))
    tail = dist_final[-W:]
    tail_progress = float(tail[0] - tail[-1])   # >0 = approaching the goal
    tail_std = float(np.std(tail))
    final_dist = float(dist_final[-1])

    # The freeze must occur on the FINAL goal leg (the last waypoint). Otherwise a
    # slow-to-reach inserted goal G1' eats the whole horizon and the run freezes on
    # the *G1' leg* with no budget left for G1 -- a horizon artifact, not a trap.
    final_wp = (len(schedule) - 1) if schedule else 0
    on_final_leg = all(r.wp_idx >= final_wp for r in records[-W:])

    if collided:
        label = "COLLISION"
    elif reached_final:
        label = "REACHED"
    else:
        far = final_dist > far_factor * reach_radius
        frozen = (tail_std < stall_std_eps) and (abs(tail_progress) < progress_eps)
        full_length = n >= stall_window
        label = "DEADLOCK" if (far and frozen and full_length and on_final_leg) else "TIMEOUT"

    return TrialOutcome(
        label=label,
        reached_final=reached_final,
        steps_to_final=steps_to_final,
        collided=collided,
        worst_min_dist_env=float(np.min(min_dists)),
        peak_slack=float(np.max(slacks)) if slacks.size else 0.0,
        mean_slack=float(np.mean(slacks)) if slacks.size else 0.0,
        n_steps=n,
        final_dist=final_dist,
        tail_progress=tail_progress,
        tail_std=tail_std,
        schedule=schedule,
    )
