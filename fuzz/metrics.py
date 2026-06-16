"""
Per-step records and trial-outcome classification.

This is the metric layer the SSA-paper evaluation lacks (RESEARCH_PLAN_0611.md
§15.2): their metrics (J tracking, S collision, R_Feas feasibility) never detect
a *goal-trap* — a run that stays safe but never reaches G1. We add exactly that.

Outcome labels:
  REACHED   - the end-effector reached the final goal G1.
  COLLISION - the robot penetrated an obstacle (min robot-obstacle dist < margin).
  DEADLOCK  - never reached G1, never collided, and the EE stalled (the trap).
  TIMEOUT   - never reached G1, never collided, but still moving at the cutoff.
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
    schedule: List = field(default_factory=list)   # waypoints used (base frame xyz)

    def is_attack_success(self) -> bool:
        """An inserted goal 'succeeds' as an attack if it prevents a safe reach."""
        return self.label in ("DEADLOCK", "COLLISION", "TIMEOUT")


def classify_trial(records: List[StepRecord], schedule=None,
                   collision_margin: float = 0.0,
                   stall_window: int = 30, stall_eps: float = 5e-3) -> TrialOutcome:
    """Reduce per-step records to a single labelled outcome."""
    n = len(records)
    schedule = list(schedule) if schedule is not None else []

    if n == 0:
        return TrialOutcome("TIMEOUT", False, None, False, np.inf, 0.0, 0.0, 0, schedule)

    slacks = np.array([r.peak_slack for r in records], dtype=float)
    min_dists = np.array([r.min_dist_env for r in records], dtype=float)

    collided = bool(np.any(min_dists < -abs(collision_margin)))

    steps_to_final = None
    for r in records:
        if r.reached_final:
            steps_to_final = r.step
            break
    reached_final = steps_to_final is not None

    # Stall detection over the trailing window (EE distance-to-goal barely changes).
    tail = np.array([r.dist_final for r in records[-stall_window:]], dtype=float)
    stalled = tail.size >= 2 and float(np.std(tail)) < stall_eps

    if collided:
        label = "COLLISION"
    elif reached_final:
        label = "REACHED"
    elif stalled:
        label = "DEADLOCK"
    else:
        label = "TIMEOUT"

    return TrialOutcome(
        label=label,
        reached_final=reached_final,
        steps_to_final=steps_to_final,
        collided=collided,
        worst_min_dist_env=float(np.min(min_dists)),
        peak_slack=float(np.max(slacks)) if slacks.size else 0.0,
        mean_slack=float(np.mean(slacks)) if slacks.size else 0.0,
        n_steps=n,
        schedule=schedule,
    )
