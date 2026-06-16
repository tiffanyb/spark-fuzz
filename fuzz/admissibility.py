"""
Admissibility of a candidate inserted goal G1'.

The whole point of the study is that every inserted goal must be a *legitimate*
goal — one the operator could plausibly issue. We define admissibility
operationally (RESEARCH_PLAN_0611.md §15) as:

  1. within the arm-goal workspace bounds (right_arm_goal_range), AND
  2. clear of every obstacle by at least `keepout` (world frame), AND
  3. (optional) inverse-kinematics solvable from the current configuration.

The clearance check mirrors SPARK's own goal-sampling check in
BenchmarkTask._init_goal (benchmark_task.py:232):

    norm(goal_world_center - obstacle_frame_center) < arm_goal_keepout  -> reject

VERIFIED: SPARK measures goal-CENTER to obstacle-CENTER and is geometry-agnostic
(it does NOT subtract the obstacle radius / use the surface). We replicate that
center-to-center convention exactly, so an admissible G1' is indistinguishable
from a goal the benchmark itself would have sampled. (A surface-based clearance
would be physically stricter but would make G1' distinguishable from a real
benchmark goal, which we explicitly do not want.)

QP-feasibility / phi>=0-at-issue is a stronger admissibility notion noted in the
plan; it is left as a hook (`ik_check` here is the kinematic stand-in) and can be
layered on later by evaluating the safety index at the issue state.
"""

import numpy as np


def _within_bounds(xyz, bounds):
    for d in range(3):
        lo, hi = bounds[d]
        if xyz[d] < lo or xyz[d] > hi:
            return False
    return True


def _obstacle_clearance(xyz_base, base_frame, obstacles_world, keepout):
    """Minimum world-frame distance from the goal to any obstacle center."""
    if obstacles_world is None or len(obstacles_world) == 0:
        return np.inf
    goal_world = (base_frame @ np.block([[np.eye(3), xyz_base.reshape(3, 1)],
                                         [np.zeros((1, 3)), np.ones((1, 1))]]))[:3, 3]
    dmin = np.inf
    for ow in obstacles_world:
        d = float(np.linalg.norm(goal_world - np.asarray(ow)[:3, 3]))
        dmin = min(dmin, d)
    return dmin


def is_admissible(xyz_base, bounds, base_frame, obstacles_world, keepout,
                  robot_kinematics=None, current_q=None, ik_check=False):
    """Return (ok: bool, reason: str) for a candidate inserted goal (base frame)."""
    xyz_base = np.asarray(xyz_base, dtype=float).reshape(3)

    if not _within_bounds(xyz_base, bounds):
        return False, "out_of_bounds"

    clearance = _obstacle_clearance(xyz_base, base_frame, obstacles_world, keepout)
    if clearance < keepout:
        return False, f"too_close_to_obstacle({clearance:.3f}<{keepout:.3f})"

    if ik_check and robot_kinematics is not None:
        goal_frame = np.eye(4)
        goal_frame[:3, 3] = xyz_base
        try:
            robot_kinematics.inverse_kinematics([goal_frame], current_q)
        except Exception as e:  # IK failure -> unreachable
            return False, f"ik_unreachable({type(e).__name__})"

    return True, "ok"


def sample_admissible(rng, bounds, base_frame, obstacles_world, keepout,
                      max_tries=1000, robot_kinematics=None, current_q=None,
                      ik_check=False):
    """Rejection-sample one admissible candidate in the workspace box.

    Returns the 3-D position (base frame) or None if no admissible point was
    found within `max_tries`."""
    for _ in range(max_tries):
        xyz = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
        ok, _ = is_admissible(xyz, bounds, base_frame, obstacles_world, keepout,
                              robot_kinematics=robot_kinematics, current_q=current_q,
                              ik_check=ik_check)
        if ok:
            return xyz
    return None
