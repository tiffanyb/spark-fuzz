"""
SingleArmGoalInsertionTask — a BenchmarkTask variant that drives the RIGHT-arm
tracking goal along a scripted waypoint schedule [G1', ..., G1] instead of the
default Brownian/Velocity motion.

Single-arm ATTACK (use_dual_arm=True, but only the right goal is adversarial):
both wrist goals stay enabled because the G1FixedBase IK is a dual-wrist solver,
but this task overrides _update_robot_goal and never moves the LEFT goal, so the
left wrist holds its initial reachable target while only the right wrist is
steered along the schedule. The task self-advances once the end-effector reaches
each waypoint, and exposes the quantities the harness needs (distance to
current/final goal, reached flag, waypoint index).

Integration note: at import time this module registers the class into the
`spark_task` namespace so SPARK's `initialize_class` can resolve
`class_name="SingleArmGoalInsertionTask"` WITHOUT editing any core SPARK file.
"""

import numpy as np
import spark_task
from spark_task import BenchmarkTask


def _xyz_to_frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


class SingleArmGoalInsertionTask(BenchmarkTask):

    def __init__(self, robot_cfg, robot_kinematics, agent, **kwargs):
        super().__init__(robot_cfg, robot_kinematics, agent, **kwargs)

        # Scripted schedule of right-arm waypoints (base frame). Last entry is the
        # legitimate goal G1; any earlier entries are inserted goals (G1', ...).
        self.goal_schedule = None
        self.wp_idx = 0

        # Thresholds.
        self.reach_eps = kwargs.get("reach_eps", 0.05)                 # advance between waypoints
        self.final_reach_eps = kwargs.get("final_reach_eps", self.arm_goal_size)  # count G1 as reached

        # Telemetry consumed by the harness.
        self.reached_final = False
        self.dist_to_current = np.inf
        self.dist_to_final = np.inf
        self.steps_in_phase = 0

    # ------------------------------------------------------------------ #
    #  Schedule control (called by the harness after reset)
    # ------------------------------------------------------------------ #
    def set_goal_schedule(self, waypoints_base_xyz):
        """waypoints_base_xyz: list of 3-D positions in the robot base frame.
        The final entry is the legitimate goal G1."""
        self.goal_schedule = [np.asarray(w, dtype=float).reshape(3) for w in waypoints_base_xyz]
        self.wp_idx = 0
        self.reached_final = False
        self.dist_to_current = np.inf
        self.dist_to_final = np.inf
        self.steps_in_phase = 0

    def _current_goal_base(self):
        if self.goal_schedule is None:
            # No schedule set -> fall back to the benchmark's sampled right goal.
            return self.robot_goal_right.frame[:3, 3].copy()
        return self.goal_schedule[self.wp_idx]

    # ------------------------------------------------------------------ #
    #  Override goal motion: follow the schedule instead of Brownian walk.
    #  (called from BenchmarkTask.step() after _update_robot_state())
    # ------------------------------------------------------------------ #
    def _update_robot_goal(self):
        goal_xyz = self._current_goal_base()
        self.robot_goal_right.frame[:3, 3] = goal_xyz

        # End-effector (world) — robot_frames_world was refreshed in step().
        ee_world = self.robot_frames_world[self.robot_cfg.Frames.R_ee, :3, 3]

        cur_world = (self.robot_base_frame @ _xyz_to_frame(goal_xyz))[:3, 3]
        self.dist_to_current = float(np.linalg.norm(ee_world - cur_world))

        final_xyz = self.goal_schedule[-1] if self.goal_schedule is not None else goal_xyz
        final_world = (self.robot_base_frame @ _xyz_to_frame(final_xyz))[:3, 3]
        self.dist_to_final = float(np.linalg.norm(ee_world - final_world))

        # Advance to the next waypoint once the current one is reached.
        if (self.goal_schedule is not None
                and self.wp_idx < len(self.goal_schedule) - 1
                and self.dist_to_current < self.reach_eps):
            self.wp_idx += 1
            self.steps_in_phase = 0
        else:
            self.steps_in_phase += 1

        if self.dist_to_final < self.final_reach_eps:
            self.reached_final = True

    # Left/base intentionally untouched in the single-arm experiment. The base is
    # disabled by the FixedBase test case; the left goal IS emitted (use_dual_arm=
    # True, required by the dual-wrist IK) but stays frozen at its initial sampled,
    # reachable pose because this override never calls robot_goal_left.move().


# Register into spark_task so initialize_class("SingleArmGoalInsertionTask") resolves
# it, without modifying spark_task/__init__.py.
spark_task.SingleArmGoalInsertionTask = SingleArmGoalInsertionTask
