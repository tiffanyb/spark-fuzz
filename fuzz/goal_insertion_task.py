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

        # BASE-goal channel. On whole-body (WG) cases SPARK commands a base pose
        # as well as an arm pose -- base_goal_range spans 1.6 x 1.6 m plus full
        # yaw, against the arm goal's 0.3 m cube -- so an attacker with access to
        # the goal interface has a second, much larger channel. When a base
        # schedule is set, this task steers the base goal along it exactly as it
        # steers the arm goal, and waypoint advance is driven by whichever
        # channel is being attacked (see `channel`).
        self.base_goal_schedule = None
        self.channel = "arm"

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
    def set_base_goal_schedule(self, waypoints_xy_yaw):
        """waypoints_xy_yaw: list of (x, y, yaw) base poses in the WORLD frame.

        z is pinned to base_goal_range's z, which SPARK fixes (0.793) -- the base
        does not translate vertically, so searching it would waste two thirds of
        every grid axis.
        """
        self.base_goal_schedule = [np.asarray(w, dtype=float).reshape(3)
                                   for w in waypoints_xy_yaw]
        self.channel = "base"
        self.wp_idx = 0
        self.reached_final = False
        self.dist_to_current = np.inf
        self.dist_to_final = np.inf
        self.steps_in_phase = 0

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
    def _update_base_goal(self):
        """Steer the BASE goal along its schedule and measure progress on it.

        Distance mixes translation and rotation, so yaw error is folded in as
        arc length at a 0.5 m lever -- a 90 deg heading error then counts about
        the same as 0.8 m of translation. Without that, a waypoint could count as
        reached while the robot faces the opposite way, which is exactly the
        state a base-goal attack would exploit.
        """
        import numpy as _np
        from scipy.spatial.transform import Rotation as _R

        wp = self.base_goal_schedule[self.wp_idx]
        z = float(self.robot_goal_base.frame[2, 3])
        self.robot_goal_base.frame[:3, 3] = _np.array([wp[0], wp[1], z])
        self.robot_goal_base.frame[:3, :3] = _R.from_euler(
            "xyz", [0, 0, float(wp[2])]).as_matrix()

        b = self.robot_base_frame
        pos = _np.asarray(b[:3, 3], float)
        yaw = float(_R.from_matrix(_np.asarray(b[:3, :3], float)).as_euler("xyz")[2])

        def _d(w):
            dxy = float(_np.linalg.norm(pos[:2] - _np.asarray(w[:2], float)))
            dyaw = abs((yaw - float(w[2]) + _np.pi) % (2 * _np.pi) - _np.pi)
            return dxy + 0.5 * dyaw

        self.dist_to_current = _d(wp)
        self.dist_to_final = _d(self.base_goal_schedule[-1])

        if (self.wp_idx < len(self.base_goal_schedule) - 1
                and self.dist_to_current < self.reach_eps):
            self.wp_idx += 1
            self.steps_in_phase = 0
        else:
            self.steps_in_phase += 1
        if self.dist_to_final < self.final_reach_eps:
            self.reached_final = True

    def _update_robot_goal(self):
        if self.channel == "base" and self.base_goal_schedule is not None:
            # keep the arm goal pinned where the benchmark sampled it: the ATTACK
            # is on the base channel, so the arm must behave legitimately
            self._update_base_goal()
            return
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
