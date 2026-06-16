"""
SingleArmHarness — runs ONE goal-insertion trial end-to-end and returns a
classified TrialOutcome.

It reuses SPARK's env/algo wrappers exactly as BasePipeline does (instantiate
robot config + kinematics, SparkEnvWrapper, SparkAlgoWrapper), but drives the
loop itself so the fuzzer has step-level control and so the scene stays fixed
across trials.

Design choices:
  - The scene is fixed by seed_list=[seed] in the config, so every env.reset()
    regenerates the identical obstacle layout, start pose, and benchmark goal G1.
  - Goal scheduling lives in the Task (SingleArmGoalInsertionTask); the harness
    only reads the Task's telemetry and computes collision/slack from action_info.
"""

import numpy as np

# Importing this registers SingleArmGoalInsertionTask before SparkEnvWrapper
# resolves the task class_name.
from . import goal_insertion_task  # noqa: F401
from .metrics import StepRecord, classify_trial


def _xyz_to_frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


class SingleArmHarness:

    def __init__(self, cfg):
        from spark_utils import initialize_class
        from spark_env import SparkEnvWrapper
        from spark_algo import SparkAlgoWrapper

        self.cfg = cfg
        self.robot_cfg = initialize_class(cfg.robot.cfg)
        cfg.robot.kinematics.class_name = self.robot_cfg.kinematics_class_name
        self.robot_kinematics = initialize_class(cfg.robot.kinematics, robot_cfg=self.robot_cfg)

        self.env = SparkEnvWrapper(cfg.env, robot_cfg=self.robot_cfg,
                                   robot_kinematics=self.robot_kinematics)
        self.algo = SparkAlgoWrapper(cfg.algo, robot_cfg=self.robot_cfg,
                                     robot_kinematics=self.robot_kinematics)

        self.R_ee = self.robot_cfg.Frames.R_ee
        self.max_steps = cfg.max_num_steps

    # ------------------------------------------------------------------ #
    def _reset_scene(self):
        agent_feedback, task_info = self.env.reset()
        # Warm-up acts: BenchmarkPipeline does this to settle the seeded init.
        for _ in range(10):
            u_safe, action_info = self.algo.act(agent_feedback, task_info)
        return agent_feedback, task_info

    def scene_info(self):
        """Reset once and report the fixed scene: start, benchmark goal G1,
        obstacles (world), workspace bounds, and the goal keepout."""
        agent_feedback, task_info = self._reset_scene()
        base = agent_feedback["robot_base_frame"]

        G1_base = self.env.task.robot_goal_right.frame[:3, 3].copy()
        ee_world = self.env.task.robot_frames_world[self.R_ee, :3, 3].copy()
        G0_base = (np.linalg.inv(base) @ _xyz_to_frame(ee_world))[:3, 3]

        obstacles_world = np.array(task_info["obstacle"]["frames_world"]) \
            if len(task_info["obstacle"]["frames_world"]) > 0 else np.zeros((0, 4, 4))

        return {
            "G0_base": G0_base,
            "G1_base": G1_base,
            "base_frame": base,
            "obstacles_world": obstacles_world,
            "bounds": self.env.task.right_arm_goal_range,
            "keepout": self.env.task.arm_goal_keepout,
        }

    # ------------------------------------------------------------------ #
    def run_trial(self, goal_schedule, max_steps=None, stop_on_terminal=True):
        """Run one trial with the given right-arm waypoint schedule (base frame).

        goal_schedule: list of 3-D positions; the last is the legitimate goal G1.
        Returns a TrialOutcome.
        """
        from spark_utils import compute_masked_distance_matrix

        max_steps = max_steps if max_steps is not None else self.max_steps

        agent_feedback, task_info = self._reset_scene()
        self.env.task.set_goal_schedule(goal_schedule)

        u_safe, action_info = self.algo.act(agent_feedback, task_info)

        records = []
        for step in range(max_steps):
            agent_feedback, task_info = self.env.step(u_safe, action_info)
            u_safe, action_info = self.algo.act(agent_feedback, task_info)

            task = self.env.task

            # Collision: recompute min robot-obstacle distance (mirrors the pipeline).
            obs_frames = task_info["obstacle"]["frames_world"]
            obs_geom = task_info["obstacle"]["geom"]
            if len(obs_frames) > 0:
                dmat, _ = compute_masked_distance_matrix(
                    frame_list_1=task.robot_frames_world,
                    geom_list_1=self.robot_cfg.CollisionVol.values(),
                    frame_list_2=obs_frames,
                    geom_list_2=obs_geom,
                )
                min_dist_env = float(dmat.min()) if dmat is not None else np.inf
            else:
                min_dist_env = np.inf

            # Slack: relaxed controllers report per-constraint slack in "violation".
            viol = action_info.get("violation", None)
            peak_slack = float(np.max(viol)) if (viol is not None and np.size(viol) > 0) else 0.0

            rec = StepRecord(
                step=step,
                wp_idx=task.wp_idx,
                dist_final=task.dist_to_final,
                reached_final=task.reached_final,
                min_dist_env=min_dist_env,
                peak_slack=peak_slack,
                trigger_safe=bool(action_info.get("trigger_safe", False)),
                collided=(min_dist_env < 0.0),
            )
            records.append(rec)

            if stop_on_terminal and (task.reached_final or rec.collided):
                break

        return classify_trial(records, schedule=goal_schedule)
