"""
Harness — holds SPARK's env + algo and exposes step/reset primitives.

This is a thin holder, not a simulator. SPARK's SparkEnvWrapper owns the MuJoCo
model, the robot and the obstacles; SparkAlgoWrapper owns the safety filter. We
drive the loop ourselves (in world/run.py) so the run loop has step-level
control and so the scene stays identical across trials.
"""

import numpy as np

from . import task                      # noqa: F401  registers the schedule Task
from .config import build_config, retune_live
from ..types import Scene, FilterSpec


def _xyz_to_frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


class Harness:
    """One built SPARK world. Construct once, run many schedules against it."""

    def __init__(self, cfg, seed=-1, test_case="", spec: FilterSpec = None):
        from spark_utils import initialize_class
        from spark_env import SparkEnvWrapper
        from spark_algo import SparkAlgoWrapper

        self.cfg = cfg
        self.seed = seed
        self.test_case = test_case
        self.spec = spec or FilterSpec()

        self.robot_cfg = initialize_class(cfg.robot.cfg)
        cfg.robot.kinematics.class_name = self.robot_cfg.kinematics_class_name
        self.robot_kinematics = initialize_class(cfg.robot.kinematics,
                                                 robot_cfg=self.robot_cfg)
        self.env = SparkEnvWrapper(cfg.env, robot_cfg=self.robot_cfg,
                                   robot_kinematics=self.robot_kinematics)
        self.algo = SparkAlgoWrapper(cfg.algo, robot_cfg=self.robot_cfg,
                                     robot_kinematics=self.robot_kinematics)

        self.R_ee = self.robot_cfg.Frames.R_ee
        self.max_steps = cfg.max_num_steps
        self._infeasible = [0]           # filled by the probe

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, seed=0, spec: FilterSpec = None,
              test_case="G1FixedBase_D1_AG_SO_v0", max_steps=400, **kw):
        spec = spec or FilterSpec()
        cfg = build_config(seed=seed, spec=spec, test_case=test_case,
                           max_steps=max_steps, **kw)
        return cls(cfg, seed=seed, test_case=test_case, spec=spec)

    # ------------------------------------------------------------------ #
    @property
    def supports_index(self) -> str:
        """Which safety index this world runs — dictated by its control mode.

        SPARK asserts that the first-order (distance) index requires a Dynamic1
        velocity-controlled robot and the second-order (velocity-augmented) index
        requires a Dynamic2 acceleration-controlled one. So this is a property of
        the robot, not a free choice: see config.index_required_by.
        """
        from .config import index_required_by
        return index_required_by(self.robot_cfg.__class__.__name__)

    def accepts(self, spec: FilterSpec) -> bool:
        """Can this world be retuned to that spec without rebuilding? Only the
        demand can be retuned live; the index is fixed by the robot."""
        return spec.index == self.supports_index

    def retune(self, spec: FilterSpec):
        """Apply a FilterSpec's demand parameters to the live filter."""
        retune_live(self, spec)
        self.spec = spec
        return self

    # ------------------------------------------------------------------ #
    def reset(self, warmup: int = 10):
        """Reset to the fixed scene. The warm-up acts settle the seeded init,
        mirroring what SPARK's own benchmark pipeline does."""
        agent_feedback, task_info = self.env.reset()
        for _ in range(warmup):
            u_safe, action_info = self.algo.act(agent_feedback, task_info)
        return agent_feedback, task_info

    def scene(self) -> Scene:
        """Read the fixed world: start, legitimate goal, obstacles, bounds."""
        agent_feedback, task_info = self.reset()
        base = agent_feedback["robot_base_frame"]

        G1 = self.env.task.robot_goal_right.frame[:3, 3].copy()
        ee_world = self.env.task.robot_frames_world[self.R_ee, :3, 3].copy()
        G0 = (np.linalg.inv(base) @ _xyz_to_frame(ee_world))[:3, 3]

        obs = task_info["obstacle"]["frames_world"]
        obstacles = np.array(obs) if len(obs) > 0 else np.zeros((0, 4, 4))

        return Scene(G0=G0, G1=G1, base_frame=base, obstacles_world=obstacles,
                     bounds=self.env.task.right_arm_goal_range,
                     keepout=self.env.task.arm_goal_keepout,
                     seed=self.seed, test_case=self.test_case)

    # ------------------------------------------------------------------ #
    def clearance(self, task_info, guarded_only: bool = True) -> float:
        """Min robot-obstacle distance, using SPARK's own distance computation.

        BUG-1 FIX. SPARK deliberately excludes some robot volumes from
        ENVIRONMENT collision checking via `env_collision_vol_ignore` -- on the
        G1 those are the three waist joints and the three pelvis links, which sit
        near the base and would otherwise trip constantly. The safety index does
        not watch them, so the filter is not accountable for them.

        Measuring over ALL volumes therefore reports "collisions" the filter was
        never asked to prevent: every collision examined during the Kind-0
        investigation was `pelvis_link_3`, while phi simultaneously read -0.066
        ("safe") because it was describing the guarded pairs. Two numbers meant
        to describe the same event were describing different pairs.

        With guarded_only=True the two views are aligned: this measures exactly
        the pairs the filter monitors. Pass False to see raw geometric contact
        (useful for reporting that the robot touched something at all, but NOT
        for attributing the failure to the filter).
        """
        from spark_utils import compute_masked_distance_matrix

        obs_frames = task_info["obstacle"]["frames_world"]
        obs_geom = task_info["obstacle"]["geom"]
        if len(obs_frames) == 0:
            return np.inf
        dmat, _ = compute_masked_distance_matrix(
            frame_list_1=self.env.task.robot_frames_world,
            geom_list_1=self.robot_cfg.CollisionVol.values(),
            frame_list_2=obs_frames, geom_list_2=obs_geom)
        if dmat is None:
            return np.inf
        dmat = np.asarray(dmat, dtype=float)

        if guarded_only:
            si = self.algo.safe_controller.safe_algo.safety_index
            mask = getattr(si, "env_collision_mask", None)
            if mask is not None and np.shape(mask) == dmat.shape:
                # ignored pairs pushed to +inf so they cannot set the minimum
                dmat = np.where(np.asarray(mask, dtype=bool), dmat, np.inf)
        return float(dmat.min())
