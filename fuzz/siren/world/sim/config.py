"""
Build SPARK's pipeline config, and apply a FilterSpec to a live safety filter.

Everything here is SPARK's own configuration machinery; we only override the few
fields the experiment needs:
  * our schedule-driven Task instead of the wandering benchmark goal,
  * a FIXED scene (seed_list=[seed]) so the baseline and every candidate share
    the same obstacles, start and legitimate goal — without that invariance a
    "sequence-induced" failure could just be a different world,
  * headless + kinematic (deterministic and fast),
  * no auto-termination on reach (the run loop decides when to stop).

apply_filter_spec() is the operation that makes the black-box ensemble possible:
it retunes the deployed filter in place, so the SAME candidate goal can be rolled
out under several assumed defenses.
"""

import numpy as np

# importing these registers the custom task / controller into SPARK's namespaces
from . import task                      # noqa: F401  -> spark_task
from fuzz import projected_ssa          # noqa: F401  -> spark_policy

from ..types import FilterSpec


# 20-length control weights: waist 3 + left arm 7 + right arm 7 + base 3.
# The FixedBase robot uses the first 17 (drops the base), matching SPARK's example.
_FULL_CONTROL_WEIGHT = [1.0] * 20

# SPARK ships both index orders. This is the distance-vs-velocity distinction
# that the black-box speed sweep is designed to identify:
#   first order   phi = d_min - d            engages at a FIXED clearance
#   second order  phi = ... - k * d_dot      engages EARLIER when closing fast
_INDEX_CLASS = {
    "distance": "FirstOrderCollisionSafetyIndex",
    "velocity": "SecondOrderCollisionSafetyIndex",
}


def index_required_by(robot_class_name: str) -> str:
    """Which safety index this robot MUST use.

    The choice is not free: SPARK asserts that the first-order index runs only on
    a Dynamic1 (velocity-controlled) robot, and the second-order index only on a
    Dynamic2 (acceleration-controlled) one. That is the relative-degree argument
    showing up as a runtime check -- with velocity control the clearance responds
    to the command immediately, so distance alone suffices; with force control it
    does not, so the index must also count how fast clearance is closing.

    Consequence for the threat model: the index family is pinned by the robot's
    CONTROL MODE, which is a published specification. An attacker who knows the
    robot is velocity-controlled already knows the index is distance-based, and
    gets weak-black-box knowledge for free. The speed sweep earns its keep when
    the control mode is not known.
    """
    return "velocity" if "Dynamic2" in robot_class_name else "distance"


def apply_filter_spec(cfg, spec: FilterSpec):
    """Write a FilterSpec into a (not yet instantiated) SPARK config."""
    sa = cfg.algo.safe_controller.safe_algo
    sa.class_name = spec.class_name

    if spec.demand_shape == "constant":
        sa.eta_ssa = float(spec.eta if spec.eta is not None else 0.5)
    else:
        lam = float(spec.lam if spec.lam is not None else 10.0)
        if spec.algo in ("cbf", "rcbf"):
            sa.lambda_cbf = lam
        else:
            sa.lambda_sss = lam

    if spec.slack_weight is not None:
        sa.slack_weight = float(spec.slack_weight)
    elif spec.algo in ("rssa", "rcbf", "rsss"):
        sa.slack_weight = 1e3
    elif spec.algo == "pssa":
        sa.slack_weight = 1.0

    weight = list(_FULL_CONTROL_WEIGHT)
    if "FixedBase" in cfg.robot.cfg.class_name:
        weight = weight[:-3]
    elif "RightArm" in cfg.robot.cfg.class_name:
        weight = weight[3:10]
    sa.control_weight = weight

    # The robot's control mode decides the index; a spec asking for the other one
    # cannot be honoured (SPARK asserts on it), so we use the required one.
    required = index_required_by(cfg.robot.cfg.class_name)
    si = cfg.algo.safe_controller.safety_index
    si.class_name = _INDEX_CLASS[required]
    si.min_distance["environment"] = float(spec.d_min)
    if required == "velocity":
        si.phi_n = 1                                   # linear form
        si.phi_k = float(spec.k if spec.k is not None else 0.1)
    return cfg


def retune_live(harness, spec: FilterSpec):
    """Retune an ALREADY-BUILT harness in place — the ensemble path.

    Only the demand parameter is changed at runtime; the index class is fixed at
    construction, so a spec whose index differs from the harness's is rolled out
    under the harness's index (see Harness.supports_index).
    """
    algo = harness.algo.safe_controller.safe_algo
    if spec.demand_shape == "constant":
        if hasattr(algo, "eta_default"):
            algo.eta_default = float(spec.eta if spec.eta is not None else 0.5)
        if hasattr(algo, "eta_ssa"):
            algo.eta_ssa = float(spec.eta if spec.eta is not None else 0.5)
    else:
        lam = float(spec.lam if spec.lam is not None else 10.0)
        if hasattr(algo, "lambda_default"):
            algo.lambda_default = lam
    return harness


def build_config(seed: int = 0,
                 spec: FilterSpec = None,
                 test_case: str = "G1FixedBase_D1_AG_SO_v0",
                 max_steps: int = 400,
                 reach_eps: float = 0.05,
                 use_sim_dynamics: bool = False,
                 enable_viewer: bool = False):
    """A SPARK G1 benchmark config wired for goal attacks."""
    from spark_pipeline import G1BenchmarkPipelineConfig, generate_benchmark_test_case

    spec = spec or FilterSpec()

    cfg = G1BenchmarkPipelineConfig()
    cfg = generate_benchmark_test_case(cfg, test_case)

    # --- task: our scripted waypoint schedule, on a FIXED scene ---------- #
    cfg.env.task.class_name = "SingleArmGoalInsertionTask"
    # Both wrist goals stay enabled (G1FixedBase IK is a dual-wrist solver), but
    # the task never moves the left one — only the right wrist is adversarial.
    cfg.env.task.use_dual_arm = True
    cfg.env.task.arm_goal_reach_done = False
    cfg.env.task.max_episode_length = max_steps
    cfg.env.task.reach_eps = reach_eps
    cfg.env.task.seed = seed
    cfg.env.task.seed_list = [seed]

    # --- agent: headless, kinematic ------------------------------------- #
    cfg.env.agent.enable_viewer = enable_viewer
    cfg.env.agent.use_sim_dynamics = use_sim_dynamics

    apply_filter_spec(cfg, spec)

    cfg.max_num_steps = max_steps
    cfg.max_num_reset = -1
    cfg.enable_logger = False
    cfg.enable_plotter = False
    cfg.enable_safe_zone_render = False
    return cfg
