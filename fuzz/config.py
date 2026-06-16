"""
Config builder for the Phase-A single-arm goal-insertion experiment.

Reuses SPARK's own benchmark test-case generator for the robot/scene/safety
defaults, then overrides the few fields the experiment needs:
  - single arm (use_dual_arm = False),
  - our scripted task (class_name = SingleArmGoalInsertionTask),
  - a FIXED scene across all trials (seed_list = [seed]) so baseline and every
    candidate share the same obstacles + start + G1,
  - headless, kinematic, no auto-termination on reach (the harness controls it).

The safety-controller block mirrors example/g1/run_g1_benchmark.py::config_safety_module.
Default controller is r-SSA (RelaxedSafeSetAlgorithm) because its QP exposes a
graded slack signal (action_info["violation"]); SSA/CBF only report slack on
infeasibility.
"""

# importing the task module registers SingleArmGoalInsertionTask into spark_task
from . import goal_insertion_task  # noqa: F401


# 20-length control weights (waist 3 + left arm 7 + right arm 7 + base 3); the
# FixedBase robot uses the first 17 (drop the 3 base weights), matching the example.
_FULL_CONTROL_WEIGHT = [
    1.0, 1.0, 1.0,                      # waist
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm
    1.0, 1.0, 1.0,                      # base / locomotion
]


def _config_safety(cfg, safe_algo: str):
    sa = cfg.algo.safe_controller.safe_algo
    weight = list(_FULL_CONTROL_WEIGHT)

    if safe_algo == "ssa":
        sa.class_name = "BasicSafeSetAlgorithm"
        sa.eta_ssa = 0.1
        sa.control_weight = weight
    elif safe_algo == "rssa":
        sa.class_name = "RelaxedSafeSetAlgorithm"
        sa.eta_ssa = 0.1
        sa.slack_weight = 1e3
        sa.control_weight = weight
    elif safe_algo == "cbf":
        sa.class_name = "BasicControlBarrierFunction"
        sa.lambda_cbf = 10.0
        sa.control_weight = weight
    elif safe_algo == "rcbf":
        sa.class_name = "RelaxedControlBarrierFunction"
        sa.lambda_cbf = 10.0
        sa.slack_weight = 1e3
        sa.control_weight = weight
    elif safe_algo == "sss":
        sa.class_name = "BasicSublevelSafeSetAlgorithm"
        sa.lambda_sss = 10.0
        sa.control_weight = weight
    else:
        raise ValueError(f"unsupported safe_algo: {safe_algo}")

    # FixedBase uses 17 controls -> drop the 3 base weights (mirrors the example).
    if "FixedBase" in cfg.robot.cfg.class_name:
        sa.control_weight = sa.control_weight[:-3]
    elif "RightArm" in cfg.robot.cfg.class_name:
        sa.control_weight = sa.control_weight[3:10]

    cfg.algo.safe_controller.safety_index.class_name = "FirstOrderCollisionSafetyIndex"


def build_single_arm_config(test_case: str = "G1FixedBase_D1_AG_SO_v0",
                            safe_algo: str = "rssa",
                            seed: int = 0,
                            max_steps: int = 400,
                            reach_eps: float = 0.05,
                            use_sim_dynamics: bool = False,
                            enable_viewer: bool = False):
    """Build a G1BenchmarkPipelineConfig wired for single-arm goal insertion."""
    from spark_pipeline import G1BenchmarkPipelineConfig, generate_benchmark_test_case

    cfg = G1BenchmarkPipelineConfig()
    cfg = generate_benchmark_test_case(cfg, test_case)

    # --- task: single-arm ATTACK, scripted goals, fixed scene ---
    cfg.env.task.class_name = "SingleArmGoalInsertionTask"
    # Keep BOTH arm goals enabled. The G1FixedBase IK is a dual-wrist solver
    # (g1_fixed_base_kinematics.py:226 unpacks T[0],T[1]); with use_dual_arm=False
    # the policy passes a single IK target and the solver IndexErrors. We instead
    # FREEZE the left goal (the task overrides _update_robot_goal and never moves
    # the left goal) and fuzz only the right -> a single-arm attack on the intact
    # two-hand robot.
    cfg.env.task.use_dual_arm = True
    cfg.env.task.arm_goal_reach_done = False     # harness controls termination
    cfg.env.task.max_episode_length = max_steps
    cfg.env.task.reach_eps = reach_eps
    cfg.env.task.seed = seed
    cfg.env.task.seed_list = [seed]              # SAME scene every reset

    # --- agent: headless, kinematic (deterministic, fast) ---
    cfg.env.agent.enable_viewer = enable_viewer
    cfg.env.agent.use_sim_dynamics = use_sim_dynamics

    # --- safety controller ---
    _config_safety(cfg, safe_algo)

    # --- pipeline-level ---
    cfg.max_num_steps = max_steps
    cfg.max_num_reset = -1
    cfg.enable_logger = False
    cfg.enable_plotter = False
    cfg.enable_safe_zone_render = False

    return cfg
