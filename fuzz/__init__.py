"""
Phase-A single-arm adversarial goal-insertion fuzzer for SPARK.

Research question (see ../RESEARCH_PLAN_0611.md §15): can a single *admissible*
intermediate goal G1' inserted before a legitimate goal G1 trap a reactive
safe controller so the robot can no longer reach G1 safely?

This package is intentionally self-contained: it plugs into SPARK by
*registering* a custom Task into the `spark_task` namespace at import time
(see goal_insertion_task.py) rather than editing any core SPARK file.

Public entry point:
    from fuzz import build_single_arm_config, SingleArmHarness, GoalInsertionFuzzer
"""

from .config import build_single_arm_config
from .goal_insertion_task import SingleArmGoalInsertionTask
from .harness import SingleArmHarness
from .metrics import StepRecord, TrialOutcome, classify_trial
from .admissibility import is_admissible, sample_admissible
from .fuzzer import GoalInsertionFuzzer

__all__ = [
    "build_single_arm_config",
    "SingleArmGoalInsertionTask",
    "SingleArmHarness",
    "StepRecord",
    "TrialOutcome",
    "classify_trial",
    "is_admissible",
    "sample_admissible",
    "GoalInsertionFuzzer",
]
