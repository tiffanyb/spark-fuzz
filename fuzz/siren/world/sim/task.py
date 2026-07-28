"""
The schedule-driven Task.

SPARK's BenchmarkTask wanders its goals (Brownian / constant velocity). For a
goal-insertion or goal-modification attack we need the goal to follow a scripted
list of waypoints instead: [G1', G1] for insertion, [G1'] for modification.

The class itself already exists in fuzz/goal_insertion_task.py and registers
itself into the spark_task namespace at import time, so importing this module is
enough for SPARK's initialize_class("SingleArmGoalInsertionTask") to resolve.
Re-importing it here (rather than copying) keeps a single registration and a
single definition.
"""

from fuzz.goal_insertion_task import SingleArmGoalInsertionTask  # noqa: F401  (registers)

__all__ = ["SingleArmGoalInsertionTask"]
