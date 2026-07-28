"""
world/ — ground truth. What actually happens when the robot pursues a goal.

    types.py    Scene, FilterSpec           (the inputs)
    measure.py  StepMeasurement, RunRecord  (the output)
    derived.py  pure math: authority C and margin g
    run.py      THE LOOP — orchestrates sim + derived + measure
    sim/        the SPARK adapter (the only place that imports spark_* / mujoco)
"""

from .types import Scene, FilterSpec
from .measure import StepMeasurement, RunRecord, classify_run

__all__ = ["Scene", "FilterSpec", "StepMeasurement", "RunRecord", "classify_run"]
