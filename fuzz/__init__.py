"""fuzz — SPARK adversarial goal-insertion research.

At import time this registers two classes into SPARK's namespaces so
`spark`'s `initialize_class` can resolve them by name:

    goal_insertion_task -> spark_task.SingleArmGoalInsertionTask
    projected_ssa       -> spark_policy.ProjectedSafeSetAlgorithm

Both are used by the SIREN pipeline under fuzz/siren/ (the task class its worlds
run, and the pssa filter). The former standalone Phase-A fuzzer that also lived
at this level has been retired to fuzz/trash/.
"""

from . import goal_insertion_task  # noqa: F401  (-> spark_task)
from . import projected_ssa        # noqa: F401  (-> spark_policy)
