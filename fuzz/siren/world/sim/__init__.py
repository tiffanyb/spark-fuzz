"""
sim/ — the SPARK adapter. The ONLY place in SIREN that imports spark_* or mujoco.

We implement no physics and no safety filters of our own: SPARK provides the
MuJoCo environment, the robot model, the benchmark scenes, the distance
computation, and every safety algorithm. What we add is thin:

    config.py   build SPARK's pipeline config, pin the scene, apply a FilterSpec
    harness.py  hold the env + algo and expose step/reset primitives
    task.py     a Task that follows a waypoint schedule instead of wandering
    probe.py    read-only instrumentation: pull (phi, L_g phi, L_f phi, u_lim)
                out of the live safety index and count QP give-ups
"""

from .config import build_config, apply_filter_spec
from .harness import Harness
from .probe import install_probe, read_raw

__all__ = ["build_config", "apply_filter_spec", "Harness", "install_probe", "read_raw"]
