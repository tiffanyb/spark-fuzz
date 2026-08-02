"""
Hand-built scenes — the positive control.

Everything measured so far says the attack surface is small or gone. That result
is only meaningful if the search would have FOUND an attack had one existed, and
nothing in this project has ever established that. "No attacks exist" and "our
fuzzer cannot find attacks" produce identical evidence.

A scene where an attack is known to exist by construction separates them:

    SIREN finds it      the search works; scarcity elsewhere is a property of
                        the system, not of the tool
    SIREN misses it     the fuzzer is the problem, and every negative result so
                        far is uninterpretable

The scenes here are legitimate in exactly the sense SPARK's own generator is:
obstacles do not overlap each other or the robot's start pose, and the
legitimate goal G1 respects the same admissibility rule the attacker must obey
(inside the workspace box, at least `arm_goal_keepout` from every obstacle
centre). The only thing hand-chosen is the LAYOUT — which is what a scenario is.

The robot's start pose G0 is NOT chosen: it is whatever the robot's home
configuration gives, identical to every benchmark scene.
"""

from .scenes import (Scenario, SCENARIOS, apply_scenario, build_scenario_world)

__all__ = ["Scenario", "SCENARIOS", "apply_scenario", "build_scenario_world"]
