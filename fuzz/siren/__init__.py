"""
SIREN — Safe-control Infeasibility via Reachable-goal Exploitation Nexus.

Automatically finds goal-based attacks against humanoid safety filters: goals
that look legitimate but drive the robot into states where the safety filter has
no safe control left.

Two halves:
    world/   what actually happens  — a thin adapter over SPARK (ground truth)
    search/  what to try, how good was it — the attacker (never touches MuJoCo)

The only thing search/ knows about world/ is two calls:
    world.scene()                     -> Scene
    world.run(schedule, filter_spec)  -> RunRecord
"""

__all__ = ["world", "search"]
