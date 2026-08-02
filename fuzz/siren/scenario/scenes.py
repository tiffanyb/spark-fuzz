"""
Hand-built scenes, and the machinery to impose them on a SPARK world.

`BenchmarkTask.reset()` calls `_init_obstacle()` and `_init_goal()` every time,
so a scene written once is erased on the next reset. `apply_scenario` therefore
wraps `Harness.reset` and re-imposes the layout after every reset — that is the
only way to hold a hand-built scene fixed across the hundreds of rollouts a
search performs.

Obstacle positions are given in the ROBOT BASE frame for legibility (the same
frame the goals live in) and converted to world on application, because SPARK
stores obstacle frames in world coordinates. Getting that backwards silently
places the obstacles somewhere else entirely — it is the same trap that made an
earlier plot draw obstacles in the wrong position.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Scenario:
    """One hand-built layout. Positions in the robot BASE frame, metres."""
    name: str
    obstacles: List[List[float]]          # obstacle centres, base frame
    G1: List[float]                       # the legitimate goal
    base_case: str = "G1FixedBase_D2_AG_SO_v0"   # supplies robot + control mode
    note: str = ""

    def n_obstacles(self) -> int:
        return len(self.obstacles)


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


def check_legitimacy(scn: Scenario, keepout: float, obstacle_radius: float,
                     G0: np.ndarray, bounds) -> list:
    """Every way a hand-built scene could be cheating. Returns a list of faults.

    A scene that fails any of these is not a fair test: the attacker is required
    to place goals inside the workspace box and `keepout` from obstacle centres,
    so a scene whose own G1 violates that is asking the filter to do something
    the benchmark would never ask.
    """
    faults = []
    obs = np.asarray(scn.obstacles, dtype=float)
    G1 = np.asarray(scn.G1, dtype=float)

    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            d = float(np.linalg.norm(obs[i] - obs[j]))
            if d < 2 * obstacle_radius:
                faults.append(f"obstacles {i},{j} overlap (centres {d:.3f} m "
                              f"< {2*obstacle_radius:.3f})")
    for i, o in enumerate(obs):
        if np.linalg.norm(G1 - o) < keepout:
            faults.append(f"G1 is {np.linalg.norm(G1-o):.3f} m from obstacle {i} "
                          f"— closer than the keepout {keepout:.3f} the attacker "
                          f"must respect")
        if np.linalg.norm(np.asarray(G0) - o) < obstacle_radius:
            faults.append(f"obstacle {i} overlaps the robot's start pose")
    for d in range(3):
        lo, hi = bounds[d]
        if not (lo <= G1[d] <= hi):
            faults.append(f"G1 axis {d} = {G1[d]:.3f} outside the workspace box "
                          f"[{lo:.3f}, {hi:.3f}]")
    return faults


def apply_scenario(harness, scn: Scenario):
    """Impose `scn` on a built world, and keep imposing it after every reset."""
    task = harness.env.task
    base = None

    def _impose():
        nonlocal base
        if base is None:
            base = np.asarray(task.robot_base_frame, dtype=float)
        n = min(len(scn.obstacles), len(task.obstacle_task))
        for i in range(n):
            world = (base @ _frame(scn.obstacles[i]))[:3, 3]
            task.obstacle_task[i].frame[:3, 3] = world
            # freeze it: a hand-built layout must not drift between rollouts
            task.obstacle_task[i].velocity = 0.0
            task.obstacle_task[i].last_direction = np.zeros(3)
        # park any surplus obstacles far away rather than leaving them where the
        # sampler put them, which would silently add unplanned geometry
        for i in range(n, len(task.obstacle_task)):
            task.obstacle_task[i].frame[:3, 3] = np.array([0.0, 0.0, -10.0])
            task.obstacle_task[i].velocity = 0.0
        # NOT converted: harness.scene() reads robot_goal_right.frame directly as
        # a BASE-frame position, while obstacle frames above are WORLD. The two
        # are stored in different frames and converting both breaks the goal.
        task.robot_goal_right.frame[:3, 3] = np.asarray(scn.G1, dtype=float)
        task.robot_goal_right.velocity = 0.0

    original_reset = harness.reset

    def patched_reset(*a, **kw):
        agent_feedback, _stale = original_reset(*a, **kw)
        _impose()
        # REBUILD the info dict. BenchmarkTask.reset() assembles `self.info`
        # from the sampled world, so a layout imposed afterwards lands in
        # task.obstacle_task but never reaches task_info -- which is the dict
        # the filter, the collision check and every measurement actually read.
        # The first version of this called a `_update_info` method that does not
        # exist, inside a bare `except: pass`, so the whole scenario framework
        # was silently inert: every scene measured the stock benchmark obstacles
        # while the hand-built layout sat unused in a parallel structure. The
        # tell was three different layouts reporting byte-identical clearance.
        #
        # get_info() is the same builder SparkEnvWrapper.reset uses, so calling
        # it here reproduces exactly what a normal reset would have produced had
        # the obstacles been in these positions all along.
        task_info = harness.env.task.get_info(agent_feedback)
        return agent_feedback, task_info

    harness.reset = patched_reset
    harness._scenario = scn
    return harness


def build_scenario_world(scn: Scenario, spec, max_steps=900):
    """Build a world and impose the scenario on it."""
    from ..world.run import World
    w = World.build(seed=0, spec=spec, test_case=scn.base_case,
                    max_steps=max_steps)
    apply_scenario(w.harness, scn)
    w._scene = None                     # force a re-read through the patched reset
    return w


# --------------------------------------------------------------------------- #
#  The scenes
# --------------------------------------------------------------------------- #
#  G0 (the robot's home pose, not chosen) sits at roughly
#  [0.25, -0.24, 0.12] in the base frame, and the workspace box is
#  x [0.1, 0.4], y [-0.4, -0.1], z [0.0, 0.3].
#
#  Design intent of each layout is stated in `note`, and each is a HYPOTHESIS
#  about what makes an insertion attack work — brute force decides which, if
#  any, actually admits one.

SCENARIOS = {

    "wall_gap": Scenario(
        name="wall_gap",
        base_case="G1FixedBase_D2_AG_SO_v1",      # 10 obstacle slots
        # A wall at x = 0.18 with a single gap at (y=-0.24, z=0.18). G0 already
        # sits PAST the wall (x = 0.25), so the direct route to G1 never meets
        # it. An inserted goal in front of the wall forces the return leg to
        # thread the gap -- geometry the unattacked task never encounters.
        obstacles=[[0.18, -0.13, 0.06], [0.18, -0.24, 0.06], [0.18, -0.35, 0.06],
                   [0.18, -0.13, 0.18],                       [0.18, -0.35, 0.18],
                   [0.18, -0.13, 0.30], [0.18, -0.24, 0.30], [0.18, -0.35, 0.30]],
        G1=[0.36, -0.24, 0.18],
        note="wall with one gap: the direct route is clear because G0 starts "
             "past the wall; an inserted goal in front of it forces a threading "
             "manoeuvre through a 0.06 m free radius",
    ),

    "blind_corner": Scenario(
        name="blind_corner",
        base_case="G1FixedBase_D2_AG_SO_v0",
        # G1 sits just past a single obstacle. Approaching from G0 is oblique;
        # approaching from the opposite corner is head-on with a long run-up.
        obstacles=[[0.24, -0.15, 0.06], [0.15, -0.36, 0.26]],
        G1=[0.37, -0.13, 0.04],
        note="run-up: an inserted goal in the far corner produces a long "
             "head-on approach to a goal tucked just past an obstacle",
    ),

    "pocket": Scenario(
        name="pocket",
        base_case="G1FixedBase_D2_AG_SO_v1",
        # Three obstacles form a pocket around the low-x end of the box. An
        # inserted goal inside it means the return leg must reverse out.
        obstacles=[[0.13, -0.13, 0.20], [0.13, -0.35, 0.20],
                   [0.24, -0.24, 0.20], [0.13, -0.24, 0.31]],
        G1=[0.37, -0.24, 0.08],
        note="trap: an inserted goal inside the pocket forces the return leg to "
             "back out through its mouth before heading to G1",
    ),
}
