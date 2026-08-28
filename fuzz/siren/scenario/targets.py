"""
Fuzz targets: a verified attack scene, packaged so the fuzzer can search it.

A verified control says "from G0, inserting G1' breaks the task". To hand that to
the fuzzer we cannot just record G0's POSITION, because the fuzzer would then
start the robot at its home pose and search a different problem. What the attack
depends on is the STATE the arm is in when it departs G0 — joint positions and
velocities both. The whole insertion mechanism is about departing the handover
in motion; a target restored to the right position but zero velocity is a
different system.

So a target stores the full simulator state captured the moment G0 was reached:

    qpos, qvel, act, ctrl          MuJoCo
    dof_{pos,vel,acc}_cmd/fbk      the agent's held command state
    obstacles, G1, G1'_truth       the scene, and the answer

`apply_target` then patches Harness.reset to restore that state after every
reset, exactly as apply_scenario does for layouts — because BenchmarkTask.reset
re-samples the world and would otherwise erase it.

HOW G0 IS REPRODUCED — and why not by restoring the state.

Restoring (qpos, qvel, command state, and even MuJoCo's qacc_warmstart) does NOT
reproduce these attacks. Measured on three controls: the restored run misses by
+0.000075 m where the continuous run contacts at -0.00008 m. The handover states
are bit-identical between runs, so the discrepancy is accumulated integration
history, and the contacts are only 50-280 microns deep -- a 0.155 mm divergence
over ~160 steps is enough to erase them.

So G0 is reproduced the only way that is exactly repeatable: as a PREFIX
WAYPOINT. Every run starts from the robot's home pose and the schedules are

    legitimate   [G0, G1]
    attack       [G0, G1', G1]

The simulation regenerates the state at G0 identically each time because it is
the same rollout, not a restored snapshot. `state_at_G0` is still recorded, as
a description of the handover for analysis, but nothing depends on replaying it.

The stored `G1_prime_truth` is the ground truth the search should find, and must
never be given to it.
"""

import json
from pathlib import Path

import numpy as np


def capture_state(harness) -> dict:
    """Everything env.step advances, as plain lists."""
    ag = harness.env.agent
    def L(x):
        return None if x is None else np.asarray(x, dtype=float).tolist()
    # qacc_warmstart is the solver's initial guess and it PERSISTS between
    # steps. Restoring position and velocity but not the warm start left a
    # 0.15 mm trajectory divergence over ~160 steps — enough to turn a -0.08 mm
    # contact into a +0.08 mm miss, i.e. to lose the attack entirely. Identical
    # (qpos, qvel) with a different warm start is NOT the same system.
    return {
        "qpos": L(ag.data.qpos), "qvel": L(ag.data.qvel),
        "act": L(ag.data.act) if ag.data.act.size else None,
        "ctrl": L(ag.data.ctrl), "time": float(ag.data.time),
        "qacc_warmstart": L(ag.data.qacc_warmstart),
        "qacc": L(ag.data.qacc),
        "qfrc_applied": L(ag.data.qfrc_applied),
        "xfrc_applied": L(ag.data.xfrc_applied),
        "dof_pos_cmd": L(ag.dof_pos_cmd), "dof_vel_cmd": L(ag.dof_vel_cmd),
        "dof_acc_cmd": L(ag.dof_acc_cmd),
        "dof_pos_fbk": L(ag.dof_pos_fbk), "dof_vel_fbk": L(ag.dof_vel_fbk),
    }


def restore_state(harness, st: dict):
    import mujoco
    ag = harness.env.agent
    ag.data.qpos[:] = np.asarray(st["qpos"], float)
    ag.data.qvel[:] = np.asarray(st["qvel"], float)
    if st.get("act") is not None and ag.data.act.size:
        ag.data.act[:] = np.asarray(st["act"], float)
    ag.data.ctrl[:] = np.asarray(st["ctrl"], float)
    ag.data.time = float(st["time"])
    for key in ("qacc_warmstart", "qacc", "qfrc_applied", "xfrc_applied"):
        if st.get(key) is not None:
            getattr(ag.data, key)[:] = np.asarray(st[key], float).reshape(
                getattr(ag.data, key).shape)
    for key, attr in (("dof_pos_cmd", "dof_pos_cmd"), ("dof_vel_cmd", "dof_vel_cmd"),
                      ("dof_acc_cmd", "dof_acc_cmd"), ("dof_pos_fbk", "dof_pos_fbk"),
                      ("dof_vel_fbk", "dof_vel_fbk")):
        if st.get(key) is not None and getattr(ag, attr) is not None:
            getattr(ag, attr)[:] = np.asarray(st[key], float)
    mujoco.mj_forward(ag.model, ag.data)


def apply_target(harness, target: dict):
    """Make every reset land in the target's captured state."""
    original_reset = harness.reset

    def patched_reset(*a, **kw):
        agent_feedback, _stale = original_reset(*a, **kw)
        restore_state(harness, target["state_at_G0"])
        # the goal must survive the restore: BenchmarkTask re-samples it on reset
        harness.env.task.robot_goal_right.frame[:3, 3] = np.asarray(
            target["G1"], dtype=float)
        harness.env.task.robot_goal_right.velocity = 0.0
        # rebuild the info dict AFTER the state and goal are in place, or every
        # downstream reader sees the pre-restore world (see scenes.py)
        agent_feedback = (harness.env.agent.get_feedback()
                          if hasattr(harness.env.agent, "get_feedback")
                          else agent_feedback)
        task_info = harness.env.task.get_info(agent_feedback)
        return agent_feedback, task_info

    harness.reset = patched_reset
    harness._target = target
    return harness


def load_target(path, max_steps=None):
    """Build a world already sitting at the target's G0 state.

    Returns (world, target). `target["G1_prime_truth"]` is the known attack and
    must not be shown to the search — it is the answer key.
    """
    from ..world.run import World
    from ..world.types import real_filter

    target = json.loads(Path(path).read_text())
    spec = real_filter(algo=target["algo"], index=target["index"],
                       d_min=target["d_min"], eta=target["eta"],
                       lam=target["lam"], k=target["k"])
    steps = max_steps or target["max_steps"]
    w = World.build(seed=target["seed"], spec=spec,
                    test_case=target["case"], max_steps=steps)
    # deliberately NOT apply_target(): G0 is replayed as a prefix waypoint, so
    # the world is the stock scene and every schedule begins from home.

    # ...EXCEPT when the target carries its own obstacles. Some attacks were
    # found in a scene the search built rather than the one the seed generates
    # -- the constructed searches park unused obstacles metres away and place
    # one deliberately. Rebuilding the stock scene for such a target searches a
    # DIFFERENT world than the attack lives in: it does not error, it just never
    # finds anything, which is a false negative and exactly the ambiguity a
    # target is supposed to remove.
    #
    # Pinning has to survive reset(), because every rollout resets first, so the
    # harness reset is wrapped rather than the obstacles being set once.
    obs = target.get("obstacles_world")
    if obs:
        import numpy as _np
        from fuzz.siren.world.big_obstacle import set_obstacles as _set_obs
        _pos = [_np.asarray(q, float) for q in obs]
        _rad = float(target.get("obstacle_radius") or 0.05)
        _h = w.harness
        _orig = _h.reset

        def _pinned_reset(*aa, **kk):
            af, _ = _orig(*aa, **kk)
            _set_obs(w, _pos, _rad)
            return af, _h.env.task.get_info(af)

        _h.reset = _pinned_reset
        target = dict(target, _obstacles_pinned=True)
    return w, target


def list_targets(folder="fuzz/siren/scenario/targets"):
    return sorted(str(p) for p in Path(folder).glob("*.json"))
