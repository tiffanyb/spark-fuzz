"""
Capture a simulator state and resume a run from it.

WHY THIS EXISTS. The G0 search currently replays `home -> G0` as a prefix
waypoint on every candidate, because an earlier measurement said restoring a
snapshot did not reproduce the attacks. That measurement predates capturing
`qacc_warmstart` and the agent's command arrays; with those included, 4 of 4
controls spanning the depth range (-0.000811 to -0.000018) reproduced from a
restored state. So resuming is viable, and it removes a whole leg of simulation
from every candidate.

WHAT A STATE HAS TO CONTAIN. Two halves, and leaving out either one breaks it:

  robot    qpos, qvel, act, ctrl, time, and -- critically -- qacc_warmstart,
           which is the solver's initial guess and PERSISTS between steps.
           Restoring position and velocity but not the warm start left a 0.15 mm
           divergence over ~160 steps, enough to turn a -0.08 mm contact into a
           +0.08 mm miss. Plus the agent's five dof_*_cmd / dof_*_fbk arrays:
           with use_sim_dynamics=False these ARE the state the filter sees, and
           MuJoCo's qvel stays at zero throughout.

  world    every obstacle's frame, last_direction, last_frame, step_counter and
           its own RandomState, plus the task's RandomState. On dynamic-obstacle
           scenes the obstacles keep walking; restoring the robot alone resumes
           it into a world whose obstacles are at the wrong PHASE. An earlier
           probe was invalidated by exactly this and had to be thrown away.

Static-obstacle scenes do not need the world half, but capturing it costs
nothing and makes the helper safe to use on either.
"""

import numpy as np

from ..scenario.targets import capture_state, restore_state


def capture_world(harness) -> dict:
    """Full resumable state: robot + obstacles + task RNG."""
    task = harness.env.task
    obs = []
    for o in getattr(task, "obstacle_task", []) or []:
        rs = getattr(o, "rs", None)
        st = rs.get_state() if rs is not None else None
        obs.append({
            "frame": np.asarray(o.frame, float).tolist(),
            "last_frame": (np.asarray(o.last_frame, float).tolist()
                           if getattr(o, "last_frame", None) is not None else None),
            "last_direction": np.asarray(o.last_direction, float).tolist(),
            "step_counter": int(o.step_counter),
            # RandomState.get_state() is a tuple whose second entry is a uint32
            # array; keep it as-is and only convert on the way back in.
            "rs_state": (st[0], np.asarray(st[1]).tolist(), st[2], st[3], st[4])
                        if st is not None else None,
        })
    # The whole-body IK is an iterative casadi solve that WARM-STARTS from its
    # own previous solution: g1_dual_arm_kinematics.py sets self.init_data =
    # sol_q after every call, and feeds it to both opti.set_initial() and
    # var_q_last. It lives on the kinematics object, so nothing above captures
    # it -- and a resumed run whose IK starts from a different guess converges
    # slightly differently. Measured on G1MobileBase: steps +0 and +1 matched
    # exactly, then dof_pos_cmd diverged from step +2 and grew monotonically to
    # 7.8e-03. Capturing it is what makes the resume exact.
    kin = getattr(task, "robot_kinematics", None)
    ik_init = getattr(kin, "init_data", None) if kin is not None else None

    trs = getattr(task, "rs", None)
    t_st = trs.get_state() if trs is not None else None
    return {
        "robot": capture_state(harness),
        "obstacles": obs,
        "task_rs_state": ((t_st[0], np.asarray(t_st[1]).tolist(), t_st[2],
                           t_st[3], t_st[4]) if t_st is not None else None),
        # ALL three goals, full 4x4 frames. BenchmarkTask re-samples every goal
        # on reset, and on whole-body (WG) scenes reached_final depends on the
        # BASE goal as well as the arm goal -- restoring only robot_goal_right
        # left the base goal freshly sampled, which collapsed resumed runs to a
        # handful of steps because the robot was already "at" a different goal.
        # The base goal carries a rotation too, hence the full frame.
        "ik_init_data": (np.asarray(ik_init, float).tolist()
                         if ik_init is not None else None),
        "goals": {name: np.asarray(getattr(task, name).frame, float).tolist()
                  for name in ("robot_goal_right", "robot_goal_left",
                               "robot_goal_base")
                  if getattr(task, name, None) is not None},
    }


def restore_world(harness, st: dict):
    restore_state(harness, st["robot"])
    task = harness.env.task
    obs_list = getattr(task, "obstacle_task", []) or []
    for o, rec in zip(obs_list, st.get("obstacles", [])):
        o.frame[:] = np.asarray(rec["frame"], float)
        if rec.get("last_frame") is not None:
            o.last_frame = np.asarray(rec["last_frame"], float)
        o.last_direction = np.asarray(rec["last_direction"], float)
        # SIREN FIX (2026-08-03). get_info() publishes the obstacle velocity from
        # `last_displacement` (the REALISED step, measured after the bound clamp),
        # which older snapshots do not carry. It is recoverable EXACTLY rather
        # than approximately: snapshots are taken between steps, so `frame` is
        # post-clamp and `last_frame` is that step's pre-move snapshot, and their
        # difference is the definition of last_displacement. Without this, a
        # resumed run publishes a stale velocity on its first restored step.
        if rec.get("last_frame") is not None:
            o.last_displacement = (np.asarray(rec["frame"], float)[:3, 3]
                                   - np.asarray(rec["last_frame"], float)[:3, 3])
        else:
            o.last_displacement = np.zeros(3)
        o.step_counter = int(rec["step_counter"])
        if rec.get("rs_state") is not None and getattr(o, "rs", None) is not None:
            k, keys, pos, has_gauss, cached = rec["rs_state"]
            o.rs.set_state((k, np.asarray(keys, dtype=np.uint32), int(pos),
                            int(has_gauss), float(cached)))
    if st.get("task_rs_state") is not None and getattr(task, "rs", None) is not None:
        k, keys, pos, has_gauss, cached = st["task_rs_state"]
        task.rs.set_state((k, np.asarray(keys, dtype=np.uint32), int(pos),
                           int(has_gauss), float(cached)))
    kin = getattr(task, "robot_kinematics", None)
    if st.get("ik_init_data") is not None and kin is not None:
        kin.init_data = np.asarray(st["ik_init_data"], float)

    for name, frame in (st.get("goals") or {}).items():
        obj = getattr(task, name, None)
        if obj is not None:
            obj.frame[:] = np.asarray(frame, float)


def refresh_task_cache(harness, agent_feedback):
    """Recompute the task's CACHED view of the robot after a restore.

    THE BUG THIS FIXES. get_info() builds the goal the reference controller
    actually tracks as

        info["goal_teleop"]["right"] = self.robot_base_frame @ robot_goal_right.frame

    but `self.robot_base_frame` and `self.robot_frames_world` are cached
    attributes, refreshed ONLY inside _update_robot_state(). get_info() never
    calls it -- BenchmarkTask.step() does, just before. So a resume that went
    straight to get_info() left the task holding the base frame from reset()
    while the restored robot was somewhere else entirely.

    On a FIXED base that is harmless: the base frame never moves, so the stale
    copy is correct by accident. On a mobile or locomoting base it is not:
    goal_teleop came out 0.34 wrong, u_ref 2.69 wrong, and the resumed run
    diverged from step +1. That is exactly the observed split -- all 8
    G1FixedBase variants resumed bit-exact (56/56) while all 8 G1MobileBase
    variants and G1SportMode failed (0/66).

    Calling _update_robot_state() first is what BenchmarkTask.step() does, so
    this simply restores the ordering the simulator itself relies on.
    """
    task = harness.env.task
    upd = getattr(task, "_update_robot_state", None)
    if upd is not None and agent_feedback is not None:
        upd(agent_feedback)


def run_from_state(world, state: dict, schedule, max_steps=None,
                   exact_margin=False):
    """Run `schedule` starting from `state` instead of the home pose.

    Patches harness.reset for the duration and restores it afterwards, so the
    same world object can be reused for other runs.
    """
    h = world.harness
    original_reset = h.reset

    def patched_reset(*a, **kw):
        agent_feedback, _stale = original_reset(*a, **kw)
        restore_world(h, state)
        agent_feedback = (h.env.agent.get_feedback()
                          if hasattr(h.env.agent, "get_feedback")
                          else agent_feedback)
        refresh_task_cache(h, agent_feedback)
        task_info = h.env.task.get_info(agent_feedback)
        return agent_feedback, task_info

    h.reset = patched_reset
    try:
        return world.run(schedule, max_steps=max_steps,
                         exact_margin=exact_margin)
    finally:
        h.reset = original_reset


def run_capturing(world, schedule, capture_at_wp: int, max_steps=None,
                  exact_margin=False):
    """Run `schedule` from home, capturing the state the instant wp_idx first
    reaches `capture_at_wp`.

    Returns (record, state_at_wp or None, step_index or None).

    The capture must happen on a run where the waypoint is INTERMEDIATE. Running
    [g0] alone makes g0 final, so the reference controller decelerates into it
    and the arm arrives at REST -- a different system from the one the attack
    sees, where the arm passes through in motion. The handover state depends only
    on the APPROACH to g0 (the controller retargets at that instant), so it is
    the same whichever waypoint follows.
    """
    from ..world.sim import probe

    h = world.harness
    spec = h.spec
    steps = max_steps if max_steps is not None else h.max_steps
    probe.reset_giveups(h)
    agent_feedback, task_info = h.reset()
    h.env.task.set_goal_schedule(schedule)
    u, ai = h.algo.act(agent_feedback, task_info)

    captured, captured_at = None, None
    for t in range(steps):
        agent_feedback, task_info = h.env.step(u, ai)
        u, ai = h.algo.act(agent_feedback, task_info)
        if captured is None and int(getattr(h.env.task, "wp_idx", 0)) >= capture_at_wp:
            captured, captured_at = capture_world(h), t
        if h.env.task.reached_final:
            break
    # the record itself comes from a normal run so the labels are computed by
    # the usual classifier rather than reimplemented here
    rec = world.run(schedule, max_steps=steps, exact_margin=exact_margin)
    return rec, captured, captured_at
