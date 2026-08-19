"""Place obstacles using MEASURED swept volumes. Plant untouched.

Guessing geometry failed because the "baseline path" is not the straight line
between endpoints -- it is the volume swept by the ENTIRE arm (elbow, forearm,
wrist, torso). Obstacles that clear the end-effector line still hit the elbow, so
269 of 270 guessed scenes broke the baseline instead of attacking it.

So measure it: roll out the baseline and the attack with NO obstacles, record
every guarded collision-volume centre at every step, and place the obstacle where
the RETURN leg sweeps but the BASELINE never does. That makes baseline-safety a
property of the construction rather than something to hope for.
"""
import numpy as np


def sweep_points(world, schedule, steps, leg=None):
    """All guarded robot collision-volume centres over a rollout.
    leg=None -> whole rollout; leg=2 -> only the return leg."""
    from fuzz.siren.world.sim import probe
    from fuzz.siren.pipeline.stage1_search import set_channel
    h = world.harness
    probe.reset_giveups(h)
    af, ti = h.reset()
    set_channel(world, "arm")
    h.env.task.set_goal_schedule([np.asarray(x, float) for x in schedule])
    u, ai = h.algo.act(af, ti)
    pts = []
    for t in range(steps):
        af, ti = h.env.step(u, ai)
        try:
            u, ai = h.algo.act(af, ti)
        except Exception:
            u = np.zeros_like(np.asarray(u, float))
        wp = int(getattr(h.env.task, "wp_idx", 0))
        if leg is None or wp == leg:
            fr = h.env.task.robot_frames_world
            for idx in h.robot_cfg.CollisionVol:
                pts.append(np.asarray(fr[idx], float)[:3, 3].copy())
        if h.env.task.reached_final:
            break
    return np.array(pts) if pts else np.zeros((0, 3))


def candidate_spots(base_pts, atk_pts, R, d_min, n=12, clear=0.05):
    """Points the RETURN leg passes close to but the BASELINE never approaches.

    An obstacle of radius R centred here cannot touch the baseline (its nearest
    baseline approach exceeds R + d_min + clear) but sits right on the return
    path, so only the inserted detour meets it.
    """
    if len(atk_pts) == 0 or len(base_pts) == 0:
        return []
    need = R + d_min + clear
    # subsample for speed
    A = atk_pts[:: max(1, len(atk_pts) // 4000)]
    B = base_pts[:: max(1, len(base_pts) // 4000)]
    out = []
    for p in A:
        if np.min(np.linalg.norm(B - p, axis=1)) > need:
            out.append(p)
    if not out:
        return []
    out = np.array(out)
    # spread the picks out so they are not all the same instant
    keep = [out[0]]
    for p in out:
        if min(np.linalg.norm(np.array(keep) - p, axis=1)) > 0.05:
            keep.append(p)
        if len(keep) >= n:
            break
    return keep
