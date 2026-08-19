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


# --------------------------------------------------------------------------- #
# Surface separation.
#
# sweep_points returns collision-volume CENTRES. The original placement metric
# was `min |p - b| - R_obstacle`, which subtracts only the obstacle's radius and
# silently treats the robot as a point cloud. The robot is not a point cloud:
# SPARK models it as spheres of 0.05 m on arm links, 0.06 at the shoulder rolls,
# 0.08-0.10 at the torso. The true free space between an obstacle of radius R
# centred at p and a robot sphere of radius r_b centred at b is
#
#     |p - b| - R - r_b
#
# Omitting r_b overstates free space by 5-10 cm, which is larger than the entire
# band the placement search operates in -- every placement selected under the old
# metric sits INSIDE the swept volume rather than clear of it. Both call sites
# had the same bug, so the corrected metric lives here and is shared.
# --------------------------------------------------------------------------- #

def volume_radii(world):
    """Radius of each guarded collision volume, in CollisionVol iteration order.

    sweep_points and legs_sweep append one point per volume per step in exactly
    this order, so the per-point radii are this array tiled over the steps.
    """
    out = []
    for g in world.harness.robot_cfg.CollisionVol.values():
        r = (g.attributes["radius"] if hasattr(g, "attributes")
             else getattr(g, "radius", 0.05))
        out.append(float(r))
    return np.array(out, float)


def tile_radii(world, n_points):
    """Per-point radii for a sweep of n_points, tiled from volume_radii."""
    r = volume_radii(world)
    if n_points % len(r):
        raise ValueError(f"{n_points} points is not a whole number of sweeps "
                         f"over {len(r)} collision volumes")
    return np.resize(r, n_points)


def surface_gap(probe_pts, swept_pts, swept_radii, r_obstacle, chunk=256):
    """Free space between a sphere of radius r_obstacle at each probe point and
    the nearest surface of the swept volume.

    Returns one value per probe point. Negative means the sphere overlaps the
    swept volume. Chunked so the pairwise distance never materialises in full.
    """
    probe_pts = np.asarray(probe_pts, float)
    swept_pts = np.asarray(swept_pts, float)
    swept_radii = np.asarray(swept_radii, float)
    if len(probe_pts) == 0 or len(swept_pts) == 0:
        return np.full(len(probe_pts), np.inf)
    out = np.empty(len(probe_pts))
    for i in range(0, len(probe_pts), chunk):
        blk = probe_pts[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - swept_pts[None, :, :], axis=2)
        out[i:i + chunk] = (d - swept_radii[None, :]).min(axis=1)
    return out - float(r_obstacle)
