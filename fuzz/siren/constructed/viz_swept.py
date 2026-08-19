"""Draw the three swept volumes phase_spots reasons about, in the 3-D scene.

The placement criterion compares three sampled swept volumes:

    B   [G0, G1]            the legitimate task
    L1  leg 1 of [G0,G1',G1]  outbound, home -> G1'
    L2  leg 2 of [G0,G1',G1]  return,   G1' -> G1

and keeps points of L2 that are far from both B and L1. This renders those three
point sets in world coordinates together with the robot and the obstacle that
was actually selected, so the geometry the criterion depends on is visible
rather than inferred from numbers.

Every simulation here runs with the obstacles parked outside the workspace, so
the safety filter never activates and the trajectories are the reference
controller's own -- the same sets phase_spots used.

    python -m fuzz.siren.constructed.viz_swept --g1p 0.207,-0.38,0.28
"""

import argparse
import json
import os

import numpy as np

from .separated_search import CASE, park_all


def sweep_by_volume(world, schedule, steps):
    """Sampled swept-volume points tagged with (leg, collision-volume index).

    sweep_points and legs_sweep flatten across volumes, which is all the
    placement criterion needs but loses the identity required to tell a moving
    link from a stationary one.
    """
    from ..pipeline.stage1_search import set_channel
    from ..world.sim import probe
    h = world.harness
    probe.reset_giveups(h)
    af, ti = h.reset()
    set_channel(world, "arm")
    h.env.task.set_goal_schedule([np.asarray(x, float) for x in schedule])
    u, ai = h.algo.act(af, ti)
    vols = list(h.robot_cfg.CollisionVol)
    P, V, L = [], [], []
    for _ in range(steps):
        af, ti = h.env.step(u, ai)
        try:
            u, ai = h.algo.act(af, ti)
        except Exception:
            u = np.zeros_like(np.asarray(u, float))
        wp = int(getattr(h.env.task, "wp_idx", 0))
        fr = h.env.task.robot_frames_world
        for vi, idx in enumerate(vols):
            P.append(np.asarray(fr[idx], float)[:3, 3].copy())
            V.append(vi)
            L.append(wp)
        if h.env.task.reached_final:
            break
    return np.array(P), np.array(V), np.array(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--g1p", default="0.207,-0.38,0.28")
    p.add_argument("--attack", default="fuzz/siren/constructed/stock_attacks/ssa_0.json",
                   help="attack whose selected obstacle is drawn; '' for none")
    p.add_argument("--npts", type=int, default=550, help="points drawn per volume")
    p.add_argument("--out", default="fuzz/siren/constructed/stock_visualizations")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--distance", type=float, default=0.80)
    p.add_argument("--still-az", type=float, default=125.0)
    p.add_argument("--move-min", type=float, default=0.02,
                   help="a volume is drawn only if its position varies by more "
                        "than this over the legitimate task")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    a = p.parse_args(argv)

    import cv2
    import mujoco

    from ..pipeline import trialconf
    from ..world.run import World
    from .swept import sweep_points

    G1p = np.array([float(x) for x in a.g1p.split(",")])
    cfg = trialconf.load("fuzz/siren/pipeline/configs/trial2_ours.yaml")
    spec = trialconf.spec_for(cfg, "ssa", CASE)

    w = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=a.steps)
    sc = w.scene()
    G0, G1 = np.asarray(sc.G0, float), np.asarray(sc.G1, float)
    bf = np.asarray(sc.base_frame, float)
    park_all(w)
    Pb, Vb, _ = sweep_by_volume(w, [G0, G1], a.steps)

    w2 = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=a.steps)
    park_all(w2)
    Pa, Va, La = sweep_by_volume(w2, [G0, G1p, G1], a.steps)

    # A collision volume whose position barely changes contributes the same
    # points to all three sets, so it can never set the minimum separation
    # between them. Those are the torso, waist and left-arm volumes, which the
    # arm channel does not command. Drawing them only obscures the comparison,
    # so the figure shows the volumes that actually move; the criterion itself
    # still uses every guarded volume.
    span = {}
    for vi in np.unique(Vb):
        q = Pb[Vb == vi]
        span[int(vi)] = float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))
    moving = {vi for vi, d in span.items() if d > a.move_min}
    print(f"{len(moving)} of {len(span)} collision volumes move more than "
          f"{a.move_min} m", flush=True)
    mb = np.array([v in moving for v in Vb])
    ma = np.array([v in moving for v in Va])
    B = Pb[mb]
    L1 = Pa[ma & (La == 1)]
    L2 = Pa[ma & (La == 2)]
    print(f"B {len(B)}   L1 {len(L1)}   L2 {len(L2)} sampled points", flush=True)

    # residual separation of every L2 point, the quantity the criterion uses
    Bs = B[:: max(1, len(B) // 1500)]
    L1s = L1[:: max(1, len(L1) // 1500)]
    db = np.min(np.linalg.norm(L2[:, None] - Bs[None], axis=2), axis=1)
    d1 = np.min(np.linalg.norm(L2[:, None] - L1s[None], axis=2), axis=1)
    gap = np.minimum(db, d1) - 0.05
    for lo, hi in ((0.002, 0.022), (0.016, 0.040), (0.024, 0.030)):
        frac = float(((gap >= lo) & (gap <= hi)).mean())
        print(f"  gap in [{lo}, {hi}]: {frac*100:5.2f}% of L2 points", flush=True)
    print(f"gap over L2: max {gap.max():+.4f}  median {np.median(gap):+.4f}",
          flush=True)

    obs_pos, obs_r = None, 0.05
    if a.attack and os.path.exists(a.attack):
        v = json.load(open(a.attack))
        obs_pos = np.asarray(v["obstacles_world"][0], float)
        obs_r = float(v["obstacle_radius"])

    # fresh world for rendering, so the robot sits at its start configuration
    w3 = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=a.steps)
    park_all(w3)
    h3 = w3.harness
    h3.reset()
    ag = h3.env.agent
    qpos = np.asarray(ag.data.qpos, float).copy()

    def sub(P, n):
        return P[:: max(1, len(P) // n)]

    Bd, L1d, L2d = sub(B, a.npts), sub(L1, a.npts), sub(L2, a.npts)
    tw = lambda q: (bf @ np.append(np.asarray(q, float), 1.0))[:3]

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    r = mujoco.Renderer(ag.model, a.height, a.width, max_geom=40000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = np.mean(np.vstack([Bd, L1d, L2d]), axis=0)
    cam.distance, cam.elevation = a.distance, -12.0

    os.makedirs(a.out, exist_ok=True)
    mp4 = f"{a.out}/swept_volumes.mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                         (a.width, a.height))
    F = cv2.FONT_HERSHEY_SIMPLEX
    still = None
    for fi in range(a.frames):
        cam.azimuth = 360.0 * fi / a.frames
        ag.data.qpos[:] = qpos
        mujoco.mj_forward(ag.model, ag.data)
        r.update_scene(ag.data, camera=cam)
        s = r.scene

        def add(pos, rad, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([rad, rad, rad], float),
                                np.asarray(pos, float).reshape(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            s.ngeom += 1

        for q in Bd:
            add(q, 0.006, (0.10, 0.85, 0.25, 0.65))     # B  legitimate
        for q in L1d:
            add(q, 0.006, (0.25, 0.55, 1.00, 0.65))     # L1 outbound
        for q in L2d:
            add(q, 0.006, (1.00, 0.50, 0.05, 0.75))     # L2 return
        if obs_pos is not None:
            add(obs_pos, obs_r, (1.00, 0.05, 0.05, 0.55))
        add(tw(G0), 0.026, (0.20, 0.45, 1.00, 1.0))
        add(tw(G1p), 0.030, (1.00, 0.85, 0.10, 1.0))
        add(tw(G1), 0.030, (0.10, 0.90, 0.10, 1.0))
        img = np.ascontiguousarray(r.render())

        def txt(y, t, col=(255, 255, 255), sc=0.58, th=2):
            cv2.putText(img, t, (16, y), F, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, t, (16, y), F, sc, col, th, cv2.LINE_AA)

        txt(32, f"SWEPT VOLUMES   {CASE}  seed {a.seed}   "
                f"G1' = {np.round(G1p,3).tolist()}", sc=0.60)
        txt(60, "green   B  = legitimate task [G0, G1]", (120, 255, 140))
        txt(86, "blue    L1 = outbound leg, home -> G1'", (120, 190, 255))
        txt(112, "orange  L2 = return leg, G1' -> G1", (255, 150, 60))
        txt(138, f"red = selected obstacle (r = {obs_r*100:.0f} cm): on L2, "
                 f"clear of B and L1", (255, 120, 120))
        txt(164, "markers: G0 blue, G1' yellow, G1 green")
        txt(190, f"moving collision volumes only; "
                 f"{len(Bd)}/{len(L1d)}/{len(L2d)} of "
                 f"{len(B)}/{len(L1)}/{len(L2)} points drawn", sc=0.50)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # capture the still at the angle the attack videos use, not at azimuth 0
        # where the torso occludes the region the criterion is about
        if still is None or abs(cam.azimuth - a.still_az) < 360.0 / a.frames:
            still = img.copy()
    vw.release()
    png = f"{a.out}/swept_volumes.png"
    cv2.imwrite(png, cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
    print(f"wrote {mp4} and {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
