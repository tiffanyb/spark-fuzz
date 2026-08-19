"""Draw ONLY the return-leg swept volume L2, coloured by how far each link moves.

L2 is the swept volume of every guarded collision volume over leg 2, not the
hand's path. This renders all of it with no other set drawn, colouring each
point by the distance its own link travels during that leg, so the composition
of the set is visible directly:

    orange  travel > 0.15 m   hand and outer wrist -- these track G1' -> G1
    yellow  0.05 - 0.15 m     inner wrist and elbow
    grey    travel < 0.05 m   shoulders, torso, waist, pelvis, whole left arm

The arm channel commands only the right arm, so the grey links are stationary
under every schedule and occupy identical world positions in L2 and in the
legitimate task's swept volume.

    python -m fuzz.siren.constructed.viz_l2
"""

import argparse
import os

import numpy as np

from .separated_search import CASE, park_all
from .viz_swept import sweep_by_volume


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--g1p", default="0.207,-0.38,0.28")
    p.add_argument("--out", default="fuzz/siren/constructed/stock_visualizations")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--distance", type=float, default=0.85)
    p.add_argument("--still-az", type=float, default=125.0)
    a = p.parse_args(argv)

    import cv2
    import mujoco

    from ..pipeline import trialconf
    from ..world.run import World

    G1p = np.array([float(x) for x in a.g1p.split(",")])
    cfg = trialconf.load("fuzz/siren/pipeline/configs/trial2_ours.yaml")
    spec = trialconf.spec_for(cfg, "ssa", CASE)

    w = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=a.steps)
    sc = w.scene()
    G0, G1 = np.asarray(sc.G0, float), np.asarray(sc.G1, float)
    bf = np.asarray(sc.base_frame, float)
    park_all(w)
    P, V, L = sweep_by_volume(w, [G0, G1p, G1], a.steps)
    names = [str(k).replace("Frames.", "")
             for k in w.harness.robot_cfg.CollisionVol]
    P2, V2 = P[L == 2], V[L == 2]

    travel = {}
    for vi in np.unique(V2):
        q = P2[V2 == vi]
        travel[int(vi)] = float(np.linalg.norm(q.max(axis=0) - q.min(axis=0)))

    def tier(vi):
        t = travel[int(vi)]
        return 0 if t > 0.15 else (1 if t > 0.05 else 2)

    COL = [(1.00, 0.45, 0.05, 0.95),      # moving: hand / outer wrist
           (1.00, 0.85, 0.15, 0.85),      # intermediate
           (0.55, 0.55, 0.58, 0.75)]      # stationary
    counts = [0, 0, 0]
    links = [[], [], []]
    for vi in np.unique(V2):
        counts[tier(vi)] += int((V2 == vi).sum())
        links[tier(vi)].append(names[int(vi)])
    tot = len(P2)
    print(f"L2 total {tot} points from {len(np.unique(V2))} collision volumes")
    for i, lab in enumerate(("travel > 0.15 m", "0.05 - 0.15 m", "< 0.05 m")):
        print(f"  {lab:<16} {len(links[i]):2d} links  {counts[i]:5d} pts "
              f"({100*counts[i]/tot:5.1f}%)  {sorted(links[i])[:4]}", flush=True)

    w2 = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=a.steps)
    park_all(w2)
    w2.harness.reset()
    ag = w2.harness.env.agent
    qpos = np.asarray(ag.data.qpos, float).copy()
    tw = lambda q: (bf @ np.append(np.asarray(q, float), 1.0))[:3]

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    r = mujoco.Renderer(ag.model, a.height, a.width, max_geom=40000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = P2.mean(axis=0)
    cam.distance, cam.elevation = a.distance, -12.0

    os.makedirs(a.out, exist_ok=True)
    mp4 = f"{a.out}/L2_only.mp4"
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

        for q, vi in zip(P2, V2):
            add(q, 0.006, COL[tier(vi)])
        add(tw(G0), 0.026, (0.20, 0.45, 1.00, 1.0))
        add(tw(G1p), 0.030, (1.00, 0.85, 0.10, 1.0))
        add(tw(G1), 0.030, (0.10, 0.90, 0.10, 1.0))
        img = np.ascontiguousarray(r.render())

        def txt(y, t, col=(255, 255, 255), sc=0.58, th=2):
            cv2.putText(img, t, (16, y), F, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, t, (16, y), F, sc, col, th, cv2.LINE_AA)

        txt(32, f"L2 ONLY - return leg swept volume, {tot} points   "
                f"G1' = {np.round(G1p,3).tolist()}", sc=0.60)
        txt(60, f"orange  {len(links[0])} links travel > 0.15 m   "
                f"{counts[0]} pts ({100*counts[0]/tot:.0f}%)", (255, 150, 60))
        txt(86, f"yellow  {len(links[1])} links travel 0.05-0.15 m  "
                f"{counts[1]} pts ({100*counts[1]/tot:.0f}%)", (255, 220, 90))
        txt(112, f"grey    {len(links[2])} links travel < 0.05 m    "
                 f"{counts[2]} pts ({100*counts[2]/tot:.0f}%)", (190, 190, 195))
        txt(138, "only the orange links track G1' -> G1; the grey ones are "
                 "stationary under every schedule")
        txt(164, "markers: G0 blue, G1' yellow, G1 green")
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if still is None or abs(cam.azimuth - a.still_az) < 360.0 / a.frames:
            still = img.copy()
    vw.release()
    png = f"{a.out}/L2_only.png"
    cv2.imwrite(png, cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
    print(f"wrote {mp4} and {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
