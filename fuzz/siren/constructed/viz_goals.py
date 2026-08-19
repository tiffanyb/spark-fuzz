"""Render the candidate inserted goals G1' for one seed, in the 3-D scene.

phase_spots enumerates G1' over the task's own goal box and keeps those whose
return leg offers a placement in the gap band. The result is a list of numbers;
this shows where those goals actually are relative to the robot, the legitimate
task, and the box they were drawn from.

Drawn:
  wireframe box   right_arm_goal_range, the domain G1' is enumerated over
  blue            G0, the arm's start (which is also where the legitimate task begins)
  green           G1, the legitimate goal
  yellow->red     each surviving G1', shaded by how many placements it yielded
  small orange    the placements themselves (candidate obstacle centres)

Note the placements are hypothetical obstacle positions: phase_spots runs with
every scene obstacle parked outside the workspace, so the scene's own obstacles
are NOT what these mark.

    python -m fuzz.siren.constructed.viz_goals --seed 10
"""

import argparse
import collections
import json
import os

import numpy as np

from .separated_search import CASE, park_all


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--spots", default=None)
    p.add_argument("--out", default="fuzz/siren/constructed/rq1_results")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--distance", type=float, default=1.30)
    p.add_argument("--still-az", type=float, default=210.0)
    p.add_argument("--max-spots-drawn", type=int, default=420)
    a = p.parse_args(argv)

    import cv2
    import mujoco

    from ..pipeline import trialconf
    from ..world.run import World

    spots_file = a.spots or f"{a.out}/stock_spots_{a.seed}.json"
    S = json.load(open(spots_file))
    G0 = np.asarray(S["G0"], float)
    G1 = np.asarray(S["G1"], float)
    bounds = S["bounds"]
    per_goal = collections.Counter(tuple(x["G1_prime"]) for x in S["spots"])
    goals = sorted(per_goal.items(), key=lambda kv: kv[1])
    spots = [np.asarray(x["spot"], float) for x in S["spots"]]
    spots = spots[:: max(1, len(spots) // a.max_spots_drawn)]
    print(f"seed {a.seed}: {len(S['spots'])} placements from "
          f"{len(per_goal)} distinct G1'", flush=True)

    cfg = trialconf.load("fuzz/siren/pipeline/configs/trial2_ours.yaml")
    spec = trialconf.spec_for(cfg, "ssa", CASE)
    w = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=200)
    park_all(w)
    h = w.harness
    h.reset()
    ag = h.env.agent
    qpos = np.asarray(ag.data.qpos, float).copy()
    bf = np.asarray(w.scene().base_frame, float)

    def tw(q):
        return (bf @ np.append(np.asarray(q, float), 1.0))[:3]

    # goal-box wireframe, as strings of small spheres (a capsule would need an
    # orientation matrix per edge for no visual gain at this size)
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    corners = [(x, y, z) for x in (xlo, xhi) for y in (ylo, yhi)
               for z in (zlo, zhi)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if sum(abs(np.array(corners[i]) - np.array(corners[j])) > 1e-9) == 1]
    box_pts = []
    for i, j in edges:
        ci, cj = np.array(corners[i]), np.array(corners[j])
        for t in np.linspace(0, 1, 26):
            box_pts.append(tw(ci + t * (cj - ci)))

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    r = mujoco.Renderer(ag.model, a.height, a.width, max_geom=40000)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    cam.lookat[:] = 0.5 * (tw(G0) + tw(G1)) + np.array([0.0, -0.05, 0.02])
    cam.distance, cam.elevation = a.distance, -14.0

    os.makedirs(a.out, exist_ok=True)
    mp4 = f"{a.out}/goals_seed{a.seed}.mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                         (a.width, a.height))
    F = cv2.FONT_HERSHEY_SIMPLEX
    nmax = max(per_goal.values())
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

        for q in box_pts:
            add(q, 0.0032, (0.55, 0.55, 0.58, 0.5))
        for q in spots:
            add(q, 0.0035, (1.00, 0.45, 0.05, 0.32))
        # Sequential PURPLE ramp: one hue, light -> dark, so productivity reads
        # as depth of colour rather than a hue change. Purple also separates the
        # goals cleanly from the orange placements -- the earlier yellow->red
        # ramp collided with them, which is what made the shading unreadable.
        for g, n in goals:
            f = n / nmax
            add(tw(g), 0.0070 + 0.0060 * f,
                (0.78 - 0.49 * f, 0.72 - 0.49 * f, 0.92 - 0.27 * f,
                 0.50 + 0.45 * f))
        add(tw(G0), 0.030, (0.16, 0.47, 0.84, 1.0))
        add(tw(G1), 0.032, (0.05, 0.55, 0.36, 1.0))
        img = np.ascontiguousarray(r.render())

        def txt(y, t, col=(255, 255, 255), sc=0.58, th=2):
            cv2.putText(img, t, (16, y), F, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, t, (16, y), F, sc, col, th, cv2.LINE_AA)

        txt(32, f"CANDIDATE INSERTED GOALS   seed {a.seed}   "
                f"{len(per_goal)} of 125 enumerated G1' survived", sc=0.60)
        txt(60, "grey wireframe = right_arm_goal_range, the box G1' is drawn from",
            (200, 200, 205))
        txt(86, f"light -> deep PURPLE = a surviving G1', deeper = more "
                f"placements it yielded (1 to {nmax})", (190, 150, 245))
        txt(112, f"orange = the {len(S['spots'])} placements themselves "
                 f"(hypothetical obstacle centres; scene obstacles are parked)",
            (120, 190, 255))
        txt(138, "blue = G0 (arm start)     green = G1 (legitimate goal)",
            (200, 230, 210))
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if still is None or abs(cam.azimuth - a.still_az) < 360.0 / a.frames:
            still = img.copy()
    vw.release()
    png = f"{a.out}/goals_seed{a.seed}.png"
    cv2.imwrite(png, cv2.cvtColor(still, cv2.COLOR_RGB2BGR))
    print(f"wrote {mp4} and {png}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
