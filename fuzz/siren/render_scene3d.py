"""
Orbiting render of one scene with the robot, obstacles, and every goal tried.

Same visual language as render_case.py, but static: the robot is held at its
start pose and the camera orbits, so the 3-D structure of the goal cloud reads
properly. Candidates come from the sweep record — the world is built once and
never stepped, so this costs a single reset plus the frames.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.render_scene3d \
        --case G1FixedBase_D1_AG_SO_v0 --seed 0
"""

import argparse
import glob
import json

import numpy as np

_RGBA = {"REACHED":   (0.62, 0.62, 0.62, 0.75),
         "TIMEOUT":   (1.00, 0.60, 0.10, 0.85),
         "COLLISION": (1.00, 0.10, 0.10, 0.95),
         "DEADLOCK":  (0.80, 0.10, 0.90, 0.95)}


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, float).reshape(3)
    return f


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", default="/tmp/rep_shard*.jsonl")
    p.add_argument("--out", default=None)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--elevation", type=float, default=-16.0)
    p.add_argument("--still", action="store_true",
                   help="write a single PNG instead of an orbit")
    p.add_argument("--azimuth", type=float, default=135.0,
                   help="camera azimuth for --still")
    p.add_argument("--distance", type=float, default=1.25)
    a = p.parse_args(argv)

    import mujoco
    import cv2
    from .world.run import World
    from .world.types import real_filter

    pts = []
    for f in sorted(glob.glob(a.sweep)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or r["case"] != a.case or r["seed"] != a.seed:
                continue
            for e in r["evaluations"]:
                if e["stage"] == "inadmissible":
                    continue
                lab = (e["leg2_label"] if e["stage"] == "evaluated"
                       else (e["leg1_label"] or "TIMEOUT"))
                pts.append((np.asarray(e["candidate"], float), lab or "TIMEOUT"))
    if not pts:
        print(f"no records for {a.case} seed {a.seed}")
        return 1
    counts = {}
    for _c, l in pts:
        counts[l] = counts.get(l, 0) + 1

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
    w = World.build(seed=a.seed, spec=spec, test_case=a.case, max_steps=10)
    sc = w.scene()
    h = w.harness
    ag = h.env.agent
    h.reset()                      # robot at its start pose; never stepped after
    mujoco.mj_forward(ag.model, ag.data)
    base = sc.base_frame

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, a.height, a.width,
                               max_geom=len(pts) + 4096)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    obs_w = np.asarray(sc.obstacles_world)
    focus = (obs_w[:, :3, 3].mean(0) if len(obs_w)
             else (base @ _frame(sc.G1))[:3, 3])
    # bias the look-at toward the goal box rather than the obstacle centroid,
    # since several obstacles sit well outside the reachable region
    focus = 0.5 * focus + 0.5 * (base @ _frame(sc.G0))[:3, 3]
    cam.lookat[:] = focus
    cam.distance, cam.elevation = a.distance, a.elevation

    out = a.out or (f"/tmp/scene3d_{a.case}_s{a.seed}."
                    + ("png" if a.still else "mp4"))
    vw = None if a.still else cv2.VideoWriter(
        out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (a.width, a.height))
    n_frames = 1 if a.still else a.frames

    def world_of(xyz_base):
        return (base @ _frame(xyz_base))[:3, 3]

    F = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n_frames):
        cam.azimuth = a.azimuth if a.still else 360.0 * i / a.frames
        renderer.update_scene(ag.data, camera=cam)
        s = renderer.scene

        def add(pos, r, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, r, r], float),
                                np.asarray(pos, float).reshape(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            s.ngeom += 1

        for of in obs_w:
            add(np.asarray(of)[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.38))
        for c, lab in pts:
            add(world_of(c), 0.008, _RGBA.get(lab, (0.5, 0.5, 0.5, 0.7)))
        add(world_of(sc.G0), 0.028, (0.20, 0.45, 1.00, 0.95))
        add(world_of(sc.G1), 0.034, (0.10, 0.90, 0.10, 0.97))

        img = np.ascontiguousarray(renderer.render())

        def txt(y, t, col=(255, 255, 255), sc_=0.6, th=2):
            cv2.putText(img, t, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, t, (16, y), F, sc_, col, th, cv2.LINE_AA)

        txt(34, f"{a.case}   seed {a.seed}", sc_=0.68)
        txt(62, f"{len(pts)} inserted goals tried    "
                f"blue = G0    green = G1    red = obstacles")
        y = 92
        for lab in ("REACHED", "TIMEOUT", "DEADLOCK", "COLLISION"):
            if lab in counts:
                col = tuple(int(255 * v) for v in _RGBA[lab][:3])
                txt(y, f"{lab}: {counts[lab]}", col, 0.58)
                y += 26
        if "COLLISION" not in counts:
            txt(y + 6, "NO COLLISIONS", (90, 230, 120), 0.72, 2)
        if a.still:
            cv2.imwrite(out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if vw is not None:
        vw.release()
    print(f"{len(pts)} goals, outcomes={counts}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
