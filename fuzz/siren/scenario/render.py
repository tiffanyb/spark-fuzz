"""
Render a hand-built scene, with the robot, so you can see why it works or fails.

A layout that reads fine as a list of coordinates can be physically absurd —
`wall_gap` put a wall through the robot's own forearm and collided at step 1.
Coordinates do not show that; a picture does.

Obstacles that are actually intersecting the robot at its start pose are drawn
in solid red and named, the rest translucent. Emits an orbit .mp4 and a still
.png, the format kept in fuzz/siren/experiment/.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.render --scenario wall_gap
"""

import argparse

import numpy as np


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="wall_gap")
    p.add_argument("--algo", default="ssa")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=145.0)
    p.add_argument("--elevation", type=float, default=-14.0)
    p.add_argument("--distance", type=float, default=1.35)
    a = p.parse_args(argv)

    import mujoco
    import cv2
    from spark_utils import compute_masked_distance_matrix
    from ..world.types import real_filter
    from .scenes import SCENARIOS, build_scenario_world

    scn = SCENARIOS[a.scenario]
    index = "velocity" if "_D2_" in scn.base_case else "distance"
    spec = real_filter(algo=a.algo, index=index, d_min=0.02, eta=0.02,
                       lam=10.0, k=0.1)
    w = build_scenario_world(scn, spec, max_steps=50)
    sc = w.scene()
    h = w.harness
    ag = h.env.agent
    af, ti = h.reset()
    mujoco.mj_forward(ag.model, ag.data)
    base = sc.base_frame

    # which robot volume, if any, is intersecting which obstacle right now
    si = h.algo.safe_controller.safe_algo.safety_index
    env_mask = np.asarray(si.env_collision_mask, bool)
    vol_names = list(h.robot_cfg.CollisionVol.keys())
    dmat, _ = compute_masked_distance_matrix(
        frame_list_1=h.env.task.robot_frames_world,
        geom_list_1=h.robot_cfg.CollisionVol.values(),
        frame_list_2=ti["obstacle"]["frames_world"],
        geom_list_2=ti["obstacle"]["geom"])
    dmat = np.asarray(dmat, float)
    guarded = np.where(env_mask, dmat, np.inf)
    n_obs_live = guarded.shape[1]

    print(f"scenario '{scn.name}'  ({scn.base_case}, {a.algo})")
    print(f"  {scn.note}\n")
    print(f"  G0 {np.round(sc.G0,3)}   G1 {np.round(sc.G1,3)}")
    print(f"  clearance at reset: {guarded.min():+.4f} m "
          f"({'IN CONTACT' if guarded.min() < 0 else 'clear'})\n")
    bad = set()
    print(f"  {'robot volume':<26}{'obstacle':>9}{'distance':>11}")
    order = np.dstack(np.unravel_index(np.argsort(guarded, axis=None),
                                       guarded.shape))[0]
    for vi, oi in order[:8]:
        d = guarded[vi, oi]
        if not np.isfinite(d):
            continue
        flag = "  <-- INTERSECTING" if d < 0 else ("  <- inside keep-out"
                                                  if d < 0.02 else "")
        print(f"  {vol_names[vi]:<26}{oi:>9}{d:>11.4f}{flag}")
        if d < 0.02:
            bad.add(int(oi))

    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, a.height, a.width, max_geom=4096)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)
    obs_w = np.asarray(ti["obstacle"]["frames_world"])
    cam.lookat[:] = (base @ _frame(sc.G0))[:3, 3]
    cam.distance, cam.elevation = a.distance, a.elevation

    prefix = a.out_prefix or f"/tmp/scenario_{scn.name}"
    vw = cv2.VideoWriter(prefix + "_orbit.mp4",
                         cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (a.width, a.height))
    F = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(a.frames + 1):
        still = (i == a.frames)
        cam.azimuth = a.azimuth if still else 360.0 * i / a.frames
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

        for oi in range(min(n_obs_live, len(obs_w))):
            centre = np.asarray(obs_w[oi])[:3, 3]
            if centre[2] < -1.0:            # parked surplus obstacle
                continue
            hot = oi in bad
            add(centre, 0.05,
                (1.0, 0.05, 0.05, 0.92) if hot else (0.85, 0.15, 0.15, 0.30))
        add((base @ _frame(sc.G0))[:3, 3], 0.028, (0.20, 0.45, 1.00, 0.95))
        add((base @ _frame(sc.G1))[:3, 3], 0.034, (0.10, 0.90, 0.10, 0.97))

        img = np.ascontiguousarray(renderer.render())

        def txt(y, t, col=(255, 255, 255), sc_=0.6, th=2):
            cv2.putText(img, t, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, t, (16, y), F, sc_, col, th, cv2.LINE_AA)

        txt(34, f"hand-built scene: {scn.name}", sc_=0.7)
        txt(62, f"{len(scn.obstacles)} obstacles   blue = G0 (start)   "
                f"green = G1 (goal)")
        c = guarded.min()
        txt(90, f"clearance at reset: {c:+.4f} m",
            (60, 60, 255) if c < 0 else (120, 230, 140), 0.62)
        if c < 0:
            txt(118, "ROBOT ALREADY IN CONTACT — baseline collides at step 1",
                (60, 60, 255), 0.62)
            txt(146, "solid red = obstacle intersecting the arm", (60, 60, 255), 0.55)
        if still:
            cv2.imwrite(prefix + "_still.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"\nwrote {prefix}_orbit.mp4 and {prefix}_still.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
