"""
Render one attack end to end: G0 -> G1' -> G1, with the filter's state on screen.

Produces an mp4 plus a step-by-step dissection of the contact. The overlay shows,
every frame:

    blue sphere      G0, where the hand started
    yellow sphere    G1', the attacker's inserted goal
    green sphere     G1, the legitimate goal
    red spheres      obstacles
    white ring       the robot volume currently closest to an obstacle

    red border + "FILTER ENGAGED"   a constraint is actively being enforced
    clearance / phi / braking margin, and the two moments that matter:
        t0    the first step the filter intervened on leg 2
        PNR   the point of no return -- the last step from which a best-effort
              controller could still have avoided contact (from the brute-force
              escape search, not from a model)

The point of the video is the gap between those two: the filter engages, works
for many steps, and the state still becomes unrecoverable while it is working.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.render_case
"""

import argparse
import json

import numpy as np


def _xyz_to_frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from-json", default="/tmp/picked_kind2.json")
    p.add_argument("--out", default="/tmp/kind2_case.mp4")
    p.add_argument("--dump", default="/tmp/kind2_case.json")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=150.0)
    p.add_argument("--elevation", type=float, default=-18.0)
    p.add_argument("--distance", type=float, default=1.2)
    a = p.parse_args(argv)

    import mujoco
    import cv2
    from spark_utils import compute_masked_distance_matrix
    from .world.run import World, BRAKE_SCALE
    from .world.types import real_filter
    from .world.sim import probe
    from .world import derived

    case = json.load(open(a.from_json))
    cand = np.asarray(case["candidate"], float)
    t0, pnr, contact = case["t0_leg2"], case["point_of_no_return"], case["contact_step"]

    index = "velocity" if "_D2_" in case["case"] else "distance"
    ms = 300 if "_D2_" in case["case"] else 500
    spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
    w = World.build(seed=case["seed"], spec=spec, test_case=case["case"], max_steps=ms)
    sc = w.scene()
    h = w.harness
    si = h.algo.safe_controller.safe_algo.safety_index
    env_mask = np.asarray(si.env_collision_mask, bool)
    n_obs = si.num_obstacle_vol
    vol_names = list(h.robot_cfg.CollisionVol.keys())
    interval = float(h.env.agent.dt * h.env.agent.control_decimation)
    base = sc.base_frame

    agent = h.env.agent
    agent.model.vis.global_.offwidth = max(a.width, agent.model.vis.global_.offwidth)
    agent.model.vis.global_.offheight = max(a.height, agent.model.vis.global_.offheight)
    renderer = mujoco.Renderer(agent.model, a.height, a.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(agent.model, cam)
    obs_w = np.asarray(sc.obstacles_world)
    cam.lookat[:] = (obs_w[:, :3, 3].mean(0) if len(obs_w)
                     else (base @ _xyz_to_frame(sc.G1))[:3, 3])
    cam.distance, cam.elevation, cam.azimuth = a.distance, a.elevation, a.azimuth

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (a.width, a.height))

    def world_of(xyz_base):
        return (base @ _xyz_to_frame(xyz_base))[:3, 3]

    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule([cand, np.asarray(sc.G1)])
    u, ai = h.algo.act(af, ti)

    log, prev_clear = [], None
    for t in range(ms):
        af, ti = h.env.step(u, ai)
        u_ref = ai.get("u_ref", None)
        u_applied = np.asarray(u, float).reshape(-1)
        u, ai = h.algo.act(af, ti)

        dmat, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=ti["obstacle"]["frames_world"],
            geom_list_2=ti["obstacle"]["geom"])
        guarded = np.where(env_mask, np.asarray(dmat, float), np.inf)
        flat = int(np.argmin(guarded))
        vi, oi = divmod(flat, n_obs)
        clear = float(guarded[vi, oi])
        v_close = 0.0 if prev_clear is None else (prev_clear - clear) / interval
        prev_clear = clear

        raw = probe.read_raw(h)
        phi = C_phi = np.nan
        engaged = False
        if raw:
            d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"], raw["phi_mask"],
                                 raw["u_lim"], demand_shape=spec.demand_shape,
                                 eta=spec.eta, lam=spec.lam, exact=False)
            phi, C_phi = d.get("phi", np.nan), d.get("C_phi", np.inf)
            engaged = bool(d.get("engaged", False))
        brake = (clear - v_close ** 2 / (2 * BRAKE_SCALE * C_phi)
                 if (v_close > 0 and np.isfinite(C_phi) and C_phi > 0) else np.inf)
        dev = (float(np.linalg.norm(u_applied - np.asarray(u_ref, float).reshape(-1)))
               if u_ref is not None else 0.0)
        wp = int(getattr(h.env.task, "wp_idx", 0))

        log.append({"t": t, "wp": wp, "clear": clear, "vol": vol_names[vi],
                    "vol_idx": vi, "obs": oi, "phi": float(phi),
                    "C_phi": float(C_phi), "v_close": float(v_close),
                    "brake": float(brake), "engaged": engaged, "deviation": dev})

        # ---------------- render ---------------- #
        renderer.update_scene(agent.data, camera=cam)
        s = renderer.scene

        def add(pos, r, rgba):
            if s.ngeom >= s.maxgeom:
                return
            mujoco.mjv_initGeom(s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([r, r, r], float),
                                np.asarray(pos, float).reshape(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            s.ngeom += 1

        for of in ti["obstacle"]["frames_world"]:
            add(np.asarray(of)[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.40))
        add(world_of(sc.G0), 0.028, (0.20, 0.45, 1.00, 0.95))
        add(world_of(cand), 0.034, (1.00, 0.85, 0.10, 0.95))
        add(world_of(sc.G1), 0.034, (0.10, 0.90, 0.10, 0.95))
        # ring the robot volume that is currently closest to an obstacle
        add(h.env.task.robot_frames_world[vi][:3, 3], 0.045,
            (1.0, 1.0, 1.0, 0.30 if clear > 0.01 else 0.75))

        img = renderer.render()
        img = np.ascontiguousarray(img)

        if engaged:                                   # red border while enforcing
            cv2.rectangle(img, (0, 0), (a.width - 1, a.height - 1), (255, 40, 40), 8)

        leg = "leg 1:  G0 -> G1'  (attacker's inserted goal)" if wp == 0 \
            else "leg 2:  G1' -> G1  (back to the legitimate goal)"
        F = cv2.FONT_HERSHEY_SIMPLEX
        def txt(y, s_, col=(255, 255, 255), sc_=0.62, th=2):
            cv2.putText(img, s_, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(img, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

        txt(34, f"step {t:3d}    {leg}", sc_=0.7)
        txt(64, f"clearance {clear:+.4f} m   ({log[-1]['vol']} vs obstacle {oi})")
        txt(92, f"phi {phi:+.4f}   closing {v_close:+.3f} m/s   "
                f"braking margin {brake:+.4f}" if np.isfinite(brake)
                else f"phi {phi:+.4f}   closing {v_close:+.3f} m/s   braking margin  --")
        txt(120, "FILTER ENGAGED — enforcing a constraint" if engaged
                 else "filter idle — no constraint active",
            (60, 90, 255) if engaged else (190, 190, 190))
        if t0 is not None and t >= t0:
            txt(150, f"t0 = {t0}: filter first intervened on leg 2", (120, 200, 255), 0.58)
        if pnr is not None and t >= pnr:
            txt(178, f"PNR = {pnr}: past this, NO control avoids contact",
                (80, 160, 255), 0.58)
        if clear < 0:
            txt(212, "COLLISION", (60, 60, 255), 1.0, 3)

        hold = 1
        if t in (t0, pnr, contact) or clear < 0:
            hold = a.fps * 2                          # linger on the key moments
        for _ in range(hold):
            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if h.env.task.reached_final or clear < 0.0:
            break

    vw.release()
    json.dump({"case": case, "log": log}, open(a.dump, "w"), indent=2, default=float)
    print(f"wrote {a.out}  ({len(log)} steps)")
    print(f"wrote {a.dump}")

    # ---------------- dissection ---------------- #
    print(f"\n{'='*96}\nCONTACT at step {log[-1]['t']}: {log[-1]['vol']} vs obstacle "
          f"{log[-1]['obs']}, clearance {log[-1]['clear']:+.5f}\n{'='*96}")
    print(f"{'t':>4} {'leg':>4} {'clear':>9} {'phi':>9} {'v_close':>9} "
          f"{'brake':>10} {'C_phi':>8} {'|u-u_ref|':>10} {'eng':>4}  note")
    for r in log:
        if r["t"] < min(t0, pnr) - 6:
            continue
        note = ""
        if r["t"] == t0: note = "<- t0: filter first engages on leg 2"
        if r["t"] == pnr: note = "<- POINT OF NO RETURN"
        if r["t"] == log[-1]["t"]: note = "<- CONTACT"
        b = f"{r['brake']:+.4f}" if np.isfinite(r["brake"]) else "    --"
        print(f"{r['t']:>4} {r['wp']:>4} {r['clear']:>+9.4f} {r['phi']:>+9.4f} "
              f"{r['v_close']:>+9.3f} {b:>10} {r['C_phi']:>8.3f} "
              f"{r['deviation']:>10.4f} {str(r['engaged']):>4}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
