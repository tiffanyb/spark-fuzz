"""
Simulate and render one discovered attack: home -> G0 -> G1' -> G1.

This SIMULATES -- it does not replay a stored trajectory. Physics and rendering
share one loop and one MjData: h.env.step() advances mj_step, and
renderer.update_scene() reads the state that step just wrote. So producing the
video costs a full rollout (~90 s here), and the frames are of the run happening
now, not of a recorded one. Hence the name.

That is a property of this design rather than a necessity: logging qpos per step
and replaying it into the renderer afterwards would decouple the two. Not worth
it at one video per case, but it is the escape hatch if rendering ever needs to
be cheap or repeatable at a different camera angle without re-running physics.

Unlike render_case.py this is driven by a fuzz TARGET plus a candidate goal the
search actually found, so it uses the target's own filter and scene rather than
a hardcoded one. The schedule carries G0 as a prefix waypoint, matching the
contract every other check in scenario/ uses:

    leg 0   home -> G0     setup; identical on every run
    leg 1   G0   -> G1'    reaching the inserted goal
    leg 2   G1'  -> G1     the attack leg, where contact is expected

Obstacles are drawn translucent pink throughout. The moment a guarded robot
volume goes to negative clearance, THE OBSTACLE RESPONSIBLE turns solid red and
stays that way — so the video says not just "a collision happened" but which of
the obstacles the robot hit. Attribution comes from the argmin of the same
masked distance matrix the harness uses for clearance, so the highlighted
obstacle is by construction the one that set the reported clearance.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.render_simulate_attack \
        --fuzz-result fuzz/siren/experiment/case/fuzz_<name>.json --pick farthest
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, float).reshape(3)
    return f


def pick_candidate(res, mode, pick_seed=0):
    """Choose which attack found by the SEARCH to render.

    Reads the per-evaluation log written by stage3_fuzz -- every location tried,
    with `is_attack` already decided against the target's own kind (contact on
    leg 2 for INSERTION, leg 1 for MODIFICATION). The older renderer read a
    `hits` list that stage 3b no longer emits.

    modes:
        random    uniform over distinct attacks, seeded by --pick-seed so a
                  "random" pick is still reproducible
        farthest  furthest from the planted G1' -- the least obvious attack
        nearest   closest to it
        deepest   deepest contact
    """
    uniq = {}
    for row in res["results"]:
        for e in row["evaluations"]:
            if e.get("stage") != "evaluated" or not e.get("is_attack"):
                continue
            k = tuple(np.round(e["cand"], 9))
            rec = uniq.setdefault(k, {"dist": e.get("dist_to_truth", np.nan),
                                      "clear": e.get("min_clearance", 0.0),
                                      "round": e.get("round"),
                                      "eval": e.get("eval"),
                                      "id": e.get("id"),
                                      "pickers": set()})
            rec["pickers"].add(row["picker"])
            rec["clear"] = min(rec["clear"], e.get("min_clearance", 0.0))
    if not uniq:
        return None, None, 0
    items = sorted(uniq.items())          # sort first: dict order must not
                                          # decide what "random" means
    if mode == "random":
        rng = np.random.RandomState(pick_seed)
        k, e = items[int(rng.randint(len(items)))]
    elif mode == "farthest":
        k, e = max(items, key=lambda kv: kv[1]["dist"])
    elif mode == "nearest":
        k, e = min(items, key=lambda kv: kv[1]["dist"])
    else:
        k, e = min(items, key=lambda kv: kv[1]["clear"])
    return np.asarray(k, float), e, len(uniq)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fuzz-result", required=True)
    p.add_argument("--targets-dir", default="fuzz/siren/pipeline/targets")
    p.add_argument("--pick", default="random",
                   choices=["random", "farthest", "nearest", "deepest"])
    p.add_argument("--pick-seed", type=int, default=0,
                   help="makes --pick random reproducible")
    p.add_argument("--out", default=None)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=55.0)
    p.add_argument("--elevation", type=float, default=-14.0)
    p.add_argument("--distance", type=float, default=1.05)
    p.add_argument("--stride", type=int, default=2,
                   help="render every Nth control step")
    a = p.parse_args(argv)

    import mujoco
    import cv2
    from spark_utils import compute_masked_distance_matrix
    from ..world.run import World, BRAKE_SCALE
    from ..world.sim import probe
    from ..world import derived
    from ..scenario.targets import load_target

    res = json.loads(Path(a.fuzz_result).read_text())
    tpath = f"{a.targets_dir}/{res['target']}.json"
    w, tgt = load_target(tpath)
    sc = w.scene()
    h = w.harness
    spec = h.spec
    steps = tgt["max_steps"]

    cand, meta, n_uniq = pick_candidate(res, a.pick, a.pick_seed)
    if cand is None:
        print("no attacks in this fuzz result — nothing to render")
        return 1
    truth = np.asarray(tgt["G1_prime_truth"], float)
    G0 = np.asarray(tgt["G0_commanded"], float)
    G1 = np.asarray(tgt["G1"], float)

    print(f"target {res['target']}   filter={tgt['algo']}")
    print(f"  picked ({a.pick} of {n_uniq}): {np.round(cand,4)}  "
          f"{meta['dist']:.4f} m from planted G1' {np.round(truth,4)}")
    print(f"  round={meta['round']} eval={meta['eval']}  found by {sorted(meta['pickers'])}  "
          f"recorded clearance {meta['clear']:+.6f}", flush=True)

    si = h.algo.safe_controller.safe_algo.safety_index
    env_mask = np.asarray(si.env_collision_mask, bool)
    n_obs = si.num_obstacle_vol
    # CollisionVol keys are an IntEnum, so f-string formatting silently renders
    # them as bare integers ("15") instead of the link name. Take .name here.
    vol_names = [getattr(k, "name", str(k)) for k in h.robot_cfg.CollisionVol]
    interval = float(h.env.agent.dt * h.env.agent.control_decimation)
    base = sc.base_frame

    ag = h.env.agent
    ag.model.vis.global_.offwidth = max(a.width, ag.model.vis.global_.offwidth)
    ag.model.vis.global_.offheight = max(a.height, ag.model.vis.global_.offheight)
    renderer = mujoco.Renderer(ag.model, a.height, a.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(ag.model, cam)

    def world_of(xyz_base):
        return (base @ _frame(xyz_base))[:3, 3]

    # frame on the goals near the action; the inserted goal can sit far out and
    # would drag the look-at off the arm entirely
    anchors = np.array([world_of(G0), world_of(G1), world_of(truth)])
    cam.lookat[:] = anchors.mean(0)
    cam.distance, cam.elevation, cam.azimuth = a.distance, a.elevation, a.azimuth

    out = a.out or str(Path(a.fuzz_result).parent /
                       f"found_attack_{res['target']}_{a.pick}.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                         (a.width, a.height))

    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule([G0, cand, G1])
    u, ai = h.algo.act(af, ti)

    LEGS = ["leg 0:  home -> G0        (setup)",
            "leg 1:  G0 -> G1'         (reaching the inserted goal)",
            "leg 2:  G1' -> G1         (back to the legitimate goal)"]
    F = cv2.FONT_HERSHEY_SIMPLEX
    log, prev_clear, hit_obs, hit_step, hit_vol = [], None, None, None, None

    for t in range(steps):
        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)

        dmat, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=ti["obstacle"]["frames_world"],
            geom_list_2=ti["obstacle"]["geom"])
        guarded = np.where(env_mask, np.asarray(dmat, float), np.inf)
        vi, oi = divmod(int(np.argmin(guarded)), n_obs)
        clear = float(guarded[vi, oi])
        v_close = 0.0 if prev_clear is None else (prev_clear - clear) / interval
        prev_clear = clear

        raw = probe.read_raw(h)
        phi, C_phi, engaged = np.nan, np.inf, False
        if raw:
            d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"],
                                 raw["phi_mask"], raw["u_lim"],
                                 demand_shape=spec.demand_shape,
                                 eta=spec.eta, lam=spec.lam, exact=False)
            phi, C_phi = d.get("phi", np.nan), d.get("C_phi", np.inf)
            engaged = bool(d.get("engaged", False))
        brake = (clear - v_close ** 2 / (2 * BRAKE_SCALE * C_phi)
                 if (v_close > 0 and np.isfinite(C_phi) and C_phi > 0) else np.inf)
        wp = int(getattr(h.env.task, "wp_idx", 0))

        if clear < 0.0 and hit_obs is None:
            hit_obs, hit_step, hit_vol = oi, t, vol_names[vi]

        log.append({"t": t, "wp": wp, "clear": clear, "vol": vol_names[vi],
                    "obs": int(oi), "phi": float(phi), "v_close": float(v_close),
                    "brake": float(brake), "engaged": engaged})

        collided = hit_obs is not None
        render_now = (t % a.stride == 0) or collided or engaged
        if render_now:
            renderer.update_scene(ag.data, camera=cam)
            s = renderer.scene

            def add(pos, r, rgba):
                if s.ngeom >= s.maxgeom:
                    return
                mujoco.mjv_initGeom(
                    s.geoms[s.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([r, r, r], float),
                    np.asarray(pos, float).reshape(3), np.eye(3).flatten(),
                    np.asarray(rgba, np.float32))
                s.ngeom += 1

            for j, of in enumerate(ti["obstacle"]["frames_world"]):
                if collided and j == hit_obs:
                    add(np.asarray(of)[:3, 3], 0.052, (1.00, 0.00, 0.00, 1.00))
                else:
                    add(np.asarray(of)[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.28))
            add(world_of(G0), 0.024, (0.20, 0.45, 1.00, 0.95))
            add(world_of(cand), 0.030, (1.00, 0.85, 0.10, 0.95))
            add(world_of(G1), 0.030, (0.10, 0.90, 0.10, 0.95))
            add(world_of(truth), 0.014, (0.65, 0.65, 0.20, 0.45))
            add(h.env.task.robot_frames_world[vi][:3, 3], 0.045,
                (1.0, 1.0, 1.0, 0.30 if clear > 0.01 else 0.75))

            img = np.ascontiguousarray(renderer.render())
            if collided:
                cv2.rectangle(img, (0, 0), (a.width - 1, a.height - 1),
                              (0, 0, 255), 12)
            elif engaged:
                cv2.rectangle(img, (0, 0), (a.width - 1, a.height - 1),
                              (255, 40, 40), 6)

            def txt(y, s_, col=(255, 255, 255), sc_=0.6, th=2):
                cv2.putText(img, s_, (16, y), F, sc_, (0, 0, 0), th + 3, cv2.LINE_AA)
                cv2.putText(img, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

            txt(34, f"step {t:4d}    {LEGS[min(wp, 2)]}", sc_=0.66)
            txt(62, f"clearance {clear:+.5f} m   ({vol_names[vi]} vs obstacle {oi})",
                sc_=0.56)
            bs = f"{brake:+.4f}" if np.isfinite(brake) else "  --"
            txt(88, f"phi {phi:+.4f}   closing {v_close:+.3f} m/s   "
                    f"braking margin {bs}", sc_=0.56)
            # ASCII only: cv2.putText has no glyph for en/em dashes and draws "?"
            txt(114, "FILTER ENGAGED - enforcing a constraint" if engaged
                     else "filter idle - no constraint active",
                (60, 90, 255) if engaged else (190, 190, 190), 0.56)
            txt(146, f"inserted goal {np.round(cand,3)}   "
                     f"{meta['dist']:.3f} m from the planted G1'", sc_=0.52)
            txt(170, "blue = G0   yellow = inserted G1'   green = G1   "
                     "faint = planted G1'", sc_=0.5)
            if collided:
                txt(212, f"COLLISION at step {hit_step}", (0, 0, 255), 1.0, 3)
                txt(246, f"{hit_vol} vs obstacle {hit_obs} "
                         f"(solid red)", (0, 0, 255), 0.62, 2)

            vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if hit_obs is not None:
            # Orbit the contact rather than holding one frame. The arm that made
            # the contact is between the camera and the obstacle it hit, so from
            # the trajectory's own viewpoint the solid-red sphere is largely
            # occluded -- exactly at the moment it needs to be legible. Orbiting
            # guarantees several seconds of unoccluded angles.
            contact_pos = np.asarray(
                ti["obstacle"]["frames_world"][hit_obs])[:3, 3]
            hold_frames = a.fps * 5
            for i in range(hold_frames):
                cam.lookat[:] = 0.5 * contact_pos + 0.5 * anchors.mean(0)
                cam.azimuth = a.azimuth + 360.0 * i / hold_frames
                cam.distance = a.distance * 0.8
                renderer.update_scene(ag.data, camera=cam)
                s = renderer.scene
                for j, of in enumerate(ti["obstacle"]["frames_world"]):
                    pos = np.asarray(of)[:3, 3]
                    if j == hit_obs:
                        add(pos, 0.052, (1.00, 0.00, 0.00, 1.00))
                    else:
                        add(pos, 0.05, (0.85, 0.15, 0.15, 0.28))
                add(world_of(G0), 0.024, (0.20, 0.45, 1.00, 0.95))
                add(world_of(cand), 0.030, (1.00, 0.85, 0.10, 0.95))
                add(world_of(G1), 0.030, (0.10, 0.90, 0.10, 0.95))
                add(h.env.task.robot_frames_world[vi][:3, 3], 0.045,
                    (1.0, 1.0, 1.0, 0.75))
                fr = np.ascontiguousarray(renderer.render())
                cv2.rectangle(fr, (0, 0), (a.width - 1, a.height - 1),
                              (0, 0, 255), 12)

                def ftxt(y, s_, col=(255, 255, 255), sc_=0.6, th=2):
                    cv2.putText(fr, s_, (16, y), F, sc_, (0, 0, 0),
                                th + 3, cv2.LINE_AA)
                    cv2.putText(fr, s_, (16, y), F, sc_, col, th, cv2.LINE_AA)

                ftxt(40, f"COLLISION at step {hit_step}, on leg 2",
                     (0, 0, 255), 0.95, 3)
                ftxt(76, f"{hit_vol}  vs  obstacle {hit_obs} (solid red)",
                     (0, 0, 255), 0.62)
                ftxt(104, f"clearance {clear:+.6f} m", (255, 255, 255), 0.58)
                ftxt(132, f"inserted goal {np.round(cand,3)}, "
                          f"{meta['dist']:.3f} m from the planted G1'",
                     (255, 255, 255), 0.52)
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            break
        if h.env.task.reached_final:
            break

    vw.release()
    dump = str(Path(out).with_suffix(".json"))
    json.dump({"target": res["target"], "candidate": cand.tolist(),
               "pick_mode": a.pick, "dist_to_truth": meta["dist"],
               "round": meta["round"], "eval": meta["eval"],
               "eval_id": meta["id"], "pickers": sorted(meta["pickers"]),
               "contact_step": hit_step, "contact_obstacle": hit_obs,
               "contact_volume": hit_vol, "n_steps": len(log),
               "log": log}, open(dump, "w"), indent=2, default=float)

    if hit_obs is None:
        print(f"\n  NO CONTACT in {len(log)} steps — the attack did not reproduce")
    else:
        legs = [r["wp"] for r in log if r["clear"] < 0]
        print(f"\n  CONTACT at step {hit_step} on leg {legs[0]}: "
              f"{hit_vol} vs obstacle {hit_obs}, "
              f"clearance {log[-1]['clear']:+.6f}")
    print(f"wrote {out}")
    print(f"wrote {dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
