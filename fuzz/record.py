"""
Offscreen video recorder for goal-insertion trials.

Headless (no mjpython, no GLFW window): it builds its OWN mujoco.Renderer against
the agent's model/data and draws the robot plus overlay markers (obstacles, the
active goal, the end-effector) each step, then encodes an MP4 with cv2.

Why a separate renderer: in headless mode the agent's own renderer is None, and
the pipeline's overlay path targets the interactive viewer's user_scn (which the
offscreen renderer doesn't use). So we add overlay geoms ourselves AFTER
update_scene(), every frame.

Run (inside the SPARK conda env, with the single-thread env vars):

    export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    python -m fuzz.record --seed 0 --schedule G1 --out /tmp/deadlock.mp4
    python -m fuzz.record --seed 4 --schedule "0.314,-0.191,0.065 ; G1" --out /tmp/attack.mp4
"""

import argparse

import numpy as np
import mujoco
import cv2

from .config import build_single_arm_config
from .harness import SingleArmHarness


def _xyz_to_frame(xyz):
    f = np.eye(4)
    f[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return f


def _add_sphere(scene, pos, radius, rgba):
    """Append one overlay sphere to a mujoco scene (after update_scene)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=float),
        np.asarray(pos, dtype=float).reshape(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _project(scene, model, p, W, H):
    """Project a 3-D world point to pixel (u, v) for the scene's current camera.
    Returns None if behind the camera."""
    cam = scene.camera[0]
    pos = np.array(cam.pos, dtype=float)
    fwd = np.array(cam.forward, dtype=float); fwd /= np.linalg.norm(fwd)
    up = np.array(cam.up, dtype=float)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = np.asarray(p, dtype=float) - pos
    zc = float(rel @ fwd)
    if zc <= 1e-6:
        return None
    xc = float(rel @ right); yc = float(rel @ up)
    tan_v = np.tan(np.radians(float(model.vis.global_.fovy)) / 2.0)
    ndc_x = xc / (zc * tan_v * (W / H)); ndc_y = yc / (zc * tan_v)
    return (int((ndc_x * 0.5 + 0.5) * W), int((1 - (ndc_y * 0.5 + 0.5)) * H))


def _label(img, uv, text, rgb, W, H):
    """Draw a colored text label with a white outline at a projected point."""
    if uv is None:
        return
    u, v = uv
    if not (-80 <= u < W + 80 and -40 <= v < H + 40):
        return
    cv2.circle(img, (u, v), 5, rgb, -1)
    cv2.putText(img, text, (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, text, (u + 10, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.62, rgb, 2, cv2.LINE_AA)


def _draw_prompt(img, base, inject, W):
    """Top banner showing the natural-language command. The attacker-injected part
    (`inject`) is drawn in red; the legitimate command (`base`) in white."""
    if not base:
        return
    F = cv2.FONT_HERSHEY_SIMPLEX
    bh = 50
    ov = img.copy(); cv2.rectangle(ov, (0, 0), (W, bh), (18, 18, 24), -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    x, y = 18, 33
    cv2.putText(img, "COMMAND:", (x, y), F, 0.6, (150, 165, 190), 2, cv2.LINE_AA)
    x += cv2.getTextSize("COMMAND:", F, 0.6, 2)[0][0] + 14
    if inject:
        t = inject + ", then "
        cv2.putText(img, t, (x, y), F, 0.62, (255, 80, 80), 2, cv2.LINE_AA)
        x += cv2.getTextSize(t, F, 0.62, 2)[0][0]
        cv2.putText(img, base, (x, y), F, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(img, "red = attacker-injected step (benign, passes validation)",
                    (18, bh + 20), F, 0.46, (255, 90, 90), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, base, (x, y), F, 0.62, (245, 245, 245), 2, cv2.LINE_AA)


def record_trial(seed, schedule_base, out_path, safe_algo="rssa",
                 max_steps=400, width=1280, height=720, fps=30,
                 azimuth=135.0, elevation=-20.0, distance=1.2, lookat=None,
                 d_min_env=None, prompt_base="", prompt_inject=""):
    """Run one trial and write an MP4. schedule_base: list of 3-D base-frame goals
    (last = legitimate G1). Returns (out_path, n_frames, label)."""
    from .metrics import StepRecord, classify_trial

    cfg = build_single_arm_config(seed=seed, safe_algo=safe_algo, max_steps=max_steps,
                                  d_min_env=d_min_env)
    h = SingleArmHarness(cfg)
    sc = h.scene_info()
    agent = h.env.agent

    # enlarge the offscreen framebuffer (default is 640x480) so we can render HD
    agent.model.vis.global_.offwidth = max(width, agent.model.vis.global_.offwidth)
    agent.model.vis.global_.offheight = max(height, agent.model.vis.global_.offheight)
    renderer = mujoco.Renderer(agent.model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(agent.model, cam)
    obs = sc["obstacles_world"]
    G1_world = (sc["base_frame"] @ _xyz_to_frame(sc["G1_base"]))[:3, 3]
    if lookat is None:
        lookat = obs[:, :3, 3].mean(0) if len(obs) else G1_world
    cam.lookat[:] = np.asarray(lookat, dtype=float)
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth

    R_ee = h.robot_cfg.Frames.R_ee

    # reset + program the schedule (same as harness.run_trial, but we render too)
    agent_feedback, task_info = h._reset_scene()
    h.env.task.set_goal_schedule(schedule_base)
    u_safe, action_info = h.algo.act(agent_feedback, task_info)

    # fixed labelled markers: dose = start (G0), barcode scanner = inserted goals
    # (all but the last), IV = the legitimate goal (G1).
    base = sc["base_frame"]
    G0_world = (base @ _xyz_to_frame(sc["G0_base"]))[:3, 3]
    inserted_world = [(base @ _xyz_to_frame(g))[:3, 3] for g in schedule_base[:-1]]

    frames, records = [], []
    from spark_utils import compute_masked_distance_matrix

    collided_at = None
    tail_after_event = 25   # keep rendering a few frames past a collision, then stop
    for step in range(max_steps):
        agent_feedback, task_info = h.env.step(u_safe, action_info)
        u_safe, action_info = h.algo.act(agent_feedback, task_info)
        task = h.env.task

        # --- collision / slack bookkeeping (same as the harness) ---
        obs_frames = task_info["obstacle"]["frames_world"]
        obs_geom = task_info["obstacle"]["geom"]
        if len(obs_frames) > 0:
            dmat, _ = compute_masked_distance_matrix(
                frame_list_1=task.robot_frames_world,
                geom_list_1=h.robot_cfg.CollisionVol.values(),
                frame_list_2=obs_frames, geom_list_2=obs_geom)
            min_dist_env = float(dmat.min()) if dmat is not None else np.inf
        else:
            min_dist_env = np.inf
        viol = action_info.get("violation", None)
        peak_slack = float(np.max(viol)) if (viol is not None and np.size(viol) > 0) else 0.0
        records.append(StepRecord(step=step, wp_idx=task.wp_idx, dist_final=task.dist_to_final,
                                  reached_final=task.reached_final, min_dist_env=min_dist_env,
                                  peak_slack=peak_slack,
                                  trigger_safe=bool(action_info.get("trigger_safe", False)),
                                  collided=(min_dist_env < 0.0)))

        in_collision = min_dist_env < 0.0
        if in_collision and collided_at is None:
            collided_at = step
        # per-obstacle min distance -> only the obstacle(s) actually penetrated turn solid
        per_obs = dmat.min(axis=0) if (len(obs_frames) > 0 and dmat is not None) else None

        # --- render this frame ---
        mujoco.mj_forward(agent.model, agent.data)
        renderer.update_scene(agent.data, camera=cam)
        scene = renderer.scene
        # obstacles ("Patient & equipment"): dim red; ONLY the colliding one turns solid
        for j, (of, og) in enumerate(zip(obs_frames, obs_geom)):
            rad = og.attributes.get("radius", 0.05)
            colliding = per_obs is not None and j < len(per_obs) and per_obs[j] < 0.0
            rgba = (1.0, 0.05, 0.05, 0.98) if colliding else (0.85, 0.15, 0.15, 0.45)
            _add_sphere(scene, of[:3, 3], rad, rgba)
        # labelled goal markers
        _add_sphere(scene, G0_world, 0.028, (0.20, 0.45, 1.0, 0.90))     # dose (start)   - blue
        for gp in inserted_world:
            _add_sphere(scene, gp, 0.028, (0.95, 0.55, 0.10, 0.92))      # barcode scanner - orange
        _add_sphere(scene, G1_world, 0.034, (0.10, 0.85, 0.10, 0.95))    # IV (goal)       - green
        ee = task.robot_frames_world[R_ee, :3, 3]
        _add_sphere(scene, ee, 0.022, (1.0, 0.9, 0.1, 0.95))            # hand (syringe)  - yellow

        img = renderer.render()
        W, H = width, height
        # prompt banner (top): the command, with the injected step in red
        _draw_prompt(img, prompt_base, prompt_inject, W)
        # project the fixed 3-D markers to screen and label them
        _label(img, _project(scene, agent.model, G0_world, W, H), "dose", (40, 110, 230), W, H)
        for gp in inserted_world:
            _label(img, _project(scene, agent.model, gp, W, H), "barcode scanner", (240, 150, 20), W, H)
        _label(img, _project(scene, agent.model, G1_world, W, H), "IV", (40, 190, 60), W, H)
        # live collision flag (below the prompt banner)
        if in_collision:
            cv2.putText(img, "COLLISION", (16, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (230, 40, 40), 3, cv2.LINE_AA)
        # step counter (bottom-right) + obstacle legend (bottom-left)
        cv2.putText(img, f"{step+1}/{max_steps}", (W - 116, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.circle(img, (26, H - 22), 9, (230, 40, 40), -1)
        for col, th in [((255, 255, 255), 4), ((20, 20, 20), 1)]:
            cv2.putText(img, "Patient & other equipment (safety filter must avoid)", (44, H - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, th, cv2.LINE_AA)
        frames.append(img)

        if task.reached_final:
            break
        if collided_at is not None and step - collided_at >= tail_after_event:
            break

    outcome = classify_trial(records, schedule=schedule_base)
    # stamp the verdict on the last 30 frames so the ending is self-explanatory.
    # Color by the LABEL (green = REACHED/good; red = COLLISION/DEADLOCK/TIMEOUT/bad) --
    # NOT by reached_final, since a COLLISION can also happen to reach the goal.
    vcol = (60, 200, 60) if outcome.label == "REACHED" else (230, 50, 50)
    for img in frames[-30:]:
        cv2.putText(img, outcome.label, (16, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.9, vcol, 3, cv2.LINE_AA)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for img in frames:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    return out_path, len(frames), outcome.label


def _parse_schedule(s, G1):
    """'G1' -> [G1]; '0.3,-0.2,0.1 ; G1' -> [[...], G1]."""
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if tok.upper() == "G1":
            out.append(np.asarray(G1, dtype=float))
        else:
            out.append(np.array([float(x) for x in tok.split(",")], dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser(description="Record an MP4 of a goal-insertion trial")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safe-algo", default="rssa")
    ap.add_argument("--schedule", default="G1",
                    help="';'-separated base-frame goals; 'G1' = benchmark goal. "
                         "e.g. '0.314,-0.191,0.065 ; G1'")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--d-min", type=float, default=None,
                    help="override safety-index keep-out distance (e.g. 0.02 for a feasible filter)")
    ap.add_argument("--out", default="/tmp/trial.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=1.2)
    ap.add_argument("--prompt", default="", help="legitimate command shown in the top banner")
    ap.add_argument("--inject", default="", help="attacker-injected step (shown in red before the command)")
    args = ap.parse_args()

    # need G1 to resolve the schedule -> build a throwaway harness for the scene
    cfg = build_single_arm_config(seed=args.seed, safe_algo=args.safe_algo,
                                  max_steps=args.max_steps, d_min_env=args.d_min)
    G1 = SingleArmHarness(cfg).scene_info()["G1_base"]
    schedule = _parse_schedule(args.schedule, G1)
    print(f"[record] seed={args.seed} schedule={[np.round(w,3).tolist() for w in schedule]}")

    out, n, label = record_trial(args.seed, schedule, args.out, safe_algo=args.safe_algo,
                                 max_steps=args.max_steps, fps=args.fps, d_min_env=args.d_min,
                                 azimuth=args.azimuth, elevation=args.elevation, distance=args.distance,
                                 prompt_base=args.prompt, prompt_inject=args.inject)
    print(f"[record] wrote {out}  frames={n}  outcome={label}")


if __name__ == "__main__":
    main()
