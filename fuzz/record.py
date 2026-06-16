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


def record_trial(seed, schedule_base, out_path, safe_algo="rssa",
                 max_steps=400, width=1280, height=720, fps=30,
                 azimuth=135.0, elevation=-20.0, distance=1.2, lookat=None):
    """Run one trial and write an MP4. schedule_base: list of 3-D base-frame goals
    (last = legitimate G1). Returns (out_path, n_frames, label)."""
    from .metrics import StepRecord, classify_trial

    cfg = build_single_arm_config(seed=seed, safe_algo=safe_algo, max_steps=max_steps)
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

    frames, records = [], []
    from spark_utils import compute_masked_distance_matrix

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

        # --- render this frame ---
        mujoco.mj_forward(agent.model, agent.data)
        renderer.update_scene(agent.data, camera=cam)
        scene = renderer.scene
        # obstacles (red, semi-transparent)
        for of, og in zip(obs_frames, obs_geom):
            rad = og.attributes.get("radius", 0.05)
            _add_sphere(scene, of[:3, 3], rad, (0.85, 0.15, 0.15, 0.55))
        # active right goal (green) and final G1 (cyan, faint)
        cur_world = (task.robot_base_frame @ _xyz_to_frame(task._current_goal_base()))[:3, 3]
        _add_sphere(scene, G1_world, 0.035, (0.1, 0.8, 0.9, 0.35))
        _add_sphere(scene, cur_world, 0.03, (0.1, 0.9, 0.1, 0.85))
        # end-effector (yellow)
        ee = task.robot_frames_world[R_ee, :3, 3]
        _add_sphere(scene, ee, 0.022, (1.0, 0.9, 0.1, 0.95))

        img = renderer.render()
        # HUD text
        txt = f"seed {seed}  step {step+1}/{max_steps}  d(EE->G1)={task.dist_to_final:.3f}  wp {task.wp_idx}"
        cv2.putText(img, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        frames.append(img)

        if task.reached_final:
            break

    outcome = classify_trial(records, schedule=schedule_base)
    # stamp the verdict on the last 30 frames so the ending is self-explanatory
    for img in frames[-30:]:
        # img is RGB here (converted to BGR at write time): green reached / red trapped
        cv2.putText(img, outcome.label, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (60, 200, 60) if outcome.reached_final else (230, 50, 50), 3, cv2.LINE_AA)

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
    ap.add_argument("--out", default="/tmp/trial.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=1.2)
    args = ap.parse_args()

    # need G1 to resolve the schedule -> build a throwaway harness for the scene
    cfg = build_single_arm_config(seed=args.seed, safe_algo=args.safe_algo, max_steps=args.max_steps)
    G1 = SingleArmHarness(cfg).scene_info()["G1_base"]
    schedule = _parse_schedule(args.schedule, G1)
    print(f"[record] seed={args.seed} schedule={[np.round(w,3).tolist() for w in schedule]}")

    out, n, label = record_trial(args.seed, schedule, args.out, safe_algo=args.safe_algo,
                                 max_steps=args.max_steps, fps=args.fps,
                                 azimuth=args.azimuth, elevation=args.elevation, distance=args.distance)
    print(f"[record] wrote {out}  frames={n}  outcome={label}")


if __name__ == "__main__":
    main()
