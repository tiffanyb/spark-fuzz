"""
Visualize a goal-insertion search: render the MuJoCo scene (robot at G0, the
obstacles, the start G0, the legitimate goal G1) with EVERY tried inserted goal
G1' overlaid as a small sphere, colored by the attack outcome.

  G0        = blue          (start: right-hand position)
  G1        = green         (legitimate goal)
  obstacles = translucent red
  G1' tried = REACHED gray / TIMEOUT orange / COLLISION red / DEADLOCK magenta

Reusable for any seed / controller / d_min:
    python -m fuzz.plot_candidates --seed 17 --d-min 0.02 --n-candidates 120 --out /tmp/seed17.png

It re-runs a (random by default) admissible search to collect the candidates and
their outcomes, so the picture reflects exactly what the search explored.
"""

import argparse

import numpy as np
import mujoco
import cv2

from .config import build_single_arm_config
from .harness import SingleArmHarness
from .fuzzer import instrument_infeasibility, _trial
from .admissibility import sample_admissible
from .record import _add_sphere, _xyz_to_frame


_OUTCOME_RGBA = {
    "REACHED":   (0.62, 0.62, 0.62, 0.85),
    "TIMEOUT":   (1.00, 0.60, 0.10, 0.95),
    "COLLISION": (1.00, 0.10, 0.10, 0.98),
    "DEADLOCK":  (0.80, 0.10, 0.90, 0.98),
}


def load_candidates_from_json(paths):
    """Load already-simulated candidates from one or more run_search JSON files
    (no re-simulation). Returns [(G1'_base, label, infeasible), ...]."""
    import json
    out = []
    for p in paths:
        d = json.load(open(p))
        for r in d.get("all_results", []):
            if r is None:
                continue
            out.append((np.asarray(r["candidate"], dtype=float),
                        r["label"], r.get("infeasible_QP", 0)))
    return out


def collect_candidates(harness, scene, n_candidates, max_steps, search_seed=0, verbose=True):
    """Run the random-admissible search, returning the list of
    (G1'_base, outcome_label, infeasible_QP) for every screened candidate."""
    rng = np.random.RandomState(search_seed)
    G1 = scene["G1_base"]
    out = []
    tried = 0
    attempts = 0
    while tried < n_candidates and attempts < n_candidates * 20:
        attempts += 1
        cand = sample_admissible(rng, bounds=scene["bounds"], base_frame=scene["base_frame"],
                                 obstacles_world=scene["obstacles_world"], keepout=scene["keepout"])
        if cand is None:
            continue
        screen, _ = _trial(harness, [cand], max_steps)
        if not screen.reached_final:
            continue                     # not individually reachable -> not a valid inserted goal
        outcome, infeas = _trial(harness, [cand, G1], max_steps)
        out.append((cand, outcome.label, infeas))
        tried += 1
        if verbose and tried % 10 == 0:
            print(f"  collected {tried}/{n_candidates}", flush=True)
    return out


def render(harness, scene, candidates, out_path,
           width=1280, height=720, azimuth=135.0, elevation=-20.0, distance=1.3):
    agent = harness.env.agent
    base = harness.env.task.robot_base_frame
    R_ee = harness.robot_cfg.Frames.R_ee

    # robot at G0 (reset/warmed pose)
    harness._reset_scene()
    mujoco.mj_forward(agent.model, agent.data)
    obs_f = scene["obstacles_world"]

    agent.model.vis.global_.offwidth = max(width, agent.model.vis.global_.offwidth)
    agent.model.vis.global_.offheight = max(height, agent.model.vis.global_.offheight)
    renderer = mujoco.Renderer(agent.model, height, width)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(agent.model, cam)
    cam.lookat[:] = (obs_f[:, :3, 3].mean(0) if len(obs_f)
                     else (base @ _xyz_to_frame(scene["G1_base"]))[:3, 3])
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth

    renderer.update_scene(agent.data, camera=cam)
    sc = renderer.scene

    def w(xyz_base):
        return (base @ _xyz_to_frame(xyz_base))[:3, 3]

    for of in obs_f:
        _add_sphere(sc, of[:3, 3], 0.05, (0.85, 0.15, 0.15, 0.45))
    # all tried G1'
    counts = {}
    for cand, label, _inf in candidates:
        counts[label] = counts.get(label, 0) + 1
        _add_sphere(sc, w(cand), 0.012, _OUTCOME_RGBA.get(label, (0.5, 0.5, 0.5, 0.8)))
    # G0 (start EE) and G1 (goal) on top, larger
    _add_sphere(sc, w(scene["G0_base"]), 0.030, (0.20, 0.45, 1.00, 0.95))   # blue
    _add_sphere(sc, w(scene["G1_base"]), 0.040, (0.10, 0.90, 0.10, 0.97))   # green

    img = renderer.render()

    # HUD: title + legend (BGR-at-write, so pass RGB)
    cv2.putText(img, f"tried G1' = {len(candidates)}   (blue=G0  green=G1)", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    y = 58
    for label in ("REACHED", "TIMEOUT", "COLLISION", "DEADLOCK"):
        rgba = _OUTCOME_RGBA[label]
        col = tuple(int(255 * c) for c in rgba[:3])
        cv2.putText(img, f"{label}: {counts.get(label, 0)}", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        y += 26

    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[plot] wrote {out_path}  candidates={len(candidates)}  outcomes={counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description="Plot all tried G1' over the scene")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--safe-algo", default="ssa")
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument("--n-candidates", type=int, default=120)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--search-seed", type=int, default=0)
    ap.add_argument("--from-json", default=None,
                    help="comma-separated run_search JSON(s) to load candidates from "
                         "(skips re-simulation -> renders in seconds)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated outcome labels to drop before rendering "
                         "(e.g. TIMEOUT to hide soft failures)")
    ap.add_argument("--out", default="/tmp/candidates.png")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=1.3)
    a = ap.parse_args()

    cfg = build_single_arm_config(seed=a.seed, safe_algo=a.safe_algo,
                                  max_steps=a.max_steps, d_min_env=a.d_min)
    h = SingleArmHarness(cfg); instrument_infeasibility(h)
    scene = h.scene_info()
    if a.from_json:
        cands = load_candidates_from_json(a.from_json.split(","))
        print(f"[setup] seed={a.seed} loaded {len(cands)} candidates from json (no re-sim)", flush=True)
    else:
        print(f"[setup] seed={a.seed} d_min={a.d_min} collecting {a.n_candidates} candidates...", flush=True)
        cands = collect_candidates(h, scene, a.n_candidates, a.max_steps, a.search_seed)
    if a.exclude:
        drop = {s.strip().upper() for s in a.exclude.split(",") if s.strip()}
        before = len(cands)
        cands = [c for c in cands if c[1].upper() not in drop]
        print(f"[filter] dropped {before - len(cands)} candidates with label in {sorted(drop)}", flush=True)
    render(h, scene, cands, a.out, azimuth=a.azimuth, elevation=a.elevation, distance=a.distance)


if __name__ == "__main__":
    main()
