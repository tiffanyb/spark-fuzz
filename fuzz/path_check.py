"""
Configuration-space path-existence check for a goal-insertion failure.

Question it answers: when G0 -> G1' -> G1 fails (collision/deadlock), is that
because NO collision-free path exists from the post-G1' configuration to a G1
configuration (a true topological trap, case B), or because a feasible path
EXISTS but the reactive safe controller could not find it (controller myopia,
case A -- the stronger result)?

Method (faithful to the controller's own collision model -- the same
forward_kinematics + compute_masked_distance_matrix the harness/safety filter
use, so the SPARK virtual sphere obstacles are accounted for):

  q'  = the arm configuration when the controller reaches G1'  (from run_trial([G1']))
  q1  = a collision-free arm configuration at G1               (from run_trial([G1]) baseline)
  1) straight-line check: is the linear joint-space interpolation q' -> q1
     collision-free? If yes, a feasible path trivially exists -> case A.
  2) else RRT-Connect over the joints that differ, looking for a detour.
       found    -> case A (myopia): a path existed, the controller failed.
       not found-> likely case B (topological trap) -- with the usual caveat that
                   sampling planners are only probabilistically complete (they
                   cannot prove non-existence, only fail to find a path).

Clearance convention: min robot-obstacle surface distance. >0 = collision-free;
we also report whether the path clears the controller's d_min = 0.05 m margin
(a fully *safe* path, not merely collision-free).
"""

import numpy as np

from .config import build_single_arm_config
from .harness import SingleArmHarness


# ----------------------------------------------------------------------------- #
def make_clearance_fn(h, obs_frames, obs_geom):
    """Returns clearance(dof) -> min robot-obstacle distance, using the harness's
    exact collision model (forward_kinematics -> world -> masked distance matrix)."""
    from spark_utils import compute_masked_distance_matrix
    base = h.env.task.robot_base_frame
    vols = list(h.robot_cfg.CollisionVol.values())

    def clearance(dof):
        frames = h.robot_kinematics.forward_kinematics(np.asarray(dof, dtype=float))
        fw = np.stack([base @ frames[i] for i in range(len(frames))])
        dmat, _ = compute_masked_distance_matrix(fw, vols, obs_frames, obs_geom)
        return float(dmat.min()) if dmat is not None else np.inf

    return clearance


def _final_config(h, schedule, max_steps):
    """Run a trial and return (final dof config, reached flag)."""
    out = h.run_trial(schedule, max_steps=max_steps)
    dof = np.asarray(h.env.agent.dof_pos_cmd, dtype=float).copy()
    return dof, out.reached_final, out.label


# ----------------------------------------------------------------------------- #
def _edge_clear(clearance, qa, qb, plan_idx, base_cfg, margin, res=0.02):
    """Is the straight segment qa->qb (over plan_idx) clear at >= margin?
    Returns (ok, min_clear). res = joint-space step in radians."""
    qa, qb = np.asarray(qa), np.asarray(qb)
    dist = np.linalg.norm(qb[plan_idx] - qa[plan_idx])
    n = max(2, int(np.ceil(dist / res)) + 1)
    mn = np.inf
    for t in np.linspace(0.0, 1.0, n):
        cfg = base_cfg.copy()
        cfg[plan_idx] = (1 - t) * qa[plan_idx] + t * qb[plan_idx]
        c = clearance(cfg)
        mn = min(mn, c)
        if c < margin:
            return False, mn
    return True, mn


def rrt_connect(clearance, q_start, q_goal, plan_idx, margin,
                max_iter=3000, step=0.15, pad=1.0, seed=0, res=0.03):
    """Minimal RRT-Connect over plan_idx joints. Returns (found, path|None, n_iter)."""
    rng = np.random.RandomState(seed)
    base_cfg = q_start.copy()
    lo = np.minimum(q_start[plan_idx], q_goal[plan_idx]) - pad
    hi = np.maximum(q_start[plan_idx], q_goal[plan_idx]) + pad

    def clear_at(qvec):
        cfg = base_cfg.copy(); cfg[plan_idx] = qvec
        return clearance(cfg) >= margin

    def steer(q_from, q_to):
        d = q_to - q_from
        n = np.linalg.norm(d)
        return q_to if n <= step else q_from + step * d / n

    def extend(tree, parent, q_target):
        # nearest
        i = int(np.argmin([np.linalg.norm(tree[k] - q_target) for k in range(len(tree))]))
        q_new = steer(tree[i], q_target)
        ok, _ = _edge_clear(clearance, _full(tree[i]), _full(q_new), plan_idx, base_cfg, margin, res)
        if ok and clear_at(q_new):
            tree.append(q_new); parent.append(i)
            return q_new, len(tree) - 1
        return None, None

    def _full(qvec):
        cfg = base_cfg.copy(); cfg[plan_idx] = qvec; return cfg

    qs, qg = q_start[plan_idx].copy(), q_goal[plan_idx].copy()
    if not clear_at(qs) or not clear_at(qg):
        return False, None, 0
    A, pA = [qs], [-1]
    B, pB = [qg], [-1]
    for it in range(max_iter):
        q_rand = rng.uniform(lo, hi)
        q_newA, iA = extend(A, pA, q_rand)
        if q_newA is None:
            A, pA, B, pB = B, pB, A, pA
            continue
        # try to connect B toward q_newA
        q_newB, iB = extend(B, pB, q_newA)
        while q_newB is not None and np.linalg.norm(q_newB - q_newA) > 1e-6:
            nxt, iB2 = extend(B, pB, q_newA)
            if nxt is None:
                break
            if np.linalg.norm(nxt - q_newB) < 1e-9:
                break
            q_newB, iB = nxt, iB2
        if q_newB is not None and np.linalg.norm(q_newB - q_newA) <= step + 1e-6:
            return True, "connected", it + 1
        A, pA, B, pB = B, pB, A, pA
    return False, None, max_iter


# ----------------------------------------------------------------------------- #
def check(seed, safe_algo, G1p, max_steps=400, margins=(0.0, 0.05), rrt_iter=3000, d_min_env=None):
    cfg = build_single_arm_config(seed=seed, safe_algo=safe_algo, max_steps=max_steps, d_min_env=d_min_env)
    h = SingleArmHarness(cfg)
    sc = h.scene_info()
    G1 = sc["G1_base"]
    G1p = np.asarray(G1p, dtype=float)

    # static obstacles for the fixed scene
    af, ti = h._reset_scene()
    obs_frames = ti["obstacle"]["frames_world"]
    obs_geom = ti["obstacle"]["geom"]
    clearance = make_clearance_fn(h, obs_frames, obs_geom)

    # endpoint configurations
    q1, reached1, lab1 = _final_config(h, [G1], max_steps)        # config at G1 (baseline)
    qp, reachedp, labp = _final_config(h, [G1p], max_steps)       # config at G1'

    print(f"[seed {seed} | {safe_algo}] baseline [G1] -> {lab1} (reached={reached1}), "
          f"one-hop [G1'] -> {labp} (reached={reachedp})")
    print(f"  clearance at q1(G1) = {clearance(q1):+.4f}   clearance at q'(G1') = {clearance(qp):+.4f}")

    plan_idx = np.where(np.abs(qp - q1) > 1e-3)[0]
    print(f"  planning over {len(plan_idx)} joints that differ between q' and q1")

    cqp, cq1 = clearance(qp), clearance(q1)
    for margin in margins:
        # If an endpoint itself violates the margin, the margin-level path question
        # is moot (it's a property of the goal, not a trap).
        if cqp < margin or cq1 < margin:
            print(f"  margin {margin:.2f}: endpoint clearance below margin "
                  f"(q'={cqp:+.4f}, q1={cq1:+.4f}) -> not a trap, just a tight goal; skipping")
            continue
        line_ok, line_min = _edge_clear(clearance, qp, q1, plan_idx, qp.copy(), margin)
        if line_ok:
            print(f"  margin {margin:.2f}: STRAIGHT-LINE path q'->q1 is clear "
                  f"(min clearance {line_min:+.4f}) -> FEASIBLE PATH EXISTS (case A: controller myopia)")
            continue
        found, _, it = rrt_connect(clearance, qp, q1, plan_idx, margin,
                                   max_iter=rrt_iter, seed=seed)
        if found:
            print(f"  margin {margin:.2f}: straight line blocked (min {line_min:+.4f}), "
                  f"but RRT-Connect FOUND a detour in {it} iters -> FEASIBLE PATH EXISTS (case A)")
        else:
            print(f"  margin {margin:.2f}: straight line blocked AND RRT-Connect found NO path "
                  f"in {it} iters -> likely TOPOLOGICAL TRAP (case B; sampling planner cannot prove non-existence)")


def record_path(seed, safe_algo, G1p, out_path, n_interp=160, max_steps=400,
                fps=30, width=1280, height=720,
                azimuth=135.0, elevation=-20.0, distance=1.2, d_min_env=None):
    """Render the robot following the collision-free STRAIGHT-LINE joint-space path
    q'(G1') -> q1(G1) -- the feasible path the reactive controller failed to take."""
    import mujoco
    import cv2
    from .record import _add_sphere, _xyz_to_frame

    cfg = build_single_arm_config(seed=seed, safe_algo=safe_algo, max_steps=max_steps, d_min_env=d_min_env)
    h = SingleArmHarness(cfg)
    sc = h.scene_info()
    G1 = sc["G1_base"]
    G1p = np.asarray(G1p, dtype=float)

    af, ti = h._reset_scene()
    obs_frames = ti["obstacle"]["frames_world"]
    obs_geom = ti["obstacle"]["geom"]
    clearance = make_clearance_fn(h, obs_frames, obs_geom)

    q1, _, _ = _final_config(h, [G1], max_steps)
    qp, _, _ = _final_config(h, [G1p], max_steps)

    agent = h.env.agent
    base = h.env.task.robot_base_frame
    R_ee = h.robot_cfg.Frames.R_ee

    agent.model.vis.global_.offwidth = max(width, agent.model.vis.global_.offwidth)
    agent.model.vis.global_.offheight = max(height, agent.model.vis.global_.offheight)
    renderer = mujoco.Renderer(agent.model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(agent.model, cam)
    cam.lookat[:] = (obs_frames[:, :3, 3].mean(0) if len(obs_frames)
                     else (base @ _xyz_to_frame(G1))[:3, 3])
    cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth

    G1_world = (base @ _xyz_to_frame(G1))[:3, 3]
    G1p_world = (base @ _xyz_to_frame(G1p))[:3, 3]

    frames, worst = [], np.inf
    for t in np.linspace(0.0, 1.0, n_interp):
        q = (1.0 - t) * qp + t * q1                       # straight line in joint space
        agent._set_dof_pos(q)
        mujoco.mj_forward(agent.model, agent.data)
        c = clearance(q); worst = min(worst, c)
        fk = h.robot_kinematics.forward_kinematics(q)
        ee = (base @ fk[R_ee])[:3, 3]

        renderer.update_scene(agent.data, camera=cam)
        scene = renderer.scene
        for of, og in zip(obs_frames, obs_geom):
            _add_sphere(scene, of[:3, 3], og.attributes.get("radius", 0.05), (0.85, 0.15, 0.15, 0.55))
        _add_sphere(scene, G1p_world, 0.03, (0.95, 0.55, 0.1, 0.7))   # G1' = orange (start)
        _add_sphere(scene, G1_world, 0.03, (0.1, 0.9, 0.1, 0.85))     # G1  = green (end)
        _add_sphere(scene, ee, 0.022, (1.0, 0.9, 0.1, 0.95))          # EE  = yellow
        img = renderer.render()
        cv2.putText(img, f"FEASIBLE PATH  straight line G1'->G1  min clearance {worst:+.3f}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "NO COLLISION" if worst > 0 else "COLLISION", (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 200, 60) if worst > 0 else (230, 50, 50),
                    3, cv2.LINE_AA)
        frames.append(img)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for img in frames:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[record_path] wrote {out_path}  frames={len(frames)}  min_clearance={worst:+.4f}")
    return out_path, worst


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--safe-algo", required=True)
    ap.add_argument("--g1p", required=True, help="comma-separated x,y,z of G1'")
    ap.add_argument("--rrt-iter", type=int, default=3000)
    ap.add_argument("--record", default=None, help="output mp4 path; render the feasible path instead of checking")
    a = ap.parse_args()
    G1p = [float(x) for x in a.g1p.split(",")]
    if a.record:
        record_path(a.seed, a.safe_algo, G1p, a.record)
    else:
        check(a.seed, a.safe_algo, G1p, rrt_iter=a.rrt_iter)
