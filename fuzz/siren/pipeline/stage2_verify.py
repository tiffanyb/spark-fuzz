"""
Stage 2 -- verify each candidate attack from scratch, record its trace, render it.

Merges what used to be three steps (collect / render / trace). The merge is a
CORRECTNESS change, not a convenience one: previously the renderer ran its own
independent rollout, so on robot families where a resume drifts there was no
guarantee the rendered run was the one that passed verification. Here a single
instrumented rollout produces the verdict, the trace, and the frames.

    baseline [G0, G1]           must REACH *and* keep clearance > 0 throughout.
                                reached_final alone is not enough -- a rollout
                                can penetrate an obstacle and still arrive, and
                                measure.classify_run calls that COLLISION
                                because collision is tested before reached.
    attack [G0, G1', G1] x N    must COLLIDE or DEADLOCK on every repeat.
                                Repeat 0 is instrumented (keep_q + obstacle
                                frames); the rest are label-only.
    contact leg                 2 -> INSERTION, 1 -> MODIFICATION, 0 -> reject,
                                and must agree across all repeats.

Rendering replays the recorded trace through mj_forward -- kinematics only, no
physics and no filter -- so the video is provably the verified rollout and a
re-render at a new camera angle costs no simulation.

    python -m fuzz.siren.pipeline.stage2_verify \
        --src '/abs/stage1/control_*.json' --out-dir /abs/verified --render
"""

import argparse
import glob
import json
import os

import numpy as np


def contact_leg(steps):
    for s in steps:
        if s.clearance < 0.0:
            return int(s.wp_idx)
    return None


def instrumented_run(world, schedule, max_steps):
    """One rollout, keeping everything needed to re-render and to investigate.

    World.run already records per-step clearance/phi/mu/g/demand/brake_margin/
    engaged/deviation/gave_up and, with keep_q, the configuration. What it does
    NOT keep is the obstacle geometry per step, which dynamic-obstacle scenes
    need for replay, nor MuJoCo's qpos -- `q` is the agent's command state, and
    with use_sim_dynamics=False those are not the same object. Both are captured
    here so replay is unambiguous.
    """
    from ..world.sim import probe

    h = world.harness
    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule(schedule)
    u, ai = h.algo.act(af, ti)

    qpos, obst, ctrl = [], [], []
    for t in range(max_steps):
        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)
        qpos.append(np.asarray(h.env.agent.data.qpos, float).copy())
        ctrl.append(np.asarray(u, float).reshape(-1).copy())
        obst.append(np.asarray([np.asarray(o)[:3, 3]
                                for o in ti["obstacle"]["frames_world"]], float))
        if h.clearance(ti) < 0.0 or h.env.task.reached_final:
            break
    rec = world.run(schedule, max_steps=max_steps, exact_margin=True,
                    keep_q=True)
    return rec, np.array(qpos), np.array(obst), np.array(ctrl)


def write_trace(path, rec, qpos, obst, ctrl, stride=1):
    S = rec.steps
    sl = slice(None, None, stride)
    np.savez_compressed(
        path,
        step=np.array([s.step for s in S])[sl],
        wp_idx=np.array([s.wp_idx for s in S])[sl],
        clearance=np.array([s.clearance for s in S], float)[sl],
        phi=np.array([s.phi for s in S], float)[sl],
        mu=np.array([s.mu for s in S], float)[sl],
        g=np.array([s.g for s in S], float)[sl],
        demand=np.array([s.demand for s in S], float)[sl],
        brake_margin=np.array([s.brake_margin for s in S], float)[sl],
        engaged=np.array([s.engaged for s in S], bool)[sl],
        deviation=np.array([s.deviation for s in S], float)[sl],
        gave_up=np.array([s.gave_up for s in S], bool)[sl],
        dist_final=np.array([s.dist_final for s in S], float)[sl],
        qpos=qpos[sl], obstacles=obst[sl], u=ctrl[sl])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--trace", default="full", choices=["full", "decimated",
                                                       "none"])
    p.add_argument("--trace-stride", type=int, default=5)
    p.add_argument("--render", action="store_true", default=False)
    p.add_argument("--only", default=None)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f for f in glob.glob(a.src)
                   if "INDEX" not in os.path.basename(f)
                   and (a.only is None or a.only in f))
    print(f"{len(files)} control files\n", flush=True)

    kept, rejected = [], []
    for path in files:
        src = json.load(open(path))
        spec = real_filter(algo=src["algo"], index=src["index"],
                           d_min=src["d_min"], eta=src["eta"],
                           lam=src["lam"], k=src["k"])
        steps = src["max_steps"]
        w = World.build(seed=src["seed"], spec=spec, test_case=src["case"],
                        max_steps=steps)
        G1 = np.asarray(src["G1"], float)
        G1p = np.asarray(src["G1_prime"], float)

        for i, c in enumerate(src.get("controls", [])):
            G0 = np.asarray(c["G0"], float)
            base = w.run([G0, G1], max_steps=steps)
            base_clear = min((s.clearance for s in base.steps), default=np.inf)
            if base.label != "REACHED" or base_clear <= 0.0:
                rejected.append((os.path.basename(path), i,
                                 f"baseline {base.label} {base_clear:+.6f}"))
                continue

            rec, qpos, obst, ctrl = instrumented_run(w, [G0, G1p, G1], steps)
            labels = [rec.label]
            legs = [contact_leg(rec.steps)]
            for _ in range(a.repeats - 1):
                r2 = w.run([G0, G1p, G1], max_steps=steps)
                labels.append(r2.label)
                legs.append(contact_leg(r2.steps))
            if not all(l in ("COLLISION", "DEADLOCK") for l in labels):
                rejected.append((os.path.basename(path), i, f"attack {labels}"))
                continue
            if labels[0] == "DEADLOCK":
                kind = "INSERTION"
            elif all(x == 2 for x in legs):
                kind = "INSERTION"
            elif all(x == 1 for x in legs):
                kind = "MODIFICATION"
            else:
                rejected.append((os.path.basename(path), i, f"legs {legs}"))
                continue

            tag = (f"{src['case']}_{src['algo']}_s{src['seed']}"
                   f"_lam{src['lam']}_{kind}{len(kept)}")
            out = {"kind": kind, "case": src["case"], "algo": src["algo"],
                   "seed": src["seed"], "index": src["index"],
                   "max_steps": steps, "d_min": src["d_min"],
                   "eta": src["eta"], "lam": src["lam"], "k": src["k"],
                   "G0": G0.tolist(), "G1_prime": G1p.tolist(),
                   "G1": G1.tolist(), "bounds": src["bounds"],
                   "keepout": src["keepout"],
                   "obstacles_world": src["obstacles_world"],
                   "baseline": {"label": base.label,
                                "steps": int(base.n_steps),
                                "min_clearance": float(base_clear)},
                   "attack": {"labels": labels, "contact_legs": legs,
                              "min_clearance": float(rec.min_clearance),
                              "steps": int(rec.n_steps)},
                   "verified": True, "source_file": path}
            # trace and verdict are written BEFORE any rendering: a renderer
            # failure must not cost the verification result
            json.dump(out, open(f"{a.out_dir}/{tag}.json", "w"), indent=2,
                      default=float)
            if a.trace != "none":
                stride = a.trace_stride if a.trace == "decimated" else 1
                write_trace(f"{a.out_dir}/{tag}_trace.npz", rec, qpos, obst,
                            ctrl, stride)
            kept.append(tag)
            print(f"  {kind:<12} {src['algo']:<5} {src['case'][:26]:<26} "
                  f"s{src['seed']} G0={np.round(G0,3)} "
                  f"clear={rec.min_clearance:+.6f}  -> {tag}", flush=True)

    print(f"\n{'='*76}\n{len(kept)} verified -> {a.out_dir}")
    print(f"{len(rejected)} rejected")
    for f, i, why in rejected[:10]:
        print(f"   {f[:52]:<52} #{i}  {why}")
    json.dump({"kept": kept,
               "rejected": [{"file": f, "idx": i, "why": w}
                            for f, i, w in rejected]},
              open(f"{a.out_dir}/INDEX.json", "w"), indent=2, default=float)

    if a.render and kept:
        from .stage2_render import render_dir
        render_dir(a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
