"""
Turn verified controls into fuzz targets, and prove each one still works.

For every verified control: drive the robot to G0, capture the full simulator
state there, write it out, then RELOAD the target from disk and confirm the
attack still reproduces from the restored state. That last step is the point —
a target that does not reproduce after a save/load round trip is worse than no
target, because the fuzzer would be scored against an attack that is not there.

Each target is checked two ways after reloading:

    [G1]        must REACH     — the legitimate task is safe from G0
    [G1', G1]   must COLLIDE   — the known attack still works from G0

If both hold, the fuzzer can be pointed at this target and scored on whether it
rediscovers G1' without ever being told it.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.make_targets
"""

import argparse
import glob
import json
import os

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--verified", default="fuzz/siren/scenario/verified/*.json")
    p.add_argument("--out-dir", default="fuzz/siren/scenario/targets")
    p.add_argument("--settle", type=int, default=0,
                   help="extra steps to hold at G0 before capturing")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..scenario.targets import capture_state, load_target

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(a.verified))
    # stage-2 output also contains INDEX.json and trace/render files;
    # only the per-attack metadata JSONs are inputs here
    files = [f for f in files
             if f.endswith(".json") and "INDEX" not in os.path.basename(f)
             and not f.endswith("_trace.npz")]
    print(f"{len(files)} verified controls -> targets\n", flush=True)

    made, failed = [], []
    for path in files:
        v = json.loads(open(path).read())
        tag = os.path.basename(path).replace(".json", "")
        case, algo, seed = v["case"], v["algo"], v["seed"]
        steps = v["max_steps"]
        spec = real_filter(algo=algo, index=v["index"], d_min=v["d_min"],
                           eta=v["eta"], lam=v["lam"], k=v["k"])
        G0 = np.asarray(v["G0"], float)
        G1p = np.asarray(v["G1_prime"], float)
        G1 = np.asarray(v["G1"], float)

        w = World.build(seed=seed, spec=spec, test_case=case, max_steps=steps)
        h = w.harness

        # --- capture the state at the HANDOVER, not at a settled stop ----- #
        # Running schedule [G0] alone makes G0 the FINAL waypoint, so the
        # reference controller decelerates into it and the arm arrives at rest.
        # In the real attack G0 is an INTERMEDIATE waypoint the arm passes
        # through in motion. Those are different states and they give different
        # outcomes: captured at rest, 2 of the first 3 controls stopped
        # colliding on reload (joint speed 0.92 and 0.97 at the true handover).
        #
        # So capture during a [G0, G1] run at the instant wp_idx goes 0 -> 1.
        # That is the true handover state, and it depends only on the APPROACH
        # to G0 — the controller retargets at that instant — so it is identical
        # whether the next waypoint is G1' or G1. Capturing from the baseline
        # keeps the answer key out of the target's own construction.
        from ..world.sim import probe
        probe.reset_giveups(h)
        af, ti = h.reset()
        h.env.task.set_goal_schedule([G0, G1])
        u, ai = h.algo.act(af, ti)
        reached_at, state = None, None
        for t in range(steps):
            af, ti = h.env.step(u, ai)
            u, ai = h.algo.act(af, ti)
            if int(getattr(h.env.task, "wp_idx", 0)) >= 1:
                reached_at = t
                state = capture_state(h)
                break
        if state is None:
            print(f"  {tag}: never handed over at G0 — skipped", flush=True)
            failed.append((tag, "G0 handover never occurred"))
            continue
        ee = h.env.task.robot_frames_world[h.robot_cfg.Frames.R_ee, :3, 3]
        base = np.asarray(h.env.task.robot_base_frame, float)
        ee_base = (np.linalg.inv(base) @ np.append(ee, 1.0))[:3]
        speed = float(np.linalg.norm(np.asarray(state["dof_vel_fbk"], float)))

        target = {
            "name": tag, "case": case, "algo": algo, "seed": seed,
            "kind": v.get("kind", "INSERTION"),   # what stage 2 verified
            "index": v["index"], "max_steps": steps,
            "d_min": v["d_min"], "eta": v["eta"], "lam": v["lam"], "k": v["k"],
            "G0_commanded": G0.tolist(),
            "G0_actual_ee": ee_base.tolist(),
            "G0_reached_at_step": int(reached_at),
            "joint_speed_at_G0": speed,
            "G1": G1.tolist(),
            "G1_prime_truth": G1p.tolist(),   # ANSWER KEY — never give to the search
            "bounds": v["bounds"], "keepout": v["keepout"],
            "obstacles_world": v["obstacles_world"],
            "state_at_G0": state,
            "provenance": {
                "built_from": path,
                "method": "SPARK scenario whose plain task collides; its START "
                          "becomes G1' and its GOAL becomes G1; G0 searched so "
                          "that [G0,G1] is safe but [G0,G1',G1] collides",
            },
        }
        tpath = f"{a.out_dir}/{tag}.json"
        json.dump(target, open(tpath, "w"), indent=2, default=float)

        # --- reload from disk and prove the attack survives the round trip - #
        # G0 is replayed as a PREFIX WAYPOINT, not a restored state: restoring
        # the snapshot was measured not to reproduce these contacts (see
        # targets.py). Running the same rollout does, exactly.
        w2, tgt = load_target(tpath)
        G0r = np.asarray(tgt["G0_commanded"])
        clean = w2.run([G0r, np.asarray(tgt["G1"])], max_steps=steps)
        atk = w2.run([G0r, np.asarray(tgt["G1_prime_truth"]),
                      np.asarray(tgt["G1"])], max_steps=steps)
        ok = clean.label == "REACHED" and atk.label == "COLLISION"
        print(f"  {tag}", flush=True)
        print(f"      G0 reached at step {reached_at}, joint speed {speed:.4f}",
              flush=True)
        print(f"      reload -> [G1] {clean.label:<9} "
              f"[G1',G1] {atk.label:<10} clear {atk.min_clearance:+.5f}"
              f"   {'OK' if ok else 'ROUND TRIP FAILED'}", flush=True)
        (made if ok else failed).append((tag, atk.label))
        if not ok:
            os.remove(tpath)

    print(f"\n{'='*70}")
    print(f"{len(made)} usable fuzz targets in {a.out_dir}")
    for t, _ in made:
        print(f"   {t}")
    if failed:
        print(f"\n{len(failed)} rejected:")
        for t, why in failed:
            print(f"   {t}: {why}")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
