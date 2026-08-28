"""Stage 2c -- record the BASELINE rollout for each verified attack.

Stage 2 already runs [G0, G1] and requires it to REACH safely; that is what
makes the attack attributable to the inserted goal rather than to a world that
was doomed anyway. But it keeps no trace of that run, so there is nothing to
replay -- and an attack video on its own shows a robot crashing without
establishing that it had any business not crashing.

This re-runs the baseline instrumented and writes `<tag>_baseline_trace.npz`
next to the attack trace, so stage2_render can produce the matching
`<tag>_baseline.mp4`. Same world, same seed, same filter, same G0 -- the only
difference from the attack is the absence of G1'.

    python -m fuzz.siren.pipeline.stage2_baseline --dir <verified_dir>
"""

import argparse
import glob
import json
import os

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True,
                   help="stage-2 output dir (verified attack JSONs)")
    p.add_argument("--trace-stride", type=int, default=1)
    p.add_argument("--only", default=None)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from .stage2_verify import instrumented_run, write_trace

    metas = sorted(f for f in glob.glob(f"{a.dir}/*.json")
                   if "INDEX" not in os.path.basename(f)
                   and (a.only is None or a.only in f))
    print(f"{len(metas)} verified attacks -> baseline traces\n", flush=True)

    ok, bad = 0, []
    for m in metas:
        tag = os.path.basename(m)[:-5]
        r = json.load(open(m))
        spec = real_filter(algo=r["algo"], index=r["index"], d_min=r["d_min"],
                           eta=r["eta"], lam=r["lam"], k=r["k"])
        steps = r["max_steps"]
        w = World.build(seed=r["seed"], spec=spec, test_case=r["case"],
                        max_steps=steps)
        channel = r.get("channel", "arm")
        G0 = np.asarray(r["G0"], float)
        G1 = np.asarray(r["G1"], float)

        rec, qpos, obst, ctrl = instrumented_run(w, [G0, G1], steps,
                                                 channel=channel)
        clear = min((s.clearance for s in rec.steps), default=np.inf)
        # Re-assert the property the video is meant to demonstrate. If this
        # ever fails the attack is not attributable to the insertion, and a
        # "look, it is safe without the attack" video would be a lie.
        if rec.label != "REACHED" or clear <= 0.0:
            bad.append((tag, f"{rec.label} {clear:+.6f}"))
            print(f"  {tag}: BASELINE NOT SAFE -- {rec.label} {clear:+.6f}",
                  flush=True)
            continue
        write_trace(f"{a.dir}/{tag}_baseline_trace.npz", rec, qpos, obst, ctrl,
                    stride=a.trace_stride)
        ok += 1
        print(f"  {tag}: {rec.label} in {rec.n_steps} steps, "
              f"min clearance {clear:+.6f}", flush=True)

    print(f"\n{'='*70}\n{ok} baseline traces written to {a.dir}")
    if bad:
        print(f"{len(bad)} baselines NOT safe -- their attacks are not "
              f"attributable to the insertion:")
        for t, why in bad:
            print(f"   {t}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
