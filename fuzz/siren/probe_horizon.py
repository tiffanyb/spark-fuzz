"""
Are the D1 timeouts hiding collisions, or hiding deadlocks?

985 D1 candidates reached the inserted goal and then failed to get back to G1
within the 500-step horizon. They ran the full horizon, sat at a median minimum
clearance of +0.0197 (i.e. exactly on d_min = 0.020), and had the filter engaged
on 85.6% of runs against 47.7% for runs that reached. That is the profile of a
robot that has been STOPPED, not one that was merely slow — but the outcome label
says TIMEOUT, which the taxonomy treats as inconclusive and discards.

Re-running them on a much longer horizon separates three possibilities that the
500-step label cannot:

    COLLISION  the horizon was masking a failure  -> the invariance argument is
               wrong and D1 does have an attack surface
    REACHED    the robot was just slow            -> TIMEOUT was a horizon
               artefact and the candidate is harmless
    still stuck  the robot is genuinely trapped   -> a denial-of-service attack
               that "0 collisions" was never going to count

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.probe_horizon --shard 0/4
"""

import argparse
import glob
import json
from collections import Counter, defaultdict

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="/tmp/rep_shard*.jsonl")
    p.add_argument("--family", default="_D1_")
    p.add_argument("--long-horizon", type=int, default=2500)
    p.add_argument("--per-scene", type=int, default=8)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from .world.run import World
    from .world.types import real_filter

    pool = defaultdict(list)
    for f in glob.glob(a.sweep):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or a.family not in r["case"]:
                continue
            for e in r["evaluations"]:
                if e["stage"] == "evaluated" and e["leg2_label"] == "TIMEOUT":
                    pool[(r["case"], r["seed"], r["max_steps"])].append(e)

    si, sn = (int(x) for x in a.shard.split("/"))
    keys = [k for i, k in enumerate(sorted(pool)) if i % sn == si]
    print(f"shard {si}/{sn}: {len(keys)} scenes, re-running leg-2 timeouts at "
          f"{a.long_horizon} steps (was 500)\n", flush=True)

    rows = []
    for case, seed, ms in keys:
        evs = pool[(case, seed, ms)]
        uniq = {tuple(np.round(e["candidate"], 9)): e for e in evs}
        picks = list(uniq.values())[:a.per_scene]
        index = "velocity" if "_D2_" in case else "distance"
        spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
        w = World.build(seed=seed, spec=spec, test_case=case,
                        max_steps=a.long_horizon)
        sc = w.scene()
        out = Counter()
        for e in picks:
            rec = w.run([np.asarray(e["candidate"], float),
                         np.asarray(sc.G1)], max_steps=a.long_horizon)
            # did it make progress after the old horizon expired?
            d_at_old = next((s.dist_final for s in rec.steps if s.step >= ms), None)
            rows.append({
                "case": case, "seed": seed, "candidate": e["candidate"],
                "label_500": "TIMEOUT", "label_long": rec.label,
                "n_steps": len(rec.steps),
                "min_clearance": float(rec.min_clearance),
                "final_dist": float(rec.final_dist),
                "dist_at_old_horizon": (float(d_at_old) if d_at_old is not None else None),
                "n_gave_up": int(rec.n_gave_up),
            })
            out[rec.label] += 1
        print(f"  {case} seed={seed}: {len(picks)} timeouts -> {dict(out)}", flush=True)

    path = a.out or f"/tmp/horizon_{si}.json"
    json.dump(rows, open(path, "w"), indent=2, default=float)

    c = Counter(r["label_long"] for r in rows)
    print(f"\n{'='*66}\n{len(rows)} previously-TIMEOUT candidates at "
          f"{a.long_horizon} steps:")
    for k, v in c.most_common():
        print(f"   {k:<12}{v:>5}  ({100*v/max(1,len(rows)):.0f}%)")
    mc = np.array([r["min_clearance"] for r in rows])
    print(f"\n   min clearance over all: {mc.min():+.5f}   "
          f"(negative would mean the horizon was hiding a collision)")
    stuck = [r for r in rows if r["label_long"] in ("TIMEOUT", "DEADLOCK")]
    if stuck:
        prog = [r["dist_at_old_horizon"] - r["final_dist"] for r in stuck
                if r["dist_at_old_horizon"] is not None]
        if prog:
            print(f"   still-stuck runs: progress made AFTER step {a.per_scene and ms}: "
                  f"median {np.median(prog):+.4f} m "
                  f"({'genuinely frozen' if abs(np.median(prog)) < 0.01 else 'still moving'})")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
