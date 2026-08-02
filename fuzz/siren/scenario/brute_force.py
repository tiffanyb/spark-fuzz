"""
Ground truth on a REAL scene: sweep every inserted goal, find what exists.

This is the control the project has been missing. SIREN's search reports few or
no attacks post-fix, and that reading is ambiguous:

    attacks do not exist          -> the fixes worked, the result is real
    the search cannot find them   -> every negative result is uninterpretable

A dense exhaustive sweep settles it, because it uses no guidance, no optimiser
and no budget. Whatever it finds is what EXISTS up to grid resolution. Then:

    brute force finds attacks SIREN missed  -> the fuzzer is the problem
    brute force finds nothing either        -> the scene is genuinely clean

Unlike the hand-built scenes in scenes.py this needs no layout design, so it
cannot be spoiled by an unfair or unreachable scene — it runs on the benchmark's
own geometry, on scenes already known to pass the baseline gate.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.brute_force \
        --case G1FixedBase_D2_AG_SO_v0 --seed 0 --grid 8
"""

import argparse
import json
import time
from collections import Counter

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algo", default="ssa")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..search.pick import is_admissible

    index = "velocity" if "_D2_" in a.case else "distance"
    steps = a.max_steps or (900 if "_D2_" in a.case else 1500)
    spec = real_filter(algo=a.algo, index=index, d_min=a.d_min,
                       eta=a.eta, lam=10.0, k=0.1)
    w = World.build(seed=a.seed, spec=spec, test_case=a.case, max_steps=steps)
    sc = w.scene()

    base = w.run([np.asarray(sc.G1)], spec, max_steps=steps)
    print(f"{a.case} seed={a.seed} algo={a.algo}")
    print(f"  baseline G0 -> G1: {base.label}  steps={base.n_steps}  "
          f"gave_up={base.n_gave_up}", flush=True)
    if not base.reached:
        print("  baseline fails — this scene cannot host an attack by definition")
        return 1

    lo = [b[0] for b in sc.bounds]
    hi = [b[1] for b in sc.bounds]
    axes = [np.linspace(lo[d], hi[d], a.grid) for d in range(3)]
    cands = [np.array([x, y, z])
             for x in axes[0] for y in axes[1] for z in axes[2]]
    cands = [c for c in cands if is_admissible(c, sc)[0]]
    si, sn = (int(x) for x in a.shard.split("/"))
    cands = [c for i, c in enumerate(cands) if i % sn == si]
    print(f"  shard {si}/{sn}: {len(cands)} admissible goals from a "
          f"{a.grid}^3 grid — exhaustive, no guidance", flush=True)

    t0, rows, hits = time.time(), [], []
    for i, c in enumerate(cands):
        one = w.run([c], max_steps=steps)
        if not one.reached:
            rows.append({"cand": c.tolist(), "leg1": one.label, "leg2": None})
            continue
        full = w.run([c, np.asarray(sc.G1)], max_steps=steps)
        r = {"cand": c.tolist(), "leg1": "REACHED", "leg2": full.label,
             "min_clearance": float(full.min_clearance),
             "n_gave_up": int(full.n_gave_up)}
        rows.append(r)
        if full.label in ("COLLISION", "DEADLOCK"):
            hits.append(r)
            print(f"    HIT {full.label} at {np.round(c,3)}  "
                  f"clearance {full.min_clearance:+.5f}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(cands)}  hits={len(hits)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    viable = sum(1 for r in rows if r["leg2"])
    print(f"\n{'='*66}")
    print(f"leg-1: {dict(Counter(r['leg1'] for r in rows))}")
    print(f"leg-2: {dict(Counter(r['leg2'] for r in rows if r['leg2']))}")
    print(f"ATTACKS EXISTING (brute force): {len(hits)} / {viable} viable goals")
    out = a.out or f"/tmp/brute_{a.case}_s{a.seed}_{si}.json"
    json.dump({"case": a.case, "seed": a.seed, "algo": a.algo,
               "grid": a.grid, "baseline": base.label,
               "n_viable": viable, "hits": hits, "rows": rows},
              open(out, "w"), indent=2, default=float)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
