"""
Replay the exact goals that collided BEFORE the L_f sign fix, on the fixed filter.

The post-fix sweep runs a fresh search and finds nothing on the static scenes.
That is strong, but it leaves one gap: a search that stops *proposing* dangerous
goals for some unrelated reason looks identical to a filter that stops failing.
This removes the search from the loop entirely — same goal, same scene, same
seed, same everything, with only the sign of one term changed — so any change in
outcome is attributable to the fix and nothing else.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.replay_collisions --shard 0/4
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="/tmp/rep_shard*.jsonl")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from .world.run import World
    from .world.types import real_filter

    old = defaultdict(list)
    for f in glob.glob(a.sweep):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or "_D2_" not in r["case"]:
                continue
            for e in r["evaluations"]:
                if e["stage"] == "evaluated" and e["collided"]:
                    old[(r["case"], r["seed"], r["max_steps"])].append(e["candidate"])

    si, sn = (int(x) for x in a.shard.split("/"))
    keys = [k for i, k in enumerate(sorted(old)) if i % sn == si]
    total = sum(len({tuple(np.round(c, 9)) for c in old[k]}) for k in keys)
    print(f"shard {si}/{sn}: {total} unique previously-colliding goals "
          f"across {len(keys)} scenes", flush=True)

    rows = []
    for case, seed, ms in keys:
        # the same goal can be found by several arms/search seeds — dedupe first
        uniq = {tuple(np.round(c, 9)): c for c in old[(case, seed, ms)]}
        spec = real_filter(algo="ssa", index="velocity", d_min=0.02,
                           eta=0.02, k=0.1)
        w = World.build(seed=seed, spec=spec, test_case=case, max_steps=ms)
        sc = w.scene()
        still = 0
        for c in uniq.values():
            rec = w.run([np.asarray(c, float), np.asarray(sc.G1)], max_steps=ms)
            hit = rec.label == "COLLISION"
            still += hit
            rows.append({"case": case, "seed": seed,
                         "candidate": [float(x) for x in c],
                         "label_after": rec.label,
                         "min_clearance_after": float(rec.min_clearance),
                         "still_collides": bool(hit)})
        print(f"  {case} seed={seed}: {len(uniq)} goals -> {still} still collide "
              f"({100.0*still/max(1,len(uniq)):.0f}%)", flush=True)

    out = a.out or f"/tmp/replay_{si}.json"
    json.dump(rows, open(out, "w"), indent=2, default=float)
    n_hit = sum(1 for r in rows if r["still_collides"])
    print(f"\nshard {si}: {n_hit}/{len(rows)} previously-colliding goals still collide")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
