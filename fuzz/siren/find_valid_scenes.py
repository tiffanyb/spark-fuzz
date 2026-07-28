"""
Which (test case, seed) scenes are VALID for a goal-attack experiment?

A scene is valid only if the legitimate goal G0 -> G1 is reached with the safety
filter active and feasible. If the robot already fails without any attack, then
nothing we do to it afterwards demonstrates an attack — so those scenes must be
excluded, and excluded BY NAME rather than quietly dropped.

    python -m fuzz.siren.find_valid_scenes --seeds 0-9 --out /tmp/valid.json
"""

import argparse
import json
import os
import sys

from .world.run import World
from .world.types import real_filter

CASES = [
    "G1FixedBase_D1_AG_SO_v0", "G1FixedBase_D1_AG_SO_v1",
    "G1FixedBase_D1_AG_DO_v0", "G1FixedBase_D1_AG_DO_v1",
    "G1FixedBase_D2_AG_SO_v0", "G1FixedBase_D2_AG_SO_v1",
    "G1FixedBase_D2_AG_DO_v0", "G1FixedBase_D2_AG_DO_v1",
]


def parse_seeds(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out", default="/tmp/valid_scenes.json")
    a = p.parse_args(argv)

    cases = [c.strip() for c in a.cases.split(",") if c.strip()]
    seeds = parse_seeds(a.seeds)
    rows = []

    for case in cases:
        index = "velocity" if "_D2_" in case else "distance"
        spec = real_filter(algo="ssa", index=index, d_min=a.d_min,
                           eta=a.eta, lam=10.0, k=0.1)
        for seed in seeds:
            row = {"test_case": case, "seed": seed, "index": index}
            try:
                w = World.build(seed=seed, spec=spec, test_case=case,
                                max_steps=a.max_steps)
                rec = w.run([w.scene().G1], spec, max_steps=a.max_steps)
                row.update({
                    "ok": True,
                    "label": rec.label,
                    "gave_up": rec.n_gave_up,
                    "min_clearance": (None if rec.min_clearance != rec.min_clearance
                                      else round(float(rec.min_clearance), 5)),
                    "steps": rec.n_steps,
                    # valid = reached AND the filter never had to give up
                    "valid": bool(rec.reached and rec.n_gave_up == 0),
                })
            except Exception as e:
                row.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                            "valid": False})
            rows.append(row)
            print(f"{case:<28} seed={seed:<3} "
                  f"{row.get('label', 'ERR'):<10} gave_up={row.get('gave_up')} "
                  f"{'VALID' if row.get('valid') else '-'}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=2)

    print("\n" + "=" * 64)
    for case in cases:
        good = [r["seed"] for r in rows if r["test_case"] == case and r.get("valid")]
        print(f"{case:<28} valid seeds: {good if good else 'NONE'}")
    print("=" * 64)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
