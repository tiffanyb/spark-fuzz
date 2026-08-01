"""
Is the demand physically achievable for this robot?

The margin is g = C - eta (kinematic case). If eta exceeds the authority C the
robot can muster in the poses where the filter actually engages, then g < 0
WHENEVER the filter is active -- the certificate is saturated, carries no
information, and the filter never once operates feasibly. It is then binary:
dormant, or broken.

This sweeps eta and measures, for a known attack and a known non-attack:
  * how often the filter engages,
  * of those steps, how many it can actually SOLVE (feasible) vs gives up on,
  * whether the attack still succeeds,
  * how much g varies -- i.e. whether it could rank anything.

Units matter: phi is in metres, so eta is a demanded retreat rate in m/s.

    python -m fuzz.siren.diagnose_eta --etas 0.005,0.01,0.02,0.05,0.1,0.5
"""

import argparse
import json

import numpy as np

from .world.run import World
from .world.types import real_filter


def probe(world, scene, schedule, spec, max_steps):
    rec = world.run(schedule, spec, max_steps=max_steps)
    steps = rec.steps
    engaged = [s for s in steps if s.phi >= 0]
    gvals = [s.g for s in engaged if np.isfinite(s.g)]
    feasible_engaged = [s for s in engaged if np.isfinite(s.g) and s.g >= 0]
    Cvals = [s.C_phi for s in engaged if np.isfinite(s.C_phi)]
    return {
        "label": rec.label,
        "n_steps": len(steps),
        "n_engaged": len(engaged),
        "n_feasible_engaged": len(feasible_engaged),
        "pct_feasible_when_engaged": (100.0 * len(feasible_engaged) / len(engaged)
                                      if engaged else float("nan")),
        "n_gave_up": rec.n_gave_up,
        "min_g_engaged": float(min(gvals)) if gvals else float("nan"),
        "g_spread_engaged": (float(max(gvals) - min(gvals)) if len(gvals) > 1
                             else 0.0),
        "min_C_engaged": float(min(Cvals)) if Cvals else float("nan"),
        "max_C_engaged": float(max(Cvals)) if Cvals else float("nan"),
        "min_clearance": float(rec.min_clearance),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from-json", default="/tmp/siren_w2.json")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--etas", default="0.005,0.01,0.02,0.05,0.1,0.5")
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", default="/tmp/eta_sweep.json")
    a = p.parse_args(argv)

    data = json.load(open(a.from_json))["results"]
    hit = [r["candidate"] for r in data if r["success"]][0]
    miss = [r["candidate"] for r in data if not r["success"]][0]
    etas = [float(x) for x in a.etas.split(",")]

    rows = []
    for eta in etas:
        spec = real_filter(algo="ssa", index="distance", d_min=a.d_min,
                           eta=eta, k=0.1)
        world = World.build(seed=a.seed, spec=spec, test_case=a.test_case,
                            max_steps=a.max_steps)
        sc = world.scene()
        base = probe(world, sc, [sc.G1], spec, a.max_steps)
        atk = probe(world, sc, [np.asarray(hit), np.asarray(sc.G1)], spec, a.max_steps)
        non = probe(world, sc, [np.asarray(miss), np.asarray(sc.G1)], spec, a.max_steps)
        rows.append({"eta": eta, "baseline": base, "attack": atk, "nonattack": non})
        print(f"  eta={eta}: baseline={base['label']}/{base['n_gave_up']}  "
              f"attack={atk['label']}  nonattack={non['label']}", flush=True)

    print("\n" + "=" * 100)
    print("DEMAND SWEEP — is the filter ever able to do its job?")
    print("=" * 100)
    print(f"{'eta':<8}{'baseline':<14}{'attack':<12}{'non-attack':<13}"
          f"{'engaged':<10}{'FEASIBLE when engaged':<24}{'g spread'}")
    print("-" * 100)
    for r in rows:
        atk = r["attack"]
        pct = atk["pct_feasible_when_engaged"]
        pcts = f"{pct:.0f}%" if pct == pct else "n/a"
        print(f"{r['eta']:<8}{r['baseline']['label']+'/'+str(r['baseline']['n_gave_up']):<14}"
              f"{atk['label']:<12}{r['nonattack']['label']:<13}"
              f"{atk['n_engaged']:<10}"
              f"{str(atk['n_feasible_engaged'])+'/'+str(atk['n_engaged'])+' = '+pcts:<24}"
              f"{atk['g_spread_engaged']:.4f}")

    print("\n" + "-" * 100)
    print("AUTHORITY ACTUALLY AVAILABLE WHERE THE FILTER ENGAGES")
    print("-" * 100)
    a0 = rows[0]["attack"]
    print(f"  C ranges over [{a0['min_C_engaged']:.4f}, {a0['max_C_engaged']:.4f}] m/s "
          f"at engaged steps")
    print(f"  => any eta above ~{a0['min_C_engaged']:.3f} m/s makes the filter")
    print(f"     infeasible at the WORST pose it meets; any eta above")
    print(f"     ~{a0['max_C_engaged']:.3f} makes it infeasible EVERYWHERE it engages.")
    print(f"  The paper's eta = 0.5 m/s demands a retreat rate "
          f"{0.5 / max(a0['min_C_engaged'], 1e-9):.0f}x the worst-case achievable.")
    json.dump(rows, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
