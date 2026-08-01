"""
Find a scene with a usable ATTACK RATE against a filter that actually works.

Comparing guidance strategies needs positives. At eta=0.02 on seed 20 the rate
was 0/150 (insertion) and 2/150 (modification) -- far too few for any comparison
to mean anything. This screens seeds for one where attacks are common enough to
measure against.

For each seed it checks, in order, and skips as soon as one fails:
  1. VALID scene      -- the legitimate goal is reached AND the filter never
                         gives up (a filter that is already giving up is inert,
                         and attacks against it prove nothing)
  2. DEMAND met       -- the filter finds a safe control at most engaged steps,
                         so it is genuinely operating rather than broken
  3. ATTACK RATE      -- unguided random search, so the number measures the
                         SCENE and not the search

Modification is screened first: it needs one rollout per candidate rather than
two, and it had the higher base rate.

    python -m fuzz.siren.find_attackable_seeds --seeds 0-19 --budget 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from .world.run import World
from .world.types import real_filter
from .search.attacks import make_attack
from .search.loop import search, probe_demand
from .search.pick import make_picker
from .search.threat import threat_model


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


def screen_seed(seed, test_case, eta, d_min, budget, max_steps, attacks):
    index = "velocity" if "_D2_" in test_case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=d_min, eta=eta, k=0.1)
    row = {"seed": seed, "test_case": test_case, "eta": eta}
    t0 = time.time()
    try:
        world = World.build(seed=seed, spec=spec, test_case=test_case,
                            max_steps=max_steps)
        scene = world.scene()

        # 1. is the scene valid at all?
        base = world.run([scene.G1], spec, max_steps=max_steps)
        row["baseline"] = base.label
        row["baseline_gave_up"] = base.n_gave_up
        if not base.reached or base.n_gave_up > 0:
            row["status"] = "invalid_scene"
            row["elapsed"] = round(time.time() - t0, 1)
            return row

        # 2. is the filter actually operating?
        feas = probe_demand(world, scene, spec, max_steps=max_steps)
        row["demand_verdict"] = feas.get("verdict")
        row["frac_feasible"] = feas.get("frac_feasible_when_engaged")
        row["C_min"] = feas.get("C_min")
        row["C_max"] = feas.get("C_max")
        if feas.get("verdict") in ("never_feasible", "mostly_infeasible"):
            row["status"] = "demand_not_met"
            row["elapsed"] = round(time.time() - t0, 1)
            return row

        # 3. the attack rate, measured WITHOUT guidance so it describes the scene
        tm = threat_model("random", d_min=d_min, index=index, real=spec)
        for atk in attacks:
            rep = search(world, make_attack(atk),
                         make_picker("random", scene, seed=0), tm,
                         budget=budget, batch=budget, max_steps=max_steps,
                         verbose=False)
            row[f"{atk}_success"] = rep.n_success
            row[f"{atk}_screened"] = rep.n_screened
            row[f"{atk}_rate"] = ((rep.n_success / rep.n_screened)
                                  if rep.n_screened else None)
            row[f"{atk}_labels"] = sorted({l for r in rep.ranked() if r.success
                                           for l in r.labels})
        row["status"] = "screened"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    row["elapsed"] = round(time.time() - t0, 1)
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-19")
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--attacks", default="modification,insertion")
    p.add_argument("--out", default="/tmp/attackable_seeds.json")
    a = p.parse_args(argv)

    seeds = parse_seeds(a.seeds)
    attacks = [s.strip() for s in a.attacks.split(",") if s.strip()]
    rows = []
    for sd in seeds:
        r = screen_seed(sd, a.test_case, a.eta, a.d_min, a.budget,
                        a.max_steps, attacks)
        rows.append(r)
        bits = [f"seed={sd:<3}", f"{r['status']:<16}"]
        for atk in attacks:
            if r.get(f"{atk}_rate") is not None:
                bits.append(f"{atk[:4]}={r[f'{atk}_success']}/"
                            f"{r[f'{atk}_screened']}")
        bits.append(f"({r['elapsed']}s)")
        print("  ".join(bits), flush=True)
        json.dump(rows, open(a.out, "w"), indent=2, default=float)

    print("\n" + "=" * 74)
    print(f"SEED SCREEN — {a.test_case}, eta={a.eta}, d_min={a.d_min}, "
          f"budget={a.budget}")
    print("=" * 74)
    good = [r for r in rows if r["status"] == "screened"]
    print(f"{len(good)}/{len(rows)} seeds gave a valid scene with a working filter")
    for reason in ("invalid_scene", "demand_not_met", "error"):
        n = sum(1 for r in rows if r["status"] == reason)
        if n:
            print(f"   {n} rejected: {reason}")

    for atk in attacks:
        ranked = sorted((r for r in good if r.get(f"{atk}_rate") is not None),
                        key=lambda r: -r[f"{atk}_rate"])
        print(f"\n{atk.upper()} — seeds by attack rate:")
        if not ranked or ranked[0][f"{atk}_rate"] == 0:
            print("   no seed produced any attack at this budget")
            continue
        for r in ranked[:8]:
            if r[f"{atk}_rate"] == 0:
                break
            print(f"   seed {r['seed']:<4} {r[f'{atk}_success']}/"
                  f"{r[f'{atk}_screened']} = {100*r[f'{atk}_rate']:.0f}%   "
                  f"{r.get(f'{atk}_labels')}   "
                  f"feasible={100*(r.get('frac_feasible') or 0):.0f}%")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
