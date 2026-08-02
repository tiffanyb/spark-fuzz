"""
Build a positive control from a collision SPARK already produces.

Designing a collision from scratch failed repeatedly: post-fix the static
scenarios admit none (0 attacks in 417 exhaustively-swept goals on a real scene,
0 in 105 on a designed one), so there is no mechanism left to design around.

This inverts the problem. Some (scenario, filter) pairs ALREADY collide on the
plain task — 42 of them on static obstacles. Take that known-bad segment and
re-use it as the attack's second leg:

    known bad:   home ----------------> G1     collides
    becomes:     G1' = home position,  G1 = the scenario goal

Then search for a first waypoint G0 such that

    legitimate:  [G0, G1]        REACHES      the task is safe without the attacker
    leg one:     [G0, G1']       no collision, no deadlock   (engaged is fine)
    attack:      [G0, G1', G1]   COLLIDES     inserting G1' breaks it

The robot always starts from its home pose, so G0 is the first waypoint rather
than a new spawn position — nothing about the robot's initialisation is altered.

Why this can work where direct design could not: the colliding segment is known
to exist, and the only question is whether some G0 both avoids it directly and
still permits it after the detour. The arrival state at G1' differs from the home
rest state (the arm arrives in motion, from a different direction), so the
collision is not guaranteed to reproduce — which is exactly what the search tests.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.from_known_collision \
        --shard 0/4 --grid 5
"""

import argparse
import json
import time

import numpy as np

#: prefer the arms our tooling is best validated on, then the rest
PRIORITY = ["G1FixedBase_D1_AG_SO_v0", "G1FixedBase_D2_AG_SO_v0",
            "G1FixedBase_D1_AG_SO_v1", "G1FixedBase_D2_AG_SO_v1"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="/tmp/colliding_baselines.json")
    p.add_argument("--grid", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--relax", action="store_true",
                   help="accept DEADLOCK as well as COLLISION on the attack leg")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--lam", type=float, default=10.0,
                   help="CBF demand gain. The default 10.0 makes the FIXED-BASE "
                        "D1 scene infeasible at every step (lambda*phi > C), so "
                        "any control found there is an artifact of the gain.")
    p.add_argument("--only-case", default=None,
                   help="restrict to one scenario")
    p.add_argument("--only-algos", default=None,
                   help="comma-separated filters to keep")
    p.add_argument("--only-seeds", default=None,
                   help="comma-separated seeds to keep")
    p.add_argument("--out-dir", default="fuzz/siren/scenario")
    a = p.parse_args(argv)

    from ..world.run import World
    from ..world.types import real_filter
    from ..search.pick import is_admissible

    cands = json.load(open(a.candidates))
    if a.only_case:
        cands = [c for c in cands if c["case"] == a.only_case]
    if a.only_algos:
        keep = {x.strip() for x in a.only_algos.split(",")}
        cands = [c for c in cands if c["algo"] in keep]
    if a.only_seeds:
        keep = {int(x) for x in a.only_seeds.split(",")}
        cands = [c for c in cands if c["seed"] in keep]
    cands.sort(key=lambda r: (PRIORITY.index(r["case"])
                              if r["case"] in PRIORITY else 99,
                              r["case"], r["algo"], r["seed"]))
    si, sn = (int(x) for x in a.shard.split("/"))
    cands = [c for i, c in enumerate(cands) if i % sn == si]
    print(f"shard {si}/{sn}: {len(cands)} colliding baselines to build from",
          flush=True)

    found = []
    for cand in cands:
        case, algo, seed = cand["case"], cand["algo"], cand["seed"]
        index = "velocity" if "_D2_" in case else "distance"
        steps = a.max_steps or (900 if "_D2_" in case else 1500)
        spec = real_filter(algo=algo, index=index, d_min=0.02, eta=0.02,
                           lam=a.lam, k=0.1)
        try:
            w = World.build(seed=seed, spec=spec, test_case=case, max_steps=steps)
            sc = w.scene()
        except Exception as e:
            print(f"  {case} {algo} s{seed}: build failed ({type(e).__name__})",
                  flush=True)
            continue

        G1p = np.asarray(sc.G0, float)     # the scenario's START becomes G1'
        G1 = np.asarray(sc.G1, float)      # the scenario's GOAL stays G1

        # reproduce the known-bad segment before building on it
        known = w.run([G1], spec, max_steps=steps)
        print(f"\n{case} {algo} seed={seed}: home->G1 = {known.label} "
              f"(clearance {known.min_clearance:+.5f})", flush=True)
        if known.label not in ("COLLISION", "DEADLOCK"):
            print("   does not reproduce — skipping", flush=True)
            continue
        print(f"   G1' := {np.round(G1p,3)} (scenario start)   "
              f"G1 := {np.round(G1,3)} (scenario goal)", flush=True)

        lo = [b[0] for b in sc.bounds]
        hi = [b[1] for b in sc.bounds]
        axes = [np.linspace(lo[i], hi[i], a.grid) for i in range(3)]
        g0s = [np.array([x, y, z])
               for x in axes[0] for y in axes[1] for z in axes[2]]
        g0s = [g for g in g0s if is_admissible(g, sc)[0]]

        t0, hits, mods = time.time(), [], []
        ok_attack = ("COLLISION", "DEADLOCK") if a.relax else ("COLLISION",)
        n_unreachable = 0
        for g0 in g0s:
            # is_admissible checks the workspace box and obstacle keep-out only;
            # it does not know about REACHABILITY. On the mobile base the
            # whole-body IK raises for goals it cannot solve, and one such point
            # used to kill the whole sweep. Skip and count them instead.
            try:
                # 1. the legitimate task must SUCCEED, else there is no attack
                base = w.run([g0, G1], max_steps=steps)
                if base.label != "REACHED":
                    continue
                # 2. the attack must break it. There is deliberately NO separate
                #    [g0, G1'] run: G1' IS the home position, so that schedule's
                #    final waypoint is where the robot already stands and
                #    reached_final fires at step 0 -- it returns REACHED in one
                #    step having never driven to g0, testing nothing. The leg
                #    classification below supersedes it.
                atk = w.run([g0, G1p, G1], max_steps=steps)
            except Exception as e:
                n_unreachable += 1
                print(f"   G0={np.round(g0,3)} unusable "
                      f"({type(e).__name__}) — skipped", flush=True)
                continue
            # C4: the damage must land on the ATTACK leg. Without this the
            # search accepts runs that crash on the way TO the inserted goal --
            # goal modification, not goal insertion. Both the rcbf and rsss
            # controls found before this check existed failed verification for
            # exactly that reason (contact on leg 1, 3/3 reproducible).
            # This also makes C2 redundant: contact at wp_idx == 2 already
            # implies legs 0 and 1 were clean, whereas [g0, G1'] is vacuous --
            # G1' IS the home position, so reached_final fires at step 0.
            contact_leg = None
            for s in atk.steps:
                if s.clearance < 0.0:
                    contact_leg = int(s.wp_idx)
                    break
            # DEADLOCK needs no leg test: measure.classify_run already requires
            # the stall to be on the FINAL waypoint (on_final_leg), so a leg-1
            # freeze is labelled TIMEOUT, not DEADLOCK.
            if atk.label not in ok_attack:
                continue
            if atk.label == "COLLISION" and contact_leg != 2:
                #   leg 0 -> crashed on home->G0; bad G0, not an attack
                #   leg 1 -> crashed reaching the inserted goal; MODIFICATION
                if contact_leg == 1:
                    mods.append({"G0": g0.tolist(), "contact_leg": 1,
                                 "attack_min_clearance": float(atk.min_clearance),
                                 "attack_steps": int(atk.n_steps)})
                continue
            hit = {"G0": g0.tolist(), "attack_label": atk.label,
                   "contact_leg": contact_leg,
                   "attack_min_clearance": float(atk.min_clearance),
                   "attack_steps": int(atk.n_steps),
                   "baseline_steps": int(base.n_steps)}
            hits.append(hit)
            print(f"   *** POSITIVE CONTROL: G0={np.round(g0,3)}  "
                  f"[G0,G1]=REACHED  [G0,G1',G1]={atk.label} "
                  f"({atk.min_clearance:+.5f}) on leg {contact_leg}", flush=True)
        print(f"   swept {len(g0s)} G0 candidates -> {len(hits)} controls, "
              f"{len(mods)} modification-only, {n_unreachable} unusable "
              f"({time.time()-t0:.0f}s)", flush=True)

        if hits:
            rec = {"case": case, "algo": algo, "seed": seed,
                   "index": index, "max_steps": steps,
                   "d_min": 0.02, "eta": 0.02, "lam": a.lam, "k": 0.1,
                   "G1_prime": G1p.tolist(), "G1": G1.tolist(),
                   "home_pose_ee": sc.G0.tolist(),
                   "known_bad_segment": {"label": known.label,
                                         "min_clearance": float(known.min_clearance)},
                   "obstacles_world": [list(map(float, np.asarray(o)[:3, 3]))
                                       for o in sc.obstacles_world],
                   "bounds": [[float(l), float(h)] for l, h in sc.bounds],
                   "keepout": float(sc.keepout),
                   "n_G0_swept": len(g0s), "controls": hits,
                   "modification_hits": mods}
            tag = "" if a.lam == 10.0 else f"_lam{a.lam}"
            name = f"{a.out_dir}/control_{case}_{algo}_s{seed}{tag}.json"
            json.dump(rec, open(name, "w"), indent=2, default=float)
            print(f"   wrote {name}", flush=True)
            found.append(name)

    print(f"\n{'='*66}\nshard {si}: {len(found)} positive controls saved")
    for f in found:
        print(f"   {f}")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
