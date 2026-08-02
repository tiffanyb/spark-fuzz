"""
Self-check: is the scene legitimate, and is each discovered goal a real attack?

Criteria, as agreed. Anything failing is reported, not quietly dropped.

SCENE LEGITIMACY
  L1  obstacles disjoint from each other and from the robot
  L2  initial state FEASIBLE — engaged at t=0 is fine, infeasible is not
  L3  robot not in contact at reset
  L4  the legitimate task succeeds:  [G0,G1] -> REACHED
  L5  the filter is not inert on the baseline: n_gave_up == 0
  L6  G1 obeys the same admissibility the attacker must obey
  L7  deterministic: rebuild reproduces the scene, baseline stable over repeats

DISCOVERED GOAL CORRECTNESS
  C1  admissible — in bounds, >= keepout from obstacle centres
  C2  reaching it is clean:  [G0,G1'] not COLLISION/DEADLOCK
      -- NOT a pass/fail. It CLASSIFIES the attack type:
           clean leg-1 + contact on leg 2  ->  GOAL INSERTION attack
           leg-1 collides or deadlocks     ->  GOAL MODIFICATION attack
         Both are genuine attacks; they differ in which goal does the damage.
  C3  the attack collides:   [G0,G1',G1] -> COLLISION
  C4  which leg the contact lands on — the classifier, not a filter
  C5  same scene without G1' is safe (inherited from L4)
  C6  reproducible: COLLISION on every repeat
  C7  contact on a GUARDED pair (clearance is masked by env_collision_vol_ignore)

C2 and C6 are the ones the fuzzer run never tested. C2 tells insertion apart
from modification rather than rejecting either; C6 matters because these
contacts are 50-280 microns deep and a 0.155 mm perturbation was measured to
flip the outcome.

Distance to G1_prime_truth is deliberately NOT a criterion: any candidate
satisfying C1-C7 is a genuine attack wherever it sits.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.scenario.self_check \
        --fuzz-result /tmp/fuzz_G1FixedBase_D2_AG_SO_v0_pssa_s1_G0-14.json
"""

import argparse
import json

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--fuzz-result",
                   default="/tmp/fuzz_G1FixedBase_D2_AG_SO_v0_pssa_s1_G0-14.json")
    p.add_argument("--targets-dir", default="fuzz/siren/scenario/targets")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from .targets import load_target
    from ..search.pick import is_admissible
    from ..world.sim import probe
    from ..world import derived

    res = json.load(open(a.fuzz_result))
    tpath = f"{a.targets_dir}/{res['target']}.json"
    w, tgt = load_target(tpath)
    sc = w.scene()
    steps = tgt["max_steps"]
    G0 = np.asarray(tgt["G0_commanded"], float)
    G1 = np.asarray(tgt["G1"], float)
    h = w.harness

    print(f"SELF-CHECK  target {tgt['name']}")
    print(f"  {tgt['case']}  filter={tgt['algo']}  seed={tgt['seed']}\n")

    # ------------------------- SCENE LEGITIMACY ------------------------- #
    scene_faults = []
    obs = np.asarray([o for o in sc.obstacles_world])
    inv = np.linalg.inv(sc.base_frame)
    obs_b = np.array([(inv @ np.append(np.asarray(o)[:3, 3], 1.0))[:3]
                      for o in obs]) if len(obs) else np.zeros((0, 3))

    bad = [(i, j, float(np.linalg.norm(obs_b[i] - obs_b[j])))
           for i in range(len(obs_b)) for j in range(i + 1, len(obs_b))
           if np.linalg.norm(obs_b[i] - obs_b[j]) < 0.10]
    if bad:
        scene_faults.append(f"L1 obstacles overlap: {bad[:3]}")

    af, ti = h.reset()
    clear0 = h.clearance(ti)
    if clear0 <= 0.0:
        scene_faults.append(f"L3 robot in contact at reset ({clear0:+.5f})")

    u, ai = h.algo.act(af, ti)
    raw = probe.read_raw(h)
    mu0, engaged0 = np.nan, False
    if raw:
        d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"], raw["phi_mask"],
                             raw["u_lim"], demand_shape=h.spec.demand_shape,
                             eta=h.spec.eta, lam=h.spec.lam, exact=True)
        engaged0 = bool(d.get("engaged"))
        mu0 = float(d.get("mu", np.nan))
        if engaged0 and np.isfinite(mu0) and mu0 > 1e-9:
            scene_faults.append(f"L2 initial state INFEASIBLE (mu={mu0:+.5f})")

    base = w.run([G0, G1], max_steps=steps)
    if base.label != "REACHED":
        scene_faults.append(f"L4 legitimate task fails ({base.label})")
    if base.n_gave_up > 0:
        scene_faults.append(f"L5 filter inert on baseline "
                            f"({base.n_gave_up} give-ups)")
    if not is_admissible(G1, sc)[0]:
        scene_faults.append(f"L6 G1 inadmissible: {is_admissible(G1, sc)[1]}")

    # L7 determinism: rebuild from scratch, compare, and repeat the baseline
    w2, _ = load_target(tpath)
    sc2 = w2.scene()
    same = (np.allclose(sc.G0, sc2.G0) and np.allclose(sc.G1, sc2.G1)
            and np.allclose(np.asarray(sc.obstacles_world),
                            np.asarray(sc2.obstacles_world)))
    if not same:
        scene_faults.append("L7 scene differs on rebuild")
    labels = [w.run([G0, G1], max_steps=steps).label for _ in range(2)]
    if any(l != base.label for l in labels):
        scene_faults.append(f"L7 baseline unstable across repeats: "
                            f"{[base.label] + labels}")

    print(f"  L1 obstacles disjoint          {'FAIL' if bad else 'pass'}")
    print(f"  L2 initial state feasible      "
          f"{'pass (engaged, mu=%+.4f)' % mu0 if engaged0 else 'pass (not engaged)'}")
    print(f"  L3 not in contact at reset     pass (clearance {clear0:+.5f})")
    print(f"  L4 [G0,G1] reaches             {base.label} ({base.n_steps} steps)")
    print(f"  L5 filter not inert            {base.n_gave_up} give-ups")
    print(f"  L6 G1 admissible               {is_admissible(G1, sc)[1]}")
    print(f"  L7 deterministic               "
          f"{'rebuild matches' if same else 'REBUILD DIFFERS'}, "
          f"baseline {[base.label] + labels}")
    print(f"\n  SCENE: {'LEGITIMATE' if not scene_faults else 'FAULTS'}")
    for f in scene_faults:
        print(f"     - {f}")

    # ---------------------- DISCOVERED GOAL CHECKS ---------------------- #
    hits = []
    for r in res["results"]:
        for hgt in r["hits"]:
            hits.append((r["picker"], tuple(np.round(hgt["cand"], 9))))
    uniq = {}
    for pk, c in hits:
        uniq.setdefault(c, []).append(pk)
    keys = sorted(uniq)
    si, sn = (int(x) for x in a.shard.split("/"))
    keys = [k for i, k in enumerate(keys) if i % sn == si]
    print(f"\n  {len(hits)} reported hits -> {len(uniq)} unique candidates "
          f"(shard {si}/{sn}: {len(keys)})\n", flush=True)

    print(f"  {'candidate':<26}{'C1':>4}{'C2':>10}{'C3':>11}{'C4':>5}"
          f"{'C6':>12}  verdict")
    rows, counts = [], {}
    for c in keys:
        cand = np.asarray(c, float)
        c1 = is_admissible(cand, sc)[0]
        leg1 = w.run([G0, cand], max_steps=steps)
        c2 = leg1.label not in ("COLLISION", "DEADLOCK")
        labs, legs = [], []
        for _ in range(a.repeats):
            rec = w.run([G0, cand, G1], max_steps=steps)
            labs.append(rec.label)
            leg = None
            for s in rec.steps:
                if s.clearance < 0.0:
                    leg = int(s.wp_idx)
                    break
            legs.append(leg)
        c3 = labs[0] == "COLLISION"
        c6 = all(l == "COLLISION" for l in labs)
        # C2 + contact leg classify the TYPE rather than accepting/rejecting:
        # damage done on the way TO the inserted goal is a modification attack,
        # damage done on the way BACK to G1 is an insertion attack.
        if c1 and c3 and c6 and legs[0] == 2 and c2:
            kind = "INSERTION"
        elif c1 and c3 and c6 and (legs[0] in (0, 1) or not c2):
            kind = "MODIFICATION"
        else:
            kind = "REJECT"
        counts[kind] = counts.get(kind, 0) + 1
        rows.append({"cand": list(map(float, cand)), "pickers": uniq[c],
                     "C1": bool(c1), "C2_leg1_clean": bool(c2),
                     "leg1_label": leg1.label,
                     "C3": bool(c3), "contact_legs": legs,
                     "C6": bool(c6), "repeat_labels": labs, "kind": kind})
        print(f"  [{cand[0]:+.3f},{cand[1]:+.3f},{cand[2]:+.3f}]"
              f"{'ok' if c1 else 'FAIL':>4}{leg1.label:>10}"
              f"{labs[0]:>11}{str(legs[0]):>5}"
              f"{('%d/%d' % (sum(l=='COLLISION' for l in labs), len(labs))):>12}"
              f"  {kind}", flush=True)

    ins = counts.get("INSERTION", 0)
    mod = counts.get("MODIFICATION", 0)
    rej = counts.get("REJECT", 0)
    print(f"\n  of {len(keys)} discovered goals: {ins} INSERTION, "
          f"{mod} MODIFICATION, {rej} rejected")
    out = a.out or f"/tmp/selfcheck_{res['target']}_{si}.json"
    json.dump({"target": tgt["name"], "scene_faults": scene_faults,
               "scene_legitimate": not scene_faults,
               "n_unique": len(uniq), "n_checked": len(keys),
               "counts": counts,
               "candidates": rows}, open(out, "w"), indent=2, default=float)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
