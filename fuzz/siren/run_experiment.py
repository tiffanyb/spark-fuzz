"""
The full experiment: every threat model x both attacks x every in-scope SPARK case.

    python -m fuzz.siren.run_experiment --test-case G1FixedBase_D1_AG_SO_v0 \
        --seeds 0,20 --budget 15 --out /tmp/exp_D1_SO_v0.json

Runs all (attack, threat) combinations for one test case and reports whether any
goal attack was found. For WHITE BOX it additionally separates the two failure
mechanisms:

  (A) INTRINSIC authority void
      The goal routes the robot into a pose where the motors simply cannot meet
      the filter's demand. The infeasibility is a property of the geometry; the
      filter's own choices did not create it.

  (B) FILTER-INDUCED infeasibility
      Every individual leg is fine -- the legitimate goal is reachable AND the
      inserted goal is reachable on its own, each with the filter feasible
      throughout -- yet running them IN SEQUENCE drives the filter into a state
      where it has no safe control left. Nothing about either goal is dangerous;
      the filter walked itself into the trap by taking locally-safe corrections
      that stranded it. This is the myopia result: instantaneous safety without
      future feasibility.

  The discriminator is therefore a three-way comparison, all under the real
  filter:
        baseline  G0 -> G1        gave up?   must be NO
        one-hop   G0 -> G1'       gave up?   must be NO
        attack    G0 -> G1' -> G1 gave up?   must be YES
  If all three hold, neither goal alone can be blamed and the failure is purely
  an artefact of ordering plus the filter's own trajectory -> (B).
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

from .world.run import World
from .world.types import FilterSpec, real_filter
from .search.attacks import make_attack
from .search.loop import search
from .search.pick import make_picker
from .search.threat import threat_model

THREATS = ["white", "gray", "gray-proportional", "weak-black", "strict-black", "random"]
ATTACKS = ["insertion", "modification"]


def _f(x, nd=4):
    try:
        v = float(x)
        return None if not np.isfinite(v) else round(v, nd)
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
#  White-box: which mechanism produced the failure?
# ---------------------------------------------------------------------------- #
def classify_mechanism(world, scene, cand, spec, attack_name, max_steps):
    """Separate a filter-induced trap (B) from an intrinsic authority void (A).

    Returns a dict with the three probe runs and the verdict.
    """
    out = {}

    base = world.run([scene.G1], spec, max_steps=max_steps)
    out["baseline_label"] = base.label
    out["baseline_gave_up"] = base.n_gave_up
    out["baseline_infeasible_steps"] = base.n_predicted_infeasible

    atk_schedule = ([np.asarray(cand), np.asarray(scene.G1)]
                    if attack_name == "insertion" else [np.asarray(cand)])
    atk = world.run(atk_schedule, spec, max_steps=max_steps)
    out["attack_label"] = atk.label
    out["attack_gave_up"] = atk.n_gave_up
    out["attack_infeasible_steps"] = atk.n_predicted_infeasible
    out["attack_min_g"] = _f(atk.min_g)
    out["attack_min_C"] = _f(atk.min_C_phi)
    out["attack_penetration"] = _f(atk.max_penetration)

    if attack_name == "insertion":
        one = world.run([np.asarray(cand)], spec, max_steps=max_steps)
        out["onehop_label"] = one.label
        out["onehop_gave_up"] = one.n_gave_up
        out["onehop_infeasible_steps"] = one.n_predicted_infeasible

        # Did the filter lose feasibility only when the goals were SEQUENCED?
        legs_clean = (base.n_gave_up == 0 and one.n_gave_up == 0
                      and base.reached and one.reached)
        seq_broke = atk.n_gave_up > 0 or atk.n_predicted_infeasible > 0
        out["filter_induced"] = bool(legs_clean and seq_broke)
        out["legs_clean"] = bool(legs_clean)
    else:
        # A modified goal is a SINGLE leg, so there is no ordering to blame and
        # the three-way comparison does not exist. We therefore cannot separate
        # the mechanisms here at all -- claiming "filter-induced" from a clean
        # baseline alone would just be restating "the attack worked on a valid
        # scene". Report it as undetermined rather than overclaiming.
        out["onehop_label"] = None
        out["onehop_gave_up"] = None
        out["filter_induced"] = None
        out["legs_clean"] = bool(base.n_gave_up == 0 and base.reached)

    out["mechanism"] = ("undetermined (single leg)" if out["filter_induced"] is None
                        else "filter-induced" if out["filter_induced"]
                        else "intrinsic")
    return out


# ---------------------------------------------------------------------------- #
def run_cell(test_case, seed, attack_name, threat_name, budget, max_steps,
             d_min, eta, lam, picker_name, verbose=False):
    """One (case, seed, attack, threat) experiment."""
    index = "velocity" if "_D2_" in test_case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=d_min, eta=eta, lam=lam, k=0.1)

    t0 = time.time()
    cell = {"test_case": test_case, "seed": seed, "attack": attack_name,
            "threat": threat_name, "index": index}
    try:
        world = World.build(seed=seed, spec=spec, test_case=test_case,
                            max_steps=max_steps)
        scene = world.scene()
        tm = threat_model(threat_name, d_min=d_min, index=index, real=spec)
        attack = make_attack(attack_name)
        picker = make_picker(picker_name, scene, seed=0)

        # Batch must be SMALLER than the budget or a concentrating picker never
        # gets to concentrate: with batch == budget, CEM draws one generation from
        # its initial (narrow) distribution, receives feedback once, and the loop
        # ends -- strictly worse coverage than uniform sampling. Give it several
        # generations instead.
        batch = max(4, budget // 4)
        rep = search(world, attack, picker, tm, budget=budget, batch=batch,
                     max_steps=max_steps, verbose=verbose)

        cell.update({
            "ok": True,
            "aborted": rep.aborted,
            "n_screened": rep.n_screened,
            "n_gated_out": rep.n_gated_out,
            "n_success": rep.n_success,
            "success_rate": (round(rep.n_success / rep.n_screened, 4)
                             if rep.n_screened else None),
            "baseline_label": (rep.baseline.label if rep.baseline else None),
            "baseline_gave_up": (rep.baseline.n_gave_up if rep.baseline else None),
            "n_specs": len(tm.specs),
            "elapsed_s": round(time.time() - t0, 1),
        })

        hits = [r for r in rep.ranked() if r.success]
        cell["hit_labels"] = sorted({l for r in hits for l in r.labels})
        cell["hits"] = [{"candidate": [round(float(x), 4) for x in r.candidate],
                         "score": _f(r.score), "labels": r.labels}
                        for r in hits[:5]]

        # --- white box only: which mechanism? ------------------------------ #
        if threat_name == "white" and hits:
            mech = []
            for r in hits[:5]:
                try:
                    mech.append(classify_mechanism(world, scene, r.candidate, spec,
                                                   attack_name, max_steps))
                except Exception as e:
                    mech.append({"error": f"{type(e).__name__}: {e}"})
            cell["mechanisms"] = mech
            cell["n_filter_induced"] = sum(1 for m in mech if m.get("filter_induced"))

    except Exception as e:
        cell.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()[-1500:],
                     "elapsed_s": round(time.time() - t0, 1)})
    return cell


# ---------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description="SIREN full experiment matrix")
    p.add_argument("--test-case", required=True)
    p.add_argument("--seeds", default="20")
    p.add_argument("--attacks", default=",".join(ATTACKS))
    p.add_argument("--threats", default=",".join(THREATS))
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--picker", default="random")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    attacks = [s.strip() for s in a.attacks.split(",") if s.strip()]
    threats = [s.strip() for s in a.threats.split(",") if s.strip()]

    cells = []
    total = len(seeds) * len(attacks) * len(threats)
    i = 0
    for seed in seeds:
        for attack in attacks:
            for threat in threats:
                i += 1
                print(f"[{a.test_case}] {i}/{total} seed={seed} {attack}/{threat}",
                      flush=True)
                c = run_cell(a.test_case, seed, attack, threat, a.budget,
                             a.max_steps, a.d_min, a.eta, a.lam, a.picker)
                status = ("ERR" if not c.get("ok") else
                          f"success={c.get('n_success')}/{c.get('n_screened')}")
                print(f"    -> {status}  ({c.get('elapsed_s')}s)", flush=True)
                cells.append(c)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"args": vars(a), "cells": cells}, f, indent=2)
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
