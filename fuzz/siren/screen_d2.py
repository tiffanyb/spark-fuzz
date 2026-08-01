"""
Does g-guidance work in the regime it was DESIGNED for?

On the kinematic (D1) benchmark the margin objective was unusable, but for a
reason that let it off the hook: g never went negative there. The robot is
velocity-controlled, so the drift L_f phi is zero, the best achievable rate is
-C <= 0, and danger is always reducible by something. The filter was feasible on
every run, including the ones that collided -- so g was describing a failure mode
that never occurred.

A D2 (torque-controlled) world is different. Momentum and gravity make
L_f phi != 0, which opens the corner from Appendix A part (4):

        g < -demand   <=>   C < L_f phi   <=>   danger GROWS under every
                                                admissible command

That is genuine infeasibility, and it is exactly what the margin objective was
built to find. This screens D2 scenes for the preconditions:

  1. VALID scene       -- legitimate goal reached, filter never gives up
  2. DEMAND met        -- the filter can actually satisfy its own demand
  3. **g GOES NEGATIVE** -- the decisive one. If the margin never turns negative
                          here either, the objective still has nothing to rank by
                          and the D2 test is inconclusive rather than positive.
  4. ATTACK RATE       -- measured unguided, so it describes the scene

    python -m fuzz.siren.screen_d2 --seeds 0-9 --etas 0.02,0.1,0.5
"""

import argparse
import json

import numpy as np

from .world.run import World
from .world.types import real_filter
from .search.attacks import make_attack
from .search.loop import probe_demand, search
from .search.pick import make_picker, sample_admissible
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


def probe_scene(seed, case, eta, d_min, max_steps, n_probe, budget):
    spec = real_filter(algo="ssa", index="velocity", d_min=d_min, eta=eta, k=0.1)
    row = {"seed": seed, "case": case, "eta": eta}
    try:
        world = World.build(seed=seed, spec=spec, test_case=case,
                            max_steps=max_steps)
        scene = world.scene()

        base = world.run([scene.G1], spec, max_steps=max_steps)
        row["baseline"] = base.label
        row["baseline_gave_up"] = base.n_gave_up
        if not base.reached or base.n_gave_up > 0:
            row["status"] = "invalid_scene"
            return row

        feas = probe_demand(world, scene, spec, max_steps=max_steps)
        row["demand"] = feas.get("verdict")
        row["frac_feasible"] = feas.get("frac_feasible_when_engaged")
        if feas.get("verdict") in ("never_feasible", "mostly_infeasible"):
            row["status"] = "demand_not_met"
            return row

        # ---- the decisive measurement: does the margin ever go NEGATIVE? ----
        rng = np.random.RandomState(0)
        n_runs = n_neg_runs = 0
        neg_steps = tot_eng = 0
        drift = []
        gmins = []
        for _ in range(n_probe):
            c = sample_admissible(rng, scene)
            if c is None:
                continue
            rec = world.run([c, scene.G1], spec, max_steps=max_steps)
            eng = [s for s in rec.steps if s.engaged]
            if not eng:
                continue
            n_runs += 1
            tot_eng += len(eng)
            neg = [s for s in eng if np.isfinite(s.g) and s.g < 0]
            neg_steps += len(neg)
            if neg:
                n_neg_runs += 1
            if np.isfinite(rec.min_g):
                gmins.append(float(rec.min_g))
        row["n_runs_engaged"] = n_runs
        row["runs_with_g_negative"] = n_neg_runs
        row["frac_engaged_steps_g_negative"] = (neg_steps / tot_eng) if tot_eng else None
        row["min_g_seen"] = (min(gmins) if gmins else None)
        row["mean_min_g"] = (float(np.mean(gmins)) if gmins else None)

        # ---- attack rate, unguided ----
        tm = threat_model("random", d_min=d_min, index="velocity", real=spec)
        rep = search(world, make_attack("insertion"),
                     make_picker("random", scene, seed=0), tm,
                     budget=budget, batch=budget, max_steps=max_steps,
                     verbose=False)
        row["attacks"] = rep.n_success
        row["screened"] = rep.n_screened
        row["rate"] = (rep.n_success / rep.n_screened) if rep.n_screened else None
        row["status"] = "screened"
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--cases", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--etas", default="0.02,0.1,0.5")
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--n-probe", type=int, default=4)
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--out", default="/tmp/d2_screen.json")
    a = p.parse_args(argv)

    rows = []
    for case in [c.strip() for c in a.cases.split(",") if c.strip()]:
        for eta in [float(x) for x in a.etas.split(",")]:
            for sd in parse_seeds(a.seeds):
                r = probe_scene(sd, case, eta, a.d_min, a.max_steps,
                                a.n_probe, a.budget)
                rows.append(r)
                bits = [f"{case.replace('G1FixedBase_','')}", f"eta={eta:<5}",
                        f"seed={sd:<3}", f"{r['status']:<15}"]
                if r["status"] == "screened":
                    fn = r.get("frac_engaged_steps_g_negative")
                    bits.append(f"g<0 in {r['runs_with_g_negative']}/"
                                f"{r['n_runs_engaged']} runs "
                                f"({100*(fn or 0):.0f}% of engaged steps)")
                    bits.append(f"min_g={r.get('min_g_seen')}")
                    bits.append(f"attacks={r['attacks']}/{r['screened']}")
                print("  ".join(str(b) for b in bits), flush=True)
                json.dump(rows, open(a.out, "w"), indent=2, default=float)

    print("\n" + "=" * 96)
    print("D2 SCREEN — is g EVER negative? (the precondition for g-guidance)")
    print("=" * 96)
    good = [r for r in rows if r["status"] == "screened"]
    withneg = [r for r in good if (r.get("runs_with_g_negative") or 0) > 0]
    print(f"{len(good)} scenes screened; "
          f"**{len(withneg)} showed g < 0 at any point**")
    if withneg:
        print("\nScenes where the margin actually goes negative:")
        for r in sorted(withneg, key=lambda r: -(r.get("rate") or 0)):
            print(f"   {r['case'].replace('G1FixedBase_','')} eta={r['eta']} "
                  f"seed={r['seed']}: g<0 in {r['runs_with_g_negative']}/"
                  f"{r['n_runs_engaged']} runs, min_g={r.get('min_g_seen')}, "
                  f"attacks={r['attacks']}/{r['screened']}")
        print("\n-> run the guidance comparison on the one with a usable attack rate")
    else:
        print("\nNo D2 scene produced a negative margin either. The margin has")
        print("nothing to rank by here as well, so the D2 test is INCONCLUSIVE")
        print("for the concept rather than a positive result for it.")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
