"""
Why does the c(x)/g(x) guidance not discriminate attacks from non-attacks?

Runs known-successful and known-unsuccessful candidates on the same scene and
dissects the per-step signal, testing each hypothesis in turn:

  H1  the summary is taken over the WRONG STEPS -- `min over the whole
      trajectory` includes the G0->G1' leg and the approach to the FIXED goal
      G1, which every candidate shares, so the minimum is common to all.
  H2  the inactive-constraint FALLBACK contaminates it -- when no constraint is
      enforced we still report C - demand at the nearest pair, which is a
      meaningless (and large-negative) "margin" for a constraint the filter is
      not enforcing.
  H3  the constant demand DOMINATES -- g = C - eta with C ~ 0.06 and eta = 0.5
      means g ~ -0.44 for everything, and the candidate-to-candidate variation in
      C is swamped.
  H4  C genuinely does not vary -- all trajectories converge to the same fixed
      goal, so the configuration at the critical moment is nearly identical.
  H5  the certificate is the wrong TEST -- a single-constraint margin cannot see
      a pincer (several constraints each satisfiable alone, none jointly), so the
      real infeasibility may be invisible to min_i g_i.

    python -m fuzz.siren.diagnose_guidance --n-hits 3 --n-miss 3
"""

import argparse
import json

import numpy as np

from .world.run import World
from .world.types import real_filter


def leg_of(step, n_wp):
    return "G0->G1'" if step.wp_idx < n_wp - 1 else "G1'->G1"


def dissect(world, scene, cand, spec, max_steps, label):
    """Run one candidate and pull the signal apart step by step."""
    rec = world.run([np.asarray(cand), np.asarray(scene.G1)], spec,
                    max_steps=max_steps)
    steps = rec.steps
    n_wp = 2

    engaged = [s for s in steps if s.phi >= 0]
    inactive = [s for s in steps if s.phi < 0]
    final_leg = [s for s in steps if s.wp_idx >= n_wp - 1]
    final_engaged = [s for s in final_leg if s.phi >= 0]

    def mn(seq, attr):
        v = [getattr(s, attr) for s in seq if np.isfinite(getattr(s, attr))]
        return float(min(v)) if v else np.inf

    argmin_step = None
    if steps:
        finite = [(s.g, s) for s in steps if np.isfinite(s.g)]
        if finite:
            argmin_step = min(finite, key=lambda t: t[0])[1]

    return {
        "label": label,
        "outcome": rec.label,
        "cand": [round(float(x), 3) for x in cand],
        "n_steps": len(steps),
        "n_engaged": len(engaged),
        "pct_engaged": round(100.0 * len(engaged) / max(1, len(steps)), 1),
        "n_gave_up": rec.n_gave_up,
        # --- the reported summary (what the objective actually uses) -------
        "min_g_ALL": mn(steps, "g"),
        "min_C_ALL": mn(steps, "C_phi"),
        # --- H2: restrict to steps the filter actually enforces ------------
        "min_g_ENGAGED": mn(engaged, "g"),
        "min_C_ENGAGED": mn(engaged, "C_phi"),
        # --- H1: restrict to the leg where the attack happens --------------
        "min_g_FINAL_LEG": mn(final_leg, "g"),
        "min_C_FINAL_LEG": mn(final_leg, "C_phi"),
        "min_g_FINAL_ENGAGED": mn(final_engaged, "g"),
        "min_C_FINAL_ENGAGED": mn(final_engaged, "C_phi"),
        # --- where is the minimum attained? --------------------------------
        "argmin_leg": (leg_of(argmin_step, n_wp) if argmin_step else "-"),
        "argmin_step": (argmin_step.step if argmin_step else -1),
        "argmin_engaged": (bool(argmin_step.phi >= 0) if argmin_step else False),
        # --- H3/H4: how much does C actually move? -------------------------
        "C_range": (round(float(max(s.C_phi for s in steps if np.isfinite(s.C_phi))
                                - min(s.C_phi for s in steps if np.isfinite(s.C_phi))), 5)
                    if steps else 0.0),
        "max_phi": round(float(max((s.phi for s in steps if np.isfinite(s.phi)),
                                   default=-np.inf)), 5),
        "min_clearance": round(float(rec.min_clearance), 5),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--from-json", default="/tmp/siren_w2.json")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--test-case", default="G1FixedBase_D1_AG_SO_v0")
    p.add_argument("--n-hits", type=int, default=3)
    p.add_argument("--n-miss", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--d-min", type=float, default=0.02)
    a = p.parse_args(argv)

    data = json.load(open(a.from_json))["results"]
    hits = [r["candidate"] for r in data if r["success"]][:a.n_hits]
    miss = [r["candidate"] for r in data if not r["success"]][:a.n_miss]

    spec = real_filter(algo="ssa", index="distance", d_min=a.d_min,
                       eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.test_case,
                        max_steps=a.max_steps)
    scene = world.scene()

    rows = []
    for c in hits:
        rows.append(dissect(world, scene, c, spec, a.max_steps, "HIT"))
    for c in miss:
        rows.append(dissect(world, scene, c, spec, a.max_steps, "miss"))

    # ---------------- report ------------------------------------------- #
    print("\n" + "=" * 100)
    print("PER-CANDIDATE SIGNAL DISSECTION")
    print("=" * 100)
    hdr = (f"{'':<6}{'outcome':<11}{'steps':<7}{'engaged':<10}{'gave_up':<9}"
           f"{'min_g ALL':<12}{'min_g ENG':<12}{'min_g FINAL':<13}{'argmin at':<14}")
    print(hdr)
    for r in rows:
        print(f"{r['label']:<6}{r['outcome']:<11}{r['n_steps']:<7}"
              f"{str(r['n_engaged'])+' ('+str(r['pct_engaged'])+'%)':<10}"
              f"{r['n_gave_up']:<9}"
              f"{r['min_g_ALL']:<12.4f}{r['min_g_ENGAGED']:<12.4f}"
              f"{r['min_g_FINAL_ENGAGED']:<13.4f}"
              f"{r['argmin_leg']+'/'+('eng' if r['argmin_engaged'] else 'INACTIVE'):<14}")

    # ---------------- separation of each variant ------------------------ #
    print("\n" + "-" * 100)
    print("DOES ANY VARIANT SEPARATE HITS FROM MISSES?")
    print("-" * 100)
    H = [r for r in rows if r["label"] == "HIT"]
    M = [r for r in rows if r["label"] == "miss"]
    print(f"{'statistic':<24}{'hits mean':<14}{'miss mean':<14}{'separation':<14}{'verdict'}")
    for key in ("min_g_ALL", "min_g_ENGAGED", "min_g_FINAL_LEG", "min_g_FINAL_ENGAGED",
                "min_C_ALL", "min_C_ENGAGED", "min_C_FINAL_ENGAGED", "max_phi"):
        hv = [r[key] for r in H if np.isfinite(r[key])]
        mv = [r[key] for r in M if np.isfinite(r[key])]
        if not hv or not mv:
            print(f"{key:<24}{'(inf)':<14}{'(inf)':<14}{'-':<14}unusable")
            continue
        sep = np.mean(hv) - np.mean(mv)
        scale = max(1e-9, np.std(hv + mv))
        verdict = ("USABLE" if abs(sep) > 0.5 * scale and abs(sep) > 1e-3
                   else "no signal")
        print(f"{key:<24}{np.mean(hv):<14.5f}{np.mean(mv):<14.5f}{sep:<+14.5f}{verdict}")

    # ---------------- hypothesis verdicts -------------------------------- #
    print("\n" + "-" * 100)
    print("HYPOTHESIS CHECKS")
    print("-" * 100)
    inactive_argmin = sum(1 for r in rows if not r["argmin_engaged"])
    print(f"H2  minimum attained at an INACTIVE constraint: "
          f"{inactive_argmin}/{len(rows)} candidates")
    early_leg = sum(1 for r in rows if r["argmin_leg"] == "G0->G1'")
    print(f"H1  minimum attained on the FIRST leg (not the attack leg): "
          f"{early_leg}/{len(rows)} candidates")
    eng = np.mean([r["pct_engaged"] for r in rows])
    print(f"    filter actually engaged on {eng:.1f}% of steps on average")
    cr = np.mean([r["C_range"] for r in rows])
    print(f"H4  C varies by {cr:.5f} within a trajectory on average")
    print(f"H3  demand eta = {a.eta};  typical C ~ "
          f"{np.mean([r['min_C_ALL'] for r in rows if np.isfinite(r['min_C_ALL'])]):.4f}"
          f"  =>  g ~ C - eta is dominated by the constant")
    gu = [r["n_gave_up"] for r in rows]
    print(f"H5  give-ups: hits={[r['n_gave_up'] for r in H]}  misses={[r['n_gave_up'] for r in M]}")
    print("=" * 100)

    json.dump(rows, open("/tmp/guidance_diagnosis.json", "w"), indent=2, default=float)
    print("wrote /tmp/guidance_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
