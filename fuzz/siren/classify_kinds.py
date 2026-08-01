"""
For every collision on one scene: was it Kind 1 or Kind 2?

    Kind 1  the filter was ALREADY infeasible the first moment it intervened.
            Nothing it could have done would have helped -- the robot arrived at
            the engagement point already doomed. Blame lies upstream of the
            filter (configuration, authority, how late the index engages).

    Kind 2  the filter WAS feasible when it first intervened, acted, and the
            state it steered into was later infeasible. The filter walked the
            robot into the trap. This is the myopia result, and it is the one
            that makes the paper's argument.

Separating them needs per-step timing, which the existing three-way probe in
run_experiment.classify_mechanism does not record -- it answers the different
question of whether SEQUENCING two goals broke a filter that handles each goal
alone. Both are reported here, because they are complementary: the timing says
WHEN feasibility was lost, the three-way probe says whether the attacker's
ordering CAUSED it.

Infeasibility is read two independent ways, and they are NOT merged:

    predicted   g < 0        the certificate says no safe control exists
    observed    gave_up      the solver actually failed to find one

They can disagree, and the disagreement matters: g is computed from a
single-constraint margin, so a pincer of several constraints that are jointly
unsatisfiable can show g > 0 while the solver still fails. Reporting only one
would hide that.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python -m fuzz.siren.classify_kinds --seed 1 --case G1FixedBase_D2_AG_SO_v0 --n 80
"""

import argparse
import json

import numpy as np

from .search.pick import sample_admissible
from .world.run import World
from .world.types import real_filter


def timeline(rec, leg2_only=True):
    """When did the filter engage on LEG 2, and when did it lose feasibility?

    Restricted to leg 2 (wp_idx >= 1) on purpose. Leg 1 is the attacker's setup
    move; the trap is sprung on the departure toward the legitimate goal, so t0
    must be the first intervention AFTER the handover. Scanning the whole
    schedule would let an engagement during the approach to G1' stand in for t0
    and misclassify a Kind 2 as a Kind 1.
    """
    steps = [s for s in rec.steps if (s.wp_idx >= 1 or not leg2_only)]
    if not steps:
        steps = rec.steps

    def first(pred):
        return next((s.step for s in steps if pred(s)), None)

    # mu > 0 means NO control satisfies every active constraint. mu is nan on
    # steps where nothing is engaged, and nan comparisons are False, so an
    # unengaged step can never be mistaken for an infeasible one.
    return {
        "n_steps": len(rec.steps),
        "n_steps_leg2": len(steps),
        "t_engage": first(lambda s: s.engaged),
        "t_gave_up": first(lambda s: s.gave_up),
        "t_mu_positive": first(lambda s: np.isfinite(s.mu) and s.mu > 1e-9),
        "t_g_negative": first(lambda s: np.isfinite(s.g) and s.g < 0.0),
        "n_engaged": sum(1 for s in steps if s.engaged),
        "n_gave_up": rec.n_gave_up,
        "max_mu": max((float(s.mu) for s in steps if np.isfinite(s.mu)),
                      default=None),
        "min_g_engaged": float(rec.min_g) if np.isfinite(rec.min_g) else None,
        "max_phi": max((float(s.phi) for s in steps
                        if np.isfinite(s.phi)), default=None),
        "min_clearance": float(rec.min_clearance),
    }


def classify(tl, signal):
    """Kind from the timing of one infeasibility signal.

    `signal` is the step at which feasibility was first lost (or None). The
    comparison against t_engage is what distinguishes the two kinds; a filter
    that was never engaged at all is neither kind, and saying so is the point --
    lumping it in with Kind 1 would credit the filter with a failure it was never
    consulted about.
    """
    t_e = tl["t_engage"]
    if t_e is None:
        return "NEVER_ENGAGED"
    if signal is None:
        return "NO_INFEASIBILITY"
    if signal <= t_e:
        return "KIND_1"
    return "KIND_2"


def three_way(world, scene, cand, max_steps):
    """Did SEQUENCING break a filter that handles each goal alone?"""
    base = world.run([np.asarray(scene.G1)], max_steps=max_steps)
    one = world.run([np.asarray(cand)], max_steps=max_steps)
    legs_clean = (base.reached and one.reached
                  and base.n_gave_up == 0 and one.n_gave_up == 0)
    return {
        "baseline_label": base.label, "baseline_gave_up": base.n_gave_up,
        "onehop_label": one.label, "onehop_gave_up": one.n_gave_up,
        "legs_clean": bool(legs_clean),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--case", default="G1FixedBase_D2_AG_SO_v0")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--d-min", type=float, default=0.02)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--out", default="/tmp/kinds.json")
    a = p.parse_args(argv)

    index = "velocity" if "_D2_" in a.case else "distance"
    spec = real_filter(algo="ssa", index=index, d_min=a.d_min, eta=a.eta, k=0.1)
    world = World.build(seed=a.seed, spec=spec, test_case=a.case,
                        max_steps=a.max_steps)
    scene = world.scene()
    print(f"[scene] {a.case} seed={a.seed} eta={a.eta}", flush=True)

    # Same sampler and search seed as plot_scene, so the collisions examined here
    # are the same ones drawn in the picture.
    rng = np.random.RandomState(a.search_seed)
    rows, tried, attempts = [], 0, 0
    leg1 = {"COLLISION": 0, "DEADLOCK": 0, "TIMEOUT": 0, "REACHED": 0}
    while tried < a.n and attempts < a.n * 20:
        attempts += 1
        cand = sample_admissible(rng, scene)
        if cand is None:
            continue

        # Step 1 of the pipeline. The three leg-1 outcomes are NOT the same
        # event and must not be collapsed into one "discard": a leg that
        # collides or deadlocks on its own is already a modification hit and is
        # counted as such, while a timeout is genuinely uninformative. Only a
        # leg that REACHES gives a well-defined state to depart from.
        screen = world.run([cand], max_steps=a.max_steps)
        leg1[screen.label] = leg1.get(screen.label, 0) + 1
        if screen.label != "REACHED":
            continue

        # Step 2: one continuous run, so leg 2 departs from the state leg 1
        # actually ended in (in motion), not from a fresh settle at G1'.
        rec = world.run([cand, np.asarray(scene.G1)], max_steps=a.max_steps,
                        exact_margin=True)
        tried += 1
        if rec.label != "COLLISION":
            continue
        tl = timeline(rec)
        row = {"candidate": [float(x) for x in cand], "label": rec.label, **tl}
        row["kind_mu"] = classify(tl, tl["t_mu_positive"])       # the design's test
        row["kind_g"] = classify(tl, tl["t_g_negative"])         # single-constraint proxy
        row["kind_observed"] = classify(tl, tl["t_gave_up"])     # solver actually failed
        row.update(three_way(world, scene, cand, a.max_steps))
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} collisions classified "
                  f"({tried}/{a.n} screened)", flush=True)

    # ------------------------------- report ------------------------------- #
    print(f"\n{'='*74}\nD2 seed {a.seed}: {len(rows)} COLLISIONS of {tried} "
          f"inserted goals\n{'='*74}")
    print(f"\nleg-1 outcomes over {attempts} sampled goals: {leg1}"
          f"\n   (COLLISION/DEADLOCK on leg 1 = modification hits, not insertion "
          f"candidates; TIMEOUT discarded; only REACHED continues to leg 2)")
    for key, title in (("kind_mu", "by EXACT LP margin  mu(x_t0) > 0   <- the designed test"),
                       ("kind_g", "by single-constraint proxy  g < 0"),
                       ("kind_observed", "by OBSERVED infeasibility (solver gave up)")):
        counts = {}
        for r in rows:
            counts[r[key]] = counts.get(r[key], 0) + 1
        print(f"\n{title}")
        for k in ("KIND_1", "KIND_2", "NO_INFEASIBILITY", "NEVER_ENGAGED"):
            n = counts.get(k, 0)
            print(f"   {k:<20} {n:>3}  ({100.0*n/max(1,len(rows)):.0f}%)")

    eng = [r for r in rows if r["t_engage"] is not None]
    print(f"\nengagement: {len(eng)}/{len(rows)} collisions had the filter engage "
          f"at all before contact")
    if eng:
        gaps = [r["n_steps"] - r["t_engage"] for r in eng]
        print(f"   steps between first engagement and the collision: "
              f"median {np.median(gaps):.0f}, min {min(gaps)}, max {max(gaps)}")
    clean = sum(1 for r in rows if r["legs_clean"])
    print(f"\nthree-way probe: {clean}/{len(rows)} had BOTH legs clean on their own "
          f"(so the ordering is what broke them)")

    json.dump({"seed": a.seed, "case": a.case, "eta": a.eta,
               "n_screened": tried, "rows": rows}, open(a.out, "w"),
              indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
