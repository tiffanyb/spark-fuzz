"""Static-obstacle insertion attacks at the STOCK obstacle size.

Constraints this search respects, which the earlier ones did not:

  * obstacle radius fixed at 0.05 m -- the size the SPARK benchmark ships
    (10 cm across). Nothing is resized to make an attack fit.
  * fewer than 10 obstacles; these scenes use one.
  * d_min stays in a deployable range. SPARK's own configs span 0.02 (this
    study) to 0.1 (its benchmark), with 0.03 in the WBC example, so the ladder
    only goes 0.020 -> 0.015 and never near zero.
  * eta and lambda are free -- they are deployment tuning, and SPARK ships
    eta_ssa = 0.5 / lambda = 10.0 against this study's eta = 0.02.

Two corrections over separated_search:

1. The baseline is [G0, G1], not [G1]. big_obstacle.evaluate checks the single
   waypoint [G1] -- home straight to the goal -- but the attack is [G0, G1', G1],
   so the run the attack must be compared against is the same handover at G0 with
   the inserted waypoint removed. The separation geometry was measured against
   [G0, G1] all along, so checking [G1] validated a different task than the one
   the placement was built for.

2. Spots are not limited to the single best-separated point per G1'. With the
   radius pinned, what matters is how much room is left AROUND the sphere:
   the attacks land in a narrow band of sep - R, so every qualifying point in
   that band is a candidate placement, not just the argmax.

    python -m fuzz.siren.constructed.stock_radius --phase spots
    python -m fuzz.siren.constructed.stock_radius --phase hunt
"""

import argparse
import gc
import json
import os

import numpy as np

from .separated_search import CASE, PARK, goal_grid, park_all

R_STOCK = 0.05
SPOTS = "fuzz/siren/constructed/rq1_results/stock_spots"
OUT = "fuzz/siren/constructed/rq1_results/stock_attacks"


def run(world, schedule, pos_w, R, steps):
    """Rollout with the obstacles pinned; None if the solver gave out.

    SPARK's whole-body IK is a casadi problem that can fail outright
    ("Maximum_Iterations_Exceeded"), and the exception escapes World.run. A scan
    that does not catch it loses every remaining setting the moment one
    configuration is hard to solve -- one such failure ended a cbf scan after 10
    of 500 rollouts. A failed solve is not evidence about the filter, so the
    setting is skipped rather than scored.
    """
    from .big_obstacle import run_pinned
    try:
        return run_pinned(world, [np.asarray(x, float) for x in schedule],
                          pos_w, R, steps)
    except Exception as e:
        print(f"     solver failed ({type(e).__name__}) -- setting skipped",
              flush=True)
        return None


def free_worlds(*names_locals):
    """Drop Worlds and reclaim promptly.

    Each World holds a MuJoCo model plus a SPARK harness -- measured at about
    1.06 GB. Letting one fall out of scope is not enough: six built and dropped
    left RSS 3.6 GB above baseline, and a hunt building 66 Worlds in three
    minutes climbed 4.6 -> 15.3 GB and kept climbing. Collecting explicitly
    after each use holds the cost to ~82 MB per World, a 13x reduction.

    This is not a tidiness nicety. Running six such workers in parallel is what
    starved the machine: the kernel panicked with a userspace watchdog timeout
    (no WindowServer check-in for 122 s) under the memory pressure.
    """
    gc.collect()


def legwise(rec):
    """min clearance per leg, first leg to make contact, step count."""
    per, hit = {}, None
    for s in rec.steps:
        wp = int(s.wp_idx)
        per[wp] = min(per.get(wp, np.inf), float(s.clearance))
        if s.clearance < 0.0 and hit is None:
            hit = wp
    return per, hit, len(rec.steps)


def phase_spots(seed, grid, steps, gap_lo, gap_hi, per_goal):
    """Every placement whose free room around a 5 cm sphere lands in the band."""
    from ..pipeline import trialconf
    from ..world.run import World
    from .separated_search import legs_sweep
    from .swept import surface_gap, sweep_points, tile_radii_for

    cfg = trialconf.load("fuzz/siren/pipeline/configs/trial2_ours.yaml")
    spec0 = trialconf.spec_for(cfg, "ssa", CASE)
    w = World.build(seed=seed, spec=spec0, test_case=CASE, max_steps=steps)
    sc = w.scene()
    G0, G1 = np.asarray(sc.G0, float), np.asarray(sc.G1, float)
    bounds = [(float(x), float(y)) for x, y in sc.bounds]
    park_all(w)
    vol_r = None
    B = sweep_points(w, [G0, G1], steps)
    from .swept import volume_radii
    vol_r = volume_radii(w)
    nv = len(vol_r)
    B = B.reshape(-1, nv, 3)[:: max(1, len(B) // nv // 1500)].reshape(-1, 3)
    rB = tile_radii_for(vol_r, len(B))
    print(f"seed {seed}: baseline sweep {len(B)} pts, "
          f"{nv} collision volumes (radii {vol_r.min():.2f}-{vol_r.max():.2f} m)",
          flush=True)

    spots = []
    for i, G1p in enumerate(goal_grid(bounds, grid)):
        w2 = World.build(seed=seed, spec=spec0, test_case=CASE, max_steps=steps)
        park_all(w2)
        legs = legs_sweep(w2, [G0, G1p, G1], steps)
        w2 = None
        free_worlds()
        leg1, leg2 = legs.get(1, np.zeros((0, 3))), legs.get(2, np.zeros((0, 3)))
        if len(leg1) == 0 or len(leg2) == 0:
            continue
        # Subsample by whole sweeps so the point order still matches the
        # CollisionVol order the radii are tiled from.
        nv = len(vol_r)
        A = leg2.reshape(-1, nv, 3)[:: max(1, len(leg2) // nv // 1500)].reshape(-1, 3)
        L1 = leg1.reshape(-1, nv, 3)[:: max(1, len(leg1) // nv // 1500)].reshape(-1, 3)
        # No radii for A: those are probe points (obstacle centres), so only the
        # obstacle radius is charged on that side, inside surface_gap.
        r1 = tile_radii_for(vol_r, len(L1))
        # free space between a radius-R sphere at each return-leg point and the
        # nearest SURFACE of the legitimate volume / the outbound leg
        db = surface_gap(A, B, rB, R_STOCK)
        d1 = surface_gap(A, L1, r1, R_STOCK)
        gap = np.minimum(db, d1)
        # the old centre-to-centre quantity, kept so placements recorded under
        # the previous metric remain locatable
        legacy = np.minimum(
            np.min(np.linalg.norm(A[:, None] - B[None], axis=2), axis=1),
            np.min(np.linalg.norm(A[:, None] - L1[None], axis=2), axis=1)) - R_STOCK
        ok = (gap >= gap_lo) & (gap <= gap_hi)
        if not ok.any():
            continue
        pts, gs, lg = A[ok], gap[ok], legacy[ok]
        # Keep the two components, not just their minimum. An insertion has to
        # clear TWO volumes and the failures are asymmetric: a placement tight
        # against the baseline is fine (the filter routes around it and the
        # legitimate task still reaches G1), but one inside the OUTBOUND
        # corridor is fatal -- leg 1 collides, world/run.py:183 breaks on the
        # first negative clearance, and the return leg never executes. Stored
        # as a single min the two are indistinguishable, which is what left
        # seeds 3/5/6/7 with zero leg-2 contacts over 4148 Worlds.
        gb_, g1_ = db[ok], d1[ok]
        # Least-overlapping first. Under the old centre-to-centre metric the
        # productive end of the band was the TIGHTEST fit, so this sorted
        # ascending. With surface separation the sign flips: every placement
        # overlaps the legitimate volume, and the ones that overlap LEAST are
        # the ones that can leave the legitimate task completable. Sorting
        # ascending here would now select the placements most likely to break
        # the baseline outright.
        order = np.argsort(-gs)
        chosen = []
        for j in order:
            if all(np.linalg.norm(pts[j] - c) > 0.03 for c in chosen):
                chosen.append(pts[j])
                spots.append({"G1_prime": G1p.tolist(),
                              "spot": pts[j].tolist(),
                              "gap": float(gs[j]),
                              "gap_baseline": float(gb_[j]),
                              "gap_leg1": float(g1_[j]),
                              "gap_centre_legacy": float(lg[j])})
            if len(chosen) >= per_goal:
                break
        print(f"  [{i:3d}] G1'={np.round(G1p,3)} {int(ok.sum())} in band -> "
              f"{len(chosen)} spots (tightest gap {gs[order[0]]:.4f})",
              flush=True)

    spots.sort(key=lambda s: s["gap"])
    spot_file = f"{SPOTS}_{seed}.json"
    os.makedirs(os.path.dirname(spot_file), exist_ok=True)
    # Write via a temp file and rename. A hunt worker in another process reads
    # this corpus, and json.dump straight onto the live path leaves a window
    # where that reader sees a truncated file; os.replace is atomic on POSIX.
    tmp_file = f"{spot_file}.tmp"
    json.dump({"case": CASE, "seed": seed, "R": R_STOCK,
               "metric_version": 3,
               "metric": "gap = free space between a radius-R sphere at the "
                         "placement and the nearest SURFACE of the legitimate "
                         "swept volume or the outbound leg; negative means the "
                         "sphere overlaps it. gap_centre_legacy is the previous "
                         "centre-to-centre quantity, which omitted the robot's "
                         "own collision radius and overstated free space by "
                         "0.05-0.10 m.",
               "grid": grid, "gap_lo": gap_lo, "gap_hi": gap_hi,
               "per_goal": per_goal,
               "G0": G0.tolist(), "G1": G1.tolist(), "bounds": bounds,
               "spots": spots}, open(tmp_file, "w"), indent=1)
    os.replace(tmp_file, spot_file)
    print(f"\n{len(spots)} placements at R={R_STOCK} -> {spot_file}", flush=True)
    return 0


def phase_hunt(algos, steps, ladder, dmins, want, max_spots,
               seed, gap_target=0.027, ks=None, min_leg1=None):
    from ..pipeline import trialconf
    from ..world.run import World
    from ..world.types import real_filter

    spot_file = f"{SPOTS}_{seed}.json"
    s = json.load(open(spot_file))
    # Try the placements whose free room matches the band the confirmed attacks
    # landed in first. Every insertion found so far sits at sep - R between
    # 0.024 and 0.030: tighter and the filter simply stops short of the sphere,
    # looser and it has room to route around. Ordering tightest-first spent the
    # budget on the end of the range that never produces a hit.
    # Drop placements buried in the OUTBOUND corridor. gap is min(baseline,
    # leg1), so a very negative gap can mean either "tight against the
    # legitimate path" (fine -- the filter routes around it) or "inside leg 1"
    # (fatal -- leg 1 collides, the rollout stops at world/run.py:183 and the
    # return leg never runs). Only the second kind is worthless for an
    # insertion, and only gap_leg1 separates them.
    if min_leg1 is not None:
        have = [x for x in s["spots"] if "gap_leg1" in x]
        if not have:
            print(f"--min-leg1 ignored: {spot_file} is metric_version "
                  f"{s.get('metric_version')} and records no gap_leg1; "
                  f"regenerate with --phase spots to use it", flush=True)
        else:
            before = len(s["spots"])
            s["spots"] = [x for x in have if x["gap_leg1"] >= min_leg1]
            print(f"--min-leg1 {min_leg1}: {len(s['spots'])}/{before} "
                  f"placements clear the outbound leg", flush=True)
            if not s["spots"]:
                raise SystemExit(
                    f"no placement in {spot_file} has gap_leg1 >= {min_leg1}; "
                    f"max available is "
                    f"{max(x['gap_leg1'] for x in have):.4f}")
    s["spots"].sort(key=lambda x: abs(x["gap"] - gap_target))
    G0, G1 = np.asarray(s["G0"], float), np.asarray(s["G1"], float)
    # The corpus records the seed it was generated on. Trust the file over the
    # argument, but refuse to run if they disagree -- silently ignoring --seed
    # would hunt one scene while reporting another.
    if int(s["seed"]) != int(seed):
        raise SystemExit(f"--seed {seed} but {spot_file} was generated on seed "
                         f"{s['seed']}; regenerate the corpus or pass --seed "
                         f"{s['seed']}")
    cfg = trialconf.load("fuzz/siren/pipeline/configs/trial2_ours.yaml")
    os.makedirs(OUT, exist_ok=True)

    w0 = World.build(seed=seed,
                     spec=trialconf.spec_for(cfg, "ssa", CASE),
                     test_case=CASE, max_steps=steps)
    n_obs = len(w0.scene().obstacles_world)

    total = 0
    for algo in algos:
        base = trialconf.spec_for(cfg, algo, CASE)
        const = base.demand_shape == "constant"
        hits, seen = 0, []
        # stock settings first, so an attack that needs no retuning is preferred
        # phi_k weights the closing-velocity term of the second-order index,
        # so a small k makes phi nearly distance-only on an acceleration
        # controlled robot -- the filter stops anticipating and reacts late.
        # SPARK ships both 0.1 (its WBC example) and 1.0 (its benchmark), so
        # the whole ladder stays inside values it deploys.
        settings = [(m, d, kk) for d in dmins for m in ladder
                    for kk in (ks or [base.k])]
        for sp in s["spots"][:max_spots]:
            if hits >= want:
                break
            G1p = np.asarray(sp["G1_prime"], float)
            if any(np.allclose(G1p, q) for q in seen):
                continue
            pos_w = [np.asarray(sp["spot"], float)] + [PARK] * (n_obs - 1)
            # Probe the spot once at stock settings before scanning the ladder.
            # Running the baseline for every (eta, d_min) pair doubled the cost
            # of a search whose hits are rare, and a sphere the return leg never
            # comes near cannot be made to bite by retuning the filter -- the
            # demand terms only matter once the constraint is active at all.
            # Probe with the WEAKEST setting in the ladder, not the stock one.
            # The probe exists to drop spots the return leg never approaches,
            # but a strong filter holds the arm far from the sphere, so probing
            # at stock pruned exactly the placements that only pay off once the
            # filter is detuned -- cbf, the strongest at lambda=10, lost all of
            # its candidates that way.
            weak = min(ladder)
            probe_spec = real_filter(
                algo=algo, index=base.index, d_min=min(dmins),
                eta=(base.eta or 0.02) * weak if const else base.eta,
                lam=base.lam if const else (base.lam or 10.0) * weak,
                k=min(ks or [base.k]))
            wp_ = World.build(seed=seed, spec=probe_spec, test_case=CASE,
                              max_steps=steps)
            pr = run(wp_, [G0, G1p, G1], pos_w, R_STOCK, steps)
            wp_ = None
            free_worlds()
            if pr is None:
                continue
            pper, phit, _ = legwise(pr)
            if phit is None and pper.get(2, np.inf) > 0.03:
                continue
            for mult, dmin, kval in settings:
                if hits >= want:
                    break
                # No gap >= d_min guard here. It was meant to keep the sphere
                # off the legitimate volume, but gap >= d_min is only the
                # condition for the filter never to ENGAGE on the baseline --
                # the baseline can engage the filter and still reach G1 safely,
                # which is all an insertion requires. The guard threw away every
                # tight placement, and tight is exactly where cbf comes closest
                # (its closest approach over 44 placements, +0.0123, is at gaps
                # of 0.008-0.019). The baseline is measured directly below, so
                # nothing is assumed.
                eta = (base.eta or 0.02) * mult if const else base.eta
                lam = base.lam if const else (base.lam or 10.0) * mult
                spec = real_filter(algo=algo, index=base.index, d_min=dmin,
                                   eta=eta, lam=lam, k=kval)
                w2 = World.build(seed=seed, spec=spec, test_case=CASE,
                                 max_steps=steps)
                atk = run(w2, [G0, G1p, G1], pos_w, R_STOCK, steps)
                if atk is None:
                    continue
                aper, ahit, an = legwise(atk)
                if ahit != 2:
                    w2 = None
                    free_worlds()
                    continue
                w = World.build(seed=seed, spec=spec, test_case=CASE,
                                max_steps=steps)
                b = run(w, [G0, G1], pos_w, R_STOCK, steps)
                if b is None:
                    continue
                bper, bhit, bn = legwise(b)
                w = None
                free_worlds()
                bmin = min(bper.values()) if bper else np.inf
                if b.label != "REACHED" or bmin <= 0:
                    print(f"  {algo:<5} gap={sp['gap']:.4f} leg2 hit but "
                          f"baseline {b.label}{bmin:+.6f} -- not an insertion",
                          flush=True)
                    continue
                dem = (f"eta={eta:.4f}" if const else f"lam={lam:.3f}")
                dem += f" k={kval}"
                print(f"  {algo:<5} gap={sp['gap']:.4f} {dem} dm={dmin:.3f} "
                      f"base REACHED{bmin:+.6f} | {atk.label} "
                      f"leg1 {aper.get(1, float('nan')):+.5f} "
                      f"leg2 {aper.get(2, float('nan')):+.6f}  <== INSERTION",
                      flush=True)
                # fresh world: the IK warm start persists across reset(), so a
                # hit inside a reused World is not yet a result
                w2 = None
                free_worlds()
                w3 = World.build(seed=seed, spec=spec, test_case=CASE,
                                 max_steps=steps)
                a2 = run(w3, [G0, G1p, G1], pos_w, R_STOCK, steps)
                w3 = None
                free_worlds()
                if a2 is None:
                    continue
                p2, h2, _ = legwise(a2)
                if h2 != 2:
                    print("     did not reproduce in a fresh world", flush=True)
                    continue
                json.dump({"case": CASE, "algo": algo, "seed": seed,
                           "index": spec.index, "d_min": dmin,
                           "eta": eta, "lam": lam or 1.0, "k": kval,
                           "demand_multiplier": mult,
                           "stock_config": bool(mult == 1.0 and dmin == 0.02
                                                and kval == base.k),
                           "max_steps": steps, "channel": "arm",
                           "bounds": s["bounds"], "keepout": 0.0,
                           "obstacles_world": [list(map(float, q))
                                               for q in pos_w],
                           "obstacle_radius": R_STOCK, "n_obstacles": 1,
                           "u_lim_scale": 1.0, "plant_modified": False,
                           "static": True, "env_only": False,
                           "gap": sp["gap"],
                           "G1": G1.tolist(), "G1_prime": G1p.tolist(),
                           "controls": [{"G0": G0.tolist()}],
                           "baseline_label": b.label, "baseline_min": float(bmin),
                           "baseline_steps": bn,
                           "leg1_min": float(aper.get(1, np.inf)),
                           "leg2_min": float(min(aper.get(2, np.inf),
                                                 p2.get(2, np.inf))),
                           "hit_leg": "INSERTION"},
                          open(f"{OUT}/{algo}_{hits}_{seed}.json", "w"),
                          indent=1, default=float)
                hits += 1
                total += 1
                seen.append(G1p)
                print(f"*** STATIC INSERTION {algo} #{hits}  R=0.05  {dem}  "
                      f"d_min={dmin}  baseline {bmin:+.6f}  "
                      f"leg2 {p2.get(2, np.inf):+.6f}", flush=True)
        print(f"{algo}: {hits}", flush=True)
    print(f"TOTAL {total}", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["spots", "hunt"], required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--min-leg1", type=float, default=None,
                   help="hunt only placements whose surface gap to the "
                        "OUTBOUND leg is at least this; keeps the rollout "
                        "alive to leg 2 (needs a metric_version 3 corpus)")
    p.add_argument("--gap-lo", type=float, default=-0.034,
                   help="metric v2: true surface separation. The v1 band "
                        "[0.016, 0.040] was centre-to-centre and corresponds to "
                        "[-0.034, -0.010] here.")
    p.add_argument("--gap-hi", type=float, default=-0.010)
    p.add_argument("--per-goal", type=int, default=6)
    p.add_argument("--max-spots", type=int, default=40)
    p.add_argument("--ks", default=None,
                   help="phi_k values to scan (SPARK ships 0.1 and 1.0)")
    p.add_argument("--gap-target", type=float, default=-0.023,
                   help="free room sep-R the confirmed attacks land at; spots "
                        "are tried in order of distance from it")
    p.add_argument("--want", type=int, default=1)
    p.add_argument("--ladder", default="1,0.5,0.2,2,5,0.05")
    p.add_argument("--dmins", default="0.020,0.018,0.015")
    p.add_argument("--algos", default="ssa,rssa,pssa,cbf,rcbf,sss,rsss")
    a = p.parse_args(argv)
    if a.phase == "spots":
        return phase_spots(a.seed, a.grid, a.steps, a.gap_lo, a.gap_hi, a.per_goal)
    else:
        return phase_hunt(a.algos.split(","), a.steps,
                          [float(x) for x in a.ladder.split(",")],
                          [float(x) for x in a.dmins.split(",")],
                          a.want, a.max_spots, a.seed, a.gap_target,
                          ([float(x) for x in a.ks.split(",")] if a.ks else None),
                          a.min_leg1)


if __name__ == "__main__":
    raise SystemExit(main())
