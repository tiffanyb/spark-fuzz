"""Static insertion by maximising SWEPT-VOLUME SEPARATION over G1'.

Placing an obstacle on the return leg broke the legitimate task in 36 of 42
attempts -- the baseline collided too, so there was nothing to attack. That is
not evidence the filter is safe; it is evidence the two swept volumes overlapped
wherever the obstacle went.

The fix is to choose G1' for separation rather than picking one and hoping. The
quantity that decides whether a static insertion exists is

    sep(G1') = max over return-leg points p of  min over baseline points b  |p-b|

An obstacle of radius R centred at that argmax leaves the baseline untouched
whenever sep > R + d_min + clear, yet sits directly on the return path. Earlier
runs used one G1' close to G1, where the legs converge and sep is small; sep
grows with how far the detour carries the arm off the legitimate path.

Geometry is computed ONCE and reused across all seven filters: with every
obstacle parked far away the filter never engages, so the rollout is the
reference controller's own path and does not depend on which filter is fitted.

Plant untouched throughout -- u_lim, kinematics and dt are all stock. The only
things varied are obstacle position/radius and the goals, which the attacker
controls.

    python -m fuzz.siren.constructed.separated_search --phase geom
    python -m fuzz.siren.constructed.separated_search --phase test
"""

import argparse
import json
import os

import numpy as np

CASE = "G1FixedBase_D2_AG_SO_v0"
SEED = 1
STEPS = 900
GEOM = "fuzz/siren/constructed/separation_geom.json"
OUT = "fuzz/siren/constructed/final_attacks"
PARK = np.array([9.0, 9.0, 9.0])
#: obstacle radius the separation geometry is computed for
R_STOCK_GEOM = 0.05


def park_all(world):
    """Move every obstacle out of the workspace so the filter stays disengaged."""
    from .big_obstacle import set_obstacles
    n = len(world.scene().obstacles_world)
    set_obstacles(world, [PARK] * n, 0.05)


def legs_sweep(world, schedule, steps):
    """Per-leg swept volumes from ONE rollout: {leg index -> (N,3) centres}.

    sweep_points would need a second rollout per leg, and the outbound and
    return legs must come from the same run to be comparable.
    """
    from ..pipeline.stage1_search import set_channel
    from ..world.sim import probe
    h = world.harness
    probe.reset_giveups(h)
    af, ti = h.reset()
    set_channel(world, "arm")
    h.env.task.set_goal_schedule([np.asarray(x, float) for x in schedule])
    u, ai = h.algo.act(af, ti)
    out = {}
    for _ in range(steps):
        af, ti = h.env.step(u, ai)
        try:
            u, ai = h.algo.act(af, ti)
        except Exception:
            u = np.zeros_like(np.asarray(u, float))
        wp = int(getattr(h.env.task, "wp_idx", 0))
        fr = h.env.task.robot_frames_world
        out.setdefault(wp, []).extend(
            np.asarray(fr[i], float)[:3, 3].copy()
            for i in h.robot_cfg.CollisionVol)
        if h.env.task.reached_final:
            break
    return {k: np.array(v) for k, v in out.items()}


def goal_grid(bounds, n=4):
    """Interior grid over the goal box -- candidate detour targets G1'."""
    axes = [np.linspace(lo + 0.02, hi - 0.02, n) for lo, hi in bounds]
    return [np.array([x, y, z])
            for x in axes[0] for y in axes[1] for z in axes[2]]


def phase_geom(n_grid, sub):
    from ..pipeline import trialconf
    from ..world.run import World
    from .swept import (surface_gap, sweep_points, tile_radii_for,
                        volume_radii)  # noqa: F401

    cfg = trialconf.load("fuzz/siren/pipeline/configs/config2.yaml")
    spec = trialconf.spec_for(cfg, "ssa", CASE)
    w = World.build(seed=SEED, spec=spec, test_case=CASE, max_steps=STEPS)
    sc = w.scene()
    G0 = np.asarray(sc.G0, float)
    G1 = np.asarray(sc.G1, float)
    bounds = [(float(a), float(b)) for a, b in sc.bounds]

    park_all(w)
    base = sweep_points(w, [G0, G1], STEPS)
    vol_r = volume_radii(w)
    nv = len(vol_r)
    B = base.reshape(-1, nv, 3)[:: max(1, len(base) // nv // sub)].reshape(-1, 3)
    print(f"baseline sweep {len(base)} pts (subsampled {len(B)}), "
          f"{nv} collision volumes", flush=True)

    rows = []
    for i, G1p in enumerate(goal_grid(bounds, n_grid)):
        w2 = World.build(seed=SEED, spec=spec, test_case=CASE, max_steps=STEPS)
        park_all(w2)
        legs = legs_sweep(w2, [G0, G1p, G1], STEPS)
        leg1, leg2 = legs.get(1, np.zeros((0, 3))), legs.get(2, np.zeros((0, 3)))
        if len(leg2) == 0 or len(leg1) == 0:
            print(f"  [{i:3d}] G1'={np.round(G1p,3)} return leg never ran",
                  flush=True)
            continue
        # subsample by whole sweeps so point order still matches CollisionVol
        A = leg2.reshape(-1, nv, 3)[:: max(1, len(leg2) // nv // sub)].reshape(-1, 3)
        L1 = leg1.reshape(-1, nv, 3)[:: max(1, len(leg1) // nv // sub)].reshape(-1, 3)
        # An insertion needs contact on the RETURN leg only, so the obstacle has
        # to clear two volumes, not one: the baseline (or the legitimate task
        # breaks and there is nothing to attack) and the OUTBOUND leg (or the
        # contact lands on leg 1, which is a modification, not an insertion).
        # Scoring on the min of the two distances is what separates the cases;
        # scoring on the baseline alone put every contact on leg 1.
        # Surface separation, not centre-to-centre: the robot's own collision
        # spheres (0.05 m arm, up to 0.10 m torso) must be subtracted too, or a
        # placement reported as clear is actually inside the swept volume. `d`
        # now already accounts for the obstacle radius, so callers must not
        # subtract R again.
        db = surface_gap(A, B, tile_radii_for(vol_r, len(B)), R_STOCK_GEOM)
        d1 = surface_gap(A, L1, tile_radii_for(vol_r, len(L1)), R_STOCK_GEOM)
        d = np.minimum(db, d1)
        j = int(np.argmax(d))
        rows.append({"G1_prime": G1p.tolist(), "spot": A[j].tolist(),
                     "sep": float(d[j]), "sep_base": float(db[j]),
                     "sep_leg1": float(d1[j]), "n_leg2": int(len(leg2))})
        print(f"  [{i:3d}] G1'={np.round(G1p,3)} sep={d[j]:.4f} "
              f"(base {db[j]:.3f}, leg1 {d1[j]:.3f}) spot={np.round(A[j],3)}",
              flush=True)

    rows.sort(key=lambda r: -r["sep"])
    json.dump({"case": CASE, "seed": SEED, "G0": G0.tolist(), "G1": G1.tolist(),
               "bounds": bounds, "rows": rows}, open(GEOM, "w"), indent=1)
    print(f"\n{len(rows)} candidates -> {GEOM}")
    for r in rows[:8]:
        print(f"   sep {r['sep']:.4f}  G1'={np.round(r['G1_prime'],3)}")
    # an obstacle of radius R fits without touching the baseline when
    # sep > R + d_min + clear
    # `sep` is now true surface separation for a 0.05 m sphere, so a placement
    # genuinely clears the legitimate volume when sep > 0 (add d_min if the
    # filter must also never engage on it).
    for thr, label in ((0.0, "clear of the swept volume"),
                       (0.02, "clear by d_min as well")):
        k = sum(1 for r in rows if r["sep"] > thr)
        print(f"   sep > {thr}: {k} candidates {label}")
    return 0


def phase_test(top, radii, algos, clear=0.015, offsets=None):
    from ..pipeline import trialconf
    from ..world.run import World
    from .big_obstacle import evaluate

    g = json.load(open(GEOM))
    G0 = np.asarray(g["G0"], float)
    G1 = np.asarray(g["G1"], float)
    cfg = trialconf.load("fuzz/siren/pipeline/configs/config2.yaml")
    os.makedirs(OUT, exist_ok=True)

    total = 0
    for algo in algos:
        spec = trialconf.spec_for(cfg, algo, CASE)
        hits = 0
        for r in g["rows"][:top]:
            if hits >= 2:
                break
            G1p = np.asarray(r["G1_prime"], float)
            spot = np.asarray(r["spot"], float)
            # The three attacks found so far all sit in a narrow band,
            # sep - R in [0.024, 0.030]: the sphere nearly fills the corridor
            # between the return leg and the baseline volume, leaving the filter
            # too little room to route around yet still clearing the legitimate
            # path. Sizing R per candidate from its own sep searches that band
            # directly instead of testing absolute radii that miss it on most
            # candidates.
            rs = ([r["sep"] - o for o in offsets] if offsets else radii)
            for R in rs:
                if R <= 0.01:
                    continue
                if r["sep"] <= R + spec.d_min + clear:
                    continue          # obstacle would foul the baseline
                w = World.build(seed=SEED, spec=spec, test_case=CASE,
                                max_steps=STEPS)
                n = len(w.scene().obstacles_world)
                pw = [spot] + [PARK] * (n - 1)
                ok, info = evaluate(w, G0, G1p, G1, pw, R, STEPS)
                print(f"  {algo:<5} sep={r['sep']:.3f} R={R} "
                      f"base {info['blabel']}{info['bmin']:+.5f} "
                      f"| {info.get('label','-'):<9} "
                      f"leg2 {info.get('leg2', float('nan')):+.6f}"
                      f"{'  <== INSERTION' if ok else ''}", flush=True)
                if not ok:
                    continue
                # reproduce in a fresh world: IK warm-start persists across
                # reset(), so an attack found in a reused World is not yet a
                # result
                w2 = World.build(seed=SEED, spec=spec, test_case=CASE,
                                 max_steps=STEPS)
                ok2, i2 = evaluate(w2, G0, G1p, G1, pw, R, STEPS)
                if not ok2:
                    print("     did not reproduce in a fresh world", flush=True)
                    continue
                rec = {"case": CASE, "algo": algo, "seed": SEED,
                       "index": spec.index, "d_min": spec.d_min,
                       "eta": spec.eta, "lam": spec.lam or 1.0, "k": spec.k,
                       "max_steps": STEPS, "channel": "arm",
                       "bounds": g["bounds"], "keepout": 0.0,
                       "obstacles_world": [list(map(float, q)) for q in pw],
                       "obstacle_radius": float(R), "n_obstacles": 1,
                       "u_lim_scale": 1.0, "plant_modified": False,
                       "static": True, "env_only": True,
                       "separation": r["sep"],
                       "G1": G1.tolist(), "G1_prime": G1p.tolist(),
                       "controls": [{"G0": G0.tolist()}],
                       "baseline_min": i2["bmin"], "leg1_min": i2["leg1"],
                       "leg2_min": i2["leg2"], "hit_leg": "INSERTION"}
                json.dump(rec, open(f"{OUT}/sep_s{SEED}_{algo}_{hits}.json", "w"),
                          indent=1, default=float)
                hits += 1
                total += 1
                print(f"*** STATIC INSERTION {algo} #{hits} "
                      f"(baseline {i2['bmin']:+.5f}, leg2 {i2['leg2']:+.6f})",
                      flush=True)
                break
        print(f"{algo}: {hits} static insertions", flush=True)
    print(f"TOTAL {total}", flush=True)
    return 0


def phase_multi(top, R, K, clear, algos, spacing=2.2, offsets=None,
                prefix='multi'):
    """Several obstacles along the return leg instead of one.

    A single obstacle is easy for the filter: one active constraint, and the
    measured surplus of authority over demand is large, so it deflects and the
    arm goes around. Deflecting away from one obstacle in a CORRIDOR of them
    pushes the arm toward the next, which is the situation the theory says the
    filter cannot solve -- several constraints active at once, with no single
    control satisfying all of them (mu > 0). The scan measured up to 81
    simultaneously active constraints, so the regime is reachable.

    Every obstacle still has to clear the baseline volume and the outbound leg,
    so the legitimate task stays safe and any contact is an insertion.
    """
    from ..pipeline import trialconf
    from ..world.run import World
    from .big_obstacle import evaluate
    from .swept import sweep_points

    g = json.load(open(GEOM))
    G0 = np.asarray(g["G0"], float)
    G1 = np.asarray(g["G1"], float)
    cfg = trialconf.load("fuzz/siren/pipeline/configs/config2.yaml")
    spec0 = trialconf.spec_for(cfg, "ssa", CASE)
    os.makedirs(OUT, exist_ok=True)

    w = World.build(seed=SEED, spec=spec0, test_case=CASE, max_steps=STEPS)
    n_obs = len(w.scene().obstacles_world)
    park_all(w)
    B = sweep_points(w, [G0, G1], STEPS)
    B = B[:: max(1, len(B) // 1500)]

    plans = []
    for r in g["rows"][:top]:
        G1p = np.asarray(r["G1_prime"], float)
        w2 = World.build(seed=SEED, spec=spec0, test_case=CASE, max_steps=STEPS)
        park_all(w2)
        legs = legs_sweep(w2, [G0, G1p, G1], STEPS)
        leg1 = legs.get(1, np.zeros((0, 3)))
        leg2 = legs.get(2, np.zeros((0, 3)))
        if len(leg1) == 0 or len(leg2) == 0:
            continue
        A = leg2[:: max(1, len(leg2) // 1500)]
        L1 = leg1[:: max(1, len(leg1) // 1500)]
        db = np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2), axis=1)
        d1 = np.min(np.linalg.norm(A[:, None, :] - L1[None, :, :], axis=2), axis=1)
        # Size the spheres per candidate from its own separation. The single
        # obstacle attacks all landed at sep - R in [0.024, 0.030]; a fixed
        # radius misses that band on every candidate whose sep differs.
        for R_i in ([r["sep"] - o for o in offsets] if offsets else [R]):
            if R_i <= 0.01:
                continue
            need = R_i + spec0.d_min + clear
            okp = A[(db > need) & (d1 > need)]
            if len(okp) == 0:
                continue
            # spread the corridor out: obstacles must not overlap each other, or
            # they act as one large sphere and the constraints stay collinear
            picked = []
            for p in okp:
                if all(np.linalg.norm(p - q) > spacing * R_i for q in picked):
                    picked.append(p)
                if len(picked) >= min(K, n_obs):
                    break
            if len(picked) >= 2:
                plans.append((G1p, picked, R_i))
                print(f"  plan G1'={np.round(G1p,3)} R={R_i:.4f}: "
                      f"{len(okp)} clear pts -> {len(picked)} obstacles",
                      flush=True)

    print(f"\n{len(plans)} multi-obstacle plans, R={R}\n", flush=True)
    total = 0
    for algo in algos:
        spec = trialconf.spec_for(cfg, algo, CASE)
        hits = 0
        for G1p, picked, R in plans:
            if hits >= 2:
                break
            pw = list(picked) + [PARK] * (n_obs - len(picked))
            w3 = World.build(seed=SEED, spec=spec, test_case=CASE,
                             max_steps=STEPS)
            ok, info = evaluate(w3, G0, G1p, G1, pw, R, STEPS)
            print(f"  {algo:<5} n={len(picked)} base {info['blabel']}"
                  f"{info['bmin']:+.5f} | {info.get('label','-'):<9} "
                  f"leg1 {info.get('leg1', float('nan')):+.5f} "
                  f"leg2 {info.get('leg2', float('nan')):+.6f}"
                  f"{'  <== INSERTION' if ok else ''}", flush=True)
            if not ok:
                continue
            w4 = World.build(seed=SEED, spec=spec, test_case=CASE,
                             max_steps=STEPS)
            ok2, i2 = evaluate(w4, G0, G1p, G1, pw, R, STEPS)
            if not ok2:
                print("     did not reproduce in a fresh world", flush=True)
                continue
            json.dump({"case": CASE, "algo": algo, "seed": SEED,
                       "index": spec.index, "d_min": spec.d_min,
                       "eta": spec.eta, "lam": spec.lam or 1.0, "k": spec.k,
                       "max_steps": STEPS, "channel": "arm",
                       "bounds": g["bounds"], "keepout": 0.0,
                       "obstacles_world": [list(map(float, q)) for q in pw],
                       "obstacle_radius": float(R),
                       "n_obstacles": len(picked),
                       "u_lim_scale": 1.0, "plant_modified": False,
                       "static": True, "env_only": True,
                       "G1": G1.tolist(), "G1_prime": G1p.tolist(),
                       "controls": [{"G0": G0.tolist()}],
                       "baseline_min": i2["bmin"], "leg1_min": i2["leg1"],
                       "leg2_min": i2["leg2"], "hit_leg": "INSERTION"},
                      open(f"{OUT}/{prefix}_{algo}_{hits}.json", "w"),
                      indent=1, default=float)
            hits += 1
            total += 1
            print(f"*** STATIC INSERTION {algo} #{hits} "
                  f"({len(picked)} obstacles, baseline {i2['bmin']:+.5f}, "
                  f"leg2 {i2['leg2']:+.6f})", flush=True)
        print(f"{algo}: {hits} static insertions", flush=True)
    print(f"TOTAL {total}", flush=True)
    return 0


def phase_demand(top, R, K, clear, algos, spacing, ladder,
                 dmins=None, prefix='dem'):
    """Scan the filter's DEMAND on separated corridor geometry.

    The corridor already leaves the return leg tighter than the baseline (leg 2
    at +0.0104 while the baseline holds +0.019 at eta=0.02), so the two legs are
    ordered: as demand rises, the filter runs out of authority on the return leg
    BEFORE it does on the legitimate one. That ordering is what turns a demand
    scan into an insertion rather than a scene that simply breaks.

    Demand is a deployment setting, not a plant property: SPARK ships
    eta_ssa = 0.5 and lambda = 10.0, while this study's config uses eta = 0.02,
    so the ladder stays inside values SPARK itself deploys. The mechanism is the
    c(x) < d(x) condition directly -- the geometry fixes supply, the ladder
    raises demand until it crosses.
    """
    from ..pipeline import trialconf
    from ..world.run import World
    from ..world.types import real_filter
    from .big_obstacle import evaluate
    from .swept import sweep_points

    g = json.load(open(GEOM))
    G0 = np.asarray(g["G0"], float)
    G1 = np.asarray(g["G1"], float)
    cfg = trialconf.load("fuzz/siren/pipeline/configs/config2.yaml")
    spec0 = trialconf.spec_for(cfg, "ssa", CASE)
    os.makedirs(OUT, exist_ok=True)

    w = World.build(seed=SEED, spec=spec0, test_case=CASE, max_steps=STEPS)
    n_obs = len(w.scene().obstacles_world)
    park_all(w)
    B = sweep_points(w, [G0, G1], STEPS)
    B = B[:: max(1, len(B) // 1500)]

    plans = []
    for r in g["rows"][:top]:
        G1p = np.asarray(r["G1_prime"], float)
        w2 = World.build(seed=SEED, spec=spec0, test_case=CASE, max_steps=STEPS)
        park_all(w2)
        legs = legs_sweep(w2, [G0, G1p, G1], STEPS)
        leg1 = legs.get(1, np.zeros((0, 3)))
        leg2 = legs.get(2, np.zeros((0, 3)))
        if len(leg1) == 0 or len(leg2) == 0:
            continue
        A = leg2[:: max(1, len(leg2) // 1500)]
        L1 = leg1[:: max(1, len(leg1) // 1500)]
        need = R + spec0.d_min + clear
        db = np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2), axis=1)
        d1 = np.min(np.linalg.norm(A[:, None, :] - L1[None, :, :], axis=2), axis=1)
        okp = A[(db > need) & (d1 > need)]
        picked = []
        for p in okp:
            if all(np.linalg.norm(p - q) > spacing * R for q in picked):
                picked.append(p)
            if len(picked) >= min(K, n_obs):
                break
        if picked:
            plans.append((G1p, picked))
            print(f"  plan G1'={np.round(G1p,3)}: {len(picked)} obstacles",
                  flush=True)

    print(f"\n{len(plans)} plans, R={R}, ladder={ladder}\n", flush=True)
    total = 0
    for algo in algos:
        base_spec = trialconf.spec_for(cfg, algo, CASE)
        const = base_spec.demand_shape == "constant"
        hits = 0
        for G1p, picked in plans:
            if hits >= 2:
                break
            pw = list(picked) + [PARK] * (n_obs - len(picked))
            for mult, dmin in [(m, d) for m in ladder
                               for d in (dmins or [base_spec.d_min])]:
                if hits >= 2:
                    break
                eta = (base_spec.eta or 0.02) * mult if const else base_spec.eta
                lam = base_spec.lam if const else (base_spec.lam or 10.0) * mult
                # d_min sets how far out the shell reaches, so shrinking it makes
                # the filter engage LATER -- less distance to bleed off speed in.
                # The baseline is untouched by it: the obstacle clears the
                # legitimate swept volume by construction, so that run never
                # engages the filter at any d_min.
                spec = real_filter(algo=algo, index=base_spec.index,
                                   d_min=dmin, eta=eta, lam=lam, k=base_spec.k)
                w3 = World.build(seed=SEED, spec=spec, test_case=CASE,
                                 max_steps=STEPS)
                ok, info = evaluate(w3, G0, G1p, G1, pw, R, STEPS)
                dem = (f"eta={eta:.3f}" if const else f"lam={lam:.2f}")
                dem += f" dm={dmin:.3f}"
                print(f"  {algo:<5} n={len(picked)} {dem:<11} "
                      f"base {info['blabel']}{info['bmin']:+.5f} | "
                      f"{info.get('label','-'):<9} "
                      f"leg1 {info.get('leg1', float('nan')):+.5f} "
                      f"leg2 {info.get('leg2', float('nan')):+.6f}"
                      f"{'  <== INSERTION' if ok else ''}", flush=True)
                if not ok:
                    continue
                w4 = World.build(seed=SEED, spec=spec, test_case=CASE,
                                 max_steps=STEPS)
                ok2, i2 = evaluate(w4, G0, G1p, G1, pw, R, STEPS)
                if not ok2:
                    print("     did not reproduce in a fresh world", flush=True)
                    continue
                json.dump({"case": CASE, "algo": algo, "seed": SEED,
                           "index": spec.index, "d_min": spec.d_min,
                           "eta": spec.eta, "lam": spec.lam or 1.0, "k": spec.k,
                           "max_steps": STEPS, "channel": "arm",
                           "bounds": g["bounds"], "keepout": 0.0,
                           "obstacles_world": [list(map(float, q)) for q in pw],
                           "obstacle_radius": float(R),
                           "n_obstacles": len(picked),
                           "u_lim_scale": 1.0, "plant_modified": False,
                           "static": True, "env_only": True,
                           "demand_multiplier": float(mult),
                           "G1": G1.tolist(), "G1_prime": G1p.tolist(),
                           "controls": [{"G0": G0.tolist()}],
                           "baseline_min": i2["bmin"], "leg1_min": i2["leg1"],
                           "leg2_min": i2["leg2"], "hit_leg": "INSERTION"},
                          open(f"{OUT}/{prefix}_{algo}_{hits}.json", "w"),
                          indent=1, default=float)
                hits += 1
                total += 1
                print(f"*** STATIC INSERTION {algo} #{hits}  {dem}  "
                      f"{len(picked)} obstacles  baseline {i2['bmin']:+.5f}  "
                      f"leg2 {i2['leg2']:+.6f}", flush=True)
        print(f"{algo}: {hits} static insertions", flush=True)
    print(f"TOTAL {total}", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["geom", "test", "multi", "demand"],
                   required=True)
    p.add_argument("-R", type=float, default=0.03)
    p.add_argument("-K", type=int, default=5)
    p.add_argument("--clear", type=float, default=0.015)
    p.add_argument("--spacing", type=float, default=2.2,
                   help="minimum centre spacing in units of R; ~1.2 packs the "
                        "obstacles into a near-continuous wall")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--prefix", default="multi")
    p.add_argument("--dmins", default=None,
                   help="comma list of d_min values to scan; smaller means the "
                        "filter engages later, with less room to stop")
    p.add_argument("--seed", type=int, default=0,
                   help="scene seed; each seed is a different G0/G1 pair and so "
                        "a different achievable separation")
    p.add_argument("--offsets", default=None,
                   help="comma list of sep-minus-R offsets; sizes the obstacle "
                        "per candidate instead of using absolute radii")
    p.add_argument("--ladder", default="1,2.5,5,10,25,50",
                   help="demand multipliers relative to the study config; "
                        "25x eta reaches SPARK's shipped eta_ssa=0.5")
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--sub", type=int, default=1500)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--radii", default="0.05,0.07,0.03")
    p.add_argument("--algos", default="ssa,rssa,pssa,cbf,rcbf,sss,rsss")
    a = p.parse_args(argv)
    if a.seed:
        global SEED, GEOM
        SEED = a.seed
        GEOM = f"fuzz/siren/constructed/separation_geom_s{a.seed}.json"
    if a.steps:
        global STEPS
        STEPS = a.steps
    if a.phase == "geom":
        return phase_geom(a.grid, a.sub)
    if a.phase == "demand":
        return phase_demand(a.top, a.R, a.K, a.clear, a.algos.split(","),
                            a.spacing,
                            [float(x) for x in a.ladder.split(",")],
                            ([float(x) for x in a.dmins.split(",")]
                             if a.dmins else None), a.prefix)
    if a.phase == "multi":
        return phase_multi(a.top, a.R, a.K, a.clear, a.algos.split(","),
                           a.spacing,
                           ([float(x) for x in a.offsets.split(",")]
                            if a.offsets else None), a.prefix)
    return phase_test(a.top, [float(x) for x in a.radii.split(",")],
                      a.algos.split(","), a.clear,
                      ([float(x) for x in a.offsets.split(",")]
                       if a.offsets else None))


if __name__ == "__main__":
    raise SystemExit(main())
