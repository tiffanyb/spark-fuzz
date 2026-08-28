"""How near does the return leg actually come to the placement?

phase_hunt prints nothing for a rollout that produces no leg-2 contact, so a
seed that yields zero attacks leaves no record of whether it missed by a
millimetre or by ten centimetres. Those two cases need opposite fixes -- a
weaker demand / deeper overlap for the first, different placements for the
second -- so the distinction has to be measured before more budget is spent.

Reports, per placement, the minimum clearance reached on each leg.

    python -m fuzz.siren.constructed.reach_probe --seed 3 --algo rcbf --spots 6
"""

import argparse
import json

import numpy as np

from .separated_search import CASE, PARK
from .run_rq1 import R_STOCK, SPOTS, free_worlds, legwise, run


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--algo", default="rcbf")
    p.add_argument("--spots", type=int, default=6)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--gap-target", type=float, default=-0.023)
    p.add_argument("--mult", type=float, default=0.02,
                   help="demand multiplier; the weakest rung of the ladder")
    p.add_argument("--d-min", type=float, default=0.015)
    p.add_argument("--k", type=float, default=0.1)
    a = p.parse_args(argv)

    from ..pipeline import trialconf
    from ..world.run import World
    from ..world.types import real_filter

    s = json.load(open(f"{SPOTS}_{a.seed}.json"))
    if int(s["seed"]) != a.seed:
        raise SystemExit(f"corpus is seed {s['seed']}, not {a.seed}")
    s["spots"].sort(key=lambda x: abs(x["gap"] - a.gap_target))
    G0, G1 = np.asarray(s["G0"], float), np.asarray(s["G1"], float)

    cfg = trialconf.load("fuzz/siren/pipeline/configs/config2.yaml")
    base = trialconf.spec_for(cfg, a.algo, CASE)
    const = base.demand_shape == "constant"
    spec = real_filter(algo=a.algo, index=base.index, d_min=a.d_min, k=a.k,
                       eta=(base.eta or 0.02) * a.mult if const else base.eta,
                       lam=base.lam if const else (base.lam or 10.0) * a.mult)

    w0 = World.build(seed=a.seed, spec=spec, test_case=CASE, max_steps=a.steps)
    n_obs = len(w0.scene().obstacles_world)
    w0 = None
    free_worlds()

    dem = f"eta*{a.mult}" if const else f"lam*{a.mult}"
    print(f"seed {a.seed}  {a.algo}  {dem}  d_min={a.d_min}  k={a.k}\n"
          f"{'#':>3} {'gap':>9} {'leg1 min':>10} {'leg2 min':>10} "
          f"{'hit':>4}  verdict", flush=True)

    rows = []
    for i, sp in enumerate(s["spots"][:a.spots]):
        G1p = np.asarray(sp["G1_prime"], float)
        pos_w = [np.asarray(sp["spot"], float)] + [PARK] * (n_obs - 1)
        w = World.build(seed=a.seed, spec=spec, test_case=CASE,
                        max_steps=a.steps)
        rec = run(w, [G0, G1p, G1], pos_w, R_STOCK, a.steps)
        w = None
        free_worlds()
        if rec is None:
            continue
        per, hit, _ = legwise(rec)
        l1, l2 = per.get(1, float("nan")), per.get(2, float("nan"))
        if hit == 2:
            v = "CONTACT on return leg"
        elif np.isnan(l2):
            v = "return leg never ran"
        elif l2 < 0.01:
            v = "near miss -- weaken demand"
        elif l2 < 0.03:
            v = "approaches, held off"
        else:
            v = "far -- placement off the filtered path"
        rows.append(l2)
        print(f"{i:>3} {sp['gap']:>9.4f} {l1:>10.5f} {l2:>10.5f} "
              f"{str(hit):>4}  {v}", flush=True)

    if rows:
        r = np.array([x for x in rows if not np.isnan(x)])
        if len(r):
            print(f"\nleg-2 closest approach over {len(r)} placements: "
                  f"min {r.min():+.5f}  median {np.median(r):+.5f}  "
                  f"max {r.max():+.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
