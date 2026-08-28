"""eta case study on a single constructed attack: sweep the demand, record the outcome.

Scene, obstacle placement and the goal triple (G0, G1', G1) are read from the
saved attack and held FIXED; only the filter's demand parameter eta changes. Two
filters are swept:

    ssa   -- no slack. An infeasible QP cannot be solved at all and SPARK's
             fallback returns the UNFILTERED reference control, so this filter
             can fail open.
    pssa  -- p-SSA. Phase I solves for the nearest feasible relaxation, so the
             QP is feasible by construction and it degrades via slack instead.

Two outcome columns, deliberately kept separate:

    collision      min clearance < 0  -- the safety failure. This is what
                   "the attack succeeds" means.
    reached_final  the robot finished the attacker's [G0, G1', G1] schedule.
                   A run can end with NO collision and NO goal (the filter
                   stalled the task); that is a task failure, not a defence,
                   and collapsing it into "attack failed" would hide it.

Data lives in data/: the attack is read from data/pssa_0_1.json and the sweep is
written to data/eta_sweep.csv + data/eta_sweep_meta.json by default.

Usage (see README.md -- DYLD_INSERT_LIBRARIES is required):
    python -m fuzz.siren.experiment.rq1.rq1_results.casestudy.sweep_eta
    python .../sweep_eta.py --attack <path.json> --out data/eta_sweep.csv
"""

import argparse
import csv
import json
import os

import numpy as np

# 0.004 is the demand the saved attack was actually found at, so it is the
# one point on this axis that reproduces the recorded collision.
ETAS = [0.0001, 0.001, 0.004, 0.005, 0.01, 0.05, 0.5, 1, 2, 3, 4, 5, 10, 15,
        20, 50, 100, 200, 1000, 10000]
ALGOS = ["ssa"]
HERE = os.path.dirname(os.path.abspath(__file__))
# All data files live in casestudy/data/.
DATA = os.path.join(HERE, "data")
DEFAULT_ATTACK = os.path.join(DATA, "pssa_0_1.json")


def run_one(V, algo, eta):
    """One rollout of the attack schedule under (algo, eta). Returns a row dict."""
    from fuzz.siren.world.run import World
    from fuzz.siren.world.types import real_filter
    from fuzz.siren.world.sim import probe
    from fuzz.siren.world import derived
    from fuzz.siren.pipeline.stage1_search import set_channel
    from fuzz.siren.experiment.rq1.big_obstacle import set_obstacles

    pos_w = [np.asarray(q, float) for q in V["obstacles_world"]]
    R = float(V["obstacle_radius"])
    steps = int(V["max_steps"])
    G0 = np.asarray(V["controls"][0]["G0"], float)
    G1 = np.asarray(V["G1"], float)
    G1p = np.asarray(V["G1_prime"], float)

    spec = real_filter(algo=algo, index=V["index"], d_min=V["d_min"],
                       eta=eta, lam=V["lam"], k=V["k"])
    w = World.build(seed=V["seed"], spec=spec, test_case=V["case"],
                    max_steps=steps)
    h = w.harness
    orig = h.reset

    def patched(*a, **kk):
        af_, _ = orig(*a, **kk)
        set_obstacles(w, pos_w, R)
        return af_, h.env.task.get_info(af_)

    h.reset = patched
    af, ti = h.reset()
    # Harness.reset() runs 10 warm-up algo.act calls. Zero the counter AFTER it
    # so warm-up infeasibility is not billed to the episode.
    warm = probe.giveups(h)
    probe.reset_giveups(h)
    set_channel(w, "arm")
    h.env.task.set_goal_schedule([G0, G1p, G1])
    ctrl, ai = h.algo.act(af, ti)

    clears, legs, engaged = [], [], 0
    for _ in range(steps):
        af, ti = h.env.step(ctrl, ai)
        try:
            ctrl, ai = h.algo.act(af, ti)
        except Exception:
            # SPARK answers a failed whole-body IK with a zero command.
            ctrl = np.zeros_like(np.asarray(ctrl, float))
        raw = probe.read_raw(h)
        if raw:
            act = derived.active_set(raw["phi"], raw["phi_mask"], "constant")
            engaged += int(bool(act.any()))
        clears.append(float(h.clearance(ti)))
        legs.append(int(getattr(h.env.task, "wp_idx", 0)))
        if clears[-1] < 0.0 or h.env.task.reached_final:
            break
    h.reset = orig

    cmin = float(min(clears))
    hit = next((i for i, c in enumerate(clears) if c < 0.0), None)
    return {
        "algo": algo,
        "eta": eta,
        "steps": len(clears),
        "min_clearance": round(cmin, 9),
        "collision": int(cmin < 0.0),
        "attack_succeeds": int(cmin < 0.0),
        "reached_final": int(bool(h.env.task.reached_final)),
        "contact_leg": ("" if hit is None else legs[hit]),
        "engaged_steps": engaged,
        "warmup_giveups": int(warm),
        # Only BasicSafeSetAlgorithm gets the give-up counter installed, so for
        # the slack filters this is not a measurement -- recorded as blank.
        "episode_giveups": (int(probe.giveups(h)) if w._giveup_counter else ""),
        "giveups_instrumented": int(bool(w._giveup_counter)),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default=DEFAULT_ATTACK)
    p.add_argument("--out", default=os.path.join(DATA, "eta_sweep.csv"))
    p.add_argument("--algos", default=",".join(ALGOS),
                   help="comma list; 'ssa,pssa' to include the slack filter")
    a = p.parse_args(argv)

    V = json.load(open(a.attack))
    algos = [x.strip() for x in a.algos.split(",") if x.strip()]
    cols = ["algo", "eta", "steps", "min_clearance", "collision",
            "attack_succeeds", "reached_final", "contact_leg", "engaged_steps",
            "warmup_giveups", "episode_giveups", "giveups_instrumented"]

    rows = []
    print(f"attack {os.path.basename(a.attack)}  seed {V['seed']}  "
          f"case {V['case']}\n  d_min {V['d_min']}  phi_k {V['k']}  "
          f"lam {V['lam']}  R {V['obstacle_radius']}  "
          f"n_obstacles {V['n_obstacles']}\n", flush=True)
    for algo in algos:
        for eta in ETAS:
            r = run_one(V, algo, eta)
            rows.append(r)
            print(f"  {algo:<5} eta={eta:<7g} steps {r['steps']:>4}  "
                  f"min clearance {r['min_clearance']:+.6f}  "
                  f"{'COLLISION' if r['collision'] else 'no contact':<10}  "
                  f"reached {bool(r['reached_final'])!s:<5}  "
                  f"episode give-ups {r['episode_giveups']}", flush=True)

    with open(a.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)
    # Provenance next to the data, so the csv is reproducible on its own.
    meta = {
        "attack_file": os.path.abspath(a.attack),
        "etas": ETAS, "algos": algos,
        "fixed": {k: V[k] for k in ("seed", "case", "index", "d_min", "k",
                                    "lam", "obstacle_radius", "n_obstacles",
                                    "max_steps", "G1", "G1_prime")},
        "obstacles_world": V["obstacles_world"],
        "G0": V["controls"][0]["G0"],
    }
    with open(os.path.join(os.path.dirname(a.out), "eta_sweep_meta.json"),
              "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n{len(rows)} rows -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
