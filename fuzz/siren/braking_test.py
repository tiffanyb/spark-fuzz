"""
A cheap stand-in for the escape search: is the robot already past braking distance?

WHY. Kind 1 vs Kind 2 turns on WHEN the state became infeasible. mu cannot answer
that -- it called 100% of the collisions feasible -- and the brute-force escape
search that can costs ~600 simulator rollouts per probed step, about a week of
compute for the full sweep. This is an O(1)-per-step approximation of the same
question, validated against the 77 escape-search labels.

THE PHYSICS mu misses. Under acceleration control the robot cannot stop
instantly. If a guarded pair is closing at speed v and the largest deceleration
of that closing available from the actuators is a, arresting needs

        stopping distance  ~  v^2 / (2a)

so the state is already committed when   clearance < v^2 / (2a).

mu is a FIRST-ORDER condition on the rate ("is some control making phi_dot
acceptable right now?"); braking distance is the INTEGRATED second-order
quantity. That gap is exactly the relative-degree-2 problem, and it is why D1
(velocity control, no braking distance) never collides while D2 does.

WHAT COULD GO WRONG, stated up front. This is a point-mass approximation along
the single closest-approach direction, applied to a 17-DoF arm that could escape
SIDEWAYS instead of braking. That failure mode makes it over-report
infeasibility, i.e. it will tend to place the point of no return too early and so
over-report Kind 1. Whether that matters is an empirical question, which is what
the validation below answers. If agreement is poor the honest fallback is to keep
the expensive probe on a sample and report ratios from the sample rather than
per-candidate labels.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python -m fuzz.siren.braking_test
"""

import argparse
import glob
import json

import numpy as np


def trace(world, scene, cand, max_steps):
    """One plain run, recording what a braking test needs. No LP, no rollouts."""
    from .world.sim import probe
    from spark_utils import compute_masked_distance_matrix

    h = world.harness
    si = h.algo.safe_controller.safe_algo.safety_index
    env_mask = np.asarray(si.env_collision_mask, bool)
    interval = h.env.agent.dt * h.env.agent.control_decimation

    probe.reset_giveups(h)
    af, ti = h.reset()
    h.env.task.set_goal_schedule([np.asarray(cand), np.asarray(scene.G1)])
    u, ai = h.algo.act(af, ti)

    rows = []
    for t in range(max_steps):
        af, ti = h.env.step(u, ai)
        u, ai = h.algo.act(af, ti)
        dm, _ = compute_masked_distance_matrix(
            frame_list_1=h.env.task.robot_frames_world,
            geom_list_1=h.robot_cfg.CollisionVol.values(),
            frame_list_2=ti["obstacle"]["frames_world"],
            geom_list_2=ti["obstacle"]["geom"])
        clear = float(np.where(env_mask, np.asarray(dm, float), np.inf).min())
        raw = probe.read_raw(h)
        C_phi = C_d = np.nan
        engaged = False
        if raw:
            from . import world as _w  # noqa
            from .world import derived
            d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"], raw["phi_mask"],
                                 raw["u_lim"], demand_shape=h.spec.demand_shape,
                                 eta=h.spec.eta, lam=h.spec.lam, exact=False)
            C_phi, C_d = d.get("C_phi", np.nan), d.get("C_d", np.nan)
            engaged = bool(d.get("engaged", False))
        rows.append({"t": t, "wp": int(getattr(h.env.task, "wp_idx", 0)),
                     "clear": clear, "C_phi": float(C_phi), "C_d": float(C_d),
                     "engaged": engaged})
        if h.env.task.reached_final or clear < 0.0:
            break

    # closing speed by finite difference -- deliberately model-free, so the test
    # does not inherit the same linearisation that made mu wrong.
    for i, r in enumerate(rows):
        r["vclose"] = ((rows[i - 1]["clear"] - r["clear"]) / interval) if i else 0.0
    return rows, interval


def predict(rows, a_source, scale):
    """First step at which clearance < v^2/(2a). None if it never happens."""
    for r in rows:
        v = r["vclose"]
        if v <= 0:                      # receding: braking distance irrelevant
            continue
        a = r.get(a_source, np.nan)
        if not np.isfinite(a) or a <= 0:
            continue
        if r["clear"] < (v * v) / (2.0 * scale * a):
            return r["t"]
    return None


def classify(pnr, t0):
    if pnr is None:
        return "NO_INFEASIBILITY"
    if t0 is None:
        return "DOOMED_BUT_NEVER_ENGAGED"
    return "KIND_1 (discrete)" if pnr <= t0 else "KIND_2 (discrete)"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default="/tmp/dfeas_do*.json")
    p.add_argument("--out", default="/tmp/braking_validation.json")
    a = p.parse_args(argv)

    from .world.run import World
    from .world.types import real_filter

    truth = []
    for f in sorted(glob.glob(a.labels)):
        truth += [r for r in json.load(open(f)) if r.get("faithful")]
    print(f"validating against {len(truth)} escape-search labels\n", flush=True)

    traces = []
    for i, r in enumerate(truth):
        index = "velocity" if "_D2_" in r["case"] else "distance"
        spec = real_filter(algo="ssa", index=index, d_min=0.02, eta=0.02, k=0.1)
        ms = 300 if "_D2_" in r["case"] else 500
        w = World.build(seed=r["seed"], spec=spec, test_case=r["case"], max_steps=ms)
        rows, interval = trace(w, w.scene(), r["candidate"], ms)
        leg2 = [x for x in rows if x["wp"] >= 1]
        traces.append({"truth": r, "rows": leg2 or rows,
                       "t0": r["t0_leg2"], "pnr_true": r["point_of_no_return"]})
        if (i + 1) % 10 == 0:
            print(f"   traced {i+1}/{len(truth)}", flush=True)

    print("\n" + "=" * 78)
    print("AGREEMENT WITH THE ESCAPE SEARCH")
    print("=" * 78)
    print(f"{'authority':<8}{'scale':>7}{'kind acc':>10}{'K1 recall':>11}"
          f"{'K2 recall':>11}{'PNR err (steps)':>18}")
    best = None
    for src in ("C_d", "C_phi"):
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
            ok = k1_ok = k1_n = k2_ok = k2_n = 0
            errs = []
            for tr in traces:
                pnr = predict(tr["rows"], src, scale)
                got = classify(pnr, tr["t0"])
                want = tr["truth"]["verdict"]
                if got == want:
                    ok += 1
                if want.startswith("KIND_1"):
                    k1_n += 1; k1_ok += got.startswith("KIND_1")
                if want.startswith("KIND_2"):
                    k2_n += 1; k2_ok += got.startswith("KIND_2")
                if pnr is not None and tr["pnr_true"] is not None:
                    errs.append(pnr - tr["pnr_true"])
            acc = ok / len(traces)
            e = f"{np.median(errs):+.0f} (n={len(errs)})" if errs else "-"
            print(f"{src:<8}{scale:>7}{100*acc:>9.0f}%"
                  f"{100*k1_ok/max(1,k1_n):>10.0f}%{100*k2_ok/max(1,k2_n):>10.0f}%{e:>18}")
            if best is None or acc > best[0]:
                best = (acc, src, scale)
    print(f"\nbest: {best[1]} x{best[2]} at {100*best[0]:.0f}% kind accuracy")
    json.dump({"n": len(traces), "best_source": best[1], "best_scale": best[2],
               "accuracy": best[0]}, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
