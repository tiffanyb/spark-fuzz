"""
Stage 3b -- run SIREN against a verified target and log every location searched.

The point of a target: an empty search is ambiguous between "no attack exists"
and "the search cannot find one". A target removes that -- an attack is known to
exist at known coordinates, and `G1_prime_truth` is never shown to the search.

Differences from the older scenario/run_fuzzer.py, all of them things that bit
us:

  C1 tests SAFETY, not arrival.   The baseline [G0, G1] must reach *and* keep
      clearance > 0 at every step. reached_final alone accepts a rollout that
      penetrates an obstacle and still arrives -- that mislabelled 18 controls.

  MODIFICATION counts as a hit.   The old code scored only contact on leg 2, so
      6 of 61 ground-truth attacks could not be scored at all. Here the target's
      own `kind` says which leg makes a hit, and both are recorded.

  PROVENANCE per location.        Each evaluated location gets a UUID, a
      wall-clock timestamp, the ask/tell round, and `parent_ids`.

On `parent_ids`: it is a LIST, because "the parent" is not well defined for two
of the three strategies and inventing one would be a lie about the genealogy.

    random  nothing -- draws are independent           -> []
    cem     the elite set that produced the current
            mean/std; a proposal descends from all of
            them, not one point                        -> ids of the elites
    bo      the whole observation history through the
            GP posterior; no single ancestor           -> ids of that round's
                                                          observations

    python -m fuzz.siren.pipeline.stage3_fuzz --target /abs/t.json --budget 60
"""

import argparse
import datetime
import json
import os
import time
import uuid

import numpy as np


def _now():
    """Absolute wall-clock stamp for one searched location.

    The existing "t" is seconds since the run began, which is what you want for
    plotting a search curve. It is NOT enough to line a run up against anything
    outside itself -- another picker's run, a machine-load trace, a second
    experiment. So both are recorded: "t" relative, "ts" absolute ISO-8601.
    """
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


def contact_leg(rec):
    for s in rec.steps:
        if s.clearance < 0.0:
            return int(s.wp_idx)
    return None


def run_schedule(world, schedule, steps, channel="arm", pin=None, radius=0.05):
    """Run a schedule in either a stock scene or a pinned custom scene."""
    sched = [np.asarray(x, float) for x in schedule]
    if pin is not None:
        if channel != "arm":
            raise RuntimeError("pinned obstacle targets are only supported for arm-channel runs")
        from ..constructed.big_obstacle import run_pinned
        return run_pinned(world, sched, pin, radius, steps)
    from .stage1_search import run_ch
    return run_ch(world, sched, channel, max_steps=steps)


def obstacle_frames(points):
    frames = []
    for q in points:
        q = np.asarray(q, float)
        if q.shape == (4, 4):
            frames.append(q)
        else:
            f = np.eye(4)
            f[:3, 3] = q.reshape(-1)[:3]
            frames.append(f)
    return np.asarray(frames, float)


def disabled_pickers():
    disabled = set()
    env = os.environ.get("SIREN_DISABLED_PICKERS", "")
    disabled.update(x.strip() for x in env.split(",") if x.strip())
    rq3_flag = os.path.join(os.path.dirname(__file__), "..", "rq3",
                            ".disable_bo")
    if os.path.exists(rq3_flag):
        disabled.update(("bo", "bo_seeded"))
    return disabled


def evaluate_candidate(world, G0, cand, G1, steps, hit_leg,
                       channel="arm", pin=None, radius=0.05):
    """Run [G0, cand, G1] and decide whether it is an attack of the wanted kind.

    Score mirrors pipeline_spec: engaged candidates outrank unengaged ones, and
    within each group the braking margin (or clearance) decides. `mu` is
    deliberately not used -- measured over 9936 engaged candidates it is
    ANTI-correlated with contact (corr -0.14), so any monotone function of it
    steers the search the wrong way.
    """
    rec = run_schedule(world, [G0, cand, G1], steps, channel,
                       pin=pin, radius=radius)
    leg = contact_leg(rec)
    is_attack = ((rec.label == "COLLISION" and leg == hit_leg)
                 or rec.label == "DEADLOCK")

    tail = [s for s in rec.steps if s.wp_idx >= hit_leg]
    engaged = [s for s in tail if s.engaged]
    if engaged:
        bm = [float(s.brake_margin) for s in tail
              if np.isfinite(s.brake_margin)]
        score = 1000.0 - (min(bm) if bm else 0.0)
    else:
        cl = [s.clearance for s in tail if np.isfinite(s.clearance)]
        score = -float(min(cl)) if cl else 0.0
    return is_attack, rec, leg, score, bool(engaged)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--pickers",
                   default="random,random_seeded,cem,cem_seeded,bo,bo_seeded")
    p.add_argument("--budget", type=int, default=60)
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--search-seeds", default="0")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    import dataclasses

    from ..scenario.targets import load_target
    from ..search.pick import make_picker, is_admissible
    from .stage1_search import run_ch

    w, tgt = load_target(a.target)
    channel = tgt.get("channel", "arm")
    _pin = tgt.get("obstacles_world") if tgt.get("pinned_scene") else None
    pin = [np.asarray(q, float) for q in _pin] if _pin else None
    pin_radius = float(tgt.get("obstacle_radius") or 0.05)
    if pin is not None:
        if channel != "arm":
            raise RuntimeError("custom pinned obstacles are only supported for arm targets")
    sc = w.scene()
    if pin is not None:
        sc = dataclasses.replace(sc, obstacles_world=obstacle_frames(pin))

    # A BASE target's coordinates are (x, y, yaw) base poses, not end-effector
    # positions. Two things follow, and both used to be wrong here:
    #   * every rollout has to go down the base channel, or the waypoints reach
    #     the arm IK as an unreachable pose and casadi dies;
    #   * the search space is base_goal_range + the yaw range, NOT the 0.3 m arm
    #     workspace box. The bounds stored in the target are the ARM box even
    #     for base targets (a stage-1 bug, fixed there too), so they are taken
    #     from the live task rather than from the file.
    admissible = is_admissible
    if channel == "base":
        task = w.harness.env.task
        br = task.base_goal_range
        rr = getattr(task, "base_goal_rot_range", (-np.pi, np.pi))
        ko = float(getattr(task, "base_goal_keepout", 0.1))
        sc = dataclasses.replace(
            sc, bounds=((br[0][0], br[0][1]), (br[1][0], br[1][1]),
                        (rr[0], rr[1])), keepout=ko)
        obs_xy = (np.array([np.asarray(o)[:2, 3] for o in sc.obstacles_world])
                  if sc.n_obstacles else None)

        def admissible(cand, scene):
            """SPARK's own rule for a sampled base goal: inside base_goal_range,
            and at least base_goal_keepout from any obstacle in XY. The yaw
            component has no keepout -- rotating in place cannot approach an
            obstacle."""
            g = np.asarray(cand, float).reshape(3)
            for d in range(3):
                lo, hi = scene.bounds[d]
                if g[d] < lo or g[d] > hi:
                    return False, "out_of_bounds"
            if obs_xy is not None and len(obs_xy):
                gap = float(np.min(np.linalg.norm(obs_xy - g[:2], axis=1)))
                if gap < scene.keepout:
                    return False, f"too_close({gap:.3f}<{scene.keepout:.3f})"
            return True, "ok"
    steps = tgt["max_steps"]
    G0 = np.asarray(tgt["G0_commanded"], float)
    G1 = np.asarray(tgt["G1"], float)
    # A target may have NO answer key. The benchmark targets are built around a
    # known attack, but a target can also be built from a scenario the filter
    # merely SURVIVES -- there the question is whether an attack exists at all,
    # so there is nothing to measure distance to and nothing to pre-check.
    _t = tgt.get("G1_prime_truth")
    truth = None if _t is None else np.asarray(_t, float)
    kind = tgt.get("kind", "INSERTION")
    hit_leg = 2 if kind == "INSERTION" else 1

    print(f"target {tgt['name']}")
    print(f"  {tgt['case']}  filter={tgt['algo']}  seed={tgt['seed']}  "
          f"channel={channel}  kind={kind} -> a hit is contact on leg {hit_leg}")
    print(f"  search space {[tuple(round(float(x), 3) for x in b) for b in sc.bounds]}"
          f"  keepout {sc.keepout}")

    base = run_schedule(w, [G0, G1], steps, channel, pin=pin,
                        radius=pin_radius)
    base_clear = min((s.clearance for s in base.steps), default=np.inf)
    known = (run_schedule(w, [G0, truth, G1], steps, channel, pin=pin,
                          radius=pin_radius)
             if truth is not None else None)
    print(f"  baseline [G0,G1]         {base.label} "
          f"(min clearance {base_clear:+.6f})")
    if known is not None:
        print(f"  known attack [G0,G1*,G1] {known.label} "
              f"(leg {contact_leg(known)}, {known.min_clearance:+.6f})")
    # C1: safe arrival, not merely arrival
    if base.label != "REACHED" or base_clear <= 0.0:
        print("  baseline is not safe — aborting (the scene is broken, so a "
              "collision would say nothing about the inserted goal)")
        return 1
    if known is not None and known.label not in ("COLLISION", "DEADLOCK"):
        print("  planted attack does not reproduce — aborting")
        return 1
    print(f"  (the search is NOT told G1* = {np.round(truth,3)})\n"
          if truth is not None else
          "  NO ground truth — open search: does an attack exist at all?\n",
          flush=True)

    rows = []
    t_start = time.time()
    _started_at = _now()
    picker_names = [x.strip() for x in a.pickers.split(",") if x.strip()]
    disabled = disabled_pickers()
    dropped = [x for x in picker_names if x in disabled]
    if dropped:
        print(f"  disabled pickers: {','.join(dropped)}", flush=True)
        picker_names = [x for x in picker_names if x not in disabled]

    for pname in picker_names:
        for ss in [int(x) for x in a.search_seeds.split(",") if x.strip()]:
            picker = make_picker(pname, sc, seed=ss)
            n_eval, first_hit, hits, evals, n_inadm = 0, None, [], [], 0
            n_err = 0
            prev_round_ids, rnd, t0 = [], 0, time.time()
            # The docstring's contract: random draws are INDEPENDENT, so they
            # have no ancestor and must record []. Writing prev_round_ids for
            # them claims a descent that does not exist -- the exact "lie about
            # the genealogy" the parent_ids design set out to avoid. Only the
            # strategies that actually condition on prior observations (cem on
            # its elite set, bo on the GP's observation history) get parents.
            _has_genealogy = not pname.startswith("random")
            while n_eval < a.budget:
                want = min(a.batch, a.budget - n_eval)
                cands = picker.ask(want)
                if not cands:
                    break
                scored, this_round = [], []
                for c in cands:
                    if not admissible(c, sc)[0]:
                        n_inadm += 1
                        evals.append({"id": str(uuid.uuid4()),
                                      "stage": "inadmissible", "round": rnd,
                                      "t": time.time() - t_start,
                                      "ts": _now(),
                                      "parent_ids": (prev_round_ids if _has_genealogy else []),
                                      "cand": [float(x) for x in
                                               np.asarray(c, float).reshape(-1)]})
                        continue
                    # A candidate the whole-body IK cannot solve raises out of
                    # casadi ("Maximum_Iterations_Exceeded") and used to kill the
                    # whole run -- 6 of 59 targets produced no result at all, and
                    # a crash midway is indistinguishable from a search that
                    # found nothing. Charge it as a spent evaluation (it cost a
                    # rollout attempt) and carry on; the picker simply never
                    # hears about it, which is correct since there is no score.
                    try:
                        atk, rec, leg, score, eng = evaluate_candidate(
                            w, G0, c, G1, steps, hit_leg, channel,
                            pin=pin, radius=pin_radius)
                    except Exception as e:
                        n_eval += 1
                        n_err += 1
                        evals.append({"id": str(uuid.uuid4()), "stage": "error",
                                      "round": rnd, "eval": int(n_eval),
                                      "t": time.time() - t_start,
                                      "ts": _now(),
                                      "parent_ids": (prev_round_ids if _has_genealogy else []),
                                      "cand": [float(x) for x in
                                               np.asarray(c, float).reshape(-1)],
                                      "error": f"{type(e).__name__}: {e}"[:200]})
                        continue
                    n_eval += 1
                    scored.append((c, score))
                    uid = str(uuid.uuid4())
                    this_round.append(uid)
                    evals.append({
                        "id": uid, "stage": "evaluated", "round": rnd,
                        "eval": int(n_eval), "t": time.time() - t_start,
                        "ts": _now(),
                        "parent_ids": (prev_round_ids if _has_genealogy else []),
                        "cand": [float(x) for x in
                                 np.asarray(c, float).reshape(-1)],
                        "score": float(score), "is_attack": bool(atk),
                        "label": rec.label,
                        "contact_leg": None if leg is None else int(leg),
                        "min_clearance": float(rec.min_clearance),
                        "n_steps": int(rec.n_steps),
                        "n_gave_up": int(getattr(rec, "n_gave_up", 0)),
                        "engaged": bool(eng),
                        "dist_to_truth": (None if truth is None else
                                          float(np.linalg.norm(
                                              np.asarray(c, float).reshape(-1)
                                              - truth))),
                        "from_initial_design": bool(rnd == 0)})
                    if atk:
                        hits.append(evals[-1])
                        if first_hit is None:
                            first_hit = n_eval
                            print(f"    {pname}/s{ss}: FIRST HIT at eval "
                                  f"{n_eval}", flush=True)
                    if n_eval >= a.budget:
                        break
                if scored:
                    picker.tell(scored)
                # the next round descends from what this round observed
                prev_round_ids = this_round
                rnd += 1
            best = min((h["dist_to_truth"] for h in hits
                        if h["dist_to_truth"] is not None), default=None)
            rows.append({"picker": pname, "search_seed": ss,
                         "n_eval": n_eval, "n_attacks": len(hits),
                         "first_hit": first_hit, "best_dist_to_truth": best,
                         "n_inadmissible": n_inadm,
                         "n_errors": n_err, "n_rounds": rnd,
                         "elapsed_s": round(time.time() - t0, 1),
                         "evaluations": evals})
            print(f"  {pname:<14} s{ss}  attacks {len(hits):>3}/{n_eval}  "
                  f"first {str(first_hit):>5}  rounds {rnd}  "
                  f"err {n_err}  ({rows[-1]['elapsed_s']}s)", flush=True)

    print(f"\n{'='*76}")
    print(f"{'picker':<16}{'attacks':>9}{'rate':>7}{'1st':>6}"
          f"{'closest':>10}{'rounds':>8}")
    for r in rows:
        rate = f"{100*r['n_attacks']/max(1,r['n_eval']):.0f}%"
        cl = f"{r['best_dist_to_truth']:.3f}" if r["best_dist_to_truth"] is not None else "-"
        print(f"{r['picker']:<16}{r['n_attacks']:>9}{rate:>7}"
              f"{str(r['first_hit']):>6}{cl:>10}{r['n_rounds']:>8}")
    total_s = time.time() - t_start
    print(f"\ntotal fuzzing time {total_s:.1f}s "
          f"({total_s/60:.1f} min) across {len(rows)} picker(s)")
    json.dump({"target": tgt["name"], "case": tgt["case"], "algo": tgt["algo"],
               "seed": tgt["seed"], "kind": kind, "hit_leg": hit_leg,
               "budget": a.budget, "batch": a.batch,
               "pinned_obstacles": bool(pin),
               "obstacle_radius": (pin_radius if pin is not None else None),
               # per-picker cost is in results[].elapsed_s; this is the whole run
               "total_elapsed_s": round(total_s, 2),
               "started_at": _started_at, "finished_at": _now(),
               "G1_prime_truth": (None if truth is None else truth.tolist()),
               "baseline_label": base.label,
               "baseline_min_clearance": float(base_clear),
               "results": rows}, open(a.out, "w"), indent=2, default=float)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
