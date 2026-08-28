"""RQ3 -- re-simulate every stored SIREN attack and decide whether it still holds.

The attacks in this repo were found over months by several different searches,
which wrote at least two record schemas, against code that has since changed
(the gap-metric fix, the fail-open findings, the D2/index rework). A record is a
CLAIM. This re-runs each one from scratch and keeps only the claims that survive.

The standard is the one pipeline/stage2_verify.py already established, and it is
stricter than "the label said COLLISION":

    baseline [G0, G1]         must REACH *and* hold clearance > 0 the whole way.
                              reached_final alone is not enough -- a run can
                              penetrate an obstacle and still arrive.
    attack   [G0, G1', G1]    must COLLIDE (or DEADLOCK) on EVERY repeat.
    contact leg               2 -> INSERTION, 1 -> MODIFICATION, 0 -> reject,
                              and it must agree across repeats.

Repeats matter here more than usual: these contacts are tens of microns deep and
a 0.155 mm perturbation was measured to flip an outcome, so a knife-edge that
fires once is not an attack.

Two record schemas are normalised on the way in:

    A  "constructed"  G0 nested in controls[0], has hit_leg, obstacle_radius
    B  "verified"     G0 top-level, has attack/baseline verdict blocks

Records carrying `obstacles_world` get those obstacles PINNED, because the scene
they were found in is part of the claim. Records without them (the teleop hunts)
run on the scene's own obstacles for that seed.

    python -m fuzz.siren.rq3.revalidate --shard 0 --of 8
"""

import argparse
import gc
import glob
import json
import os

import numpy as np

OUT = "fuzz/siren/rq3/logs/verdicts"


def norm_record(v, path):
    """Both schemas -> one shape. None if the record cannot be replayed."""
    G0 = v.get("G0")
    if G0 is None and v.get("controls"):
        c = v["controls"][0]
        G0 = c.get("G0") if isinstance(c, dict) else None
    if G0 is None or v.get("G1") is None or v.get("G1_prime") is None:
        return None
    kind = v.get("hit_leg") or v.get("kind")
    if kind is None:
        # scenario/verified stores the leg as a bool instead of a name
        if v.get("contact_on_attack_leg") is not None:
            kind = "INSERTION" if v["contact_on_attack_leg"] else "MODIFICATION"
    return {
        "src": path,
        "case": v["case"], "seed": int(v["seed"]), "algo": v["algo"],
        "index": v.get("index"), "d_min": v.get("d_min"),
        "eta": v.get("eta"), "lam": v.get("lam"), "k": v.get("k"),
        "channel": v.get("channel") or "arm",
        "max_steps": int(v.get("max_steps") or 900),
        "G0": list(map(float, G0)), "G1": list(map(float, v["G1"])),
        "G1p": list(map(float, v["G1_prime"])),
        "obstacles_world": v.get("obstacles_world"),
        "obstacle_radius": v.get("obstacle_radius"),
        "claimed_kind": kind,
        "claimed_leg2_min": v.get("leg2_min"),
        "claimed_baseline_min": v.get("baseline_min"),
    }


def legwise(rec):
    """min clearance per leg, and the first leg to make contact."""
    per, hit = {}, None
    for s in rec.steps:
        wp = int(s.wp_idx)
        per[wp] = min(per.get(wp, np.inf), float(s.clearance))
        if s.clearance < 0.0 and hit is None:
            hit = wp
    return per, hit


def free():
    gc.collect()


def run_one(r, repeats):
    """Re-simulate one attack. Returns a verdict dict."""
    from ..pipeline.stage1_search import set_channel
    from ..world.run import World
    from ..world.types import real_filter

    spec = real_filter(algo=r["algo"], index=r["index"], d_min=r["d_min"],
                       eta=r["eta"], lam=r["lam"], k=r["k"])
    out = {"verdict": "ERROR", "reason": None, **{k: r[k] for k in
           ("src", "case", "seed", "algo", "channel", "claimed_kind")}}

    pin = None
    if r["obstacles_world"]:
        pin = [np.asarray(q, float) for q in r["obstacles_world"]]

    def build():
        w = World.build(seed=r["seed"], spec=spec, test_case=r["case"],
                        max_steps=r["max_steps"])
        set_channel(w, r["channel"])
        return w

    def rollout(w, sched):
        if pin is not None:
            from ..constructed.big_obstacle import run_pinned
            return run_pinned(w, [np.asarray(x, float) for x in sched], pin,
                              float(r["obstacle_radius"] or 0.05),
                              r["max_steps"])
        return w.run([np.asarray(x, float) for x in sched],
                     max_steps=r["max_steps"])

    G0, G1, G1p = (np.asarray(r[k], float) for k in ("G0", "G1", "G1p"))

    # ---- baseline: must REACH and never penetrate ----------------------- #
    w = build()
    b = rollout(w, [G0, G1])
    w = None; free()
    if b is None:
        out.update(verdict="ERROR", reason="baseline solver failure"); return out
    bper, bhit = legwise(b)
    bmin = min(bper.values()) if bper else float("inf")
    out["baseline_label"] = b.label
    out["baseline_min"] = float(bmin)
    if b.label != "REACHED" or bmin <= 0:
        out.update(verdict="INVALID",
                   reason=f"baseline {b.label} min={bmin:+.6f}: the legitimate "
                          "task is not safe in this scene, so there is no attack")
        return out

    # ---- attack: must collide on the SAME leg every repeat -------------- #
    labels, legs, mins = [], [], []
    for i in range(repeats):
        w = build()
        a = rollout(w, [G0, G1p, G1])
        w = None; free()
        if a is None:
            out.update(verdict="ERROR", reason=f"attack solver failure rep {i}")
            return out
        aper, ahit = legwise(a)
        labels.append(a.label); legs.append(ahit)
        mins.append(float(min(aper.values()) if aper else np.inf))
    out["attack_labels"] = labels
    out["attack_legs"] = legs
    out["attack_mins"] = mins

    unsafe = [l in ("COLLISION", "DEADLOCK") for l in labels]
    if not all(unsafe):
        out.update(verdict="INVALID",
                   reason=f"attack did not reproduce: labels={labels}")
        return out
    if len(set(legs)) != 1:
        out.update(verdict="INVALID",
                   reason=f"contact leg inconsistent across repeats: {legs}")
        return out
    leg = legs[0]
    if labels[0] == "DEADLOCK":
        kind = "DEADLOCK"
    elif leg == 2:
        kind = "INSERTION"
    elif leg == 1:
        kind = "MODIFICATION"
    else:
        out.update(verdict="INVALID",
                   reason=f"contact on leg {leg}: the run crashed before the "
                          "inserted goal, so the goal is incidental")
        return out
    out["observed_kind"] = kind
    out["verdict"] = "VALID"
    out["kind_matches_claim"] = (r["claimed_kind"] in (None, kind))
    if not out["kind_matches_claim"]:
        out["reason"] = f"still an attack, but {kind} not {r['claimed_kind']}"
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--identities", default="fuzz/siren/rq3/logs/identities.json")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--limit", type=int)
    a = p.parse_args(argv)

    ids = json.load(open(a.identities))
    keys = sorted(ids)
    keys = [k for i, k in enumerate(keys) if i % a.of == a.shard]
    if a.limit:
        keys = keys[:a.limit]
    os.makedirs(OUT, exist_ok=True)
    print(f"shard {a.shard}/{a.of}: {len(keys)} identities", flush=True)

    for n, h in enumerate(keys):
        dest = f"{OUT}/{h}.json"
        if os.path.exists(dest):
            continue
        src = ids[h]["src"][0]
        try:
            v = json.load(open(src))
            r = norm_record(v, src)
            if r is None:
                res = {"verdict": "SKIP", "reason": "record lacks G0/G1/G1'",
                       "src": src}
            else:
                res = run_one(r, a.repeats)
        except Exception as e:
            res = {"verdict": "ERROR", "reason": f"{type(e).__name__}: {e}",
                   "src": src}
        res["id"] = h
        res["all_sources"] = ids[h]["src"]
        json.dump(res, open(dest, "w"), indent=1, default=float)
        print(f"  [{n+1}/{len(keys)}] {h} {res['verdict']:<8} "
              f"{res.get('observed_kind') or res.get('reason') or ''}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
