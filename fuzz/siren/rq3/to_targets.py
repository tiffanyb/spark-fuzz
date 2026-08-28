"""Convert re-validated rq3/dataset attacks into stage3_fuzz target files.

stage3_fuzz wants a "target": an attack known to exist at known coordinates, so
that an empty search means "the search failed" rather than "nothing was there".
The dataset records carry all the substance but under different names, and one
class of them carries a scene stage3_fuzz cannot reconstruct.

Renames
    G0 / controls[0].G0   ->  G0_commanded
    G1_prime              ->  G1_prime_truth      (the answer key, never shown)
    hit_leg / kind        ->  kind
    <filename>            ->  name

REFUSED, deliberately
    `load_target` builds the STOCK world for (case, seed) and never pins
    obstacles_world. For 69 of the 265 dataset entries -- all from the
    `constructed` searches -- the recorded scene is not the stock scene: those
    searches park unused obstacles ~9 m away and place one sphere on purpose.
    Pointing stage3_fuzz at one of those would search a 5-obstacle scene for an
    attack that only exists in a 1-obstacle scene. It would not crash; it would
    report "no attack found", which is a FALSE NEGATIVE and precisely the
    ambiguity a target is supposed to remove.

    So those are skipped with a reason rather than emitted. Fixing them needs
    load_target to honour obstacles_world, not an adapter.

DEADLOCK entries are also skipped: stage3_fuzz scores a hit as contact on leg 2
(INSERTION) or leg 1 (MODIFICATION), and a deadlock has no contact at all.

    python -m fuzz.siren.rq3.to_targets --out fuzz/siren/rq3/targets
"""

import argparse
import collections
import json
import os

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="fuzz/siren/rq3/dataset/INDEX.json")
    p.add_argument("--out", default="fuzz/siren/rq3/targets")
    p.add_argument("--allow-custom-scene", action="store_true",
                   help="emit targets whose recorded scene differs from the "
                        "stock scene. Only valid now that load_target pins "
                        "obstacles_world; before that fix these produced false "
                        "negatives")
    p.add_argument("--kinds", default="INSERTION,MODIFICATION",
                   help="which kinds to emit (DEADLOCK is never scorable)")
    a = p.parse_args(argv)

    from .revalidate import norm_record
    from ..world.run import World
    from ..world.types import real_filter

    idx = json.load(open(a.index))
    want = {k.strip() for k in a.kinds.split(",")}
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        if f.endswith(".json"):
            os.unlink(os.path.join(a.out, f))

    stock = {}
    stats = collections.Counter()
    skipped = []
    written = []

    for name, e in sorted(idx.items()):
        r = norm_record(json.load(open(e["source"])), e["source"])
        kind = e["kind"]
        if kind not in want:
            stats[f"skip:kind={kind}"] += 1
            skipped.append((name, f"kind {kind} is not scorable by stage3_fuzz"))
            continue

        # scene check: does the STOCK world for (case, seed) match the record?
        if r.get("obstacles_world"):
            key = (r["case"], r["seed"])
            if key not in stock:
                spec = real_filter(algo=r["algo"], index=r["index"],
                                   d_min=r["d_min"], eta=r["eta"],
                                   lam=r["lam"], k=r["k"])
                w = World.build(seed=r["seed"], spec=spec,
                                test_case=r["case"], max_steps=50)
                stock[key] = np.array([np.asarray(o, float)[:3, 3]
                                       for o in w.scene().obstacles_world])
                w = None
            rec = np.array(r["obstacles_world"], float)
            if (not a.allow_custom_scene) and (rec.shape != stock[key].shape or
                    np.abs(rec - stock[key]).max() > 1e-6):
                stats["skip:custom_scene"] += 1
                skipped.append((name, "recorded scene differs from the stock "
                                      "scene; load_target cannot rebuild it"))
                continue

        tgt = {
            "name": name.replace(".json", ""),
            "case": r["case"], "seed": r["seed"], "algo": r["algo"],
            "index": r["index"], "d_min": r["d_min"], "eta": r["eta"],
            "lam": r["lam"], "k": r["k"],
            "channel": r["channel"], "max_steps": r["max_steps"],
            "G0_commanded": r["G0"], "G1": r["G1"],
            "G1_prime_truth": r["G1p"],
            "kind": kind,
            "obstacles_world": r.get("obstacles_world"),
            # load_target pins these; without the radius it would fall back to
            # 0.05, which is wrong for the 0.055 constructed scenes.
            "obstacle_radius": r.get("obstacle_radius"),
            "provenance": {"rq3_id": e["id"], "source": e["source"],
                           "revalidated": True,
                           "baseline_min": e.get("baseline_min"),
                           "attack_mins": e.get("attack_mins")},
        }
        json.dump(tgt, open(os.path.join(a.out, name), "w"), indent=1,
                  default=float)
        written.append(name)
        stats[f"emit:{kind}"] += 1

    json.dump({"written": written,
               "skipped": [{"name": n, "reason": w} for n, w in skipped]},
              open(os.path.join(a.out, "_manifest.json"), "w"), indent=1)
    print(f"  wrote {len(written)} targets -> {a.out}")
    for k, v in sorted(stats.items()):
        print(f"    {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
