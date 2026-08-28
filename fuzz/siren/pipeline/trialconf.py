"""
Load a trial config and turn it into FilterSpecs.

The point of the config file is that every parameter reaching SPARK is written
down in one place. That matters most for the SPARK-default trial: a claim about
"SPARK as shipped" is only meaningful if nothing was silently overridden, and
several of our long-standing values (d_min, eta, phi_k) turned out to differ
from SPARK's by 5-10x without that ever being explicit.

Fields FilterSpec has a named slot for (eta, lam, d_min, k) are set directly.
Everything else -- safety_buffer, use_slack, slack_regularization_order, the
self min_distance, enable_self_collision, phi_n -- goes through
FilterSpec.overrides as dotted paths applied last in apply_filter_spec.
"""

import numpy as np
import yaml


ALL_FILTERS = ["ssa", "rssa", "pssa", "cbf", "rcbf", "sss", "rsss"]

#: Benchmark families that are BOTH state-round-trippable and physically
#: searchable. ALL FOUR G1FixedBase_*_AG_DO_* are excluded: a fixed base cannot
#: evade a 0.400 m/s obstacle. D1 is blind to the motion (Cartesian_Lf = 0)
#: and has no authority (C = 0.0268); D2 sees it but the guarded torso is
#: driven only by three waist joints. Measured 0/47 and 0/55 G0 pass C1.
USABLE_SCENES = [
    "G1FixedBase_D1_AG_SO_v0", "G1FixedBase_D1_AG_SO_v1",
    "G1FixedBase_D2_AG_SO_v0", "G1FixedBase_D2_AG_SO_v1",
    "G1MobileBase_D1_WG_SO_v0", "G1MobileBase_D1_WG_SO_v1",
    "G1MobileBase_D1_WG_DO_v0", "G1MobileBase_D1_WG_DO_v1",
    "G1MobileBase_D2_WG_SO_v0", "G1MobileBase_D2_WG_SO_v1",
    "G1MobileBase_D2_WG_DO_v0", "G1MobileBase_D2_WG_DO_v1",
]


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def index_for(case):
    """D2 cases use the velocity-augmented (second-order) index, D1 the
    distance one. SPARK asserts if they disagree with the robot's control mode,
    so this is determined by the case name, not by the config."""
    return "velocity" if "_D2_" in case else "distance"


def spec_for(cfg, algo, case):
    """Build the FilterSpec for one (filter, scenario) pair from the config."""
    from ..world.types import FilterSpec

    f = cfg["filters"][algo]
    idx = cfg.get("index", {})
    overrides = dict(idx.get("overrides", {}))
    overrides.update(cfg.get("safe_algo_overrides", {}))
    # phi_n has no SPARK-wide default -- it is a required kwarg of the
    # second-order index, supplied by whichever runner constructs it. Carry it
    # from the config so it is explicit rather than hardcoded.
    if index_for(case) == "velocity" and idx.get("phi_n") is not None:
        overrides["safety_index.phi_n"] = float(idx["phi_n"])

    return FilterSpec(
        algo=algo,
        index=index_for(case),
        d_min=float(idx.get("d_min", 0.02)),
        eta=(float(f["eta"]) if f.get("eta") is not None else None),
        lam=(float(f["lam"]) if f.get("lam") is not None else None),
        k=(float(idx["phi_k"]) if idx.get("phi_k") is not None else None),
        slack_weight=(float(f["slack_weight"])
                      if f.get("slack_weight") is not None else None),
        c=(float(f["c"]) if f.get("c") is not None else None),
        overrides=overrides,
        label=cfg.get("name", "trial"))


def describe(cfg):
    lines = [f"config: {cfg.get('name')}  -- {cfg.get('description','')}"]
    idx = cfg.get("index", {})
    lines.append(f"  index   d_min={idx.get('d_min')}  phi_n={idx.get('phi_n')}"
                 f"  phi_k={idx.get('phi_k')}")
    for algo in ALL_FILTERS:
        f = cfg["filters"].get(algo)
        if f is None:
            continue
        rate = (f"eta={f['eta']}" if f.get("eta") is not None
                else f"lam={f['lam']}")
        native = "" if f.get("spark_native", True) else "   [NOT a SPARK filter]"
        lines.append(f"  {algo:<5} {f['class_name']:<32} {rate:<12}"
                     f"slack={f.get('slack_weight')}{native}")
    for k, v in (cfg.get("safe_algo_overrides") or {}).items():
        lines.append(f"  raw     {k} = {v}")
    for k, v in (idx.get("overrides") or {}).items():
        lines.append(f"  raw     {k} = {v}")
    return "\n".join(lines)
