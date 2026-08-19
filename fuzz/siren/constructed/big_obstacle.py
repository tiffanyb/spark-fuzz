"""Force a STATIC-obstacle INSERTION by attacking c(x) < d(x) directly.

The stall barrier: against a static obstacle the filter can always trade
progress for safety and park at clearance == d_min. It can only FAIL when its
demand exceeds its control authority, c(x) < d(x). So instead of only moving
obstacles around, this sweeps the terms of that inequality:

  * obstacle RADIUS   -- bigger sphere = the constraint bites further out and
                         the arm has less room to route around it
  * demand eta / lam  -- raise the demand the filter must meet
  * pincer            -- two obstacles so no single escape direction works
                         (worst-blend authority C = min_y c_y is far below any
                         individual c_i)

Requirements for a genuine INSERTION are unchanged and enforced:
    baseline [G1]      REACHED with clearance > 0
    leg 1 home->G1'    no collision   (else MODIFICATION)
    leg 2 G1'->G1      collision      (the insertion)
"""
import numpy as np
from spark_utils import Geometry, VizColor

#: robot_cfg.ControlLimit is a CLASS-level dict shared by every World.build, so
#: scaling it in a loop COMPOUNDS (0.1 then 0.1 gives 0.01, not 0.1). Capture the
#: pristine values once and always set ABSOLUTE values from them.
_PRISTINE_ULIM = {}


def set_ulim_scale(world, scale):
    """Set actuator limits to `scale` x their PRISTINE values (not cumulative)."""
    rc = world.harness.robot_cfg
    key = type(rc).__name__
    if key not in _PRISTINE_ULIM:
        _PRISTINE_ULIM[key] = {k: float(v) for k, v in rc.ControlLimit.items()}
    base = _PRISTINE_ULIM[key]
    for k in rc.ControlLimit:
        rc.ControlLimit[k] = base[k] * scale
    return base


def set_obstacles(world, positions_world, radius):
    """Pin obstacle positions AND resize their collision geometry."""
    task = world.harness.env.task
    obs = task.obstacle_task
    n = min(len(obs), len(positions_world))
    for i in range(n):
        obs[i].frame[:3, 3] = np.asarray(positions_world[i], float)
        obs[i].velocity = 0.0
        if getattr(obs[i], "last_frame", None) is not None:
            obs[i].last_frame = obs[i].frame.copy()
        if hasattr(obs[i], "last_displacement"):
            obs[i].last_displacement = np.zeros(3)
        task.obstacle_task_geom[i] = Geometry(type="sphere", radius=float(radius),
                                              color=VizColor.obstacle_task)
    for i in range(n, len(obs)):
        obs[i].frame[:3, 3] = np.array([6.0, 6.0, 6.0])
        obs[i].velocity = 0.0
    return n


def run_pinned(world, schedule, positions_world, radius, max_steps):
    from fuzz.siren.pipeline.stage1_search import set_channel
    h = world.harness
    orig = h.reset

    def patched(*a, **k):
        af, ti = orig(*a, **k)
        set_obstacles(world, positions_world, radius)
        return af, h.env.task.get_info(af)
    h.reset = patched
    try:
        set_channel(world, "arm")
        return world.run(schedule, max_steps=max_steps)
    finally:
        h.reset = orig


def legs(rec):
    c1 = [s.clearance for s in rec.steps if int(s.wp_idx) <= 1]
    c2 = [s.clearance for s in rec.steps if int(s.wp_idx) == 2]
    return (min(c1) if c1 else np.inf), (min(c2) if c2 else np.inf), len(c2)


def evaluate(world, G0, G1p, G1, pos_w, radius, steps):
    """Return (is_insertion, info). Baseline must be safe; leg1 clean; leg2 hit."""
    b = run_pinned(world, [G1], pos_w, radius, steps)
    b1, b2, _ = legs(b)
    bmin = min(b1, b2)
    if b.label != "REACHED" or bmin <= 0:
        return False, dict(stage="baseline", blabel=b.label, bmin=float(bmin))
    r = run_pinned(world, [G0, G1p, G1], pos_w, radius, steps)
    c1, c2, n2 = legs(r)
    ok = (r.label == "COLLISION" and n2 > 0 and c2 < 0 and c1 > 0)
    return ok, dict(stage="attack", blabel=b.label, bmin=float(bmin),
                    label=r.label, leg1=float(c1), leg2=float(c2), n2=n2)
