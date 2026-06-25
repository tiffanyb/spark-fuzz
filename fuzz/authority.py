"""
Control-authority analysis of the SSA safety filter.

This implements the c(x) machinery derived in fuzz/cx_derivation.pdf:

    c_i(x) = max_{u in box} ( -L_g phi_i . u ) = sum_k u_lim,k |[L_g phi_i]_k|

is the largest rate at which the actuators can drive the i-th safety index DOWN.
For a single active constraint the SSA QP is feasible iff c_i(x) >= eta + L_f phi_i;
for several active constraints feasibility is the min-max margin

    mu(x) = min_{u in box} max_{i active} ( L_f phi_i + L_g phi_i . u + eta )   (<= 0 feasible)

solved here as a small LP. mu(x) > 0  <=>  no safe control exists  <=>  basic SSA
goes primal-infeasible (gives up -> u_ref). The relaxed variants never go infeasible
but must use slack s_i* = (eta + L_f phi_i - c_i)_+, so the same quantity drives them.

We read L_g phi straight from the safety index (it already equals n^T J G), so nothing
here re-derives kinematics -- we only weight |L_g phi| by the actuator limits.

Key entry points:
  - eval_state(algo)            : c_i, active set, mu, slack-pressure at the CURRENT sim state
  - rollout(harness, schedule)  : run a trajectory, return c(x_t) / pressure / mu per step
                                  and the summary J = max_t pressure (the attack objective)
"""

import types

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover
    linprog = None


# ---------------------------------------------------------------------------- #
#  Reading L_g phi from the live controller
# ---------------------------------------------------------------------------- #
def install_probe(harness):
    """Wrap safety_index.phi so each call stashes (phi, Lg, Lf) on the algo as
    `_last_phi`. Idempotent; returns the safe-algo object to read from."""
    algo = harness.algo.safe_controller.safe_algo
    if getattr(algo, "_authority_probe", False):
        return algo
    si = algo.safety_index
    orig = si.phi

    def phi_wrapped(x, task_info):
        out = orig(x, task_info)
        algo._last_phi = out            # (phi, Lg, Lf, phi0, phi0dot)
        return out

    si.phi = phi_wrapped
    algo._authority_probe = True
    return algo


def control_limit(algo):
    """Per-DoF actuator limit u_lim (the symmetric box half-width)."""
    return np.abs(np.asarray(algo.control_max, dtype=float).flatten())


def eta_of(algo):
    return float(algo.eta_default)


# ---------------------------------------------------------------------------- #
#  c(x) and the feasibility margin
# ---------------------------------------------------------------------------- #
def c_per_constraint(Lg, u_lim):
    """c_i(x) = sum_k u_lim,k |Lg_ik|  for every constraint row of Lg."""
    Lg = np.asarray(Lg, dtype=float)
    return np.abs(Lg) @ u_lim          # (num_constraint,)


def active_mask(phi, phi_mask):
    """Constraints the QP actually enforces: phi_mask>0 AND phi>=0 (matches eta_fn)."""
    phi = np.asarray(phi, dtype=float).reshape(-1)
    pm = np.asarray(phi_mask, dtype=float).reshape(-1)
    return (pm > 0) & (phi >= 0)


def feasibility_margin(Lg, Lf, eta, u_lim, active):
    """mu(x) = min_{u in box} max_{i in active} (L_f phi_i + L_g phi_i . u + eta).

    mu <= 0  => a safe control exists (QP feasible).
    mu  > 0  => infeasible (basic SSA gives up).  Returns -inf if nothing active."""
    idx = np.where(active)[0]
    if idx.size == 0:
        return -np.inf
    if linprog is None:
        raise RuntimeError("scipy.optimize.linprog unavailable")
    A = np.asarray(Lg, dtype=float)[idx]                       # (k, m)
    Lf = np.asarray(Lf, dtype=float).reshape(-1)[idx]
    k, m = A.shape
    rhs = -(Lf + eta)                                          # Lg_i u <= rhs_i
    # variables z = [u (m), t]; min t  s.t.  Lg_i u - t <= rhs_i ; box on u
    c = np.zeros(m + 1); c[-1] = 1.0
    A_ub = np.hstack([A, -np.ones((k, 1))])
    bounds = [(-u_lim[j], u_lim[j]) for j in range(m)] + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=rhs, bounds=bounds, method="highs")
    return float(res.fun) if res.success else np.nan


def eval_state(algo, eta=None, compute_mu=True):
    """Analyse the CURRENT sim state from the stashed L_g phi. Returns a dict with
    per-constraint c, the active set, the binding (lowest-c active) constraint, the
    single-constraint slack pressure, and the exact multi-constraint margin mu.

    compute_mu=False skips the (per-step) LP and uses the cheap single-constraint
    proxy: pressure>0 already certifies infeasibility (one constraint alone fails)."""
    phi, Lg, Lf = algo._last_phi[0], algo._last_phi[1], algo._last_phi[2]
    phi = np.asarray(phi, dtype=float).reshape(-1)
    Lf = np.asarray(Lf, dtype=float).reshape(-1)
    u_lim = control_limit(algo)
    eta = eta_of(algo) if eta is None else eta

    c = c_per_constraint(Lg, u_lim)
    act = active_mask(phi, algo.safety_index.phi_mask)
    idx = np.where(act)[0]

    if idx.size:
        # per-constraint slack each active constraint would need: (eta + Lf - c)_+
        need = np.maximum(0.0, (eta + Lf[idx]) - c[idx])
        binding = idx[int(np.argmin(c[idx]))]
        c_min = float(c[idx].min())
        pressure = float(need.max())               # single-constraint proxy
    else:
        binding, c_min, pressure = -1, np.inf, 0.0

    if compute_mu:
        mu = feasibility_margin(Lg, Lf, eta, u_lim, act)
        infeasible = bool(np.isfinite(mu) and mu > 1e-9)
    else:
        mu = np.nan
        infeasible = bool(pressure > 1e-9)   # single-constraint proxy (sufficient)
    return {
        "n_active": int(idx.size),
        "active_idx": idx.tolist(),
        "binding": int(binding),
        "c_min": c_min,                 # min c over active constraints
        "c_binding": float(c[binding]) if binding >= 0 else np.inf,
        "eta": float(eta),
        "pressure": pressure,           # max_i (eta+Lf-c)_+  over active i
        "mu": float(mu),                # exact margin; >0 => infeasible
        "infeasible": infeasible,
    }


# ---------------------------------------------------------------------------- #
#  Trajectory evaluation: c(x_t) along a closed-loop rollout
# ---------------------------------------------------------------------------- #
def rollout(harness, schedule, max_steps=None, eta=None, compute_mindist=True, compute_mu=True):
    """Run schedule (list of base-frame goals) through the closed loop and record,
    per step, the control-authority state. Returns (steps, summary).

    summary["J"] = max_t pressure_t  -- the attack objective (how far the trajectory
    drove the safety constraint past what control can meet). J>0 means the trajectory
    reached an SSA-infeasible state.
    """
    from spark_utils import compute_masked_distance_matrix

    algo = install_probe(harness)
    if eta is not None:
        algo.eta_default = float(eta)   # drive BOTH the controller and the c(x) test
    eta = eta_of(algo)
    max_steps = max_steps if max_steps is not None else harness.max_steps

    agent_feedback, task_info = harness._reset_scene()
    harness.env.task.set_goal_schedule(schedule)
    u_safe, action_info = harness.algo.act(agent_feedback, task_info)

    steps = []
    first_fail = None
    for t in range(max_steps):
        agent_feedback, task_info = harness.env.step(u_safe, action_info)
        u_safe, action_info = harness.algo.act(agent_feedback, task_info)
        task = harness.env.task

        s = eval_state(algo, eta=eta, compute_mu=compute_mu)
        s["step"] = t
        s["trigger_safe"] = bool(action_info.get("trigger_safe", False))
        s["dist_final"] = float(task.dist_to_final)

        if compute_mindist:
            of = task_info["obstacle"]["frames_world"]
            og = task_info["obstacle"]["geom"]
            if len(of) > 0:
                dmat, _ = compute_masked_distance_matrix(
                    frame_list_1=task.robot_frames_world,
                    geom_list_1=harness.robot_cfg.CollisionVol.values(),
                    frame_list_2=of, geom_list_2=og)
                s["min_dist_env"] = float(dmat.min()) if dmat is not None else np.inf
            else:
                s["min_dist_env"] = np.inf
        else:
            s["min_dist_env"] = np.nan
        s["collided"] = bool(np.isfinite(s["min_dist_env"]) and s["min_dist_env"] < 0.0)

        if s["infeasible"] and first_fail is None:
            first_fail = t
        steps.append(s)

        if task.reached_final or s["collided"]:
            break

    pressures = [s["pressure"] for s in steps]
    J = max(pressures) if pressures else 0.0
    t_peak = int(np.argmax(pressures)) if pressures else -1
    summary = {
        "J": float(J),                              # attack objective: max_t pressure
        "t_peak": t_peak,
        "first_infeasible_step": first_fail,
        "n_infeasible": int(sum(s["infeasible"] for s in steps)),
        "min_c_over_traj": float(min((s["c_min"] for s in steps if np.isfinite(s["c_min"])), default=np.inf)),
        "eta": float(eta),
        "n_steps": len(steps),
        "reached_final": bool(steps[-1]["dist_final"] < 0.05) if steps else False,
        "collided": bool(any(s["collided"] for s in steps)),
    }
    return steps, summary
