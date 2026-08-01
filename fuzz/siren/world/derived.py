"""
Pure math: control authority C and the feasibility margin g.

numpy in, numbers out. NO SPARK, NO MuJoCo — so every formula here is testable
against hand-made arrays in milliseconds, which matters because these two
quantities are the whole technical claim.

The one identity everything rests on
------------------------------------
A value-based safety filter enforces, for each active constraint i,

    L_f phi_i + L_g phi_i . u  <=  -demand_i          (danger must fall at
                                                       least this fast)

The control u lives in the actuator box U = [-u_lim, u_lim]. The most negative
the control term can be made is

    min_{u in U} L_g phi_i . u  =  -c_i ,
        with   c_i = sum_k u_lim,k * |L_g phi_i,k|        <-- CONTROL AUTHORITY

so a control satisfying the constraint exists  <=>  c_i - demand_i - L_f phi_i >= 0.
That quantity is the margin

    g_i = c_i - demand_i - L_f phi_i           g_i < 0  =>  guarantee is void.

Note where each piece comes from: c_i is the only place L_g phi enters; demand_i
is the filter's choice; L_f phi_i is pure drift (zero for a kinematic robot with
static obstacles).

Why C_d ("retreat capacity") is index-free
------------------------------------------
Every collision safety index is some function of clearance, phi = F(d, ddot).
Hence L_g phi = F_d * L_g d : the unknown formula RESCALES the sensitivities, it
never redirects them. So an attacker who does not know the index still gets the
right RANKING of poses from c(.) computed on whatever gradient it can see — the
missing factor |F_d| is a common multiplier. That is exactly the strict
black-box position: unscaled, not blind.
"""

from typing import Optional, Tuple

import numpy as np

try:
    from scipy.optimize import linprog
except Exception:                                    # pragma: no cover
    linprog = None


# ---------------------------------------------------------------------------- #
#  Control authority
# ---------------------------------------------------------------------------- #
def control_authority(Lg: np.ndarray, u_lim: np.ndarray) -> np.ndarray:
    """c_i = sum_k u_lim,k * |L_g phi_i,k|   for every constraint row of Lg.

    "How fast can the actuators drive danger DOWN, at full effort, in this pose."
    Lg is (n_constraint, n_dof); returns (n_constraint,).
    """
    Lg = np.atleast_2d(np.asarray(Lg, dtype=float))
    u_lim = np.abs(np.asarray(u_lim, dtype=float).reshape(-1))
    return np.abs(Lg) @ u_lim


def demand_vector(shape: str, phi: np.ndarray,
                  eta: Optional[float] = None,
                  lam: Optional[float] = None) -> np.ndarray:
    """What the filter insists on, per constraint.

        constant      d = eta         (SSA family)  — bites even at the boundary
        proportional  d = lambda*phi  (CBF/SSS)     — vanishes at phi = 0, so
                                                      grazing cannot defeat it;
                                                      the attacker must penetrate
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    if shape == "constant":
        return np.full_like(phi, float(eta if eta is not None else 0.0))
    if shape == "proportional":
        return float(lam if lam is not None else 0.0) * phi
    return np.zeros_like(phi)                        # heuristic filters (SMA/PFM)


def active_set(phi: np.ndarray, phi_mask: np.ndarray, shape: str = "constant") -> np.ndarray:
    """Which constraints the filter actually enforces right now.

    SSA/SSS engage once danger is non-negative (phi >= 0); a CBF keeps every
    masked constraint active everywhere.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    mask = np.asarray(phi_mask, dtype=float).reshape(-1) > 0
    if shape == "proportional_all":
        return mask
    return mask & (phi >= 0)


# ---------------------------------------------------------------------------- #
#  The margin
# ---------------------------------------------------------------------------- #
def margin(c: np.ndarray, demand: np.ndarray, Lf: np.ndarray,
           active: np.ndarray) -> dict:
    """g_i = c_i - demand_i - L_f phi_i over the active constraints.

    Returns the tightest one (g_min < 0 means NO control in the box can hold the
    demanded rate — the guarantee of every value-based filter is void), which
    constraint binds, and the forced-slack pressure a soft filter would have to
    pay there.
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    demand = np.asarray(demand, dtype=float).reshape(-1)
    Lf = np.asarray(Lf, dtype=float).reshape(-1)
    idx = np.where(np.asarray(active).reshape(-1))[0]

    if idx.size == 0:
        return {"g_min": np.inf, "binding": -1, "pressure": 0.0,
                "c_min": np.inf, "n_active": 0}

    g_i = c[idx] - demand[idx] - Lf[idx]
    j = int(np.argmin(g_i))
    return {
        "g_min": float(g_i.min()),
        "binding": int(idx[j]),
        "pressure": float(max(0.0, -g_i.min())),     # slack a soft filter must take
        "c_min": float(c[idx].min()),
        "n_active": int(idx.size),
    }


def exact_margin_lp(Lg: np.ndarray, demand: np.ndarray, Lf: np.ndarray,
                    u_lim: np.ndarray, active: np.ndarray) -> float:
    """Multi-constraint feasibility, exactly:

        mu(x) = min_{u in U} max_{i active} ( L_f phi_i + L_g phi_i . u + demand_i )

    mu <= 0 => some single control satisfies ALL active constraints at once.
    mu >  0 => infeasible. g_min < 0 is the (tight) single-constraint certificate;
    this LP is the honest multi-constraint version, used when several constraints
    conflict — a pincer no single control can escape.
    """
    idx = np.where(np.asarray(active).reshape(-1))[0]
    if idx.size == 0:
        return -np.inf
    if linprog is None:                              # pragma: no cover
        raise RuntimeError("scipy.optimize.linprog unavailable")

    A = np.atleast_2d(np.asarray(Lg, dtype=float))[idx]
    beta = (np.asarray(demand, dtype=float).reshape(-1)[idx]
            + np.asarray(Lf, dtype=float).reshape(-1)[idx])
    u_lim = np.abs(np.asarray(u_lim, dtype=float).reshape(-1))
    k, m = A.shape

    # variables z = [u (m), t];  minimize t  s.t.  Lg_i u - t <= -beta_i
    cost = np.zeros(m + 1); cost[-1] = 1.0
    A_ub = np.hstack([A, -np.ones((k, 1))])
    bounds = [(-u_lim[j], u_lim[j]) for j in range(m)] + [(None, None)]
    res = linprog(cost, A_ub=A_ub, b_ub=-beta, bounds=bounds, method="highs")
    return float(res.fun) if res.success else np.nan


# ---------------------------------------------------------------------------- #
#  One-stop evaluation
# ---------------------------------------------------------------------------- #
def evaluate(Lg, Lf, phi, phi_mask, u_lim, demand_shape="constant",
             eta=None, lam=None, exact=False) -> dict:
    """Everything derived, for one instant.

    C_d is reported alongside C_phi. Under the first-order index
    (phi = d_min - d) the two are identical, because F_d = -1; under any other
    index they differ by the unknown common factor |F_d|, which is precisely the
    scale a strict black-box attacker never learns and — because it is common to
    all poses — never needs.
    """
    c = control_authority(Lg, u_lim)
    demand = demand_vector(demand_shape, phi, eta, lam)
    act = active_set(phi, phi_mask, demand_shape)
    m = margin(c, demand, Lf, act)
    engaged = m["n_active"] > 0
    phi_arr = np.asarray(phi, dtype=float).reshape(-1)

    # ---------------------------------------------------------------------- #
    #  Two distinct families, deliberately NOT mixed.
    #
    #  MARGIN (g, demand, C at the binding constraint) is a statement about a
    #  constraint the filter is ACTUALLY ENFORCING. For an inactive constraint
    #  it is meaningless: the filter is not trying to satisfy it, so "the margin
    #  is negative" says nothing. An earlier version reported C - demand for the
    #  nearest inactive pair as a stand-in "guidance" value; that was not merely
    #  uninformative but INVERTED -- candidates that never engaged the filter
    #  received the most negative value of all, so ranking by it was backwards.
    #  The margin is therefore undefined (+inf) unless something is engaged.
    #
    #  PROXIMITY (phi_nearest, C_nearest) is always defined and is honest about
    #  what it is: how close the robot came to the boundary, and how much
    #  authority it had there. Authority is well defined for any pair -- it is
    #  just the actuator reach against that constraint's geometry -- so nothing
    #  is lost by reporting it while the constraint is dormant.
    # ---------------------------------------------------------------------- #
    mask_all = np.asarray(phi_mask, dtype=float).reshape(-1) > 0
    if mask_all.any():
        idx = np.where(mask_all)[0]
        j_near = int(idx[int(np.argmax(phi_arr[idx]))])     # closest to boundary
        phi_nearest = float(phi_arr[j_near])
        C_nearest = float(c[j_near])
    else:
        j_near, phi_nearest, C_nearest = -1, -np.inf, np.inf

    out = {
        "engaged": engaged,
        # ---- proximity: always defined, explicitly not a margin ----
        "phi": phi_nearest,
        "C_d": C_nearest,                # index-free ruler at the nearest pair
        # ---- margin: defined ONLY while the filter is enforcing ----
        "C_phi": m["c_min"] if engaged else np.inf,
        "g": m["g_min"] if engaged else np.inf,
        "demand": (float(demand[m["binding"]])
                   if (engaged and m["binding"] >= 0) else np.nan),
        "pressure": m["pressure"] if engaged else 0.0,
        "n_active": m["n_active"] if engaged else 0,
        "binding": m["binding"] if engaged else j_near,
        # only a genuinely enforced constraint can be violated
        "predicted_infeasible": bool(engaged and m["g_min"] < 0.0),
    }
    if exact and engaged:
        mu = exact_margin_lp(Lg, demand, Lf, u_lim, act)
        out["mu"] = mu
        out["predicted_infeasible"] = bool(np.isfinite(mu) and mu > 1e-9)
    return out


# ---------------------------------------------------------------------------- #
#  P5: is the demand even achievable here?
# ---------------------------------------------------------------------------- #
def demand_achievable(c_engaged, g_engaged, demand_value):
    """Can the filter actually meet its own demand where it engages?

    Judged EMPIRICALLY, from the fraction of engaged steps at which a safe
    control existed (g >= 0), rather than inferred from the authority range.
    That matters here: the authority across engaged constraints spans a wide
    range, but the QP is CONJUNCTIVE -- every active constraint must be
    satisfied at once -- so the weakest active constraint governs, and it is
    consistently the same weak one. Measuring feasibility directly sidesteps
    having to reason about which constraint binds.

    A filter that is never feasible while engaged does not operate at all: it is
    dormant or broken, "attack" degenerates to "get close enough to engage", and
    `g = C - demand` is a saturated constant that cannot rank candidates. That is
    the same dead end as an inert filter, reached through the demand instead of
    the keep-out distance.
    """
    c = np.asarray([v for v in np.atleast_1d(c_engaged) if np.isfinite(v)],
                   dtype=float)
    g = np.asarray([v for v in np.atleast_1d(g_engaged) if np.isfinite(v)],
                   dtype=float)
    if c.size == 0 and g.size == 0:
        return {"verdict": "never_engaged", "n": 0}

    frac_feasible = float((g >= 0).mean()) if g.size else float("nan")
    lo = float(c.min()) if c.size else np.nan
    hi = float(c.max()) if c.size else np.nan

    if g.size == 0:
        verdict = "never_engaged"
    elif frac_feasible <= 0.0:
        verdict = "never_feasible"      # the filter cannot ever do its job here
    elif frac_feasible < 0.5:
        verdict = "mostly_infeasible"
    else:
        verdict = "achievable"

    return {"verdict": verdict, "demand": float(demand_value),
            "frac_feasible_when_engaged": frac_feasible,
            "C_min": lo, "C_max": hi, "C_spread": (hi - lo) if c.size else np.nan,
            "ratio_to_worst": (float(demand_value) / lo
                               if (c.size and lo > 0) else np.inf),
            "n": int(max(c.size, g.size))}
