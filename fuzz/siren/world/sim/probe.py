"""
Read-only instrumentation of the live SPARK filter.

Two things the safety filter knows but does not report, and that we need:

  1. The raw ingredients of the margin — the danger number phi, its control
     sensitivity L_g phi, and its drift L_f phi. SPARK's safety index computes
     these every step and throws them away, so we wrap phi() and keep the last
     value. Nothing is modified; the filter behaves exactly as before.

  2. Whether the filter GAVE UP. A hard filter whose QP is infeasible falls back
     to the unmodified reference control and drives on. That is observable from
     outside (the robot stops dodging) and is the only infeasibility signal a
     gray/black-box attacker is allowed to use — as distinct from the analytic
     g < 0 prediction, which needs the model. We count it by wrapping qp_solver.

Both wrappers are idempotent and leave SPARK's source untouched.
"""

import types

import numpy as np
import osqp
from scipy import sparse


def install_probe(harness) -> bool:
    """Attach both instruments. Returns True if the give-up counter was installed
    (only meaningful for the hard, no-slack filters)."""
    algo = harness.algo.safe_controller.safe_algo
    harness._infeasible = [0]

    # ---- 1. stash (phi, L_g phi, L_f phi) each time the index is evaluated ---
    if not getattr(algo, "_siren_phi_probe", False):
        si = algo.safety_index
        original_phi = si.phi

        def phi_wrapped(x, task_info):
            out = original_phi(x, task_info)
            algo._last_phi = out            # (phi, Lg, Lf, phi0, phi0dot)
            return out

        si.phi = phi_wrapped
        algo._siren_phi_probe = True

    # ---- 2. count give-ups (hard SSA only; soft filters never go infeasible) --
    if algo.__class__.__name__ != "BasicSafeSetAlgorithm":
        return False
    if getattr(algo, "_siren_giveup_probe", False):
        return True

    def qp_solver_counted(self, u_ref, Q_u, Lg, Lf, eps_=1.00e-2, abs_=1.00e-2):
        n = u_ref.shape[0]
        m = Lf.shape[0]
        Q = sparse.csc_matrix(Q_u)
        q = -(Q_u.T @ u_ref.reshape(-1, 1)).reshape(-1)
        C = sparse.vstack([sparse.csc_matrix(Lg), sparse.eye(n)]).tocsc()
        lo = np.concatenate([-np.inf * np.ones(m), self.control_min.flatten()])
        hi = np.concatenate([-Lf.reshape(-1), self.control_max.flatten()])

        prob = osqp.OSQP()
        prob.setup(Q, q, C, lo, hi, alpha=1.0, eps_abs=abs_, eps_rel=eps_, verbose=False)
        r = prob.solve()
        if r.info.status != "solved":
            # THE GIVE-UP: no safe control exists, so SPARK hands back the
            # obstacle-blind reference control and the robot drives into trouble.
            harness._infeasible[0] += 1
            v = Lf + Lg @ u_ref.reshape(-1, 1)
            v = v.reshape(-1)
            v[v < 0] = 0
            return u_ref, v
        return r.x[:n].flatten(), np.zeros(m)

    algo.qp_solver = types.MethodType(qp_solver_counted, algo)
    algo._siren_giveup_probe = True
    return True


def read_raw(harness) -> dict:
    """Pull this step's raw ingredients out of the live filter.

    Returns phi / Lg / Lf / phi_mask / u_lim, plus the demand parameters the
    filter is currently configured with. Everything downstream (authority,
    margin) is computed from these by world/derived.py, which never sees SPARK.
    """
    algo = harness.algo.safe_controller.safe_algo
    last = getattr(algo, "_last_phi", None)
    if last is None:
        return {}

    phi = np.asarray(last[0], dtype=float).reshape(-1)
    Lg = np.atleast_2d(np.asarray(last[1], dtype=float))
    Lf = np.asarray(last[2], dtype=float).reshape(-1)

    u_lim = np.abs(np.asarray(algo.control_max, dtype=float).reshape(-1))
    phi_mask = np.asarray(algo.safety_index.phi_mask, dtype=float).reshape(-1)

    return {
        "phi": phi,
        "Lg": Lg,
        "Lf": Lf,
        "phi_mask": phi_mask,
        "u_lim": u_lim,
        "eta": float(getattr(algo, "eta_default", np.nan)),
        "lam": float(getattr(algo, "lambda_default", np.nan)),
    }


def giveups(harness) -> int:
    return harness._infeasible[0] if hasattr(harness, "_infeasible") else 0


def reset_giveups(harness):
    if hasattr(harness, "_infeasible"):
        harness._infeasible[0] = 0
