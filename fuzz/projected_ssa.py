"""
p-SSA (Projected Safe Set Algorithm) -- a faithful port of the paper's headline,
tuning-free safe controller, which SPARK does not ship.

Reference: Chen, Sun, Liu, "Dexterous Safe Control for Humanoids in Cluttered
Environments via Projected Safe Set Algorithm" (papers/ssa.pdf), Section IV-B,
equations (12)-(13).

Two phases per control step (decoupling performance and safety, so there is NO
Q_s trade-off weight to tune -- unlike r-SSA):

  Phase I  (12):  min_{u,s} ||s||_2^2
                  s.t. L_f phi + L_g phi u <= -eta + s,  u- <= u <= u+,  s >= 0
                  -> the MINIMAL projection slack s* (the nearest feasible
                     relaxation of the safety constraints). Q_s = I.

  Phase II (13):  min_u ||u - u_ref||_Q^2
                  s.t. L_f phi + L_g phi u <= -eta + s*,  u- <= u <= u+
                  -> u_safe. Guaranteed feasible because s* already made the
                     constraint set feasible. Q = I.

Paper setup matched here: Q = Q_s = I, p = 2, eta = 0.5 (set in fuzz/config.py).

SPARK's QP convention: safe_control() passes Lf already augmented with +eta (via
eta_fn), and the constraint upper bound is -Lf, i.e. L_g u <= -Lf encodes
L_f phi + L_g phi u <= -eta. Phase I reuses the relaxed-SSA matrices but with the
objective ||s||^2; Phase II is the basic-SSA QP with the bound relaxed by s*
(pass Lf - s* so the bound becomes -Lf + s*).

Registered into the spark_policy namespace at import time so
initialize_class("ProjectedSafeSetAlgorithm") resolves it WITHOUT editing any
core SPARK file (initialize_class searches spark_policy/agent/robot/task).
"""

import numpy as np
import osqp
from scipy import sparse

import spark_policy
from spark_policy.safe.safe_algo.value_based.ssa.basic_safe_set_algorithm import BasicSafeSetAlgorithm
from spark_policy.safe.safe_algo.value_based.ssa.relaxed_safe_set_algorithm import RelaxedSafeSetAlgorithm


class ProjectedSafeSetAlgorithm(RelaxedSafeSetAlgorithm):

    def __init__(self, **kwargs):
        # Relaxed.__init__ requires slack_weight; p-SSA uses Q_s = I -> weight 1.0.
        kwargs.setdefault("slack_weight", 1.0)
        super().__init__(**kwargs)
        print(f"Initializing {self.__class__.__name__}")
        self.slack_regularization_order = 2   # p = 2, per the paper

    # ------------------------------------------------------------------ #
    def _phase1_projection(self, Lg, Lf, eps_=1.00e-2, abs_=1.00e-2):
        """Eq. (12): minimal ||s||^2 that makes the safety constraints feasible.
        Returns s* (length m). Only s* is kept; the phase-I u is discarded."""
        n = Lg.shape[1]
        m = Lf.shape[0]

        # Objective: min ||s||^2. Tiny ridge on u keeps OSQP's P positive-definite
        # (u is box-bounded and discarded, so this does not affect s*).
        P = sparse.block_diag([1e-6 * sparse.eye(n), sparse.eye(m)]).tocsc()
        q = np.zeros(n + m)

        # Constraints (same structure as r-SSA): Lg u - s <= -Lf ; u in [u-,u+] ; s>=0
        C_upper = sparse.hstack([sparse.csc_matrix(Lg), -sparse.eye(m)])
        C_lower = sparse.eye(n + m)
        Cmat = sparse.vstack([C_upper, C_lower]).tocsc()
        l = np.concatenate([-np.inf * np.ones(m), self.control_min.flatten(), np.zeros(m)])
        u = np.concatenate([-Lf.reshape(-1), self.control_max.flatten(), np.inf * np.ones(m)])

        prob = osqp.OSQP()
        prob.setup(P, q, Cmat, l, u, alpha=1.0, eps_abs=abs_, eps_rel=eps_, verbose=False)
        result = prob.solve()
        if result.info.status == 'solved':
            s_star = result.x[n:].flatten()
            s_star[s_star < 0] = 0.0
            return s_star
        # Degenerate fallback: no relaxation (phase II then behaves like basic SSA).
        return np.zeros(m)

    # ------------------------------------------------------------------ #
    def qp_solver(self, u_ref, Q_u, Lg, Lf, eps_=1.00e-2, abs_=1.00e-2):
        """Two-phase p-SSA. Returns (u_safe, s*) -- s* is surfaced as the
        'violation' signal (the projection magnitude)."""
        # Phase I: project the (possibly infeasible) constraint set to nearest feasible.
        s_star = self._phase1_projection(Lg, Lf, eps_, abs_)
        # Phase II: basic-SSA QP with the bound relaxed by s* (pass Lf - s*).
        Lf_relaxed = (Lf.reshape(-1) - s_star).reshape(-1, 1)
        u_sol, _ = BasicSafeSetAlgorithm.qp_solver(self, u_ref, Q_u, Lg, Lf_relaxed, eps_, abs_)
        return u_sol, s_star


# Non-invasive registration (mirrors the task registration in goal_insertion_task.py).
spark_policy.ProjectedSafeSetAlgorithm = ProjectedSafeSetAlgorithm
