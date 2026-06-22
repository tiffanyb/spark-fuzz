"""
GoalInsertionFuzzer — searches for an admissible inserted goal G1' that prevents
the reactive safe controller from safely reaching the legitimate goal G1.

Phase-A scope: random-admissible search (the baseline search strategy in the
plan). A greedy/myopic baseline and stronger optimizers (CEM/BO) are left as
follow-ons; this class exposes a `propose` hook so they can be dropped in.

Per the plan (RESEARCH_PLAN_0611.md §15.5), the experiment is:
  1. baseline trial schedule=[G1] -> confirm G1 is reachable from G0. This is a
     HARD GATE: if the baseline does not reach G1, the scene is invalid for this
     controller and we do NOT fuzz (any "attack" would be meaningless).
  2. for each admissible candidate G1':
       a. SCREEN with schedule=[G1'] alone. G1' must itself be safely reachable
          from G0. If it is not, the trap is a *trivial one-hop trap* (e.g. G1'
          placed where reaching it alone already deadlocks/collides) and is
          discarded -- it says nothing about goal *sequencing*.
       b. only then run the attack schedule=[G1', G1]. A failure here is a
          genuine two-hop / sequence-induced trap: each goal is individually
          safe-reachable, but the order defeats the reactive controller.
  3. rank surviving candidates by how badly they break the safe reach.
"""

import types

import numpy as np
import osqp
from scipy import sparse

from .admissibility import sample_admissible, is_admissible
from .metrics import TrialOutcome


# Higher score = stronger attack. DEADLOCK (safe but trapped) is the headline
# result; COLLISION is also a failure; TIMEOUT is weak; REACHED is no attack.
_LABEL_BASE = {"DEADLOCK": 3.0, "COLLISION": 2.0, "TIMEOUT": 1.0, "REACHED": 0.0}


def instrument_infeasibility(harness):
    """Wrap the basic-SSA QP solver so each trial counts primal-infeasible steps
    (= steps where the safety filter gives up and falls back to u_ref). Sets
    harness._infeasible (a 1-element list). No-op for non-basic-SSA controllers.

    This is what distinguishes 'the filter actively failed' from 'the filter was
    never engaged': a confirmed attack with ~0 infeasible steps means the FEASIBLE
    filter was genuinely defeated; many infeasible steps means it had given up.
    """
    algo = harness.algo.safe_controller.safe_algo
    harness._infeasible = [0]
    if algo.__class__.__name__ != "BasicSafeSetAlgorithm":
        return False
    if getattr(harness, "_infeas_installed", False):
        return True

    def patched(self, u_ref, Q_u, Lg, Lf, eps_=1.00e-2, abs_=1.00e-2):
        n = u_ref.shape[0]; m = Lf.shape[0]
        Q = sparse.csc_matrix(Q_u); q = -(Q_u.T @ u_ref.reshape(-1, 1)).reshape(-1)
        Cmat = sparse.vstack([sparse.csc_matrix(Lg), sparse.eye(n)]).tocsc()
        l = np.concatenate([-np.inf * np.ones(m), self.control_min.flatten()])
        u = np.concatenate([-Lf.reshape(-1), self.control_max.flatten()])
        prob = osqp.OSQP(); prob.setup(Q, q, Cmat, l, u, alpha=1.0, eps_abs=abs_, eps_rel=eps_, verbose=False)
        r = prob.solve()
        if r.info.status != 'solved':
            harness._infeasible[0] += 1
            v = Lf + Lg @ u_ref.reshape(-1, 1); v = v.reshape(-1); v[v < 0] = 0
            return u_ref, v
        return r.x[:n].flatten(), np.zeros(m)

    algo.qp_solver = types.MethodType(patched, algo)
    harness._infeas_installed = True
    return True


def _trial(harness, schedule, max_steps):
    """Run one trial; return (outcome, infeasible_QP_steps)."""
    if hasattr(harness, "_infeasible"):
        harness._infeasible[0] = 0
    outcome = harness.run_trial(schedule, max_steps=max_steps)
    infeas = harness._infeasible[0] if hasattr(harness, "_infeasible") else 0
    return outcome, infeas


class GoalInsertionFuzzer:

    def __init__(self, harness, scene, seed=0, ik_check=False):
        self.harness = harness
        self.scene = scene
        self.rng = np.random.RandomState(seed)
        self.ik_check = ik_check

    # ------------------------------------------------------------------ #
    def _propose(self):
        """Sample one admissible candidate G1' (base frame). Override for other
        search strategies (greedy, CEM, ...)."""
        return sample_admissible(
            self.rng,
            bounds=self.scene["bounds"],
            base_frame=self.scene["base_frame"],
            obstacles_world=self.scene["obstacles_world"],
            keepout=self.scene["keepout"],
            robot_kinematics=self.harness.robot_kinematics if self.ik_check else None,
            ik_check=self.ik_check,
        )

    @staticmethod
    def _score(outcome: TrialOutcome, baseline: TrialOutcome) -> float:
        """Rank attacks. Base on outcome label, tie-break by how much worse than
        baseline (extra steps, deeper slack, closer to / into obstacles)."""
        s = _LABEL_BASE.get(outcome.label, 0.0)
        # Tie-breakers (small weights so the label dominates the ranking).
        s += 1e-3 * outcome.n_steps
        s += 1e-1 * outcome.peak_slack
        if np.isfinite(outcome.worst_min_dist_env):
            s += 1e-1 * max(0.0, -outcome.worst_min_dist_env)  # penetration depth
        return s

    # ------------------------------------------------------------------ #
    def run(self, n_candidates=50, max_steps=None, verbose=True,
            stop_on_first_hit=False, progress_every=10):
        """Random-admissible search. With stop_on_first_hit=True, keep sampling
        until the first confirmed attack (DEADLOCK/COLLISION) or until n_candidates
        screened candidates are exhausted. Reports infeasible-QP steps per attack."""
        G1 = self.scene["G1_base"]

        # --- Hard gate: G1 must be reachable from G0 for this scene/controller. ---
        baseline, base_infeas = _trial(self.harness, [G1], max_steps)
        if verbose:
            print(f"[baseline] [G1] -> {baseline.label} (reached={baseline.reached_final}, "
                  f"steps={baseline.n_steps}, infeasible_QP={base_infeas}/{baseline.n_steps})", flush=True)
        if not baseline.reached_final:
            if verbose:
                print("  ABORT: baseline G0->G1 not reached; scene invalid -- not fuzzing.")
            return {"baseline": baseline, "results": [], "aborted": True,
                    "abort_reason": "baseline_unreachable", "n_one_hop_skipped": 0,
                    "baseline_infeasible": base_infeas, "n_screened": 0, "first_hit": None}

        results = []
        n_one_hop_skipped = 0
        n_screened = 0
        first_hit = None
        for i in range(n_candidates):
            cand = self._propose()
            if cand is None:
                continue

            # --- Screen: G1' must itself be reachable from G0 (reject one-hop traps). ---
            screen, _ = _trial(self.harness, [cand], max_steps)
            if not screen.reached_final:
                n_one_hop_skipped += 1
                continue
            n_screened += 1

            # --- Attack: the genuine two-hop trap test, with infeasibility logged. ---
            outcome, infeas = _trial(self.harness, [cand, G1], max_steps)
            score = self._score(outcome, baseline)
            rec = {"candidate": cand, "outcome": outcome, "score": score, "infeasible": infeas}
            results.append(rec)
            hit = outcome.is_attack_success()
            if verbose and (hit or i % progress_every == 0):
                print(f"[{i:04d}] screened={n_screened} G1'={np.round(cand, 3)} -> "
                      f"{outcome.label} (final_d={outcome.final_dist:.3f}, "
                      f"infeasible_QP={infeas}/{outcome.n_steps})"
                      + ("  *** HIT" if hit else ""), flush=True)
            if hit and first_hit is None:
                first_hit = rec
                if stop_on_first_hit:
                    break

        results.sort(key=lambda r: r["score"], reverse=True)
        return {"baseline": baseline, "results": results, "aborted": False,
                "n_one_hop_skipped": n_one_hop_skipped, "baseline_infeasible": base_infeas,
                "n_screened": n_screened, "first_hit": first_hit}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _attack_score(outcome, infeas):
        """Continuous fitness for the targeted search: rewards confirmed attacks,
        and (as guidance toward them) NOT reaching G1, the filter being driven
        toward give-up (infeasible QP), and penetration depth."""
        s = _LABEL_BASE.get(outcome.label, 0.0)
        s += 1.0 * float(outcome.final_dist)                 # further from G1 -> nearer deadlock
        s += 0.01 * float(infeas)                            # more give-up -> nearer collision
        if np.isfinite(outcome.worst_min_dist_env):
            s += 2.0 * max(0.0, -float(outcome.worst_min_dist_env))
        return s

    def run_targeted(self, iterations=10, pop=12, elite=4, init_std=0.06,
                     max_steps=None, verbose=True, stop_on_first_hit=True):
        """Cross-Entropy-Method search over G1'. Instead of sampling uniformly, fit
        a Gaussian to the highest-scoring admissible candidates each round and
        resample around them, steering G1' toward goals that break the filter."""
        G1 = self.scene["G1_base"]
        bounds = self.scene["bounds"]
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])

        baseline, base_infeas = _trial(self.harness, [G1], max_steps)
        if verbose:
            print(f"[baseline] [G1] -> {baseline.label} (reached={baseline.reached_final}, "
                  f"infeasible_QP={base_infeas}/{baseline.n_steps})", flush=True)
        if not baseline.reached_final:
            return {"baseline": baseline, "aborted": True, "abort_reason": "baseline_unreachable",
                    "first_hit": None, "results": [], "baseline_infeasible": base_infeas}

        # init the sampling distribution at a random admissible point
        mean = self._propose()
        if mean is None:
            mean = (lo + hi) / 2.0
        std = (hi - lo) * init_std

        results = []; first_hit = None; n_screened = 0
        for it in range(iterations):
            # draw `pop` admissible candidates from N(mean, std)
            cands = []
            tries = 0
            while len(cands) < pop and tries < pop * 50:
                tries += 1
                c = np.clip(self.rng.normal(mean, std), lo, hi)
                ok, _ = is_admissible(c, bounds, self.scene["base_frame"],
                                      self.scene["obstacles_world"], self.scene["keepout"])
                if ok:
                    cands.append(c)
            scored = []
            for c in cands:
                screen, _ = _trial(self.harness, [c], max_steps)
                # G0 -> c should be reachable, therefore reached_final should be True
                if not screen.reached_final:
                    scored.append((c, -1.0))                 # not individually reachable -> drop
                    continue
                n_screened += 1
                outcome, infeas = _trial(self.harness, [c, G1], max_steps)
                sc = self._attack_score(outcome, infeas)
                scored.append((c, sc))
                rec = {"candidate": c, "outcome": outcome, "score": sc, "infeasible": infeas}
                results.append(rec)
                if outcome.is_attack_success() and first_hit is None:
                    first_hit = rec
                    if verbose:
                        print(f"[iter {it}] *** HIT G1'={np.round(c,3)} -> {outcome.label} "
                              f"(infeasible_QP={infeas}/{outcome.n_steps})", flush=True)
                    if stop_on_first_hit:
                        results.sort(key=lambda r: r["score"], reverse=True)
                        return {"baseline": baseline, "results": results, "aborted": False,
                                "first_hit": first_hit, "n_screened": n_screened,
                                "baseline_infeasible": base_infeas}
            # refit distribution to the elite (highest-scoring) candidates
            scored.sort(key=lambda x: x[1], reverse=True)
            elites = np.array([c for c, sc in scored[:elite] if sc >= 0.0])
            best_sc = scored[0][1] if scored else float("nan")
            if verbose:
                print(f"[iter {it}] best_score={best_sc:.3f} elites={len(elites)} "
                      f"mean={np.round(mean,3)}", flush=True)
            if len(elites) >= 2:
                mean = elites.mean(0)
                std = elites.std(0) + 1e-3            # +floor so it never fully collapses
            # else: keep the current mean/std and try again next round

        results.sort(key=lambda r: r["score"], reverse=True)
        return {"baseline": baseline, "results": results, "aborted": False,
                "first_hit": first_hit, "n_screened": n_screened,
                "baseline_infeasible": base_infeas}
