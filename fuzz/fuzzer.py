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

import numpy as np

from .admissibility import sample_admissible
from .metrics import TrialOutcome


# Higher score = stronger attack. DEADLOCK (safe but trapped) is the headline
# result; COLLISION is also a failure; TIMEOUT is weak; REACHED is no attack.
_LABEL_BASE = {"DEADLOCK": 3.0, "COLLISION": 2.0, "TIMEOUT": 1.0, "REACHED": 0.0}


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
    def run(self, n_candidates=50, max_steps=None, verbose=True):
        G1 = self.scene["G1_base"]

        # --- Hard gate: G1 must be reachable from G0 for this scene/controller. ---
        baseline = self.harness.run_trial([G1], max_steps=max_steps)
        if verbose:
            print(f"[baseline] schedule=[G1] -> {baseline.label} "
                  f"(reached={baseline.reached_final}, steps={baseline.n_steps})")
        if not baseline.reached_final:
            if verbose:
                print("  ABORT: baseline G0->G1 not reached; scene is invalid for "
                      "this controller -- not fuzzing. Pick another seed / sparser "
                      "scene, or constrain G1 sampling (see fuzz/TODO.md).")
            return {"baseline": baseline, "results": [], "aborted": True,
                    "abort_reason": "baseline_unreachable", "n_one_hop_skipped": 0}

        results = []
        n_one_hop_skipped = 0
        for i in range(n_candidates):
            cand = self._propose()
            if cand is None:
                if verbose:
                    print(f"[{i:03d}] no admissible candidate found (rejection limit)")
                continue

            # --- Screen: reject trivial one-hop traps. G1' alone must be safely ---
            # --- reachable from G0; otherwise the failure is about G1', not the ---
            # --- *sequence* G0->G1'->G1.                                         ---
            screen = self.harness.run_trial([cand], max_steps=max_steps)
            if not screen.reached_final:
                n_one_hop_skipped += 1
                if verbose:
                    print(f"[{i:03d}] G1'={np.round(cand, 3)} skipped: one-hop trap "
                          f"([G1'] -> {screen.label}); not a sequence attack")
                continue

            # --- Attack: genuine two-hop trap test. ---
            outcome = self.harness.run_trial([cand, G1], max_steps=max_steps)
            score = self._score(outcome, baseline)
            results.append({"candidate": cand, "outcome": outcome, "score": score})
            if verbose:
                print(f"[{i:03d}] G1'={np.round(cand, 3)} -> {outcome.label} "
                      f"(reached={outcome.reached_final}, steps={outcome.n_steps}, "
                      f"peak_slack={outcome.peak_slack:.3g}, score={score:.3f})")

        results.sort(key=lambda r: r["score"], reverse=True)
        return {"baseline": baseline, "results": results, "aborted": False,
                "n_one_hop_skipped": n_one_hop_skipped}
