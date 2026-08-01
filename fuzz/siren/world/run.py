"""
THE LOOP — the top level of world/.

This is a coordinator, not an adapter: each step it advances SPARK, reads the
raw numbers off the live filter, turns them into authority and margin, and
writes the result down. It is the only module that knows about all four
collaborators:

    sim/      advance the robot, read raw numbers   (SPARK lives here)
    derived   turn raw numbers into C and g          (pure math)
    measure   the record types                       (pure data)
    types     Scene, FilterSpec                      (the vocabulary)

Everything outside world/ talks to it through two calls:

    world.scene()                     -> Scene
    world.run(schedule, filter_spec)  -> RunRecord
"""

import numpy as np

from . import derived
from .measure import StepMeasurement, RunRecord
from .types import FilterSpec, Scene

#: Calibrated on 77 escape-search labels (see fuzz/siren/braking_test.py): the
#: authority C_phi over-states the deceleration actually available along the
#: closing direction, so the stopping distance is scaled up by 1/0.25 = 4x.
#: FITTED, not derived — chosen from a grid of 10 on those same 77 points, so
#: treat 83% agreement as an optimistic estimate until it is cross-validated.
BRAKE_SCALE = 0.25


class World:
    """A SPARK-backed world the attacker can question."""

    def __init__(self, harness):
        from .sim.probe import install_probe
        self.harness = harness
        self._giveup_counter = install_probe(harness)
        self._scene = None
        self._warned_mismatch = False      # BUG-1 guard fires at most once

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, seed=0, spec: FilterSpec = None,
              test_case="G1FixedBase_D1_AG_SO_v0", max_steps=400, **kw):
        from .sim.harness import Harness
        return cls(Harness.build(seed=seed, spec=spec, test_case=test_case,
                                 max_steps=max_steps, **kw))

    # ------------------------------------------------------------------ #
    def scene(self) -> Scene:
        """The fixed world. Read once and cached — the whole experiment depends
        on it being identical for the baseline and every candidate."""
        if self._scene is None:
            self._scene = self.harness.scene()
        return self._scene

    @property
    def supports_index(self) -> str:
        return self.harness.supports_index

    def accepts(self, spec: FilterSpec) -> bool:
        return self.harness.accepts(spec)

    # ------------------------------------------------------------------ #
    def run(self, schedule, filter_spec: FilterSpec = None,
            max_steps=None, exact_margin=False, keep_q=False) -> RunRecord:
        """Drive the robot through `schedule` under `filter_spec`, recording
        everything. `schedule` is a list of base-frame positions; the last is
        the goal that counts as "done"."""
        from .sim import probe

        h = self.harness
        if filter_spec is not None:
            h.retune(filter_spec)
        spec = h.spec
        max_steps = max_steps if max_steps is not None else h.max_steps

        demand_shape = spec.demand_shape
        eta = spec.eta
        lam = spec.lam

        probe.reset_giveups(h)
        agent_feedback, task_info = h.reset()
        h.env.task.set_goal_schedule(schedule)
        u_safe, action_info = h.algo.act(agent_feedback, task_info)

        steps = []
        prev_giveups = 0
        prev_clearance = None
        # the control is HELD for this long; braking distance is measured in it
        interval = float(h.env.agent.dt * h.env.agent.control_decimation)

        for t in range(max_steps):
            agent_feedback, task_info = h.env.step(u_safe, action_info)
            u_ref_prev = action_info.get("u_ref", None)
            u_safe, action_info = h.algo.act(agent_feedback, task_info)
            task = h.env.task

            # --- what SPARK reports directly ---------------------------- #
            clearance = h.clearance(task_info)
            # closing speed by finite difference: model-free on purpose, so the
            # braking test does not inherit the linearisation that made mu wrong
            v_close = (0.0 if prev_clearance is None
                       else (prev_clearance - clearance) / interval)
            prev_clearance = clearance

            # --- raw ingredients -> authority + margin ------------------ #
            raw = probe.read_raw(h)
            if raw:
                d = derived.evaluate(raw["Lg"], raw["Lf"], raw["phi"],
                                     raw["phi_mask"], raw["u_lim"],
                                     demand_shape=demand_shape,
                                     eta=eta if eta is not None else raw.get("eta"),
                                     lam=lam if lam is not None else raw.get("lam"),
                                     exact=exact_margin)
            else:
                d = {}

            # --- what the filter visibly did ---------------------------- #
            total_giveups = probe.giveups(h)
            gave_up = total_giveups > prev_giveups
            prev_giveups = total_giveups

            viol = action_info.get("violation", None)
            slack = float(np.max(viol)) if (viol is not None and np.size(viol) > 0) else 0.0

            u_ref = action_info.get("u_ref", u_ref_prev)
            if u_ref is not None:
                deviation = float(np.linalg.norm(
                    np.asarray(u_safe).reshape(-1) - np.asarray(u_ref).reshape(-1)))
            else:
                deviation = slack

            # BUG-1 regression guard. `clearance` and `phi` are two views of the
            # same event, so a contact on a guarded pair MUST show up as phi > 0.
            # They silently disagreed for several rounds of conclusions before
            # anything checked. Report rather than raise: a violation means the
            # two views have drifted apart again, which is a measurement fault,
            # not a reason to abort a long search.
            if (clearance < 0.0 and d and np.isfinite(d.get("phi", -np.inf))
                    and d.get("phi", -np.inf) <= 0.0 and not self._warned_mismatch):
                self._warned_mismatch = True
                print(f"[measurement warning] step {t}: clearance="
                      f"{clearance:.5f} < 0 (contact) but phi="
                      f"{d.get('phi'):.5f} <= 0 (safe). The collision check and "
                      f"the safety index are describing different pairs — see "
                      f"BUG-1.", flush=True)

            # --- braking margin (see StepMeasurement.brake_margin) --------- #
            C_brake = d.get("C_phi", np.inf)
            if v_close > 0.0 and np.isfinite(C_brake) and C_brake > 0.0:
                brake_margin = clearance - (v_close * v_close) / (
                    2.0 * BRAKE_SCALE * C_brake)
            else:
                brake_margin = np.inf        # receding, or nothing enforced

            steps.append(StepMeasurement(
                step=t,
                wp_idx=int(getattr(task, "wp_idx", 0)),
                dist_final=float(task.dist_to_final),
                reached_final=bool(task.reached_final),
                clearance=clearance,
                C_d=d.get("C_d", np.inf),
                C_phi=d.get("C_phi", np.inf),
                phi=d.get("phi", -np.inf),
                demand=d.get("demand", np.nan),
                g=d.get("g", np.inf),
                mu=d.get("mu", np.nan),      # only set when exact_margin=True
                brake_margin=brake_margin,
                engaged=bool(d.get("engaged", False)),
                trigger_safe=bool(action_info.get("trigger_safe", False)),
                deviation=deviation,
                gave_up=gave_up,
                predicted_infeasible=bool(d.get("predicted_infeasible", False)),
                slack=slack,
                q=(np.asarray(agent_feedback.get("robot_state")).copy()
                   if keep_q and "robot_state" in agent_feedback else None),
            ))

            if task.reached_final or clearance < 0.0:
                break

        return RunRecord.summarize(steps, schedule, spec)

    # ------------------------------------------------------------------ #
    def run_nominal(self, schedule, max_steps=None) -> RunRecord:
        """Cheap prefilter: the same schedule with the filter's demand set to
        ~zero, i.e. essentially the unfiltered goal-seeking drive.

        Used only to RANK a large candidate pool before spending real rollouts on
        the shortlist. It must never be the final judge: the reference controller
        is obstacle-blind, so nominal paths plough straight through obstacles and
        "penetration along the nominal path" stops discriminating.
        """
        from dataclasses import replace
        spec = replace(self.harness.spec, eta=1e-6, lam=1e-6, label="nominal")
        return self.run(schedule, spec, max_steps=max_steps)
