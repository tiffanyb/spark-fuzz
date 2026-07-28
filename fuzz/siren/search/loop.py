"""
The search: evaluate candidates against the ensemble, manage the budget, rank.

Per candidate:
    1. admissible?                    — a goal an operator could plausibly issue
    2. attack.gate                    — insertion also demands G1' reach alone
    3. for each surrogate filter:     — roll out, observe, view, score
    4. aggregate WORST-CASE           — a candidate rates highly only if it
                                        routes into the low-authority basin
                                        under EVERY plausible filter, so its
                                        success does not hinge on guessing right
    5. feed the score back            — CEM refits; random ignores it

Results are returned RANKED rather than as a single hit, so a later sim-to-real
transfer layer has an ordered list to work through.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .pick import is_admissible
from .threat import KnowledgeError, ThreatModel


@dataclass
class CandidateResult:
    candidate: np.ndarray
    records: List = field(default_factory=list)     # one per surrogate
    member_scores: List[float] = field(default_factory=list)
    score: float = -np.inf
    success: bool = False
    skipped: Optional[str] = None

    @property
    def labels(self):
        return [r.label for r in self.records]


@dataclass
class SearchReport:
    threat: str
    attack: str
    picker: str
    scene: object
    baseline: object = None
    results: List[CandidateResult] = field(default_factory=list)
    n_proposed: int = 0
    n_inadmissible: int = 0
    n_gated_out: int = 0
    n_screened: int = 0
    n_success: int = 0
    n_specs_skipped: int = 0
    elapsed_s: float = 0.0
    aborted: Optional[str] = None

    def ranked(self, k=None):
        rs = sorted([r for r in self.results if r.skipped is None],
                    key=lambda r: r.score, reverse=True)
        return rs[:k] if k else rs

    def summary(self) -> str:
        lines = [
            f"threat={self.threat}  attack={self.attack}  picker={self.picker}",
            f"proposed={self.n_proposed}  inadmissible={self.n_inadmissible}  "
            f"gated_out={self.n_gated_out}  screened={self.n_screened}",
            f"SUCCESS={self.n_success}  elapsed={self.elapsed_s:.1f}s",
        ]
        if self.aborted:
            lines.append(f"ABORTED: {self.aborted}")
        if self.n_specs_skipped:
            lines.append(f"note: {self.n_specs_skipped} surrogate rollouts skipped "
                         f"(index unsupported by this world)")
        top = self.ranked(3)
        for i, r in enumerate(top):
            lines.append(f"  #{i+1} score={r.score:.3f} labels={r.labels} "
                         f"G1'={np.round(r.candidate, 3)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- #
def evaluate_candidate(cand, world, scene, attack, threat: ThreatModel,
                       max_steps=None) -> CandidateResult:
    """Roll one candidate out under every surrogate and aggregate worst-case."""
    res = CandidateResult(candidate=np.asarray(cand, dtype=float))
    schedule = attack.build_schedule(cand, scene)

    for spec in threat.specs:
        # No silent truncation: a world built for velocity control cannot host a
        # distance index and vice versa, so incompatible members are reported.
        if not world.accepts(spec):
            res.skipped = (res.skipped or "") + f"[{spec.label}:index] "
            continue
        record = world.run(schedule, spec, max_steps=max_steps)
        record = threat.observability.observe(record)
        try:
            s = threat.objective.score(record)
        except KnowledgeError as e:
            raise RuntimeError(f"threat '{threat.name}' violated its own "
                               f"knowledge budget: {e}") from e
        res.records.append(record)
        res.member_scores.append(s)

    if not res.member_scores:
        res.skipped = res.skipped or "no_compatible_surrogate"
        return res

    res.score = float(threat.aggregate(res.member_scores))
    res.success = any(attack.is_success(r) for r in res.records)
    res.skipped = None
    return res


def search(world, attack, picker, threat: ThreatModel,
           budget: int = 100, batch: int = 12, max_steps=None,
           verbose: bool = True, stop_on_first: bool = False) -> SearchReport:
    """Run the search. `budget` counts candidates actually screened."""
    t0 = time.time()
    scene = world.scene()
    report = SearchReport(threat=threat.name, attack=attack.name,
                          picker=picker.name, scene=scene)

    # --- baseline gate: the legitimate goal must be reachable ------------ #
    #
    # The gate MUST run. Picking threat.specs[0] blindly can select a surrogate
    # this world cannot host (e.g. a distance-index member against an
    # acceleration-controlled world), which would skip the gate silently and let
    # an already-broken scene report a full set of "attacks". So pick the first
    # spec the world actually accepts, and refuse to search if there is none.
    compatible = [s for s in threat.specs if world.accepts(s)]
    if not compatible:
        report.aborted = (f"no_compatible_surrogate "
                          f"(world runs '{world.supports_index}' index; "
                          f"threat offers "
                          f"{sorted({s.index for s in threat.specs})})")
        report.elapsed_s = time.time() - t0
        return report

    primary = compatible[0]
    baseline = world.run([scene.G1], primary, max_steps=max_steps)
    report.baseline = baseline
    if verbose:
        print(f"[baseline] G0->G1 {baseline.brief()}", flush=True)
    if not baseline.reached:
        report.aborted = f"baseline_unreachable({baseline.label})"
        report.elapsed_s = time.time() - t0
        return report
    # A scene where the filter is already giving up is INERT, not defeated: the
    # obstacle-blind reference controller is driving. Attacks there say nothing
    # about the filter, so exclude the scene rather than bank a false positive.
    if baseline.n_gave_up > 0:
        report.aborted = (f"baseline_filter_inert"
                          f"({baseline.n_gave_up} give-ups of {baseline.n_steps})")
        report.elapsed_s = time.time() - t0
        return report

    def run_fn(sched):
        return world.run(sched, primary, max_steps=max_steps)

    # --- the loop -------------------------------------------------------- #
    while report.n_screened < budget:
        want = min(batch, budget - report.n_screened)
        cands = picker.ask(want)
        if not cands:
            break
        report.n_proposed += len(cands)

        scored = []
        for cand in cands:
            if not is_admissible(cand, scene)[0]:
                report.n_inadmissible += 1
                continue
            ok, reason = attack.gate(cand, scene, run_fn)
            if not ok:
                report.n_gated_out += 1
                continue

            res = evaluate_candidate(cand, world, scene, attack, threat,
                                     max_steps=max_steps)
            if res.skipped and not res.records:
                report.n_specs_skipped += 1
                continue

            report.n_screened += 1
            report.results.append(res)
            scored.append((cand, res.score))
            if res.success:
                report.n_success += 1
                if verbose:
                    print(f"  *** HIT  score={res.score:.3f} labels={res.labels} "
                          f"G1'={np.round(cand, 3)}", flush=True)
                if stop_on_first:
                    picker.tell(scored)
                    report.elapsed_s = time.time() - t0
                    return report

        picker.tell(scored)
        if verbose:
            print(f"  screened={report.n_screened}/{budget} "
                  f"success={report.n_success}", flush=True)

    report.elapsed_s = time.time() - t0
    return report
