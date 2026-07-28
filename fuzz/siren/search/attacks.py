"""
What counts as an attack.

SIREN supports two attack types. They are duals — insertion varies the START of
the failing leg (with the real goal pinned), modification varies its ENDPOINT —
and they differ in exactly three places:

                     INSERTION                      MODIFICATION
    what runs        G0 -> G1' -> G1                G0 -> G1'
    legitimacy       G1' must be reachable ALONE    plausible goal (admissible)
    success          the SEQUENCE fails             the direct motion fails

Everything else — the picker, the rollout, the measurements, the scoring, the
loop — is shared. That is why the attack type is a small plug-in and not a fork
of the whole tool.

Insertion's extra gate is what makes its result meaningful: if G1' alone already
traps the robot, the "attack" says nothing about goal SEQUENCING; it is just a
bad goal. Requiring G0->G1' to succeed on its own means any failure of
G0->G1'->G1 is purely an ordering effect.
"""

from abc import ABC, abstractmethod

import numpy as np


class Attack(ABC):
    name: str

    @abstractmethod
    def build_schedule(self, cand, scene) -> list:
        """The waypoint list the robot will be driven through."""

    @abstractmethod
    def gate(self, cand, scene, run_fn) -> tuple:
        """Is this a legitimate candidate? Returns (ok, reason).
        `run_fn(schedule)` is available for gates that need a rollout."""

    def is_success(self, record) -> bool:
        """Only a confirmed unsafe outcome counts; TIMEOUT is inconclusive."""
        return record.is_attack_success()


class InsertionAttack(Attack):
    """Insert one admissible goal BEFORE the legitimate goal."""

    name = "insertion"

    def build_schedule(self, cand, scene) -> list:
        return [np.asarray(cand, dtype=float), np.asarray(scene.G1, dtype=float)]

    def gate(self, cand, scene, run_fn) -> tuple:
        # G1' must itself be safely reachable, else the trap is trivial.
        screen = run_fn([np.asarray(cand, dtype=float)])
        if not screen.reached:
            return False, f"one_hop_unreachable({screen.label})"
        return True, "ok"


class ModificationAttack(Attack):
    """Replace the legitimate goal with a different, still-plausible one."""

    name = "modification"

    def build_schedule(self, cand, scene) -> list:
        return [np.asarray(cand, dtype=float)]

    def gate(self, cand, scene, run_fn) -> tuple:
        # Admissibility (bounds + keepout) is already enforced by the picker;
        # a modified goal needs no reachability screen, since failing to reach
        # it IS the attack.
        return True, "ok"


_ATTACKS = {"insertion": InsertionAttack, "modification": ModificationAttack}


def make_attack(name: str) -> Attack:
    if name not in _ATTACKS:
        raise ValueError(f"unknown attack '{name}'; choose from {sorted(_ATTACKS)}")
    return _ATTACKS[name]()
