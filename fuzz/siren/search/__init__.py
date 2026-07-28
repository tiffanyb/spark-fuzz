"""
search/ — the attacker. What to try, and how good was it.

    attacks.py  what counts as an attack (insertion vs modification)
    pick.py     what to try next (admissibility + random / CEM)
    threat.py   what the attacker may know, see, and want
    loop.py     run the search, rank the results
    cli.py      entry point

Imports world/ only through its contract (Scene, FilterSpec, RunRecord and the
two World calls). It must never import spark_* or mujoco — if it needs
something from the simulator, that capability belongs behind World instead.
"""

from .attacks import Attack, InsertionAttack, ModificationAttack, make_attack
from .threat import AttackerCapability, ThreatModel, threat_model, PRESETS
from .pick import RandomPicker, CEMPicker, make_picker
from .loop import search, SearchReport

__all__ = [
    "Attack", "InsertionAttack", "ModificationAttack", "make_attack",
    "AttackerCapability", "ThreatModel", "threat_model", "PRESETS",
    "RandomPicker", "CEMPicker", "make_picker",
    "search", "SearchReport",
]
