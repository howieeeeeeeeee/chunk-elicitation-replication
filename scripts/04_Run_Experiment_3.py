"""Run formal Experiment 3 and temperature robustness into local JSON.

Scope: four deterministic tasks and random-number generation, eight formal
model IDs, ``phase_2``, Atomic/ChunkN=10, plus Atomic temperature=2 cells tagged
``small_experiments_on_temperature``. All 15 attempted temperature cells stay
in the plan even when a model has no completed historical result.
"""

from __future__ import annotations

from _common import main_for_experiment
from experiment_specs import EXPERIMENT_3


if __name__ == "__main__":
    raise SystemExit(main_for_experiment(EXPERIMENT_3))
