"""Run formal Experiment 2 and its reasoning robustness into local JSON.

Scope: ten behavioral games, eight formal model IDs, ``phase_2``, Atomic and
ChunkN 10/20/25/50/100, plus the
``small_experiments_on_removing_reasoning`` cells. Gemini 3 Flash keeps both
reasoning-enabled values from the historical run; Qwen uses false only.
"""

from __future__ import annotations

from _common import main_for_experiment
from experiment_specs import EXPERIMENT_2


if __name__ == "__main__":
    raise SystemExit(main_for_experiment(EXPERIMENT_2))
