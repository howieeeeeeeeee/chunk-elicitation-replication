"""Run the formal Experiment 1 factorial design into local JSON.

Scope: eight behavioral games, GPT-5.2, ``phase_2_context``, empty
``extraFlag``, Atomic/ChunkN=10, explicit reasoning on/off, seven valid
context-incentive cells, and three privacy cells. Planned cells are distinct
from successfully completed cells reported by the preflight plan.
"""

from __future__ import annotations

from _common import main_for_experiment
from experiment_specs import EXPERIMENT_1


if __name__ == "__main__":
    raise SystemExit(main_for_experiment(EXPERIMENT_1))
