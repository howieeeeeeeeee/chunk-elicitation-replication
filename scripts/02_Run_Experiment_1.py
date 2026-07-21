"""Run Experiment 1 into local JSON.

Covers eight behavioral games with GPT-5.2, phase_2_context, an empty
extraFlag, Atomic and ChunkN=10 elicitation, reasoning on/off, seven
context-incentive cells, and three privacy treatments (672 cells).
"""

from __future__ import annotations

from _common import (
    base_arg_parser,
    build_jobs,
    confirm_or_exit,
    make_combos,
    run_jobs,
    setup_logger,
)

from games.treatments import PRIVACY_TREATMENTS
from simluations.treatment_check import validate_treatment


logger = setup_logger(__name__)

EXPERIMENT_NAME = "exp1"
PHASE_NAME = "phase_2_context"
EXTRA_FLAG = []

GAMES = [
    "Dictator",
    "Ultimatum Strategy (Proposer)",
    "Ultimatum Strategy (Responder)",
    "Linear Public Good",
    "Prisoner's Dilemma",
    "Bomb Risk",
    "Trust in CC09 (trustor)",
    "Trust in CC09 (trustee)",
]

MODELS = {
    "with_thinking": ["openai/gpt-5.2"],
}

COMMON_COMBOS = [
    (True, "IterativeSimulate", 1, "basic", None),
    (False, "IterativeSimulate", 1, "basic", None),
    (True, "BatchSimulate", 10, "basic", None),
    (False, "BatchSimulate", 10, "basic", None),
]

PARAMETER_COMBINATIONS = {
    "with_thinking": make_combos(COMMON_COMBOS, reasoning_values=[False]),
}

CONTEXT_INCENTIVE_CELLS = [
    ("Not Specified", "Not Specified"),
    ("Classroom", "Not Specified"),
    ("Classroom", "Standard"),
    ("Classroom", "High"),
    ("Lab", "Not Specified"),
    ("Lab", "Standard"),
    ("Lab", "High"),
]

TREATMENT_OVERRIDES = [
    {
        "instruction_config.context": context,
        "instruction_config.incentive_size": incentive,
        "instruction_config.privacy_treatment": privacy,
    }
    for context, incentive in CONTEXT_INCENTIVE_CELLS
    for privacy in PRIVACY_TREATMENTS.keys()
]

LIST_OF_PARAMETERS = [
    "instruction_config.explain_reasoning",
    "instruction_config.explain_reasoning_mode",
    "instruction_config.split_n",
    "instruction_config.context",
    "instruction_config.incentive_size",
    "instruction_config.privacy_treatment",
    "instruction_config.theoretical_prediction",
    "simulation_config.simulation_mode",
    "simulation_config.batch_simulation_n",
    "llm_config.reasoning_enabled",
    "llm_config.model",
    "simulation_config.game_type",
]

BASELINE_CONFIG = {
    "simulation_config": {
        "simulation_mode": None,
        "target_simulation_n": 100,
        "game_type": None,
        "output_format": "json",
        "batch_mode": "Independent",
        "batch_simulation_n": None,
        "previous_responses": [],
        "save_messages_n_contents": False,
        "iterative_workers": 10,
    },
    "instruction_config": {
        "background": "",
        "personality_traits": "",
        "explain_reasoning": None,
        "explain_reasoning_mode": "basic",
        "split_n": None,
        "theoretical_prediction": False,
        "additional_instructions": [],
        "include_simulation_id": True,
        "context": "Not Specified",
        "incentive_size": "Not Specified",
        "privacy_treatment": "Not Specified",
        "focal_point": False,
    },
    "llm_config": {
        "llm_service": "openrouter",
        "temperature": 1,
        "frequency_penalty": 1,
        "model": None,
        "reasoning_enabled": None,
    },
}


def main() -> None:
    parser = base_arg_parser("Run Experiment 1 into data/exp1 local JSON files.")
    args = parser.parse_args()
    jobs, counts = build_jobs(
        games=GAMES,
        models=MODELS,
        parameter_combinations=PARAMETER_COMBINATIONS,
        baseline_config=BASELINE_CONFIG,
        phase_name=PHASE_NAME,
        list_of_parameters=LIST_OF_PARAMETERS,
        experiment_name=EXPERIMENT_NAME,
        data_root=args.data_root,
        extra_flag=EXTRA_FLAG,
        validate_fn=validate_treatment,
        treatment_overrides=TREATMENT_OVERRIDES,
    )
    logger.info("Plan: %s", counts)
    confirm_or_exit(args, counts["to_run"])
    run_jobs(
        jobs=jobs,
        data_root=args.data_root,
        max_workers=args.max_workers,
        logger=logger,
    )


if __name__ == "__main__":
    main()
