"""Run Experiment 2 and its reasoning-removal check into local JSON.

Covers ten behavioral games, eight models, phase_2, Atomic and ChunkN
10/20/25/50/100, plus the small_experiments_on_removing_reasoning tag
(1,320 cells).
"""

from __future__ import annotations

from _common import (
    FORMAL_SIGNATURE_FIELDS,
    base_arg_parser,
    build_jobs,
    confirm_or_exit,
    make_combos,
    run_jobs,
    setup_logger,
)

from games.instructions import BEHAVIOR_GAMES
from simluations.treatment_check import validate_treatment


logger = setup_logger(__name__)

EXPERIMENT_NAME = "exp2"
PHASE_NAME = "phase_2"

MODELS = {
    "with_thinking": [
        "openai/gpt-5.2",
        "anthropic/claude-sonnet-4.5",
        "x-ai/grok-4",
        "google/gemini-3-flash-preview",
        "x-ai/grok-4.1-fast",
        "deepseek/deepseek-v3.2",
        "google/gemini-3.1-pro-preview",
    ],
    "without_thinking": ["qwen/qwen3-235b-a22b-2507"],
}

MAIN_COMBOS = [
    (True, "IterativeSimulate", 1, "basic", None),
    (True, "BatchSimulate", 10, "basic", None),
    (True, "BatchSimulate", 20, "basic", None),
    (True, "BatchSimulate", 25, "basic", None),
    (False, "BatchSimulate", 50, "basic", None),
    (False, "BatchSimulate", 100, "basic", None),
    (True, "BatchSimulate", 50, "split", 25),
    (True, "BatchSimulate", 100, "split", 25),
]

REASONING_OFF_COMBOS = [
    (False, "IterativeSimulate", 1, "basic", None),
    (False, "BatchSimulate", 10, "basic", None),
    (False, "BatchSimulate", 20, "basic", None),
    (False, "BatchSimulate", 25, "basic", None),
]

PARAMETER_COMBINATIONS = {
    "with_thinking": make_combos(MAIN_COMBOS, reasoning_values=[True, False]),
    "without_thinking": make_combos(MAIN_COMBOS, reasoning_values=[False]),
}

REASONING_OFF_MODELS = {
    "with_thinking": ["google/gemini-3-flash-preview"],
    "without_thinking": ["qwen/qwen3-235b-a22b-2507"],
}

REASONING_OFF_PARAMETER_COMBINATIONS = {
    "with_thinking": make_combos(
        REASONING_OFF_COMBOS, reasoning_values=[True, False]
    ),
    "without_thinking": make_combos(REASONING_OFF_COMBOS, reasoning_values=[False]),
}

LIST_OF_PARAMETERS = list(FORMAL_SIGNATURE_FIELDS)

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
    parser = base_arg_parser("Run Experiment 2 into data/exp2 local JSON files.")
    args = parser.parse_args()
    main_jobs, main_counts = build_jobs(
        games=BEHAVIOR_GAMES,
        models=MODELS,
        parameter_combinations=PARAMETER_COMBINATIONS,
        baseline_config=BASELINE_CONFIG,
        phase_name=PHASE_NAME,
        list_of_parameters=LIST_OF_PARAMETERS,
        experiment_name=EXPERIMENT_NAME,
        data_root=args.data_root,
        extra_flag=[],
        validate_fn=validate_treatment,
    )
    off_jobs, off_counts = build_jobs(
        games=BEHAVIOR_GAMES,
        models=REASONING_OFF_MODELS,
        parameter_combinations=REASONING_OFF_PARAMETER_COMBINATIONS,
        baseline_config=BASELINE_CONFIG,
        phase_name=PHASE_NAME,
        list_of_parameters=LIST_OF_PARAMETERS,
        experiment_name=EXPERIMENT_NAME,
        data_root=args.data_root,
        extra_flag=["small_experiments_on_removing_reasoning"],
        validate_fn=validate_treatment,
    )
    jobs = main_jobs + off_jobs
    to_run = main_counts["to_run"] + off_counts["to_run"]
    logger.info("Main plan: %s", main_counts)
    logger.info("Reasoning-off robustness plan: %s", off_counts)
    confirm_or_exit(args, to_run)
    run_jobs(
        jobs=jobs,
        data_root=args.data_root,
        max_workers=args.max_workers,
        logger=logger,
    )


if __name__ == "__main__":
    main()
