"""Run Experiment 3 and its temperature check into local JSON.

Covers four deterministic tasks and random-number generation across eight
models in phase_2, plus the Atomic temperature=2 cells tagged
small_experiments_on_temperature (165 cells).
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

from simluations.treatment_check import validate_treatment


logger = setup_logger(__name__)

EXPERIMENT_NAME = "exp3"
PHASE_NAME = "phase_2"

GAMES = [
    "TicTacToe Logic - L2",
    "TicTacToe Logic",
    "Arithmetic Verification",
    "Trivial Dominance",
    "Random Number Generation",
]

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
]

TEMP_ROBUSTNESS_COMBOS = [
    (True, "IterativeSimulate", 1, "basic", None),
]

PARAMETER_COMBINATIONS = {
    "with_thinking": make_combos(MAIN_COMBOS, reasoning_values=[True, False]),
    "without_thinking": make_combos(MAIN_COMBOS, reasoning_values=[False]),
}

TEMP_PARAMETER_COMBINATIONS = {
    "with_thinking": make_combos(
        TEMP_ROBUSTNESS_COMBOS, reasoning_values=[True, False]
    ),
    "without_thinking": make_combos(TEMP_ROBUSTNESS_COMBOS, reasoning_values=[False]),
}

LIST_OF_PARAMETERS = [
    "instruction_config.explain_reasoning",
    "instruction_config.explain_reasoning_mode",
    "instruction_config.split_n",
    "simulation_config.simulation_mode",
    "simulation_config.batch_simulation_n",
    "llm_config.reasoning_enabled",
    "llm_config.temperature",
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

TEMP_BASELINE_CONFIG = {
    **BASELINE_CONFIG,
    "llm_config": {
        **BASELINE_CONFIG["llm_config"],
        "temperature": 2,
    },
}


def main() -> None:
    parser = base_arg_parser("Run Experiment 3 into data/exp3 local JSON files.")
    args = parser.parse_args()
    main_jobs, main_counts = build_jobs(
        games=GAMES,
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
    temp_jobs, temp_counts = build_jobs(
        games=["Random Number Generation"],
        models=MODELS,
        parameter_combinations=TEMP_PARAMETER_COMBINATIONS,
        baseline_config=TEMP_BASELINE_CONFIG,
        phase_name=PHASE_NAME,
        list_of_parameters=LIST_OF_PARAMETERS,
        experiment_name=EXPERIMENT_NAME,
        data_root=args.data_root,
        extra_flag=["small_experiments_on_temperature"],
        validate_fn=validate_treatment,
    )
    jobs = main_jobs + temp_jobs
    to_run = main_counts["to_run"] + temp_counts["to_run"]
    logger.info("Main plan: %s", main_counts)
    logger.info("Temperature robustness plan: %s", temp_counts)
    confirm_or_exit(args, to_run)
    run_jobs(
        jobs=jobs,
        data_root=args.data_root,
        max_workers=args.max_workers,
        logger=logger,
    )


if __name__ == "__main__":
    main()
