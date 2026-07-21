"""Plain configuration dictionaries for the three formal experiments."""

from __future__ import annotations

from games.instructions import BEHAVIOR_GAMES


COORDINATION_GAMES = [
    "BoS in CDJFR89",
    "Stag Hunt in CDFR92",
]
EXPERIMENT_1_GAMES = [
    game for game in BEHAVIOR_GAMES if game not in COORDINATION_GAMES
]
EXPERIMENT_2_GAMES = list(BEHAVIOR_GAMES)

GROUND_TRUTH_GAMES = [
    "TicTacToe Logic - L2",
    "TicTacToe Logic",
    "Arithmetic Verification",
    "Trivial Dominance",
]
RANDOM_NUMBER_GAME = "Random Number Generation"
EXPERIMENT_3_GAMES = [*GROUND_TRUTH_GAMES, RANDOM_NUMBER_GAME]

THINKING_MODEL_IDS = [
    "openai/gpt-5.2",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
    "google/gemini-3-flash-preview",
    "x-ai/grok-4.1-fast",
    "deepseek/deepseek-v3.2",
    "google/gemini-3.1-pro-preview",
]
QWEN_MODEL_ID = "qwen/qwen3-235b-a22b-2507"
FORMAL_MODEL_IDS = [*THINKING_MODEL_IDS, QWEN_MODEL_ID]

FULL_MODELS = {
    **{model_id: [True, False] for model_id in THINKING_MODEL_IDS},
    QWEN_MODEL_ID: [False],
}

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

# explain_reasoning, simulation_mode, batch_simulation_n,
# explain_reasoning_mode, split_n
EXPERIMENT_1_COMBOS = [
    (True, "IterativeSimulate", 1, "basic", None),
    (False, "IterativeSimulate", 1, "basic", None),
    (True, "BatchSimulate", 10, "basic", None),
    (False, "BatchSimulate", 10, "basic", None),
]

EXPERIMENT_2_MAIN_COMBOS = [
    (True, "IterativeSimulate", 1, "basic", None),
    (True, "BatchSimulate", 10, "basic", None),
    (True, "BatchSimulate", 20, "basic", None),
    (True, "BatchSimulate", 25, "basic", None),
    (False, "BatchSimulate", 50, "basic", None),
    (False, "BatchSimulate", 100, "basic", None),
    (True, "BatchSimulate", 50, "split", 25),
    (True, "BatchSimulate", 100, "split", 25),
]

REASONING_REMOVAL_COMBOS = [
    (False, "IterativeSimulate", 1, "basic", None),
    (False, "BatchSimulate", 10, "basic", None),
    (False, "BatchSimulate", 20, "basic", None),
    (False, "BatchSimulate", 25, "basic", None),
]

EXPERIMENT_3_MAIN_COMBOS = [
    (True, "IterativeSimulate", 1, "basic", None),
    (True, "BatchSimulate", 10, "basic", None),
]

TEMPERATURE_COMBOS = [(True, "IterativeSimulate", 1, "basic", None)]

CONTEXT_INCENTIVE_CELLS = [
    ("Not Specified", "Not Specified"),
    ("Classroom", "Not Specified"),
    ("Classroom", "Standard"),
    ("Classroom", "High"),
    ("Lab", "Not Specified"),
    ("Lab", "Standard"),
    ("Lab", "High"),
]

EXPERIMENT_1_TREATMENTS = [
    {
        "instruction_config.context": context,
        "instruction_config.incentive_size": incentive,
        "instruction_config.privacy_treatment": privacy,
    }
    for context, incentive in CONTEXT_INCENTIVE_CELLS
    for privacy in ["Not Specified", "Public", "Private"]
]

EXPERIMENT_1 = {
    "key": "exp1",
    "title": "Experiment 1: factorial prompt treatments",
    "paper_scope": (
        "Eight behavioral games, GPT-5.2, Atomic versus ChunkN=10, reasoning "
        "on/off, seven context-incentive cells, and three privacy cells."
    ),
    "expected_cells": 672,
    "sections": [
        {
            "name": "main",
            "phase_name": "phase_2_context",
            "extra_flag": [],
            "games": EXPERIMENT_1_GAMES,
            "models": {"openai/gpt-5.2": [False]},
            "combos": EXPERIMENT_1_COMBOS,
            "treatments": EXPERIMENT_1_TREATMENTS,
            "config_overrides": {},
        }
    ],
}

EXPERIMENT_2 = {
    "key": "exp2",
    "title": "Experiment 2: chunk size, models, and split reasoning",
    "paper_scope": (
        "Ten behavioral games, eight models, ChunkN 1/10/20/25/50/100, and "
        "the reasoning-removal robustness check."
    ),
    "expected_cells": 1320,
    "sections": [
        {
            "name": "main",
            "phase_name": "phase_2",
            "extra_flag": [],
            "games": EXPERIMENT_2_GAMES,
            "models": FULL_MODELS,
            "combos": EXPERIMENT_2_MAIN_COMBOS,
            "treatments": [{}],
            "config_overrides": {},
        },
        {
            "name": "reasoning_removal",
            "phase_name": "phase_2",
            "extra_flag": ["small_experiments_on_removing_reasoning"],
            "games": EXPERIMENT_2_GAMES,
            "models": {
                "google/gemini-3-flash-preview": [True, False],
                QWEN_MODEL_ID: [False],
            },
            "combos": REASONING_REMOVAL_COMBOS,
            "treatments": [{}],
            "config_overrides": {},
        },
    ],
}

EXPERIMENT_3 = {
    "key": "exp3",
    "title": "Experiment 3: ground-truth and random-number tasks",
    "paper_scope": (
        "Four deterministic tasks and random-number generation across eight "
        "models, plus an Atomic temperature=2 robustness check."
    ),
    "expected_cells": 165,
    "sections": [
        {
            "name": "main",
            "phase_name": "phase_2",
            "extra_flag": [],
            "games": EXPERIMENT_3_GAMES,
            "models": FULL_MODELS,
            "combos": EXPERIMENT_3_MAIN_COMBOS,
            "treatments": [{}],
            "config_overrides": {},
        },
        {
            "name": "temperature",
            "phase_name": "phase_2",
            "extra_flag": ["small_experiments_on_temperature"],
            "games": [RANDOM_NUMBER_GAME],
            "models": FULL_MODELS,
            "combos": TEMPERATURE_COMBOS,
            "treatments": [{}],
            "config_overrides": {"llm_config.temperature": 2},
        },
    ],
}

EXPERIMENTS = {
    "exp1": EXPERIMENT_1,
    "exp2": EXPERIMENT_2,
    "exp3": EXPERIMENT_3,
}


def get_experiment(key: str) -> dict:
    if key not in EXPERIMENTS:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiment '{key}'. Choose one of: {choices}")
    return EXPERIMENTS[key]
