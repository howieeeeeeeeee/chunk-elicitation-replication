"""Constants and small shared helpers for the analysis pipeline."""

from __future__ import annotations

import pandas as pd

from data_summarizer.visualizations import SHORT_GAME_NAMES
from games.instructions import BEHAVIOR_GAMES

CONTEXT_INCENTIVE_REGRESSOR = "Context-Incentive Cell"
CONTEXT_INCENTIVE_BASELINE = "Not Specified / Not Specified"

# Paper regression regressors. Context and incentive are intentionally combined
# because the observed phase_2_context design is not a complete Context x
# Incentive factorial design.
DEFAULT_REGRESSORS = [
    CONTEXT_INCENTIVE_REGRESSOR,
    "Privacy Treatment",
    "Explain Reasoning",
    "Theoretical Prediction",
    "LLM Model",
    "Mode",
]

GROUND_TRUTH_GAMES = [
    "TicTacToe Logic - L2",
    "TicTacToe Logic",
    "Arithmetic Verification",
    "Trivial Dominance",
]

RANDOM_NUMBER_GAME = ["Random Number Generation"]
TEMPERATURE_EXPERIMENT_FLAG = '["small_experiments_on_temperature"]'

FIGURE_FILES = [
    "table2_raincloud_model_performance.png",
    "task1b_w1_by_game.png",
    "task2_decision_distributions.png",
    "task2b_ground_truth_decision_distributions.png",
    "task1c_atomic_vs_chunk10_scatter.png",
]

# ---------------------------------------------------------------------------
# Figure styling + model/game ordering (ported from analysis.ipynb)
# ---------------------------------------------------------------------------

ATOMIC_COLOR = "#3D405B"
ATOMIC_EDGE = "#2a2c40"
CHUNK_COLOR = "#E07A5F"
CHUNK_EDGE = "#c55a3f"

# Typography aligned with Task 1c (scatter, ``Figure 3`` in ``result.tex``).
_FIG_TITLE_FS = 20
_FIG_SUPTITLE_FS = 22
_FIG_AXIS_LABEL_FS = 17
_FIG_TICK_FS = 14
_FIG_LEGEND_FS = 14
_FIG_ANNOTATION_FS = 13
_FIG_TICK_PARAMS = {"width": 1.1, "length": 5}
# Raincloud: two-line mean (std) per model (ChunkN=10 then Atomic); no in-text prefixes.
_FIG_RAINCLOUD_STAT_FS = _FIG_ANNOTATION_FS * 0.9
_FIG_RAINCLOUD_STAT_LINESPACING = 1.18
_FIG_RAINCLOUD_STAT_X = 0.755
_FIG_RAINCLOUD_XLIM_HI = 0.82
_FIG_RAINCLOUD_Y_TICK_FS = 15
_FIG_RAINCLOUD_LEGEND_FS = 11
# Task 1b + Task 2 (+ Task 2b): panel titles/ticks; no figure suptitle (caption in LaTeX).
_FIG_TASK12_PANEL_TITLE_FS = 22
_FIG_TASK12_TICK_FS = 18
# Grids without suptitle: larger top legend + layout rect.
_FIG_GRID_LEGEND_FS = 22
_FIG_GRID_LEGEND_Y = 0.998
_FIG_GRID_NO_SUPTITLE_RECT = (0, 0, 1, 0.90)

EXCLUDED_MODEL_IDS = {"google/gemini-3-pro-preview", "gemini-3-pro-preview"}

MODEL_NAME_MAP = {
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gpt-5.2": "GPT-5.2",
    "qwen3-235b-a22b-2507": "Qwen 3",
    "grok-4": "Grok 4",
    "grok-4.1-fast": "Grok 4.1 Fast",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek-v3.2": "DeepSeek V3.2",
}
MODEL_LABEL_ORDER = list(MODEL_NAME_MAP.values())

GAME_ORDER_SHORT_GRID = [
    [
        "Stag Hunt",
        "Battle of Sexes",
        "Prisoner's Dilemma",
        "Trust (Trustor)",
        "Trust (Trustee)",
    ],
    [
        "Ultimatum (Prop)",
        "Ultimatum (Resp)",
        "Linear Public Good",
        "Bomb Risk",
        "Dictator",
    ],
]
GAME_SHORT_TO_FULL = {SHORT_GAME_NAMES.get(game, game): game for game in BEHAVIOR_GAMES}
GAME_ORDER = [
    GAME_SHORT_TO_FULL[game_short]
    for row in GAME_ORDER_SHORT_GRID
    for game_short in row
    if game_short in GAME_SHORT_TO_FULL
]


def _short_model_name(model_name) -> str:
    """Map raw provider/model IDs to paper-friendly labels with fallback."""
    if model_name is None:
        return ""
    model_key = str(model_name).split("/")[-1].lower()
    return MODEL_NAME_MAP.get(model_key, str(model_name))


def _exclude_models(df: pd.DataFrame, model_col: str = "LLM Model") -> pd.DataFrame:
    if model_col not in df.columns:
        return df.copy()
    return df.loc[~df[model_col].isin(EXCLUDED_MODEL_IDS)].copy()
