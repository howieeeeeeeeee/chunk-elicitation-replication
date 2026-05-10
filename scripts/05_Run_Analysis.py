"""Replication analysis pipeline: rebuild every LaTeX artifact under ``tex/``.

Run via:

    uv run python scripts/05_Run_Analysis.py

This script reads local JSON exports from ``data/exp1``, ``data/exp2``,
``data/exp3``, and ``data/benchmark``. It:

1. Loads ``phase_2`` and ``phase_2_context`` simulations from local JSON,
2. Writes every ``tex/tables/*.tex`` (regression + summaries),
3. Generates every ``tex/figs/*.png`` directly from those DataFrames
   (mirroring a copy into ``output/figures/`` for inspection),
4. Overwrites ``tex/result.tex`` with an ``article`` skeleton that
   ``\\input``s each table and ``\\includegraphics``es each figure.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPLICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analysis.regression import (
    _prepare_regression_frame,
    _build_formula,
    _is_categorical_column,
)
from data_summarizer import (
    show_all_simulations_df,
    attach_ks_test_results_to_simulations_df,
    create_grouped_summary,
)
from data_summarizer.simulations import process_decisions
from data_summarizer.visualizations import SHORT_GAME_NAMES
from db_ops.config import get_combined_database
from db_ops.retrievers import get_all_simulation_results, get_benchmark_results
from games.instructions import BEHAVIOR_GAMES, GAME_DECISION_ARRAY_CONFIG


# ---------------------------------------------------------------------------
# Paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = REPLICATION_ROOT
DATA_DIR = REPO_ROOT / "data"
TEX_DIR = REPO_ROOT / "tex"
TABLES_DIR = TEX_DIR / "tables"
FIGS_DIR = TEX_DIR / "figs"
SRC_FIGS_DIR = REPO_ROOT / "output" / "figures"

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


def _save_figure(fig, filename: str, *, dpi: int = 300) -> Path:
    """Save ``fig`` as ``output/figures/<filename>`` with tight bbox."""
    SRC_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SRC_FIGS_DIR / filename
    fig.patch.set_facecolor("white")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value) -> str:
    s = "" if value is None else str(value)
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in s)


def _format_number(x, digits: int = 3) -> str:
    if x is None:
        return "--"
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "--"
    except TypeError:
        pass
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{int(x):,}"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return _latex_escape(x)


def _sig_marker(p_value: float) -> str:
    if p_value is None or np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Regression label shortening + reference-group overrides
# ---------------------------------------------------------------------------

# Categorical regressors that should use their no-prompt level as the reference
# whenever that level is present in the data. Matched against the *display* name.
_NOT_SPECIFIED_REF_DISPLAYS = {
    "Context",
    "Incentive Size",
    CONTEXT_INCENTIVE_REGRESSOR,
    "Privacy Treatment",
}
_REFERENCE_LEVELS_BY_DISPLAY = {
    CONTEXT_INCENTIVE_REGRESSOR: CONTEXT_INCENTIVE_BASELINE,
}

# Compact labels for the regression table.  Keys are (display name, level);
# value is the string shown in the leftmost column.  Anything not in this
# table falls back to "<short-var> = <level>".
_LEVEL_SHORT: dict[tuple[str, str], str] = {
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Not Specified"): "Lab",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Standard"): "Lab x Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / High"): "Lab x High",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Not Specified"): "Classroom",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Standard"): "Classroom x Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / High"): "Classroom x High",
    ("Context", "Lab"): "Lab",
    ("Context", "Classroom"): "Classroom",
    ("Incentive Size", "Standard"): "Std Pay",
    ("Incentive Size", "High"): "High Pay",
    ("Privacy Treatment", "Private"): "Private",
    ("Privacy Treatment", "Public"): "Public",
    ("Mode", "Chunk"): "Chunk",
    ("Mode", "Atomic"): "Atomic",
    ("Explain Reasoning", "True"): "Reasoning",
    ("Explain Reasoning", "False"): "No Reasoning",
    ("Theoretical Prediction", "True"): "ThePred",
    ("Theoretical Prediction", "False"): "No ThePred",
}

_LEVEL_LATEX: dict[tuple[str, str], str] = {
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Not Specified"): "Lab",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / Standard"): r"Lab $\times$ Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Lab / High"): r"Lab $\times$ High",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Not Specified"): "Classroom",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / Standard"): r"Classroom $\times$ Low",
    (CONTEXT_INCENTIVE_REGRESSOR, "Classroom / High"): r"Classroom $\times$ High",
}

_VAR_SHORT: dict[str, str] = {
    CONTEXT_INCENTIVE_REGRESSOR: "Prompt",
    "Explain Reasoning": "Reasoning",
    "Theoretical Prediction": "ThePred",
    "Incentive Size": "Inc",
    "Privacy Treatment": "Priv",
    "LLM Model": "LLM",
}


def _short_label(display_name: str, level: str | None) -> str:
    if level is None:
        return _VAR_SHORT.get(display_name, display_name)
    key = (display_name, level)
    if key in _LEVEL_SHORT:
        return _LEVEL_SHORT[key]
    return f"{_VAR_SHORT.get(display_name, display_name)} = {level}"


def _short_label_latex(display_name: str, level: str | None) -> str:
    if level is not None:
        key = (display_name, level)
        if key in _LEVEL_LATEX:
            return _LEVEL_LATEX[key]
    return _latex_escape(_short_label(display_name, level))


def _clean_prompt_factor(value) -> str:
    if value is None or pd.isna(value):
        return "Not Specified"
    text = str(value).strip()
    return text if text else "Not Specified"


def _add_context_incentive_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Add the observed Context x Incentive prompt-condition cell."""
    required = ["Context", "Incentive Size"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot build '{CONTEXT_INCENTIVE_REGRESSOR}'; missing columns: {missing}"
        )
    out = df.copy()
    context = out["Context"].map(_clean_prompt_factor)
    incentive = out["Incentive Size"].map(_clean_prompt_factor)
    out[CONTEXT_INCENTIVE_REGRESSOR] = context + " / " + incentive
    return out


def _write_tabular(
    path: Path,
    header: list[str],
    rows: list[list[str]],
    *,
    col_spec: str,
    pre_header: str | None = None,
    midrule_after_header: bool = True,
) -> None:
    """Write a bare tabular environment to ``path``.

    ``header`` and each row in ``rows`` must already be LaTeX-safe strings.
    """
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"]
    if pre_header:
        lines.append(pre_header)
    lines.append(" & ".join(header) + r" \\")
    if midrule_after_header:
        lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Section 1: Regression (phase_2_context, OLS + Fractional Logit, W1)
# ---------------------------------------------------------------------------


def _build_formula_with_refs(
    dep_id: str,
    regressor_ids: list[str],
    work: pd.DataFrame,
    ident_to_display: dict[str, str],
) -> str:
    """Build a statsmodels formula with no-prompt reference levels."""
    if not regressor_ids:
        return f"{dep_id} ~ 1"
    parts: list[str] = []
    for rid in regressor_ids:
        series = work[rid]
        if _is_categorical_column(series):
            display = ident_to_display.get(rid, rid)
            levels = {str(v) for v in series.dropna().unique()}
            ref_level = _REFERENCE_LEVELS_BY_DISPLAY.get(display, "Not Specified")
            if display in _NOT_SPECIFIED_REF_DISPLAYS and ref_level in levels:
                parts.append(f"C({rid}, Treatment(reference='{ref_level}'))")
            else:
                parts.append(f"C({rid})")
        else:
            parts.append(rid)
    return f"{dep_id} ~ " + " + ".join(parts)


def _fit_ols(
    work: pd.DataFrame,
    dep_id: str,
    regressor_ids: list[str],
    ident_to_display: dict[str, str],
):
    formula = _build_formula_with_refs(dep_id, regressor_ids, work, ident_to_display)
    return smf.ols(formula=formula, data=work).fit(), formula


def _fit_fractional_logit(
    work: pd.DataFrame,
    dep_id: str,
    regressor_ids: list[str],
    ident_to_display: dict[str, str],
):
    formula = _build_formula_with_refs(dep_id, regressor_ids, work, ident_to_display)
    model = smf.glm(formula=formula, data=work, family=sm.families.Binomial())
    return model.fit(cov_type="HC1"), formula


# Matches both ``C(name)[T.level]`` and
# ``C(name, Treatment(reference='X'))[T.level]`` forms that statsmodels emits.
# ``.*`` is greedy so the trailing ``)`` anchor lands on the outermost paren
# of ``C(...)`` even when ``Treatment(...)`` introduces nested parens.
_TERM_RE = re.compile(
    r"^C\((?P<name>[A-Za-z_][A-Za-z_0-9]*)(?:\s*,.*)?\)\[T\.(?P<level>.+?)\]$"
)


def _prettify_term(term: str, ident_to_display: dict[str, str]) -> str:
    """Turn a statsmodels term name into a compact, human-readable label.

    ``ident_to_display`` maps normalized identifiers back to their original
    display column names (e.g. ``"context"`` -> ``"Context"``).
    """

    def _display(name: str) -> str:
        return ident_to_display.get(name, name)

    if term == "Intercept":
        return "Intercept"

    m = _TERM_RE.match(term)
    if m:
        return _short_label(_display(m.group("name")), m.group("level"))

    # Fallback: strip a trailing ``[T.level]`` off a plain identifier.
    m = re.match(r"^(?P<name>.+?)\[T\.(?P<level>.+)\]$", term)
    if m:
        return _short_label(_display(m.group("name")), m.group("level"))

    return _short_label(_display(term), None)


def _prettify_term_latex(term: str, ident_to_display: dict[str, str]) -> str:
    """Return a LaTeX-safe display label for a statsmodels term."""

    def _display(name: str) -> str:
        return ident_to_display.get(name, name)

    if term == "Intercept":
        return "Intercept"

    m = _TERM_RE.match(term)
    if m:
        return _short_label_latex(_display(m.group("name")), m.group("level"))

    m = re.match(r"^(?P<name>.+?)\[T\.(?P<level>.+)\]$", term)
    if m:
        return _short_label_latex(_display(m.group("name")), m.group("level"))

    return _short_label_latex(_display(term), None)


def _drop_zero_variance(work: pd.DataFrame, regressor_ids: list[str]) -> list[str]:
    usable = []
    for rid in regressor_ids:
        if work[rid].nunique(dropna=True) >= 2:
            usable.append(rid)
    return usable


def _build_regression_table(
    df: pd.DataFrame,
    dependent_var: str,
    regressors: list[str],
) -> tuple[list[str], list[list[str]]]:
    if (
        CONTEXT_INCENTIVE_REGRESSOR in regressors
        and CONTEXT_INCENTIVE_REGRESSOR not in df.columns
    ):
        df = _add_context_incentive_cell(df)

    work, dep_id, regressor_ids, name_map = _prepare_regression_frame(
        df, dependent_var, regressors
    )
    regressor_ids = _drop_zero_variance(work, regressor_ids)
    ident_to_display = {ident: display for display, ident in name_map.items()}

    ols_res, _ = _fit_ols(work, dep_id, regressor_ids, ident_to_display)
    glm_res, _ = _fit_fractional_logit(work, dep_id, regressor_ids, ident_to_display)

    # Union of parameter names, preserving OLS ordering first.
    ordered_terms: list[str] = []
    seen = set()
    for term in list(ols_res.params.index) + list(glm_res.params.index):
        if term not in seen:
            seen.add(term)
            ordered_terms.append(term)

    # Reorder to the preferred presentation order, with Intercept pushed last.
    _DISPLAY_ORDER = [
        "Chunk",
        "Lab",
        "Lab x Low",
        "Lab x High",
        "Classroom",
        "Classroom x Low",
        "Classroom x High",
        "Private",
        "Public",
        "Reasoning",
    ]
    _display_rank = {name: idx for idx, name in enumerate(_DISPLAY_ORDER)}

    def _term_rank(term: str) -> tuple[int, int]:
        if term == "Intercept":
            return (2, 0)
        label = _prettify_term(term, ident_to_display)
        if label in _display_rank:
            return (0, _display_rank[label])
        return (1, 0)

    ordered_terms.sort(key=_term_rank)

    def _cell(res, term: str) -> tuple[str, str]:
        if term not in res.params.index:
            return "--", ""
        coef = res.params[term]
        se = res.bse[term]
        p = res.pvalues[term]
        marker = _sig_marker(p)
        coef_str = _format_number(coef, 3)
        if marker:
            coef_str = f"{coef_str}$^{{{marker}}}$"
        se_str = f"({_format_number(se, 3)})"
        return coef_str, se_str

    rows: list[list[str]] = []
    for term in ordered_terms:
        label = _prettify_term_latex(term, ident_to_display)
        ols_coef, ols_se = _cell(ols_res, term)
        glm_coef, glm_se = _cell(glm_res, term)
        rows.append([label, ols_coef, glm_coef])
        rows.append(["", ols_se, glm_se])

    # Footer rows.
    rows.append(["\\midrule N", f"{int(ols_res.nobs):,}", f"{int(glm_res.nobs):,}"])
    rows.append(
        [
            "R$^2$",
            _format_number(getattr(ols_res, "rsquared", None), 3),
            "--",
        ]
    )
    rows.append(
        [
            "Adj.\\ R$^2$",
            _format_number(getattr(ols_res, "rsquared_adj", None), 3),
            "--",
        ]
    )
    pseudo_r2 = None
    try:
        pseudo_r2 = 1.0 - (glm_res.deviance / glm_res.null_deviance)
    except Exception:  # noqa: BLE001
        pseudo_r2 = None
    rows.append(["Pseudo R$^2$", "--", _format_number(pseudo_r2, 3)])
    rows.append(
        [
            "Log-likelihood",
            _format_number(getattr(ols_res, "llf", None), 3),
            _format_number(getattr(glm_res, "llf", None), 3),
        ]
    )

    header = [
        "",
        "(1) OLS",
        "(2) Fractional Logit",
    ]
    return header, rows


def write_regression_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    dependent_var: str = "Wasserstein-1",
    regressors: Iterable[str] = DEFAULT_REGRESSORS,
) -> None:
    header, rows = _build_regression_table(df, dependent_var, list(regressors))
    _write_tabular(
        out_path,
        header=header,
        rows=rows,
        col_spec="lcc",
    )
    # Append a trailing \par with the significance legend so the \input site
    # gets the legend automatically.
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n\\par\\smallskip\\footnotesize "
            "\\textit{Notes:} Dependent variable = Wasserstein-1. "
            "Standard errors in parentheses (HC1 robust for Fractional Logit). "
            "Prompt-condition reference = ``Not Specified / Not Specified'' "
            "(no context, no incentive). Privacy Treatment reference = "
            "``Not Specified''. "
            "Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.\n"
        )


# ---------------------------------------------------------------------------
# Section 2: Overall W1 summary (phase_2 behavior)
# ---------------------------------------------------------------------------


def filter_behavior(sims_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate notebook Cell 3 filters for phase_2 behavior analysis."""
    excluded_models = {"google/gemini-3-pro-preview", "gemini-3-pro-preview"}
    df = sims_df[sims_df["Game"].isin(BEHAVIOR_GAMES)].copy()
    df = df[df["Extra Flag"].astype(str) == "[]"]
    if "Explain Reasoning" in df.columns:
        df = df[df["Explain Reasoning"].astype(bool) == True]  # noqa: E712
    df = df[~df["LLM Model"].astype(str).isin(excluded_models)]
    return df


_LABEL_COLUMNS = {
    "Mode",
    "Game",
    "LLM Model",
    "Explain Reasoning",
    "Enabled Reasoning",
    "Reasoning Mode",
    "Context",
    "Incentive Size",
    "Privacy Treatment",
}

_INT_COLUMNS = {"N", "ChunkN", "SplitN", "Misaligned N", "# Non-zero W1"}


_HEADER_MULTILINE: dict[str, str] = {
    "Explain Reasoning": r"\begin{tabular}[c]{@{}l@{}}Explain\\Reasoning\end{tabular}",
    "Enabled Reasoning": r"\begin{tabular}[c]{@{}l@{}}Enabled\\Reasoning\end{tabular}",
    "Reasoning Mode": r"\begin{tabular}[c]{@{}l@{}}Reasoning\\Mode\end{tabular}",
    "# Non-zero W1": r"\begin{tabular}[c]{@{}l@{}}Num of\\W1$>$0\end{tabular}",
}


def _summary_to_rows(summary: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    """Format a ``create_grouped_summary`` output as LaTeX-safe rows."""
    header = [
        _HEADER_MULTILINE[c] if c in _HEADER_MULTILINE else _latex_escape(c)
        for c in summary.columns
    ]
    rows = []
    for _, r in summary.iterrows():
        row = []
        for col in summary.columns:
            val = r[col]
            if col in {"Explain Reasoning", "Enabled Reasoning"}:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    row.append("--")
                else:
                    row.append("True" if bool(val) else "False")
            elif col in _LABEL_COLUMNS:
                row.append(_latex_escape(val))
            elif col in _INT_COLUMNS:
                row.append(_format_number(val, 0))
            elif isinstance(
                val, (int, float, np.floating, np.integer)
            ) and not isinstance(val, bool):
                row.append(_format_number(val, 3))
            else:
                row.append(_latex_escape(val))
        rows.append(row)
    return header, rows


def write_overall_w1_summary(df_behavior: pd.DataFrame, out_path: Path) -> None:
    summary = create_grouped_summary(
        df=df_behavior,
        metric="Wasserstein-1",
        group_cols=["Mode", "ChunkN"],
    )
    # Sort: Atomic first, then Chunk by ChunkN ascending (matches notebook).
    if "Mode" in summary.columns and "ChunkN" in summary.columns:
        summary = summary.sort_values(["Mode", "ChunkN"]).reset_index(drop=True)
    header, rows = _summary_to_rows(summary)
    col_spec = "l" * 2 + "r" * (len(header) - 2)
    _write_tabular(out_path, header=header, rows=rows, col_spec=col_spec)


# ---------------------------------------------------------------------------
# Section 2b: Random Number Generation summary (phase_2)
# ---------------------------------------------------------------------------


def write_random_number_generation_summary(
    sims_p2: pd.DataFrame, out_path: Path
) -> pd.DataFrame:
    """Wasserstein-1 summary for the Random Number Generation task.

    This mirrors the original notebook scope: Random Number Generation only,
    excluded models removed, ChunkN in {1, 10}, and Extra Flag in the mainline
    or small temperature experiment runs.
    """
    df = sims_p2[sims_p2["Game"].isin(RANDOM_NUMBER_GAME)].copy()
    df = df[df["Extra Flag"].astype(str).isin({"[]", TEMPERATURE_EXPERIMENT_FLAG})]
    df = _exclude_models(df)
    df = df[pd.to_numeric(df["ChunkN"], errors="coerce").isin([1, 10])]

    if df.empty:
        header = [
            "Mode",
            "ChunkN",
            "Temperature",
            "N",
            "avg",
            "std",
            "min",
            "p25",
            "p50",
            "p75",
            "p90",
        ]
        _write_tabular(
            out_path,
            header=[_latex_escape(c) for c in header],
            rows=[],
            col_spec="lrrrrrrrrrr",
        )
        return pd.DataFrame(columns=header)

    summary = create_grouped_summary(
        df=df,
        metric="Wasserstein-1",
        group_cols=["Mode", "ChunkN", "Temperature"],
    )
    sort_cols = [
        c for c in ("Mode", "ChunkN", "Temperature") if c in summary.columns
    ]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    header, rows = _summary_to_rows(summary)
    col_spec = "lrr" + "r" * (len(header) - 3)
    _write_tabular(out_path, header=header, rows=rows, col_spec=col_spec)
    return summary


# ---------------------------------------------------------------------------
# Section 3: Reasoning on/off (phase_2, screenshot filter)
# ---------------------------------------------------------------------------


def _pick_model(models: Iterable[str], patterns: list[str]) -> str | None:
    models = list(models)
    for pat in patterns:
        for m in models:
            if re.search(pat, m, re.IGNORECASE):
                return m
    return None


def write_reasoning_on_off(sims_p2: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    unique_models = sorted(sims_p2["LLM Model"].dropna().astype(str).unique())
    gemini = _pick_model(unique_models, [r"gemini-3.*flash", r"gemini-3"])
    qwen = _pick_model(unique_models, [r"qwen3-235", r"qwen3-23", r"qwen"])
    picked = [m for m in (gemini, qwen) if m is not None]
    if not picked:
        raise RuntimeError(
            "Could not match Gemini-3-Flash or Qwen3-235 in phase_2 data. "
            f"Available models: {unique_models}"
        )

    small_exp_flag = '["small_experiments_on_removing_reasoning"]'
    df = sims_p2.copy()
    df = df[df["Game"].isin(BEHAVIOR_GAMES)]
    df = df[df["Extra Flag"].astype(str).isin({"[]", small_exp_flag})]
    df = df[df["LLM Model"].astype(str).isin(picked)]
    df = df[pd.to_numeric(df["ChunkN"], errors="coerce").isin([1, 10, 20, 25])]

    if df.empty:
        raise RuntimeError(
            "Reasoning on/off filter produced zero rows. "
            f"Picked models: {picked}. Unique Extra Flag values: "
            f"{sorted(sims_p2['Extra Flag'].dropna().astype(str).unique())}"
        )

    summary = create_grouped_summary(
        df=df,
        metric="Wasserstein-1",
        group_cols=["Explain Reasoning", "Mode", "ChunkN"],
    )
    # Keep only the most useful columns (the grouped summary exposes many;
    # the user asked for N/avg/std/p50).
    keep = ["Explain Reasoning", "Mode", "ChunkN", "N", "avg", "std", "p50"]
    keep = [c for c in keep if c in summary.columns]
    summary = summary[keep]
    sort_cols = [
        c for c in ("Mode", "ChunkN", "Explain Reasoning") if c in summary.columns
    ]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)
    if set(["Mode", "ChunkN", "Explain Reasoning"]).issubset(summary.columns):
        summary = summary[
            ["Mode", "ChunkN", "Explain Reasoning"]
            + [
                c
                for c in summary.columns
                if c not in {"Mode", "ChunkN", "Explain Reasoning"}
            ]
        ]

    header, rows = _summary_to_rows(summary)
    col_spec = "l" * 3 + "r" * (len(header) - 3)
    _write_tabular(out_path, header=header, rows=rows, col_spec=col_spec)

    model_list = ", ".join(f"\\texttt{{{_latex_escape(m)}}}" for m in picked)
    # with out_path.open("a", encoding="utf-8") as fh:
    #     fh.write(
    #         "\n\\par\\smallskip\\footnotesize \\textit{Models:} " f"{model_list}.\n"
    #     )
    return summary


# ---------------------------------------------------------------------------
# Section 3b: Reasoning on/off for large chunks (ChunkN in {50, 100})
# ---------------------------------------------------------------------------


_LARGE_CHUNK_GROUP_COLS = [
    "ChunkN",
    "Explain Reasoning",
    "SplitN",
    "Reasoning Mode",
]
_LARGE_CHUNK_SIZES = [50, 100]


def write_reasoning_large_chunks(sims_p2: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Wasserstein-1 summary on phase_2 behavioral games with ChunkN in
    {50, 100} (chunk elicitation), separating Explain-Reasoning on vs. off
    and stratifying by SplitN + Reasoning Mode. Uses all LLM models present
    in phase_2 (no model filter). The tabular omits ``Mode`` (redundant with
    chunk elicitation here); model list in the footnote uses
    ``_short_model_name`` like the figures."""
    small_exp_flag = '["small_experiments_on_removing_reasoning"]'
    df = sims_p2.copy()
    df = df[df["Game"].isin(BEHAVIOR_GAMES)]
    df = df[df["Extra Flag"].astype(str).isin({"[]", small_exp_flag})]
    df = df[pd.to_numeric(df["ChunkN"], errors="coerce").isin(_LARGE_CHUNK_SIZES)]
    df = _exclude_models(df)

    if df.empty:
        raise RuntimeError(
            "Large-chunk reasoning filter produced zero rows. "
            f"Extra Flag values seen: "
            f"{sorted(sims_p2['Extra Flag'].dropna().astype(str).unique())}"
        )

    group_cols = [c for c in _LARGE_CHUNK_GROUP_COLS if c in df.columns]
    summary = create_grouped_summary(
        df=df,
        metric="Wasserstein-1",
        group_cols=group_cols,
    )
    keep = [*group_cols, "N", "avg", "std", "p50"]
    keep = [c for c in keep if c in summary.columns]
    summary = summary[keep]

    sort_cols = [
        c
        for c in ("ChunkN", "Explain Reasoning", "SplitN", "Reasoning Mode")
        if c in summary.columns
    ]
    summary = summary.sort_values(sort_cols).reset_index(drop=True)

    header, rows = _summary_to_rows(summary)
    # All group cols render as labels ("l"), metric cols as right-aligned ("r").
    col_spec = "l" * len(group_cols) + "r" * (len(header) - len(group_cols))
    _write_tabular(out_path, header=header, rows=rows, col_spec=col_spec)

    model_seen = ", ".join(
        _latex_escape(_short_model_name(m))
        for m in sorted(df["LLM Model"].dropna().astype(str).unique())
    )
    # with out_path.open("a", encoding="utf-8") as fh:
    #     fh.write(
    #         "\n\\par\\smallskip\\footnotesize "
    #         "\\textit{Scope:} phase\\_2 behavioral games, "
    #         f"ChunkN $\\in \\{{{', '.join(str(n) for n in _LARGE_CHUNK_SIZES)}\\}}$, "
    #         "Extra Flag $\\in$ \\{``[]'', small\\_experiments\\_on\\_removing\\_reasoning\\}. "
    #         f"\\textit{{Models:}} {model_seen}.\n"
    #     )
    return summary


# ---------------------------------------------------------------------------
# Section 4: Ground-truth simulations with non-zero W1
# ---------------------------------------------------------------------------


def _filter_ground_truth(sims_p2: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth tables: ground-truth games, ChunkN in {1, 10}, Explain
    Reasoning = True (instruction_config.explain_reasoning), and Extra Flag = []."""
    df = sims_p2.copy()
    df = df[df["Extra Flag"].astype(str) == "[]"]
    df = df[df["Game"].isin(GROUND_TRUTH_GAMES)]
    df = df[pd.to_numeric(df["ChunkN"], errors="coerce").isin([1, 10])]
    if "Explain Reasoning" in df.columns:
        df = df[df["Explain Reasoning"].astype(bool) == True]  # noqa: E712
    return df


def write_ground_truth_summary_by_mode(
    sims_p2: pd.DataFrame, out_path: Path
) -> pd.DataFrame:
    """W1 summary over ground-truth games (Explain Reasoning = True), grouped
    by Mode, with a count of non-zero W1 simulations. Excludes
    ``EXCLUDED_MODEL_IDS`` like the rest of ``main_analysis``."""
    df = _exclude_models(_filter_ground_truth(sims_p2))
    summary = create_grouped_summary(
        df=df,
        metric="Wasserstein-1",
        group_cols=["Mode"],
    )
    w1 = pd.to_numeric(df["Wasserstein-1"], errors="coerce").fillna(0)
    nonzero_by_mode = (
        df.assign(_nz=(w1 != 0).astype(int)).groupby("Mode")["_nz"].sum().astype(int)
    )
    summary["# Non-zero W1"] = (
        summary["Mode"].map(nonzero_by_mode).fillna(0).astype(int)
    )
    if "N" in summary.columns:
        cols = list(summary.columns)
        cols.remove("# Non-zero W1")
        insert_at = cols.index("N") + 1
        cols.insert(insert_at, "# Non-zero W1")
        summary = summary[cols]
    if "Mode" in summary.columns:
        summary = summary.sort_values("Mode").reset_index(drop=True)
    header, rows = _summary_to_rows(summary)
    col_spec = "l" + "r" * (len(header) - 1)
    _write_tabular(out_path, header=header, rows=rows, col_spec=col_spec)
    return summary


# ---------------------------------------------------------------------------
# Section 5: Ground-truth non-zero with exact error rates
# ---------------------------------------------------------------------------


def _benchmark_ground_truth_value(db, game_type: str) -> float | None:
    decisions = get_benchmark_results(_db=db, game_type=game_type)
    values = process_decisions(decisions, 0)
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def write_ground_truth_error_rates(
    db,
    sims_p2: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Rows with non-zero W1 on ground-truth games, restricted to Explain
    Reasoning = True via ``_filter_ground_truth``. LaTeX omits ChunkN (Mode
    suffices), Ground Truth, N, Misaligned counts, and the reasoning flag
    (constant True in this scope). Misaligned \\% only; ``N=100`` is stated
    in ``result.tex`` caption. LLM Model uses ``_short_model_name``.
    Excludes ``EXCLUDED_MODEL_IDS`` like the rest of ``main_analysis``."""
    df_gt = _exclude_models(_filter_ground_truth(sims_p2))
    sim_ids = list(df_gt.index)
    all_results = get_all_simulation_results(_db=db, simulation_ids=sim_ids)

    gt_cache: dict[str, float | None] = {}

    records = []
    for sim_id, row in df_gt.iterrows():
        decisions = all_results.get(sim_id, [])
        values = process_decisions(decisions, 0)
        if not values:
            continue
        game = row["Game"]
        if game not in gt_cache:
            gt_cache[game] = _benchmark_ground_truth_value(db, game)
        gt_value = gt_cache[game]
        if gt_value is None:
            continue
        misaligned = sum(1 for v in values if v != gt_value)
        total = len(values)
        pct = (misaligned / total * 100) if total else 0.0
        records.append(
            {
                "Game": game,
                "Mode": row["Mode"],
                "ChunkN": row["ChunkN"],
                "LLM Model": row["LLM Model"],
                "Ground Truth": gt_value,
                "N": total,
                "Misaligned N": misaligned,
                "Misaligned %": pct,
                "Wasserstein-1": pd.to_numeric(
                    row.get("Wasserstein-1"), errors="coerce"
                ),
            }
        )

    df = pd.DataFrame(records)

    def _error_table_header() -> list[str]:
        return [
            _latex_escape("Game"),
            _latex_escape("Mode"),
            _latex_escape("LLM Model"),
            _latex_escape("Misaligned %"),
            _latex_escape("Wasserstein-1"),
        ]

    if df.empty:
        _write_tabular(
            out_path,
            header=_error_table_header(),
            rows=[],
            col_spec="lllrr",
        )
        return df

    w1 = pd.to_numeric(df["Wasserstein-1"], errors="coerce").fillna(0)
    df = df[w1 != 0].sort_values(["Game", "Mode", "LLM Model"]).reset_index(drop=True)

    header = _error_table_header()
    rows: list[list[str]] = []
    for _, r in df.iterrows():
        rows.append(
            [
                _latex_escape(r["Game"]),
                _latex_escape(r["Mode"]),
                _latex_escape(_short_model_name(r["LLM Model"])),
                _format_number(r["Misaligned %"], 1),
                _format_number(r["Wasserstein-1"], 3),
            ]
        )
    _write_tabular(out_path, header=header, rows=rows, col_spec="lllrr")
    return df


# ---------------------------------------------------------------------------
# Section 6: Figure generation
# ---------------------------------------------------------------------------


def _prep_behavior_frame(sims_p2: pd.DataFrame) -> pd.DataFrame:
    """Behavior-game subset used for Task 1b + Table 2 (all ChunkN, not just
    Explain-Reasoning = True, matching the notebook's ``df_all_behavior``)."""
    df = sims_p2[sims_p2["Game"].isin(BEHAVIOR_GAMES)].copy()
    df = df[df["Extra Flag"].astype(str) == "[]"]
    return _exclude_models(df)


def _make_fig_task1b_w1_by_game(df_1_10: pd.DataFrame) -> None:
    """Task 1b: W1 histograms per game, ChunkN=10 vs Atomic."""
    game_order = [g for g in GAME_ORDER if g in df_1_10["Game"].unique()]

    df_game = df_1_10.copy()
    df_game["game_short"] = df_game["Game"].map(lambda g: SHORT_GAME_NAMES.get(g, g))
    df_game["wasserstein_1"] = df_game["Wasserstein-1"]

    n_games = len(game_order)
    ncols = 5
    nrows = (n_games + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.0 * nrows))
    axes = axes.flatten()

    w1_means = df_1_10.groupby(["Game", "Mode"])["Wasserstein-1"].mean()
    bins = np.linspace(0, 0.75, 12)

    sources = [
        ("ChunkN=10", "Chunk", CHUNK_COLOR, CHUNK_EDGE, 0.45),
        ("Atomic", "Atomic", ATOMIC_COLOR, ATOMIC_EDGE, 0.45),
    ]

    for idx, game in enumerate(game_order):
        ax = axes[idx]
        short = SHORT_GAME_NAMES.get(game, game)

        for label, mode_key, fill, line, alpha in sources:
            vals = (
                df_game.loc[
                    (df_game["Game"] == game) & (df_game["Mode"] == mode_key),
                    "wasserstein_1",
                ]
                .dropna()
                .values
            )
            if len(vals) == 0:
                continue
            weights = np.ones_like(vals) / len(vals)
            ax.hist(
                vals,
                bins=bins,
                weights=weights,
                label=label if idx == 0 else None,
                color=fill,
                edgecolor=line,
                linewidth=0.7,
                alpha=alpha,
            )

        w1_c = w1_means.get((game, "Chunk"), float("nan"))
        w1_a = w1_means.get((game, "Atomic"), float("nan"))
        ax.text(
            0.5,
            0.95,
            f"W1: C={w1_c:.3f}, A={w1_a:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=_FIG_ANNOTATION_FS,
            color="#333333",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )
        ax.set_title(short, fontsize=_FIG_TASK12_PANEL_TITLE_FS, pad=8)
        ax.tick_params(
            axis="both",
            labelsize=_FIG_TASK12_TICK_FS,
            **_FIG_TICK_PARAMS,
        )
        ax.set_facecolor("white")
        ax.grid(axis="y", alpha=0.25)
        if idx >= ncols:
            ax.set_xlabel("W1 Distance", fontsize=_FIG_AXIS_LABEL_FS)
        else:
            ax.set_xlabel("")

    for ax in axes[n_games:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, _FIG_GRID_LEGEND_Y),
        frameon=False,
        fontsize=_FIG_GRID_LEGEND_FS,
    )
    fig.tight_layout(rect=_FIG_GRID_NO_SUPTITLE_RECT)
    _save_figure(fig, "task1b_w1_by_game.png")


def _make_fig_task1c_atomic_vs_chunk_scatter(df_1_10: pd.DataFrame) -> None:
    """Task 1c: scatter of per-(Model, Game) mean Wasserstein-1 with Atomic
    on the x-axis and ChunkN=10 on the y-axis, plus a 45-degree reference
    line. Uses the same phase_2 behavioral scope as Task 1b."""
    work = df_1_10.dropna(subset=["Wasserstein-1", "LLM Model", "Game", "Mode"]).copy()
    work = work[work["Mode"].isin(["Atomic", "Chunk"])]
    if work.empty:
        print("  [warn] task1c scatter: no rows after filtering")
        return

    agg = work.groupby(["LLM Model", "Game", "Mode"], as_index=False)[
        "Wasserstein-1"
    ].mean()
    wide = agg.pivot_table(
        index=["LLM Model", "Game"],
        columns="Mode",
        values="Wasserstein-1",
    ).reset_index()
    wide = wide.dropna(subset=["Atomic", "Chunk"])
    if wide.empty:
        print("  [warn] task1c scatter: no Atomic/Chunk pairs found")
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.8, 7.8))

    ax.scatter(
        wide["Atomic"].to_numpy(),
        wide["Chunk"].to_numpy(),
        s=62,
        color=CHUNK_COLOR,
        edgecolor=CHUNK_EDGE,
        linewidth=0.45,
        alpha=0.3,
    )

    lo = 0.0
    hi = float(max(wide["Atomic"].max(), wide["Chunk"].max(), 0.6))
    hi = min(hi * 1.05, 1.0)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="grey",
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Wasserstein-1 -- Atomic", fontsize=_FIG_AXIS_LABEL_FS)
    ax.set_ylabel("Wasserstein-1 -- ChunkN=10", fontsize=_FIG_AXIS_LABEL_FS)
    # ax.set_title(
    #     "Wasserstein-1 Distance per Model, Game",
    #     fontsize=_FIG_TITLE_FS,
    #     pad=14,
    # )
    ax.tick_params(
        axis="both",
        labelsize=_FIG_TICK_FS,
        **_FIG_TICK_PARAMS,
    )
    ax.set_facecolor("white")

    fig.tight_layout()
    _save_figure(fig, "task1c_atomic_vs_chunk10_scatter.png")


def _make_fig_table2_raincloud(df_1_10: pd.DataFrame) -> None:
    """Table 2 raincloud: W1 distribution by model, split by elicitation mode."""
    df_t2 = df_1_10.copy()
    df_t2["model"] = df_t2["LLM Model"].map(_short_model_name)
    df_t2["elicitation_mode"] = df_t2["Mode"].replace(
        {"Chunk": "ChunkN=10", "Atomic": "Atomic"}
    )
    df_t2["wasserstein_1"] = df_t2["Wasserstein-1"]
    model_order = [m for m in MODEL_LABEL_ORDER if m in df_t2["model"].unique()]

    palette = {"Atomic": ATOMIC_COLOR, "ChunkN=10": CHUNK_COLOR}
    hue_order = ["Atomic", "ChunkN=10"]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    sns.violinplot(
        data=df_t2,
        x="wasserstein_1",
        y="model",
        hue="elicitation_mode",
        order=model_order,
        hue_order=hue_order,
        orient="h",
        split=True,
        inner=None,
        cut=0,
        bw_method=0.2,
        linewidth=0.8,
        alpha=0.4,
        palette=palette,
        saturation=1,
        ax=ax,
    )
    sns.boxplot(
        data=df_t2,
        x="wasserstein_1",
        y="model",
        hue="elicitation_mode",
        order=model_order,
        hue_order=hue_order,
        orient="h",
        dodge=True,
        width=0.22,
        showfliers=False,
        palette=palette,
        boxprops={"alpha": 0.9},
        whiskerprops={"linewidth": 1},
        capprops={"linewidth": 1},
        medianprops={"color": "white", "linewidth": 1.2},
        ax=ax,
    )
    sns.stripplot(
        data=df_t2,
        x="wasserstein_1",
        y="model",
        hue="elicitation_mode",
        order=model_order,
        hue_order=hue_order,
        orient="h",
        dodge=True,
        size=3,
        jitter=0.08,
        alpha=0.6,
        palette=palette,
        linewidth=0,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for h, l in zip(handles, labels):
        if l in hue_order and l not in unique:
            unique[l] = h
    ax.legend(
        [unique[h] for h in hue_order if h in unique],
        [h for h in hue_order if h in unique],
        title="",
        loc="lower right",
        bbox_to_anchor=(0.92, 0.08),
        borderaxespad=0,
        frameon=False,
        fontsize=_FIG_RAINCLOUD_LEGEND_FS,
    )

    stats = (
        df_t2.groupby(["model", "elicitation_mode"])["wasserstein_1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    model_to_y = {m: float(i) for i, m in enumerate(model_order)}
    for model in model_order:
        sub = stats[stats["model"] == model]
        lines: list[str] = []
        # ChunkN=10 line first, then Atomic (split-violin vertical order).
        for mode_label in ("ChunkN=10", "Atomic"):
            row_m = sub[sub["elicitation_mode"] == mode_label]
            if row_m.empty:
                continue
            r = row_m.iloc[0]
            lines.append(f"{r['mean']:.2f} ({r['std']:.2f})")
        if not lines:
            continue
        y = model_to_y[model]
        ax.text(
            _FIG_RAINCLOUD_STAT_X,
            y,
            "\n".join(lines),
            fontsize=_FIG_RAINCLOUD_STAT_FS,
            linespacing=_FIG_RAINCLOUD_STAT_LINESPACING,
            va="center",
            ha="left",
            clip_on=False,
        )

    ax.set_xlim(0, _FIG_RAINCLOUD_XLIM_HI)
    ax.set_xlabel("Wasserstein-1 Distance", fontsize=_FIG_AXIS_LABEL_FS)
    ax.set_ylabel("")
    ax.tick_params(
        axis="x",
        labelsize=_FIG_TASK12_TICK_FS,
        **_FIG_TICK_PARAMS,
    )
    ax.tick_params(
        axis="y",
        labelsize=_FIG_RAINCLOUD_Y_TICK_FS,
        **_FIG_TICK_PARAMS,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.08)
    sns.despine(ax=ax, left=False, bottom=False)

    fig.tight_layout()
    _save_figure(fig, "table2_raincloud_model_performance.png")


def _pool_decisions(
    db,
    df_subset: pd.DataFrame,
    games: list[str],
) -> tuple[dict[tuple[str, str], list], dict[str, list]]:
    """Pool processed decisions per (game, mode) from simulations in ``df_subset``
    (expects ChunkN in {1, 10}), plus benchmark decisions per game."""
    sim_ids_by_group: dict[tuple[str, str], list] = {}
    for game in games:
        for mode_label, chunk_n in [("Atomic", 1), ("Chunk", 10)]:
            mask = (
                (df_subset["Game"] == game)
                & (df_subset["Mode"] == mode_label)
                & (df_subset["ChunkN"] == chunk_n)
            )
            sim_ids_by_group[(game, mode_label)] = df_subset[mask].index.tolist()

    all_sim_ids = [sid for ids in sim_ids_by_group.values() for sid in ids]
    all_raw = get_all_simulation_results(_db=db, simulation_ids=tuple(all_sim_ids))

    pooled: dict[tuple[str, str], list] = {}
    for (game, mode_label), sids in sim_ids_by_group.items():
        decs: list = []
        for sid in sids:
            decs.extend(process_decisions(all_raw.get(sid, []), 0))
        pooled[(game, mode_label)] = decs

    benchmarks: dict[str, list] = {}
    for game in games:
        benchmarks[game] = process_decisions(
            get_benchmark_results(_db=db, game_type=game), 0
        )

    return pooled, benchmarks


def _plot_distribution_grid(
    axes,
    games: list[str],
    pooled: dict[tuple[str, str], list],
    benchmarks: dict[str, list],
    *,
    benchmark_label: str,
    benchmark_key: str,
    config_overrides: dict[str, tuple] | None = None,
    width_overrides: dict[str, float] | None = None,
    title_fontsize: float | None = None,
    tick_labelsize: float | None = None,
) -> None:
    panel_title_fs = title_fontsize if title_fontsize is not None else _FIG_TITLE_FS
    panel_tick_fs = tick_labelsize if tick_labelsize is not None else _FIG_TICK_FS
    sources = [
        (benchmark_label, benchmark_key, "#9a9a9a", "#555555", 0.55),
        ("ChunkN=10", "Chunk", CHUNK_COLOR, CHUNK_EDGE, 0.60),
        ("Atomic", "Atomic", ATOMIC_COLOR, ATOMIC_EDGE, 0.60),
    ]

    for idx, game in enumerate(games):
        ax = axes[idx]
        if config_overrides and game in config_overrides:
            x_range, is_discrete = config_overrides[game]
        else:
            x_range, is_discrete = GAME_DECISION_ARRAY_CONFIG[game][0]

        src = {
            benchmark_key: benchmarks[game],
            "Chunk": pooled[(game, "Chunk")],
            "Atomic": pooled[(game, "Atomic")],
        }

        if is_discrete:
            cats = (
                sorted(x_range)
                if isinstance(x_range, list)
                else list(range(x_range[0], x_range[1] + 1))
            )
            x = np.arange(len(cats))
            width = (width_overrides or {}).get(game, 0.26)
            for j, (label, key, fill, line, alpha) in enumerate(sources):
                vals = src[key]
                if not vals:
                    continue
                total = len(vals)
                cts = Counter(vals)
                props = np.array([cts.get(cat, 0) / total for cat in cats])
                ax.bar(
                    x + (j - 1) * width,
                    props,
                    width=width,
                    label=label if idx == 0 else None,
                    color=fill,
                    edgecolor=line,
                    linewidth=0.8,
                    alpha=alpha,
                )
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in cats])
        else:
            lo, hi = x_range
            edges = np.linspace(lo, hi, 11)
            for label, key, fill, line, alpha in sources:
                vals = src[key]
                if not vals:
                    continue
                vals_arr = np.asarray(vals)
                weights = np.ones_like(vals_arr, dtype=float) / len(vals_arr)
                ax.hist(
                    vals_arr,
                    bins=edges,
                    weights=weights,
                    label=label if idx == 0 else None,
                    color=fill,
                    edgecolor=line,
                    linewidth=0.7,
                    alpha=alpha,
                )

        ax.set_title(SHORT_GAME_NAMES.get(game, game), fontsize=panel_title_fs, pad=8)
        ax.tick_params(
            axis="both",
            labelsize=panel_tick_fs,
            **_FIG_TICK_PARAMS,
        )
        ax.set_facecolor("white")
        ax.grid(axis="y", alpha=0.25)


def _make_fig_task2_decision_distributions(db, df_1_10: pd.DataFrame) -> None:
    """Task 2: decision distributions across 10 behavioral games."""
    game_order = [g for g in GAME_ORDER if g in df_1_10["Game"].unique()]
    pooled, human_bm = _pool_decisions(db, df_1_10, game_order)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    axes = axes.flatten()
    _plot_distribution_grid(
        axes,
        game_order,
        pooled,
        human_bm,
        benchmark_label="Human",
        benchmark_key="human",
        title_fontsize=_FIG_TASK12_PANEL_TITLE_FS,
        tick_labelsize=_FIG_TASK12_TICK_FS,
    )
    for ax in axes[len(game_order) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, _FIG_GRID_LEGEND_Y),
        frameon=False,
        fontsize=_FIG_GRID_LEGEND_FS,
    )
    fig.tight_layout(rect=_FIG_GRID_NO_SUPTITLE_RECT)
    _save_figure(fig, "task2_decision_distributions.png")


def _make_fig_task2b_ground_truth_distributions(db, sims_p2: pd.DataFrame) -> None:
    """Task 2b: decision distributions on the four ground-truth games (Extra Flag = [])."""
    df_gt = _exclude_models(sims_p2[sims_p2["Game"].isin(GROUND_TRUTH_GAMES)].copy())
    df_gt = df_gt[df_gt["Extra Flag"].astype(str) == "[]"]
    df_gt = df_gt[df_gt["ChunkN"].isin([1, 10])]
    ngt_games = [g for g in GROUND_TRUTH_GAMES if g in df_gt["Game"].unique()]

    pooled_gt, bm_gt = _pool_decisions(db, df_gt, ngt_games)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    _plot_distribution_grid(
        axes,
        ngt_games,
        pooled_gt,
        bm_gt,
        benchmark_label="Benchmark",
        benchmark_key="benchmark",
        config_overrides={
            "Arithmetic Verification": ((36, 45), True),
        },
        title_fontsize=_FIG_TASK12_PANEL_TITLE_FS,
        tick_labelsize=_FIG_TASK12_TICK_FS,
    )
    for ax in axes[len(ngt_games) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, _FIG_GRID_LEGEND_Y),
        frameon=False,
        fontsize=_FIG_GRID_LEGEND_FS,
    )
    fig.tight_layout(rect=_FIG_GRID_NO_SUPTITLE_RECT)
    _save_figure(fig, "task2b_ground_truth_decision_distributions.png")


def generate_figures(db, sims_p2: pd.DataFrame) -> list[str]:
    """Render every figure consumed by ``tex/result.tex`` directly from the
    phase_2 simulation DataFrame, saving into ``output/figures/`` and then
    mirroring the four paper figures into ``tex/figs/``.
    """
    sns.set_theme(style="whitegrid")

    df_all_behavior = _prep_behavior_frame(sims_p2)
    df_1_10 = df_all_behavior[df_all_behavior["ChunkN"].isin([1, 10])].copy()

    _make_fig_task1b_w1_by_game(df_1_10)
    _make_fig_task1c_atomic_vs_chunk_scatter(df_1_10)
    _make_fig_table2_raincloud(df_1_10)
    _make_fig_task2_decision_distributions(db, df_1_10)
    _make_fig_task2b_ground_truth_distributions(db, sims_p2)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in FIGURE_FILES:
        src = SRC_FIGS_DIR / name
        if not src.exists():
            print(f"  [warn] figure missing after generation: {src}")
            continue
        shutil.copy2(src, FIGS_DIR / name)
        copied.append(name)
    return copied


# ---------------------------------------------------------------------------
# Section 7: Master result.tex
# ---------------------------------------------------------------------------


RESULT_TEX_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\graphicspath{{figs/}}
\setlength{\tabcolsep}{5pt}

\title{Phase 2 / Phase 2 Context Analysis}
\author{Howie Zhi-Hong Jian}
\date{\today}

\begin{document}
\maketitle

\section{Regression: Wasserstein-1 on phase\_2\_context}
\begin{table}[H]
\centering
\caption{Regression of Wasserstein-1 on elicitation mode, observed context-incentive prompt-condition cells, and controls (phase\_2\_context). Column (1) is OLS; column (2) is a fractional logit (GLM, Binomial, logit link) with HC1 robust standard errors.}
\label{tab:regression-w1-p2c}
\input{tables/regression_w1_phase2_context}
\end{table}

\section{Overall Wasserstein-1 Summary (phase\_2)}
\begin{table}[H]
\centering
\caption{Summary statistics of Wasserstein-1 by Mode and ChunkN over phase\_2 behavioral games (Explain Reasoning = True, Extra Flag = []).}
\label{tab:w1-overall-p2}
\input{tables/w1_overall_summary_phase2}
\end{table}

\section{Explain Reasoning On vs.\ Off (phase\_2)}
\begin{table}[H]
\centering
\caption{Wasserstein-1 with Explain Reasoning on vs.\ off, restricted to Gemini-3 Flash and Qwen3-235 over ChunkN $\in \{1, 10, 20, 25\}$.}
\label{tab:w1-reasoning-p2}
\input{tables/w1_reasoning_on_off_phase2}
\end{table}

\section{Explain Reasoning On vs.\ Off -- Large Chunks (phase\_2)}
\begin{table}[H]
\centering
\caption{Wasserstein-1 with Explain Reasoning on vs.\ off on phase\_2 behavioral games for ChunkN $\in \{50, 100\}$, grouped by ChunkN, Explain Reasoning, SplitN, and Reasoning Mode (chunk elicitation only). All LLM models included.}
\label{tab:w1-reasoning-large-chunks-p2}
\input{tables/w1_reasoning_on_off_large_chunks_phase2}
\end{table}

\section{Wasserstein-1 Summary by Mode (Ground-truth Games)}
\begin{table}[H]
\centering
\caption{Wasserstein-1 summary statistics over the four ground-truth games (TicTacToe Logic L2, TicTacToe Logic, Arithmetic Verification, Trivial Dominance), phase\_2, ChunkN $\in \{1, 10\}$, Explain Reasoning = True.}
\label{tab:gt-summary-mode}
\input{tables/w1_summary_ground_truth_by_mode}
\end{table}

\section{Ground-truth Non-zero W1 with Error Rates}
\begin{table}[H]
\centering
\caption{Simulations on ground-truth games (Explain Reasoning = True) where Wasserstein-1 $\neq 0$, with the share of decisions misaligned with the ground-truth benchmark (Misaligned \%). Each listed simulation pools $N=100$ decisions.}
\label{tab:gt-errors}
\input{tables/ground_truth_non_zero_error_rates}
\end{table}

\section{Random Number Generation Summary}
\begin{table}[H]
\centering
\caption{Wasserstein-1 summary statistics for Random Number Generation, phase\_2, grouped by Mode, ChunkN, and Temperature, with Extra Flag = [] or small\_experiments\_on\_temperature.}
\label{tab:w1-random-number-generation-p2}
\input{tables/w1_random_number_generation_phase2}
\end{table}

\section{Figures}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{table2_raincloud_model_performance.png}
\caption{Raincloud of W1 by LLM Model, split by elicitation mode (ChunkN=10 vs.\ Atomic).}
\label{fig:raincloud-model}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{task1b_w1_by_game.png}
\caption{Distribution of W1 by game, ChunkN=10 vs.\ Atomic.}
\label{fig:w1-by-game}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.7\textwidth]{task1c_atomic_vs_chunk10_scatter.png}
\caption{Atomic vs.\ ChunkN=10 per-(Model, Game) mean Wasserstein-1. Each point is one (LLM Model, Game) pair over the phase\_2 behavioral-game scope; the dashed grey line is the equality reference. Points with $\alpha=0.1$ convey density; points below the line indicate ChunkN=10 reduces W1 relative to Atomic.}
\label{fig:atomic-vs-chunk-scatter}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{task2_decision_distributions.png}
\caption{Decision distributions across behavioral games: human benchmark vs.\ ChunkN=10 vs.\ Atomic.}
\label{fig:dist-behavior}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{task2b_ground_truth_decision_distributions.png}
\caption{Decision distributions for ground-truth games: benchmark vs.\ ChunkN=10 vs.\ Atomic.}
\label{fig:dist-groundtruth}
\end{figure}

\end{document}
"""


def write_result_tex() -> None:
    (TEX_DIR / "result.tex").write_text(RESULT_TEX_TEMPLATE, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    db = get_combined_database(
        data_root=DATA_DIR,
        experiment_names=("exp1", "exp2", "exp3"),
    )

    print("[1/7] Loading phase_2_context simulations...")
    sims_p2c = show_all_simulations_df(
        _db=db,
        filter_incomplete=True,
        phase_name="phase_2_context",
    )
    print(f"      phase_2_context sims: {len(sims_p2c)}")
    sims_p2c = attach_ks_test_results_to_simulations_df(
        _db=db, simulations_df=sims_p2c, decision_index=0, alpha=0.05
    )
    sims_p2c = sims_p2c[sims_p2c["Extra Flag"].astype(str) == "[]"]
    print(f"      phase_2_context sims (Extra Flag = []): {len(sims_p2c)}")

    print("[2/7] Writing regression table...")
    write_regression_table(
        sims_p2c,
        TABLES_DIR / "regression_w1_phase2_context.tex",
        dependent_var="Wasserstein-1",
        regressors=DEFAULT_REGRESSORS,
    )

    print("[3/7] Loading phase_2 simulations...")
    sims_p2 = show_all_simulations_df(
        _db=db,
        filter_incomplete=True,
        phase_name="phase_2",
    )
    print(f"      phase_2 sims: {len(sims_p2)}")
    sims_p2 = attach_ks_test_results_to_simulations_df(
        _db=db, simulations_df=sims_p2, decision_index=0, alpha=0.05
    )

    print("[4/7] Writing overall W1 summary (phase_2 behavior)...")
    df_behavior = filter_behavior(sims_p2)
    print(f"      behavior sims: {len(df_behavior)}")
    write_overall_w1_summary(df_behavior, TABLES_DIR / "w1_overall_summary_phase2.tex")

    print("[5/7] Writing Explain Reasoning on/off tables...")
    write_reasoning_on_off(sims_p2, TABLES_DIR / "w1_reasoning_on_off_phase2.tex")
    write_reasoning_large_chunks(
        sims_p2, TABLES_DIR / "w1_reasoning_on_off_large_chunks_phase2.tex"
    )

    print("[6/7] Writing ground-truth and random-number tables...")
    write_ground_truth_summary_by_mode(
        sims_p2, TABLES_DIR / "w1_summary_ground_truth_by_mode.tex"
    )
    write_ground_truth_error_rates(
        db, sims_p2, TABLES_DIR / "ground_truth_non_zero_error_rates.tex"
    )
    write_random_number_generation_summary(
        sims_p2, TABLES_DIR / "w1_random_number_generation_phase2.tex"
    )
    # Drop the deprecated non-zero listing if it was generated by an older run.
    old_nonzero = TABLES_DIR / "ground_truth_non_zero_w1.tex"
    if old_nonzero.exists():
        old_nonzero.unlink()

    print("[7/7] Generating figures + writing result.tex...")
    copied = generate_figures(db, sims_p2)
    print(f"      figures ready: {copied}")
    write_result_tex()

    print("Done. Artifacts under:", TEX_DIR)


if __name__ == "__main__":
    main()
