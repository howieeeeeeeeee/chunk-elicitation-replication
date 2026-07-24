"""Summary table generation for phase_2 analysis outputs."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_summarizer import create_grouped_summary
from data_summarizer.simulations import process_decisions
from db_ops.retrievers import get_all_simulation_results, get_benchmark_results
from games.instructions import BEHAVIOR_GAMES

from .config import (
    GROUND_TRUTH_GAMES,
    RANDOM_NUMBER_GAME,
    TEMPERATURE_EXPERIMENT_FLAG,
    _exclude_models,
    _short_model_name,
)
from .latex import _format_number, _latex_escape, _write_tabular

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
                "Simulation ID": str(sim_id),
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
    df = (
        df[w1 != 0]
        .sort_values(
            [
                "Game",
                "Mode",
                "LLM Model",
                "Misaligned %",
                "Wasserstein-1",
                "Simulation ID",
            ]
        )
        .reset_index(drop=True)
    )

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
