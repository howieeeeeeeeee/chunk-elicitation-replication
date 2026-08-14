"""Selected within-response mechanism outputs for the canonical paper."""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.perform_test import perform_wasserstein_1
from data_summarizer.simulations import get_simulation_info, process_decisions
from db_ops.retrievers import get_benchmark_results
from games.instructions import BEHAVIOR_GAMES

from .config import ATOMIC_COLOR, CHUNK_COLOR, EXCLUDED_MODEL_IDS, _short_model_name
from .environment import FIGS_DIR, TABLES_DIR
from .latex import _format_number, _write_tabular

plt.switch_backend("Agg")

FIRST_COLOR = "#E9C46A"
_TITLE_FS = 15
_LABEL_FS = 13
_TICK_FS = 11
_LEGEND_FS = 11


def _extract_value(item) -> float | None:
    if isinstance(item, dict):
        item = item.get("decision", [])
    if isinstance(item, (list, tuple)):
        if not item:
            return None
        if isinstance(item[0], (list, tuple)):
            item = item[0]
        try:
            return float(item[0])
        except (IndexError, TypeError, ValueError):
            return None
    if isinstance(item, (int, float)):
        return float(item)
    return None


def _decision_frame(db) -> pd.DataFrame:
    sims = db.simulations.find(
        {
            "archived": False,
            "completed": True,
            "phase_name": "phase_2",
            "simulation_config.game_type": {"$in": BEHAVIOR_GAMES},
            "instruction_config.explain_reasoning": True,
        }
    )
    selected = []
    session_ids = []
    for sim in sims:
        info = get_simulation_info(sim)
        if str(info["Extra Flag"]) != "[]":
            continue
        if str(info["LLM Model"]) in EXCLUDED_MODEL_IDS:
            continue
        mode, chunk_n = info["Mode"], int(info["ChunkN"])
        if not ((mode == "Atomic" and chunk_n == 1) or
                (mode == "Chunk" and chunk_n == 10)):
            continue
        selected.append((sim, info))
        session_ids.extend(sim.get("simulation_sessions", []))

    sessions = {
        session["_id"]: session
        for session in db.simulation_sessions.find(
            {"_id": {"$in": session_ids}}, {"_id": 1, "decisions": 1}
        )
    }
    rows = []
    for sim, info in selected:
        for session_id in sim.get("simulation_sessions", []):
            session = sessions.get(session_id)
            if not session:
                continue
            for position, item in enumerate(session.get("decisions", []) or []):
                value = _extract_value(item)
                if value is not None:
                    rows.append(
                        {
                            "sim_id": sim["_id"],
                            "model": _short_model_name(info["LLM Model"]),
                            "Game": info["Game"],
                            "Mode": info["Mode"],
                            "ChunkN": int(info["ChunkN"]),
                            "session_id": session_id,
                            "position": position,
                            "value": value,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No Atomic or ChunkN=10 mechanism observations found.")
    return frame


def _w1(sample, reference, game: str) -> float:
    value = perform_wasserstein_1(list(sample), list(reference), game)
    return np.nan if value is None else float(value)


def _modal_share(values) -> float:
    counts = Counter(round(value) for value in values)
    return max(counts.values()) / len(values)


def _mechanism_aggregates(frame: pd.DataFrame, db):
    atomic_pools = {
        key: group["value"].tolist()
        for key, group in frame[frame["Mode"] == "Atomic"].groupby(
            ["model", "Game"]
        )
    }
    benchmarks = {}
    first_rows, position_rows, cumulative_rows = [], [], []

    for sim_id, group in frame.groupby("sim_id"):
        game = group["Game"].iloc[0]
        if game not in benchmarks:
            raw = get_benchmark_results(_db=db, game_type=game)
            benchmarks[game] = process_decisions(raw, 0) if raw else []
        human = benchmarks[game]
        if not human:
            continue

        mode = group["Mode"].iloc[0]
        model = group["model"].iloc[0]
        values = group["value"].tolist()
        base = {
            "sim_id": sim_id,
            "Game": game,
            "Mode": mode,
            "modal_human": _modal_share(human),
            "w1_human": _w1(values, human, game),
            "modal": _modal_share(values),
        }
        if mode == "Atomic":
            first_rows.append(base)
            continue

        atomic = atomic_pools[(model, game)]
        first = group[group["position"] == 0]["value"].tolist()
        first_rows.append(
            base
            | {
                "w1_first": _w1(first, human, game),
                "modal_first": _modal_share(first),
                "w1_first_vs_atomic": _w1(first, atomic, game),
                "w1_first_vs_full": _w1(first, values, game),
            }
        )

        for position, position_group in group.groupby("position"):
            position_values = position_group["value"].tolist()
            position_rows.append(
                {
                    "sim_id": sim_id,
                    "position": int(position),
                    "answer": int(position) + 1,
                    "n_values": len(position_values),
                    "w1_human": _w1(position_values, human, game),
                    "modal": _modal_share(position_values),
                    "w1_vs_atomic": _w1(position_values, atomic, game),
                    "w1_vs_full": _w1(position_values, values, game),
                }
            )

        for prefix in range(1, int(group["position"].max()) + 2):
            prefix_values = group[group["position"] < prefix]["value"].tolist()
            cumulative_rows.append(
                {
                    "sim_id": sim_id,
                    "prefix": prefix,
                    "n_values": len(prefix_values),
                    "w1_human": _w1(prefix_values, human, game),
                    "modal": _modal_share(prefix_values),
                    "w1_vs_atomic": _w1(prefix_values, atomic, game),
                    "w1_vs_full": _w1(prefix_values, values, game),
                }
            )

    first = pd.DataFrame(first_rows)
    atomic = first[first["Mode"] == "Atomic"]
    chunk = first[first["Mode"] == "Chunk"]
    reference = pd.Series(
        {
            "n_atomic_sims": len(atomic),
            "n_chunk_sims": len(chunk),
            "w1_atomic": atomic["w1_human"].mean(),
            "w1_first": chunk["w1_first"].mean(),
            "w1_full": chunk["w1_human"].mean(),
            "w1_first_vs_atomic": chunk["w1_first_vs_atomic"].mean(),
            "w1_first_vs_full": chunk["w1_first_vs_full"].mean(),
            "modal_atomic": atomic["modal"].mean(),
            "modal_first": chunk["modal_first"].mean(),
            "modal_full": chunk["modal"].mean(),
            "modal_human": first.groupby("Game")["modal_human"].first().mean(),
        }
    )
    aggregations = {
        "n_sims": ("sim_id", "nunique"),
        "mean_n_values": ("n_values", "mean"),
        "w1_human": ("w1_human", "mean"),
        "modal": ("modal", "mean"),
        "w1_vs_atomic": ("w1_vs_atomic", "mean"),
        "w1_vs_full": ("w1_vs_full", "mean"),
    }
    positions = (
        pd.DataFrame(position_rows)
        .groupby(["position", "answer"], as_index=False)
        .agg(**aggregations)
        .sort_values("position")
    )
    cumulative = (
        pd.DataFrame(cumulative_rows)
        .groupby("prefix", as_index=False)
        .agg(**aggregations)
        .sort_values("prefix")
    )
    if positions["answer"].tolist() != list(range(1, 11)):
        raise RuntimeError("Mechanism analysis requires complete Answers 1--10.")
    return reference, positions, cumulative


def _write_position_table(data: pd.DataFrame, ref: pd.Series) -> None:
    rows = [["Atomic", _format_number(ref["w1_atomic"]),
             _format_number(ref["modal_atomic"]), "--", "--"]]
    rows.extend(
        [
            f"Answer {int(row.answer)}",
            _format_number(row.w1_human),
            _format_number(row.modal),
            _format_number(row.w1_vs_atomic),
            _format_number(row.w1_vs_full),
        ]
        for row in data.itertuples()
    )
    rows.extend(
        [
            ["Chunk: full", _format_number(ref["w1_full"]),
             _format_number(ref["modal_full"]), "--", "--"],
            ["Human benchmark", _format_number(0.0),
             _format_number(ref["modal_human"]), "--", "--"],
        ]
    )
    _write_tabular(
        TABLES_DIR / "mechanism_position_answer_w1.tex",
        ["Distribution", r"$W_1\!\to$ Human", "Modal share",
         r"$W_1\!\to$ Atomic", r"$W_1\!\to$ Full"],
        rows,
        col_spec="lcccc",
    )


def _write_cumulative_table(data: pd.DataFrame, ref: pd.Series) -> None:
    rows = [["Atomic", "100", _format_number(ref["w1_atomic"]),
             _format_number(ref["modal_atomic"]), "--", "--"]]
    for row in data.itertuples():
        noun = "answer" if row.prefix == 1 else "answers"
        rows.append(
            [
                f"First {row.prefix} {noun}",
                f"{row.mean_n_values:.0f}",
                _format_number(row.w1_human),
                _format_number(row.modal),
                _format_number(row.w1_vs_atomic),
                _format_number(row.w1_vs_full),
            ]
        )
    rows.append(
        ["Human benchmark", "--", _format_number(0.0),
         _format_number(ref["modal_human"]), "--", "--"]
    )
    _write_tabular(
        TABLES_DIR / "mechanism_cumulative_answer_w1.tex",
        ["Distribution", "Mean $n$", r"$W_1\!\to$ Human", "Modal share",
         r"$W_1\!\to$ Atomic", r"$W_1\!\to$ Full"],
        rows,
        col_spec="lrcccc",
    )


def _save(fig, filename: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(FIGS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _first_answer_figure(ref: pd.Series) -> None:
    labels = ["Atomic", "Chunk:\nfirst answer", "Chunk:\nfull"]
    colors = [ATOMIC_COLOR, FIRST_COLOR, CHUNK_COLOR]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, values, title in (
        (axes[0], [ref["w1_atomic"], ref["w1_first"], ref["w1_full"]],
         r"Mean $W_1$ distance to human benchmark"),
        (axes[1], [ref["modal_atomic"], ref["modal_first"], ref["modal_full"]],
         "Mean modal-mass share"),
    ):
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=_TITLE_FS)
        ax.tick_params(labelsize=_TICK_FS)
        ax.set_ylim(0, max(values) * 1.18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=_LABEL_FS)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].axhline(ref["modal_human"], ls="--", color="black", linewidth=1.2,
                    label=f"Human ({ref['modal_human']:.3f})")
    axes[1].legend(fontsize=_LEGEND_FS, frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, "mechanism_first_answer.png")


def _figure_legend(fig, axes) -> None:
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.02, 0.9, 0.96, 0.09),
               mode="expand", ncol=6, fontsize=13, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.76))


def _trajectory_figure(data: pd.DataFrame, ref: pd.Series, *, cumulative: bool) -> None:
    x_name = "prefix" if cumulative else "answer"
    x = data[x_name].to_numpy()
    series_label = "First k answers" if cumulative else "Answer number"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    axes[0].plot(x, data["w1_human"], "-o", color=FIRST_COLOR, linewidth=2.2,
                 label=series_label)
    axes[0].axhline(ref["w1_atomic"], ls="--", color=ATOMIC_COLOR, label="Atomic")
    axes[0].axhline(ref["w1_full"], ls="--", color=CHUNK_COLOR, label="Chunk full")
    axes[0].axhline(0, ls=":", color="black", linewidth=1.1, label="Human")
    axes[0].set_title(r"$W_1$ to human", fontsize=_TITLE_FS)
    axes[0].set_ylabel(r"Mean $W_1$", fontsize=_LABEL_FS)

    axes[1].plot(x, data["modal"], "-o", color=FIRST_COLOR, linewidth=2.2,
                 label=series_label)
    axes[1].axhline(ref["modal_atomic"], ls="--", color=ATOMIC_COLOR,
                    label="Atomic")
    axes[1].axhline(ref["modal_full"], ls="--", color=CHUNK_COLOR,
                    label="Chunk full")
    axes[1].axhline(ref["modal_human"], ls=":", color="black", linewidth=1.1,
                    label="Human")
    axes[1].set_title("Modal-mass share", fontsize=_TITLE_FS)
    axes[1].set_ylabel("Mean modal share", fontsize=_LABEL_FS)

    axes[2].plot(x, data["w1_vs_atomic"], "-o", color=ATOMIC_COLOR, linewidth=2.2,
                 label=r"$W_1$ to atomic")
    axes[2].plot(x, data["w1_vs_full"], "-o", color=CHUNK_COLOR, linewidth=2.2,
                 label=r"$W_1$ to full chunk")
    axes[2].set_title("Distance to references", fontsize=_TITLE_FS)
    axes[2].set_ylabel(r"Mean $W_1$", fontsize=_LABEL_FS)

    x_label = "First k answers included" if cumulative else "Answer number"
    for ax in axes:
        ax.set_xlabel(x_label, fontsize=_LABEL_FS)
        ax.set_xticks(x)
        ax.tick_params(labelsize=_TICK_FS)
        ax.spines[["top", "right"]].set_visible(False)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0, ymin), ymax)
    _figure_legend(fig, axes)
    filename = ("mechanism_cumulative_answer_w1.png" if cumulative
                else "mechanism_position_answer_w1.png")
    _save(fig, filename)


def generate_mechanism_outputs(db) -> list[str]:
    """Generate the five mechanism artifacts selected in HD-0001."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    reference, positions, cumulative = _mechanism_aggregates(_decision_frame(db), db)
    _write_position_table(positions, reference)
    _write_cumulative_table(cumulative, reference)
    _first_answer_figure(reference)
    _trajectory_figure(positions, reference, cumulative=False)
    _trajectory_figure(cumulative, reference, cumulative=True)
    return [
        "mechanism_first_answer.png",
        "mechanism_position_answer_w1.tex",
        "mechanism_position_answer_w1.png",
        "mechanism_cumulative_answer_w1.tex",
        "mechanism_cumulative_answer_w1.png",
    ]
