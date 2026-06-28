"""Figure generation for phase_2 analysis outputs."""

from __future__ import annotations

import shutil
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_summarizer.simulations import process_decisions
from data_summarizer.visualizations import SHORT_GAME_NAMES
from db_ops.retrievers import get_all_simulation_results, get_benchmark_results
from games.instructions import BEHAVIOR_GAMES, GAME_DECISION_ARRAY_CONFIG

from .config import (
    ATOMIC_COLOR,
    ATOMIC_EDGE,
    CHUNK_COLOR,
    CHUNK_EDGE,
    FIGURE_FILES,
    GAME_ORDER,
    GROUND_TRUTH_GAMES,
    MODEL_LABEL_ORDER,
    _FIG_ANNOTATION_FS,
    _FIG_AXIS_LABEL_FS,
    _FIG_GRID_LEGEND_FS,
    _FIG_GRID_LEGEND_Y,
    _FIG_GRID_NO_SUPTITLE_RECT,
    _FIG_LEGEND_FS,
    _FIG_RAINCLOUD_LEGEND_FS,
    _FIG_RAINCLOUD_STAT_FS,
    _FIG_RAINCLOUD_STAT_LINESPACING,
    _FIG_RAINCLOUD_STAT_X,
    _FIG_RAINCLOUD_XLIM_HI,
    _FIG_RAINCLOUD_Y_TICK_FS,
    _FIG_TASK12_PANEL_TITLE_FS,
    _FIG_TASK12_TICK_FS,
    _FIG_TICK_FS,
    _FIG_TICK_PARAMS,
    _FIG_TITLE_FS,
    _exclude_models,
    _short_model_name,
)
from .environment import FIGS_DIR, SRC_FIGS_DIR

def _save_figure(fig, filename: str, *, dpi: int = 300) -> Path:
    """Save ``fig`` as ``output/figures/<filename>`` with tight bbox."""
    SRC_FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SRC_FIGS_DIR / filename
    fig.patch.set_facecolor("white")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


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
    np.random.seed(0)
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
