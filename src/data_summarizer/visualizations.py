import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import scipy.stats as stats

COLORS = px.colors.qualitative.Plotly  # Using Plotly's default qualitative color set

# Molandic color palette for KS summary visualization
KS_COLORS = {
    "chunk": "#7eb8a2",  # Sage green
    "atomic": "#e8a87c",  # Warm coral
    # Keep old keys for backward compatibility during transition
    "batch": "#7eb8a2",  # Sage green
    "iterative": "#e8a87c",  # Warm coral
    "accent": "#85677b",  # Dusty mauve
    "neutral": "#c9b1a0",  # Warm taupe
    "background": "#f5f2ef",  # Cream
    "text": "#4a4a4a",  # Soft charcoal
    "grid": "#e0dbd5",  # Light warm gray
}

# Short game names for compact display
SHORT_GAME_NAMES = {
    "Random Number Generation": "Random Number",
    "Trust in CC09 (trustor)": "Trust (Trustor)",
    "Trust in CC09 (trustee)": "Trust (Trustee)",
    "BoS in CDJFR89": "Battle of Sexes",
    "Stag Hunt in CDFR92": "Stag Hunt",
    "Ultimatum Strategy (Proposer)": "Ultimatum (Prop)",
    "Ultimatum Strategy (Responder)": "Ultimatum (Resp)",
    "Linear Public Good": "Linear Public Good",
    "Prisoner's Dilemma": "Prisoner's Dilemma",
    "Dictator": "Dictator",
    "Bomb Risk": "Bomb Risk",
}


def create_ks_summary_figure(df: pd.DataFrame, metric: str = "Unified P Value") -> go.Figure:
    """
    Create a combined 2x2 plotly figure summarizing distributional statistics.

    Args:
        df: DataFrame with columns including 'Mode', 'Game', 'ChunkN',
            'LLM Model', and the specified metric column.
        metric: Column name for the metric to visualize (default: "Unified P Value").

    Returns:
        A plotly Figure with 4 subplots:
        - A: Overall Chunk vs Atomic comparison
        - B: Performance by Economic Game
        - C: Effect of Chunk Size
        - D: Improvement by LLM Model
    """
    # Fallback chain for metric selection
    if metric not in df.columns:
        for fallback in ["Unified P Value", "KS Statistic", "Chi2 Statistic"]:
            if fallback in df.columns:
                metric = fallback
                break
        else:
            metric = "KS Statistic"  # Ultimate fallback
    
    # Filter out rows where the metric is null (for game-specific metrics)
    df = df[df[metric].notna()].copy()
    # Dynamically calculate number of games and models for flexible height
    n_games = df["Game"].nunique() if "Game" in df.columns else 1
    n_models = df["LLM Model"].nunique() if "LLM Model" in df.columns else 1
    max_items = max(n_games, n_models, 6)  # Minimum 6 for reasonable spacing
    dynamic_height = max(1125, 500 + max_items * 56)  # 25% taller

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>A. Overall: Chunk vs. Atomic Elicitation</b>",
            "<b>B. Performance by Economic Game</b>",
            "<b>C. Effect of Chunk Size</b>",
            "<b>D. Performance by LLM Model</b>",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )

    # ============ Plot 1: Main Comparison ============
    available_modes = [m for m in ["Atomic", "Chunk"] if m in df["Mode"].unique()]
    mode_data = (
        df.groupby("Mode")[metric].agg(["mean", "std"]).reindex(available_modes)
    )

    if not mode_data.empty and mode_data["mean"].notna().any():
        mode_labels = []
        mode_colors = []
        for m in available_modes:
            if m == "Atomic":
                mode_labels.append("Atomic<br>(Baseline)")
                mode_colors.append(KS_COLORS["atomic"])
            else:
                mode_labels.append("Chunk<br>(Proposed)")
                mode_colors.append(KS_COLORS["chunk"])

        fig.add_trace(
            go.Bar(
                x=mode_labels,
                y=mode_data["mean"],
                error_y=dict(
                    type="data",
                    array=mode_data["std"],
                    color=KS_COLORS["text"],
                    thickness=1.5,
                ),
                marker_color=mode_colors,
                marker_line=dict(width=0),
                text=[f"{v:.3f}" for v in mode_data["mean"]],
                textposition="outside",
                textfont=dict(size=12, color=KS_COLORS["text"]),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ============ Plot 2: By Game ============
    game_data = df.groupby(["Game", "Mode"])[metric].mean().unstack()
    # Sort by difference (Atomic - Chunk) if both available
    if "Atomic" in game_data.columns and "Chunk" in game_data.columns:
        game_data["_diff"] = game_data["Atomic"] - game_data["Chunk"]
        game_data = game_data.sort_values("_diff")
        game_data = game_data.drop(columns=["_diff"])
    elif "Chunk" in game_data.columns:
        game_data = game_data.sort_values("Chunk")
    game_labels = [SHORT_GAME_NAMES.get(g, g) for g in game_data.index]

    if "Atomic" in game_data.columns:
        fig.add_trace(
            go.Bar(
                y=game_labels,
                x=game_data["Atomic"],
                orientation="h",
                name="Atomic",
                marker_color=KS_COLORS["atomic"],
                marker_line=dict(width=0),
                showlegend=True,
                legendgroup="mode",
            ),
            row=1,
            col=2,
        )

    if "Chunk" in game_data.columns:
        fig.add_trace(
            go.Bar(
                y=game_labels,
                x=game_data["Chunk"],
                orientation="h",
                name="Chunk",
                marker_color=KS_COLORS["chunk"],
                marker_line=dict(width=0),
                showlegend=True,
                legendgroup="mode",
            ),
            row=1,
            col=2,
        )

    # ============ Plot 3: By Chunk Size (Categorical) ============
    chunk_size_data = df.groupby("ChunkN")[metric].agg(["mean", "std"])
    chunk_size_data = chunk_size_data.sort_index()

    # Create categorical labels
    chunk_labels = []
    chunk_colors = []
    for cn in chunk_size_data.index:
        if cn == 1:
            chunk_labels.append("1 (Atomic)")
            chunk_colors.append(KS_COLORS["atomic"])
        else:
            chunk_labels.append(str(int(cn)))
            chunk_colors.append(KS_COLORS["chunk"])

    fig.add_trace(
        go.Bar(
            x=chunk_labels,
            y=chunk_size_data["mean"],
            error_y=dict(
                type="data",
                array=chunk_size_data["std"],
                color=KS_COLORS["text"],
                thickness=1.5,
            ),
            marker_color=chunk_colors,
            marker_line=dict(width=0),
            text=[f"{v:.3f}" for v in chunk_size_data["mean"]],
            textposition="outside",
            textfont=dict(size=11, color=KS_COLORS["text"]),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Add baseline reference line if ChunkN=1 exists
    if 1 in chunk_size_data.index:
        fig.add_hline(
            y=chunk_size_data.loc[1, "mean"],
            line_dash="dash",
            line_color=KS_COLORS["atomic"],
            opacity=0.7,
            annotation_text="Atomic baseline",
            annotation_position="top right",
            annotation_font=dict(size=10, color=KS_COLORS["atomic"]),
            row=2,
            col=1,
        )

    # ============ Plot 4: By Model ============
    model_data = df.groupby(["LLM Model", "Mode"])[metric].mean().unstack()
    # Sort by difference (Atomic - Chunk) if both available
    if "Atomic" in model_data.columns and "Chunk" in model_data.columns:
        model_data["_diff"] = model_data["Atomic"] - model_data["Chunk"]
        model_data = model_data.sort_values("_diff")
        model_data = model_data.drop(columns=["_diff"])
    elif "Chunk" in model_data.columns:
        model_data = model_data.sort_values("Chunk")

    # Shorten model names for all models
    short_model_names = [m.split("/")[-1] for m in model_data.index]

    if "Atomic" in model_data.columns:
        fig.add_trace(
            go.Bar(
                y=short_model_names,
                x=model_data["Atomic"],
                orientation="h",
                name="Atomic",
                marker_color=KS_COLORS["atomic"],
                marker_line=dict(width=0),
                showlegend=False,
                legendgroup="mode",
            ),
            row=2,
            col=2,
        )

    if "Chunk" in model_data.columns:
        fig.add_trace(
            go.Bar(
                y=short_model_names,
                x=model_data["Chunk"],
                orientation="h",
                name="Chunk",
                marker_color=KS_COLORS["chunk"],
                marker_line=dict(width=0),
                showlegend=False,
                legendgroup="mode",
            ),
            row=2,
            col=2,
        )

    # ============ Layout Styling ============
    fig.update_layout(
        height=dynamic_height,
        plot_bgcolor=KS_COLORS["background"],
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=KS_COLORS["text"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.52,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=KS_COLORS["text"],
            borderwidth=1,
            font=dict(size=12, color=KS_COLORS["text"]),
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        title=dict(
            text=f"<b>LLM Behavioral Simulation: Chunk vs. Atomic Elicitation ({metric})</b>",
            font=dict(size=18, color=KS_COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        barmode="group",
    )

    # Update axes styling with darker tick colors
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                gridcolor=KS_COLORS["grid"],
                linecolor=KS_COLORS["grid"],
                tickfont=dict(size=10, color=KS_COLORS["text"]),
                row=i,
                col=j,
            )
            fig.update_yaxes(
                gridcolor=KS_COLORS["grid"],
                linecolor=KS_COLORS["grid"],
                tickfont=dict(size=11, color=KS_COLORS["text"]),
                row=i,
                col=j,
            )

    # Specific axis labels with proper title styling
    axis_title_font = dict(size=12, color=KS_COLORS["text"])
    fig.update_yaxes(
        title_text=f"Mean {metric}", title_font=axis_title_font, row=1, col=1
    )
    fig.update_xaxes(
        title_text=f"Mean {metric}", title_font=axis_title_font, row=1, col=2
    )
    fig.update_yaxes(
        title_text=f"Mean {metric}", title_font=axis_title_font, row=2, col=1
    )
    fig.update_xaxes(
        title_text="Chunk Size (N)", title_font=axis_title_font, row=2, col=1
    )
    fig.update_xaxes(
        title_text=f"Mean {metric}", title_font=axis_title_font, row=2, col=2
    )

    # Categorical x-axis for batch size
    fig.update_xaxes(type="category", row=2, col=1)

    # Dynamic axis ranges based on data (no cap for metrics like Wasserstein)
    max_val = df[metric].max() if metric in df.columns else 1.0
    y_range_max = max_val * 1.3  # Add 30% headroom
    x_range_max = max_val * 1.2

    fig.update_yaxes(range=[0, y_range_max], row=1, col=1)
    fig.update_xaxes(range=[0, x_range_max], row=1, col=2)
    fig.update_yaxes(range=[0, y_range_max], row=2, col=1)
    fig.update_xaxes(range=[0, x_range_max], row=2, col=2)

    return fig


def create_ks_distribution_figure(df: pd.DataFrame, metric: str = "Unified P Value") -> go.Figure:
    """
    Create a plotly figure showing distributional analysis of statistics.
    
    Layout:
    - Row 1: Histogram (full width, spanning both columns)
    - Row 2: A. Overall Distribution, B. Distribution by Game
    - Row 3: C. Effect of Batch Size, D. Distribution by Model

    Args:
        df: DataFrame with columns including 'Mode', 'Game', 'ChunkN',
            'LLM Model', and the specified metric column.
        metric: Column name for the metric to visualize (default: "Unified P Value").

    Returns:
        A plotly Figure with 5 subplots showing histogram, violin plots, and box plots.
    """
    # Fallback chain for metric selection
    if metric not in df.columns:
        for fallback in ["Unified P Value", "KS Statistic", "Chi2 Statistic"]:
            if fallback in df.columns:
                metric = fallback
                break
        else:
            metric = "KS Statistic"  # Ultimate fallback
    
    # Filter out rows where the metric is null (for game-specific metrics)
    df = df[df[metric].notna()].copy()
    # Add short game names
    df["Game_Short"] = df["Game"].map(lambda g: SHORT_GAME_NAMES.get(g, g))

    # Dynamically calculate height based on content
    n_games = df["Game"].nunique() if "Game" in df.columns else 1
    n_models = df["LLM Model"].nunique() if "LLM Model" in df.columns else 1
    dynamic_height = max(1200, 1000 + max(n_games, n_models) * 20)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "",  # No title for the top histogram
            "",  # Empty title for merged cell
            f"<b>A. Overall {metric} Distribution (Violin + Box)</b>",
            "<b>B. Distribution by Economic Game</b>",
            "<b>C. Effect of Chunk Size</b>",
            "<b>D. Distribution by Model</b>",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
        row_heights=[0.25, 0.38, 0.37],
        specs=[
            [{"colspan": 2}, None],  # Row 1: Histogram spans both columns
            [{}, {}],                 # Row 2: Two plots
            [{}, {}],                 # Row 3: Two plots
        ],
    )

    # ============ Row 1: Histogram (Full Width) ============
    for mode in ["Atomic", "Chunk"]:
        if mode not in df["Mode"].unique():
            continue
        mode_data = df[df["Mode"] == mode][metric]
        color = KS_COLORS["atomic"] if mode == "Atomic" else KS_COLORS["chunk"]

        fig.add_trace(
            go.Histogram(
                x=mode_data,
                name=mode,
                marker_color=color,
                opacity=0.6,
                nbinsx=30,
                showlegend=True,
                legendgroup="mode",
            ),
            row=1,
            col=1,
        )

    # ============ Row 2, Col 1: Violin + Box (Overall) ============
    for mode in ["Atomic", "Chunk"]:
        if mode not in df["Mode"].unique():
            continue
        mode_data = df[df["Mode"] == mode][metric]
        color = KS_COLORS["atomic"] if mode == "Atomic" else KS_COLORS["chunk"]

        fig.add_trace(
            go.Violin(
                x=[mode] * len(mode_data),
                y=mode_data,
                name=mode,
                side="both",
                meanline_visible=True,
                fillcolor=color,
                opacity=0.6,
                line_color=color,
                showlegend=False,
                box_visible=True,
                points="all",
                pointpos=0,
                jitter=0.3,
                marker=dict(size=3, color=color, opacity=0.4),
            ),
            row=2,
            col=1,
        )

    # Add mean annotations for overall plot
    for mode in ["Atomic", "Chunk"]:
        if mode not in df["Mode"].unique():
            continue
        mean_val = df[df["Mode"] == mode][metric].mean()
        fig.add_annotation(
            x=mode,
            y=mean_val + 0.05,
            text=f"μ={mean_val:.3f}",
            showarrow=False,
            font=dict(size=10, color=KS_COLORS["text"]),
            xref="x2",
            yref="y2",
        )

    # ============ Row 2, Col 2: Box plots by Game ============
    game_stats = df.groupby(["Game_Short", "Mode"])[metric].mean().unstack()
    if "Atomic" in game_stats.columns and "Chunk" in game_stats.columns:
        game_stats["_diff"] = game_stats["Atomic"] - game_stats["Chunk"]
        game_order = game_stats.sort_values("_diff").index.tolist()
    else:
        game_order = df.groupby("Game_Short")[metric].mean().sort_values().index.tolist()

    for mode in ["Atomic", "Chunk"]:
        if mode not in df["Mode"].unique():
            continue
        mode_df = df[df["Mode"] == mode]
        color = KS_COLORS["atomic"] if mode == "Atomic" else KS_COLORS["chunk"]

        fig.add_trace(
            go.Box(
                x=mode_df["Game_Short"],
                y=mode_df[metric],
                name=mode,
                marker_color=color,
                fillcolor=color,
                opacity=0.7,
                boxmean=True,
                showlegend=False,
                legendgroup="mode",
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(categoryorder="array", categoryarray=game_order, row=2, col=2)

    # ============ Row 3, Col 1: Violin by Chunk Size ============
    chunk_values = sorted(df["ChunkN"].unique())
    chunk_labels = {1: "1 (Atomic)"}
    for cn in chunk_values:
        if cn != 1:
            chunk_labels[cn] = str(int(cn))

    for chunk_n in chunk_values:
        chunk_data = df[df["ChunkN"] == chunk_n][metric]
        color = KS_COLORS["atomic"] if chunk_n == 1 else KS_COLORS["chunk"]
        label = chunk_labels.get(chunk_n, str(int(chunk_n)))

        fig.add_trace(
            go.Violin(
                x=[label] * len(chunk_data),
                y=chunk_data,
                name=label,
                fillcolor=color,
                opacity=0.6,
                line_color=color,
                meanline_visible=True,
                box_visible=True,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_xaxes(
        categoryorder="array", categoryarray=list(chunk_labels.values()), row=3, col=1
    )

    # ============ Row 3, Col 2: Box by Model ============
    model_stats = df.groupby(["LLM Model", "Mode"])[metric].mean().unstack()
    if "Atomic" in model_stats.columns and "Chunk" in model_stats.columns:
        model_stats["_diff"] = model_stats["Atomic"] - model_stats["Chunk"]
        model_order = model_stats.sort_values("_diff").index.tolist()
    else:
        model_order = df.groupby("LLM Model")[metric].mean().sort_values().index.tolist()
    short_model_order = [m.split("/")[-1] for m in model_order]

    for mode in ["Atomic", "Chunk"]:
        if mode not in df["Mode"].unique():
            continue
        mode_df = df[df["Mode"] == mode].copy()
        mode_df["Model_Short"] = mode_df["LLM Model"].apply(lambda x: x.split("/")[-1])
        color = KS_COLORS["atomic"] if mode == "Atomic" else KS_COLORS["chunk"]

        fig.add_trace(
            go.Box(
                x=mode_df["Model_Short"],
                y=mode_df[metric],
                name=mode,
                marker_color=color,
                fillcolor=color,
                opacity=0.7,
                boxmean=True,
                showlegend=False,
            ),
            row=3,
            col=2,
        )

    fig.update_xaxes(
        categoryorder="array", categoryarray=short_model_order, row=3, col=2
    )

    # ============ Layout Styling ============
    fig.update_layout(
        height=dynamic_height,
        plot_bgcolor=KS_COLORS["background"],
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color=KS_COLORS["text"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=KS_COLORS["text"],
            borderwidth=1,
            font=dict(size=12, color=KS_COLORS["text"]),
        ),
        margin=dict(l=70, r=50, t=100, b=70),
        title=dict(text=""),
        boxmode="group",
        boxgap=0.3,
        boxgroupgap=0.2,
        barmode="overlay",
    )

    # Update axes styling
    for i in range(1, 4):
        for j in range(1, 3):
            # Skip the merged cell position
            if i == 1 and j == 2:
                continue
            fig.update_xaxes(
                gridcolor=KS_COLORS["grid"],
                linecolor=KS_COLORS["grid"],
                tickfont=dict(size=10, color=KS_COLORS["text"]),
                tickangle=-45 if (i == 2 and j == 2) or (i == 3 and j == 2) else 0,
                row=i,
                col=j,
            )
            fig.update_yaxes(
                gridcolor=KS_COLORS["grid"],
                linecolor=KS_COLORS["grid"],
                tickfont=dict(size=11, color=KS_COLORS["text"]),
                row=i,
                col=j,
            )

    # Axis labels
    axis_title_font = dict(size=12, color=KS_COLORS["text"])
    # Row 1: Histogram
    fig.update_xaxes(title_text=metric, title_font=axis_title_font, row=1, col=1)
    fig.update_yaxes(title_text="Count", title_font=axis_title_font, row=1, col=1)
    # Row 2
    fig.update_yaxes(title_text=metric, title_font=axis_title_font, row=2, col=1)
    fig.update_yaxes(title_text=metric, title_font=axis_title_font, row=2, col=2)
    # Row 3
    fig.update_xaxes(title_text="Chunk Size", title_font=axis_title_font, row=3, col=1)
    fig.update_yaxes(title_text=metric, title_font=axis_title_font, row=3, col=1)
    fig.update_yaxes(title_text=metric, title_font=axis_title_font, row=3, col=2)

    # Set y-axis ranges dynamically
    max_val = df[metric].max() if metric in df.columns else 1.0
    y_range_max = max_val * 1.1

    fig.update_yaxes(range=[0, y_range_max], row=2, col=1)
    fig.update_yaxes(range=[0, y_range_max], row=2, col=2)
    fig.update_yaxes(range=[0, y_range_max], row=3, col=1)
    fig.update_yaxes(range=[0, y_range_max], row=3, col=2)

    return fig


def create_histogram(
    df: pd.DataFrame,
    is_discrete: bool,
    x_range: Union[Tuple[int, int], List[Union[int, str]]],
) -> go.Figure:
    fig = go.Figure()
    if is_discrete:
        # Use x_range if provided, otherwise use unique values from data
        if isinstance(x_range, list):
            unique_values = sorted(x_range)
        else:
            unique_values = sorted(df["Decision Value"].unique())

        for simulation in df["Simulation"].unique():
            simulation_data = df[df["Simulation"] == simulation]
            counts = (
                simulation_data["Decision Value"].value_counts(normalize=True) * 100
            )
            fig.add_trace(
                go.Bar(
                    x=unique_values,
                    y=[counts.get(val, 0) for val in unique_values],
                    name=simulation,
                    marker_color=COLORS[
                        list(df["Simulation"].unique()).index(simulation) % len(COLORS)
                    ],
                    opacity=0.3,
                    hovertemplate="<br>".join(
                        [
                            "Simulation: %{fullData.name}",
                            "Decision Value: %{x}",
                            "Percentage: %{y:.2f}%",
                            "<extra></extra>",
                        ]
                    ),
                )
            )

        # Set x-axis for discrete data
        fig.update_xaxes(
            tickmode="array",
            tickvals=unique_values,
            ticktext=[str(val) for val in unique_values],
            range=(
                [min(unique_values) - 0.5, max(unique_values) + 0.5]
                if all(isinstance(x, (int, float)) for x in unique_values)
                else None
            ),
        )
    else:
        min_val, max_val = x_range
        bin_edges = np.linspace(min_val, max_val, 11)  # 11 edges to create 10 bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for simulation in df["Simulation"].unique():
            simulation_data = df[df["Simulation"] == simulation]["Decision Value"]
            hist, _ = np.histogram(simulation_data, bins=bin_edges, density=True)
            hist = hist / hist.sum() * 100  # Convert to percentages
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist,
                    name=simulation,
                    marker_color=COLORS[
                        list(df["Simulation"].unique()).index(simulation) % len(COLORS)
                    ],
                    opacity=0.3,
                    hovertemplate="<br>".join(
                        [
                            "Simulation: %{fullData.name}",
                            "Range: [%{customdata[0]:.2f}, %{customdata[1]:.2f})"
                            + (
                                "<br>(Includes upper bound)"
                                if "%{customdata[2]}" == "1"
                                else ""
                            ),
                            "Percentage: %{y:.2f}%",
                            "<extra></extra>",
                        ]
                    ),
                    customdata=np.column_stack(
                        (
                            bin_edges[:-1],
                            bin_edges[1:],
                            np.arange(len(bin_edges) - 1)
                            == len(bin_edges) - 2,  # Flag for last bin
                        )
                    ),
                    visible="legendonly",
                )
            )
        fig.update_xaxes(
            tickvals=bin_edges, ticktext=[f"{val:.1f}" for val in bin_edges]
        )

    fig.update_layout(
        barmode="group" if is_discrete else "overlay",
        bargap=0.15,
        bargroupgap=0.1,
        xaxis_title="Decision Value",
        yaxis_title="Percent",
    )

    return fig


def add_density_curve(
    fig: go.Figure, df: pd.DataFrame, simulation_name: str, color_index: int
):
    try:
        filtered_df = df[df["Simulation"] == simulation_name]
        density = stats.gaussian_kde(filtered_df["Decision Value"])
        min_val = filtered_df["Decision Value"].min()
        max_val = filtered_df["Decision Value"].max()
        x = np.linspace(min_val, max_val, 100)

        y = density(x) * 100
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{simulation_name} Density",
                line=dict(color=COLORS[color_index % len(COLORS)]),
                visible="legendonly" if min_val == max_val else None,
            )
        )
    except Exception as e:
        print(f"Error adding density curve for {simulation_name}: {str(e)}")
