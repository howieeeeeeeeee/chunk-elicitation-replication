from db_ops.retrievers import (
    get_simulation,
    get_simulation_results,
    get_benchmark_results,
    get_all_simulation_results,
)
from .visualizations import create_histogram, add_density_curve
from analysis.perform_test import (
    perform_ks_test,
    perform_chi_square_test,
    perform_wasserstein_1,
)
from games.instructions import (
    GAME_DECISION_ARRAY_DESCRIPTION,
    GAME_DECISION_ARRAY_CONFIG,
)
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from typing import List, Dict, Any, Tuple, Optional
# P-value metrics that should show % < 0.1 and % < 0.05 columns
P_VALUE_METRICS = {"KS P Value", "Chi2 P Value"}


def create_grouped_summary(
    df: pd.DataFrame,
    metric: str,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create a grouped summary table for a given metric.

    Args:
        df: DataFrame with simulation data (must contain the metric column)
        metric: Column name of the metric to summarize (e.g., "Wasserstein-1", "KS P Value")
        group_cols: List of columns to group by. If None or empty, returns overall summary.

    Returns:
        DataFrame with summary statistics (N, avg, std, min, p25, p50, p75, p90)
        For p-value metrics, also includes % < 0.1 and % < 0.05.
    """
    is_p_value_metric = metric in P_VALUE_METRICS

    def _series_stats(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").dropna()

        def _r4(x: float) -> float:
            return round(float(x), 4)

        def _r2(x: float) -> float:
            return round(float(x), 2)

        if s.empty:
            base_stats = {"N": 0}
            if is_p_value_metric:
                base_stats["% < 0.1"] = None
                base_stats["% < 0.05"] = None
            base_stats.update({
                "avg": None,
                "std": None,
                "min": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p90": None,
            })
            return pd.Series(base_stats)

        # Build stats dict with conditional p-value percentage columns
        stats = {"N": int(s.size)}

        # Add p-value threshold columns only for p-value metrics
        if is_p_value_metric:
            pct_lt_01 = (s < 0.1).sum() / s.size * 100
            pct_lt_005 = (s < 0.05).sum() / s.size * 100
            stats["% < 0.1"] = _r2(pct_lt_01)
            stats["% < 0.05"] = _r2(pct_lt_005)

        # Add standard stats
        stats.update({
            "avg": _r4(s.mean()),
            "std": _r4(s.std()) if s.size > 1 else 0.0,
            "min": _r4(s.min()),
            "p25": _r4(s.quantile(0.25)),
            "p50": _r4(s.quantile(0.50)),
            "p75": _r4(s.quantile(0.75)),
            "p90": _r4(s.quantile(0.90)),
        })

        return pd.Series(stats)

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
        summary = grouped[metric].apply(_series_stats).unstack().reset_index()
    else:
        summary = _series_stats(df[metric]).to_frame().T

    return summary


def _normalize_extra_flag(extra_flag: Any) -> List[Any]:
    if extra_flag is None:
        return []
    if isinstance(extra_flag, list):
        return list(extra_flag)
    if isinstance(extra_flag, tuple):
        return list(extra_flag)
    return [extra_flag]


def _format_extra_flag(extra_flag: Any) -> str:
    return json.dumps(_normalize_extra_flag(extra_flag))


def get_simulation_info(simulation):
    # Map simulation_mode to display mode: Iterative -> Atomic, Batch -> Chunk
    simulation_mode = simulation["simulation_config"]["simulation_mode"]
    mode_display = simulation_mode.replace("Simulate", "")
    if mode_display == "Iterative":
        mode_display = "Atomic"
    elif mode_display == "Batch":
        mode_display = "Chunk"
    
    dic = {
        "_id": simulation["_id"],
        "Name": simulation["name"],
        "Game": simulation["simulation_config"]["game_type"],
        "Mode": mode_display,
        "ChunkN": (
            simulation["simulation_config"].get("batch_simulation_n", 1)
            if simulation["simulation_config"]["simulation_mode"] == "BatchSimulate"
            else 1
        ),
        "Explain Reasoning": simulation["instruction_config"]["explain_reasoning"],
        "Reasoning Mode": simulation["instruction_config"].get(
            "explain_reasoning_mode", "basic"
        ),
        "SplitN": simulation["instruction_config"].get("split_n"),
        "LLM Model": simulation["llm_config"]["model"],
        "Temperature": simulation["llm_config"].get("temperature"),
        "Enabled Reasoning": simulation["llm_config"].get("reasoning_enabled", False),
        "Completed": simulation["completed"] == True,
        "updatedTime": simulation["updated_at"],
        "Context": simulation["instruction_config"].get("context", ""),
        "Incentive Size": simulation["instruction_config"].get("incentive_size", ""),
        "Privacy Treatment": simulation["instruction_config"].get(
            "privacy_treatment", ""
        ),
        "Theoretical Prediction": simulation["instruction_config"][
            "theoretical_prediction"
        ],
        "Focal Point": simulation["instruction_config"].get("focal_point", False),
        "Extra Flag": _format_extra_flag(simulation.get("extraFlag")),
    }
    return dic


def show_all_simulations_df(
    _db,
    filter_incomplete=True,
    phase_name="phase_1",
    game_type=None,
    columns=(),
    simulation_ids=(),
):
    """Load simulations from the local JSON database.

    Always omits archived rows (``archived: False``). When
    ``filter_incomplete`` is True (default), only completed simulations are
    returned (``completed: True``); set ``filter_incomplete=False`` only when
    you intentionally need incomplete runs (e.g.\ admin UI).
    """

    filter: dict = {"archived": False}
    if filter_incomplete:
        filter["completed"] = True
    if phase_name:
        filter["phase_name"] = phase_name
    if game_type:
        filter["simulation_config.game_type"] = game_type
    if simulation_ids:
        filter["_id"] = {"$in": list(simulation_ids)}
    ls = []
    for simulation in _db.simulations.find(filter):
        dic = get_simulation_info(simulation)
        ls.append(dic)

    df = pd.DataFrame(ls)

    if columns:
        columns = columns + ["_id"] if "_id" not in columns else columns
        # Ensure all columns exist in the DataFrame, if not fill with NaN
        for col in columns:
            if col not in df.columns:
                df[col] = None  # or use pd.NA for missing values
        df = df[columns]

    return df.set_index("_id")


def attach_ks_test_results_to_simulations_df(
    _db,
    simulations_df: pd.DataFrame,
    decision_index: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Attach statistical test results (vs benchmarks) to a simulations DataFrame.

    Assumes `simulations_df` is indexed by simulation_id and contains a `Game` column.
    Uses GAME_DECISION_ARRAY_CONFIG to determine if a game is discrete or continuous:
    - Discrete games: Chi-Square test
    - Continuous games: KS test, Wasserstein distance

    Computes:
    - Unified P Value: Always populated (Chi-Square P for discrete, KS P for continuous)
    - Test Type: "Chi-Square" or "KS"
    - KS Statistic / P Value: Standard KS test (continuous only)
    - KS Statistic (Std) / P Value (Std): KS on standardized data (continuous only)
    - Wasserstein: Earth Mover's Distance (continuous only)
    - Chi2 Statistic / P Value: Chi-Square test (discrete only)
    """
    empty_cols = [
        "N",
        "Wasserstein-1",  # New unified metric (always populated)
        "Test Type",
        "KS Statistic",
        "KS P Value",
        "KS Reject Null?",
        "KS Alpha",
        "Chi2 Statistic",
        "Chi2 P Value",
        "Chi2 Reject Null?",
        # Deprecated columns (kept for backward compatibility, always None)
        "Unified P Value",
        "KS Statistic (Std)",
        "KS P Value (Std)",
        "KS Reject Null? (Std)",
        "Wasserstein",
        "Wasserstein (Std)",
    ]
    if simulations_df is None or simulations_df.empty:
        out = simulations_df.copy() if simulations_df is not None else pd.DataFrame()
        for col in empty_cols:
            if col not in out.columns:
                out[col] = None
        return out

    if "Game" not in simulations_df.columns:
        raise ValueError("`simulations_df` must contain a `Game` column.")

    sim_ids = simulations_df.index.tolist()
    all_results = get_all_simulation_results(_db=_db, simulation_ids=tuple(sim_ids))

    benchmark_cache = {}  # (game_type, decision_index) -> processed benchmark decisions

    # Sample size
    sample_n = []
    # Wasserstein-1 (always populated - unified metric)
    wasserstein_1 = []
    test_type = []
    # Standard KS metrics (continuous only)
    ks_stat = []
    ks_p = []
    ks_reject = []
    ks_alpha = []
    # Chi-Square metrics (discrete only)
    chi2_stat = []
    chi2_p = []
    chi2_reject = []

    for sim_id in sim_ids:
        game_type = simulations_df.loc[sim_id, "Game"]
        decisions = all_results.get(sim_id, [])

        # Determine if this game has discrete decisions
        game_config = GAME_DECISION_ARRAY_CONFIG.get(game_type, [])
        is_discrete = (
            game_config[decision_index][1]
            if game_config and len(game_config) > decision_index
            else False
        )

        processed = process_decisions(decisions, decision_index) if decisions else []
        if not processed:
            sample_n.append(0)
            wasserstein_1.append(None)
            test_type.append("Chi-Square" if is_discrete else "KS")
            ks_stat.append(None)
            ks_p.append(None)
            ks_reject.append(None)
            ks_alpha.append(alpha)
            chi2_stat.append(None)
            chi2_p.append(None)
            chi2_reject.append(None)
            continue

        cache_key = (game_type, decision_index)
        if cache_key not in benchmark_cache:
            benchmark_decisions = get_benchmark_results(_db=_db, game_type=game_type)
            benchmark_cache[cache_key] = (
                process_decisions(benchmark_decisions, decision_index)
                if benchmark_decisions
                else []
            )

        benchmark_processed = benchmark_cache[cache_key]
        if not benchmark_processed:
            sample_n.append(len(processed))
            wasserstein_1.append(None)
            test_type.append("Chi-Square" if is_discrete else "KS")
            ks_stat.append(None)
            ks_p.append(None)
            ks_reject.append(None)
            ks_alpha.append(alpha)
            chi2_stat.append(None)
            chi2_p.append(None)
            chi2_reject.append(None)
            continue

        sample_n.append(len(processed))

        # Wasserstein-1 is always calculated (works for both discrete and continuous)
        w1 = perform_wasserstein_1(
            processed, benchmark_processed, game_type, decision_index
        )
        wasserstein_1.append(w1)

        if is_discrete:
            # Chi-Square test for discrete data
            res = perform_chi_square_test(processed, benchmark_processed, alpha=alpha)
            chi2_stat.append(res.get("Statistic"))
            chi2_p.append(res.get("P Value"))
            chi2_reject.append(res.get("Reject Null?"))
            test_type.append("Chi-Square")

            # Leave KS columns as None (not applicable for discrete)
            ks_stat.append(None)
            ks_p.append(None)
            ks_reject.append(None)
            ks_alpha.append(alpha)
        else:
            # KS test for continuous data
            res = perform_ks_test(processed, benchmark_processed, alpha=alpha)
            ks_stat.append(res.get("Statistic"))
            ks_p.append(res.get("P Value"))
            ks_reject.append(res.get("Reject Null?"))
            ks_alpha.append(res.get("Alpha", alpha))
            test_type.append("KS")

            # Leave Chi-Square columns as None (not applicable for continuous)
            chi2_stat.append(None)
            chi2_p.append(None)
            chi2_reject.append(None)

    out = simulations_df.copy()
    out["N"] = sample_n
    out["Wasserstein-1"] = wasserstein_1
    out["Test Type"] = test_type
    out["KS Statistic"] = ks_stat
    out["KS P Value"] = ks_p
    out["KS Reject Null?"] = ks_reject
    out["KS Alpha"] = ks_alpha
    out["Chi2 Statistic"] = chi2_stat
    out["Chi2 P Value"] = chi2_p
    out["Chi2 Reject Null?"] = chi2_reject
    # Deprecated columns (kept for backward compatibility)
    out["Unified P Value"] = None
    out["KS Statistic (Std)"] = None
    out["KS P Value (Std)"] = None
    out["KS Reject Null? (Std)"] = None
    out["Wasserstein"] = None
    out["Wasserstein (Std)"] = None
    return out


def summarize_decisions_for_simulations(
    _db,
    simulation_ids: List[str],
) -> Dict[str, List[Tuple[pd.DataFrame, go.Figure, pd.DataFrame]]]:
    simulations = [
        get_simulation(_db=_db, simulation_id=simulation_id)
        for simulation_id in simulation_ids
    ]

    # Group simulations by game type
    game_type_groups = {}
    for simulation in simulations:
        game_type = simulation["simulation_config"]["game_type"]
        if game_type not in game_type_groups:
            game_type_groups[game_type] = []
        game_type_groups[game_type].append(simulation)

    results_by_game = {}

    for game_type, game_simulations in game_type_groups.items():
        decision_description = GAME_DECISION_ARRAY_DESCRIPTION[game_type]
        game_graph_range = GAME_DECISION_ARRAY_CONFIG[game_type]
        decision_length = game_simulations[0]["simulation_config"].get(
            "decision_length", 1
        )

        data = {}
        benchmarks = get_benchmark_results(_db=_db, game_type=game_type)

        if benchmarks:
            data["Benchmarks"] = {
                "decisions": benchmarks,
                "simulation_name": "Benchmarks",
                "sample_size": len(benchmarks),
                "simulation_info": {},
            }

        for simulation in game_simulations:
            simulation_id = simulation["_id"]
            decisions = get_simulation_results(_db=_db, simulation_id=simulation_id)
            data[simulation_id] = {
                "decisions": decisions,
                "simulation_name": simulation["name"],
                "sample_size": len(decisions),
                "simulation_info": get_simulation_info(simulation),
            }

        game_results = []
        for i in range(decision_length):
            x_range, is_discrete = game_graph_range[i]
            all_decisions = []
            ks_test_results = []
            stats_data = []
            color_index = 0

            for simulation_id, simulation_data in data.items():
                simulation_name = simulation_data["simulation_name"]
                context = simulation_data.get("simulation_info", {}).get("Context", "")
                incentive_size = simulation_data.get("simulation_info", {}).get(
                    "Incentive Size", ""
                )
                privacy_treatment = simulation_data.get("simulation_info", {}).get(
                    "Privacy Treatment", ""
                )
                focal_point = simulation_data.get("simulation_info", {}).get(
                    "Focal Point", False
                )
                mode = simulation_data.get("simulation_info", {}).get("Mode", "")
                model = simulation_data.get("simulation_info", {}).get("LLM Model", "")
                explain_reasoning = simulation_data.get("simulation_info", {}).get(
                    "Explain Reasoning", False
                )
                theoretical_prediction = simulation_data.get("simulation_info", {}).get(
                    "Theoretical Prediction", False
                )

                sample_size = simulation_data["sample_size"]
                decisions = process_decisions(simulation_data["decisions"], i)

                for decision in decisions:
                    all_decisions.append(
                        {
                            "Simulation": f"{simulation_name} (n={sample_size})",
                            "Decision Value": decision,
                        }
                    )
                # Perform appropriate statistical test with benchmarks
                if benchmarks:
                    benchmark_decisions = process_decisions(
                        data["Benchmarks"]["decisions"], i
                    )
                    dic = {
                        "Simulation": simulation_name,
                        "Context": context,
                        "Incentive Size": incentive_size,
                        "Privacy Treatment": privacy_treatment,
                        "Focal Point": focal_point,
                        "Model": model,
                        "Mode": mode,
                        "Explain Reasoning": explain_reasoning,
                        "Theoretical Prediction": theoretical_prediction,
                        "Sample Size": sample_size,
                    }
                    # Use Chi-Square for discrete, KS for continuous
                    if is_discrete:
                        test_res = perform_chi_square_test(
                            decisions, benchmark_decisions
                        )
                        dic["Test Type"] = "Chi-Square"
                    else:
                        test_res = perform_ks_test(decisions, benchmark_decisions)
                        dic["Test Type"] = "KS"
                    dic.update(test_res)
                    ks_test_results.append(dic)

                # Collect descriptive statistics
                decision_series = pd.Series(decisions)
                stats_data.append(
                    {
                        "Simulation": simulation_name,
                        "n": len(decisions),
                        "mean": decision_series.mean(),
                        "std": decision_series.std(),
                        "min": decision_series.min(),
                        "p25": decision_series.quantile(0.25),
                        "p50": decision_series.median(),
                        "p75": decision_series.quantile(0.75),
                        "max": decision_series.max(),
                    }
                )

            df = pd.DataFrame(all_decisions)
            ks_test_results_df = pd.DataFrame(ks_test_results)

            fig = create_histogram(df, is_discrete, x_range)

            if not is_discrete:
                # Add density curves
                for simulation_name in df["Simulation"].unique():
                    add_density_curve(fig, df, simulation_name, color_index)
                    color_index += 1

            fig.update_layout(
                title=f"Distribution of Decision {i + 1} ({decision_description[i]})",
                xaxis_title="Decision Value",
                yaxis_title="Proportion (%)",
            )

            stats_df = pd.DataFrame(stats_data).set_index("Simulation")
            game_results.append((stats_df, fig, ks_test_results_df))

        results_by_game[game_type] = game_results

    return results_by_game


def process_decisions(decisions: List[Any], index: int) -> List[float]:
    processed = []
    for decision in decisions:
        if isinstance(decision, dict):
            value = decision.get("decision", [])
            processed.append(float(value[index]) if index < len(value) else None)
        elif isinstance(decision, list):
            if all(isinstance(item, (int, float)) for item in decision):
                processed.append(
                    float(decision[index]) if index < len(decision) else None
                )
            elif isinstance(decision[0], (list, tuple)):
                processed.append(
                    float(decision[0][index]) if index < len(decision[0]) else None
                )
        elif isinstance(decision, (int, float)):
            processed.append(float(decision) if index == 0 else None)
    return [d for d in processed if d is not None]
