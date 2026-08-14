"""Orchestration layer for the full analysis pipeline."""

from __future__ import annotations

from data_summarizer import (
    attach_ks_test_results_to_simulations_df,
    show_all_simulations_df,
)

from .config import DEFAULT_REGRESSORS
from .environment import FIGS_DIR, TABLES_DIR, TEX_DIR, get_analysis_database
from .figures import generate_figures
from .mechanism import generate_mechanism_outputs
from .regression_tables import write_regression_table
from .summary_tables import (
    filter_behavior,
    write_ground_truth_error_rates,
    write_ground_truth_summary_by_mode,
    write_overall_w1_summary,
    write_random_number_generation_summary,
    write_reasoning_large_chunks,
    write_reasoning_on_off,
)


def run_pipeline() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    db = get_analysis_database()

    print("[1/8] Loading phase_2_context simulations...")
    sims_p2c = show_all_simulations_df(
        _db=db,
        filter_incomplete=True,
        phase_name="phase_2_context",
    )
    print(f"      phase_2_context sims: {len(sims_p2c)}")
    sims_p2c = attach_ks_test_results_to_simulations_df(
        _db=db, simulations_df=sims_p2c, decision_index=0, alpha=0.05
    ).sort_index()
    sims_p2c = sims_p2c[sims_p2c["Extra Flag"].astype(str) == "[]"]
    print(f"      phase_2_context sims (Extra Flag = []): {len(sims_p2c)}")

    print("[2/8] Writing regression table...")
    write_regression_table(
        sims_p2c,
        TABLES_DIR / "regression_w1_phase2_context.tex",
        dependent_var="Wasserstein-1",
        regressors=DEFAULT_REGRESSORS,
    )

    print("[3/8] Loading phase_2 simulations...")
    sims_p2 = show_all_simulations_df(
        _db=db,
        filter_incomplete=True,
        phase_name="phase_2",
    )
    print(f"      phase_2 sims: {len(sims_p2)}")
    sims_p2 = attach_ks_test_results_to_simulations_df(
        _db=db, simulations_df=sims_p2, decision_index=0, alpha=0.05
    ).sort_index()

    print("[4/8] Writing overall W1 summary (phase_2 behavior)...")
    df_behavior = filter_behavior(sims_p2)
    print(f"      behavior sims: {len(df_behavior)}")
    write_overall_w1_summary(
        df_behavior, TABLES_DIR / "w1_overall_summary_phase2.tex"
    )

    print("[5/8] Writing Explain Reasoning on/off tables...")
    write_reasoning_on_off(sims_p2, TABLES_DIR / "w1_reasoning_on_off_phase2.tex")
    write_reasoning_large_chunks(
        sims_p2, TABLES_DIR / "w1_reasoning_on_off_large_chunks_phase2.tex"
    )

    print("[6/8] Writing ground-truth and random-number tables...")
    write_ground_truth_summary_by_mode(
        sims_p2, TABLES_DIR / "w1_summary_ground_truth_by_mode.tex"
    )
    write_ground_truth_error_rates(
        db, sims_p2, TABLES_DIR / "ground_truth_non_zero_error_rates.tex"
    )
    write_random_number_generation_summary(
        sims_p2, TABLES_DIR / "w1_random_number_generation_phase2.tex"
    )
    old_nonzero = TABLES_DIR / "ground_truth_non_zero_w1.tex"
    if old_nonzero.exists():
        old_nonzero.unlink()

    print("[7/8] Generating figures...")
    copied = generate_figures(db, sims_p2)
    print(f"      figures ready: {copied}")
    print(f"      static result.tex retained: {TEX_DIR / 'result.tex'}")

    print("[8/8] Generating selected within-response mechanism outputs...")
    mechanism_outputs = generate_mechanism_outputs(db)
    print(f"      mechanism outputs ready: {mechanism_outputs}")

    print("Done. Artifacts under:", TEX_DIR)


def main() -> None:
    run_pipeline()
