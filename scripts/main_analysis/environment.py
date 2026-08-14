"""Repository paths and database factory for the replication analysis pipeline."""

from __future__ import annotations

from pathlib import Path

from db_ops.config import get_combined_database


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TEX_DIR = REPO_ROOT / "tex"
TABLES_DIR = TEX_DIR / "tables"
FIGS_DIR = TEX_DIR / "figs"
SRC_FIGS_DIR = REPO_ROOT / "output" / "figures"
MAIN_ANALYSIS_OUTPUT_DIR = REPO_ROOT / "output" / "main_analysis"


def get_analysis_database():
    """Return the combined local-JSON database used by the replication bundle."""
    return get_combined_database(
        data_root=DATA_DIR,
        experiment_names=("exp1", "exp2", "exp3"),
    )
