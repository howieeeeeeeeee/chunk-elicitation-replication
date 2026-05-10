from __future__ import annotations

from pathlib import Path

from .local_json_db import CombinedJsonDatabase, LocalJsonDatabase


REPLICATION_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPLICATION_ROOT / "data"


def get_database(
    target: str = "local_json",
    db_name: str | None = None,
    *,
    data_root: str | Path | None = None,
    experiment_name: str = "exp1",
):
    if target not in {"local", "local_json"}:
        raise ValueError("Replication bundle only supports target='local_json'.")
    root = Path(data_root) if data_root else DATA_ROOT
    return LocalJsonDatabase(
        root / experiment_name,
        benchmark_dir=root / "benchmark",
    )


def get_combined_database(
    *,
    data_root: str | Path | None = None,
    experiment_names: tuple[str, ...] = ("exp1", "exp2", "exp3"),
):
    root = Path(data_root) if data_root else DATA_ROOT
    return CombinedJsonDatabase(
        [root / name for name in experiment_names],
        benchmark_dir=root / "benchmark",
    )
