"""CLI and local-JSON execution shared by the experiment entrypoints."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPLICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(REPLICATION_ROOT / ".env")

from simluations.run_simulation import run_simulation_to_json

from plan import build_plan, manifest_digest


logger = logging.getLogger("formal_replication_runner")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_existing_records(data_root: Path, experiment_key: str) -> list[dict]:
    path = data_root / experiment_key / "simulations.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")
    return records


def log_plan(experiment: dict, plan: dict, data_root: Path) -> None:
    counts = plan["counts"]
    logger.info(experiment["title"])
    logger.info("Paper scope: %s", experiment["paper_scope"])
    logger.info(
        "Persistence: local JSON under %s",
        data_root / experiment["key"],
    )
    logger.info("Manifest SHA-256: %s", manifest_digest(experiment))
    logger.info(
        "Plan: planned=%d completed=%d incomplete=%d missing=%d "
        "runnable=%d duplicates=%d legacy=%d",
        counts.get("planned", 0),
        counts.get("completed", 0),
        counts.get("incomplete", 0),
        counts.get("missing", 0),
        counts.get("runnable", 0),
        counts.get("duplicate_completed_signatures", 0),
        counts.get("legacy_records", 0),
    )
    for section, section_counts in plan["per_section"].items():
        logger.info("Section %-20s %s", section, section_counts)
    for game, game_counts in plan["per_game"].items():
        logger.info("Game %-35s %s", game, game_counts)


def run_jobs(jobs: list[dict], data_root: Path, max_workers: int) -> None:
    jobs = list(jobs)
    random.shuffle(jobs)

    def run_job(job: dict) -> None:
        config = copy.deepcopy(job["config"])
        run_simulation_to_json(
            experiment_name=job["experiment"],
            simulation_name=job["simulation_name"],
            phase_name=job["phase_name"],
            simulation_config=config["simulation_config"],
            instruction_config=config["instruction_config"],
            llm_config=config["llm_config"],
            data_root=data_root,
            extraFlag=job["extra_flag"],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                logger.info(
                    "Finished section=%s game=%s name=%s model=%s",
                    job["section"],
                    job["game"],
                    job["simulation_name"],
                    job["model"],
                )
            except Exception:
                logger.exception(
                    "Failed section=%s game=%s name=%s model=%s",
                    job["section"],
                    job["game"],
                    job["simulation_name"],
                    job["model"],
                )


def argument_parser(experiment: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{experiment['title']}. Local JSON storage only."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPLICATION_ROOT / "data",
        help="Top-level local-JSON data directory.",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    return parser


def main_for_experiment(experiment: dict) -> int:
    args = argument_parser(experiment).parse_args()
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be at least 1")

    existing_records = load_existing_records(args.data_root, experiment["key"])
    plan = build_plan(
        experiment,
        existing_records,
        retry_incomplete=args.retry_incomplete,
    )
    log_plan(experiment, plan, args.data_root)

    if args.dry_run or not plan["runnable_jobs"]:
        return 0
    if not args.yes:
        response = input(
            f"Run {len(plan['runnable_jobs'])} {experiment['key']} local-JSON "
            "simulations? [y/N]: "
        ).strip().lower()
        if response != "y":
            logger.info("Cancelled without writes or LLM calls.")
            return 0

    run_jobs(
        plan["runnable_jobs"],
        data_root=args.data_root,
        max_workers=args.max_workers,
    )
    return 0
