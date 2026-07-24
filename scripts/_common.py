"""Shared local-JSON helpers used by the three public runners."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv


REPLICATION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPLICATION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
load_dotenv(REPLICATION_ROOT / ".env")

from db_ops.config import get_database
from simluations.run_simulation import run_simulation_to_json


FORMAL_SIGNATURE_FIELDS = (
    "simulation_config.game_type",
    "llm_config.model",
    "simulation_config.simulation_mode",
    "simulation_config.batch_simulation_n",
    "simulation_config.target_simulation_n",
    "instruction_config.explain_reasoning",
    "instruction_config.explain_reasoning_mode",
    "instruction_config.split_n",
    "llm_config.reasoning_enabled",
    "llm_config.llm_service",
    "llm_config.temperature",
    "llm_config.frequency_penalty",
    "instruction_config.context",
    "instruction_config.incentive_size",
    "instruction_config.privacy_treatment",
    "instruction_config.theoretical_prediction",
    "instruction_config.focal_point",
    "simulation_config.batch_mode",
    "simulation_config.output_format",
    "simulation_config.previous_responses",
    "simulation_config.save_messages_n_contents",
    "instruction_config.background",
    "instruction_config.personality_traits",
    "instruction_config.additional_instructions",
    "instruction_config.include_simulation_id",
)

HISTORICAL_SIGNATURE_DEFAULTS = {
    "simulation_config.target_simulation_n": 100,
    "simulation_config.output_format": "json",
    "simulation_config.batch_mode": "Independent",
    "simulation_config.previous_responses": [],
    "simulation_config.save_messages_n_contents": False,
    "instruction_config.background": "",
    "instruction_config.personality_traits": "",
    "instruction_config.explain_reasoning_mode": "basic",
    "instruction_config.split_n": None,
    "instruction_config.theoretical_prediction": False,
    "instruction_config.additional_instructions": [],
    "instruction_config.include_simulation_id": True,
    "instruction_config.context": "Not Specified",
    "instruction_config.incentive_size": "Not Specified",
    "instruction_config.privacy_treatment": "Not Specified",
    "instruction_config.focal_point": False,
    "llm_config.llm_service": "openrouter",
    "llm_config.temperature": 1,
    "llm_config.frequency_penalty": 1,
}


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger


def normalize_extra_flag(extra_flag: Any) -> list:
    if extra_flag is None:
        return []
    if isinstance(extra_flag, list):
        return list(extra_flag)
    if isinstance(extra_flag, tuple):
        return list(extra_flag)
    return [extra_flag]


def get_by_dot_path(d: dict, dot_path: str) -> Any:
    cur: Any = d
    for part in dot_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_signature_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_signature_value(item) for item in value]
    return value


def normalized_simulation_signature(
    config: dict,
    *,
    phase_name: str,
    extra_flag: Any,
    fields: Iterable[str] = FORMAL_SIGNATURE_FIELDS,
) -> dict[str, Any]:
    signature = {
        "phase_name": phase_name,
        "extraFlag": normalize_extra_flag(extra_flag),
    }
    for field in fields:
        value = get_by_dot_path(config, field)
        if value is None and field in HISTORICAL_SIGNATURE_DEFAULTS:
            value = copy.deepcopy(HISTORICAL_SIGNATURE_DEFAULTS[field])
        signature[field] = _normalize_signature_value(value)
    return signature


def simulation_signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def apply_overrides(config: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        parts = key.split(".")
        cur = config
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value


def make_combos(combo_specs: list, reasoning_values: Iterable[bool]) -> list[dict]:
    combos = []
    for spec in combo_specs:
        (
            explain_reasoning,
            simulation_mode,
            batch_simulation_n,
            explain_reasoning_mode,
            split_n,
        ) = spec
        for reasoning_enabled in reasoning_values:
            combos.append(
                {
                    "instruction_config.explain_reasoning": explain_reasoning,
                    "instruction_config.explain_reasoning_mode": explain_reasoning_mode,
                    "instruction_config.split_n": split_n,
                    "simulation_config.simulation_mode": simulation_mode,
                    "simulation_config.batch_simulation_n": batch_simulation_n,
                    "llm_config.reasoning_enabled": reasoning_enabled,
                }
            )
    return combos


def simulation_exists(
    *,
    db,
    config: dict,
    phase_name: str,
    list_of_parameters: list[str],
    extra_flag: Any,
) -> bool:
    normalized = normalize_extra_flag(extra_flag)
    signature = normalized_simulation_signature(
        config,
        phase_name=phase_name,
        extra_flag=normalized,
        fields=list_of_parameters,
    )
    query: dict = {
        "phase_name": phase_name,
        "archived": False,
        "completed": True,
        "simulation_config.game_type": signature["simulation_config.game_type"],
        "llm_config.model": signature["llm_config.model"],
    }
    if normalized:
        query["extraFlag"] = normalized
    else:
        query["$or"] = [
            {"extraFlag": []},
            {"extraFlag": None},
            {"extraFlag": {"$exists": False}},
        ]
    return any(
        normalized_simulation_signature(
            candidate,
            phase_name=candidate.get("phase_name"),
            extra_flag=candidate.get("extraFlag"),
            fields=list_of_parameters,
        )
        == signature
        for candidate in db.simulations.find(query)
    )


def build_jobs(
    *,
    games: Iterable[str],
    models: dict[str, list[str]],
    parameter_combinations: dict[str, list[dict]],
    baseline_config: dict,
    phase_name: str,
    list_of_parameters: list[str],
    experiment_name: str,
    data_root: Path,
    extra_flag: Any = None,
    validate_fn: Callable[[dict, dict], bool] | None = None,
    treatment_overrides: Iterable[dict] | None = None,
):
    db = get_database(data_root=data_root, experiment_name=experiment_name)
    treatment_overrides = list(treatment_overrides or [{}])
    extra_flag = normalize_extra_flag(extra_flag)

    jobs = []
    counts = {"total": 0, "invalid": 0, "existing": 0, "to_run": 0}
    for game in games:
        game_index = 0
        for model_type, model_list in models.items():
            combos = parameter_combinations.get(model_type, [])
            for model in model_list:
                for overrides in combos:
                    for treatment in treatment_overrides:
                        game_index += 1
                        counts["total"] += 1
                        config = copy.deepcopy(baseline_config)
                        config["simulation_config"]["game_type"] = game
                        config["llm_config"]["model"] = model
                        apply_overrides(config, {**overrides, **treatment})

                        if validate_fn and not validate_fn(
                            config["simulation_config"], config["instruction_config"]
                        ):
                            counts["invalid"] += 1
                            continue

                        if simulation_exists(
                            db=db,
                            config=config,
                            phase_name=phase_name,
                            list_of_parameters=list_of_parameters,
                            extra_flag=extra_flag,
                        ):
                            counts["existing"] += 1
                            continue

                        counts["to_run"] += 1
                        jobs.append(
                            {
                                "experiment_name": experiment_name,
                                "simulation_name": f"{game_index:03d}",
                                "phase_name": phase_name,
                                "config": config,
                                "extraFlag": extra_flag,
                                "signature": normalized_simulation_signature(
                                    config,
                                    phase_name=phase_name,
                                    extra_flag=extra_flag,
                                    fields=list_of_parameters,
                                ),
                                "game": game,
                                "model": model,
                                "model_type": model_type,
                            }
                        )
    return jobs, counts


def run_jobs(
    *,
    jobs: list[dict],
    data_root: Path,
    max_workers: int,
    logger: logging.Logger,
) -> None:
    random.shuffle(jobs)

    def _run(job: dict) -> None:
        config = job["config"]
        run_simulation_to_json(
            experiment_name=job["experiment_name"],
            simulation_name=job["simulation_name"],
            phase_name=job["phase_name"],
            simulation_config=config["simulation_config"],
            instruction_config=config["instruction_config"],
            llm_config=config["llm_config"],
            data_root=data_root,
            extraFlag=job["extraFlag"],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
                logger.info(
                    "Finished game=%s simulation=%s model=%s",
                    job["game"],
                    job["simulation_name"],
                    job["model"],
                )
            except Exception as exc:
                logger.error(
                    "Failed game=%s simulation=%s model=%s error=%s",
                    job["game"],
                    job["simulation_name"],
                    job["model"],
                    exc,
                )


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPLICATION_ROOT / "data",
        help="Top-level replication data directory.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of simulation configurations to run in parallel.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run without an interactive confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without calling any LLM APIs.",
    )
    return parser


def confirm_or_exit(args, to_run: int) -> None:
    if args.dry_run or to_run == 0:
        raise SystemExit(0)
    if args.yes:
        return
    response = input(f"Proceed with {to_run} simulations? [y/N]: ").strip().lower()
    if response != "y":
        raise SystemExit(0)
