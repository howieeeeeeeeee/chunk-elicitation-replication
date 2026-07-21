"""Build and compare plain simulation job dictionaries."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from typing import Any

from simluations.treatment_check import validate_treatment

from experiment_specs import BASELINE_CONFIG


_MISSING = object()

IDENTITY_FIELDS = [
    "simulation_config.game_type",
    "simulation_config.simulation_mode",
    "simulation_config.target_simulation_n",
    "simulation_config.output_format",
    "simulation_config.batch_mode",
    "simulation_config.batch_simulation_n",
    "simulation_config.previous_responses",
    "simulation_config.save_messages_n_contents",
    "instruction_config.background",
    "instruction_config.personality_traits",
    "instruction_config.explain_reasoning",
    "instruction_config.explain_reasoning_mode",
    "instruction_config.split_n",
    "instruction_config.theoretical_prediction",
    "instruction_config.additional_instructions",
    "instruction_config.include_simulation_id",
    "instruction_config.context",
    "instruction_config.incentive_size",
    "instruction_config.privacy_treatment",
    "instruction_config.focal_point",
    "llm_config.llm_service",
    "llm_config.temperature",
    "llm_config.frequency_penalty",
    "llm_config.model",
    "llm_config.reasoning_enabled",
]


def normalize_extra_flag(extra_flag: Any) -> list[str]:
    if extra_flag is None:
        return []
    if isinstance(extra_flag, (list, tuple)):
        return [str(value) for value in extra_flag]
    return [str(extra_flag)]


def get_value(data: dict, path: str, default: Any = None) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def apply_overrides(config: dict, overrides: dict) -> None:
    for path, value in overrides.items():
        parts = path.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def combo_overrides(combo: tuple) -> dict:
    explain_reasoning, mode, chunk_n, reasoning_mode, split_n = combo
    return {
        "instruction_config.explain_reasoning": explain_reasoning,
        "instruction_config.explain_reasoning_mode": reasoning_mode,
        "instruction_config.split_n": split_n,
        "simulation_config.simulation_mode": mode,
        "simulation_config.batch_simulation_n": chunk_n,
    }


def config_signature(phase_name: str, extra_flag: Any, config: dict) -> str:
    values = [phase_name, normalize_extra_flag(extra_flag)]
    for path in IDENTITY_FIELDS:
        value = get_value(config, path, _MISSING)
        if value is _MISSING:
            value = get_value(BASELINE_CONFIG, path)
        values.append(value)
    return json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)


def record_signature(record: dict) -> str:
    config = {
        "simulation_config": record.get("simulation_config", {}),
        "instruction_config": record.get("instruction_config", {}),
        "llm_config": record.get("llm_config", {}),
    }
    return config_signature(record.get("phase_name", ""), record.get("extraFlag"), config)


def make_jobs(experiment: dict) -> tuple[list[dict], int]:
    jobs = []
    invalid = 0
    seen_signatures = set()

    for section in experiment["sections"]:
        for game in section["games"]:
            simulation_number = 0
            for model, reasoning_values in section["models"].items():
                for combo in section["combos"]:
                    for reasoning_enabled in reasoning_values:
                        for treatment in section["treatments"]:
                            simulation_number += 1
                            config = copy.deepcopy(BASELINE_CONFIG)
                            config["simulation_config"]["game_type"] = game
                            config["llm_config"]["model"] = model
                            config["llm_config"]["reasoning_enabled"] = reasoning_enabled
                            apply_overrides(config, section["config_overrides"])
                            apply_overrides(config, combo_overrides(combo))
                            apply_overrides(config, treatment)

                            if not validate_treatment(
                                config["simulation_config"],
                                config["instruction_config"],
                            ):
                                invalid += 1
                                continue

                            signature = config_signature(
                                section["phase_name"],
                                section["extra_flag"],
                                config,
                            )
                            if signature in seen_signatures:
                                raise ValueError(
                                    f"Duplicate planned config: {section['name']} "
                                    f"{game} {model}"
                                )
                            seen_signatures.add(signature)
                            jobs.append(
                                {
                                    "experiment": experiment["key"],
                                    "section": section["name"],
                                    "game": game,
                                    "simulation_name": f"{simulation_number:03d}",
                                    "phase_name": section["phase_name"],
                                    "extra_flag": list(section["extra_flag"]),
                                    "model": model,
                                    "config": config,
                                    "signature": signature,
                                }
                            )

    if len(jobs) != experiment["expected_cells"]:
        raise ValueError(
            f"{experiment['key']} has {len(jobs)} valid jobs; "
            f"expected {experiment['expected_cells']}"
        )
    return jobs, invalid


def manifest_digest(experiment: dict) -> str:
    jobs, _ = make_jobs(experiment)
    payload = "\n".join(sorted(job["signature"] for job in jobs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_plan(
    experiment: dict,
    existing_records: list[dict],
    retry_incomplete: bool = False,
) -> dict:
    jobs, invalid = make_jobs(experiment)
    existing_by_signature = defaultdict(list)
    for record in existing_records:
        if record.get("archived") is not True:
            existing_by_signature[record_signature(record)].append(record)

    planned_signatures = {job["signature"] for job in jobs}
    legacy_records = [
        record
        for signature, records in existing_by_signature.items()
        if signature not in planned_signatures
        for record in records
    ]

    counts = defaultdict(int)
    counts.update(
        planned=len(jobs) + invalid,
        valid=len(jobs),
        invalid=invalid,
        legacy_records=len(legacy_records),
    )
    per_section = defaultdict(lambda: defaultdict(int))
    per_game = defaultdict(lambda: defaultdict(int))
    runnable_jobs = []
    duplicate_signatures = []

    for job in jobs:
        per_section[job["section"]]["planned"] += 1
        per_game[job["game"]]["planned"] += 1
        records = existing_by_signature.get(job["signature"], [])
        completed = [record for record in records if record.get("completed") is True]
        incomplete = [record for record in records if record.get("completed") is not True]

        if completed:
            counts["completed"] += 1
            counts["skipped"] += 1
            per_section[job["section"]]["completed"] += 1
            per_game[job["game"]]["completed"] += 1
            if len(completed) > 1:
                duplicate_signatures.append(job["signature"])
            continue

        if incomplete:
            counts["incomplete"] += 1
            per_section[job["section"]]["incomplete"] += 1
            per_game[job["game"]]["incomplete"] += 1
            if not retry_incomplete:
                counts["skipped"] += 1
                continue
        else:
            counts["missing"] += 1
            per_section[job["section"]]["missing"] += 1
            per_game[job["game"]]["missing"] += 1

        counts["runnable"] += 1
        per_section[job["section"]]["runnable"] += 1
        per_game[job["game"]]["runnable"] += 1
        runnable_jobs.append(job)

    counts["duplicate_completed_signatures"] = len(duplicate_signatures)
    return {
        "jobs": jobs,
        "runnable_jobs": runnable_jobs,
        "counts": dict(counts),
        "per_section": {key: dict(value) for key, value in per_section.items()},
        "per_game": {key: dict(value) for key, value in per_game.items()},
        "legacy_records": legacy_records,
        "duplicate_signatures": duplicate_signatures,
    }
