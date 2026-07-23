"""Deterministic entity contract for PCA results over reasoning embeddings."""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime

from derived_analysis.common import (
    config_hash,
    copy_normalized_config,
    deterministic_entity_id,
    embedding_set_hash,
    normalize_embedding_ids,
    normalize_json_value,
    sanitize_error,
    utc_timestamp,
    validate_nonnegative_int,
    validate_sanitized_error,
    validate_timestamp,
)


PCA_SCHEMA_VERSION = 1
PCA_STATUSES = {"pending", "complete", "failed"}
IMMUTABLE_PCA_FIELDS = (
    "_id",
    "schema_version",
    "embedding_ids",
    "embedding_set_hash",
    "pca_config",
    "pca_config_hash",
    "created_at",
)


def normalize_pca_config(pca_config: dict) -> dict:
    return copy_normalized_config(pca_config, "pca_config")


def pca_config_hash(pca_config: dict) -> str:
    return config_hash(pca_config, "pca_config")


def pca_analysis_id(embedding_ids, pca_config: dict) -> str:
    normalized_ids = normalize_embedding_ids(embedding_ids)
    identity = {
        "embedding_set_hash": embedding_set_hash(normalized_ids),
        "pca_config_hash": pca_config_hash(pca_config),
    }
    return deterministic_entity_id("pca-analysis", identity)


def _validate_attempt(attempt: dict) -> None:
    required = {"started_at", "finished_at", "success", "error"}
    if not isinstance(attempt, dict) or set(attempt) != required:
        raise ValueError("PCA attempt has an unexpected schema.")
    validate_timestamp(attempt["started_at"], "PCA attempt started_at")
    validate_timestamp(
        attempt["finished_at"], "PCA attempt finished_at", optional=True
    )
    if attempt["success"] is not None and not isinstance(
        attempt["success"], bool
    ):
        raise ValueError("PCA attempt success must be boolean or null.")
    if attempt["finished_at"] is None:
        if attempt["success"] is not None or attempt["error"] is not None:
            raise ValueError("Active PCA attempt has invalid completion metadata.")
    elif attempt["success"]:
        if attempt["error"] is not None:
            raise ValueError("Successful PCA attempt cannot contain an error.")
    else:
        validate_sanitized_error(attempt["error"], "PCA attempt failed.")


def _finite_vector(values, field: str) -> list[float | int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a nonempty list.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(f"{field} must contain finite numbers.")
    return deepcopy(values)


def normalize_pca_output(output: dict, embedding_ids) -> dict:
    required = {
        "coordinates",
        "n_samples",
        "n_input_dimensions",
        "n_components",
        "diagnostics",
    }
    if not isinstance(output, dict) or set(output) != required:
        raise ValueError("PCA output has an unexpected schema.")
    normalized_ids = normalize_embedding_ids(embedding_ids)
    coordinates = output["coordinates"]
    if not isinstance(coordinates, list) or len(coordinates) != len(normalized_ids):
        raise ValueError("PCA coordinates must align with every embedding id.")

    normalized_coordinates = []
    component_count = None
    for expected_id, coordinate in zip(normalized_ids, coordinates):
        if not isinstance(coordinate, dict) or set(coordinate) != {
            "embedding_id",
            "values",
        }:
            raise ValueError("PCA coordinate has an unexpected schema.")
        if coordinate["embedding_id"] != expected_id:
            raise ValueError("PCA coordinate order must match embedding_ids.")
        values = _finite_vector(coordinate["values"], "PCA coordinate values")
        if component_count is None:
            component_count = len(values)
        elif len(values) != component_count:
            raise ValueError("PCA coordinate dimensions must be consistent.")
        normalized_coordinates.append(
            {"embedding_id": expected_id, "values": values}
        )

    for field in ("n_samples", "n_input_dimensions", "n_components"):
        validate_nonnegative_int(output[field], f"PCA output {field}")
        if output[field] < 1:
            raise ValueError(f"PCA output {field} must be positive.")
    if output["n_samples"] != len(normalized_ids):
        raise ValueError("PCA output n_samples is inconsistent.")
    if output["n_components"] != component_count:
        raise ValueError("PCA output n_components is inconsistent.")
    if output["n_components"] > output["n_input_dimensions"]:
        raise ValueError("PCA components cannot exceed input dimensions.")
    if output["n_components"] > output["n_samples"]:
        raise ValueError("PCA components cannot exceed sample count.")

    diagnostics = normalize_json_value(output["diagnostics"], "PCA diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("PCA diagnostics must be a dictionary.")
    return {
        "coordinates": normalized_coordinates,
        "n_samples": output["n_samples"],
        "n_input_dimensions": output["n_input_dimensions"],
        "n_components": output["n_components"],
        "diagnostics": diagnostics,
    }


def build_pca_analysis(
    embedding_ids,
    pca_config: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    normalized_ids = normalize_embedding_ids(embedding_ids)
    normalized_config = normalize_pca_config(pca_config)
    created_at = utc_timestamp(timestamp)
    entity = {
        "_id": pca_analysis_id(normalized_ids, normalized_config),
        "schema_version": PCA_SCHEMA_VERSION,
        "embedding_ids": normalized_ids,
        "embedding_set_hash": embedding_set_hash(normalized_ids),
        "pca_config": normalized_config,
        "pca_config_hash": pca_config_hash(normalized_config),
        "status": "pending",
        "output": None,
        "attempt_count": 0,
        "attempts": [],
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
    }
    validate_pca_analysis(entity)
    return entity


def _copy_retryable(entity: dict) -> dict:
    validate_pca_analysis(entity)
    if entity["status"] == "complete":
        raise ValueError("A complete PCA analysis cannot receive another attempt.")
    return deepcopy(entity)


def record_pca_started(
    entity: dict, *, timestamp: datetime | None = None
) -> dict:
    updated = _copy_retryable(entity)
    if updated["attempts"] and updated["attempts"][-1]["finished_at"] is None:
        raise ValueError("PCA analysis already has an active attempt.")
    started_at = utc_timestamp(timestamp)
    updated["attempts"].append(
        {
            "started_at": started_at,
            "finished_at": None,
            "success": None,
            "error": None,
        }
    )
    updated["attempt_count"] = len(updated["attempts"])
    updated["status"] = "pending"
    updated["output"] = None
    updated["updated_at"] = started_at
    updated["completed_at"] = None
    validate_pca_analysis(updated)
    return updated


def _active_attempt(entity: dict) -> dict:
    if not entity["attempts"] or entity["attempts"][-1]["finished_at"] is not None:
        raise ValueError("PCA analysis has no active attempt.")
    return entity["attempts"][-1]


def record_pca_completed(
    entity: dict,
    output: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    updated = _copy_retryable(entity)
    attempt = _active_attempt(updated)
    completed_at = utc_timestamp(timestamp)
    normalized_output = normalize_pca_output(output, updated["embedding_ids"])
    attempt["finished_at"] = completed_at
    attempt["success"] = True
    attempt["error"] = None
    updated["status"] = "complete"
    updated["output"] = normalized_output
    updated["updated_at"] = completed_at
    updated["completed_at"] = completed_at
    validate_pca_analysis(updated)
    return updated


def record_pca_failed(
    entity: dict,
    error: Exception,
    *,
    timestamp: datetime | None = None,
) -> dict:
    updated = _copy_retryable(entity)
    attempt = _active_attempt(updated)
    failed_at = utc_timestamp(timestamp)
    attempt["finished_at"] = failed_at
    attempt["success"] = False
    attempt["error"] = sanitize_error(error, "PCA attempt failed.")
    updated["status"] = "failed"
    updated["output"] = None
    updated["updated_at"] = failed_at
    updated["completed_at"] = None
    validate_pca_analysis(updated)
    return updated


def validate_pca_analysis(entity: dict) -> None:
    required = {
        "_id",
        "schema_version",
        "embedding_ids",
        "embedding_set_hash",
        "pca_config",
        "pca_config_hash",
        "status",
        "output",
        "attempt_count",
        "attempts",
        "created_at",
        "updated_at",
        "completed_at",
    }
    if not isinstance(entity, dict) or set(entity) != required:
        raise ValueError("PCA analysis has an unexpected schema.")
    if entity["schema_version"] != PCA_SCHEMA_VERSION:
        raise ValueError("PCA schema version is unsupported.")
    normalized_ids = normalize_embedding_ids(entity["embedding_ids"])
    if normalized_ids != entity["embedding_ids"]:
        raise ValueError("Stored PCA embedding_ids are not normalized.")
    if entity["embedding_set_hash"] != embedding_set_hash(normalized_ids):
        raise ValueError("PCA embedding-set hash does not match embedding_ids.")
    normalized_config = normalize_pca_config(entity["pca_config"])
    if normalized_config != entity["pca_config"]:
        raise ValueError("Stored pca_config is not normalized.")
    if entity["pca_config_hash"] != pca_config_hash(normalized_config):
        raise ValueError("PCA configuration hash does not match configuration.")
    if entity["_id"] != pca_analysis_id(normalized_ids, normalized_config):
        raise ValueError("PCA id does not match identity fields.")
    if entity["status"] not in PCA_STATUSES:
        raise ValueError("PCA status is invalid.")
    if (
        isinstance(entity["attempt_count"], bool)
        or not isinstance(entity["attempt_count"], int)
        or entity["attempt_count"] != len(entity["attempts"])
    ):
        raise ValueError("PCA attempt_count does not match attempts.")
    for attempt in entity["attempts"]:
        _validate_attempt(attempt)
    if any(
        attempt["finished_at"] is None
        for attempt in entity["attempts"][:-1]
    ):
        raise ValueError("Only the latest PCA attempt may be active.")
    validate_timestamp(entity["created_at"], "PCA created_at")
    validate_timestamp(entity["updated_at"], "PCA updated_at")
    validate_timestamp(entity["completed_at"], "PCA completed_at", optional=True)

    latest = entity["attempts"][-1] if entity["attempts"] else None
    if entity["status"] == "complete":
        if (
            latest is None
            or latest["success"] is not True
            or entity["completed_at"] is None
            or entity["output"] is None
        ):
            raise ValueError("Complete PCA analysis is missing successful output.")
        normalize_pca_output(entity["output"], normalized_ids)
    elif entity["output"] is not None or entity["completed_at"] is not None:
        raise ValueError("Incomplete PCA analysis cannot retain completed output.")
    elif entity["status"] == "failed":
        if latest is None or latest["success"] is not False:
            raise ValueError("Failed PCA analysis requires a failed latest attempt.")
    elif latest is not None and latest["finished_at"] is not None:
        raise ValueError("Pending PCA analysis requires an active latest attempt.")
