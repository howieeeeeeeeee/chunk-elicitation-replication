"""Entity contract for k-means results and per-cluster AI summaries."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from derived_analysis.common import (
    config_hash,
    copy_normalized_config,
    deterministic_entity_id,
    embedding_set_hash,
    normalize_embedding_ids,
    sanitize_error,
    utc_timestamp,
    validate_nonblank_string,
    validate_sanitized_error,
    validate_timestamp,
)
from derived_analysis.kmeans_outputs import normalize_clustering_output
from derived_analysis.kmeans_summaries import (
    add_summary_run,
    normalize_summary_config,
    record_cluster_summary_completed,
    record_cluster_summary_failed,
    record_cluster_summary_reused,
    record_cluster_summary_started,
    rendered_prompt_hash,
    summary_config_hash,
    validate_summary_run,
)
from derived_analysis.pca import validate_pca_analysis


KMEANS_SCHEMA_VERSION = 2
LEGACY_KMEANS_SCHEMA_VERSION = 1
STAGE_STATUSES = {"pending", "complete", "failed"}
IMMUTABLE_KMEANS_FIELDS = (
    "_id",
    "schema_version",
    "embedding_ids",
    "embedding_set_hash",
    "feature_source",
    "clustering_config",
    "clustering_config_hash",
    "created_at",
)


def normalize_clustering_config(clustering_config: dict) -> dict:
    return copy_normalized_config(clustering_config, "clustering_config")


def clustering_config_hash(clustering_config: dict) -> str:
    return config_hash(clustering_config, "clustering_config")


def normalize_feature_source(feature_source: dict) -> dict:
    if not isinstance(feature_source, dict):
        raise ValueError("feature_source must be a dictionary.")
    kind = feature_source.get("kind")
    if kind == "embeddings" and set(feature_source) == {"kind"}:
        return {"kind": "embeddings"}
    if kind == "pca" and set(feature_source) == {"kind", "pca_analysis_id"}:
        validate_nonblank_string(
            feature_source["pca_analysis_id"], "pca_analysis_id"
        )
        return {
            "kind": "pca",
            "pca_analysis_id": feature_source["pca_analysis_id"],
        }
    raise ValueError("feature_source must identify embeddings or one PCA analysis.")


def kmeans_analysis_id(
    embedding_ids, feature_source: dict, clustering_config: dict
) -> str:
    normalized_ids = normalize_embedding_ids(embedding_ids)
    identity = {
        "embedding_set_hash": embedding_set_hash(normalized_ids),
        "feature_source": normalize_feature_source(feature_source),
        "clustering_config_hash": clustering_config_hash(clustering_config),
    }
    return deterministic_entity_id("kmeans-analysis", identity)


def _new_attempt(started_at: str) -> dict:
    return {
        "started_at": started_at,
        "finished_at": None,
        "success": None,
        "error": None,
    }


def _validate_attempt(attempt: dict) -> None:
    required = {"started_at", "finished_at", "success", "error"}
    if not isinstance(attempt, dict) or set(attempt) != required:
        raise ValueError("Analysis attempt has an unexpected schema.")
    validate_timestamp(attempt["started_at"], "attempt started_at")
    validate_timestamp(attempt["finished_at"], "attempt finished_at", optional=True)
    if attempt["success"] is not None and not isinstance(
        attempt["success"], bool
    ):
        raise ValueError("Attempt success must be boolean or null.")
    if attempt["finished_at"] is None:
        if any(value is not None for value in (attempt["success"], attempt["error"])):
            raise ValueError("Active attempt has invalid completion metadata.")
    elif attempt["success"]:
        if attempt["error"] is not None:
            raise ValueError("Successful attempt cannot contain an error.")
    else:
        validate_sanitized_error(attempt["error"], "Clustering attempt failed.")


def build_kmeans_analysis(
    embedding_ids,
    feature_source: dict,
    clustering_config: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    normalized_ids = normalize_embedding_ids(embedding_ids)
    normalized_source = normalize_feature_source(feature_source)
    normalized_config = normalize_clustering_config(clustering_config)
    created_at = utc_timestamp(timestamp)
    entity = {
        "_id": kmeans_analysis_id(
            normalized_ids, normalized_source, normalized_config
        ),
        "schema_version": KMEANS_SCHEMA_VERSION,
        "embedding_ids": normalized_ids,
        "embedding_set_hash": embedding_set_hash(normalized_ids),
        "feature_source": normalized_source,
        "clustering_config": normalized_config,
        "clustering_config_hash": clustering_config_hash(normalized_config),
        "clustering": {
            "status": "pending",
            "output": None,
            "attempt_count": 0,
            "attempts": [],
            "updated_at": created_at,
            "completed_at": None,
        },
        "summaries": [],
        "created_at": created_at,
        "updated_at": created_at,
    }
    validate_kmeans_analysis(entity)
    return entity


def upgrade_kmeans_analysis(entity: dict) -> dict:
    validate_kmeans_analysis(entity)
    if entity["schema_version"] == KMEANS_SCHEMA_VERSION:
        return deepcopy(entity)
    if entity["schema_version"] != LEGACY_KMEANS_SCHEMA_VERSION:
        raise ValueError("K-means schema version is unsupported.")

    upgraded = deepcopy(entity)
    upgraded["schema_version"] = KMEANS_SCHEMA_VERSION
    for summary_run in upgraded["summaries"]:
        for cluster in summary_run["clusters"]:
            cluster.update(
                {
                    "prompt": None,
                    "exact_prompt_verified": False,
                    "reuse": None,
                }
            )
            for attempt in cluster["attempts"]:
                attempt.update(
                    {
                        "input_hash": cluster["input_hash"],
                        "prompt": None,
                        "prompt_hash": cluster["prompt_hash"],
                        "exact_prompt_verified": False,
                    }
                )
    validate_kmeans_analysis(upgraded)
    return upgraded


def _current_kmeans_analysis(entity: dict) -> dict:
    validate_kmeans_analysis(entity)
    return upgrade_kmeans_analysis(entity)


def _copy_clustering_retryable(entity: dict) -> dict:
    current = _current_kmeans_analysis(entity)
    if current["clustering"]["status"] == "complete":
        raise ValueError("A complete clustering stage cannot receive another attempt.")
    return current


def record_clustering_started(
    entity: dict, *, timestamp: datetime | None = None
) -> dict:
    updated = _copy_clustering_retryable(entity)
    attempts = updated["clustering"]["attempts"]
    if attempts and attempts[-1]["finished_at"] is None:
        raise ValueError("Clustering already has an active attempt.")
    started_at = utc_timestamp(timestamp)
    attempts.append(_new_attempt(started_at))
    updated["clustering"].update(
        {
            "status": "pending",
            "output": None,
            "attempt_count": len(attempts),
            "updated_at": started_at,
            "completed_at": None,
        }
    )
    updated["updated_at"] = started_at
    validate_kmeans_analysis(updated)
    return updated


def _active_attempt(attempts: list[dict], label: str) -> dict:
    if not attempts or attempts[-1]["finished_at"] is not None:
        raise ValueError(f"{label} has no active attempt.")
    return attempts[-1]


def _validate_pca_source(entity: dict, pca_analysis: dict | None) -> None:
    source = entity["feature_source"]
    if source["kind"] == "embeddings":
        return
    if pca_analysis is None:
        raise ValueError("PCA-derived clustering requires a PCA analysis.")
    validate_pca_analysis(pca_analysis)
    if pca_analysis["status"] != "complete":
        raise ValueError("PCA-derived clustering requires a complete PCA analysis.")
    if pca_analysis["_id"] != source["pca_analysis_id"]:
        raise ValueError("PCA analysis id does not match feature_source.")
    if pca_analysis["embedding_ids"] != entity["embedding_ids"]:
        raise ValueError("PCA and clustering embedding sets must be identical.")


def record_clustering_completed(
    entity: dict,
    output: dict,
    *,
    pca_analysis: dict | None = None,
    timestamp: datetime | None = None,
) -> dict:
    updated = _copy_clustering_retryable(entity)
    _validate_pca_source(updated, pca_analysis)
    attempt = _active_attempt(updated["clustering"]["attempts"], "Clustering")
    completed_at = utc_timestamp(timestamp)
    attempt.update(
        {
            "finished_at": completed_at,
            "success": True,
            "error": None,
        }
    )
    updated["clustering"].update(
        {
            "status": "complete",
            "output": normalize_clustering_output(
                output, updated["embedding_ids"]
            ),
            "updated_at": completed_at,
            "completed_at": completed_at,
        }
    )
    updated["updated_at"] = completed_at
    validate_kmeans_analysis(updated)
    return updated


def record_clustering_failed(
    entity: dict,
    error: Exception,
    *,
    timestamp: datetime | None = None,
) -> dict:
    updated = _copy_clustering_retryable(entity)
    attempt = _active_attempt(updated["clustering"]["attempts"], "Clustering")
    failed_at = utc_timestamp(timestamp)
    attempt.update(
        {
            "finished_at": failed_at,
            "success": False,
            "error": sanitize_error(error, "Clustering attempt failed."),
        }
    )
    updated["clustering"].update(
        {
            "status": "failed",
            "output": None,
            "updated_at": failed_at,
            "completed_at": None,
        }
    )
    updated["updated_at"] = failed_at
    validate_kmeans_analysis(updated)
    return updated


def _validate_stage_attempts(stage: dict) -> None:
    if (
        isinstance(stage["attempt_count"], bool)
        or not isinstance(stage["attempt_count"], int)
        or stage["attempt_count"] != len(stage["attempts"])
    ):
        raise ValueError("Attempt count does not match attempt history.")
    for attempt in stage["attempts"]:
        _validate_attempt(attempt)
    if any(
        attempt["finished_at"] is None for attempt in stage["attempts"][:-1]
    ):
        raise ValueError("Only the latest attempt may be active.")


def _validate_clustering_stage(stage: dict, embedding_ids) -> None:
    required = {
        "status",
        "output",
        "attempt_count",
        "attempts",
        "updated_at",
        "completed_at",
    }
    if not isinstance(stage, dict) or set(stage) != required:
        raise ValueError("Clustering stage has an unexpected schema.")
    if stage["status"] not in STAGE_STATUSES:
        raise ValueError("Clustering status is invalid.")
    _validate_stage_attempts(stage)
    validate_timestamp(stage["updated_at"], "clustering updated_at")
    validate_timestamp(
        stage["completed_at"], "clustering completed_at", optional=True
    )
    latest = stage["attempts"][-1] if stage["attempts"] else None
    if stage["status"] == "complete":
        if (
            latest is None
            or latest["success"] is not True
            or stage["output"] is None
            or stage["completed_at"] is None
        ):
            raise ValueError("Complete clustering is missing successful output.")
        normalize_clustering_output(stage["output"], embedding_ids)
    elif stage["output"] is not None or stage["completed_at"] is not None:
        raise ValueError("Incomplete clustering cannot retain completed output.")
    elif stage["status"] == "failed":
        if latest is None or latest["success"] is not False:
            raise ValueError("Failed clustering requires a failed latest attempt.")
    elif latest is not None and latest["finished_at"] is not None:
        raise ValueError("Pending clustering requires an active latest attempt.")


def validate_kmeans_analysis(entity: dict) -> None:
    required = {
        "_id",
        "schema_version",
        "embedding_ids",
        "embedding_set_hash",
        "feature_source",
        "clustering_config",
        "clustering_config_hash",
        "clustering",
        "summaries",
        "created_at",
        "updated_at",
    }
    if not isinstance(entity, dict) or set(entity) != required:
        raise ValueError("K-means analysis has an unexpected schema.")
    if entity["schema_version"] not in {
        LEGACY_KMEANS_SCHEMA_VERSION,
        KMEANS_SCHEMA_VERSION,
    }:
        raise ValueError("K-means schema version is unsupported.")
    normalized_ids = normalize_embedding_ids(entity["embedding_ids"])
    if normalized_ids != entity["embedding_ids"]:
        raise ValueError("Stored k-means embedding_ids are not normalized.")
    if entity["embedding_set_hash"] != embedding_set_hash(normalized_ids):
        raise ValueError("K-means embedding-set hash is invalid.")
    normalized_source = normalize_feature_source(entity["feature_source"])
    if normalized_source != entity["feature_source"]:
        raise ValueError("Stored feature_source is not normalized.")
    normalized_config = normalize_clustering_config(entity["clustering_config"])
    if normalized_config != entity["clustering_config"]:
        raise ValueError("Stored clustering_config is not normalized.")
    if entity["clustering_config_hash"] != clustering_config_hash(normalized_config):
        raise ValueError("Clustering configuration hash is invalid.")
    if entity["_id"] != kmeans_analysis_id(
        normalized_ids, normalized_source, normalized_config
    ):
        raise ValueError("K-means id does not match identity fields.")
    _validate_clustering_stage(entity["clustering"], normalized_ids)
    if not isinstance(entity["summaries"], list):
        raise ValueError("K-means summaries must be a list.")
    hashes = [item.get("summary_config_hash") for item in entity["summaries"]]
    if hashes != sorted(set(hashes)):
        raise ValueError("Summary runs must have unique sorted hashes.")
    if entity["summaries"] and entity["clustering"]["status"] != "complete":
        raise ValueError("Summaries require complete clustering.")
    for summary_run in entity["summaries"]:
        validate_summary_run(
            summary_run,
            entity["clustering"]["output"],
            entity["schema_version"],
        )
    validate_timestamp(entity["created_at"], "k-means created_at")
    validate_timestamp(entity["updated_at"], "k-means updated_at")
