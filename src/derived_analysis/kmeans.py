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
    validate_nonnegative_int,
    validate_sanitized_error,
    validate_timestamp,
)
from derived_analysis.pca import validate_pca_analysis
from derived_analysis.kmeans_outputs import (
    normalize_clustering_output,
    normalize_summary_usage,
)


KMEANS_SCHEMA_VERSION = 1
STAGE_STATUSES = {"pending", "complete", "failed"}
FORBIDDEN_SUMMARY_PAYLOAD_KEYS = {
    "prompt",
    "prompt_template",
    "messages",
    "input",
    "input_text",
    "reasoning_text",
    "raw_response",
    "reasoning_trace",
    "system_prompt",
    "user_prompt",
    "template",
}
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


def normalize_summary_config(summary_config: dict) -> dict:
    normalized = copy_normalized_config(summary_config, "summary_config")

    def reject_payload_keys(value) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() in FORBIDDEN_SUMMARY_PAYLOAD_KEYS:
                    raise ValueError(
                        "summary_config cannot contain prompts or source text."
                    )
                reject_payload_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_payload_keys(nested)

    reject_payload_keys(normalized)
    validate_nonblank_string(normalized.get("model"), "summary_config.model")
    validate_nonblank_string(
        normalized.get("prompt_version"), "summary_config.prompt_version"
    )
    reasoning = normalized.get("reasoning")
    if not isinstance(reasoning, dict):
        raise ValueError("summary_config.reasoning must be a dictionary.")
    validate_nonblank_string(
        reasoning.get("effort"), "summary_config.reasoning.effort"
    )
    if "provider" in normalized and not isinstance(normalized["provider"], dict):
        raise ValueError("summary_config.provider must be a dictionary.")
    if "generation" in normalized and not isinstance(
        normalized["generation"], dict
    ):
        raise ValueError("summary_config.generation must be a dictionary.")
    return normalized


def summary_config_hash(summary_config: dict) -> str:
    return config_hash(normalize_summary_config(summary_config), "summary_config")


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


def _new_attempt(started_at: str, *, summary: bool = False) -> dict:
    attempt = {
        "started_at": started_at,
        "finished_at": None,
        "success": None,
        "error": None,
    }
    if summary:
        attempt.update(
            {
                "response_id": None,
                "resolved_model": None,
                "finish_reason": None,
                "native_finish_reason": None,
                "usage": None,
            }
        )
    return attempt


def _validate_attempt(attempt: dict, *, summary: bool = False) -> None:
    required = {"started_at", "finished_at", "success", "error"}
    if summary:
        required |= {
            "response_id",
            "resolved_model",
            "finish_reason",
            "native_finish_reason",
            "usage",
        }
    if not isinstance(attempt, dict) or set(attempt) != required:
        raise ValueError("Analysis attempt has an unexpected schema.")
    validate_timestamp(attempt["started_at"], "attempt started_at")
    validate_timestamp(attempt["finished_at"], "attempt finished_at", optional=True)
    if attempt["success"] is not None and not isinstance(
        attempt["success"], bool
    ):
        raise ValueError("Attempt success must be boolean or null.")
    if summary:
        for field in (
            "response_id",
            "resolved_model",
            "finish_reason",
            "native_finish_reason",
        ):
            if attempt[field] is not None and not isinstance(attempt[field], str):
                raise ValueError(f"Summary attempt {field} must be a string or null.")
        if attempt["usage"] is not None:
            normalize_summary_usage(attempt["usage"])
    if attempt["finished_at"] is None:
        populated = [attempt["success"], attempt["error"]]
        if summary:
            populated.extend(
                [
                    attempt["response_id"],
                    attempt["resolved_model"],
                    attempt["finish_reason"],
                    attempt["native_finish_reason"],
                    attempt["usage"],
                ]
            )
        if any(value is not None for value in populated):
            raise ValueError("Active attempt has invalid completion metadata.")
    elif attempt["success"]:
        if attempt["error"] is not None:
            raise ValueError("Successful attempt cannot contain an error.")
        if summary and (
            attempt["resolved_model"] is None or attempt["usage"] is None
        ):
            raise ValueError("Successful summary attempt lacks response metadata.")
    else:
        validate_sanitized_error(
            attempt["error"],
            "Cluster summary attempt failed."
            if summary
            else "Clustering attempt failed.",
        )


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


def _copy_clustering_retryable(entity: dict) -> dict:
    validate_kmeans_analysis(entity)
    if entity["clustering"]["status"] == "complete":
        raise ValueError("A complete clustering stage cannot receive another attempt.")
    return deepcopy(entity)


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


def _summary_index(entity: dict, summary_hash: str) -> int:
    for index, summary_run in enumerate(entity["summaries"]):
        if summary_run["summary_config_hash"] == summary_hash:
            return index
    raise ValueError("Summary configuration is not registered.")


def add_summary_run(
    entity: dict,
    summary_config: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    validate_kmeans_analysis(entity)
    if entity["clustering"]["status"] != "complete":
        raise ValueError("Cluster summaries require complete clustering.")
    updated = deepcopy(entity)
    normalized_config = normalize_summary_config(summary_config)
    summary_hash = summary_config_hash(normalized_config)
    for existing in updated["summaries"]:
        if existing["summary_config_hash"] == summary_hash:
            if existing["summary_config"] != normalized_config:
                raise ValueError("Summary hash conflicts with stored configuration.")
            return updated
    created_at = utc_timestamp(timestamp)
    clusters = [
        {
            "cluster_id": centroid["cluster_id"],
            "status": "pending",
            "input_hash": None,
            "prompt_hash": None,
            "output": None,
            "attempt_count": 0,
            "attempts": [],
            "updated_at": created_at,
            "completed_at": None,
        }
        for centroid in updated["clustering"]["output"]["centroids"]
    ]
    updated["summaries"].append(
        {
            "summary_config": normalized_config,
            "summary_config_hash": summary_hash,
            "status": "pending",
            "clusters": clusters,
            "created_at": created_at,
            "updated_at": created_at,
            "completed_at": None,
        }
    )
    updated["summaries"].sort(key=lambda item: item["summary_config_hash"])
    updated["updated_at"] = created_at
    validate_kmeans_analysis(updated)
    return updated


def _cluster_summary(entity: dict, summary_hash: str, cluster_id: int) -> dict:
    summary_run = entity["summaries"][_summary_index(entity, summary_hash)]
    for cluster in summary_run["clusters"]:
        if cluster["cluster_id"] == cluster_id:
            return cluster
    raise ValueError("Cluster id is not present in the summary run.")


def _refresh_summary_status(summary_run: dict, timestamp: str) -> None:
    statuses = [cluster["status"] for cluster in summary_run["clusters"]]
    if statuses and all(status == "complete" for status in statuses):
        summary_run["status"] = "complete"
        summary_run["completed_at"] = timestamp
    elif any(status == "pending" for status in statuses):
        summary_run["status"] = "pending"
        summary_run["completed_at"] = None
    else:
        summary_run["status"] = "failed"
        summary_run["completed_at"] = None
    summary_run["updated_at"] = timestamp


def record_cluster_summary_started(
    entity: dict,
    summary_hash: str,
    cluster_id: int,
    *,
    input_hash: str,
    prompt_hash: str,
    timestamp: datetime | None = None,
) -> dict:
    validate_kmeans_analysis(entity)
    validate_nonnegative_int(cluster_id, "cluster_id")
    validate_nonblank_string(input_hash, "input_hash")
    validate_nonblank_string(prompt_hash, "prompt_hash")
    updated = deepcopy(entity)
    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be retried.")
    if cluster["attempts"] and cluster["attempts"][-1]["finished_at"] is None:
        raise ValueError("Cluster summary already has an active attempt.")
    if cluster["input_hash"] not in {None, input_hash}:
        raise ValueError("Cluster summary input hash is immutable.")
    if cluster["prompt_hash"] not in {None, prompt_hash}:
        raise ValueError("Cluster summary prompt hash is immutable.")
    started_at = utc_timestamp(timestamp)
    cluster["attempts"].append(_new_attempt(started_at, summary=True))
    cluster.update(
        {
            "status": "pending",
            "input_hash": input_hash,
            "prompt_hash": prompt_hash,
            "output": None,
            "attempt_count": len(cluster["attempts"]),
            "updated_at": started_at,
            "completed_at": None,
        }
    )
    summary_run = updated["summaries"][_summary_index(updated, summary_hash)]
    _refresh_summary_status(summary_run, started_at)
    updated["updated_at"] = started_at
    validate_kmeans_analysis(updated)
    return updated


def _normalize_summary_response(response: dict, cluster: dict) -> dict:
    required = {
        "summary",
        "response_id",
        "resolved_model",
        "finish_reason",
        "native_finish_reason",
        "usage",
    }
    if not isinstance(response, dict) or set(response) != required:
        raise ValueError("Summary response has an unexpected schema.")
    validate_nonblank_string(response["summary"], "summary")
    validate_nonblank_string(response["resolved_model"], "resolved_model")
    for field in ("response_id", "finish_reason", "native_finish_reason"):
        if response[field] is not None and not isinstance(response[field], str):
            raise ValueError(f"{field} must be a string or null.")
    usage = normalize_summary_usage(response["usage"])
    return {
        "summary": response["summary"],
        "response_id": response["response_id"],
        "resolved_model": response["resolved_model"],
        "finish_reason": response["finish_reason"],
        "native_finish_reason": response["native_finish_reason"],
        "input_hash": cluster["input_hash"],
        "prompt_hash": cluster["prompt_hash"],
        "usage": usage,
    }


def record_cluster_summary_completed(
    entity: dict,
    summary_hash: str,
    cluster_id: int,
    response: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    validate_kmeans_analysis(entity)
    updated = deepcopy(entity)
    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be retried.")
    attempt = _active_attempt(cluster["attempts"], "Cluster summary")
    completed_at = utc_timestamp(timestamp)
    output = _normalize_summary_response(response, cluster)
    attempt.update(
        {
            "finished_at": completed_at,
            "success": True,
            "response_id": response["response_id"],
            "resolved_model": response["resolved_model"],
            "finish_reason": response["finish_reason"],
            "native_finish_reason": response["native_finish_reason"],
            "usage": output["usage"],
            "error": None,
        }
    )
    cluster.update(
        {
            "status": "complete",
            "output": output,
            "updated_at": completed_at,
            "completed_at": completed_at,
        }
    )
    summary_run = updated["summaries"][_summary_index(updated, summary_hash)]
    _refresh_summary_status(summary_run, completed_at)
    updated["updated_at"] = completed_at
    validate_kmeans_analysis(updated)
    return updated


def record_cluster_summary_failed(
    entity: dict,
    summary_hash: str,
    cluster_id: int,
    error: Exception,
    *,
    timestamp: datetime | None = None,
) -> dict:
    validate_kmeans_analysis(entity)
    updated = deepcopy(entity)
    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be retried.")
    attempt = _active_attempt(cluster["attempts"], "Cluster summary")
    failed_at = utc_timestamp(timestamp)
    usage = getattr(error, "usage", None)
    normalized_usage = normalize_summary_usage(usage) if usage is not None else None
    attempt.update(
        {
            "finished_at": failed_at,
            "success": False,
            "response_id": getattr(error, "response_id", None),
            "resolved_model": getattr(error, "resolved_model", None),
            "finish_reason": getattr(error, "finish_reason", None),
            "native_finish_reason": getattr(error, "native_finish_reason", None),
            "usage": normalized_usage,
            "error": sanitize_error(error, "Cluster summary attempt failed."),
        }
    )
    cluster.update(
        {
            "status": "failed",
            "output": None,
            "updated_at": failed_at,
            "completed_at": None,
        }
    )
    summary_run = updated["summaries"][_summary_index(updated, summary_hash)]
    _refresh_summary_status(summary_run, failed_at)
    updated["updated_at"] = failed_at
    validate_kmeans_analysis(updated)
    return updated


def _validate_stage_attempts(stage: dict, *, summary: bool = False) -> None:
    if (
        isinstance(stage["attempt_count"], bool)
        or not isinstance(stage["attempt_count"], int)
        or stage["attempt_count"] != len(stage["attempts"])
    ):
        raise ValueError("Attempt count does not match attempt history.")
    for attempt in stage["attempts"]:
        _validate_attempt(attempt, summary=summary)
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


def _validate_summary_cluster(cluster: dict) -> None:
    required = {
        "cluster_id",
        "status",
        "input_hash",
        "prompt_hash",
        "output",
        "attempt_count",
        "attempts",
        "updated_at",
        "completed_at",
    }
    if not isinstance(cluster, dict) or set(cluster) != required:
        raise ValueError("Summary cluster has an unexpected schema.")
    validate_nonnegative_int(cluster["cluster_id"], "summary cluster_id")
    if cluster["status"] not in STAGE_STATUSES:
        raise ValueError("Summary cluster status is invalid.")
    for field in ("input_hash", "prompt_hash"):
        if cluster[field] is not None:
            validate_nonblank_string(cluster[field], field)
    _validate_stage_attempts(cluster, summary=True)
    validate_timestamp(cluster["updated_at"], "summary cluster updated_at")
    validate_timestamp(
        cluster["completed_at"], "summary cluster completed_at", optional=True
    )
    latest = cluster["attempts"][-1] if cluster["attempts"] else None
    if cluster["status"] == "complete":
        if (
            latest is None
            or latest["success"] is not True
            or cluster["output"] is None
            or cluster["completed_at"] is None
        ):
            raise ValueError("Complete cluster summary is missing output.")
        output = cluster["output"]
        required_output = {
            "summary",
            "response_id",
            "resolved_model",
            "finish_reason",
            "native_finish_reason",
            "input_hash",
            "prompt_hash",
            "usage",
        }
        if not isinstance(output, dict) or set(output) != required_output:
            raise ValueError("Cluster summary output has an unexpected schema.")
        validate_nonblank_string(output["summary"], "summary")
        validate_nonblank_string(output["resolved_model"], "resolved_model")
        for field in ("response_id", "finish_reason", "native_finish_reason"):
            if output[field] is not None and not isinstance(output[field], str):
                raise ValueError(
                    f"Cluster summary output {field} must be a string or null."
                )
        if (
            output["input_hash"] != cluster["input_hash"]
            or output["prompt_hash"] != cluster["prompt_hash"]
        ):
            raise ValueError("Cluster summary provenance hashes do not match.")
        normalize_summary_usage(output["usage"])
        for field in (
            "response_id",
            "resolved_model",
            "finish_reason",
            "native_finish_reason",
            "usage",
        ):
            if output[field] != latest[field]:
                raise ValueError(
                    "Cluster summary output must match the latest attempt."
                )
    elif cluster["output"] is not None or cluster["completed_at"] is not None:
        raise ValueError("Incomplete cluster summary cannot retain output.")
    elif cluster["status"] == "failed":
        if latest is None or latest["success"] is not False:
            raise ValueError("Failed cluster summary requires a failed latest attempt.")
    elif latest is not None and latest["finished_at"] is not None:
        raise ValueError("Pending cluster summary requires an active latest attempt.")


def _validate_summary_run(summary_run: dict, clustering_output: dict) -> None:
    required = {
        "summary_config",
        "summary_config_hash",
        "status",
        "clusters",
        "created_at",
        "updated_at",
        "completed_at",
    }
    if not isinstance(summary_run, dict) or set(summary_run) != required:
        raise ValueError("Summary run has an unexpected schema.")
    normalized_config = normalize_summary_config(summary_run["summary_config"])
    if normalized_config != summary_run["summary_config"]:
        raise ValueError("Stored summary_config is not normalized.")
    if summary_run["summary_config_hash"] != summary_config_hash(normalized_config):
        raise ValueError("Summary configuration hash is invalid.")
    if summary_run["status"] not in STAGE_STATUSES:
        raise ValueError("Summary run status is invalid.")
    validate_timestamp(summary_run["created_at"], "summary created_at")
    validate_timestamp(summary_run["updated_at"], "summary updated_at")
    validate_timestamp(
        summary_run["completed_at"], "summary completed_at", optional=True
    )
    expected_ids = [
        centroid["cluster_id"] for centroid in clustering_output["centroids"]
    ]
    actual_ids = [cluster["cluster_id"] for cluster in summary_run["clusters"]]
    if actual_ids != expected_ids:
        raise ValueError("Summary clusters must match clustering centroids.")
    for cluster in summary_run["clusters"]:
        _validate_summary_cluster(cluster)
    statuses = [cluster["status"] for cluster in summary_run["clusters"]]
    if summary_run["status"] == "complete":
        if (
            not statuses
            or not all(status == "complete" for status in statuses)
            or summary_run["completed_at"] is None
        ):
            raise ValueError("Complete summary run has incomplete clusters.")
    elif summary_run["completed_at"] is not None:
        raise ValueError("Incomplete summary run cannot have completed_at.")
    elif summary_run["status"] == "failed":
        if any(status == "pending" for status in statuses) or not any(
            status == "failed" for status in statuses
        ):
            raise ValueError("Failed summary run has inconsistent cluster states.")
    elif not any(status == "pending" for status in statuses):
        raise ValueError("Pending summary run requires a pending cluster.")


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
    if entity["schema_version"] != KMEANS_SCHEMA_VERSION:
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
        _validate_summary_run(summary_run, entity["clustering"]["output"])
    validate_timestamp(entity["created_at"], "k-means created_at")
    validate_timestamp(entity["updated_at"], "k-means updated_at")
