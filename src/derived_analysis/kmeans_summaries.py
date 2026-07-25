"""Per-cluster summary configuration, lifecycle, and validation."""

from __future__ import annotations

from datetime import datetime
import hashlib

from derived_analysis.common import (
    config_hash,
    copy_normalized_config,
    sanitize_error,
    utc_timestamp,
    validate_nonblank_string,
    validate_nonnegative_int,
    validate_sanitized_error,
    validate_timestamp,
)
from derived_analysis.kmeans_outputs import normalize_summary_usage


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


def rendered_prompt_hash(prompt: str) -> str:
    validate_nonblank_string(prompt, "prompt")
    try:
        prompt_bytes = prompt.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("prompt must be valid UTF-8 text.") from error
    return hashlib.sha256(prompt_bytes).hexdigest()


def _new_summary_attempt(started_at: str, input_hash: str, prompt: str) -> dict:
    return {
        "started_at": started_at,
        "finished_at": None,
        "success": None,
        "error": None,
        "input_hash": input_hash,
        "prompt": prompt,
        "prompt_hash": rendered_prompt_hash(prompt),
        "exact_prompt_verified": True,
        "response_id": None,
        "resolved_model": None,
        "finish_reason": None,
        "native_finish_reason": None,
        "usage": None,
    }


def _validate_prompt_provenance(
    prompt,
    prompt_hash,
    exact_prompt_verified,
    *,
    label: str,
) -> None:
    if not isinstance(exact_prompt_verified, bool):
        raise ValueError(f"{label} exact_prompt_verified must be boolean.")
    if exact_prompt_verified:
        validate_nonblank_string(prompt, f"{label} prompt")
        if prompt_hash != rendered_prompt_hash(prompt):
            raise ValueError(f"{label} prompt hash does not match exact UTF-8 bytes.")
    else:
        if prompt is not None:
            raise ValueError(f"{label} unverified prompt must be null.")
        if prompt_hash is not None:
            validate_nonblank_string(prompt_hash, f"{label} prompt_hash")


def _validate_summary_attempt(attempt: dict, schema_version: int) -> None:
    required = {
        "started_at",
        "finished_at",
        "success",
        "error",
        "response_id",
        "resolved_model",
        "finish_reason",
        "native_finish_reason",
        "usage",
    }
    if schema_version >= 2:
        required |= {
            "input_hash",
            "prompt",
            "prompt_hash",
            "exact_prompt_verified",
        }
    if not isinstance(attempt, dict) or set(attempt) != required:
        raise ValueError("Analysis attempt has an unexpected schema.")
    validate_timestamp(attempt["started_at"], "attempt started_at")
    validate_timestamp(attempt["finished_at"], "attempt finished_at", optional=True)
    if attempt["success"] is not None and not isinstance(attempt["success"], bool):
        raise ValueError("Attempt success must be boolean or null.")
    if schema_version >= 2:
        if attempt["input_hash"] is not None:
            validate_nonblank_string(
                attempt["input_hash"], "summary attempt input_hash"
            )
        _validate_prompt_provenance(
            attempt["prompt"],
            attempt["prompt_hash"],
            attempt["exact_prompt_verified"],
            label="summary attempt",
        )
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
        if any(
            attempt[field] is not None
            for field in (
                "success",
                "error",
                "response_id",
                "resolved_model",
                "finish_reason",
                "native_finish_reason",
                "usage",
            )
        ):
            raise ValueError("Active attempt has invalid completion metadata.")
    elif attempt["success"]:
        if attempt["error"] is not None:
            raise ValueError("Successful attempt cannot contain an error.")
        if attempt["resolved_model"] is None or attempt["usage"] is None:
            raise ValueError("Successful summary attempt lacks response metadata.")
    else:
        validate_sanitized_error(
            attempt["error"],
            "Cluster summary attempt failed.",
        )


def _validate_attempt_history(cluster: dict, schema_version: int) -> None:
    if (
        isinstance(cluster["attempt_count"], bool)
        or not isinstance(cluster["attempt_count"], int)
        or cluster["attempt_count"] != len(cluster["attempts"])
    ):
        raise ValueError("Attempt count does not match attempt history.")
    for attempt in cluster["attempts"]:
        _validate_summary_attempt(attempt, schema_version)
    if any(
        attempt["finished_at"] is None for attempt in cluster["attempts"][:-1]
    ):
        raise ValueError("Only the latest attempt may be active.")


def _validate_reuse(reuse: dict) -> None:
    required = {
        "source_analysis_id",
        "source_summary_config_hash",
        "source_cluster_id",
        "reused_at",
    }
    if not isinstance(reuse, dict) or set(reuse) != required:
        raise ValueError("Cluster summary reuse has an unexpected schema.")
    validate_nonblank_string(reuse["source_analysis_id"], "reuse source_analysis_id")
    validate_nonblank_string(
        reuse["source_summary_config_hash"],
        "reuse source_summary_config_hash",
    )
    validate_nonnegative_int(reuse["source_cluster_id"], "reuse source_cluster_id")
    validate_timestamp(reuse["reused_at"], "reuse reused_at")


def _validate_provider_output(output: dict, cluster: dict, latest: dict) -> None:
    required = {
        "summary",
        "response_id",
        "resolved_model",
        "finish_reason",
        "native_finish_reason",
        "input_hash",
        "prompt_hash",
        "usage",
    }
    if not isinstance(output, dict) or set(output) != required:
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
            raise ValueError("Cluster summary output must match the latest attempt.")


def _validate_reused_output(output: dict, cluster: dict) -> None:
    required = {"summary", "input_hash", "prompt_hash"}
    if not isinstance(output, dict) or set(output) != required:
        raise ValueError("Cluster summary output has an unexpected schema.")
    validate_nonblank_string(output["summary"], "summary")
    if (
        output["input_hash"] != cluster["input_hash"]
        or output["prompt_hash"] != cluster["prompt_hash"]
    ):
        raise ValueError("Cluster summary provenance hashes do not match.")


def _validate_summary_cluster(cluster: dict, schema_version: int) -> None:
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
    if schema_version >= 2:
        required |= {"prompt", "exact_prompt_verified", "reuse"}
    if not isinstance(cluster, dict) or set(cluster) != required:
        raise ValueError("Summary cluster has an unexpected schema.")
    validate_nonnegative_int(cluster["cluster_id"], "summary cluster_id")
    if cluster["status"] not in STAGE_STATUSES:
        raise ValueError("Summary cluster status is invalid.")
    for field in ("input_hash", "prompt_hash"):
        if cluster[field] is not None:
            validate_nonblank_string(cluster[field], field)
    if schema_version >= 2:
        _validate_prompt_provenance(
            cluster["prompt"],
            cluster["prompt_hash"],
            cluster["exact_prompt_verified"],
            label="summary cluster",
        )
    _validate_attempt_history(cluster, schema_version)
    validate_timestamp(cluster["updated_at"], "summary cluster updated_at")
    validate_timestamp(
        cluster["completed_at"], "summary cluster completed_at", optional=True
    )
    latest = cluster["attempts"][-1] if cluster["attempts"] else None
    reuse = cluster.get("reuse") if schema_version >= 2 else None
    if reuse is not None:
        _validate_reuse(reuse)
        if cluster["status"] != "complete":
            raise ValueError("Only a complete cluster summary may record reuse.")
    if schema_version >= 2 and latest is not None and reuse is None:
        for field in ("input_hash", "prompt", "prompt_hash", "exact_prompt_verified"):
            if cluster[field] != latest[field]:
                raise ValueError(
                    "Cluster-summary provenance must match the latest attempt."
                )
    elif (
        schema_version >= 2
        and latest is None
        and reuse is None
        and (
            cluster["input_hash"] is not None
            or cluster["prompt"] is not None
            or cluster["prompt_hash"] is not None
            or cluster["exact_prompt_verified"]
        )
    ):
        raise ValueError("Unattempted cluster summary cannot retain request state.")

    if cluster["status"] == "complete":
        if cluster["output"] is None or cluster["completed_at"] is None:
            raise ValueError("Complete cluster summary is missing output.")
        if reuse is None:
            if latest is None or latest["success"] is not True:
                raise ValueError(
                    "Complete provider summary requires a successful latest attempt."
                )
            _validate_provider_output(cluster["output"], cluster, latest)
        else:
            _validate_reused_output(cluster["output"], cluster)
    elif cluster["output"] is not None or cluster["completed_at"] is not None:
        raise ValueError("Incomplete cluster summary cannot retain output.")
    elif reuse is not None:
        raise ValueError("Incomplete cluster summary cannot retain reuse.")
    elif cluster["status"] == "failed":
        if latest is None or latest["success"] is not False:
            raise ValueError("Failed cluster summary requires a failed latest attempt.")
    elif latest is not None and latest["finished_at"] is not None:
        raise ValueError("Pending cluster summary requires an active latest attempt.")


def validate_summary_run(
    summary_run: dict,
    clustering_output: dict,
    schema_version: int,
) -> None:
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
        _validate_summary_cluster(cluster, schema_version)
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


def _current_entity(entity: dict) -> dict:
    from derived_analysis.kmeans import upgrade_kmeans_analysis

    return upgrade_kmeans_analysis(entity)


def _summary_index(entity: dict, summary_hash: str) -> int:
    for index, summary_run in enumerate(entity["summaries"]):
        if summary_run["summary_config_hash"] == summary_hash:
            return index
    raise ValueError("Summary configuration is not registered.")


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


def add_summary_run(
    entity: dict,
    summary_config: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    from derived_analysis.kmeans import validate_kmeans_analysis

    updated = _current_entity(entity)
    if updated["clustering"]["status"] != "complete":
        raise ValueError("Cluster summaries require complete clustering.")
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
            "prompt": None,
            "prompt_hash": None,
            "exact_prompt_verified": False,
            "reuse": None,
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


def record_cluster_summary_started(
    entity: dict,
    summary_hash: str,
    cluster_id: int,
    *,
    input_hash: str,
    prompt: str,
    timestamp: datetime | None = None,
) -> dict:
    from derived_analysis.kmeans import validate_kmeans_analysis

    updated = _current_entity(entity)
    validate_nonnegative_int(cluster_id, "cluster_id")
    validate_nonblank_string(input_hash, "input_hash")
    prompt_digest = rendered_prompt_hash(prompt)
    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be retried.")
    if cluster["attempts"] and cluster["attempts"][-1]["finished_at"] is None:
        raise ValueError("Cluster summary already has an active attempt.")
    started_at = utc_timestamp(timestamp)
    cluster["attempts"].append(
        _new_summary_attempt(started_at, input_hash, prompt)
    )
    cluster.update(
        {
            "status": "pending",
            "input_hash": input_hash,
            "prompt": prompt,
            "prompt_hash": prompt_digest,
            "exact_prompt_verified": True,
            "reuse": None,
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


def _active_attempt(attempts: list[dict]) -> dict:
    if not attempts or attempts[-1]["finished_at"] is not None:
        raise ValueError("Cluster summary has no active attempt.")
    return attempts[-1]


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
    from derived_analysis.kmeans import validate_kmeans_analysis

    updated = _current_entity(entity)
    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be retried.")
    attempt = _active_attempt(cluster["attempts"])
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
    from derived_analysis.kmeans import validate_kmeans_analysis

    updated = _current_entity(entity)
    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be retried.")
    attempt = _active_attempt(cluster["attempts"])
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


def record_cluster_summary_reused(
    entity: dict,
    summary_hash: str,
    cluster_id: int,
    *,
    input_hash: str,
    prompt: str,
    source_analysis_id: str,
    source_summary_config_hash: str,
    source_cluster_id: int,
    source_cluster: dict,
    timestamp: datetime | None = None,
) -> dict:
    from derived_analysis.kmeans import validate_kmeans_analysis

    updated = _current_entity(entity)
    validate_nonnegative_int(cluster_id, "cluster_id")
    validate_nonblank_string(input_hash, "input_hash")
    validate_nonblank_string(source_analysis_id, "source_analysis_id")
    validate_nonnegative_int(source_cluster_id, "source_cluster_id")
    if source_summary_config_hash != summary_hash:
        raise ValueError("Reused summary configuration hash must match the target.")
    if source_analysis_id == updated["_id"] and source_cluster_id == cluster_id:
        raise ValueError("A cluster summary cannot reuse itself.")

    prompt_digest = rendered_prompt_hash(prompt)
    if (
        not isinstance(source_cluster, dict)
        or source_cluster.get("status") != "complete"
        or source_cluster.get("reuse") is not None
        or source_cluster.get("exact_prompt_verified") is not True
        or source_cluster.get("prompt") != prompt
        or source_cluster.get("prompt_hash") != prompt_digest
        or not isinstance(source_cluster.get("output"), dict)
    ):
        raise ValueError("Reused source must be a completed exact provider summary.")
    validate_nonblank_string(source_cluster["output"].get("summary"), "source summary")

    cluster = _cluster_summary(updated, summary_hash, cluster_id)
    if cluster["status"] == "complete":
        raise ValueError("A complete cluster summary cannot be replaced.")
    if cluster["attempts"] and cluster["attempts"][-1]["finished_at"] is None:
        raise ValueError("Cluster summary already has an active attempt.")
    reused_at = utc_timestamp(timestamp)
    cluster.update(
        {
            "status": "complete",
            "input_hash": input_hash,
            "prompt": prompt,
            "prompt_hash": prompt_digest,
            "exact_prompt_verified": True,
            "reuse": {
                "source_analysis_id": source_analysis_id,
                "source_summary_config_hash": source_summary_config_hash,
                "source_cluster_id": source_cluster_id,
                "reused_at": reused_at,
            },
            "output": {
                "summary": source_cluster["output"]["summary"],
                "input_hash": input_hash,
                "prompt_hash": prompt_digest,
            },
            "attempt_count": len(cluster["attempts"]),
            "updated_at": reused_at,
            "completed_at": reused_at,
        }
    )
    summary_run = updated["summaries"][_summary_index(updated, summary_hash)]
    _refresh_summary_status(summary_run, reused_at)
    updated["updated_at"] = reused_at
    validate_kmeans_analysis(updated)
    return updated
