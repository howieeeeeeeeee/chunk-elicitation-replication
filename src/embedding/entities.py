"""Deterministic MongoDB/JSON-compatible reasoning embedding entities."""

import math
from copy import deepcopy
from datetime import datetime, timezone

from embedding.client import EmbeddingRequestError
from embedding.config import (
    EmbeddingConfigurationError,
    copy_normalized_config,
    embedding_config_hash,
    embedding_entity_id,
    normalize_json_value,
)


IMMUTABLE_EMBEDDING_FIELDS = (
    "_id",
    "simulation_id",
    "simulation_session_id",
    "decision_index",
    "input_text",
    "embedding_config",
    "embedding_config_hash",
    "created_at",
)


def _timestamp(value: datetime | None = None) -> str:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("Embedding timestamps must be timezone-aware.")
    return resolved.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_nonblank_string(value, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonblank string.")


def _validate_timestamp(value, field: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be a UTC ISO timestamp.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a UTC ISO timestamp.") from error


def _validate_usage(usage: dict) -> None:
    if not isinstance(usage, dict):
        raise ValueError("Embedding usage must be a dictionary.")
    normalize_json_value(usage, "embedding usage")
    for field in ("prompt_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Embedding usage is missing required token counts.")
    if "cost" not in usage:
        raise ValueError("Embedding usage must record cost or null.")


def _validate_attempt(attempt: dict) -> None:
    required = {
        "timestamp",
        "success",
        "response_id",
        "resolved_model",
        "usage",
        "error",
    }
    if not isinstance(attempt, dict) or set(attempt) != required:
        raise ValueError("Embedding attempt has an unexpected schema.")
    _validate_timestamp(attempt["timestamp"], "attempt timestamp")
    if not isinstance(attempt["success"], bool):
        raise ValueError("Embedding attempt success must be boolean.")
    for field in ("response_id", "resolved_model"):
        if attempt[field] is not None and not isinstance(attempt[field], str):
            raise ValueError(f"Embedding attempt {field} must be a string or null.")
    if attempt["usage"] is not None:
        _validate_usage(attempt["usage"])
    if attempt["success"]:
        if attempt["usage"] is None or attempt["error"] is not None:
            raise ValueError("Successful embedding attempt has invalid metadata.")
    elif not isinstance(attempt["error"], dict):
        raise ValueError("Failed embedding attempt requires a structured error.")


def _validate_output(output: dict) -> None:
    required = {
        "response_id",
        "resolved_model",
        "object",
        "vector",
        "vector_dimension",
    }
    if not isinstance(output, dict) or set(output) != required:
        raise ValueError("Successful embedding output has an unexpected schema.")
    if output["response_id"] is not None and not isinstance(output["response_id"], str):
        raise ValueError("Embedding response_id must be a string or null.")
    _validate_nonblank_string(output["resolved_model"], "resolved_model")
    _validate_nonblank_string(output["object"], "object")
    vector = output["vector"]
    if not isinstance(vector, list) or not vector:
        raise ValueError("Embedding output vector must be nonempty.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Embedding output vector must contain finite numbers.")
    if output["vector_dimension"] != len(vector):
        raise ValueError("Embedding output vector dimension is invalid.")


def sanitize_embedding_error(error: Exception) -> dict:
    if isinstance(error, (EmbeddingRequestError, EmbeddingConfigurationError)):
        sanitized = {
            "type": type(error).__name__,
            "code": error.code,
            "message": str(error),
        }
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            sanitized["status_code"] = status_code
        return sanitized
    return {
        "type": type(error).__name__,
        "code": "unexpected_error",
        "message": "Embedding attempt failed.",
    }


def build_embedding_entity(
    simulation_id: str,
    simulation_session_id: str,
    decision_index: int,
    input_text: str | None,
    embedding_config: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    _validate_nonblank_string(simulation_id, "simulation_id")
    _validate_nonblank_string(simulation_session_id, "simulation_session_id")
    if input_text is not None and (
        not isinstance(input_text, str) or not input_text.strip()
    ):
        raise ValueError("input_text must be a nonblank string or null.")
    normalized_config = copy_normalized_config(embedding_config)
    created_at = _timestamp(timestamp)
    entity = {
        "_id": embedding_entity_id(
            simulation_session_id, decision_index, normalized_config
        ),
        "simulation_id": simulation_id,
        "simulation_session_id": simulation_session_id,
        "decision_index": decision_index,
        "input_text": input_text,
        "embedding_config": normalized_config,
        "embedding_config_hash": embedding_config_hash(normalized_config),
        "success": False,
        "output": None,
        "usage": None,
        "attempt_count": 0,
        "attempts": [],
        "created_at": created_at,
        "updated_at": created_at,
        "succeeded_at": None,
    }
    validate_embedding_entity(entity)
    return entity


def _ensure_retryable(entity: dict) -> dict:
    validate_embedding_entity(entity)
    if entity["success"]:
        raise ValueError("A successful embedding entity cannot receive another attempt.")
    return deepcopy(entity)


def record_embedding_success(
    entity: dict,
    response: dict,
    *,
    timestamp: datetime | None = None,
) -> dict:
    updated = _ensure_retryable(entity)
    if not isinstance(updated["input_text"], str) or not updated["input_text"].strip():
        raise ValueError("A successful embedding requires nonblank input_text.")
    required = {
        "response_id",
        "resolved_model",
        "object",
        "vector",
        "vector_dimension",
        "usage",
    }
    if not isinstance(response, dict) or set(response) != required:
        raise ValueError("Validated embedding response has an unexpected schema.")
    usage = normalize_json_value(response["usage"], "embedding usage")
    vector = deepcopy(response["vector"])
    if response["vector_dimension"] != len(vector) or not vector:
        raise ValueError("Validated embedding response has an invalid vector dimension.")

    attempted_at = _timestamp(timestamp)
    attempt = {
        "timestamp": attempted_at,
        "success": True,
        "response_id": response["response_id"],
        "resolved_model": response["resolved_model"],
        "usage": usage,
        "error": None,
    }
    updated["attempts"].append(attempt)
    updated["attempt_count"] = len(updated["attempts"])
    updated["success"] = True
    updated["output"] = {
        "response_id": response["response_id"],
        "resolved_model": response["resolved_model"],
        "object": response["object"],
        "vector": vector,
        "vector_dimension": response["vector_dimension"],
    }
    updated["usage"] = usage
    updated["updated_at"] = attempted_at
    updated["succeeded_at"] = attempted_at
    validate_embedding_entity(updated)
    return updated


def record_embedding_failure(
    entity: dict,
    error: Exception,
    *,
    timestamp: datetime | None = None,
) -> dict:
    updated = _ensure_retryable(entity)
    attempted_at = _timestamp(timestamp)
    usage = getattr(error, "usage", None)
    normalized_usage = (
        normalize_json_value(usage, "embedding failure usage")
        if usage is not None
        else None
    )
    attempt = {
        "timestamp": attempted_at,
        "success": False,
        "response_id": getattr(error, "response_id", None),
        "resolved_model": getattr(error, "resolved_model", None),
        "usage": normalized_usage,
        "error": sanitize_embedding_error(error),
    }
    updated["attempts"].append(attempt)
    updated["attempt_count"] = len(updated["attempts"])
    updated["success"] = False
    updated["output"] = None
    updated["usage"] = normalized_usage
    updated["updated_at"] = attempted_at
    updated["succeeded_at"] = None
    validate_embedding_entity(updated)
    return updated


def validate_embedding_entity(entity: dict) -> None:
    required_fields = {
        "_id",
        "simulation_id",
        "simulation_session_id",
        "decision_index",
        "input_text",
        "embedding_config",
        "embedding_config_hash",
        "success",
        "output",
        "usage",
        "attempt_count",
        "attempts",
        "created_at",
        "updated_at",
        "succeeded_at",
    }
    if not isinstance(entity, dict) or set(entity) != required_fields:
        raise ValueError("Embedding entity has an unexpected schema.")
    _validate_nonblank_string(entity["simulation_id"], "simulation_id")
    _validate_nonblank_string(
        entity["simulation_session_id"], "simulation_session_id"
    )
    if entity["input_text"] is not None and (
        not isinstance(entity["input_text"], str) or not entity["input_text"].strip()
    ):
        raise ValueError("input_text must be a nonblank string or null.")
    normalized_config = copy_normalized_config(entity["embedding_config"])
    if normalized_config != entity["embedding_config"]:
        raise ValueError("Stored embedding configuration is not normalized.")
    expected_hash = embedding_config_hash(entity["embedding_config"])
    if entity["embedding_config_hash"] != expected_hash:
        raise ValueError("Embedding configuration hash does not match configuration.")
    expected_id = embedding_entity_id(
        entity["simulation_session_id"],
        entity["decision_index"],
        entity["embedding_config"],
    )
    if entity["_id"] != expected_id:
        raise ValueError("Embedding entity id does not match its identity fields.")
    if not isinstance(entity["success"], bool):
        raise ValueError("Embedding success must be boolean.")
    if (
        isinstance(entity["attempt_count"], bool)
        or not isinstance(entity["attempt_count"], int)
        or entity["attempt_count"] < 0
        or not isinstance(entity["attempts"], list)
        or entity["attempt_count"] != len(entity["attempts"])
    ):
        raise ValueError("Embedding attempt_count does not match attempts.")
    for attempt in entity["attempts"]:
        _validate_attempt(attempt)
    _validate_timestamp(entity["created_at"], "created_at")
    _validate_timestamp(entity["updated_at"], "updated_at")
    _validate_timestamp(entity["succeeded_at"], "succeeded_at", optional=True)
    if entity["success"]:
        if not isinstance(entity["input_text"], str) or not entity["input_text"].strip():
            raise ValueError("Successful embedding entity requires nonblank input_text.")
        if entity["output"] is None or entity["usage"] is None or not entity["succeeded_at"]:
            raise ValueError("Successful embedding entity is missing output metadata.")
        if not entity["attempts"] or not entity["attempts"][-1]["success"]:
            raise ValueError("Successful embedding entity requires a final successful attempt.")
        _validate_output(entity["output"])
        _validate_usage(entity["usage"])
        if entity["usage"] != entity["attempts"][-1]["usage"]:
            raise ValueError("Top-level embedding usage must match the latest attempt.")
    elif entity["output"] is not None or entity["succeeded_at"] is not None:
        raise ValueError("Failed or pending embedding entity cannot retain output.")
    elif entity["attempts"]:
        if entity["attempts"][-1]["success"]:
            raise ValueError("Failed embedding entity cannot end with a success.")
        if entity["usage"] != entity["attempts"][-1]["usage"]:
            raise ValueError("Top-level embedding usage must match the latest attempt.")
    elif entity["usage"] is not None:
        raise ValueError("Pending embedding entity cannot have usage metadata.")
