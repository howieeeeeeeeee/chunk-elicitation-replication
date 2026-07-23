"""Shared deterministic identity and validation helpers for derived analyses."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timezone


SECRET_KEY_PATTERN = re.compile(
    r"api.?key|authorization|auth.?header|bearer|credential|password|secret|"
    r"access.?token|refresh.?token|(?:^|_)token$|cookie",
    re.IGNORECASE,
)


class DerivedAnalysisConfigurationError(ValueError):
    """A configuration error that never includes submitted values."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_json_value(value, label: str):
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DerivedAnalysisConfigurationError(
            "non_json_value", f"{label} must contain only finite JSON values."
        ) from error
    return json.loads(serialized)


def _reject_secret_keys(value, path: str) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise DerivedAnalysisConfigurationError(
                    "invalid_key", f"{path} keys must be strings."
                )
            if SECRET_KEY_PATTERN.search(key):
                raise DerivedAnalysisConfigurationError(
                    "secret_key", f"{path} contains a forbidden secret-bearing key."
                )
            _reject_secret_keys(nested_value, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _reject_secret_keys(nested_value, f"{path}[{index}]")


def normalize_config(config: dict, label: str) -> dict:
    if not isinstance(config, dict):
        raise DerivedAnalysisConfigurationError(
            "invalid_config", f"{label} must be a dictionary."
        )
    _reject_secret_keys(config, label)
    return normalize_json_value(config, label)


def canonical_config_json(config: dict, label: str) -> str:
    normalized = normalize_config(config, label)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_hash(config: dict, label: str) -> str:
    canonical = canonical_config_json(config, label)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_embedding_ids(embedding_ids) -> list[str]:
    if not isinstance(embedding_ids, (list, tuple)) or not embedding_ids:
        raise DerivedAnalysisConfigurationError(
            "invalid_embedding_ids", "embedding_ids must be a nonempty list."
        )
    normalized = []
    for embedding_id in embedding_ids:
        if not isinstance(embedding_id, str) or not embedding_id.strip():
            raise DerivedAnalysisConfigurationError(
                "invalid_embedding_id",
                "Every embedding id must be a nonblank string.",
            )
        normalized.append(embedding_id)
    if len(normalized) != len(set(normalized)):
        raise DerivedAnalysisConfigurationError(
            "duplicate_embedding_id", "embedding_ids must not contain duplicates."
        )
    return sorted(normalized)


def embedding_set_hash(embedding_ids) -> str:
    normalized = normalize_embedding_ids(embedding_ids)
    canonical = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deterministic_entity_id(prefix: str, identity: dict) -> str:
    canonical = json.dumps(
        normalize_json_value(identity, "analysis identity"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def utc_timestamp(value: datetime | None = None) -> str:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("Derived-analysis timestamps must be timezone-aware.")
    return resolved.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_timestamp(value, field: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be a UTC ISO timestamp.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a UTC ISO timestamp.") from error


def validate_nonblank_string(value, field: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string.")


def validate_nonnegative_int(value, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer.")


def sanitize_error(error: Exception, message: str) -> dict:
    sanitized = {
        "type": type(error).__name__,
        "code": getattr(error, "code", "unexpected_error"),
        "message": message,
    }
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        sanitized["status_code"] = status_code
    return normalize_json_value(sanitized, "sanitized error")


def validate_sanitized_error(error: dict, expected_message: str) -> None:
    allowed = {"type", "code", "message", "status_code"}
    required = {"type", "code", "message"}
    if (
        not isinstance(error, dict)
        or not required.issubset(error)
        or not set(error).issubset(allowed)
    ):
        raise ValueError("Sanitized error has an unexpected schema.")
    validate_nonblank_string(error["type"], "error type")
    validate_nonblank_string(error["code"], "error code")
    if error["message"] != expected_message:
        raise ValueError("Sanitized error message is invalid.")
    if "status_code" in error:
        validate_nonnegative_int(error["status_code"], "error status_code")


def copy_normalized_config(config: dict, label: str) -> dict:
    return deepcopy(normalize_config(config, label))


def attempt_history_extends(existing: list[dict], replacement: list[dict]) -> bool:
    """Return whether replacement preserves history and only finalizes its active tail."""
    if len(replacement) < len(existing):
        return False
    for index, previous in enumerate(existing):
        current = replacement[index]
        if previous.get("finished_at") is not None:
            if current != previous:
                return False
            continue
        if index != len(existing) - 1:
            return False
        if set(previous) != set(current):
            return False
        for key, value in previous.items():
            if value is not None and current.get(key) != value:
                return False
    return True
