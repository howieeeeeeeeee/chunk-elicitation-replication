"""Canonical embedding configuration and deterministic identity helpers."""

import hashlib
import json
import re
from copy import deepcopy


ALLOWED_CONFIG_KEYS = {
    "model",
    "dimensions",
    "input_type",
    "provider",
    "encoding_format",
}
ALLOWED_PROVIDER_KEYS = {
    "order",
    "allow_fallbacks",
    "require_parameters",
    "data_collection",
    "zdr",
    "enforce_distillable_text",
    "only",
    "ignore",
    "quantizations",
    "sort",
    "preferred_min_throughput",
    "preferred_max_latency",
    "max_price",
}
SECRET_KEY_PATTERN = re.compile(
    r"api.?key|authorization|auth.?header|bearer|credential|password|secret|token|cookie",
    re.IGNORECASE,
)


class EmbeddingConfigurationError(ValueError):
    """A safe configuration error that never includes configuration values."""

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
        raise EmbeddingConfigurationError(
            "non_json_value", f"{label} must contain only finite JSON values."
        ) from error
    return json.loads(serialized)


def _reject_secret_keys(value, path: str = "embedding_config") -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise EmbeddingConfigurationError(
                    "invalid_key", f"{path} keys must be strings."
                )
            if SECRET_KEY_PATTERN.search(key):
                raise EmbeddingConfigurationError(
                    "secret_key", f"{path} contains a forbidden secret-bearing key."
                )
            _reject_secret_keys(nested_value, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _reject_secret_keys(nested_value, f"{path}[{index}]")


def normalize_embedding_config(embedding_config: dict) -> dict:
    if not isinstance(embedding_config, dict):
        raise EmbeddingConfigurationError(
            "invalid_config", "Embedding configuration must be a dictionary."
        )
    _reject_secret_keys(embedding_config)

    unknown_keys = set(embedding_config) - ALLOWED_CONFIG_KEYS
    if unknown_keys:
        raise EmbeddingConfigurationError(
            "unknown_config_key", "Embedding configuration contains unknown keys."
        )

    model = embedding_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise EmbeddingConfigurationError(
            "invalid_model", "Embedding configuration requires a nonblank model."
        )

    encoding_format = embedding_config.get("encoding_format", "float")
    if encoding_format != "float":
        raise EmbeddingConfigurationError(
            "unsupported_encoding",
            "Reasoning embeddings require encoding_format='float'.",
        )

    normalized = {"model": model.strip(), "encoding_format": "float"}
    if "dimensions" in embedding_config:
        dimensions = embedding_config["dimensions"]
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise EmbeddingConfigurationError(
                "invalid_dimensions", "Embedding dimensions must be a positive integer."
            )
        normalized["dimensions"] = dimensions

    if "input_type" in embedding_config:
        input_type = embedding_config["input_type"]
        if not isinstance(input_type, str) or not input_type.strip():
            raise EmbeddingConfigurationError(
                "invalid_input_type", "Embedding input_type must be a nonblank string."
            )
        normalized["input_type"] = input_type.strip()

    if "provider" in embedding_config and embedding_config["provider"] is not None:
        provider = embedding_config["provider"]
        if not isinstance(provider, dict):
            raise EmbeddingConfigurationError(
                "invalid_provider", "Embedding provider preferences must be a dictionary."
            )
        unknown_provider_keys = set(provider) - ALLOWED_PROVIDER_KEYS
        if unknown_provider_keys:
            raise EmbeddingConfigurationError(
                "unknown_provider_key",
                "Embedding provider preferences contain unknown keys.",
            )
        if provider:
            normalized["provider"] = normalize_json_value(provider, "provider")

    return normalize_json_value(normalized, "embedding_config")


def canonical_embedding_config_json(embedding_config: dict) -> str:
    normalized = normalize_embedding_config(embedding_config)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def embedding_config_hash(embedding_config: dict) -> str:
    canonical = canonical_embedding_config_json(embedding_config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def embedding_entity_id(
    simulation_session_id: str,
    decision_index: int,
    embedding_config: dict,
) -> str:
    if not isinstance(simulation_session_id, str) or not simulation_session_id:
        raise EmbeddingConfigurationError(
            "invalid_session_id", "simulation_session_id must be a nonblank string."
        )
    if (
        isinstance(decision_index, bool)
        or not isinstance(decision_index, int)
        or decision_index < 0
    ):
        raise EmbeddingConfigurationError(
            "invalid_decision_index", "decision_index must be a nonnegative integer."
        )
    identity = {
        "simulation_session_id": simulation_session_id,
        "decision_index": decision_index,
        "embedding_config_hash": embedding_config_hash(embedding_config),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"reasoning-embedding:{digest}"


def copy_normalized_config(embedding_config: dict) -> dict:
    return deepcopy(normalize_embedding_config(embedding_config))
