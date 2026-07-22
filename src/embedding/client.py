"""One-attempt OpenRouter client for a single reasoning-text embedding."""

import math
import os

import requests

from embedding.config import normalize_embedding_config, normalize_json_value


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_TIMEOUT_SECONDS = 120


class EmbeddingRequestError(RuntimeError):
    """A structured, persistence-safe embedding request or response error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        usage: dict | None = None,
        response_id: str | None = None,
        resolved_model: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.usage = usage
        self.response_id = response_id
        self.resolved_model = resolved_model


def _response_error(code: str, message: str) -> EmbeddingRequestError:
    return EmbeddingRequestError(code, message)


def _validate_usage(raw_usage) -> dict:
    if not isinstance(raw_usage, dict):
        raise _response_error(
            "invalid_usage", "OpenRouter embedding response has invalid usage metadata."
        )
    usage = normalize_json_value(raw_usage, "OpenRouter usage")
    for field in ("prompt_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _response_error(
                "invalid_usage",
                "OpenRouter embedding response is missing required token usage.",
            )
    usage.setdefault("cost", None)
    return usage


def _validate_vector(raw_vector) -> list[float]:
    if not isinstance(raw_vector, list) or not raw_vector:
        raise _response_error(
            "invalid_vector", "OpenRouter embedding response has no numeric vector."
        )
    vector = []
    for value in raw_vector:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _response_error(
                "invalid_vector", "OpenRouter embedding response has no numeric vector."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise _response_error(
                "invalid_vector", "OpenRouter embedding response has no numeric vector."
            )
        vector.append(numeric_value)
    return vector


def _parse_success_response(payload) -> dict:
    if not isinstance(payload, dict):
        raise _response_error(
            "invalid_response", "OpenRouter embedding response must be a JSON object."
        )
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise _response_error(
            "invalid_response",
            "OpenRouter embedding response must contain exactly one data item.",
        )

    item = data[0]
    if item.get("index", 0) != 0:
        raise _response_error(
            "invalid_response", "OpenRouter embedding response has an invalid item index."
        )
    object_type = item.get("object")
    if not isinstance(object_type, str) or not object_type:
        raise _response_error(
            "invalid_response", "OpenRouter embedding response has no object type."
        )
    resolved_model = payload.get("model")
    if not isinstance(resolved_model, str) or not resolved_model:
        raise _response_error(
            "invalid_response", "OpenRouter embedding response has no resolved model."
        )
    response_id = payload.get("id")
    if response_id is not None and not isinstance(response_id, str):
        raise _response_error(
            "invalid_response", "OpenRouter embedding response has an invalid id."
        )

    vector = _validate_vector(item.get("embedding"))
    usage = _validate_usage(payload.get("usage"))
    return {
        "response_id": response_id,
        "resolved_model": resolved_model,
        "object": object_type,
        "vector": vector,
        "vector_dimension": len(vector),
        "usage": usage,
    }


def _optional_error_metadata(payload) -> dict:
    if not isinstance(payload, dict):
        return {}
    metadata = {}
    if "usage" in payload:
        try:
            metadata["usage"] = _validate_usage(payload["usage"])
        except EmbeddingRequestError:
            pass
    if isinstance(payload.get("id"), str):
        metadata["response_id"] = payload["id"]
    if isinstance(payload.get("model"), str):
        metadata["resolved_model"] = payload["model"]
    return metadata


def request_reasoning_embedding(
    reason_text: str,
    embedding_config: dict,
    *,
    timeout_seconds: int | float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Make exactly one OpenRouter request and return validated embedding data.

    The API key is read only from ``OPENROUTER_API_KEY``. The function does not
    retry, log request data, or return headers/raw responses.
    """
    if not isinstance(reason_text, str) or not reason_text.strip():
        raise EmbeddingRequestError(
            "invalid_input", "Reasoning text must be a nonblank string."
        )
    if isinstance(timeout_seconds, bool) or not isinstance(
        timeout_seconds, (int, float)
    ) or timeout_seconds <= 0:
        raise EmbeddingRequestError(
            "invalid_timeout", "Embedding timeout must be positive."
        )

    normalized_config = normalize_embedding_config(embedding_config)
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise EmbeddingRequestError(
            "missing_api_key", "OPENROUTER_API_KEY is required for embeddings."
        )

    request_body = {"input": reason_text, **normalized_config}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers=headers,
            json=request_body,
            timeout=timeout_seconds,
        )
    except requests.RequestException as error:
        raise EmbeddingRequestError(
            "transport_error", "OpenRouter embedding request failed."
        ) from error

    status_code = response.status_code
    try:
        payload = response.json()
    except ValueError as error:
        raise EmbeddingRequestError(
            "invalid_json",
            "OpenRouter embedding response was not valid JSON.",
            status_code=status_code,
        ) from error

    if status_code < 200 or status_code >= 300 or (
        isinstance(payload, dict) and "error" in payload
    ):
        raise EmbeddingRequestError(
            "api_error",
            "OpenRouter rejected the embedding request.",
            status_code=status_code,
            **_optional_error_metadata(payload),
        )
    return _parse_success_response(payload)
