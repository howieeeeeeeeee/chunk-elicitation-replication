"""Reasoning-text embedding contracts for OpenRouter and persistence."""

from embedding.client import (
    EmbeddingRequestError,
    request_reasoning_embedding,
)
from embedding.config import (
    EmbeddingConfigurationError,
    embedding_config_hash,
    embedding_entity_id,
    normalize_embedding_config,
)
from embedding.entities import (
    build_embedding_entity,
    record_embedding_failure,
    record_embedding_success,
    validate_embedding_entity,
)
from embedding.orchestration import (
    EmbeddingEligibilityError,
    embed_decision_reasoning,
    embed_simulation,
    embed_simulation_session,
    summarize_embedding_plan,
)


__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingEligibilityError",
    "EmbeddingRequestError",
    "build_embedding_entity",
    "embedding_config_hash",
    "embedding_entity_id",
    "embed_decision_reasoning",
    "embed_simulation",
    "embed_simulation_session",
    "normalize_embedding_config",
    "record_embedding_failure",
    "record_embedding_success",
    "request_reasoning_embedding",
    "summarize_embedding_plan",
    "validate_embedding_entity",
]
